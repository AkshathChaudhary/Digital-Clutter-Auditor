from pathlib import Path
import shutil
import time
import uuid

import db
from config import DB_PATH, QUARANTINE_DIR
from dedup.exact import compute_sha256

def generate_quarantine_path(file_path: Path) -> Path:
    """
    Generate a unique quarantine destination.
    """

    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

    unique_name = f"{uuid.uuid4().hex}{file_path.suffix}"

    return QUARANTINE_DIR / unique_name

def quarantine_file(file_id: int, db_path: Path = DB_PATH) -> Path:
    """
    Move one file into quarantine.

    Returns the quarantine path.
    """

    db.init_db(db_path)

    record = db.get_file_by_id(file_id, db_path=db_path)

    if record is None:
        raise ValueError("File record not found.")

    source = Path(record.path)

    if not source.exists():
        raise FileNotFoundError(source)

    file_hash = compute_sha256(source)

    destination = generate_quarantine_path(source)

    shutil.move(str(source), str(destination))

    moved_at = time.time()

    db.add_quarantine_record(
        file_id=file_id,
        original_path=str(source),
        quarantine_path=str(destination),
        sha256=file_hash,
        moved_at=moved_at,
        db_path=db_path,
    )

    db.update_columns(
        file_id,
        db_path=db_path,
        status="quarantined",
    )

    db.resolve_lone_duplicates(db_path=db_path)

    return destination



def quarantine_files(file_ids: list[int], db_path: Path = DB_PATH) -> int:
    """Move multiple files into quarantine in batch."""
    count = 0
    for fid in file_ids:
        try:
            quarantine_file(fid, db_path=db_path)
            count += 1
        except Exception:
            continue
    db.resolve_lone_duplicates(db_path=db_path)
    return count

def restore_file(
    quarantine_id: int,
    db_path: Path = DB_PATH,
) -> Path:
    """
    Restore a quarantined file to its original location.

    Verifies file integrity and refuses to overwrite existing files.
    Returns the restored file path.
    """

    db.init_db(db_path)

    record = db.get_quarantine_record(
        quarantine_id,
        db_path=db_path,
    )

    if record is None:
        raise ValueError("Quarantine record not found.")

    if record["restored_at"] is not None:
        raise ValueError("This file has already been restored.")

    quarantine_path = Path(record["quarantine_path"])
    original_path = Path(record["original_path"])

    if not quarantine_path.is_file():
        raise FileNotFoundError(
            f"Quarantined file not found: {quarantine_path}"
        )

    # Verify the file has not changed since quarantine.
    current_hash = compute_sha256(quarantine_path)

    if current_hash != record["sha256"]:
        raise ValueError(
            "File integrity check failed. Restoration cancelled."
        )

    # Never overwrite an existing file or symbolic link.
    if original_path.exists() or original_path.is_symlink():
        raise FileExistsError(
            f"Original destination is occupied: {original_path}"
        )

    original_path.parent.mkdir(parents=True, exist_ok=True)

    # Move the file back.
    shutil.move(str(quarantine_path), str(original_path))

    try:
        db.mark_quarantine_restored(
            quarantine_id=quarantine_id,
            file_id=record["file_id"],
            restored_at=time.time(),
            db_path=db_path,
        )
        db.resolve_lone_duplicates(db_path=db_path)

    except Exception:
        # Compensate if the database update fails.
        if original_path.exists() and not quarantine_path.exists():
            shutil.move(str(original_path), str(quarantine_path))

        raise

    return original_path


def permanently_delete_file(
    quarantine_id: int,
    db_path: Path = DB_PATH,
) -> tuple[bool, str, int]:
    """
    Permanently delete a file from the quarantine vault and disk.
    This action cannot be undone. Reclaims disk space permanently.

    Returns:
        tuple: (success, filename, freed_bytes)
    """
    db.init_db(db_path)
    record = db.get_quarantine_record(quarantine_id, db_path=db_path)
    if record is None:
        raise ValueError("Quarantine record not found.")

    quarantine_path = Path(record["quarantine_path"])
    original_path = Path(record["original_path"])
    file_rec = db.get_file_by_id(record["file_id"], db_path=db_path)

    freed_bytes = file_rec.size if file_rec else 0
    filename = file_rec.name if file_rec else original_path.name

    # 1. Unlink ONLY the quarantined file in the vault
    if quarantine_path.is_file() or quarantine_path.exists():
        quarantine_path.unlink()

    # 2. Mark this record and file as deleted
    db.mark_quarantine_deleted(quarantine_id, record["file_id"], db_path=db_path)

    # 3. Automatically resolve duplicate groups and refresh health stats
    db.resolve_lone_duplicates(db_path=db_path)

    return True, filename, freed_bytes


def permanently_delete_all(
    db_path: Path = DB_PATH,
) -> tuple[int, int]:
    """
    Permanently purge all active files from the quarantine vault and disk.

    Returns:
        tuple: (count_deleted, total_bytes_freed)
    """
    conn = db.get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM quarantine_log WHERE restored_at IS NULL")
        q_ids = [row[0] for row in cur.fetchall()]
    finally:
        conn.close()

    total_count = 0
    total_bytes = 0
    for qid in q_ids:
        try:
            _, _, freed = permanently_delete_file(qid, db_path=db_path)
            total_count += 1
            total_bytes += freed
        except Exception:
            continue

    return total_count, total_bytes
