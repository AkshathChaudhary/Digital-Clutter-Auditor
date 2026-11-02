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

    return destination

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

    except Exception:
        # Compensate if the database update fails.
        if original_path.exists() and not quarantine_path.exists():
            shutil.move(str(original_path), str(quarantine_path))

        raise

    return original_path