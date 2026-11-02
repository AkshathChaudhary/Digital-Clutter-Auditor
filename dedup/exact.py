
import hashlib
from pathlib import Path
from collections import defaultdict
import db
from config import DB_PATH

def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def group_by_size(file_paths: list[Path]) -> dict[int, list[Path]]:
    
    size_groups = defaultdict(list)

    for file_path in file_paths:
        try:
            file_size = file_path.stat().st_size
            size_groups[file_size].append(file_path)

        except OSError:
            # Skip files that cannot be accessed.
            continue

    return {
        size: paths
        for size, paths in size_groups.items()
        if len(paths) > 1
    }


def group_by_hash(
    size_groups: dict[int, list[Path]]
) -> dict[str, list[Path]]:
    
    hash_groups = defaultdict(list)

    for paths in size_groups.values():
        for file_path in paths:
            try:
                file_hash = compute_sha256(file_path)
                hash_groups[file_hash].append(file_path)

            except OSError:
                # Skip files that cannot be read.
                continue

    return {
        file_hash: paths
        for file_hash, paths in hash_groups.items()
        if len(paths) > 1
    }


def find_exact_duplicates(db_path: Path = DB_PATH) -> dict[int, list[Path]]:
    

    # Ensure the database exists.
    db.init_db(db_path)

    # Retrieve all active file records.
    records = db.get_active_files(db_path=db_path)

    # Clear previous results to prevent stale group assignments.
    for record in records:
        db.update_columns(
            record.id,
            db_path=db_path,
            sha256=None,
            exact_group=None
        )

    # Convert database records into file paths.
    file_paths = [Path(record.path) for record in records]

    # Run the duplicate detection pipeline.
    size_groups = group_by_size(file_paths)
    hash_groups = group_by_hash(size_groups)

    # Map paths back to their database records.
    record_lookup = {
        Path(record.path): record
        for record in records
    }

    # Compute all hashes for files in size groups (candidates for duplicates)
    all_hashes = {}
    for paths in size_groups.values():
        for file_path in paths:
            try:
                file_hash = compute_sha256(file_path)
                all_hashes[file_path] = file_hash
            except OSError:
                continue

    # Save hashes and assign group IDs only for actual duplicates.
    duplicate_results = {}

    for group_id, (file_hash, paths) in enumerate(
        hash_groups.items(), start=1
    ):
        duplicate_results[group_id] = paths

        for file_path in paths:
            record = record_lookup[file_path]
            db.update_columns(
                record.id,
                db_path=db_path,
                sha256=file_hash,
                exact_group=group_id
            )

    # For files in size collision groups but not duplicates, still store their hash
    for paths in size_groups.values():
        for file_path in paths:
            if file_path in all_hashes:
                # Only update if not already updated as duplicate
                record = record_lookup.get(file_path)
                if record:
                    # Check if already assigned to a group
                    current_record = db.get_file_by_path(str(file_path), db_path=db_path)
                    if current_record and current_record.exact_group is None:
                        db.update_columns(
                            record.id,
                            db_path=db_path,
                            sha256=all_hashes[file_path],
                            exact_group=None
                        )

    return duplicate_results

