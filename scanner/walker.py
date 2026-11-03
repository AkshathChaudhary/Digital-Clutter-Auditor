from models import FileRecord
from pathlib import Path
from typing import Iterator, Optional

from config import kind_from_ext 

def iter_files(
        root: Path,
        errors:Optional[list]= None
) -> Iterator[Path]:
    """
    Recursively iterate over all files in the given root directory.

    Args:
        root (Path): The root directory to start the search from.

    Yields:
        Iterator[Path]: An iterator of file paths.
    """
    root = Path(root).expanduser()

    if errors is None:
        errors = []

    if not root.exists():
        raise FileNotFoundError(f"The specified root directory '{root}' does not exist.")
    if not root.is_dir():
        raise NotADirectoryError(f"The specified path '{root}' is not a directory.")
    if root.is_symlink():
        raise ValueError(f"The specified root directory '{root}' is a symbolic link, which is not allowed.")

    try:
        entries = list(root.iterdir())
    except OSError as exc:
        errors.append({
            "path": str(root),
            "error": str(exc) 
        })

    for entry in entries:
        try:
            if entry.is_symlink():
                continue
            if entry.is_dir():
                yield from iter_files(entry, errors)
            elif entry.is_file():
                yield entry
        except OSError as exc:
            errors.append({
                "path": str(entry),
                "error": str(exc)
            })
            continue


def extract_metadata(file_path: Path) -> dict:
    """
    Extract metadata from the given file path.
    """


    file_path = Path(file_path)
    info=file_path.stat()
    name = file_path.name
    ext = file_path.suffix.lower().lstrip(".")
    kind = kind_from_ext(ext)

    metadata = {
        "path": str(file_path.resolve()),
        "name": name,
        "ext": ext,
        "kind": kind,
        "size": info.st_size,
        "created": getattr(info, "st_birthtime", info.st_ctime),
        "modified": info.st_mtime,
        "accessed": info.st_atime,
    }
    return metadata
def create_file_record(file_path: Path) -> FileRecord:
    """
    Create a FileRecord instance from the given file path.
    """
    metadata = extract_metadata(file_path)
    record= FileRecord(**metadata)
    return record

def scan_directory(root: Path) -> tuple[list[FileRecord], list[dict]]:

    records = []
    errors = []

    
    for file_path in iter_files(root, errors):

        try:
            record = create_file_record(file_path)
            records.append(record)

        except OSError as exc:
            errors.append({
                "path": str(file_path),
                "error": str(exc)
            })

            continue

    return records, errors


def scan_folder(
    root_folder: str,
    *args,
    clean_slate: bool = False,
    db_path: Optional[Path] = None,
    **kwargs,
) -> int:
    """
    Scan a directory and store file metadata in SQLite.
    Returns the count of files indexed.

    Args:
        root_folder: Path to folder to scan.
        clean_slate: If True, clears previous active files to provide a fresh audit.
        db_path: Target SQLite database path (defaults to DB_PATH).
    """
    import db
    from config import DB_PATH

    # Support clean_slate passed positionally or via kwargs
    if args:
        clean_slate = bool(args[0])
    elif "clean_slate" in kwargs:
        clean_slate = bool(kwargs["clean_slate"])

    target_db = db_path if db_path is not None else DB_PATH
    db.init_db(target_db)

    root = Path(root_folder).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Folder does not exist: {root}")

    if clean_slate:
        # Clear previous active files for a fresh, isolated audit
        conn = db.get_conn(target_db)
        try:
            with conn:
                conn.execute("DELETE FROM files WHERE status = 'active'")
        finally:
            conn.close()

    records, errors = scan_directory(root)

    saved_count = 0
    for record in records:
        db.upsert_file(record, db_path=target_db)
        saved_count += 1

    for error in errors:
        import logging
        logger = logging.getLogger("clutteriq")
        logger.warning(f"Could not process {error['path']}: {error['error']}")

    return saved_count
