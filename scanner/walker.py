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
        "created": info.st_birthtime,
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
