
from pathlib import Path
from collections import defaultdict
import db
from config import DB_PATH
from models import FileRecord

from config import PHASH_MAX_DISTANCE

import imagehash
from PIL import Image


def compute_phash(file_path: Path) -> str:
    

    with Image.open(file_path) as image:
        perceptual_hash = imagehash.phash(image)

    return str(perceptual_hash)


def compare_phashes(hash_a: str, hash_b: str) -> int:
    

    phash_a = imagehash.hex_to_hash(hash_a)
    phash_b = imagehash.hex_to_hash(hash_b)

    return phash_a - phash_b

def group_by_phash(
    phashes: dict[Path, str],
    max_distance: int = PHASH_MAX_DISTANCE,
) -> dict[int, list[Path]]:
    

    paths = list(phashes.keys())

    # Initially, every image belongs to its own group.
    parent = {path: path for path in paths}

    def find(path):
        if parent[path] != path:
            parent[path] = find(parent[path])

        return parent[path]

    def union(path_a, path_b):
        root_a = find(path_a)
        root_b = find(path_b)

        if root_a != root_b:
            parent[root_b] = root_a

    
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            path_a = paths[i]
            path_b = paths[j]

            distance = compare_phashes(
                phashes[path_a],
                phashes[path_b]
            )

            if distance <= max_distance:
                union(path_a, path_b)

    # Collect connected images into grps.
    components = defaultdict(list)

    for path in paths:
        components[find(path)].append(path)

    
    return {
        group_id: group_paths
        for group_id, group_paths in enumerate(
            (
                paths_in_group
                for paths_in_group in components.values()
                if len(paths_in_group) > 1
            ),
            start=1
        )
    }

def find_near_duplicates(
    db_path: Path = DB_PATH
) -> dict[int, list[Path]]:
    

    db.init_db(db_path)

    # Retrive active records.
    records = db.get_active_files(db_path=db_path)

    
    image_records = [
        record for record in records
        if record.kind == "image"
    ]

    
    for record in image_records:
        db.update_columns(
            record.id,
            db_path=db_path,
            phash=None,
            near_group=None
        )

    
    phashes = {}
    record_lookup = {}

    for record in image_records:
        file_path = Path(record.path)

        try:
            image_hash = compute_phash(file_path)

            phashes[file_path] = image_hash
            record_lookup[file_path] = record

        except (OSError, ValueError):
            
            continue

    
    for file_path, image_hash in phashes.items():
        record = record_lookup[file_path]

        db.update_columns(
            record.id,
            db_path=db_path,
            phash=image_hash
        )

    
    groups = group_by_phash(phashes)

    
    for group_id, paths in groups.items():
        for file_path in paths:
            record = record_lookup[file_path]

            db.update_columns(
                record.id,
                db_path=db_path,
                near_group=group_id
            )

    return groups

def test_find_near_duplicates_updates_database(tmp_path):
    database = tmp_path / "test.db"

    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    third = tmp_path / "third.png"

    # Identical images should have identical perceptual hashes.
    Image.new("RGB", (100, 100), "blue").save(first)
    Image.new("RGB", (100, 100), "blue").save(second)
    Image.new("RGB", (100, 100), "red").save(third)

    db.init_db(database)

    for path in [first, second, third]:
        record = FileRecord(
            path=str(path),
            name=path.name,
            ext="png",
            kind="image",
            size=path.stat().st_size,
        )
        db.upsert_file(record, db_path=database)

    groups = find_near_duplicates(db_path=database)

    assert len(groups) == 1
    assert len(groups[1]) == 2

    first_record = db.get_file_by_path(str(first), db_path=database)
    second_record = db.get_file_by_path(str(second), db_path=database)
    third_record = db.get_file_by_path(str(third), db_path=database)

    assert first_record.phash is not None
    assert second_record.phash is not None
    assert first_record.phash == second_record.phash

    assert first_record.near_group == 1
    assert second_record.near_group == 1
    assert third_record.near_group is None


def test_find_near_duplicates_ignores_non_images(tmp_path):
    database = tmp_path / "test.db"

    text_file = tmp_path / "notes.txt"
    text_file.write_text("This is not an image.")

    db.init_db(database)

    record = FileRecord(
        path=str(text_file),
        name=text_file.name,
        ext="txt",
        kind="document",
        size=text_file.stat().st_size,
    )

    db.upsert_file(record, db_path=database)

    groups = find_near_duplicates(db_path=database)

    assert groups == {}
