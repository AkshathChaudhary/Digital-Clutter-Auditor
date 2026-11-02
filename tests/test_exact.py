
import hashlib
from dedup.exact import group_by_hash
import db
from models import FileRecord
from dedup.exact import find_exact_duplicates

from dedup.exact import compute_sha256


def test_compute_sha256_matches_standard_library(tmp_path):
    file_path = tmp_path / "sample.txt"
    content = b"Hello ClutterIQ"

    file_path.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()

    assert compute_sha256(file_path) == expected


def test_identical_files_have_identical_hashes(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"

    first.write_bytes(b"same content")
    second.write_bytes(b"same content")

    assert compute_sha256(first) == compute_sha256(second)


def test_different_files_have_different_hashes(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"

    first.write_bytes(b"content A")
    second.write_bytes(b"content B")

    assert compute_sha256(first) != compute_sha256(second)


def test_empty_file_can_be_hashed(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_bytes(b"")

    assert compute_sha256(file_path) == hashlib.sha256(b"").hexdigest()


from dedup.exact import group_by_size


def test_group_by_size_keeps_only_repeated_sizes(tmp_path):
    files = []

    test_data = {
        "A.txt": b"A" * 500,
        "B.txt": b"B" * 500,
        "C.txt": b"C" * 800,
        "D.txt": b"D" * 500,
        "E.txt": b"E" * 1200,
    }

    for name, content in test_data.items():
        file_path = tmp_path / name
        file_path.write_bytes(content)
        files.append(file_path)

    groups = group_by_size(files)

    assert list(groups.keys()) == [500]
    assert len(groups[500]) == 3


def test_group_by_size_returns_empty_when_all_sizes_are_unique(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"

    first.write_bytes(b"A" * 100)
    second.write_bytes(b"B" * 200)

    assert group_by_size([first, second]) == {}

    
def test_group_by_hash_finds_exact_duplicates(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    third = tmp_path / "third.txt"

    first.write_bytes(b"same content")
    second.write_bytes(b"same content")
    third.write_bytes(b"different!")

    size_groups = group_by_size([first, second, third])
    duplicate_groups = group_by_hash(size_groups)

    assert len(duplicate_groups) == 1

    duplicate_files = next(iter(duplicate_groups.values()))

    assert set(duplicate_files) == {first, second}


def test_group_by_hash_ignores_unique_files(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"

    first.write_bytes(b"AAAA")
    second.write_bytes(b"BBBB")

    size_groups = group_by_size([first, second])

    assert group_by_hash(size_groups) == {}


def test_group_by_hash_handles_empty_files(tmp_path):
    first = tmp_path / "empty1.txt"
    second = tmp_path / "empty2.txt"

    first.write_bytes(b"")
    second.write_bytes(b"")

    size_groups = group_by_size([first, second])
    duplicate_groups = group_by_hash(size_groups)

    assert len(duplicate_groups) == 1
    assert len(next(iter(duplicate_groups.values()))) == 2


def test_find_exact_duplicates_updates_database(tmp_path):
    database = tmp_path / "test.db"

    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    third = tmp_path / "third.txt"

    first.write_bytes(b"duplicate content")
    second.write_bytes(b"duplicate content")
    third.write_bytes(b"unique content!")

    db.init_db(database)

    for path in [first, second, third]:
        record = FileRecord(
            path=str(path),
            name=path.name,
            ext="txt",
            kind="document",
            size=path.stat().st_size,
        )
        db.upsert_file(record, db_path=database)

    groups = find_exact_duplicates(db_path=database)

    assert len(groups) == 1
    assert len(groups[1]) == 2

    first_record = db.get_file_by_path(str(first), db_path=database)
    second_record = db.get_file_by_path(str(second), db_path=database)
    third_record = db.get_file_by_path(str(third), db_path=database)

    assert first_record.exact_group == 1
    assert second_record.exact_group == 1
    assert first_record.sha256 == second_record.sha256

    assert third_record.exact_group is None


def test_find_exact_duplicates_clears_old_groups(tmp_path):
    database = tmp_path / "test.db"

    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"

    first.write_bytes(b"same content")
    second.write_bytes(b"same content")

    db.init_db(database)

    for path in [first, second]:
        record = FileRecord(
            path=str(path),
            name=path.name,
            ext="txt",
            kind="document",
            size=path.stat().st_size,
            exact_group=99,
            sha256="old_hash",
        )
        db.upsert_file(record, db_path=database)

    # Change the second file so they are no longer duplicates.
    second.write_bytes(b"changed content")

    groups = find_exact_duplicates(db_path=database)

    assert groups == {}

    first_record = db.get_file_by_path(str(first), db_path=database)
    second_record = db.get_file_by_path(str(second), db_path=database)

    assert first_record.exact_group is None
    assert second_record.exact_group is None

def test_same_size_content_change_breaks_duplicate_group(tmp_path):
    database = tmp_path / "test.db"

    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"

    first.write_bytes(b"same content")
    second.write_bytes(b"same content")

    db.init_db(database)

    for path in [first, second]:
        record = FileRecord(
            path=str(path),
            name=path.name,
            ext="txt",
            kind="document",
            size=path.stat().st_size,
        )
        db.upsert_file(record, db_path=database)

    # Confirm they initially form a duplicate group.
    initial_groups = find_exact_duplicates(db_path=database)
    assert len(initial_groups) == 1

    # Change content without changing file size.
    second.write_bytes(b"new! content")

    assert first.stat().st_size == second.stat().st_size

    updated_groups = find_exact_duplicates(db_path=database)

    assert updated_groups == {}

    first_record = db.get_file_by_path(str(first), db_path=database)
    second_record = db.get_file_by_path(str(second), db_path=database)

    assert first_record.exact_group is None
    assert second_record.exact_group is None
    assert first_record.sha256 != second_record.sha256


def test_missing_file_does_not_crash_duplicate_detection(tmp_path):
    database = tmp_path / "test.db"

    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"

    first.write_bytes(b"same content")
    second.write_bytes(b"same content")

    db.init_db(database)

    for path in [first, second]:
        record = FileRecord(
            path=str(path),
            name=path.name,
            ext="txt",
            kind="document",
            size=path.stat().st_size,
        )
        db.upsert_file(record, db_path=database)

    # Simulate a file disappearing after scanning.
    second.unlink()

    groups = find_exact_duplicates(db_path=database)

    assert groups == {}

