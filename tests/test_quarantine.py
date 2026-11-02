import pytest

import db
from safety.quarantine import quarantine_file, restore_file
from dedup.exact import compute_sha256
from models import FileRecord


def create_test_file(tmp_path, name="sample.txt", content="ClutterIQ test data"):
    file_path = tmp_path / name
    file_path.write_text(content, encoding="utf-8")
    return file_path

def test_quarantine_and_restore(tmp_path, monkeypatch):
    # Arrange: create an isolated test environment
    original = create_test_file(tmp_path)

    test_db = tmp_path / "test.db"
    quarantine_dir = tmp_path / "quarantine"

    monkeypatch.setattr(
        "safety.quarantine.QUARANTINE_DIR",
        quarantine_dir
    )

    record = FileRecord(
        path=str(original),
        name=original.name,
        ext=original.suffix.lower().lstrip('.'),
        kind="document",
        size=original.stat().st_size,
        created=original.stat().st_ctime,
        modified=original.stat().st_mtime,
        accessed=original.stat().st_atime,
    )

    db.init_db(test_db)
    db.upsert_file(record, db_path=test_db)

    saved_record = db.get_active_files(test_db)[0]

    # Act: quarantine the file
    quarantined_path = quarantine_file(
        saved_record.id,
        db_path=test_db
    )

    # Assert: original is gone from its original location
    assert not original.exists()
    assert quarantined_path.exists()

    # Get the quarantine log entry
    connection = db.get_conn(test_db)
    try:
        row = connection.execute(
            "SELECT id FROM quarantine_log WHERE file_id = ?",
            (saved_record.id,)
        ).fetchone()
    finally:
        connection.close()

    assert row is not None

    # Act: restore the file
    restored_path = restore_file(
        row[0],
        db_path=test_db
    )

    # Assert: original content is restored
    assert restored_path == original
    assert original.read_text(encoding="utf-8") == "ClutterIQ test data"
    assert not quarantined_path.exists()

    restored_record = db.get_file_by_id(
        saved_record.id,
        db_path=test_db
    )
    assert restored_record.status == "restored"

def test_quarantine_missing_file_raises(tmp_path):
    test_db = tmp_path / "test.db"

    db.init_db(test_db)

    with pytest.raises(ValueError):
        quarantine_file(9999, db_path=test_db)

def test_restore_refuses_existing_original(tmp_path, monkeypatch):
    original = create_test_file(tmp_path)
    test_db = tmp_path / "test.db"

    quarantine_dir = tmp_path / "quarantine"
    monkeypatch.setattr(
        "safety.quarantine.QUARANTINE_DIR",
        quarantine_dir
    )

    record = FileRecord(
        path=str(original),
        name=original.name,
        ext=original.suffix.lower().lstrip('.'),
        kind="document",
        size=original.stat().st_size,
        created=original.stat().st_ctime,
        modified=original.stat().st_mtime,
        accessed=original.stat().st_atime,
    )

    db.init_db(test_db)
    db.upsert_file(record, db_path=test_db)
    saved_record = db.get_active_files(test_db)[0]

    quarantined_path = quarantine_file(
        saved_record.id,
        db_path=test_db
    )

    # Simulate another file appearing at the original location
    original.write_text("Do not overwrite this", encoding="utf-8")

    connection = db.get_conn(test_db)
    try:
        row = connection.execute(
            "SELECT id FROM quarantine_log WHERE file_id = ?",
            (saved_record.id,)
        ).fetchone()
    finally:
        connection.close()

    with pytest.raises(FileExistsError):
        restore_file(row[0], db_path=test_db)

    # Both files must remain safe
    assert original.read_text(encoding="utf-8") == "Do not overwrite this"
    assert quarantined_path.exists()