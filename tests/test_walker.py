from pathlib import Path

import pytest
import scanner.walker as walker


def test_empty_directory(tmp_path):
    """An empty directory should produce no files."""
    errors = []

    files = list(walker.iter_files(tmp_path, errors))

    assert files == []
    assert errors == []


def test_nested_files_discovered(tmp_path):
    """The scanner should find files inside nested directories."""
    nested_folder = tmp_path / "folder_a" / "folder_b"
    nested_folder.mkdir(parents=True)

    file_one = tmp_path / "first.txt"
    file_two = nested_folder / "second.txt"

    file_one.write_text("Hello")
    file_two.write_text("ClutterIQ")

    files = list(walker.iter_files(tmp_path))

    discovered_paths = {file.resolve() for file in files}

    assert file_one.resolve() in discovered_paths
    assert file_two.resolve() in discovered_paths
    assert len(files) == 2


def test_unicode_filename(tmp_path):
    """The scanner should support filenames containing Unicode characters."""
    unicode_file = tmp_path / "আমার_ছবি.jpg"
    unicode_file.write_text("test image placeholder")

    files = list(walker.iter_files(tmp_path))

    assert len(files) == 1
    assert files[0].name == "আমার_ছবি.jpg"


def test_nonexistent_directory_raises_error(tmp_path):
    """A directory that doesn't exist should raise FileNotFoundError."""
    missing_folder = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError):
        list(walker.iter_files(missing_folder))


def test_file_instead_of_directory_raises_error(tmp_path):
    """The scanner should reject a file passed as the root directory."""
    file_path = tmp_path / "not_a_folder.txt"
    file_path.write_text("This is a file")

    with pytest.raises(NotADirectoryError):
        list(walker.iter_files(file_path))


def test_scan_directory_returns_records(tmp_path):
    """scan_directory should return FileRecord objects with metadata."""
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("ClutterIQ testing")

    records, errors = walker.scan_directory(tmp_path)

    assert len(records) == 1
    assert errors == []

    record = records[0]

    assert record.name == "sample.txt"
    assert record.ext == "txt"
    assert record.kind == "document"
    assert record.size == len("ClutterIQ testing".encode("utf-8"))
    assert record.path == str(sample_file.resolve())


def test_scan_directory_handles_file_errors(tmp_path, monkeypatch):
    """One unreadable file should not stop the rest of the scan."""
    good_file = tmp_path / "good.txt"
    bad_file = tmp_path / "bad.txt"

    good_file.write_text("This file works")
    bad_file.write_text("Simulated error")

    original_function = walker.create_file_record

    def simulated_failure(file_path):
        if file_path.name == "bad.txt":
            raise PermissionError("Simulated permission error")

        return original_function(file_path)

    monkeypatch.setattr(
        walker,
        "create_file_record",
        simulated_failure
    )

    records, errors = walker.scan_directory(tmp_path)

    assert len(records) == 1
    assert records[0].name == "good.txt"

    assert len(errors) == 1
    assert errors[0]["path"] == str(bad_file)