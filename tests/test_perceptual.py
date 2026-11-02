
from PIL import Image
from dedup.perceptual import compare_phashes, group_by_phash
from pathlib import Path

from dedup.perceptual import compute_phash
from dedup.perceptual import find_near_duplicates
import db
from models import FileRecord


def test_compute_phash_returns_string(tmp_path):
    image_path = tmp_path / "sample.png"

    image = Image.new("RGB", (100, 100), color="red")
    image.save(image_path)

    result = compute_phash(image_path)

    assert isinstance(result, str)
    assert len(result) == 16


def test_identical_images_have_identical_phashes(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"

    image = Image.new("RGB", (100, 100), color="blue")
    image.save(first)
    image.save(second)

    assert compute_phash(first) == compute_phash(second)


def test_different_images_can_have_different_phashes(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"

    Image.new("RGB", (100, 100), color="black").save(first)
    Image.new("RGB", (100, 100), color="white").save(second)

    assert compute_phash(first) != compute_phash(second)

def test_compare_phashes_identical():
    hash_a = "0000000000000000"
    hash_b = "0000000000000000"

    assert compare_phashes(hash_a, hash_b) == 0


def test_compare_phashes_one_bit_difference():
    hash_a = "0000000000000000"
    hash_b = "0000000000000001"

    assert compare_phashes(hash_a, hash_b) == 1


def test_group_by_phash_finds_similar_images():
    first = Path("first.png")
    second = Path("second.png")
    third = Path("third.png")

    phashes = {
        first: "0000000000000000",
        second: "0000000000000001",
        third: "0000000000000003",
    }

    groups = group_by_phash(phashes, max_distance=2)

    assert len(groups) == 1
    assert set(next(iter(groups.values()))) == {
        first, second, third
    }


def test_group_by_phash_ignores_unrelated_images():
    first = Path("first.png")
    second = Path("second.png")

    phashes = {
        first: "0000000000000000",
        second: "ffffffffffffffff",
    }

    groups = group_by_phash(phashes, max_distance=6)

    assert groups == {}

def test_corrupt_image_does_not_crash_detection(tmp_path):
    database = tmp_path / "test.db"

    valid_image = tmp_path / "valid.png"
    corrupt_image = tmp_path / "corrupt.png"

    Image.new("RGB", (100, 100), "blue").save(valid_image)
    corrupt_image.write_bytes(b"This is not a real image.")

    db.init_db(database)

    for path in [valid_image, corrupt_image]:
        record = FileRecord(
            path=str(path),
            name=path.name,
            ext="png",
            kind="image",
            size=path.stat().st_size,
        )
        db.upsert_file(record, db_path=database)

    groups = find_near_duplicates(db_path=database)

    assert groups == {}

    valid_record = db.get_file_by_path(
        str(valid_image), db_path=database
    )
    corrupt_record = db.get_file_by_path(
        str(corrupt_image), db_path=database
    )

    assert valid_record.phash is not None
    assert corrupt_record.phash is None

def test_changed_image_clears_old_near_group(tmp_path):
    database = tmp_path / "test.db"

    first = tmp_path / "first.png"
    second = tmp_path / "second.png"

    Image.new("RGB", (100, 100), "blue").save(first)
    Image.new("RGB", (100, 100), "blue").save(second)

    db.init_db(database)

    for path in [first, second]:
        record = FileRecord(
            path=str(path),
            name=path.name,
            ext="png",
            kind="image",
            size=path.stat().st_size,
            near_group=99,
            phash="old_hash",
        )
        db.upsert_file(record, db_path=database)

    initial_groups = find_near_duplicates(db_path=database)
    assert len(initial_groups) == 1

    # Replace the second image with a visually different image.
    from PIL import ImageDraw
    img = Image.new("RGB", (100, 100), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 50, 50], fill="black")
    img.save(second)

    updated_groups = find_near_duplicates(db_path=database)

    assert updated_groups == {}

    first_record = db.get_file_by_path(str(first), db_path=database)
    second_record = db.get_file_by_path(str(second), db_path=database)

    assert first_record.near_group is None
    assert second_record.near_group is None
