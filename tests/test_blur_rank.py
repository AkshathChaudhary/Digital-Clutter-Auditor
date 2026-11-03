from pathlib import Path
import db

from quality.blur_rank import choose_keepers

import pytest
from PIL import Image

from models import FileRecord
from quality.blur_rank import (
    calculate_blur_score,
    calculate_image_metrics,
    rank_image,
    select_keeper,
    build_duplicate_clusters,
)

@pytest.fixture
def sample_image(tmp_path):
    image_path = tmp_path / "sample.png"

    image = Image.new("RGB", (200, 100), "white")
    image.save(image_path)

    return image_path

def test_calculate_blur_score_returns_float(sample_image):
    score = calculate_blur_score(sample_image)

    assert isinstance(score, float)
    assert score >= 0


def test_calculate_blur_score_invalid_image(tmp_path):
    invalid_path = tmp_path / "invalid.png"
    invalid_path.write_text("not an image")

    with pytest.raises(ValueError):
        calculate_blur_score(invalid_path)


def test_calculate_image_metrics(sample_image):
    width, height, score = calculate_image_metrics(sample_image)

    assert width == 200
    assert height == 100
    assert isinstance(score, float)

def make_record(file_id, width, height, blur_score, path):
    return FileRecord(
        id=file_id,
        path=str(path),
        name=Path(path).name,
        kind="image",
        width=width,
        height=height,
        blur_score=blur_score,
    )

def test_rank_image_prefers_higher_resolution(tmp_path):
    image_a = make_record(1, 1000, 1000, 200, tmp_path / "a.png")
    image_b = make_record(2, 2000, 1000, 50, tmp_path / "b.png")

    assert rank_image(image_b) > rank_image(image_a)


def test_rank_image_uses_sharpness_for_tie(tmp_path):
    image_a = make_record(1, 1000, 1000, 80, tmp_path / "a.png")
    image_b = make_record(2, 1000, 1000, 150, tmp_path / "b.png")

    assert rank_image(image_b) > rank_image(image_a)


def test_select_keeper_returns_best_record(tmp_path):
    image_a = make_record(1, 1000, 1000, 80, tmp_path / "a.png")
    image_b = make_record(2, 2000, 1000, 150, tmp_path / "b.png")

    assert select_keeper([image_a, image_b]) == image_b


def test_select_keeper_empty_group():
    assert select_keeper([]) is None

def test_build_duplicate_clusters_combines_overlapping_groups(tmp_path):
    image_a = make_record(1, 100, 100, 50, tmp_path / "a.png")
    image_b = make_record(2, 100, 100, 60, tmp_path / "b.png")
    image_c = make_record(3, 100, 100, 70, tmp_path / "c.png")

    image_a.exact_group = 1
    image_b.exact_group = 1

    image_b.near_group = 2
    image_c.near_group = 2

    clusters = build_duplicate_clusters([image_a, image_b, image_c])

    assert len(clusters) == 1
    assert {record.id for record in clusters[0]} == {1, 2, 3}


def test_build_duplicate_clusters_keeps_separate_groups(tmp_path):
    image_a = make_record(1, 100, 100, 50, tmp_path / "a.png")
    image_b = make_record(2, 100, 100, 60, tmp_path / "b.png")
    image_c = make_record(3, 100, 100, 70, tmp_path / "c.png")
    image_d = make_record(4, 100, 100, 80, tmp_path / "d.png")

    image_a.exact_group = 1
    image_b.exact_group = 1

    image_c.near_group = 1
    image_d.near_group = 1

    clusters = build_duplicate_clusters(
        [image_a, image_b, image_c, image_d]
    )

    assert len(clusters) == 2
def test_choose_keepers_updates_database(tmp_path):
    db_path = tmp_path / "test.db"
    db.init_db(db_path)

    image_a_path = tmp_path / "image_a.png"
    image_b_path = tmp_path / "image_b.png"
    image_c_path = tmp_path / "image_c.png"

    Image.new("RGB", (100, 100), "blue").save(image_a_path)
    Image.new("RGB", (200, 100), "blue").save(image_b_path)
    Image.new("RGB", (50, 50), "red").save(image_c_path)

    image_a = FileRecord(
        path=str(image_a_path),
        name="image_a.png",
        ext="png",
        kind="image",
        exact_group=1,
    )

    image_b = FileRecord(
        path=str(image_b_path),
        name="image_b.png",
        ext="png",
        kind="image",
        exact_group=1,
    )

    image_c = FileRecord(
        path=str(image_c_path),
        name="image_c.png",
        ext="png",
        kind="image",
    )

    db.upsert_file(image_a, db_path)
    db.upsert_file(image_b, db_path)
    db.upsert_file(image_c, db_path)

    selected = choose_keepers(db_path)

    records = db.files_by_kind("image", db_path=db_path)

    record_by_name = {record.name: record for record in records}

    assert image_b.id in selected
    assert record_by_name["image_b.png"].is_keeper == 1
    assert record_by_name["image_a.png"].is_keeper == 0
    assert record_by_name["image_c.png"].is_keeper == 0

    assert record_by_name["image_b.png"].width == 200
    assert record_by_name["image_b.png"].height == 100
    assert record_by_name["image_b.png"].blur_score is not None