"""Unit test for feature engineering (ml/features.py).

Verifies:
1. Complete generation of all required FEATURE_COLUMNS.
2. Strict zero data leakage: asserts junk_score is never included.
3. Accuracy of engineered features (copy patterns, temp directories, size_log, blur).
4. Graceful handling of empty databases.
"""

import gc
import sys
from pathlib import Path
import tempfile
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BLUR_THRESHOLD
import db
from ml.features import (
    FEATURE_COLUMNS,
    build_feature_table,
    has_copy_pattern,
    is_in_temp_or_downloads,
)
from models import FileRecord


def test_feature_helpers():
    print("Testing feature helper functions...")

    # 1. Copy pattern detection
    assert has_copy_pattern("report (1).pdf") == 1
    assert has_copy_pattern("photo_copy.jpg") == 1
    assert has_copy_pattern("essay_final_final.docx") == 1
    assert has_copy_pattern("untitled.png") == 1
    assert has_copy_pattern("clean_resume.pdf") == 0
    assert has_copy_pattern("vacation_2024.jpg") == 0

    # 2. Temp and Downloads detection (folder component matching)
    assert is_in_temp_or_downloads(r"C:\Users\John\Downloads\setup.exe") == 1
    assert is_in_temp_or_downloads(r"/tmp/scratch.log") == 1
    assert is_in_temp_or_downloads(r"C:\Users\John\AppData\Local\Temp\file.tmp") == 1
    assert is_in_temp_or_downloads(r"C:\Users\John\Documents\work.txt") == 0
    # Must NOT false positive on 'Template' or 'Contemporary'
    assert is_in_temp_or_downloads(r"C:\Users\John\Projects\Template_Designs\file.psd") == 0

    print(" Feature helper tests passed!")


def test_build_feature_table():
    print("\nTesting build_feature_table() pipeline...")

    temp_dir_kwargs = {}
    if sys.version_info >= (3, 10):
        temp_dir_kwargs["ignore_cleanup_errors"] = True

    with tempfile.TemporaryDirectory(**temp_dir_kwargs) as tmpdir:
        tmp_path = Path(tmpdir)
        test_db = tmp_path / "test_clutteriq.db"
        db.init_db(test_db)

        # 1. Verify empty DB returns valid empty DataFrame with column schema
        empty_df = build_feature_table(db_path=test_db)
        assert empty_df.empty, "Expected empty DataFrame for empty DB"
        for col in FEATURE_COLUMNS:
            assert col in empty_df.columns, f"Missing feature column {col} in empty table schema"

        # 2. STRICT ZERO LEAKAGE CHECK:
        assert "junk_score" not in FEATURE_COLUMNS, "DATA LEAKAGE: junk_score found in FEATURE_COLUMNS!"
        assert "junk_reason" not in FEATURE_COLUMNS, "DATA LEAKAGE: junk_reason found in FEATURE_COLUMNS!"

        # 3. Create synthetic file records with known attributes
        now = time.time()

        # File 1: Stale, tiny copy in Downloads, blurry non-keeper
        r1 = FileRecord(
            path=r"C:\Users\User\Downloads\photo (1).jpg",
            name="photo (1).jpg",
            ext="jpg",
            kind="image",
            size=5000,  # < 10 KB (tiny)
            modified=now - (86400 * 400),  # 400 days old
            blur_score=50.0,  # Below BLUR_THRESHOLD (blurry)
            exact_group=10,
            is_keeper=0,
            user_label="delete",
        )

        # File 2: Recent, large, sharp keeper in Documents
        r2 = FileRecord(
            path=r"C:\Users\User\Documents\project_report.pdf",
            name="project_report.pdf",
            ext="pdf",
            kind="document",
            size=1024 * 1024 * 15,  # 15 MB
            modified=now - (86400 * 5),  # 5 days old
            blur_score=250.0,
            exact_group=10,
            is_keeper=1,
            user_label="keep",
        )

        id1 = db.upsert_file(r1, db_path=test_db)
        id2 = db.upsert_file(r2, db_path=test_db)

        # 4. Build feature table
        features_df = build_feature_table(db_path=test_db, current_time=now)
        assert len(features_df) == 2, f"Expected 2 rows, got {len(features_df)}"

        # Set file_id as index for convenient lookup
        features_df = features_df.set_index("file_id")

        row1 = features_df.loc[id1]
        row2 = features_df.loc[id2]

        # Verify File 1 features
        assert round(row1["age_days"]) == 400, f"Expected 400 age_days, got {row1['age_days']}"
        assert row1["tiny_file"] == 1, "Expected tiny_file == 1"
        assert row1["in_downloads_or_temp"] == 1, "Expected in_downloads_or_temp == 1"
        assert row1["name_copy_pattern"] == 1, "Expected name_copy_pattern == 1"
        assert row1["is_blurry"] == 1, "Expected is_blurry == 1"
        assert row1["in_exact_group"] == 1, "Expected in_exact_group == 1"
        assert row1["is_keeper"] == 0, "Expected is_keeper == 0"
        assert row1["user_label"] == "delete"

        # Verify File 2 features
        assert round(row2["age_days"]) == 5, f"Expected 5 age_days, got {row2['age_days']}"
        assert row2["tiny_file"] == 0, "Expected tiny_file == 0"
        assert row2["in_downloads_or_temp"] == 0, "Expected in_downloads_or_temp == 0"
        assert row2["name_copy_pattern"] == 0, "Expected name_copy_pattern == 0"
        assert row2["is_blurry"] == 0, "Expected is_blurry == 0"
        assert row2["is_keeper"] == 1, "Expected is_keeper == 1"
        assert row2["user_label"] == "keep"
        assert row2["size_log"] > row1["size_log"], "Expected size_log to scale with file size"

        del features_df, empty_df, row1, row2, r1, r2
        gc.collect()

    print(" Feature table pipeline tests passed successfully!")


if __name__ == "__main__":
    test_feature_helpers()
    test_build_feature_table()
