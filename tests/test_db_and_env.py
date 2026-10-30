"""Test script for db.py and Person B environment prerequisites.

Verifies:
1. SQLite schema initialization and indexing.
2. FileRecord insertion, upsert conflict handling, and column updates.
3. Querying via fetch_df and files_by_kind.
4. Python environment readiness (torch, sentence_transformers, pytesseract, pypdf, sklearn).
"""

import gc
import sys
from pathlib import Path
import tempfile
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import FileRecord
import db


def test_database_crud():
    print("Testing Database CRUD operations...")
    # Use ignore_cleanup_errors=True for robust Windows cleanup if available
    temp_dir_kwargs = {}
    if sys.version_info >= (3, 10):
        temp_dir_kwargs["ignore_cleanup_errors"] = True

    with tempfile.TemporaryDirectory(**temp_dir_kwargs) as tmpdir:
        test_db_path = Path(tmpdir) / "test_clutteriq.db"

        # 1. Initialize schema
        db.init_db(test_db_path)
        assert test_db_path.exists(), "DB file was not created!"

        # 2. Upsert a new FileRecord
        now = time.time()
        record = FileRecord(
            path=str(Path(tmpdir) / "photos" / "vacation1.jpg"),
            name="vacation1.jpg",
            ext="jpg",
            kind="image",
            size=1024 * 500,
            created=now - 86400 * 30,
            modified=now - 86400 * 10,
            accessed=now,
        )
        row_id = db.upsert_file(record, db_path=test_db_path)
        assert row_id > 0, f"Expected positive row_id, got {row_id}"
        assert record.id == row_id

        # 3. Fetch record by ID
        fetched = db.get_file_by_id(row_id, db_path=test_db_path)
        assert fetched is not None, "Failed to retrieve record by ID"
        assert fetched.name == "vacation1.jpg"
        assert fetched.kind == "image"
        assert fetched.size == 1024 * 500

        # 4. Update columns (Person B operations: semantic_group, privacy_type, junk_score)
        db.update_columns(
            row_id,
            db_path=test_db_path,
            semantic_group=42,
            privacy_type="pan",
            privacy_masked="ABCDE****F",
            junk_score=0.85,
            junk_reason="High semantic similarity duplicate",
        )

        updated = db.get_file_by_id(row_id, db_path=test_db_path)
        assert updated.semantic_group == 42
        assert updated.privacy_type == "pan"
        assert updated.privacy_masked == "ABCDE****F"
        assert updated.junk_score == 0.85
        assert updated.junk_reason == "High semantic similarity duplicate"

        # 5. Filter by kind
        images = db.files_by_kind("image", db_path=test_db_path)
        assert len(images) == 1
        assert images[0].id == row_id

        docs = db.files_by_kind("document", db_path=test_db_path)
        assert len(docs) == 0

        # 6. Fetch via pandas DataFrame
        df = db.fetch_df("SELECT id, name, semantic_group, junk_score FROM files", db_path=test_db_path)
        assert len(df) == 1
        assert df.iloc[0]["semantic_group"] == 42
        assert round(df.iloc[0]["junk_score"], 2) == 0.85

        # Explicit garbage collection before directory removal on Windows
        del df, fetched, updated, images, docs, record
        gc.collect()

    print(" Database CRUD tests passed successfully!")


def test_environment_libraries():
    print("\nChecking Person B Python environment libraries...")
    libraries = [
        ("torch", "PyTorch"),
        ("sentence_transformers", "Sentence Transformers (CLIP)"),
        ("pytesseract", "PyTesseract OCR"),
        ("pypdf", "PyPDF Text Extractor"),
        ("sklearn", "Scikit-Learn"),
        ("streamlit", "Streamlit"),
        ("plotly", "Plotly"),
        ("imagehash", "ImageHash"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
    ]

    missing = []
    for module_name, display_name in libraries:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  {display_name} ({module_name}) -> version {ver}")
        except ImportError as e:
            missing.append((module_name, display_name, str(e)))
            print(f"  FAILED: {display_name} ({module_name}) -> {e}")

    if missing:
        print(f"\n Missing {len(missing)} libraries. Please run: pip install -r requirements.txt")
    else:
        print("\n All required Person B libraries are installed and importable!")


if __name__ == "__main__":
    test_database_crud()
    test_environment_libraries()
