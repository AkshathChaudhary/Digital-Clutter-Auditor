"""Unit test for junk scoring models (ml/junk_model.py).

Verifies:
1. Cold-start heuristic scoring logic and human-readable explanation generation.
2. RandomForestClassifier training on user labels and 5-fold cross-validation metrics.
3. ML inference, probability output, and feature importance explainability.
4. Model serialization with joblib.
"""

import gc
import sys
from pathlib import Path
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import db
from ml.junk_model import (
    heuristic_score,
    predict_scores,
    retrain,
    train_model,
)
from models import FileRecord


def test_heuristic_scoring():
    print("Testing cold-start heuristic scoring engine...")

    # Case 1: Extreme Junk (non-keeper duplicate, 400 days old, blurry, copy name, downloads, tiny)
    junk_row = {
        "in_exact_group": 1,
        "is_keeper": 0,
        "age_days": 400,
        "is_blurry": 1,
        "name_copy_pattern": 1,
        "in_downloads_or_temp": 1,
        "tiny_file": 1,
    }
    score, reason = heuristic_score(junk_row)
    assert score == 1.0, f"Expected capped score 1.0, got {score}"
    assert "duplicate non-keeper copy" in reason
    assert "unmodified for 400 days" in reason
    assert "blurry" in reason

    # Case 2: Clean Keeper (keeper, recent, sharp, normal name, documents)
    clean_row = {
        "in_exact_group": 1,
        "is_keeper": 1,
        "age_days": 10,
        "is_blurry": 0,
        "name_copy_pattern": 0,
        "in_downloads_or_temp": 0,
        "tiny_file": 0,
    }
    score_clean, reason_clean = heuristic_score(clean_row)
    assert score_clean == 0.0, f"Expected score 0.0, got {score_clean}"
    assert reason_clean == "active clean file"

    print(" Heuristic scoring tests passed!")


def test_ml_training_and_inference():
    print("\nTesting Active Learning Random Forest training & prediction...")

    temp_dir_kwargs = {}
    if sys.version_info >= (3, 10):
        temp_dir_kwargs["ignore_cleanup_errors"] = True

    with tempfile.TemporaryDirectory(**temp_dir_kwargs) as tmpdir:
        tmp_path = Path(tmpdir)
        test_db = tmp_path / "test_clutteriq.db"
        test_model = tmp_path / "junk_model.joblib"
        db.init_db(test_db)

        now = time.time()

        # Insert 15 'delete' examples and 15 'keep' examples to train on
        for i in range(15):
            # Junk files: old, copies, in downloads
            r_junk = FileRecord(
                path=rf"C:\Users\User\Downloads\junk_copy_{i} (1).png",
                name=f"junk_copy_{i} (1).png",
                ext="png",
                kind="image",
                size=2000,
                modified=now - (86400 * 500),
                blur_score=40.0,
                exact_group=100 + i,
                is_keeper=0,
                user_label="delete",
            )
            db.upsert_file(r_junk, db_path=test_db)

            # Keeper files: recent, sharp, documents
            r_keep = FileRecord(
                path=rf"C:\Users\User\Documents\important_doc_{i}.pdf",
                name=f"important_doc_{i}.pdf",
                ext="pdf",
                kind="document",
                size=1024 * 1024 * 5,
                modified=now - (86400 * 10),
                blur_score=300.0,
                exact_group=100 + i,
                is_keeper=1,
                user_label="keep",
            )
            db.upsert_file(r_keep, db_path=test_db)

        # Insert 1 unlabelled file to test prediction
        r_unlabelled = FileRecord(
            path=r"C:\Users\User\Downloads\mystery_file (2).png",
            name="mystery_file (2).png",
            ext="png",
            kind="image",
            size=1500,
            modified=now - (86400 * 450),
            blur_score=30.0,
            exact_group=200,
            is_keeper=0,
            user_label=None,
        )
        unlabelled_id = db.upsert_file(r_unlabelled, db_path=test_db)

        # 1. Cold start prediction without a model (should use heuristics)
        predict_scores(db_path=test_db, model_path=test_model)
        unlabelled_record = db.get_file_by_id(unlabelled_id, db_path=test_db)
        assert unlabelled_record.junk_score is not None
        assert unlabelled_record.junk_score >= 0.7
        assert "duplicate" in unlabelled_record.junk_reason

        # 2. Train model with min_samples=20
        metrics = train_model(db_path=test_db, model_path=test_model, min_samples=20)
        assert metrics is not None, "Failed to train Random Forest!"
        assert test_model.exists(), "Model file was not created on disk!"
        assert metrics["accuracy"] >= 0.85, f"Expected high accuracy on clean data, got {metrics['accuracy']}"
        print(f"  Trained RF Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")

        # 3. Predict with trained model
        predict_scores(db_path=test_db, model_path=test_model)
        rescored = db.get_file_by_id(unlabelled_id, db_path=test_db)
        assert rescored.junk_score > 0.6, f"Expected high junk probability from ML, got {rescored.junk_score}"
        assert rescored.junk_reason.startswith("ML:"), f"Expected ML prefix in reason, got {rescored.junk_reason}"
        print(f"  ML Predicted Junk Score: {rescored.junk_score} ({rescored.junk_reason})")

        # 4. Test retrain loop
        retrain_metrics, rescored_count = retrain(db_path=test_db, model_path=test_model, min_samples=20)
        assert retrain_metrics is not None
        assert rescored_count == 31  # 30 labelled + 1 unlabelled

        del unlabelled_record, rescored, r_junk, r_keep, r_unlabelled
        gc.collect()

    print("\n All Junk Model & Active Learning tests passed successfully!")


if __name__ == "__main__":
    test_heuristic_scoring()
    test_ml_training_and_inference()

