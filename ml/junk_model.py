"""Dual-engine junk scoring model for ClutterIQ.

Combines:
1. Cold-start transparent heuristic rule engine (explainable baseline before labelling).
2. Active learning Random Forest classifier (trains on user Keep/Delete labels).
3. Human-readable reason explanation builder for both heuristics and ML predictions.
4. Model serialization and persistence via joblib.

Owned by Person B (Intelligence & Dashboard).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from config import DB_PATH, MODEL_PATH, STALE_DAYS
import db
from ml.features import FEATURE_COLUMNS, build_feature_table

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cold-Start Heuristic Rule Engine
# ---------------------------------------------------------------------------
def heuristic_score(row: pd.Series | Dict[str, Any]) -> Tuple[float, str]:
    """Calculate transparent, rule-based junk probability and explanation string.

    Rules:
    - +0.30: Non-keeper duplicate (exact, near, or semantic group without is_keeper)
    - +0.20: Stale file (unmodified for > STALE_DAYS)
    - +0.15: Blurry image (Laplacian variance below BLUR_THRESHOLD)
    - +0.15: Redundant naming convention ('copy', '(1)', 'untitled', etc.)
    - +0.10: Tiny file (< 10 KB)
    - +0.10: Located in temporary/downloads folder

    Capped at 1.0.

    Returns:
        Tuple[float, str]: (junk_score, human_readable_reason)
    """
    score = 0.0
    reasons: List[str] = []

    # 1. Non-keeper duplicate check
    in_dup = bool(
        row.get("in_exact_group", 0)
        or row.get("in_near_group", 0)
        or row.get("in_semantic_group", 0)
    )
    is_keeper = bool(row.get("is_keeper", 0))

    if in_dup and not is_keeper:
        score += 0.30
        reasons.append("duplicate non-keeper copy")

    # 2. Staleness check
    age = float(row.get("age_days", 0.0))
    if age > STALE_DAYS:
        score += 0.20
        reasons.append(f"unmodified for {int(age)} days")

    # 3. Blur / quality check
    if int(row.get("is_blurry", 0)) == 1:
        score += 0.15
        reasons.append("low sharpness / blurry")

    # 4. Copy naming check
    if int(row.get("name_copy_pattern", 0)) == 1:
        score += 0.15
        reasons.append("copy/draft filename convention")

    # 5. Tiny file check
    if int(row.get("tiny_file", 0)) == 1:
        score += 0.10
        reasons.append("tiny file (< 10 KB)")

    # 6. Temporary folder check
    if int(row.get("in_downloads_or_temp", 0)) == 1:
        score += 0.10
        reasons.append("in downloads/temp folder")

    # Normalize score between 0.0 and 1.0
    final_score = min(1.0, round(score, 2))
    reason_str = ", ".join(reasons) if reasons else "active clean file"

    return final_score, reason_str


# ---------------------------------------------------------------------------
# ML Explanation Builder
# ---------------------------------------------------------------------------
def explain_ml_prediction(
    row: pd.Series,
    feature_importances: np.ndarray,
    feature_names: List[str],
) -> str:
    """Generate a human-readable explanation sentence for an ML prediction.

    Combines global feature importances with local row-level values.
    """
    # Calculate feature contribution: importance * normalized presence
    contributions = []
    for feat, imp in zip(feature_names, feature_importances):
        val = row.get(feat, 0)
        if feat == "age_days" and val > STALE_DAYS:
            contributions.append((imp * 1.5, f"unmodified for {int(val)} days"))
        elif feat == "is_keeper" and val == 0 and (row.get("in_exact_group", 0) or row.get("in_near_group", 0)):
            contributions.append((imp * 2.0, "duplicate non-keeper"))
        elif feat == "name_copy_pattern" and val == 1:
            contributions.append((imp * 1.2, "copy filename pattern"))
        elif feat == "in_downloads_or_temp" and val == 1:
            contributions.append((imp * 1.1, "downloads/temp directory"))
        elif feat == "is_blurry" and val == 1:
            contributions.append((imp * 1.0, "low sharpness"))
        elif feat == "tiny_file" and val == 1:
            contributions.append((imp * 0.8, "tiny file"))

    # Sort by highest contribution
    contributions.sort(key=lambda x: x[0], reverse=True)
    top_reasons = [r[1] for r in contributions[:3]]

    if top_reasons:
        return "ML: " + ", ".join(top_reasons)
    return "ML: standard profile"


# ---------------------------------------------------------------------------
# Active Learning Model Training
# ---------------------------------------------------------------------------
def train_model(
    db_path: Path | str = DB_PATH,
    model_path: Path | str = MODEL_PATH,
    min_samples: int = 100,
) -> Optional[Dict[str, Any]]:
    """Train RandomForestClassifier on user-labelled rows ('keep' vs 'delete').

    Evaluates via 5-fold Stratified Cross-Validation and serializes to disk.

    Returns:
        Optional[Dict[str, Any]]: Dictionary of cross-validation metrics, or None
                                 if insufficient training labels exist.
    """
    features_df = build_feature_table(db_path=db_path)
    if features_df.empty:
        return None

    # Filter to labelled rows only
    labelled = features_df[features_df["user_label"].isin(["keep", "delete"])].copy()

    # Check minimum required sample volume
    if len(labelled) < min_samples:
        logger.info(
            f"Insufficient labels ({len(labelled)}/{min_samples}) for Random Forest. "
            "Continuing with heuristic baseline."
        )
        return None

    # Binary target: 1 = delete (junk), 0 = keep
    y = (labelled["user_label"] == "delete").astype(int).values

    # Check both classes are present
    if len(np.unique(y)) < 2:
        logger.info("Only one class represented in labels. Need both 'keep' and 'delete'.")
        return None

    X = labelled[FEATURE_COLUMNS].values

    # Initialize balanced Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation diagnostics (use min(5, class_count) folds)
    min_class_count = min(np.sum(y == 0), np.sum(y == 1))
    cv_folds = max(2, min(5, min_class_count))

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1"]
    cv_results = cross_validate(rf, X, y, cv=cv, scoring=scoring)

    # Train final model on full labelled set
    rf.fit(X, y)

    # Save model and feature columns metadata to disk
    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": rf,
        "feature_names": FEATURE_COLUMNS,
        "classes": rf.classes_,
    }
    joblib.dump(payload, str(model_file))
    logger.info(f"Model saved to {model_file} (trained on {len(labelled)} files).")

    metrics = {
        "accuracy": float(np.mean(cv_results["test_accuracy"])),
        "precision": float(np.mean(cv_results["test_precision"])),
        "recall": float(np.mean(cv_results["test_recall"])),
        "f1": float(np.mean(cv_results["test_f1"])),
        "n_samples": int(len(labelled)),
        "n_delete": int(np.sum(y == 1)),
        "n_keep": int(np.sum(y == 0)),
    }
    return metrics


# ---------------------------------------------------------------------------
# Scoring / Inference Engine
# ---------------------------------------------------------------------------
def predict_scores(
    db_path: Path | str = DB_PATH,
    model_path: Path | str = MODEL_PATH,
) -> int:
    """Compute and update junk_score and junk_reason for all active files in SQLite.

    Uses trained Random Forest if available; otherwise falls back to the transparent
    cold-start heuristic engine.

    Returns:
        int: Total number of files scored.
    """
    features_df = build_feature_table(db_path=db_path)
    if features_df.empty:
        return 0

    model_file = Path(model_path)
    trained_model = None
    feature_importances = None

    if model_file.exists():
        try:
            payload = joblib.load(str(model_file))
            trained_model = payload["model"]
            feature_importances = trained_model.feature_importances_
            logger.info(f"Using trained Random Forest from {model_file} for inference.")
        except Exception as e:
            logger.warning(f"Failed to load model from {model_file}: {e}. Using heuristics.")
            trained_model = None

    scored_count = 0

    if trained_model is not None:
        X = features_df[FEATURE_COLUMNS].values
        # Probability of class 1 ('delete')
        probs = trained_model.predict_proba(X)
        # Find index of class 1
        pos_class_idx = list(trained_model.classes_).index(1) if 1 in trained_model.classes_ else 1
        delete_probs = probs[:, pos_class_idx]

        for i, row in features_df.iterrows():
            file_id = int(row["file_id"])
            score = float(delete_probs[i])
            reason = explain_ml_prediction(row, feature_importances, FEATURE_COLUMNS)

            db.update_columns(
                file_id,
                db_path=db_path,
                junk_score=round(score, 3),
                junk_reason=reason,
            )
            scored_count += 1
    else:
        # Cold start heuristic fallback
        for _, row in features_df.iterrows():
            file_id = int(row["file_id"])
            score, reason = heuristic_score(row)

            db.update_columns(
                file_id,
                db_path=db_path,
                junk_score=score,
                junk_reason=reason,
            )
            scored_count += 1

    logger.info(f"Updated junk scores for {scored_count} files.")
    return scored_count


def retrain(
    db_path: Path | str = DB_PATH,
    model_path: Path | str = MODEL_PATH,
    min_samples: int = 100,
) -> Tuple[Optional[Dict[str, Any]], int]:
    """Retrain the model on updated user labels and re-score the database.

    Called when the user clicks 'Retrain Model' on the Suggestions page.

    Returns:
        Tuple[Optional[Dict], int]: (evaluation_metrics, count_of_files_rescored)
    """
    metrics = train_model(db_path=db_path, model_path=model_path, min_samples=min_samples)
    scored_count = predict_scores(db_path=db_path, model_path=model_path)
    return metrics, scored_count

