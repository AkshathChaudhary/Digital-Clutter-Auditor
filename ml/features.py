"""Tabular feature engineering pipeline for ClutterIQ.

Constructs invariant numerical and categorical feature vectors for active files
using filesystem metadata, deduplication group memberships, and image quality scores.

Guarantees:
- ZERO DATA LEAKAGE: Never includes `junk_score` or `user_label` in feature columns.
- Deterministic feature space for Scikit-Learn RandomForestClassifier.

Owned by Person B (Intelligence & Dashboard).
"""

import logging
from pathlib import Path
import re
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import BLUR_THRESHOLD, DB_PATH
import db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature Column Definitions
# ---------------------------------------------------------------------------
# Numerical and encoded categorical features passed directly into ML models
FEATURE_COLUMNS: List[str] = [
    "age_days",              # Days since last modification
    "days_since_access",     # Days since last access (weak signal on some OS)
    "size_log",              # Log10(size_in_bytes + 1)
    "folder_depth",          # Number of path directory levels
    "in_exact_group",        # Binary: in an exact SHA-256 group
    "in_near_group",         # Binary: in a pHash group
    "in_semantic_group",     # Binary: in a CLIP cluster
    "is_keeper",             # Binary: chosen as the best copy in its group
    "is_blurry",             # Binary: Laplacian variance below BLUR_THRESHOLD
    "name_copy_pattern",     # Binary: name matches 'copy', '(1)', 'untitled', etc.
    "in_downloads_or_temp",  # Binary: path contains downloads/temp/cache/tmp
    "tiny_file",             # Binary: size under 10 KB
    "ext_encoded",           # Integer: categorical index of file extension
    "kind_encoded",          # Integer: categorical index of file kind
]

# Top extensions assigned dedicated integer IDs; all others map to 0 ('other')
TOP_EXTENSIONS: List[str] = [
    "jpg", "jpeg", "png", "webp", "bmp",
    "pdf", "docx", "doc", "txt", "xlsx", "csv",
    "zip", "rar", "7z", "mp4", "mp3", "py"
]
EXT_MAP: Dict[str, int] = {ext: i + 1 for i, ext in enumerate(TOP_EXTENSIONS)}

# High-level file kinds
KINDS: List[str] = ["image", "document", "video", "audio", "archive", "code", "other"]
KIND_MAP: Dict[str, int] = {k: i for i, k in enumerate(KINDS)}

# Regex pattern for typical junk / redundant file names
COPY_PATTERN_REGEX = re.compile(
    r"(?:\bcopy\b|\(\d+\)|_copy|final_final|final_v\d|untitled)",
    re.IGNORECASE,
)

TEMP_FOLDER_NAMES = {"downloads", "temp", "tmp", "cache", "temporary internet files"}


# ---------------------------------------------------------------------------
# Feature Extraction Helpers
# ---------------------------------------------------------------------------
def has_copy_pattern(name: str) -> int:
    """Return 1 if filename matches common duplicate/draft conventions, else 0."""
    if not name:
        return 0
    return 1 if COPY_PATTERN_REGEX.search(name) else 0


def is_in_temp_or_downloads(path_str: str) -> int:
    """Return 1 if any parent directory is a known temp/download folder, else 0.

    Uses directory component matching to prevent false positives on names
    like 'Template' or 'Contemporary'.
    """
    if not path_str:
        return 0
    try:
        parts = [p.lower() for p in Path(path_str).parts]
        for part in parts:
            if part in TEMP_FOLDER_NAMES:
                return 1
        return 0
    except Exception:
        return 0


def calculate_folder_depth(path_str: str) -> int:
    """Return the directory depth of the file path."""
    try:
        return len(Path(path_str).parts)
    except Exception:
        return 1


# ---------------------------------------------------------------------------
# Full Feature Table Construction
# ---------------------------------------------------------------------------
def build_feature_table(
    db_path: Path | str = DB_PATH,
    current_time: Optional[float] = None,
) -> pd.DataFrame:
    """Query all active files from SQLite and compute the complete feature matrix.

    Returns:
        pd.DataFrame: Contains identifier columns ('file_id', 'path', 'name'),
                      all columns in `FEATURE_COLUMNS`, and target column 'user_label'.

    Raises:
        ValueError: If no active files exist in the database.
    """
    now = current_time if current_time is not None else time.time()

    # Fetch active rows into DataFrame
    query = """
    SELECT id, path, name, ext, kind, size, created, modified, accessed,
           blur_score, exact_group, near_group, semantic_group, is_keeper,
           user_label
    FROM files
    WHERE status = 'active'
    """
    df = db.fetch_df(query, db_path=db_path)

    if df.empty:
        logger.info("No active files found to build feature table.")
        # Return empty DataFrame with proper column structure
        cols = ["file_id", "path", "name"] + FEATURE_COLUMNS + ["user_label"]
        return pd.DataFrame(columns=cols)

    # 1. Temporal features
    modified_epoch = pd.to_numeric(df["modified"], errors="coerce").fillna(now)
    accessed_epoch = pd.to_numeric(df["accessed"], errors="coerce").fillna(modified_epoch)

    # Calculate days, clipping to 0 to prevent negative values from clock skew
    age_days = np.maximum(0.0, (now - modified_epoch) / 86400.0)
    days_since_access = np.maximum(0.0, (now - accessed_epoch) / 86400.0)

    # 2. Size features
    size_bytes = pd.to_numeric(df["size"], errors="coerce").fillna(0).astype(float)
    size_log = np.log10(size_bytes + 1.0)
    tiny_file = (size_bytes < 10_240).astype(int)

    # 3. Path and naming features
    paths = df["path"].fillna("").astype(str)
    names = df["name"].fillna("").astype(str)

    folder_depth = paths.apply(calculate_folder_depth)
    name_copy = names.apply(has_copy_pattern)
    temp_dir = paths.apply(is_in_temp_or_downloads)

    # 4. Deduplication group features
    in_exact = df["exact_group"].notna().astype(int)
    in_near = df["near_group"].notna().astype(int)
    in_semantic = df["semantic_group"].notna().astype(int)
    is_keeper = (df["is_keeper"] == 1).astype(int)

    # 5. Quality features
    blur = pd.to_numeric(df["blur_score"], errors="coerce")
    is_blurry = ((blur.notna()) & (blur < BLUR_THRESHOLD)).astype(int)

    # 6. Categorical encodings
    ext_clean = df["ext"].fillna("").str.lower().str.lstrip(".")
    ext_encoded = ext_clean.map(EXT_MAP).fillna(0).astype(int)

    kind_clean = df["kind"].fillna("other").str.lower()
    kind_encoded = kind_clean.map(KIND_MAP).fillna(KIND_MAP["other"]).astype(int)

    # Assemble structured DataFrame
    features_df = pd.DataFrame({
        # Identifiers (metadata, not model features)
        "file_id": df["id"].astype(int),
        "path": paths,
        "name": names,

        # Engineered features
        "age_days": age_days,
        "days_since_access": days_since_access,
        "size_log": size_log,
        "folder_depth": folder_depth,
        "in_exact_group": in_exact,
        "in_near_group": in_near,
        "in_semantic_group": in_semantic,
        "is_keeper": is_keeper,
        "is_blurry": is_blurry,
        "name_copy_pattern": name_copy,
        "in_downloads_or_temp": temp_dir,
        "tiny_file": tiny_file,
        "ext_encoded": ext_encoded,
        "kind_encoded": kind_encoded,

        # Supervised target (for active learning training, None for unlabelled)
        "user_label": df["user_label"],
    })

    return features_df
