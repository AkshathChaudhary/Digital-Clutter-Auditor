"""ClutterIQ configuration constants.

All thresholds, paths, and extension sets live here so they can be tuned
without touching the pipeline code. Import as: from config import DB_PATH, PHASH_MAX_DISTANCE, ...
"""

import os
from pathlib import Path
import stat

# ---------------------------------------------------------------------------
# Paths (all relative to project root; created at runtime if missing)
# ---------------------------------------------------------------------------
DB_PATH = Path("data") / "clutteriq.db"
EMB_PATH = Path("data") / "embeddings.npy"
MODEL_PATH = Path("data") / "junk_model.joblib"
QUARANTINE_DIR = Path("quarantine")

# ---------------------------------------------------------------------------
# File-type classification (lowercase, no dot)
# ---------------------------------------------------------------------------
IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "bmp", "gif", "tiff"}
DOC_EXTS = {"pdf", "doc", "docx", "txt", "xlsx", "pptx", "csv"}
VIDEO_EXTS = {"mp4", "mov", "mkv", "avi", "webm"}
AUDIO_EXTS = {"mp3", "wav", "flac", "m4a", "ogg"}
ARCHIVE_EXTS = {"zip", "rar", "7z", "tar", "gz"}
CODE_EXTS = {"py", "js", "ts", "java", "cpp", "c", "h", "rs", "go", "json", "yaml", "yml", "toml", "ini", "cfg", "md", "html", "css", "sql"}

# ---------------------------------------------------------------------------
# Deduplication thresholds (tune on benchmark data)
# ---------------------------------------------------------------------------
PHASH_MAX_DISTANCE = 6          # Hamming distance for near-duplicates (pHash)
CLIP_SIM_THRESHOLD = 0.92       # Cosine similarity for semantic groups (CLIP)
MIN_SIZE_FOR_HASH = 1           # Minimum file size (bytes) to attempt hashing

# ---------------------------------------------------------------------------
# Image quality thresholds
# ---------------------------------------------------------------------------
BLUR_THRESHOLD = 100.0          # Laplacian variance below this => blurry

# ---------------------------------------------------------------------------
# Staleness / heuristic junk scoring
# ---------------------------------------------------------------------------
STALE_DAYS = 365                # Files older than this get a staleness boost

# ---------------------------------------------------------------------------
# CLIP embedding settings
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME = "clip-ViT-B-32"
CLIP_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Privacy detection (regex patterns live in privacy/sensitive.py;
# this file only holds toggles)
# ---------------------------------------------------------------------------
ENABLE_OCR = True
ENABLE_PDF_TEXT = True
ENABLE_CARD_LUHN = True

# ---------------------------------------------------------------------------
# Helper: classify a file extension into a high-level kind
# ---------------------------------------------------------------------------
def kind_from_ext(ext: str) -> str:
    """Return one of: image|document|video|audio|archive|code|other"""
    ext = ext.lower().lstrip(".")
    if ext in IMAGE_EXTS:
        return "image"
    if ext in DOC_EXTS:
        return "document"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in ARCHIVE_EXTS:
        return "archive"
    if ext in CODE_EXTS:
        return "code"
    return "other"


# ---------------------------------------------------------------------------
# Cloud stub detection (OneDrive / iCloud dehydrated placeholders)
# ---------------------------------------------------------------------------
FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
FILE_ATTRIBUTE_OFFLINE = 0x1000


def is_cloud_stub(file_path: str | Path) -> bool:
    """Check if file is a OneDrive/iCloud dehydrated placeholder."""
    try:
        attrs = os.stat(str(file_path)).st_file_attributes
        return bool(attrs & (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_OFFLINE))
    except (AttributeError, OSError):
        return False  # Non-Windows OS or file inaccessible
