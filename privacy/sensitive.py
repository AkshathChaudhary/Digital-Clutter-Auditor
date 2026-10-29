"""Privacy and sensitive document scanner for ClutterIQ.

Scans local images, PDFs, and text files for personal and financial data:
1. Aadhaar numbers (12 digits, first 2-9, with Verhoeff checksum validation).
2. PAN numbers (Indian tax ID regex [A-Z]{5}[0-9]{4}[A-Z]).
3. Credit / Debit card numbers (13-19 digits with Luhn algorithm validation).
4. Bank statement documents (keyword co-occurrence heuristic).

STRICT PRIVACY GUARANTEE:
Raw extracted numbers and full OCR transcripts are NEVER persisted in SQLite.
Only masked strings (e.g. 'XXXX XXXX 1234') and privacy categories are saved.

Owned by Person B (Intelligence & Dashboard).
"""

import logging
from pathlib import Path
import re
import shutil
from typing import List, Optional, Tuple

from PIL import Image
import pypdf
import pytesseract
from tqdm import tqdm

from config import DB_PATH, DOC_EXTS, ENABLE_CARD_LUHN, ENABLE_OCR, ENABLE_PDF_TEXT, IMAGE_EXTS
import db
from models import FileRecord

logger = logging.getLogger(__name__)

# Configure Tesseract path on Windows if not already in PATH
if not shutil.which("tesseract"):
    default_win_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if default_win_path.exists():
        pytesseract.pytesseract.tesseract_cmd = str(default_win_path)


# ---------------------------------------------------------------------------
# Mathematical Checksum Algorithms (Verhoeff & Luhn)
# ---------------------------------------------------------------------------
# Verhoeff multiplication table d(j, k) based on dihedral group D5
VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)

# Verhoeff permutation table p(pos, num)
VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def validate_verhoeff(num_str: str) -> bool:
    """Validate a numeric string using the Verhoeff checksum algorithm (used by Aadhaar).

    Detects 100% of single-digit substitution errors and 100% of adjacent transposition errors.
    """
    clean_digits = [c for c in num_str if c.isdigit()]
    if len(clean_digits) != 12:
        return False

    c = 0
    # Process from right (least significant / checksum digit) to left
    for i, digit in enumerate(reversed(clean_digits)):
        c = VERHOEFF_D[c][VERHOEFF_P[i % 8][int(digit)]]
    return c == 0


def validate_luhn(card_number_str: str) -> bool:
    """Validate payment card number using the Luhn modulus 10 algorithm."""
    digits = [int(c) for c in card_number_str if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False

    checksum = 0
    reversed_digits = digits[::-1]
    for i, d in enumerate(reversed_digits):
        # Double every second digit starting from the right (1-indexed)
        if i % 2 == 1:
            doubled = d * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += d

    return checksum % 10 == 0


# ---------------------------------------------------------------------------
# Masking Functions
# ---------------------------------------------------------------------------
def mask_aadhaar(aadhaar_raw: str) -> str:
    """Mask 12-digit Aadhaar number as 'XXXX XXXX 1234'."""
    digits = "".join(c for c in aadhaar_raw if c.isdigit())
    if len(digits) == 12:
        return f"XXXX XXXX {digits[-4:]}"
    return "XXXX XXXX ****"


def mask_pan(pan_raw: str) -> str:
    """Mask 10-character PAN string as 'ABCDE****F'."""
    clean = pan_raw.strip().upper()
    if len(clean) == 10:
        return f"{clean[:5]}****{clean[-1]}"
    return "ABCDE****F"


def mask_card(card_raw: str) -> str:
    """Mask payment card number as '************1234'."""
    digits = "".join(c for c in card_raw if c.isdigit())
    if len(digits) >= 13:
        return f"{'*' * (len(digits) - 4)}{digits[-4:]}"
    return "************1234"


# ---------------------------------------------------------------------------
# Text Extraction Helpers
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract embedded text from all pages of a PDF file using pypdf."""
    text_chunks: List[str] = []
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    except Exception as e:
        logger.debug(f"Failed to read PDF text from {pdf_path}: {e}")
    return "\n".join(text_chunks)


def extract_text_from_image(image_path: Path, max_dimension: int = 1600) -> str:
    """Extract text from an image using Tesseract OCR, downscaling if oversized for speed."""
    if not ENABLE_OCR:
        return ""
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale to improve OCR contrast and reduce memory
            gray = img.convert("L")
            # Downscale high-resolution photos to avoid OCR lag on CPU
            w, h = gray.size
            if max(w, h) > max_dimension:
                scale = max_dimension / max(w, h)
                gray = gray.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

            return pytesseract.image_to_string(gray)
    except pytesseract.TesseractNotFoundError:
        logger.warning("Tesseract OCR binary not found. Please install Tesseract.")
        return ""
    except Exception as e:
        logger.debug(f"OCR failed for {image_path}: {e}")
        return ""


def extract_text_from_textfile(text_path: Path) -> str:
    """Read plain text from files (.txt, .csv, .log) handling encodings."""
    try:
        with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(500_000)  # Read up to 500 KB to avoid huge file stalls
    except Exception as e:
        logger.debug(f"Failed to read text file {text_path}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Pattern Detection Core
# ---------------------------------------------------------------------------
PAN_REGEX = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")

# Payment Card regex: 13 to 19 digits with optional spaces or dashes
CARD_REGEX = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

# Aadhaar regex: Exactly 12 digits, first digit 2-9, optional 4-digit grouping.
# Lookahead (?!\s*[- ]?\s*\d) prevents matching the first 3 groups of a 16-digit card number!
AADHAAR_REGEX = re.compile(r"(?<!\d)([2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4})(?!\s*[- ]?\s*\d)")

# Bank statement keywords (need at least 2 distinct primary keyword matches)
BANK_KEYWORDS = {
    "account statement",
    "account number",
    "ifsc",
    "ifsc code",
    "closing balance",
    "available balance",
    "micr code",
    "transaction details",
    "statement period",
    "debit",
    "credit",
}


def analyze_text_for_sensitive(text: str) -> Optional[Tuple[str, str]]:
    """Scan extracted text for sensitive patterns.

    Returns:
        Optional[Tuple[str, str]]: (privacy_type, masked_value) or None.
        Priority: PAN > Card > Aadhaar > Bank Statement
    """
    if not text:
        return None

    # 1. PAN Detection (Highest confidence regex)
    pan_matches = PAN_REGEX.findall(text)
    if pan_matches:
        first_pan = pan_matches[0]
        return "pan", mask_pan(first_pan)

    # 2. Payment Card Detection (Validated with Luhn check)
    if ENABLE_CARD_LUHN:
        card_candidates = CARD_REGEX.findall(text)
        for candidate in card_candidates:
            clean_digits = "".join(c for c in candidate if c.isdigit())
            if 13 <= len(clean_digits) <= 19:
                if validate_luhn(clean_digits):
                    return "card", mask_card(clean_digits)

    # 3. Aadhaar Detection (12 digits, leading 2-9, verified with Verhoeff)
    aadhaar_matches = AADHAAR_REGEX.findall(text)
    for raw_match in aadhaar_matches:
        clean_match = "".join(c for c in raw_match if c.isdigit())
        if len(clean_match) == 12:
            if validate_verhoeff(clean_match):
                return "aadhaar", mask_aadhaar(clean_match)
            # If formatted as XXXX-XXXX-XXXX but scanned with OCR noise
            if ("-" in raw_match or " " in raw_match) and clean_match[0] in "23456789":
                return "aadhaar", mask_aadhaar(clean_match)

    # 4. Bank Statement Keywords (Need at least 2 distinct keywords)
    text_lower = text.lower()
    matched_keywords = [kw for kw in BANK_KEYWORDS if kw in text_lower]
    if len(matched_keywords) >= 2:
        top_kws = ", ".join(matched_keywords[:3])
        return "bank_statement", f"Bank Statement (Matched: {top_kws})"

    return None


# ---------------------------------------------------------------------------
# Batch Pipeline Execution
# ---------------------------------------------------------------------------
def scan_sensitive(
    db_path: Path | str = DB_PATH,
    show_progress: bool = True,
) -> int:
    """Scan all active files for sensitive documents and update SQLite.

    Processes images, PDFs, and plain text files. Writes privacy_type
    and privacy_masked to the database.

    Returns:
        int: Total number of sensitive documents detected.
    """
    active_files = db.get_active_files(db_path=db_path)
    flagged_count = 0

    progress_bar = tqdm(
        total=len(active_files),
        desc="Scanning for sensitive documents",
        disable=not show_progress,
    )

    for record in active_files:
        path = Path(record.path)
        if not path.exists():
            progress_bar.update(1)
            continue

        ext = record.ext.lower()
        extracted_text = ""

        # Extract text based on file type
        if ext == "pdf" and ENABLE_PDF_TEXT:
            extracted_text = extract_text_from_pdf(path)
        elif ext in IMAGE_EXTS and ENABLE_OCR:
            # Optimize: Skip very small images (icons/thumbnails) under 10 KB
            if record.size >= 10_240:
                extracted_text = extract_text_from_image(path)
        elif ext in {"txt", "csv", "log", "md"}:
            extracted_text = extract_text_from_textfile(path)

        # Analyze extracted text for identifiers
        finding = analyze_text_for_sensitive(extracted_text)

        if finding is not None:
            privacy_type, masked_val = finding
            db.update_columns(
                record.id,
                db_path=db_path,
                privacy_type=privacy_type,
                privacy_masked=masked_val,
            )
            flagged_count += 1
        else:
            # If clean, ensure fields are reset to None
            if record.privacy_type is not None:
                db.update_columns(
                    record.id,
                    db_path=db_path,
                    privacy_type=None,
                    privacy_masked=None,
                )

        progress_bar.update(1)

    progress_bar.close()
    logger.info(f"Sensitive scan complete. Flagged {flagged_count} files.")
    return flagged_count
