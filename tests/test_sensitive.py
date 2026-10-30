"""Unit test for privacy and sensitive document detection (privacy/sensitive.py).

Verifies:
1. Luhn checksum algorithm (positive and negative card numbers).
2. Verhoeff checksum algorithm (positive and negative Aadhaar numbers).
3. PAN pattern detection and masking.
4. Bank statement keyword co-occurrence detection.
5. End-to-end scan_sensitive execution with SQLite masking verification.
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
from models import FileRecord
from privacy.sensitive import (
    VERHOEFF_D,
    VERHOEFF_P,
    analyze_text_for_sensitive,
    mask_aadhaar,
    mask_card,
    mask_pan,
    scan_sensitive,
    validate_luhn,
    validate_verhoeff,
)


def compute_verhoeff_checksum(prefix_digits: str) -> str:
    """Helper to calculate the Verhoeff check-digit for a numeric prefix."""
    inv = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)
    c = 0
    for i, digit in enumerate(reversed(prefix_digits)):
        c = VERHOEFF_D[c][VERHOEFF_P[(i + 1) % 8][int(digit)]]
    return str(inv[c])


def test_checksum_algorithms():
    print("Testing Luhn & Verhoeff checksum algorithms...")

    # 1. Luhn algorithm (Valid 16-digit card number vs Invalid)
    valid_card = "4532015112830366"
    assert validate_luhn(valid_card) is True, "Valid 16-digit Luhn card failed!"

    # Change check digit to invalidate (6 -> 7)
    invalid_card = "4532015112830367"
    assert validate_luhn(invalid_card) is False, "Invalid Luhn card should fail!"

    # Numbers under 13 digits must fail card validation
    assert validate_luhn("49927398716") is False, "Card under 13 digits should fail!"

    # 2. Verhoeff algorithm (Aadhaar 12-digit validation)
    # Generate a mathematically valid 12-digit Aadhaar starting with 2-9
    prefix = "23456789012"  # 11 digits
    check_digit = compute_verhoeff_checksum(prefix)
    valid_aadhaar = prefix + check_digit

    assert validate_verhoeff(valid_aadhaar) is True, f"Valid Verhoeff number {valid_aadhaar} failed!"

    # Invalidate by altering the checksum digit
    corrupted_digit = str((int(check_digit) + 1) % 10)
    invalid_aadhaar = prefix + corrupted_digit
    assert validate_verhoeff(invalid_aadhaar) is False, "Corrupted Verhoeff check-digit should fail!"

    # Invalidate by transposing adjacent digits (e.g. 23 -> 32)
    transposed = "32" + prefix[2:] + check_digit
    assert validate_verhoeff(transposed) is False, "Transposed adjacent digits must fail Verhoeff!"

    print(" Checksum algorithm unit tests passed!")


def test_pattern_detection_and_masking():
    print("\nTesting pattern detection and masking rules...")

    # 1. PAN detection and masking
    text_pan = "Customer Tax Identifier is ABCDE1234F for filing."
    finding = analyze_text_for_sensitive(text_pan)
    assert finding is not None, "Failed to detect PAN!"
    p_type, masked = finding
    assert p_type == "pan"
    assert masked == "ABCDE****F", f"Expected ABCDE****F, got {masked}"
    assert "1234" not in masked

    # 2. Card detection and masking
    text_card = "Please charge payment card 4532-0151-1283-0366 for the order."
    finding_card = analyze_text_for_sensitive(text_card)
    assert finding_card is not None, "Failed to detect card!"
    c_type, c_masked = finding_card
    assert c_type == "card"
    assert c_masked.endswith("0366")
    assert "*" in c_masked

    # 3. Aadhaar detection and masking
    prefix = "23456789012"
    valid_aadhaar = prefix + compute_verhoeff_checksum(prefix)
    formatted_aadhaar = f"{valid_aadhaar[:4]} {valid_aadhaar[4:8]} {valid_aadhaar[8:]}"
    text_aadhaar = f"Government ID verified: {formatted_aadhaar} in records."
    finding_aadhaar = analyze_text_for_sensitive(text_aadhaar)
    assert finding_aadhaar is not None, "Failed to detect Aadhaar!"
    a_type, a_masked = finding_aadhaar
    assert a_type == "aadhaar"
    assert a_masked == f"XXXX XXXX {valid_aadhaar[-4:]}"

    # 4. Bank statement detection
    text_bank = "Your official Account Statement. IFSC: HDFC0001234. Closing Balance: Rs 54,000."
    finding_bank = analyze_text_for_sensitive(text_bank)
    assert finding_bank is not None, "Failed to detect bank statement!"
    b_type, b_masked = finding_bank
    assert b_type == "bank_statement"
    assert "account statement" in b_masked.lower() or "ifsc" in b_masked.lower()

    # 5. Clean document (negative test)
    text_clean = "Shopping list: Apples, Bananas, Milk, Bread, Eggs."
    finding_clean = analyze_text_for_sensitive(text_clean)
    assert finding_clean is None, "Clean document was falsely flagged as sensitive!"

    print(" Detection and masking tests passed!")


def test_end_to_end_privacy_scan():
    print("\nTesting end-to-end scan_sensitive() on files...")

    temp_dir_kwargs = {}
    if sys.version_info >= (3, 10):
        temp_dir_kwargs["ignore_cleanup_errors"] = True

    with tempfile.TemporaryDirectory(**temp_dir_kwargs) as tmpdir:
        tmp_path = Path(tmpdir)
        test_db = tmp_path / "test_clutteriq.db"
        db.init_db(test_db)

        # Create sample files
        f1_path = tmp_path / "tax_doc.txt"
        f1_path.write_text("Tax Assessment. PAN: BKRPK9876Q. Keep safe.")

        f2_path = tmp_path / "receipt.txt"
        f2_path.write_text("Paid using card 4532-0151-1283-0366. Thank you.")

        f3_path = tmp_path / "notes.txt"
        f3_path.write_text("Meeting notes: discussed Q3 project architecture.")

        now = time.time()
        r1 = FileRecord(path=str(f1_path), name="tax_doc.txt", ext="txt", kind="document", size=100, modified=now)
        r2 = FileRecord(path=str(f2_path), name="receipt.txt", ext="txt", kind="document", size=100, modified=now)
        r3 = FileRecord(path=str(f3_path), name="notes.txt", ext="txt", kind="document", size=100, modified=now)

        id1 = db.upsert_file(r1, db_path=test_db)
        id2 = db.upsert_file(r2, db_path=test_db)
        id3 = db.upsert_file(r3, db_path=test_db)

        # Run scan
        flagged = scan_sensitive(db_path=test_db, show_progress=False)
        assert flagged == 2, f"Expected 2 sensitive files flagged, got {flagged}"

        rec1 = db.get_file_by_id(id1, db_path=test_db)
        rec2 = db.get_file_by_id(id2, db_path=test_db)
        rec3 = db.get_file_by_id(id3, db_path=test_db)

        # Verify PAN record
        assert rec1.privacy_type == "pan"
        assert rec1.privacy_masked == "BKRPK****Q"
        assert "9876" not in rec1.privacy_masked, "Raw PAN digits leaked into SQLite!"

        # Verify Card record
        assert rec2.privacy_type == "card"
        assert rec2.privacy_masked.endswith("0366")

        # Verify Clean record
        assert rec3.privacy_type is None
        assert rec3.privacy_masked is None

        del rec1, rec2, rec3, r1, r2, r3
        gc.collect()

    print(" End-to-end privacy scan tests passed successfully!")


def test_ocr_category_gating():
    print("\nTesting Intelligent OCR Category Gating ('photo' / 'scenery')...")
    temp_dir_kwargs = {}
    if sys.version_info >= (3, 10):
        temp_dir_kwargs["ignore_cleanup_errors"] = True

    with tempfile.TemporaryDirectory(**temp_dir_kwargs) as tmpdir:
        tmp_path = Path(tmpdir)
        test_db = tmp_path / "test_clutteriq.db"
        db.init_db(test_db)

        # Create dummy image files (> 10KB so size filter wouldn't trigger)
        img1 = tmp_path / "sunset.jpg"
        img1.write_bytes(b"\x00" * 20_000)

        img2 = tmp_path / "mountains.png"
        img2.write_bytes(b"\x00" * 20_000)

        now = time.time()
        r1 = FileRecord(
            path=str(img1),
            name="sunset.jpg",
            ext="jpg",
            kind="image",
            size=20_000,
            category="photo",
            modified=now,
        )
        r2 = FileRecord(
            path=str(img2),
            name="mountains.png",
            ext="png",
            kind="image",
            size=20_000,
            category="scenery",
            modified=now,
        )

        id1 = db.upsert_file(r1, db_path=test_db)
        id2 = db.upsert_file(r2, db_path=test_db)

        # Mock extract_text_from_image to verify it is NEVER called for 'photo' or 'scenery'
        import privacy.sensitive as sensitive_mod

        ocr_calls = []
        original_ocr = sensitive_mod.extract_text_from_image

        def mocked_ocr(path, max_dim=1600):
            ocr_calls.append(str(path))
            return "Sensitive PAN: ABCDE1234F"

        sensitive_mod.extract_text_from_image = mocked_ocr

        try:
            flagged = scan_sensitive(db_path=test_db, show_progress=False)
            assert flagged == 0, f"Expected 0 files flagged, got {flagged}"
            assert len(ocr_calls) == 0, (
                f"OCR should have been skipped, but was called {len(ocr_calls)} times!"
            )
        finally:
            sensitive_mod.extract_text_from_image = original_ocr

        del r1, r2
        gc.collect()

    print(" Intelligent OCR Category Gating verified successfully (0 OCR calls on photos)!")


if __name__ == "__main__":
    test_checksum_algorithms()
    test_pattern_detection_and_masking()
    test_end_to_end_privacy_scan()
    test_ocr_category_gating()
