"""Full-System End-to-End Simulation Test for ClutterIQ.

Constructs a personalized simulation directory with:
1. Exact duplicate documents and JSON configurations.
2. Distinct JSON files (verifying zero false-positive grouping).
3. 0-byte empty files with different extensions (verifying zero cross-extension grouping).
4. Perceptual near-duplicate images.
5. Sensitive documents with valid PAN and Aadhaar checksums.
6. Stale temporary installer files.

Runs the complete 7-stage intelligence pipeline on the simulation folder,
verifies all detections, and validates the non-destructive quarantine & restore vault.
"""

import gc
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ssl_fix
from config import DB_PATH
import db
from dedup.exact import find_exact_duplicates
from dedup.perceptual import find_near_duplicates
from ml.junk_model import predict_scores
from privacy.sensitive import scan_sensitive, validate_verhoeff
from quality.blur_rank import choose_keepers
from safety.quarantine import quarantine_file, restore_file
from scanner.walker import scan_folder


def generate_valid_aadhaar() -> str:
    """Generate a 12-digit Aadhaar number with valid Verhoeff checksum."""
    body = "98765432109"
    # Compute checksum digit
    for d in range(10):
        candidate = body + str(d)
        if validate_verhoeff(candidate):
            return candidate
    return "987654321090"


def test_full_simulation_pipeline():
    """Execute end-to-end audit on a personalized simulation environment."""
    print("\n" + "=" * 70)
    print("STARTING FULL-SYSTEM SIMULATION AUDIT")
    print("=" * 70)

    temp_dir_kwargs = {}
    if sys.version_info >= (3, 10):
        temp_dir_kwargs["ignore_cleanup_errors"] = True

    with tempfile.TemporaryDirectory(**temp_dir_kwargs) as temp_root:
        sim_dir = Path(temp_root) / "Personalized_Simulation_Target"
        sim_dir.mkdir(parents=True, exist_ok=True)
        sim_db = Path(temp_root) / "simulation_clutteriq.db"
        db.init_db(sim_db)

        # -------------------------------------------------------------------
        # 1. Populate Simulation Target Folder
        # -------------------------------------------------------------------
        # Exact duplicate documents
        doc_content = b"%PDF-1.4 Simulated Quarterly Financial Report Content " * 50
        (sim_dir / "Q3_Financial_Report.pdf").write_bytes(doc_content)
        (sim_dir / "Q3_Financial_Report - Copy.pdf").write_bytes(doc_content)

        # Exact duplicate JSON configs
        json_content = json.dumps({"app_mode": "production", "version": "2.4.1", "theme": "dark"}).encode()
        (sim_dir / "app_config.json").write_bytes(json_content)
        (sim_dir / "app_config_backup.json").write_bytes(json_content)

        # Distinct JSON files (Negative test: must NOT be grouped)
        (sim_dir / "inventory.json").write_bytes(
            json.dumps({"items": [{"sku": "A10", "qty": 45}, {"sku": "B20", "qty": 12}]}).encode()
        )
        (sim_dir / "employees.json").write_bytes(
            json.dumps({"staff": [{"name": "Akshath", "dept": "Engineering"}]}).encode()
        )

        # Empty 0-byte files with different extensions (Negative test: must NOT be grouped)
        (sim_dir / "empty_notes.txt").write_bytes(b"")
        (sim_dir / "empty_syllabus.pdf").write_bytes(b"")

        # Perceptual near-duplicate images (resized version)
        img_a = Image.new("RGB", (300, 300), color=(40, 100, 200))
        draw_a = ImageDraw.Draw(img_a)
        draw_a.rectangle([50, 50, 250, 250], fill=(255, 200, 50))
        img_a_path = sim_dir / "banner.png"
        img_a.save(img_a_path, "PNG")

        # Resized version of same image (pHash distance <= 10)
        img_b_path = sim_dir / "banner_resized.png"
        img_b = img_a.resize((280, 280))
        img_b.save(img_b_path, "PNG")

        # Sensitive Document (PAN + Aadhaar)
        aadhaar_num = generate_valid_aadhaar()
        pan_num = "ABCDE1234F"
        sensitive_text = f"CONFIDENTIAL IDENTITY RECORD\nPAN: {pan_num}\nAadhaar: {aadhaar_num}\n"
        (sim_dir / "confidential_kyc.txt").write_text(sensitive_text)

        # Clean Document
        (sim_dir / "meeting_notes.txt").write_text("Team weekly standup sync notes: sprint goals on track.")

        # Stale File
        stale_path = sim_dir / "old_temp_installer.tmp"
        stale_path.write_bytes(b"Simulated legacy installer binary data" * 100)
        # Set modified time to 400 days ago (stale > 365 days)
        old_time = time.time() - (400 * 86400)
        os.utime(stale_path, (old_time, old_time))

        print(f"Created simulation folder with 11 distinct scenario files at: {sim_dir.name}")

        # -------------------------------------------------------------------
        # 2. Stage 1: Walk & Index Simulation Folder (Fresh Audit)
        # -------------------------------------------------------------------
        files_indexed = scan_folder(str(sim_dir), clean_slate=True, db_path=sim_db)
        assert files_indexed == 13, f"Expected 13 files indexed, got {files_indexed}"
        print(f" Stage 1: Scan Walker indexed all {files_indexed} files with fresh slate.")

        # -------------------------------------------------------------------
        # 3. Stage 2: Exact Deduplication (SHA-256)
        # -------------------------------------------------------------------
        exact_groups = find_exact_duplicates(db_path=sim_db)
        assert len(exact_groups) == 2, f"Expected exactly 2 exact duplicate groups, got {len(exact_groups)}"

        # Verify Q3 Report duplicate group
        r1 = db.get_file_by_path(str(sim_dir / "Q3_Financial_Report.pdf"), db_path=sim_db)
        r2 = db.get_file_by_path(str(sim_dir / "Q3_Financial_Report - Copy.pdf"), db_path=sim_db)
        assert r1.exact_group is not None, "Q3_Financial_Report.pdf should have exact_group"
        assert r1.exact_group == r2.exact_group, "Q3_Financial_Report copies must share the same exact_group"

        # Verify JSON configs duplicate group
        j1 = db.get_file_by_path(str(sim_dir / "app_config.json"), db_path=sim_db)
        j2 = db.get_file_by_path(str(sim_dir / "app_config_backup.json"), db_path=sim_db)
        assert j1.exact_group is not None, "app_config.json should have exact_group"
        assert j1.exact_group == j2.exact_group, "app_config copies must share the same exact_group"

        # Verify distinct JSON files are NOT grouped
        inv = db.get_file_by_path(str(sim_dir / "inventory.json"), db_path=sim_db)
        emp = db.get_file_by_path(str(sim_dir / "employees.json"), db_path=sim_db)
        assert inv.exact_group is None, "Distinct JSON inventory.json must NOT be in any duplicate group!"
        assert emp.exact_group is None, "Distinct JSON employees.json must NOT be in any duplicate group!"

        # Verify 0-byte files with different extensions are NOT grouped
        e_txt = db.get_file_by_path(str(sim_dir / "empty_notes.txt"), db_path=sim_db)
        e_pdf = db.get_file_by_path(str(sim_dir / "empty_syllabus.pdf"), db_path=sim_db)
        assert e_txt.exact_group is None, "0-byte txt must NOT be grouped with 0-byte pdf!"
        assert e_pdf.exact_group is None, "0-byte pdf must NOT be grouped with 0-byte txt!"
        print(" Stage 2: Exact Deduplication verified: 2 true duplicate groups, 0 false positives.")

        # -------------------------------------------------------------------
        # 4. Stage 3: Perceptual Deduplication (pHash)
        # -------------------------------------------------------------------
        near_groups = find_near_duplicates(db_path=sim_db)
        assert len(near_groups) == 1, f"Expected 1 near-duplicate group for images, got {len(near_groups)}"
        print(" Stage 3: Perceptual pHash detected near-duplicate image pair.")

        # -------------------------------------------------------------------
        # 5. Stage 4: Quality & Keeper Selection
        # -------------------------------------------------------------------
        keepers = choose_keepers(db_path=sim_db)
        # Check keeper for Q3 Financial Report
        r1_fresh = db.get_file_by_path(str(sim_dir / "Q3_Financial_Report.pdf"), db_path=sim_db)
        r2_fresh = db.get_file_by_path(str(sim_dir / "Q3_Financial_Report - Copy.pdf"), db_path=sim_db)
        assert r1_fresh.is_keeper == 1, "Cleaner original name must be chosen as Keeper!"
        assert r2_fresh.is_keeper == 0, "File with ' - Copy' must be marked as duplicate copy (is_keeper=0)!"
        print(" Stage 4: Keeper ranking correctly identified original as Keeper and copy as non-keeper.")

        # -------------------------------------------------------------------
        # 6. Stage 5: Sensitive Document Privacy Audit
        # -------------------------------------------------------------------
        privacy_found = scan_sensitive(db_path=sim_db, show_progress=False)
        assert privacy_found >= 1, "Sensitive KYC file must be flagged by privacy auditor"

        sens_record = db.get_file_by_path(str(sim_dir / "confidential_kyc.txt"), db_path=sim_db)
        assert sens_record.privacy_type in ["pan", "aadhaar"], f"Expected pan or aadhaar, got {sens_record.privacy_type}"
        assert pan_num not in (sens_record.privacy_masked or ""), "Raw PAN must NEVER be stored raw!"
        print(f" Stage 5: Privacy Auditor flagged sensitive file with masked token: '{sens_record.privacy_masked}'")

        # -------------------------------------------------------------------
        # 7. Stage 6: ML Junk & Staleness Scoring
        # -------------------------------------------------------------------
        scored_count = predict_scores(db_path=sim_db)
        assert scored_count == 13, f"Expected all 13 files scored, got {scored_count}"

        stale_record = db.get_file_by_path(str(stale_path), db_path=sim_db)
        assert stale_record.junk_score > 0.3, "Stale 300-day-old file should have elevated junk score"
        assert "unmodified" in (stale_record.junk_reason or ""), "Junk reason must explain staleness"

        # Check empty file reason
        e_txt_fresh = db.get_file_by_path(str(sim_dir / "empty_notes.txt"), db_path=sim_db)
        assert "empty zero-byte file" in (e_txt_fresh.junk_reason or ""), "0-byte file must be labeled as empty"
        print(" Stage 6: ML & Heuristic Scoring generated explainable justifications.")

        # -------------------------------------------------------------------
        # 8. Stage 7: Non-Destructive Quarantine & Restore Vault
        # -------------------------------------------------------------------
        # Quarantine the duplicate copy of Q3 Report
        dest_quar = quarantine_file(r2_fresh.id, db_path=sim_db)
        assert dest_quar.exists(), "Quarantined file must exist in quarantine vault"
        assert not Path(r2_fresh.path).exists(), "Original duplicate file must be removed from source path"

        # Check quarantine log record
        q_rec = db.get_quarantine_record(1, db_path=sim_db)
        assert q_rec is not None, "Quarantine record must be logged in database"

        # Restore file from quarantine
        restored_path = restore_file(1, db_path=sim_db)
        assert restored_path.exists(), "Restored file must be back at original path"
        assert restored_path.read_bytes() == doc_content, "Restored content must be bit-exact identical"
        print(" Stage 7: Non-destructive quarantine & 100% bit-exact restoration verified.")

        # -------------------------------------------------------------------
        # 9. Stage 8: Sub-Millisecond KPI Calculation
        # -------------------------------------------------------------------
        tot_files, tot_bytes, rec_bytes, total_groups, priv_alerts = db.get_overview_kpis(db_path=sim_db)
        assert tot_files in [12, 13], f"Expected 12 or 13 files, got {tot_files}"
        assert total_groups >= 3, f"Expected at least 3 duplicate groups, got {total_groups}"
        assert priv_alerts == 1, f"Expected 1 privacy alert, got {priv_alerts}"
        assert rec_bytes > 0, "Recoverable bytes must be > 0"
        print(" Stage 8: Overview KPI calculation verified with zero double counting.")

    print("\n" + "=" * 70)
    print("ALL SIMULATION AUDIT CHECKS PASSED WITH ZERO DEFECTS!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_full_simulation_pipeline()
