"""Demo Folder Generator for ClutterIQ.

Generates a realistic, clean showcase folder containing:
1. Exact duplicate PDF documents and JSON configs.
2. Perceptual near-duplicate image pair (sharp PNG vs compressed JPG).
3. Sensitive privacy test file (simulated PAN and valid Aadhaar checksum).
4. Stale installer junk file.
5. Distinct unique documents and data files (negative controls).
"""

import json
import os
from pathlib import Path
import time
from PIL import Image, ImageDraw

from privacy.sensitive import validate_verhoeff


def generate_valid_aadhaar() -> str:
    """Generate a 12-digit Aadhaar number with valid Verhoeff checksum."""
    body = "98765432109"
    for d in range(10):
        candidate = body + str(d)
        if validate_verhoeff(candidate):
            return candidate
    return "987654321090"


def create_demo_folder(target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating ClutterIQ Demo showcase at: {target_dir}")

    # 1. Exact Duplicate Documents
    report_content = (
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n"
        b"4 0 obj\n<< /Length 55 >>\nstream\nBT /F1 12 Tf 100 700 Td (Semester Capstone Project Proposal) ET\nendstream\nendobj\n"
        b"xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000206 00000 n \n"
        b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n312\n%%EOF\n"
    )
    (target_dir / "Capstone_Project_Proposal.pdf").write_bytes(report_content)
    (target_dir / "Capstone_Project_Proposal - Copy.pdf").write_bytes(report_content)

    # 2. Exact Duplicate JSON Configuration
    app_settings = {
        "project": "ClutterIQ",
        "environment": "production",
        "database": "SQLite-WAL",
        "max_workers": 4,
        "logging_level": "INFO",
    }
    json_bytes = json.dumps(app_settings, indent=2).encode("utf-8")
    (target_dir / "system_config.json").write_bytes(json_bytes)
    (target_dir / "system_config_backup.json").write_bytes(json_bytes)

    # 3. Unique JSON and Data Files (Negative Controls)
    grades_data = {"student": "Akshath", "semester": 7, "subjects": [{"code": "CSE401", "grade": "A"}]}
    (target_dir / "student_grades.json").write_bytes(json.dumps(grades_data, indent=2).encode("utf-8"))

    timetable_data = {"monday": ["Computer Vision", "Machine Learning"], "tuesday": ["Cloud Computing"]}
    (target_dir / "weekly_schedule.json").write_bytes(json.dumps(timetable_data, indent=2).encode("utf-8"))

    # 4. Perceptual Near-Duplicate Images (Campus Banner)
    # Original sharp high-resolution image
    img_sharp = Image.new("RGB", (600, 400), color=(30, 80, 160))
    draw = ImageDraw.Draw(img_sharp)
    draw.rectangle([60, 60, 540, 340], fill=(240, 190, 40), outline=(255, 255, 255), width=4)
    draw.text((120, 180), "CAMPUS RESEARCH EXHIBITION 2026", fill=(20, 20, 20))
    sharp_path = target_dir / "Campus_Event_Banner.png"
    img_sharp.save(sharp_path, "PNG")

    # Resized version of the same image (pHash distance <= 6)
    img_resized = img_sharp.resize((520, 350))
    resized_path = target_dir / "Campus_Event_Banner_web.png"
    img_resized.save(resized_path, "PNG")

    # Distinct Image (Negative Control)
    img_diff = Image.new("RGB", (400, 400), color=(180, 40, 40))
    draw_diff = ImageDraw.Draw(img_diff)
    draw_diff.ellipse([80, 80, 320, 320], fill=(40, 200, 80))
    diff_path = target_dir / "Security_Badge_Photo.png"
    img_diff.save(diff_path, "PNG")

    # 5. Sensitive Identity & Financial Document
    pan_id = "ABCDE1234F"
    aadhaar_id = generate_valid_aadhaar()
    sensitive_content = (
        f"STUDENT SCHOLARSHIP VERIFICATION FORM\n"
        f"Applicant: Akshath Chaudhary\n"
        f"Income Tax PAN: {pan_id}\n"
        f"UIDAI Aadhaar Number: {aadhaar_id}\n"
        f"Bank Account: State Bank of India\n"
        f"Status: Pending Institutional Verification\n"
    )
    (target_dir / "Scholarship_KYC_Verification.txt").write_text(sensitive_content, encoding="utf-8")

    # 6. Clean Business Document
    clean_notes = (
        "Project Architecture Notes:\n"
        "- Frontend: Streamlit Multi-Page SPA\n"
        "- Database: SQLite with WAL pragma\n"
        "- ML Engine: Scikit-learn Random Forest + CLIP ViT-B-32\n"
    )
    (target_dir / "System_Architecture_Notes.txt").write_text(clean_notes, encoding="utf-8")

    # 7. Stale Temporary Installer Clutter
    stale_file = target_dir / "NodeJS_v18_Setup_old (1).exe"
    stale_file.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00Simulated Legacy Windows Installer Clutter" * 100)
    # Stamp date to 420 days ago (> 365 days STALE_DAYS threshold)
    past_timestamp = time.time() - (420 * 86400)
    os.utime(stale_file, (past_timestamp, past_timestamp))

    print(f" Successfully created 10 showcase files in '{target_dir}'.")
    return target_dir


if __name__ == "__main__":
    default_dest = Path.home() / "Desktop" / "ClutterIQ_Demo"
    create_demo_folder(default_dest)
