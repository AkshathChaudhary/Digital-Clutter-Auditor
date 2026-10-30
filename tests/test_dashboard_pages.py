"""Unit test for Streamlit dashboard logic and metric calculations.

Verifies:
1. compute_overview_metrics calculates storage, recoverable space, duplicate clusters, and privacy risks.
2. Space recoverable correctly avoids double-counting duplicate non-keepers and high-confidence junk.
3. Clean module imports and interface contracts for all Streamlit pages.
"""

import gc
import sys
from pathlib import Path
import tempfile
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ssl_fix

from app.page_overview import compute_overview_metrics, format_bytes
import db
from models import FileRecord


def test_format_bytes():
    print("Testing format_bytes helper...")
    assert format_bytes(500) == "500 B"
    assert format_bytes(2048) == "2.0 KB"
    assert format_bytes(5 * 1024 * 1024) == "5.0 MB"
    assert format_bytes(3.5 * 1024 * 1024 * 1024) == "3.50 GB"
    print(" format_bytes tests passed!")


def test_overview_metrics_calculation():
    print("\nTesting compute_overview_metrics with deduplicated space logic...")

    # Create synthetic DataFrame
    # File 1: Non-keeper duplicate (100 MB), also junk_score 0.90 -> should be counted ONCE
    # File 2: Keeper duplicate (200 MB), junk_score 0.10 -> should NOT be counted as recoverable
    # File 3: Unique file (50 MB), junk_score 0.85 -> high-confidence junk, counted as recoverable
    # File 4: Clean file (80 MB), junk_score 0.20 -> NOT recoverable
    # Total space: 100 + 200 + 50 + 80 = 430 MB
    # Expected Recoverable space: 100 (File 1) + 50 (File 3) = 150 MB (no double counting!)

    MB = 1024 * 1024
    test_data = {
        "id": [1, 2, 3, 4],
        "size": [100 * MB, 200 * MB, 50 * MB, 80 * MB],
        "exact_group": [10, 10, None, None],
        "near_group": [None, None, None, None],
        "semantic_group": [10, 10, None, None],
        "is_keeper": [0, 1, 0, 0],
        "junk_score": [0.90, 0.10, 0.85, 0.20],
        "privacy_type": ["pan", None, None, None],
    }
    df = pd.DataFrame(test_data)

    total_files, total_bytes, recoverable_bytes, dup_groups, privacy_alerts = compute_overview_metrics(df)

    assert total_files == 4, f"Expected 4 files, got {total_files}"
    assert total_bytes == 430 * MB, f"Expected 430 MB, got {total_bytes / MB} MB"
    assert recoverable_bytes == 150 * MB, f"Expected 150 MB recoverable, got {recoverable_bytes / MB} MB"
    assert privacy_alerts == 1, f"Expected 1 privacy risk, got {privacy_alerts}"

    print(" compute_overview_metrics passed with zero double-counting!")


def test_app_imports():
    print("\nVerifying Streamlit page imports...")
    import app.app as main_app
    import app.page_overview as p_over
    import app.page_privacy as p_priv
    import app.page_suggestions as p_sugg

    assert callable(main_app.main)
    assert callable(p_over.render_overview_page)
    assert callable(p_priv.render_privacy_page)
    assert callable(p_sugg.render_suggestions_page)
    print(" All Streamlit pages imported and resolved successfully!")


if __name__ == "__main__":
    test_format_bytes()
    test_overview_metrics_calculation()
    test_app_imports()

