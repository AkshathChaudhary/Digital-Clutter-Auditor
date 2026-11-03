"""Headless rendering verification for all 5 Streamlit pages in ClutterIQ."""

from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
from config import DB_PATH


def mock_columns(spec, *args, **kwargs):
    count = spec if isinstance(spec, int) else len(spec)
    return [MagicMock() for _ in range(count)]


def test_render_all_pages_headless():
    patches = [
        patch("streamlit.title"),
        patch("streamlit.header"),
        patch("streamlit.subheader"),
        patch("streamlit.markdown"),
        patch("streamlit.write"),
        patch("streamlit.caption"),
        patch("streamlit.info"),
        patch("streamlit.success"),
        patch("streamlit.warning"),
        patch("streamlit.error"),
        patch("streamlit.metric"),
        patch("streamlit.divider"),
        patch("streamlit.columns", side_effect=mock_columns),
        patch("streamlit.expander", return_value=MagicMock()),
        patch("streamlit.radio", return_value="All Items"),
        patch("streamlit.selectbox", return_value="All"),
        patch("streamlit.text_input", return_value=""),
        patch("streamlit.checkbox", return_value=False),
        patch("streamlit.button", return_value=False),
        patch("streamlit.data_editor", return_value=pd.DataFrame({"id": [1], "Select": [False]})),
        patch("streamlit.plotly_chart"),
        patch("streamlit.progress"),
        patch("streamlit.image"),
    ]

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)

        print("Testing page_overview...")
        try:
            from app.page_overview import render_overview_page
        except ModuleNotFoundError:
            from page_overview import render_overview_page
        render_overview_page(DB_PATH)
        print("  page_overview: SUCCESS")

        print("Testing page_duplicates (Exact)...")
        try:
            from app.page_duplicates import render_duplicates_page
        except ModuleNotFoundError:
            from page_duplicates import render_duplicates_page
        with patch("streamlit.radio", return_value="Exact Matches (SHA-256)"):
            render_duplicates_page(DB_PATH)
        print("  page_duplicates (Exact): SUCCESS")

        print("Testing page_duplicates (Near-Duplicates)...")
        with patch("streamlit.radio", return_value="Near-Duplicate Images (pHash)"):
            render_duplicates_page(DB_PATH)
        print("  page_duplicates (Near): SUCCESS")

        print("Testing page_duplicates (Semantic)...")
        with patch("streamlit.radio", return_value="Semantic Clusters (CLIP AI)"):
            render_duplicates_page(DB_PATH)
        print("  page_duplicates (Semantic): SUCCESS")

        print("Testing page_privacy...")
        try:
            from app.page_privacy import render_privacy_page
        except ModuleNotFoundError:
            from page_privacy import render_privacy_page
        render_privacy_page(DB_PATH)
        print("  page_privacy: SUCCESS")

        print("Testing page_suggestions...")
        try:
            from app.page_suggestions import render_suggestions_page
        except ModuleNotFoundError:
            from page_suggestions import render_suggestions_page
        render_suggestions_page(DB_PATH)
        print("  page_suggestions: SUCCESS")

        print("Testing page_quarantine...")
        try:
            from app.page_quarantine import render_quarantine_page
        except ModuleNotFoundError:
            from page_quarantine import render_quarantine_page
        render_quarantine_page(DB_PATH)
        print("  page_quarantine: SUCCESS")


if __name__ == "__main__":
    test_render_all_pages_headless()
    print("\nALL 5 DASHBOARD PAGES TESTED HEADLESS: ZERO RUNTIME ERRORS!")
