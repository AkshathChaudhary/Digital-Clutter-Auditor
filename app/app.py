"""Streamlit main entry point and navigation shell for ClutterIQ.

Provides:
1. Sidebar navigation between all 5 dashboard modules.
2. Root directory path input with 'Run Scan' action.
3. One-click 'Run Analysis' pipeline (CLIP embeddings -> Semantic dedup -> Privacy scan -> ML scoring).
4. Graceful fallbacks for Person A modules if pending integration.

Owned by Person B (Intelligence & Dashboard).
"""

import logging
import os
from pathlib import Path
import sys

# Ensure both project root and app directory are in sys.path to resolve imports cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent

for p in (PROJECT_ROOT, APP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Must import ssl_fix before any networking / ML imports
import ssl_fix

logger = logging.getLogger("clutteriq.app")

import streamlit as st

from config import DB_PATH
import db

# Page imports (supports direct sibling import to avoid 'app is not a package' collision in Streamlit)
try:
    from page_overview import render_overview_page
except ImportError:
    from app.page_overview import render_overview_page

try:
    from page_privacy import render_privacy_page
except ImportError:
    from app.page_privacy import render_privacy_page

try:
    from page_suggestions import render_suggestions_page
except ImportError:
    from app.page_suggestions import render_suggestions_page

# Graceful imports for Person A modules
try:
    from page_duplicates import render_duplicates_page
except ImportError:
    try:
        from app.page_duplicates import render_duplicates_page
    except ImportError:
        def render_duplicates_page(db_path=DB_PATH):
            st.title(" Duplicate Groups")
            st.info(" Person A module (`app/page_duplicates.py`): Exact & Perceptual duplicate groups with thumbnail previews will appear here.")

try:
    from page_quarantine import render_quarantine_page
except ImportError:
    try:
        from app.page_quarantine import render_quarantine_page
    except ImportError:
        def render_quarantine_page(db_path=DB_PATH):
            st.title(" Quarantine Vault")
            st.info(" Person A module (`app/page_quarantine.py`): Quarantine log and one-click restoration will appear here.")


# ---------------------------------------------------------------------------
# Streamlit App Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ClutterIQ | Digital Hygiene Auditor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def run_pipeline_analysis(progress_bar, status_text) -> None:
    """Execute intelligence pipeline end-to-end with step-by-step telemetry."""
    import time

    steps = [
        ("Generating CLIP ViT-B-32 Image Embeddings", lambda: __import__("dedup.semantic", fromlist=["compute_embeddings"]).compute_embeddings(db_path=DB_PATH, show_progress=False)),
        ("Clustering Semantic Duplicates (HNSW Graph)", lambda: __import__("dedup.semantic", fromlist=["cluster_semantic"]).cluster_semantic(db_path=DB_PATH)),
        ("Scanning Sensitive Documents (OCR, PDF, Luhn & Verhoeff)", lambda: __import__("privacy.sensitive", fromlist=["scan_sensitive"]).scan_sensitive(db_path=DB_PATH, show_progress=False)),
        ("Predicting ML Junk & Staleness Scores", lambda: __import__("ml.junk_model", fromlist=["predict_scores"]).predict_scores(db_path=DB_PATH)),
    ]

    total_steps = len(steps)
    for idx, (title, action) in enumerate(steps):
        t0 = time.time()
        status_text.info(f"⏳ Step {idx + 1}/{total_steps}: {title}...")
        progress_bar.progress((idx) / total_steps)
        action()
        elapsed = time.time() - t0
        logger.info(f"Completed '{title}' in {elapsed:.2f}s")

    progress_bar.progress(1.0)
    status_text.success("🎉 Full Analysis Pipeline Completed Successfully!")


def main():
    # Initialize SQLite database if missing
    db.init_db(DB_PATH)

    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.title(" ClutterIQ")
        st.caption("Local-First Digital Hygiene Auditor")
        st.divider()

        # Navigation Menu
        page_selection = st.radio(
            "Navigation",
            options=[
                "Overview",
                "Duplicate Groups",
                "Privacy Risks",
                "Clean Recommendations",
                "Quarantine Vault",
            ],
            index=0,
        )
        st.divider()

        # Target Folder Scanner Input
        st.subheader(" Scan Target")
        default_folder = str(Path.home() / "Downloads")
        scan_folder_input = st.text_input("Folder path to audit:", default_folder)

        if st.button(" Run Scan", type="secondary", use_container_width=True):
            target_path = Path(scan_folder_input)
            if not target_path.exists():
                st.error("Specified path does not exist on disk!")
            else:
                try:
                    from scanner.walker import scan_folder
                    with st.spinner("Walking filesystem and recording metadata..."):
                        files_seen = scan_folder(str(target_path))
                    st.success(f"Scanned {files_seen:,} files!")
                    st.rerun()
                except ImportError:
                    st.warning("Person A's `scanner/walker.py` is not yet available. You can test analysis on sample files.")

        st.divider()

        # Full Analysis Trigger
        st.subheader(" Intelligence Pipeline")
        st.caption("Runs CLIP embeddings, privacy auditor, and ML junk model.")
        if st.button(" Run Analysis", type="primary", use_container_width=True):
            prog = st.progress(0.0)
            status = st.empty()
            try:
                run_pipeline_analysis(prog, status)
                st.success("All intelligence stages finished successfully!")
                time_delay = 0.5
                st.rerun()
            except Exception as e:
                st.error(f"Analysis pipeline error: {e}")

        st.divider()
        st.markdown(
            "**Privacy Guarantee**: 100% Local. No files or personal data ever leave this computer."
        )

    # -----------------------------------------------------------------------
    # Page Router
    # -----------------------------------------------------------------------
    if page_selection == "Overview":
        render_overview_page(DB_PATH)
    elif page_selection == "Duplicate Groups":
        render_duplicates_page(DB_PATH)
    elif page_selection == "Privacy Risks":
        render_privacy_page(DB_PATH)
    elif page_selection == "Clean Recommendations":
        render_suggestions_page(DB_PATH)
    elif page_selection == "Quarantine Vault":
        render_quarantine_page(DB_PATH)


if __name__ == "__main__":
    main()
