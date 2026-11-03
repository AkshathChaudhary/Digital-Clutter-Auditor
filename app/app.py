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
import textwrap

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
# Streamlit App Configuration & Modern Design System
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ClutterIQ | Digital Hygiene Auditor",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Modern Aesthetics
st.markdown(
    """
    <style>
    /* Global Clean Font and Smooth UI */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Modern Headers */
    .app-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #F43F5E 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    /* Sleek Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    
    /* Visual Comparison Cards */
    .keeper-box {
        background: rgba(6, 78, 59, 0.22);
        border: 1.5px solid rgba(16, 185, 129, 0.5);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 12px;
    }
    
    .duplicate-box {
        background: rgba(136, 19, 55, 0.18);
        border: 1.5px solid rgba(244, 63, 94, 0.4);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 12px;
    }
    
    .semantic-box {
        background: rgba(49, 46, 129, 0.22);
        border: 1.5px solid rgba(129, 140, 248, 0.45);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 12px;
    }
    
    /* Status Badges */
    .badge-keeper {
        background: linear-gradient(135deg, #059669, #10B981);
        color: white !important;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        display: inline-block;
        letter-spacing: 0.03em;
    }
    
    .badge-dup {
        background: linear-gradient(135deg, #E11D48, #F43F5E);
        color: white !important;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        display: inline-block;
        letter-spacing: 0.03em;
    }
    
    .badge-semantic {
        background: linear-gradient(135deg, #4F46E5, #6366F1);
        color: white !important;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        display: inline-block;
        letter-spacing: 0.03em;
    }
    
    .badge-exact {
        background: linear-gradient(135deg, #D97706, #F59E0B);
        color: white !important;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        display: inline-block;
        letter-spacing: 0.03em;
    }
    
    .chip {
        background: rgba(255, 255, 255, 0.08);
        color: #E2E8F0;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        display: inline-block;
        margin: 2px 2px;
    }
    
    .chip-sharp {
        background: rgba(14, 165, 233, 0.2);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.4);
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .chip-blurry {
        background: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid rgba(248, 113, 113, 0.4);
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Modern Streamlit Button Polish */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #6366F1 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        border-radius: 9px !important;
        padding: 0.55rem 1rem !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-1.5px) !important;
        box-shadow: 0 7px 22px rgba(99, 102, 241, 0.55) !important;
    }
    
    div.stButton > button[kind="secondary"] {
        background: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #E2E8F0 !important;
        border-radius: 9px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background: rgba(51, 65, 85, 0.9) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        color: white !important;
    }
    
    /* Streamlit Expander Glow */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.4) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def run_pipeline_analysis(progress_bar, status_text) -> None:
    """Execute intelligence pipeline end-to-end with step-by-step telemetry."""
    import time

    steps = [
        ("Exact Deduplication (SHA-256)", lambda: __import__("dedup.exact", fromlist=["find_exact_duplicates"]).find_exact_duplicates(db_path=DB_PATH)),
        ("Perceptual Deduplication (pHash)", lambda: __import__("dedup.perceptual", fromlist=["find_near_duplicates"]).find_near_duplicates(db_path=DB_PATH)),
        ("Quality Ranking & Keeper Selection", lambda: __import__("quality.blur_rank", fromlist=["choose_keepers"]).choose_keepers(db_path=DB_PATH)),
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
        st.markdown(
            textwrap.dedent(
                """
                <div style="display: flex; align-items: center; gap: 12px; padding: 4px 0 14px 0;">
                    <div style="background: linear-gradient(135deg, #2563EB, #7C3AED); border-radius: 10px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; box-shadow: 0 4px 12px rgba(124, 58, 237, 0.4);">
                        🧹
                    </div>
                    <div>
                        <h2 style="margin: 0; font-size: 1.45rem; font-weight: 800; background: linear-gradient(135deg, #38BDF8, #A78BFA); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">ClutterIQ</h2>
                        <p style="margin: 0; font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;">Digital Hygiene Auditor</p>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
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
        st.subheader("📁 Audit Target Folder")
        demo_candidate = Path(__file__).resolve().parent.parent / "demo_showcase"
        default_folder = str(demo_candidate) if demo_candidate.exists() else str(Path.home() / "Downloads")
        scan_folder_input = st.text_input("Folder path to audit:", default_folder)
        fresh_audit = st.checkbox(
            "Fresh Audit (clear previous results)",
            value=True,
            help="When checked, clears previously scanned folders so your dashboard only reflects this folder.",
        )

        # Primary One-Click Workflow
        if st.button("🚀 Scan & Run Full Audit", type="primary", use_container_width=True):
            target_path = Path(scan_folder_input)
            if not target_path.exists() or not target_path.is_dir():
                st.error("Specified path does not exist or is not a directory!")
            else:
                try:
                    import importlib
                    import scanner.walker
                    importlib.reload(scanner.walker)
                    from scanner.walker import scan_folder

                    if fresh_audit:
                        conn = db.get_conn(DB_PATH)
                        try:
                            with conn:
                                conn.execute("DELETE FROM files WHERE status = 'active'")
                        finally:
                            conn.close()

                    with st.spinner(f"Scanning '{target_path.name}'..."):
                        files_seen = scan_folder(str(target_path))
                    
                    st.success(f"Indexed {files_seen:,} files from {target_path.name}!")
                    
                    # Run full 7-stage intelligence pipeline immediately
                    prog = st.progress(0.0)
                    status = st.empty()
                    run_pipeline_analysis(prog, status)
                    st.success("🎉 Full scan & analysis completed successfully!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Audit pipeline error: {ex}")

        st.divider()

        # Step-by-Step / Maintenance Actions
        st.subheader("⚙️ Pipeline Controls")
        col_scan, col_an = st.columns(2)
        with col_scan:
            if st.button("🔍 Scan Only", use_container_width=True):
                target_path = Path(scan_folder_input)
                if not target_path.exists():
                    st.error("Path does not exist!")
                else:
                    try:
                        import importlib
                        import scanner.walker
                        importlib.reload(scanner.walker)
                        from scanner.walker import scan_folder

                        if fresh_audit:
                            conn = db.get_conn(DB_PATH)
                            try:
                                with conn:
                                    conn.execute("DELETE FROM files WHERE status = 'active'")
                            finally:
                                conn.close()

                        with st.spinner("Indexing folder metadata..."):
                            count = scan_folder(str(target_path))
                        st.success(f"Indexed {count:,} files!")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Scan error: {ex}")

        with col_an:
            if st.button("⚡ Re-Analyze", use_container_width=True):
                prog = st.progress(0.0)
                status = st.empty()
                try:
                    run_pipeline_analysis(prog, status)
                    st.success("Analysis updated!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Analysis error: {ex}")

        if st.button("🗑️ Reset / Clear All Scan Data", use_container_width=True):
            conn = db.get_conn(DB_PATH)
            try:
                with conn:
                    conn.execute("DELETE FROM files WHERE status = 'active'")
            finally:
                conn.close()
            st.success("Database reset. Ready for a new folder scan.")
            st.rerun()

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
