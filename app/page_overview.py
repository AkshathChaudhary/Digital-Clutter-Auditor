"""Overview dashboard page for ClutterIQ.

Displays:
1. Executive KPI metric cards:
   - Total files indexed
   - Total storage occupied
   - Space recoverable in GB (de-duplicated across groups & high-confidence junk)
   - Total duplicate clusters
   - Privacy risk alerts
2. Plotly Treemap of space by directory and file kind.
3. Storage breakdown by file type.
4. Staleness / Age distribution histogram.

Owned by Person B (Intelligence & Dashboard).
"""

from pathlib import Path
import textwrap
import time
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import DB_PATH
import db


def format_bytes(bytes_val: float) -> str:
    """Format bytes into readable B, KB, MB, or GB strings."""
    if bytes_val < 1024:
        return f"{bytes_val:.0f} B"
    elif bytes_val < 1024**2:
        return f"{bytes_val / 1024:.1f} KB"
    elif bytes_val < 1024**3:
        return f"{bytes_val / (1024**2):.1f} MB"
    else:
        return f"{bytes_val / (1024**3):.2f} GB"


def compute_overview_metrics(df: pd.DataFrame) -> Tuple[int, float, float, int, int]:
    """Compute headline KPI values from active files DataFrame.

    Returns:
        Tuple: (total_files, total_bytes, recoverable_bytes, duplicate_groups, privacy_alerts)
    """
    if df.empty:
        return 0, 0.0, 0.0, 0, 0

    total_files = len(df)
    total_bytes = float(df["size"].sum())

    # Calculate Recoverable Space without double-counting:
    # A file is recoverable if:
    # 1. It is a non-keeper duplicate (exact, near, or semantic)
    # OR 2. It has high-confidence ML/heuristic junk_score >= 0.8
    is_dup_non_keeper = (
        (df["exact_group"].notna() | df["near_group"].notna() | df["semantic_group"].notna())
        & (df["is_keeper"] == 0)
    )
    is_high_junk = df["junk_score"].fillna(0.0) >= 0.8

    recoverable_mask = is_dup_non_keeper | is_high_junk
    recoverable_bytes = float(df.loc[recoverable_mask, "size"].sum())

    # Count unique unresolved duplicate clusters across all methods (groups with > 1 active file)
    def count_active_groups(col: str) -> int:
        if col not in df or df[col].dropna().empty:
            return 0
        counts = df[col].dropna().value_counts()
        return int((counts > 1).sum())

    total_dup_groups = (
        count_active_groups("exact_group")
        + count_active_groups("near_group")
        + count_active_groups("semantic_group")
    )

    # Privacy risk alerts
    privacy_alerts = int(df["privacy_type"].notna().sum())

    return total_files, total_bytes, recoverable_bytes, total_dup_groups, privacy_alerts


def render_overview_page(db_path: Path | str = DB_PATH) -> None:
    """Render the main Overview dashboard in Streamlit."""
    st.title(" Digital Hygiene Overview")
    st.markdown("Real-time telemetry on storage consumption, duplicate clusters, and privacy risks.")

    # Load active records from database
    query = """
    SELECT id, path, name, ext, kind, size, created, modified,
           exact_group, near_group, semantic_group, is_keeper,
           junk_score, privacy_type
    FROM files
    WHERE status = 'active'
    """
    df = db.fetch_df(query, db_path=db_path)

    if df.empty:
        st.info(" No files scanned yet. Use the sidebar on the left to select a folder and click **'Run Scan'** to begin.")
        return

    # Instant sub-millisecond KPI retrieval from SQLite C-engine
    total_files, total_bytes, recoverable_bytes, total_groups, privacy_alerts = db.get_overview_kpis(db_path=db_path)

    # Breakdown counts (strictly duplicate groups with > 1 active files)
    conn = db.get_conn(db_path)
    try:
        exact_count = conn.execute("SELECT COUNT(*) FROM (SELECT exact_group FROM files WHERE status = 'active' AND exact_group IS NOT NULL GROUP BY exact_group HAVING COUNT(*) > 1)").fetchone()[0]
        near_count = conn.execute("SELECT COUNT(*) FROM (SELECT near_group FROM files WHERE status = 'active' AND near_group IS NOT NULL GROUP BY near_group HAVING COUNT(*) > 1)").fetchone()[0]
        semantic_count = conn.execute("SELECT COUNT(*) FROM (SELECT semantic_group FROM files WHERE status = 'active' AND semantic_group IS NOT NULL GROUP BY semantic_group HAVING COUNT(*) > 1)").fetchone()[0]
        quarantined_count = conn.execute("SELECT COUNT(*) FROM files WHERE status = 'quarantined'").fetchone()[0]
        quarantined_bytes = conn.execute("SELECT COALESCE(SUM(size), 0) FROM files WHERE status = 'quarantined'").fetchone()[0]
        deleted_count = conn.execute("SELECT COUNT(*) FROM files WHERE status = 'deleted'").fetchone()[0]
        deleted_bytes = conn.execute("SELECT COALESCE(SUM(size), 0) FROM files WHERE status = 'deleted'").fetchone()[0]
    finally:
        conn.close()

    # Dynamic Digital Hygiene Score (Directly reflects current active risks vs resolved actions)
    hygiene_score = 100
    if total_groups > 0:
        hygiene_score -= min(40, total_groups * 10)
    if privacy_alerts > 0:
        hygiene_score -= min(40, privacy_alerts * 25)
    if recoverable_bytes > (50 * 1024):
        hygiene_score -= 10
    hygiene_score = max(15, hygiene_score)

    # When all threats and duplicates are moved to vault or deleted, score is pristine
    if total_groups == 0 and privacy_alerts == 0:
        hygiene_score = 100
        grade_letter, grade_color, grade_desc = "A+", "#10B981", "Pristine Digital Hygiene — All Clutter & Risks Neutralized"
    elif hygiene_score >= 88:
        grade_letter, grade_color, grade_desc = "A", "#10B981", "Optimal System Hygiene — Minimal Clutter"
    elif hygiene_score >= 75:
        grade_letter, grade_color, grade_desc = "B+", "#38BDF8", "Good Hygiene — Actionable Duplicate Clutter Found"
    elif hygiene_score >= 60:
        grade_letter, grade_color, grade_desc = "C", "#F59E0B", "Attention Required — Redundant Data & Sensitive PII Exposed"
    else:
        grade_letter, grade_color, grade_desc = "D", "#EF4444", "High Clutter & Critical Unprotected Privacy Risks"

    # Executive Hygiene Dashboard Hero Banner
    cleaned_badge = ""
    if quarantined_count > 0 or deleted_count > 0:
        purge_part = f'<span style="background: rgba(139, 92, 246, 0.2); color: #C084FC; border: 1px solid rgba(139, 92, 246, 0.4); padding: 4px 10px; border-radius: 6px; font-size: 0.82rem; font-weight: 600;">💥 {deleted_count} permanently purged ({format_bytes(deleted_bytes)} freed)</span>' if deleted_count > 0 else ''
        cleaned_badge = (
            f'<div style="margin-top: 10px; display: inline-flex; gap: 8px; flex-wrap: wrap;">'
            f'<span style="background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.4); padding: 4px 10px; border-radius: 6px; font-size: 0.82rem; font-weight: 600;">'
            f'🛡️ {quarantined_count} files isolated in vault ({format_bytes(quarantined_bytes)})'
            f'</span>'
            f'{purge_part}'
            f'</div>'
        )

    security_status_text = '🔒 100% Privacy Secure' if privacy_alerts == 0 else f'⚠️ {privacy_alerts} Exposed Identity Risk(s)'
    security_status_color = '#10B981' if privacy_alerts == 0 else '#F43F5E'

    hero_html = (
        f'<div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.9) 100%); '
        f'border: 1px solid rgba(255, 255, 255, 0.08); border-left: 6px solid {grade_color}; border-radius: 14px; '
        f'padding: 22px 26px; margin-bottom: 24px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">'
        f'<div>'
        f'<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">'
        f'<span style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; color: #94A3B8; font-weight: 700;">Digital Hygiene Audit</span>'
        f'<span style="background: {grade_color}22; color: {grade_color}; border: 1px solid {grade_color}66; padding: 2px 8px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700;">'
        f'Grade {grade_letter}'
        f'</span>'
        f'</div>'
        f'<h2 style="margin: 0 0 6px 0; color: #F8FAFC; font-size: 1.85rem; font-weight: 800; letter-spacing: -0.02em;">'
        f'System Hygiene Score: <span style="color: {grade_color};">{hygiene_score}/100</span>'
        f'</h2>'
        f'<p style="margin: 0; color: #CBD5E1; font-size: 0.95rem;">'
        f'{grade_desc} • <strong>{format_bytes(recoverable_bytes)}</strong> recoverable across <strong>{total_groups}</strong> active duplicate groups'
        f'</p>'
        f'{cleaned_badge}'
        f'</div>'
        f'<div style="text-align: right; margin-top: 10px;">'
        f'<div style="background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.06); padding: 12px 18px; border-radius: 10px; display: inline-block;">'
        f'<div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Security Status</div>'
        f'<div style="font-weight: 700; font-size: 1.05rem; color: {security_status_color};">{security_status_text}</div>'
        f'</div>'
        f'</div>'
        f'</div>'
        f'<!-- Progress indicator bar -->'
        f'<div style="width: 100%; height: 8px; background: rgba(255, 255, 255, 0.08); border-radius: 9999px; margin-top: 18px; overflow: hidden;">'
        f'<div style="width: {hygiene_score}%; height: 100%; background: linear-gradient(90deg, {grade_color}aa, {grade_color}); border-radius: 9999px; transition: width 0.5s ease;"></div>'
        f'</div>'
        f'</div>'
    )
    if hasattr(st, "html"):
        st.html(hero_html)
    else:
        st.markdown(hero_html, unsafe_allow_html=True)

    # Top KPI Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Files", f"{total_files:,}")
    with col2:
        st.metric("Total Storage", format_bytes(total_bytes))
    with col3:
        st.metric(
            "Recoverable Space",
            format_bytes(recoverable_bytes),
            delta=f"{(recoverable_bytes / total_bytes * 100):.1f}% of disk" if total_bytes > 0 else "0%",
            delta_color="normal",
        )
    with col4:
        st.metric(
            "Duplicate Clusters",
            f"{total_groups:,}",
            delta=f"{exact_count} Exact | {near_count} Near | {semantic_count} CLIP",
            delta_color="off",
        )
    with col5:
        st.metric(
            "Privacy Risks",
            f"{privacy_alerts:,}",
            delta=f"{privacy_alerts} sensitive files" if privacy_alerts > 0 else "None",
            delta_color="inverse" if privacy_alerts > 0 else "off",
        )

    st.divider()

    # Visualizations Row 1: Treemap & Bar Chart
    row1_left, row1_right = st.columns([3, 2])

    with row1_left:
        st.subheader(" Storage Treemap")
        # Extract immediate parent directory for visual grouping
        df["parent_folder"] = df["path"].apply(
            lambda p: Path(p).parent.name if Path(p).parent.name else "Root"
        )
        df["size_mb"] = df["size"] / (1024**2)

        # Plotly treemap
        fig_treemap = px.treemap(
            df,
            path=["parent_folder", "kind"],
            values="size_mb",
            title="Storage by Directory and Category (MB)",
            color="size_mb",
            color_continuous_scale="Teal",
        )
        fig_treemap.update_layout(template="plotly_dark", margin=dict(t=30, l=10, r=10, b=10))
        st.plotly_chart(fig_treemap, use_container_width=True)

    with row1_right:
        st.subheader(" Breakdown by Category")
        kind_summary = (
            df.groupby("kind")["size"]
            .sum()
            .reset_index()
            .sort_values(by="size", ascending=False)
        )
        kind_summary["size_formatted"] = kind_summary["size"].apply(format_bytes)
        kind_summary["size_mb"] = kind_summary["size"] / (1024**2)

        fig_bar = px.bar(
            kind_summary,
            x="kind",
            y="size_mb",
            text="size_formatted",
            color="kind",
            title="Total Size by File Kind (MB)",
            labels={"kind": "Category", "size_mb": "MB"},
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        fig_bar.update_layout(template="plotly_dark", showlegend=False, margin=dict(t=30, l=10, r=10, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # Visualizations Row 2: Staleness Distribution Histogram
    st.subheader(" File Staleness (Age Distribution)")
    now = time.time()
    df["age_days"] = np.maximum(0.0, (now - df["modified"]) / 86400.0)

    fig_hist = px.histogram(
        df,
        x="age_days",
        nbins=40,
        title="File Age Distribution (Days since Last Modification)",
        labels={"age_days": "Age (Days)"},
        color_discrete_sequence=["#818CF8"],
    )
    fig_hist.add_vline(
        x=365,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text="Stale Threshold (365 days)",
        annotation_position="top right",
    )
    fig_hist.update_layout(template="plotly_dark", margin=dict(t=30, l=10, r=10, b=10))
    st.plotly_chart(fig_hist, use_container_width=True)

