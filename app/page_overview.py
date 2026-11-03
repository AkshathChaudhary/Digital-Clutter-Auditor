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

    # Count unique duplicate clusters across all methods
    exact_groups = set(df["exact_group"].dropna().unique())
    near_groups = set(df["near_group"].dropna().unique())
    semantic_groups = set(df["semantic_group"].dropna().unique())
    total_dup_groups = len(exact_groups) + len(near_groups) + len(semantic_groups)

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
        st.metric("Duplicate Clusters", f"{total_groups:,}")
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
        fig_treemap.update_layout(margin=dict(t=30, l=10, r=10, b=10))
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
        )
        fig_bar.update_layout(showlegend=False, margin=dict(t=30, l=10, r=10, b=10))
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
        color_discrete_sequence=["#4F46E5"],
    )
    fig_hist.add_vline(
        x=365,
        line_dash="dash",
        line_color="red",
        annotation_text="Stale Threshold (365 days)",
        annotation_position="top right",
    )
    fig_hist.update_layout(margin=dict(t=30, l=10, r=10, b=10))
    st.plotly_chart(fig_hist, use_container_width=True)

