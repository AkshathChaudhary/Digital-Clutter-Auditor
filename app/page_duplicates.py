"""Duplicate Groups review page for ClutterIQ.

Displays:
1. Exact duplicates grouped by byte-for-byte SHA-256 hash.
2. Near-duplicate images grouped by perceptual hash (pHash).
3. Semantic image clusters grouped by CLIP embeddings.
4. Badges indicating which file is the designated 'Keeper' vs duplicate copy.
5. Content preview for JSON/text files and thumbnail previews for images.
6. Safe quarantine actions to isolate redundant duplicates.
"""

import json
from pathlib import Path
import textwrap
from typing import List
import pandas as pd
import streamlit as st

from config import DB_PATH
import db

try:
    from safety.quarantine import quarantine_file
except ImportError:
    def quarantine_file(file_id: int, db_path=DB_PATH):
        db.update_columns(file_id, db_path=db_path, status="quarantined")
        return Path("quarantined")


def format_bytes(bytes_val: float) -> str:
    """Format bytes into human-readable strings."""
    if bytes_val < 1024:
        return f"{bytes_val:.0f} B"
    elif bytes_val < 1024**2:
        return f"{bytes_val / 1024:.1f} KB"
    elif bytes_val < 1024**3:
        return f"{bytes_val / (1024**2):.1f} MB"
    else:
        return f"{bytes_val / (1024**3):.2f} GB"


def render_duplicates_page(db_path: Path | str = DB_PATH) -> None:
    """Render the Duplicate Groups explorer interface."""
    st.title("👥 Duplicate Groups & Classification Explorer")
    st.markdown(
        "Inspect and verify file duplicates. Every group is explicitly classified by match type: "
        "**Exact Matches** (cryptographic SHA-256 byte-for-byte identical), **Near-Duplicates** (visual pHash), "
        "and **Semantic Clusters** (CLIP AI embeddings)."
    )

    # Fetch active records that belong to any duplicate group
    query = """
    SELECT id, path, name, ext, kind, size, width, height, blur_score, sha256, phash,
           exact_group, near_group, semantic_group, is_keeper
    FROM files
    WHERE status = 'active'
      AND (exact_group IS NOT NULL OR near_group IS NOT NULL OR semantic_group IS NOT NULL)
    ORDER BY id ASC
    """
    db.resolve_lone_duplicates(db_path=db_path)
    df = db.fetch_df(query, db_path=db_path)

    if df.empty:
        st.info("ℹ️ No duplicate groups found yet. Run **'Run Analysis'** from the sidebar to detect duplicate files.")
        return

    # Filter strictly to duplicate groups with > 1 active files
    exact_df = df[df["exact_group"].notna()]
    near_df = df[df["near_group"].notna()]
    semantic_df = df[df["semantic_group"].notna()]

    valid_exact = set(exact_df["exact_group"].value_counts()[lambda c: c > 1].index)
    valid_near = set(near_df["near_group"].value_counts()[lambda c: c > 1].index)
    valid_sem = set(semantic_df["semantic_group"].value_counts()[lambda c: c > 1].index)

    exact_df = exact_df[exact_df["exact_group"].isin(valid_exact)]
    near_df = near_df[near_df["near_group"].isin(valid_near)]
    semantic_df = semantic_df[semantic_df["semantic_group"].isin(valid_sem)]

    num_exact_groups = len(valid_exact)
    num_near_groups = len(valid_near)
    num_semantic_groups = len(valid_sem)

    # Recoverable duplicate space (sum of non-keepers)
    dup_non_keepers = df[df["is_keeper"] == 0]
    total_dup_recoverable = float(dup_non_keepers["size"].sum())

    # Summary KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Exact Matches (SHA-256)", f"{num_exact_groups} groups")
    col2.metric("🖼️ Near-Duplicates (pHash)", f"{num_near_groups} groups")
    col3.metric("🧠 Semantic Clusters (CLIP)", f"{num_semantic_groups} clusters")
    col4.metric("Recoverable Space", format_bytes(total_dup_recoverable))

    st.divider()

    # Mode Selector
    view_mode = st.radio(
        "Select Duplicate Analysis & Classification Mode:",
        options=[
            f"🎯 Exact Matches (SHA-256) — {num_exact_groups} groups",
            f"🖼️ Near-Duplicate Images (pHash) — {num_near_groups} groups",
            f"🧠 Semantic Clusters (CLIP AI) — {num_semantic_groups} clusters",
        ],
        horizontal=True,
    )

    if "Exact Matches" in view_mode:
        _render_exact_section(exact_df, db_path=db_path)
    elif "Near-Duplicate Images" in view_mode:
        _render_near_section(near_df, db_path=db_path)
    else:
        _render_semantic_section(semantic_df, db_path=db_path)


def _render_exact_section(df: pd.DataFrame, db_path: Path | str) -> None:
    """Render groups for Exact Matches with content inspection."""
    st.subheader("🎯 Exact Duplicates (Cryptographically Identical via SHA-256)")
    st.info(
        "**Verification Guarantee**: Every file in an exact group has been verified with a full SHA-256 "
        "cryptographic checksum. Files share 100% bit-exact content. Even if filenames differ, the actual file "
        "payload is identical."
    )

    if df.empty:
        st.write("No exact duplicates found.")
        return

    group_ids = sorted(df["exact_group"].dropna().unique())
    if not group_ids:
        st.success("🎉 All exact duplicates have been neutralized and quarantined!")
        return

    for gid in group_ids:
        group_files = df[df["exact_group"] == gid].sort_values(by="is_keeper", ascending=False)
        non_keepers = group_files[group_files["is_keeper"] == 0]
        group_savings = float(non_keepers["size"].sum())
        sample_hash = group_files["sha256"].dropna().iloc[0] if not group_files["sha256"].dropna().empty else "Unknown"
        sample_size = group_files["size"].iloc[0]

        with st.expander(
            f"🎯 Exact Group #{int(gid)} — {len(group_files)} files | Single File Size: {format_bytes(sample_size)} | Reclaimable: {format_bytes(group_savings)}",
            expanded=True,
        ):
            # Checksum and Verification Banner
            st.markdown(
                f"**Verified SHA-256 Checksum**: `{sample_hash}`  \n"
                f"**Match Type**: `Byte-for-Byte Exact Match` | **Files**: {len(group_files)}"
            )

            # Check if group contains text/json files for preview
            is_json_or_text = any(row["ext"] in ["json", "txt", "csv", "xml", "md", "log"] for _, row in group_files.iterrows())
            if is_json_or_text:
                with st.expander("🔍 Inspect Content / Verify Why These Files Matched", expanded=False):
                    first_file_path = Path(group_files.iloc[0]["path"])
                    if first_file_path.exists():
                        try:
                            content_bytes = first_file_path.read_bytes()[:4096]
                            content_str = content_bytes.decode("utf-8", errors="replace")
                            if group_files.iloc[0]["ext"] == "json":
                                try:
                                    parsed_json = json.loads(content_str)
                                    st.caption("Parsed JSON Preview (Payload is 100% identical across all files in this group):")
                                    st.json(parsed_json)
                                except Exception:
                                    st.code(content_str, language="json")
                            else:
                                st.caption("Text Preview (Identical content):")
                                st.code(content_str)
                            st.caption(f"📁 Verified from: `{first_file_path.name}` ({len(first_file_path.read_bytes())} bytes)")
                        except Exception as read_err:
                            st.caption(f"Could not load preview: {read_err}")

            # Batch action for this group
            btn_col1, btn_col2 = st.columns([3, 1])
            with btn_col1:
                st.caption(
                    f"The original / cleanest filename is marked **👑 Keeper**. "
                    f"Redundant copies are marked **⚠️ Duplicate Copy**."
                )
            with btn_col2:
                if not non_keepers.empty:
                    if st.button(
                        f"🗑️ Quarantine {len(non_keepers)} Copies",
                        key=f"quarantine_exact_{gid}",
                    ):
                        quarantined_count = 0
                        for fid in non_keepers["id"]:
                            try:
                                quarantine_file(int(fid), db_path=db_path)
                                quarantined_count += 1
                            except Exception as ex:
                                st.error(f"Error: {ex}")
                        st.success(f"Moved {quarantined_count} duplicate files to Quarantine Vault!")
                        st.rerun()

            # Render each file card
            cols = st.columns(min(len(group_files), 4))
            for idx, (_, row) in enumerate(group_files.iterrows()):
                col = cols[idx % len(cols)]
                with col:
                    is_keeper = bool(row["is_keeper"])
                    box_class = "keeper-box" if is_keeper else "duplicate-box"
                    badge_html = '<span class="badge-keeper">👑 KEEPER (Original)</span>' if is_keeper else '<span class="badge-dup">⚠️ EXACT COPY</span>'
                    
                    card_html = textwrap.dedent(
                        f"""
                        <div class="{box_class}">
                            <div style="margin-bottom: 8px;">
                                {badge_html}
                            </div>
                            <div style="margin-bottom: 4px;">
                                <span class="chip">{row['kind'].upper()}</span>
                                <span class="chip">{format_bytes(row['size'])}</span>
                            </div>
                        </div>
                        """
                    )
                    st.markdown(card_html, unsafe_allow_html=True)
                    st.write(f"**Name**: `{row['name']}`")

                    file_path = Path(row["path"])
                    if row["kind"] == "image" and file_path.exists():
                        try:
                            st.image(str(file_path), use_container_width=True)
                        except Exception:
                            pass

                    st.caption(f"📁 `{file_path.name}`")

                    if not is_keeper:
                        if st.button("Move to Quarantine", key=f"q_sem_{row['id']}", use_container_width=True):
                            try:
                                quarantine_file(int(row["id"]), db_path=db_path)
                                db.resolve_lone_duplicates(db_path=db_path)
                                st.success("Quarantined!")
                                st.rerun()
                            except Exception as ex:
                                st.error(f"Failed: {ex}")

                    if not is_keeper:
                        if st.button("Move to Quarantine", key=f"q_exact_{row['id']}", use_container_width=True):
                            try:
                                quarantine_file(int(row["id"]), db_path=db_path)
                                st.success("Quarantined!")
                                st.rerun()
                            except Exception as ex:
                                st.error(f"Failed: {ex}")


def _render_near_section(df: pd.DataFrame, db_path: Path | str) -> None:
    """Render groups for Perceptual Near-Duplicates with quality ranking."""
    st.subheader("🖼️ Perceptual Near-Duplicates & Image Quality Ranking")
    st.info(
        "**Perceptual & Blur Ranking**: Uses DCT-based perceptual hashing (pHash) to find images with Hamming "
        r"distance $\le 6$. ClutterIQ then applies a multi-factor quality ranking algorithm: Laplacian blur variance "
        r"and resolution ($W \times H$) to designate the highest-fidelity version as **👑 Keeper**."
    )

    if df.empty:
        st.write("No perceptual near-duplicates found.")
        return

    group_ids = sorted(df["near_group"].dropna().unique())
    if not group_ids:
        st.success("🎉 All near-duplicate images have been neutralized and quarantined!")
        return

    for gid in group_ids:
        group_files = df[df["near_group"] == gid].sort_values(by="is_keeper", ascending=False)
        non_keepers = group_files[group_files["is_keeper"] == 0]
        group_savings = float(non_keepers["size"].sum())

        with st.expander(
            f"🖼️ Near-Duplicate Group #{int(gid)} — {len(group_files)} images | Reclaimable: {format_bytes(group_savings)}",
            expanded=True,
        ):
            # Batch action header
            header_col1, header_col2 = st.columns([3, 1])
            with header_col1:
                st.caption(
                    "Showing side-by-side visual comparison. The Keeper is automatically selected based on superior resolution and Laplacian sharpness."
                )
            with header_col2:
                if not non_keepers.empty:
                    if st.button(
                        f"🗑️ Quarantine {len(non_keepers)} Copies",
                        key=f"quarantine_near_group_{gid}",
                        type="secondary",
                        use_container_width=True,
                    ):
                        quarantined_count = 0
                        for fid in non_keepers["id"]:
                            try:
                                quarantine_file(int(fid), db_path=db_path)
                                quarantined_count += 1
                            except Exception as ex:
                                st.error(f"Error: {ex}")
                        st.success(f"Moved {quarantined_count} duplicate files to Quarantine Vault!")
                        st.rerun()

            # Side-by-side card grid
            cols = st.columns(min(len(group_files), 4))
            for idx, (_, row) in enumerate(group_files.iterrows()):
                col = cols[idx % len(cols)]
                with col:
                    is_keeper = bool(row["is_keeper"])
                    box_class = "keeper-box" if is_keeper else "duplicate-box"

                    blur_val = row.get("blur_score")
                    blur_text = f"{blur_val:.1f}" if pd.notna(blur_val) else "N/A"
                    w_val = int(row.get("width")) if pd.notna(row.get("width")) and row.get("width") else None
                    h_val = int(row.get("height")) if pd.notna(row.get("height")) and row.get("height") else None
                    res_text = f"{w_val} × {h_val}" if w_val and h_val else "Res N/A"

                    if is_keeper:
                        badge_html = '<span class="badge-keeper">👑 DESIGNATED KEEPER</span>'
                        blur_chip = f'<span class="chip-sharp">Sharpness: {blur_text}</span>'
                    else:
                        if pd.notna(blur_val) and blur_val < 100.0:
                            badge_html = '<span class="badge-dup">⚠️ BLURRY COPY</span>'
                            blur_chip = f'<span class="chip-blurry">Blurry: {blur_text}</span>'
                        else:
                            badge_html = '<span class="badge-dup">⚠️ SIMILAR COPY</span>'
                            blur_chip = f'<span class="chip">Sharpness: {blur_text}</span>'

                    near_card_html = textwrap.dedent(
                        f"""
                        <div class="{box_class}">
                            <div style="margin-bottom: 8px;">
                                {badge_html}
                            </div>
                            <div style="margin-bottom: 4px;">
                                {blur_chip}
                                <span class="chip">{res_text}</span>
                                <span class="chip">{format_bytes(row['size'])}</span>
                            </div>
                        </div>
                        """
                    )
                    st.markdown(near_card_html, unsafe_allow_html=True)

                    file_path = Path(row["path"])
                    if file_path.exists():
                        try:
                            st.image(str(file_path), use_container_width=True)
                        except Exception:
                            pass

                    st.caption(f"📁 `{file_path.name}`")

                    if not is_keeper:
                        if st.button("Move to Quarantine", key=f"q_sem_{row['id']}", use_container_width=True):
                            try:
                                quarantine_file(int(row["id"]), db_path=db_path)
                                db.resolve_lone_duplicates(db_path=db_path)
                                st.success("Quarantined!")
                                st.rerun()
                            except Exception as ex:
                                st.error(f"Failed: {ex}")

                    if not is_keeper:
                        if st.button("Move to Quarantine", key=f"q_near_{row['id']}", use_container_width=True):
                            try:
                                quarantine_file(int(row["id"]), db_path=db_path)
                                st.success("Quarantined!")
                                st.rerun()
                            except Exception as ex:
                                st.error(f"Failed: {ex}")


def _render_semantic_section(df: pd.DataFrame, db_path: Path | str) -> None:
    """Render groups for Semantic Image Clusters."""
    st.subheader("🧠 Semantic Clusters (CLIP ViT-B-32 Deep Learning)")
    st.info(
        "**Semantic AI Clustering**: Groups images based on high-level conceptual similarity using 512-dimensional "
        r"CLIP ViT-B-32 visual embeddings and HNSW graph cosine similarity ($\ge 0.88$). "
        "Files here share themes, scenes, or the same subject photographed from different camera angles and lighting conditions."
    )

    if df.empty:
        st.write("No semantic clusters found.")
        return

    group_ids = sorted(df["semantic_group"].dropna().unique())

    for gid in group_ids:
        group_files = df[df["semantic_group"] == gid].sort_values(by="is_keeper", ascending=False)
        with st.expander(f"🧠 Semantic Cluster #{int(gid)} — {len(group_files)} Related Images (Different Angles / Contexts)", expanded=True):
            st.caption("These images share the same visual concept or subject across varying perspectives or lighting.")
            cols = st.columns(min(len(group_files), 4))
            for idx, (_, row) in enumerate(group_files.iterrows()):
                col = cols[idx % len(cols)]
                with col:
                    is_keeper = bool(row["is_keeper"])
                    blur_val = row.get("blur_score")
                    blur_text = f"{blur_val:.1f}" if pd.notna(blur_val) else "N/A"
                    badge_html = '<span class="badge-semantic">🧠 SEMANTIC CLUSTER</span>'
                    if is_keeper:
                        badge_html = '<span class="badge-keeper">👑 BEST QUALITY</span>'

                    semantic_card_html = textwrap.dedent(
                        f"""
                        <div class="semantic-box">
                            <div style="margin-bottom: 6px;">
                                {badge_html}
                            </div>
                            <div>
                                <span class="chip">{format_bytes(row['size'])}</span>
                                <span class="chip-sharp">Sharpness: {blur_text}</span>
                            </div>
                        </div>
                        """
                    )
                    st.markdown(semantic_card_html, unsafe_allow_html=True)

                    file_path = Path(row["path"])
                    if file_path.exists():
                        try:
                            st.image(str(file_path), use_container_width=True)
                        except Exception:
                            pass
                    st.caption(f"📁 `{file_path.name}`")

                    if not is_keeper:
                        if st.button("Move to Quarantine", key=f"q_sem_{row['id']}", use_container_width=True):
                            try:
                                quarantine_file(int(row["id"]), db_path=db_path)
                                db.resolve_lone_duplicates(db_path=db_path)
                                st.success("Quarantined!")
                                st.rerun()
                            except Exception as ex:
                                st.error(f"Failed: {ex}")
