"""Quarantine Vault and Permanent Sanitization page for ClutterIQ.

Displays:
1. Two-stage hygiene workflow:
   - Stage 1: Safe non-destructive quarantine isolation.
   - Stage 2: Irreversible permanent deletion freeing disk space.
2. List of quarantined files with verified SHA-256 integrity hash.
3. One-click 'Restore File' action returning files bit-for-bit to original paths.
4. One-click 'Delete Permanently' action purging individual files from disk.
5. Batch 'Empty Entire Vault' action with safety confirmation.
"""

from datetime import datetime
from pathlib import Path
import textwrap
import pandas as pd
import streamlit as st

from config import DB_PATH, QUARANTINE_DIR
import db

try:
    from safety.quarantine import (
        restore_file,
        permanently_delete_file,
        permanently_delete_all,
    )
except ImportError:
    def restore_file(quarantine_id: int, db_path=DB_PATH):
        raise NotImplementedError("Quarantine module not available.")

    def permanently_delete_file(quarantine_id: int, db_path=DB_PATH):
        raise NotImplementedError("Quarantine module not available.")

    def permanently_delete_all(db_path=DB_PATH):
        raise NotImplementedError("Quarantine module not available.")


def format_bytes(bytes_val: float) -> str:
    """Format bytes into readable strings."""
    if bytes_val < 1024:
        return f"{bytes_val:.0f} B"
    elif bytes_val < 1024**2:
        return f"{bytes_val / 1024:.1f} KB"
    elif bytes_val < 1024**3:
        return f"{bytes_val / (1024**2):.1f} MB"
    else:
        return f"{bytes_val / (1024**3):.2f} GB"


def render_quarantine_page(db_path: Path | str = DB_PATH) -> None:
    """Render the Quarantine Vault and Permanent Sanitization interface."""
    st.title("🛡️ Quarantine Vault & Data Sanitization")
    st.markdown(
        "Manage isolated files. ClutterIQ implements a **two-stage safe deletion architecture**: "
        "files are first isolated into the vault without risk, and can either be **restored bit-for-bit** "
        "or **permanently purged** from your physical disk."
    )

    db.resolve_lone_duplicates(db_path=db_path)

    # 1. Fetch active quarantine records
    query = """
    SELECT q.id AS q_id, q.file_id, q.original_path, q.quarantine_path,
           q.sha256, q.moved_at, q.restored_at, f.name, f.size, f.kind
    FROM quarantine_log q
    LEFT JOIN files f ON q.file_id = f.id
    WHERE q.restored_at IS NULL AND (f.status = 'quarantined' OR f.status IS NULL)
    ORDER BY q.moved_at DESC
    """
    df = db.fetch_df(query, db_path=db_path)

    # Check permanently deleted stats for historical telemetry
    conn = db.get_conn(db_path)
    try:
        deleted_count = conn.execute("SELECT COUNT(*) FROM files WHERE status = 'deleted'").fetchone()[0]
        deleted_bytes = conn.execute("SELECT COALESCE(SUM(size), 0) FROM files WHERE status = 'deleted'").fetchone()[0]
    finally:
        conn.close()

    total_quarantined_bytes = float(df["size"].fillna(0).sum()) if not df.empty else 0.0

    # Top KPI Metrics Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Items in Vault", f"{len(df):,}")
    kpi2.metric("Vault Storage", format_bytes(total_quarantined_bytes))
    kpi3.metric("Permanently Purged", f"{deleted_count:,} files", delta=f"{format_bytes(deleted_bytes)} reclaimed" if deleted_bytes > 0 else None)
    kpi4.metric("Vault Directory", f"`{QUARANTINE_DIR.name}`")

    st.divider()

    if df.empty:
        st.markdown(
            textwrap.dedent(
                f"""
                <div style="background: rgba(16, 185, 129, 0.12); border: 1.5px solid rgba(16, 185, 129, 0.35); border-radius: 12px; padding: 24px; text-align: center; margin: 20px 0;">
                    <h3 style="color: #34D399; margin: 0 0 8px 0;">🎉 Quarantine Vault is Clean</h3>
                    <p style="color: #E2E8F0; margin: 0; font-size: 0.95rem;">
                        No files are currently quarantined. When you move duplicates, blurry photos, or sensitive files into the vault, they will appear here.
                    </p>
                    {f'<p style="color: #94A3B8; margin-top: 10px; font-size: 0.85rem;">Historical Reclaimed: <strong>{deleted_count}</strong> files permanently removed ({format_bytes(deleted_bytes)} disk space freed).</p>' if deleted_count > 0 else ''}
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
        return

    # Batch Action Toolbar
    st.subheader(f"📋 Quarantined Items ({len(df)})")
    
    batch_col1, batch_col2 = st.columns([3, 2])
    with batch_col1:
        st.caption(
            "Every item in the vault is stored with its verified SHA-256 checksum. "
            "Click **Restore** to return a file to its exact folder, or **Delete Permanently** to unlink it from disk."
        )
    with batch_col2:
        with st.expander("💥 Bulk Sanitization (Empty Entire Vault)", expanded=False):
            st.warning("⚠️ DANGER: This will permanently purge EVERY file currently in the Quarantine Vault at once. To delete only 1 specific file, use the individual 'Delete Permanently' button inside that file's box below.")
            confirm_empty = st.checkbox(f"I understand: Purge ALL {len(df)} files from vault simultaneously", key="confirm_bulk_purge")
            if st.button("🚨 Purge All Vault Files", type="primary", disabled=not confirm_empty, use_container_width=True):
                purged_count, freed_bytes = permanently_delete_all(db_path=db_path)
                db.resolve_lone_duplicates(db_path=db_path)
                st.success(f"💥 Successfully purged {purged_count} files! Reclaimed {format_bytes(freed_bytes)} permanently from disk.")
                st.rerun()

    # Individual Quarantined Items List
    for _, row in df.iterrows():
        qid = int(row["q_id"])
        fid = int(row["file_id"]) if pd.notna(row["file_id"]) else None
        fname = row["name"] or Path(row["original_path"]).name
        fsize = format_bytes(row["size"]) if pd.notna(row["size"]) else "Unknown size"
        moved_str = datetime.fromtimestamp(row["moved_at"]).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(row["moved_at"]) else "Unknown"
        sha_str = str(row["sha256"]) if pd.notna(row["sha256"]) else "Unknown"

        with st.expander(f"📦 {fname} ({fsize}) — Quarantined on {moved_str}", expanded=True):
            col_info, col_actions = st.columns([3, 2])
            with col_info:
                st.markdown(f"**Original Location**: `{row['original_path']}`")
                st.markdown(f"**Vault Path**: `{row['quarantine_path']}`")
                st.markdown(
                    f"**SHA-256 Checksum**: <code style='font-size: 0.78rem;'>{sha_str}</code>",
                    unsafe_allow_html=True,
                )
                if row["kind"] == "image":
                    qpath_obj = Path(row["quarantine_path"])
                    if qpath_obj.exists():
                        try:
                            st.image(str(qpath_obj), width=280)
                        except Exception:
                            pass

            with col_actions:
                st.write("**Sanitization Actions:**")
                # 1. Restore Action
                if st.button(
                    "🔄 Restore to Original Folder",
                    key=f"restore_{qid}",
                    help="Safely moves this file back to its exact original directory",
                    use_container_width=True,
                ):
                    try:
                        restored_path = restore_file(qid, db_path=db_path)
                        st.success(f"Restored to `{restored_path.name}`!")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Restoration failed: {ex}")

                # 2. Permanent Delete Action (Individual file only)
                confirm_single = st.checkbox("Confirm delete this file", key=f"conf_del_{qid}")
                if st.button(
                    "🗑️ Delete Permanently (This File Only)",
                    key=f"perm_del_{qid}",
                    type="primary",
                    disabled=not confirm_single,
                    help="Permanently unlinks only this specific file from your computer",
                    use_container_width=True,
                ):
                    try:
                        ok, deleted_name, freed = permanently_delete_file(qid, db_path=db_path)
                        db.resolve_lone_duplicates(db_path=db_path)
                        st.success(f"💥 Permanently deleted only `{deleted_name}`! Freed {format_bytes(freed)} from disk.")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Deletion failed: {ex}")
