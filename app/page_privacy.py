"""Privacy & Sensitive Documents audit page for ClutterIQ.

Displays:
1. Filterable audit table of identified sensitive personal and financial files.
2. STRICT DATA MASKING: Shows only masked values (e.g. 'XXXX XXXX 1234', 'ABCDE****F').
3. Multi-file selection with single-click 'Move Selected to Quarantine'.
4. Heuristic disclaimer explaining detection boundaries.

Owned by Person B (Intelligence & Dashboard).
"""

from pathlib import Path
import streamlit as st

from config import DB_PATH
import db

# Graceful import of Person A's quarantine engine
try:
    from safety.quarantine import quarantine_files
except ImportError:
    def quarantine_files(file_ids: list[int]) -> int:
        for fid in file_ids:
            db.update_columns(fid, status="quarantined")
        return len(file_ids)


def render_privacy_page(db_path: Path | str = DB_PATH) -> None:
    """Render the Privacy Risk Audit interface in Streamlit."""
    st.title(" Privacy Risk Auditor")
    st.markdown("Identifies exposed identity documents, payment cards, and bank statements residing on local storage.")

    # Informational Disclaimer
    st.info(
        "**Privacy & Security Disclaimer**: ClutterIQ operates strictly locally on your machine. "
        "No file contents or extracted text ever leave your computer. Sensitive identifiers are "
        "always masked and raw numbers are never stored in the database. Detection relies on OCR "
        "and checksum heuristics; please review documents before archiving."
    )

    # Fetch sensitive active files
    query = """
    SELECT id, name, path, kind, size, privacy_type, privacy_masked, modified
    FROM files
    WHERE status = 'active' AND privacy_type IS NOT NULL
    ORDER BY id DESC
    """
    df = db.fetch_df(query, db_path=db_path)

    if df.empty:
        st.success(" **No privacy risks detected!** None of your active scanned files contain exposed Aadhaar, PAN, card numbers, or bank statements.")
        return

    st.subheader(f" Identified Sensitive Files ({len(df)})")

    # Filters
    col_filter1, col_filter2 = st.columns([2, 2])
    with col_filter1:
        type_options = ["All"] + sorted(list(df["privacy_type"].unique()))
        selected_type = st.selectbox("Filter by Risk Category:", type_options)
    with col_filter2:
        search_kw = st.text_input("Search filename or path:", "")

    filtered_df = df.copy()
    if selected_type != "All":
        filtered_df = filtered_df[filtered_df["privacy_type"] == selected_type]
    if search_kw:
        filtered_df = filtered_df[
            filtered_df["name"].str.contains(search_kw, case=False)
            | filtered_df["path"].str.contains(search_kw, case=False)
        ]

    # Selection table for Quarantine action
    st.markdown("Select files to move into the **Quarantine Vault**:")

    # Multi-row selection using st.data_editor with checkbox column
    display_df = filtered_df[["id", "name", "privacy_type", "privacy_masked", "path"]].copy()
    display_df.insert(0, "Select", False)

    edited_df = st.data_editor(
        display_df,
        column_config={
            "Select": st.column_config.CheckboxColumn("Select", default=False),
            "id": st.column_config.NumberColumn("File ID", disabled=True, width="small"),
            "name": st.column_config.TextColumn("Filename", disabled=True),
            "privacy_type": st.column_config.TextColumn("Threat Category", disabled=True),
            "privacy_masked": st.column_config.TextColumn("Masked Finding (Redacted)", disabled=True),
            "path": st.column_config.TextColumn("Full Path", disabled=True),
        },
        hide_index=True,
        use_container_width=True,
        key="privacy_editor",
    )

    selected_rows = edited_df[edited_df["Select"]]
    selected_ids = selected_rows["id"].tolist()

    col_action1, col_action2 = st.columns([2, 4])
    with col_action1:
        if st.button(
            f" Move Selected ({len(selected_ids)}) to Quarantine",
            type="primary",
            disabled=len(selected_ids) == 0,
        ):
            count = quarantine_files(selected_ids)
            st.success(f"Successfully moved {count} sensitive files to the Quarantine Vault!")
            time_delay = 0.5
            st.rerun()

    with col_action2:
        if len(selected_ids) > 0:
            st.caption(f"Ready to quarantine: {', '.join(selected_rows['name'].head(3))}...")

