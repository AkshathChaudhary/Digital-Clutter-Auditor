"""Suggestions and Active Learning labelling page for ClutterIQ.

Displays:
1. Ranked junk recommendations sorted descending by `junk_score`.
2. Human-readable explainable rationale for every score.
3. Interactive Keep / Delete labelling controls directly feeding the active learning loop.
4. 'Retrain Machine Learning Model' trigger displaying live cross-validation metrics.

Owned by Person B (Intelligence & Dashboard).
"""

from pathlib import Path
import streamlit as st

from config import DB_PATH, MODEL_PATH
import db
from ml.junk_model import retrain

try:
    from safety.quarantine import quarantine_files
except ImportError:
    def quarantine_files(file_ids: list[int]) -> int:
        for fid in file_ids:
            db.update_columns(fid, status="quarantined")
        return len(file_ids)


def render_suggestions_page(db_path: Path | str = DB_PATH) -> None:
    """Render the ranked cleanup recommendations and active learning interface."""
    st.title(" Intelligent Cleanup Recommendations")
    st.markdown("Files prioritized by junk probability with explainable justifications and interactive active learning feedback.")

    # 1. Fetch active records with scores
    query = """
    SELECT id, name, path, kind, size, junk_score, junk_reason, user_label
    FROM files
    WHERE status = 'active' AND junk_score IS NOT NULL
    ORDER BY junk_score DESC, size DESC
    """
    df = db.fetch_df(query, db_path=db_path)

    if df.empty:
        st.info(" No files scored yet. Click **'Run Analysis'** in the sidebar to score active files using the ML/heuristic engine.")
        return

    # Count labelled files
    labelled_count = int(df["user_label"].isin(["keep", "delete"]).sum())
    min_required = 100

    # 2. Top Retrain Bar & Active Learning Status
    status_col1, status_col2 = st.columns([3, 2])
    with status_col1:
        progress_val = min(1.0, labelled_count / min_required)
        st.progress(
            progress_val,
            text=f"Active Learning Progress: {labelled_count}/{min_required} labelled files for Random Forest",
        )
    with status_col2:
        if st.button(" Retrain Machine Learning Model", type="primary"):
            with st.spinner("Training Random Forest with 5-fold cross-validation..."):
                metrics, rescored_count = retrain(db_path=db_path, min_samples=min_required)
                if metrics is not None:
                    st.success(
                        f" Model successfully updated and serialized! "
                        f"Accuracy: **{metrics['accuracy']:.1%}** | "
                        f"F1-Score: **{metrics['f1']:.1%}** | "
                        f"Trained on {metrics['n_samples']} samples."
                    )
                else:
                    st.warning(
                        f"Need at least {min_required} labelled files with both 'keep' and 'delete' "
                        f"labels to train the Random Forest. Currently have {labelled_count}. "
                        "Continuing with explainable heuristic scoring."
                    )
                st.rerun()

    st.divider()

    # 3. Quick Action: Batch Quarantine High-Confidence Junk
    high_junk_df = df[(df["junk_score"] >= 0.85) & (df["user_label"] != "keep")]
    if not high_junk_df.empty:
        st.subheader(" High-Confidence Recommendations")
        st.caption(f"{len(high_junk_df)} files have junk probability $\\ge 85\\%$.")
        if st.button(f" Move All {len(high_junk_df)} High-Confidence Junk to Quarantine Vault"):
            count = quarantine_files(high_junk_df["id"].tolist())
            st.success(f"Moved {count} files to quarantine!")
            st.rerun()

    st.divider()

    # 4. Interactive Review & Labelling Table
    st.subheader(" Review & Feedback Table")
    st.caption("Label files as **delete** or **keep** below. Your choices directly train your personalized ML model.")

    display_df = df[["id", "junk_score", "name", "junk_reason", "user_label", "path"]].copy()
    display_df["junk_score"] = display_df["junk_score"].apply(lambda s: f"{s:.1%}")
    display_df["user_label"] = display_df["user_label"].fillna("unlabelled")

    # Editable table
    edited_df = st.data_editor(
        display_df,
        column_config={
            "id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
            "junk_score": st.column_config.TextColumn("Junk Prob", disabled=True, width="small"),
            "name": st.column_config.TextColumn("File Name", disabled=True),
            "junk_reason": st.column_config.TextColumn("Why is this junk? (Reason)", disabled=True, width="large"),
            "user_label": st.column_config.SelectboxColumn(
                "Feedback Label",
                help="Mark file to teach the ML model",
                options=["unlabelled", "delete", "keep"],
                required=True,
            ),
            "path": st.column_config.TextColumn("Path", disabled=True),
        },
        hide_index=True,
        use_container_width=True,
        key="suggestions_editor",
    )

    # Save button for updated labels
    if st.button(" Save Updated Labels"):
        changes_saved = 0
        for _, row in edited_df.iterrows():
            fid = int(row["id"])
            lbl = row["user_label"]
            original_lbl = df.loc[df["id"] == fid, "user_label"].values[0]
            clean_lbl = None if lbl == "unlabelled" else lbl

            if clean_lbl != original_lbl:
                db.update_columns(fid, db_path=db_path, user_label=clean_lbl)
                changes_saved += 1

        if changes_saved > 0:
            st.success(f"Saved {changes_saved} label updates to database!")
            st.rerun()
        else:
            st.info("No label changes detected.")

