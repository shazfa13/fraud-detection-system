"""Read-only explorer for saved model prediction records."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from services.data_service import PROJECT_ROOT, load_prediction_history
from ui.components import configure_page, render_footer, render_hero, render_section_heading

configure_page("Transaction Explorer", "⌕")
render_hero("Transaction Explorer", "Recorded model predictions",
            "Search persisted API and streaming prediction outputs. Model predictions are separate from verified ground truth.")
history = load_prediction_history(PROJECT_ROOT)
if not history:
    st.info("No accessible prediction records were found in the configured API logs or known streaming outputs.")
else:
    frame = pd.DataFrame(history)
    st.caption(f"Showing up to {len(frame)} recent rows (maximum 500), combined from existing report and log files. Sources can overlap; records are not silently removed.")
    source_values = sorted(frame["source"].dropna().unique().tolist()) if "source" in frame else []
    source = st.selectbox("Recorded source", ["All sources", *source_values]) if source_values else "All sources"
    if source != "All sources":
        frame = frame[frame["source"] == source]
    search = st.text_input("Search", placeholder="Transaction ID, type, model, or source")
    if search:
        frame = frame[frame.astype(str).apply(lambda column: column.str.contains(search, case=False, na=False)).any(axis=1)]
    selected_predictions = st.multiselect("Predicted class", sorted(frame["prediction"].dropna().unique().tolist()))
    selected_risks = st.multiselect("Risk level", sorted(frame["risk_level"].dropna().unique().tolist()))
    selected_models = st.multiselect("Model version", sorted(frame["model_version"].dropna().unique().tolist()))
    if selected_predictions:
        frame = frame[frame["prediction"].isin(selected_predictions)]
    if selected_risks:
        frame = frame[frame["risk_level"].isin(selected_risks)]
    if selected_models:
        frame = frame[frame["model_version"].isin(selected_models)]
    exact_duplicates = int(frame.duplicated(keep=False).sum())
    if exact_duplicates:
        st.warning(f"{exact_duplicates} rows are exact duplicate values in the filtered view. They remain visible for audit.")
    sortable = [name for name in ("timestamp", "transaction_id", "fraud_probability", "risk_score") if name in frame]
    if sortable:
        sort = st.selectbox("Sort records by", sortable, index=0)
        frame = frame.sort_values(sort, ascending=(sort != "timestamp"), kind="stable", na_position="last")
    render_section_heading("Prediction records", "Missing identifiers remain blank; this table does not infer outcomes from scores.")
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.download_button("Download filtered records", frame.to_csv(index=False).encode("utf-8"), "transaction_predictions.csv", "text/csv")
    st.caption("The downloaded rows include prediction class, score, model version, source, and any identifiers/timestamps actually present in their source.")
render_footer()
