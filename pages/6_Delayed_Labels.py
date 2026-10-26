"""Read-only Stage 2 delayed-label registry view."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from services.data_service import load_stage2_label_stats, stage2_label_overview
from ui.components import configure_page, render_footer, render_hero, render_section_heading

configure_page("Delayed Labels", "◷")
render_hero("Delayed Labels", "Stage 2 label registry",
            "Counts and provenance are read from the existing SQLite database. This page never edits labels or reveals configured credentials.")
stats = load_stage2_label_stats()
overview = stage2_label_overview()
if not stats.get("store_available"):
    st.info(stats.get("error", "Stage 2 store is unavailable."))
else:
    keys = (("Predictions registered", "predictions_registered"), ("Unresolved predictions", "pending_labels"),
            ("Resolved genuine labels", "confirmed_genuine_labels"), ("Synthetic test labels", "synthetic_test_labels"))
    cols = st.columns(4)
    for col, (label, key) in zip(cols, keys):
        col.metric(label, f"{int(stats.get(key, 0)):,}")
    second = st.columns(3)
    for col, (label, key) in zip(second, (("Identical duplicate submissions", "duplicate_submissions"),
                                         ("Conflicting submissions", "conflicting_labels"),
                                         ("Unmatched pending labels", "unmatched_pending_submissions"))):
        col.metric(label, f"{int(stats.get(key, 0)):,}")
    if not stats.get("genuine_source_configured"):
        st.warning("Trusted genuine-label source is not configured. No label is treated as eligible for retraining.")
    elif not stats.get("genuine_labels_available"):
        st.info("An authorized source is configured, but the registry has no eligible genuine verified labels.")
    else:
        st.success("At least one joined genuine verified label is available; see Stage 4 gates before any training action.")

render_section_heading("Label provenance and resolution", "Grouped counts from labels(source_type, label_source, status); no record contents or tokens are displayed.")
if overview.get("available") and overview.get("labels"):
    st.dataframe(pd.DataFrame(overview["labels"]), use_container_width=True, hide_index=True)
elif overview.get("available"):
    st.info("No label records are stored.")
else:
    st.info(overview.get("error", "Label provenance is unavailable."))
render_section_heading("Submission audit dispositions", "Duplicate, accepted, rejected, and conflict decisions remain in the Stage 2 audit table.")
if overview.get("available") and overview.get("submissions"):
    st.dataframe(pd.DataFrame(overview["submissions"]), use_container_width=True, hide_index=True)
else:
    st.caption("No submission audit rows are available.")
st.caption("Synthetic test outcomes remain visibly separate from trusted verified outcomes and cannot qualify as genuine labels.")
render_footer()
