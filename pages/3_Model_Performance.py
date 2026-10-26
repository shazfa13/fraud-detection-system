"""Read-only model metadata and historical evaluation metrics."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from services.data_service import PROJECT_ROOT, load_active_model_snapshot, load_latest_phase12_evidence, read_json_report, read_project_csv
from ui.components import configure_page, render_footer, render_hero, render_section_heading, status_pill

configure_page("Model Performance", "◈")
render_hero("Model Performance", "Saved evaluation and feature contract", "Verified historical V1/V2 evaluation reports. These synthetic PaySim results are not current streaming performance.")
metadata, metadata_error = read_json_report(PROJECT_ROOT, "models/model_v2_metadata.json")
threshold, _ = read_json_report(PROJECT_ROOT, "models/model_v2_threshold.json")
if isinstance(metadata, dict):
    features = metadata.get("feature_names", [])
    c = st.columns(3)
    active = load_active_model_snapshot(PROJECT_ROOT)
    c[0].metric("Resolver-selected model", str(active.get("model_version", "Unavailable")))
    c[1].metric("Feature count", str(metadata.get("feature_count", len(features))))
    c[2].metric("V2 saved threshold", str((threshold or {}).get("selected_threshold", "Unavailable")))
    if active.get("available"):
        st.caption(f"Shared active-model resolver: {active['model_version']} · threshold {active['threshold']:.2f} · source {active['source']}")
    if isinstance(features, list) and len(features) == 34 and all(isinstance(name, str) for name in features):
        with st.expander("Inspect the ordered 34-feature contract"):
            st.dataframe(pd.DataFrame({"Order": range(1, 35), "Feature": features}), use_container_width=True, hide_index=True)
    else:
        st.warning("Saved V2 metadata does not contain the expected ordered 34-feature list.")
else:
    st.warning(metadata_error or "V2 metadata has an unexpected format.")
comparison, error = read_json_report(PROJECT_ROOT, "reports/model_v1_vs_v2_comparison.json")
render_section_heading("V1 vs V2", "Historical held-out evaluation metrics, not current live performance.")
if isinstance(comparison, dict):
    rows = []
    for key, label in (("v1_on_v2_test_split", "V1"), ("v2_on_v2_test_split", "V2")):
        values = comparison.get(key)
        if isinstance(values, dict): rows.append({"Model": label, **{k: values.get(k) for k in ("precision", "recall", "f1", "pr_auc", "roc_auc", "tp", "tn", "fp", "fn", "threshold", "rows")}})
    if rows: st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
    v2 = comparison.get("v2_on_v2_test_split", {})
    if v2:
        st.caption(f"Evaluation split: V2 held-out test steps 566–743 · {v2.get('rows', '—')} rows; {v2.get('fraud_count', '—')} fraud examples · source: reports/model_v1_vs_v2_comparison.json")
    st.caption(f"Historical Phase 11 decision: {comparison.get('decision', 'not recorded')} (kept as the original decision label).")
    st.caption("Precision: share of predicted frauds that were fraud in the evaluation set. Recall: share of actual frauds detected. F1: precision/recall balance. PR-AUC: precision–recall performance across thresholds.")
else: st.info(error or "Historical comparison report unavailable or malformed.")
importance = read_project_csv(PROJECT_ROOT, "reports/model_v1_feature_importance.csv", limit=50)
if importance:
    render_section_heading("Saved feature importance", "The available report is the historical V1 feature-importance file.")
    st.bar_chart(pd.DataFrame(importance).set_index("feature")["importance"].astype(float).head(20), color="#36c3d9")
render_section_heading("Lifecycle readiness", "The offline lifecycle coordinator is gated on trusted labels and explicit acceptance policy.")
report, _ = read_json_report(PROJECT_ROOT, "reports/recent_label_collection_report.json")
label_type = report.get("ground_truth_label_type") if isinstance(report, dict) else None
label_display = label_type or "Verified label source not configured"
st.markdown("<div class='status-row'>" + status_pill(f"Label source: {label_display}", "warning") + status_pill("Promotion requires explicit Stage 5 approval", "historical") + status_pill("Automatic retraining: Not operational", "neutral") + "</div>", unsafe_allow_html=True)
evidence = load_latest_phase12_evidence(PROJECT_ROOT)
if evidence: st.caption(f"Phase 12 deployment/integration evidence: historical {evidence.get('status')} at {evidence.get('completed_at_utc', 'unknown time')}.")
st.info("Promotion requires verified genuine labels, reliable transaction-ID matching, sufficient labels, and an explicit acceptance policy. Phase 9 currently records simulated historical PaySim labels, which do not satisfy that gate.")
render_footer()
