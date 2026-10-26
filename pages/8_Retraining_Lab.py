"""Read-only Stage 4 eligibility and Stage 6 demonstration status."""
from __future__ import annotations

import streamlit as st

from services.data_service import (PROJECT_ROOT, load_active_model_snapshot,
                                   load_latest_stage6_demo_report)
from ui.components import configure_page, render_footer, render_hero, render_section_heading, status_pill

configure_page("Retraining Lab", "⚙")
render_hero("Retraining Lab", "Eligibility and lifecycle evidence",
            "Read-only status for the genuine-label gate, shared active-model resolver, and saved Stage 6 demonstration candidate.")
st.info("No training, approval, promotion, rollback, or persistent monitor action is available from this page.")

active = load_active_model_snapshot(PROJECT_ROOT)
render_section_heading("Active model", "Resolved through the shared API/Stage 1/Stage 3 model resolver.")
if active.get("available"):
    cols = st.columns(4)
    cols[0].metric("Resolved version", active["model_version"])
    cols[1].metric("Threshold", f"{active['threshold']:.2f}")
    cols[2].metric("Feature count", active["feature_count"])
    cols[3].metric("Resolver source", "Activation manifest" if active["manifest_present"] else f"Legacy {active['model_version']} path")
    st.caption(f"Artifact source: {active['source']} · {active.get('model_path') or 'path unavailable'}")
else:
    st.error(active.get("error", "Active resolver state is unavailable."))

render_section_heading("Stage 4 genuine-label eligibility", "Computed with the existing read-only eligibility collector; this does not train or write a report.")
try:
    from src.labels import config as labels_config
    from src.monitoring import config as monitor_config
    from src.retraining.stage4_workflow import collect_eligible_labels, validate_feature_contract

    features = validate_feature_contract()
    eligible, summary = collect_eligible_labels(
        database_path=labels_config.STAGE2_DB_PATH,
        prediction_log_path=monitor_config.STAGE1_JSONL_PATH,
        features=features,
        trusted_sources=labels_config.trusted_label_sources(),
        token_configured=bool(labels_config.trusted_label_token()),
    )
    del eligible
    cols = st.columns(4)
    cols[0].metric("Labels examined", summary["labels_seen"])
    cols[1].metric("Eligible genuine labels", summary["eligible_label_count"])
    cols[2].metric("Eligible fraud labels", summary["eligible_fraud_count"])
    cols[3].metric("Minimum labels / fraud", f"{summary['minimum_verified_labels']} / {summary['minimum_verified_fraud_labels']}")
    if summary["eligible_label_count"] < summary["minimum_verified_labels"] or summary["eligible_fraud_count"] < summary["minimum_verified_fraud_labels"]:
        st.markdown(status_pill("Blocked by current eligibility counts", "blocked"), unsafe_allow_html=True)
    else:
        st.markdown(status_pill("Count gate met; separate policy gates still apply", "warning"), unsafe_allow_html=True)
    with st.expander("Observed label sources and exclusion reasons"):
        st.json({"label_source_breakdown": summary["label_source_breakdown"],
                 "label_state_breakdown": summary["label_state_breakdown"],
                 "exclusions": summary["exclusions"],
                 "trusted_source_authorization_configured": summary["trusted_source_authorization_configured"]})
except Exception as exc:
    st.warning(f"Eligibility snapshot unavailable: {exc}")

render_section_heading("Stage 6 demonstration candidate", "Metrics below are the saved synthetic PaySim demonstration report, not genuine adaptation evidence.")
demo, error = load_latest_stage6_demo_report(PROJECT_ROOT)
if not demo:
    st.info(error or "No Stage 6 report is available.")
else:
    metrics = demo.get("candidate_evaluation_metrics") or {}
    active_metrics = (demo.get("same_rows_v2_comparison") or {}).get("active_metrics") or {}
    cols = st.columns(4)
    cols[0].metric("Candidate status", demo.get("status", "unknown"))
    cols[1].metric("Candidate threshold", metrics.get("threshold", "—"))
    cols[2].metric("Evaluation rows", metrics.get("rows", "—"))
    cols[3].metric("Genuine promotion eligible", "Yes" if demo.get("eligible_for_genuine_promotion") else "No")
    st.write(f"Candidate `{demo.get('model_version', 'unknown')}` · generated {demo.get('generated_at_utc', 'unknown')} · report `{demo.get('_report_path')}`")
    if metrics and active_metrics:
        st.write("Same-row historical comparison:")
        st.dataframe([{"Model": "Active V2", **active_metrics}, {"Model": "Stage 6 demo candidate", **metrics}], use_container_width=True, hide_index=True)
    promotion = demo.get("promotion") or {}
    st.markdown(status_pill(f"Promotion: {promotion.get('status', 'not recorded')}", "historical"), unsafe_allow_html=True)
    st.caption(demo.get("dataset", {}).get("origin", "Dataset provenance not recorded"))

render_section_heading("Rollback inventory", "Read-only directory and manifest checks. Presence does not prove the rollback bundle is valid.")
manifest = PROJECT_ROOT / "models" / "active_model_manifest.json"
history_dir = PROJECT_ROOT / "models" / "promotion_history"
backup_dir = PROJECT_ROOT / "models" / "backup"
active_dir = PROJECT_ROOT / "models" / "active"
st.write(f"Resolver activation manifest: {'present' if manifest.is_file() else 'absent'}")
st.write(f"Promotion history directory: {'present' if history_dir.is_dir() else 'absent'}")
try:
    backup_files = sorted(path.name for path in backup_dir.iterdir() if path.is_file()) if backup_dir.is_dir() else []
    active_files = sorted(path.name for path in active_dir.iterdir() if path.is_file()) if active_dir.is_dir() else []
except OSError:
    backup_files, active_files = [], []
    st.caption("Local active/backup directory contents could not be read.")
st.write("Legacy backup bundle: " + (", ".join(backup_files) if backup_files else "empty or absent"))
st.write("Legacy `models/active` files: " + (", ".join(active_files) if active_files else "empty or absent"))
if active_files and not manifest.is_file():
    st.caption("The shared resolver does not select this legacy directory without an activation manifest.")
if history_dir.is_dir():
    try:
        history_entries = sorted(item.name for item in history_dir.iterdir())
        st.caption("Promotion history entries: " + (", ".join(history_entries) or "none"))
    except OSError:
        st.caption("Promotion history directory could not be read.")
render_footer()
