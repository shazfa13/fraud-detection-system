"""Historical Kafka–Spark evidence viewer; does not connect to a broker."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from services.data_service import (PROJECT_ROOT, load_latest_phase12_evidence, read_project_csv,
                                   read_project_jsonl, read_stage1_status, load_stage2_label_stats,
                                   load_stage3_monitor_report, prediction_registry_summary)
from ui.components import configure_page, render_footer, render_hero, render_section_heading, status_pill

configure_page("Live Monitoring", "⇢")
render_hero("Live Monitoring", "Stage 1 status and saved pipeline evidence", "The Stage 1 heartbeat and outputs are read from disk. Broker reachability is not inferred from them; this page never starts or restarts a service.")
stage1_status = read_stage1_status(PROJECT_ROOT)
stage1_rows = read_project_jsonl(PROJECT_ROOT, "data/stage1_streaming/predictions.jsonl")
render_section_heading("Stage 1 continuous stream", "Persisted inference output and the stream process heartbeat.")
if stage1_status:
    updated = stage1_status.get("updated_at_utc", "unknown")
    try:
        from datetime import datetime, timezone
        if not isinstance(updated, str):
            raise ValueError("heartbeat timestamp is not text")
        heartbeat = datetime.fromisoformat(updated.replace("Z", "+00:00"))
        fresh = (datetime.now(timezone.utc) - heartbeat).total_seconds() < 30
    except (TypeError, ValueError):
        fresh = False
    recorded_state = str(stage1_status.get("state", "unknown")).lower()
    state = "error" if recorded_state == "error" else recorded_state if fresh else "stale"
    tone = "healthy" if state == "running" else "failed" if state == "error" else "stale" if state == "stale" else "offline"
    st.markdown(status_pill(f"Stage 1 process state: {state}", tone), unsafe_allow_html=True)
    st.caption(f"Last heartbeat: {updated} · Model: {stage1_status.get('model_version', 'unknown')} · Topic: {stage1_status.get('topic', 'unknown')} · freshness window: 30 seconds.")
    if recorded_state == "error":
        st.error("The last recorded Stage 1 state is error. Error details are omitted here because the saved exception contains runtime diagnostics.")
else:
    st.markdown(status_pill("Stage 1: no status record", "neutral"), unsafe_allow_html=True)
if stage1_rows:
    st.caption(f"Showing {len(stage1_rows)} most recent saved Stage 1 prediction records (maximum 500). Their presence alone does not mean the stream is currently running.")
    st.dataframe(pd.DataFrame(stage1_rows), use_container_width=True, hide_index=True)
else:
    st.info("No Stage 1 prediction records are available yet.")
registry = prediction_registry_summary()
if registry.get("available"):
    st.metric("Predictions registered in Stage 2", f"{registry['total']:,}")
    st.caption(f"Stage 2 registry range: {registry.get('first_recorded_at') or '—'} → {registry.get('last_recorded_at') or '—'} · this is a persisted count, not a throughput rate.")
else:
    st.info(registry.get("error", "Prediction registry unavailable."))
render_section_heading("Stage 2 delayed labels", "Stored predictions and label dispositions. Counts are not model accuracy metrics.")
stage2 = load_stage2_label_stats()
if stage2.get("store_available"):
    metrics = [
        ("Predictions registered", "predictions_registered"),
        ("Pending labels", "pending_labels"),
        ("Genuine verified labels", "confirmed_genuine_labels"),
        ("Synthetic test labels", "synthetic_test_labels"),
        ("Duplicate submissions", "duplicate_submissions"),
        ("Conflicting labels", "conflicting_labels"),
        ("Unmatched submissions", "unmatched_pending_submissions"),
    ]
    for start in range(0, len(metrics), 4):
        columns = st.columns(min(4, len(metrics) - start))
        for column, (label_text, key) in zip(columns, metrics[start:start + 4]):
            column.metric(label_text, stage2[key])
    if not stage2.get("genuine_source_configured"):
        st.info("Genuine labels unavailable: no authorized source is configured. Synthetic test labels are tracked separately.")
    elif not stage2.get("genuine_labels_available"):
        st.info("A trusted source is configured, but no genuine verified labels have been collected yet.")
else:
    st.info(stage2.get("error", "Stage 2 label store is not available."))
render_section_heading("Stage 3 drift and performance", "Separate feature-drift and verified-label signals; monitoring never trains or promotes a model.")
stage3 = load_stage3_monitor_report()
if stage3:
    drift, performance = stage3.get("drift", {}), stage3.get("performance", {})
    for error in stage3.get("monitoring_errors", []):
        st.error(error)
    st.write(f"Model {stage3.get('model_version', 'unknown')} · Threshold {stage3.get('threshold', 'unknown')} · Last run {stage3.get('generated_at', 'unknown')}")
    cols = st.columns(4)
    cols[0].metric("Monitor status", stage3.get("overall_status", "unknown"))
    cols[1].metric("Feature drift", stage3.get("drift_status", "unknown"))
    cols[2].metric("Eligible verified labels", performance.get("eligible_labeled_transactions", 0))
    cols[3].metric("Performance status", performance.get("status", "unknown"))
    if drift.get("drifted_features"):
        st.write("Drifted features: " + ", ".join(drift["drifted_features"]))
    if performance.get("metrics"):
        metric = performance["metrics"]
        st.write(f"Precision {metric['precision']:.3f} · Recall {metric['recall']:.3f} · F1 {metric['f1']:.3f} · TP {metric['tp']} / FP {metric['fp']} / TN {metric['tn']} / FN {metric['fn']}")
    elif performance.get("label_basis") == "unavailable":
        st.info("Performance is insufficient: no authorized verified label source is configured. Synthetic labels are excluded.")
    else:
        st.info(performance.get("reason", "Insufficient eligible labels for performance evaluation."))
    for alert in stage3.get("alerts", []):
        st.warning(f"{alert['type']}: {alert.get('window_start')} to {alert.get('window_end')}")
    request = stage3.get("retraining_request", {})
    st.caption(f"Retraining decision: {request.get('status', 'unknown')} · {request.get('reason', '')}. This records a request only.")
else:
    st.info("No Stage 3 report yet. Run `python -m src.monitoring.cli` to evaluate and persist a report.")
stage1_errors = read_project_jsonl(PROJECT_ROOT, "data/stage1_streaming/errors.jsonl")
with st.expander(f"Stage 1 processing errors ({len(stage1_errors)} most recent)"):
    if stage1_errors:
        st.dataframe(pd.DataFrame(stage1_errors), use_container_width=True, hide_index=True)
    else:
        st.caption("No persisted Stage 1 errors.")
report = load_latest_phase12_evidence(PROJECT_ROOT)
if not report:
    st.info("No readable Phase 12 integration evidence report is available.")
else:
    passed = report.get("status") == "PASS"
    st.markdown(status_pill("Historical test pass" if passed else "Historical test failure", "historical" if passed else "offline"), unsafe_allow_html=True)
    st.write(f"Evidence file: `{report.get('_report_path', 'unknown')}`")
    st.write(f"Completed: {report.get('completed_at_utc', 'timestamp unavailable')} · Model: {report.get('model_version', 'V2')} · Threshold: {report.get('threshold', 'see checks')}")
    summary = report.get("summary") or {}
    if summary:
        st.caption(f"Recorded test summary: {summary.get('passed', '—')} passed, {summary.get('failed', '—')} failed, {summary.get('total', '—')} total.")
    checks = report.get("checks", report.get("evidence_checks", {}))
    render_section_heading("Recorded checks", "These values are persisted test evidence, not a live status poll.")
    if isinstance(checks, dict):
        rows = []
        for name, item in checks.items():
            status = item.get("status", "UNKNOWN") if isinstance(item, dict) else item
            rows.append({"Check": name, "Recorded status": status})
        if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    render_section_heading("Processing output", "Read-only CSV files written by prior bounded Phase 12 integration runs.")
    rel = report.get("result_file")
    candidates = [rel] if isinstance(rel, str) else []
    output_check = checks.get("output") if isinstance(checks, dict) else None
    if isinstance(output_check, dict) and isinstance(output_check.get("path"), str):
        candidates.insert(0, output_check["path"])
    rows = []
    for path in candidates:
        if not path: continue
        from services.data_service import project_relative_path
        relative = project_relative_path(PROJECT_ROOT, path.replace("\\", "/"))
        if relative:
            rows.extend(read_project_csv(PROJECT_ROOT, relative))
    if not rows:
        rows = read_project_csv(PROJECT_ROOT, "reports/phase12_kafka_test_predictions.csv")
    if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else: st.info("The referenced output CSV is unavailable or empty.")
    st.caption("Pipeline stages: transaction publish → Kafka topic → Spark Structured Streaming → V2 inference → CSV output/checkpoint. This stored evidence does not mean a broker is currently online.")
st.info("Broker connectivity: Not checked. This dashboard does not start a Kafka consumer, Spark query, or streaming job.")
render_footer()
