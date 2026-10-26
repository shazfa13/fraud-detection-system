"""Read-only Stage 3 drift and alert viewer."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from services.data_service import load_stage3_monitor_report
from ui.components import configure_page, render_footer, render_hero, render_section_heading, status_pill

configure_page("Drift & Alerts", "⌁")
render_hero("Drift & Alerts", "Persisted Stage 3 evidence",
            "Shows the latest saved monitor report and alert evidence. Viewing this page does not execute the monitor or write to its database.")
report = load_stage3_monitor_report()
if not report:
    st.info("No readable Stage 3 monitoring report is available.")
else:
    generated = report.get("generated_at")
    try:
        parsed = datetime.fromisoformat(str(generated).replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds()
        freshness, tone = ("Saved report · current (<24h)", "healthy") if age <= 86400 else ("Saved report · stale (>24h)", "stale")
    except (ValueError, TypeError):
        freshness, tone = "Saved report · timestamp unavailable", "unavailable"
    st.markdown(status_pill(freshness, tone), unsafe_allow_html=True)
    drift = report.get("drift") or {}
    reference = report.get("reference") or {}
    window = report.get("drift_window") or {}
    cols = st.columns(4)
    cols[0].metric("Model / threshold", f"{report.get('model_version', '—')} / {report.get('threshold', '—')}")
    cols[1].metric("Current records", str(window.get("records_read", window.get("record_count", "—"))))
    cols[2].metric("Reference sample", str(reference.get("reference_sample_count", reference.get("sample_count", "—"))))
    cols[3].metric("Drifted / tested", f"{drift.get('features_drifted', '—')} / {drift.get('features_tested', '—')}")
    st.caption(f"Generated: {generated or 'unknown'} · Overall status: {report.get('overall_status', 'unknown')} · Drift status: {report.get('drift_status', 'unknown')}")

    render_section_heading("Drift decision", "Values below are copied from the persisted monitor report, not recalculated by the dashboard.")
    cols = st.columns(3)
    cols[0].metric("Drift rate", f"{float(drift['drift_rate']):.3f}" if isinstance(drift.get("drift_rate"), (float, int)) else "—")
    cols[1].metric("Decision threshold", str(drift.get("drift_rate_threshold", (drift.get("configuration") or {}).get("overall_drift_rate_threshold", "—"))))
    cols[2].metric("Decision", "Drift detected" if drift.get("overall_drift_detected") else "No drift recorded")
    names = drift.get("drifted_features") or drift.get("strongest_drifted_features") or []
    if names:
        st.write("Drifted features recorded:", ", ".join(names))
    insuff = drift.get("features_insufficient_sample")
    if isinstance(insuff, (list, tuple)) and insuff:
        st.info(f"Features with insufficient reference/current samples: {', '.join(map(str, insuff))}")
    elif isinstance(insuff, int) and insuff > 0:
        st.info(f"{insuff} feature(s) had insufficient reference/current samples; the report does not persist their names.")
    st.caption(f"Detector: {drift.get('detector', 'not recorded')} · reason: {drift.get('reason', 'not recorded')}")
    details = drift.get("feature_results") or drift.get("per_feature") or drift.get("features")
    if isinstance(details, list) and details:
        render_section_heading("Per-feature detector output", "Only persisted fields are shown.")
        st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)
    else:
        st.info("This saved report does not contain per-feature PSI, raw test p-values, or adjusted p-values. It stores aggregate results and selected feature names only.")

    render_section_heading("Alerts and retraining request", "A request is an auditable signal for review; it does not start training or approve a model.")
    alerts = report.get("alerts") or []
    if alerts:
        rows = []
        for alert in alerts:
            evidence = alert.get("evidence") or {}
            rows.append({"Alert ID": alert.get("alert_id"), "Type": alert.get("type"),
                         "Window start": alert.get("window_start"), "Window end": alert.get("window_end"),
                         "Drift rate": evidence.get("drift_rate"), "Newly recorded": alert.get("newly_recorded")})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No alert is recorded in this report.")
    request = report.get("retraining_request") or {}
    st.write(f"Request status: **{request.get('status', 'not recorded')}** · reason: {request.get('reason', 'not recorded')}")
    st.caption("The alert and request belong to the saved window. Synthetic-window evidence is not production evidence.")
render_footer()
