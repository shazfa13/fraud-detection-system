"""Read-only view of recent pipeline activity and saved operational evidence."""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from services.data_service import (
    PROJECT_ROOT,
    load_latest_phase12_evidence,
    load_stage3_monitor_report,
    read_project_jsonl,
    read_stage1_status,
    recent_stage1_predictions,
)
from ui.components import configure_page, render_footer, render_hero, render_section_heading, status_pill


DISPLAY_TIMEZONE = ZoneInfo("Asia/Kolkata")
configure_page("Pipeline Evidence", "⇢")
render_hero("Pipeline Evidence", "Recent processing activity",
            "Saved streaming-process status and transaction evidence. Broker reachability is not polled and this page does not start services.")

heartbeat = read_stage1_status(PROJECT_ROOT)
predictions = recent_stage1_predictions(limit=500)
now = datetime.now(timezone.utc)

render_section_heading("Streaming process status")
if heartbeat:
    updated = heartbeat.get("updated_at_utc")
    try:
        if not isinstance(updated, str):
            raise ValueError("missing timestamp")
        parsed = datetime.fromisoformat(updated.replace("Z", "+00:00"))
        parsed = parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
        age = (now - parsed).total_seconds()
        fresh = 0 <= age < 30
        display_updated = parsed.astimezone(DISPLAY_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z")
    except (TypeError, ValueError):
        fresh, display_updated = False, str(updated or "Unknown")
    recorded_state = str(heartbeat.get("state", "unknown")).lower()
    if recorded_state == "error":
        label, tone = "Error", "failed"
    elif fresh and recorded_state == "running":
        label, tone = "Running heartbeat", "healthy"
    elif fresh:
        label, tone = recorded_state.title(), "warning"
    else:
        label, tone = "Stale or unknown", "stale"
    st.markdown(status_pill(label, tone), unsafe_allow_html=True)
    st.caption(f"Last heartbeat: {display_updated} · model: {heartbeat.get('model_version', 'Unknown')} · topic: {heartbeat.get('topic', 'Unknown')}")
    st.caption("The process heartbeat is not a direct Kafka broker or Spark query health check.")
else:
    st.markdown(status_pill("Unavailable", "unavailable"), unsafe_allow_html=True)
    st.caption("No readable streaming-process heartbeat is available.")

render_section_heading("Recent transaction activity", "Last-five-minute count within the 500 most recent registry rows.")
if predictions is None:
    st.info("Prediction registry is unavailable; recent activity cannot be confirmed.")
else:
    recent = []
    for row in predictions:
        timestamp = pd.to_datetime(row.get("prediction_timestamp"), utc=True, errors="coerce")
        if pd.notna(timestamp) and pd.Timedelta(seconds=0) <= pd.Timestamp(now) - timestamp <= pd.Timedelta(minutes=5):
            recent.append(row)
    if recent:
        st.success(f"{len(recent)} of the latest 500 saved predictions were registered in the last five minutes.")
    else:
        st.info("No prediction in the latest 500 saved rows was registered in the last five minutes.")
    if predictions:
        display_rows = []
        for row in predictions[:10]:
            timestamp = pd.to_datetime(row.get("prediction_timestamp"), utc=True, errors="coerce")
            display_rows.append({
                "Transaction ID": row.get("transaction_id"),
                "Processed (IST)": timestamp.tz_convert(DISPLAY_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z") if pd.notna(timestamp) else "Unknown",
                "Model": row.get("model_version"),
                "Prediction": row.get("prediction"),
                "Risk": row.get("risk_level"),
            })
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

render_section_heading("Latest monitoring evidence")
monitor_report = load_stage3_monitor_report()
if monitor_report:
    cols = st.columns(3)
    cols[0].metric("Saved report status", str(monitor_report.get("overall_status", "Unknown")))
    cols[1].metric("Drift status", str(monitor_report.get("drift_status", "Unknown")))
    cols[2].metric("Report time", str(monitor_report.get("generated_at", "Unknown")))
    st.page_link("pages/7_Drift_Alerts.py", label="Review drift and alerts", icon=":material/notifications_active:")
else:
    st.info("No readable monitoring report is available.")

errors = read_project_jsonl(PROJECT_ROOT, "data/stage1_streaming/errors.jsonl")
with st.expander(f"Recent processing failures ({len(errors)})"):
    if errors:
        st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)
    else:
        st.caption("No persisted processing failures are recorded.")

render_section_heading("Saved integration evidence")
integration = load_latest_phase12_evidence(PROJECT_ROOT)
if integration:
    result = integration.get("status", "Unknown")
    st.markdown(status_pill(f"Historical integration result · {result}", "historical"), unsafe_allow_html=True)
    st.caption(f"Completed: {integration.get('completed_at_utc', 'Unknown')}")
    st.caption("This saved test result does not establish current broker or streaming-service availability.")
else:
    st.info("No saved integration evidence is available.")

st.caption("Detailed drift results, label provenance, and historical model metrics are available in their dedicated pages.")
render_footer()
