"""Independent read-only API, resolver, artifact, and streaming signals."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from services.data_service import (
    PROJECT_ROOT,
    load_active_model_snapshot,
    load_latest_phase12_evidence,
    read_stage1_status,
    recent_stage1_predictions,
)
from src.retraining.active_model import resolve_active_model
from ui.components import cached_api_health, configure_page, render_footer, render_hero, render_section_heading, status_pill


configure_page("System Health", "⌁")
render_hero("System Health", "Independent service and evidence signals",
            "API, model resolution, saved artifacts, and streaming-process evidence are reported separately.")

render_section_heading("API health", "FastAPI health endpoint reachability and serving-process model state.")
api_ok, api_health, api_error = cached_api_health()
if api_ok and isinstance(api_health, dict):
    st.markdown(status_pill("Reachable", "healthy"), unsafe_allow_html=True)
    api_cols = st.columns(3)
    api_cols[0].metric("Model loaded in API", "Yes" if api_health.get("model_loaded") is True else "No")
    api_cols[1].metric("API model version", str(api_health.get("model_version", "Unknown")))
    api_cols[2].metric("Features / threshold", f"{api_health.get('feature_count', 'Unknown')} / {api_health.get('threshold', 'Unknown')}")
    st.caption("This response is cached for up to 15 seconds; the exact health-check time is not recorded.")
else:
    st.markdown(status_pill("Unavailable", "unavailable"), unsafe_allow_html=True)
    st.caption(api_error or "No health response from the configured API.")

render_section_heading("Model resolver", "The shared resolver selects the active bundle independently of API reachability.")
active = load_active_model_snapshot(PROJECT_ROOT)
if active.get("available") is True:
    st.markdown(status_pill("Resolved", "healthy"), unsafe_allow_html=True)
    st.write(f"Selected model: **{active.get('model_version', 'Unknown')}** · source: {active.get('source', 'Unknown')}")
    st.caption(f"Artifact path: {active.get('model_path') or 'Unavailable'} · threshold: {active.get('threshold', 'Unknown')} · features: {active.get('feature_count', 'Unknown')}")
else:
    st.markdown(status_pill("Unavailable", "unavailable"), unsafe_allow_html=True)
    st.warning(active.get("error", "The active model could not be resolved."))

render_section_heading("Artifact availability", "The selected bundle's model, metadata and threshold paths are checked independently.")
try:
    bundle = resolve_active_model(project_root=PROJECT_ROOT)
    artifact_rows = []
    for label, path in (("Active model", bundle.model_path), ("Active metadata", bundle.metadata_path),
                        ("Active threshold", bundle.threshold_path)):
        relative = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        artifact_rows.append({"Artifact": label, "Path": relative, "Status": "Present" if path.is_file() else "Missing"})
    st.dataframe(pd.DataFrame(artifact_rows), use_container_width=True, hide_index=True)
    st.caption("The shared resolver validated access to this bundle. The API model-loaded signal above is a separate serving-process check.")
except Exception as exc:
    st.warning(f"Selected model artifact paths are unavailable: {type(exc).__name__}: {exc}")

render_section_heading("Streaming health", "Saved worker heartbeat and transaction activity; broker/query state is not directly probed.")
stream_status = read_stage1_status(PROJECT_ROOT)
predictions = recent_stage1_predictions(limit=500)
left, right = st.columns(2)
with left:
    st.markdown("**Streaming worker heartbeat**")
    if stream_status:
        updated = stream_status.get("updated_at_utc")
        try:
            parsed = datetime.fromisoformat(str(updated).replace("Z", "+00:00"))
            parsed = parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
            fresh = 0 <= (datetime.now(timezone.utc) - parsed).total_seconds() < 30
        except (TypeError, ValueError):
            fresh = False
        state = str(stream_status.get("state", "unknown")).lower()
        label = "Error" if state == "error" else "Running heartbeat" if fresh and state == "running" else "Stale or unknown"
        tone = "failed" if state == "error" else "healthy" if fresh and state == "running" else "stale"
        st.markdown(status_pill(label, tone), unsafe_allow_html=True)
        st.caption(f"Last saved heartbeat: {updated or 'Unknown'} · process state: {state}")
    else:
        st.markdown(status_pill("Unavailable", "unavailable"), unsafe_allow_html=True)
        st.caption("No readable worker heartbeat is available.")
    st.caption("Kafka broker: Not checked · Spark query: Not directly checked")
with right:
    st.markdown("**Recent transaction activity**")
    if predictions is None:
        st.markdown(status_pill("Unavailable", "unavailable"), unsafe_allow_html=True)
        st.caption("Prediction registry could not be read.")
    else:
        now = pd.Timestamp.now(tz="UTC")
        count = 0
        latest = None
        for row in predictions:
            timestamp = pd.to_datetime(row.get("prediction_timestamp"), utc=True, errors="coerce")
            if pd.isna(timestamp):
                continue
            latest = timestamp if latest is None or timestamp > latest else latest
            if pd.Timedelta(seconds=0) <= now - timestamp <= pd.Timedelta(minutes=5):
                count += 1
        st.metric("Of latest 500 · last five minutes", count)
        st.caption(f"Latest saved prediction: {latest.isoformat() if latest is not None else 'Unknown'}")
        st.caption("Activity comes from the prediction registry; it is not a broker health check.")

evidence = load_latest_phase12_evidence(PROJECT_ROOT)
if evidence:
    st.caption(f"Saved integration evidence: {evidence.get('status', 'Unknown')} · completed {evidence.get('completed_at_utc', 'timestamp unavailable')} (historical, not a live service check).")
else:
    st.caption("Historical integration evidence is unavailable; current Kafka/Spark service health remains not checked.")

render_footer()
