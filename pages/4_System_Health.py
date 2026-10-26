"""Checks only supported API and local artifact availability."""
from __future__ import annotations

import streamlit as st

from services.data_service import PROJECT_ROOT, load_active_model_snapshot, local_artifact_status
from ui.components import cached_api_health, configure_page, render_footer, render_hero, render_section_heading, status_pill

configure_page("System Health", "⌁")
render_hero("System Health", "Service reachability and saved artifacts", "API reachability is polled separately from the Stage 1 heartbeat. Kafka and Spark are not probed or started here.")
ok, health, error = cached_api_health()
st.markdown(status_pill("FastAPI: Online" if ok else "FastAPI: Offline", "healthy" if ok else "offline"), unsafe_allow_html=True)
if ok and health:
    a, b, c = st.columns(3)
    a.metric("Model loaded", "Yes" if health.get("model_loaded") else "No")
    b.metric("API model version", health.get("model_version", "Unknown"))
    c.metric("Feature count / threshold", f"{health.get('feature_count')} / {health.get('threshold')}")
else: st.info(error or "No health response from configured API.")
active = load_active_model_snapshot(PROJECT_ROOT)
if active.get("available"):
    st.caption(f"Shared resolver selects {active['model_version']} at threshold {active['threshold']:.2f} from {active['source']}.")
else:
    st.warning(active.get("error", "Active model resolver could not be read."))
render_section_heading("Local model and lifecycle artifacts", "Presence checks only; model files are not loaded or modified.")
for item in local_artifact_status(PROJECT_ROOT)[:5]:
    if "files" in item:
        detail = "Empty" if item.get("empty") else ", ".join(item.get("files", [])) or "No directory"
    else:
        detail = "Available" if item["available"] else "Missing"
    st.write(f"**{item['label']}** · `{item['path']}` · {detail}")
st.markdown("### Architecture and scope")
st.write("FastAPI, Spark, and Stage 3 share the active-model resolver. A valid activation manifest is authoritative; if absent, the configured V1/V2 version defaults to V2.")
st.write("Kafka and Spark integration is represented by saved Phase 12 historical evidence. This page does not start services or claim they are currently healthy.")
st.info("Academic demonstration on synthetic PaySim data. Not a production banking service; genuine label provenance currently blocks operational retraining.")
render_footer()
