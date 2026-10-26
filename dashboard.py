"""Read-only Fraud Intelligence overview and navigation entry point."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from services.data_service import (PROJECT_ROOT, load_active_model_snapshot,
                                   load_latest_phase12_evidence, load_prediction_history,
                                   prediction_registry_summary)
from ui.components import (cached_api_health, configure_page, render_hero,
                           render_section_heading, status_pill)

configure_page("Overview", "◈")
render_hero("Fraud Intelligence", "Adaptive transaction risk monitoring",
            "A read-only view of saved prediction records, the configured model bundle, and historical pipeline evidence.")

registry = prediction_registry_summary()
model = load_active_model_snapshot(PROJECT_ROOT)
api_ok, api_health, api_error = cached_api_health()
history = load_prediction_history(PROJECT_ROOT, limit=100)
evidence = load_latest_phase12_evidence(PROJECT_ROOT)

if registry.get("available"):
    total = registry["total"]
    fraud = registry["predicted_fraud"]
    legit = registry["predicted_legitimate"]
    time_range = f"{registry.get('first_recorded_at') or '—'} → {registry.get('last_recorded_at') or '—'}"
    columns = st.columns(4)
    columns[0].metric("Registered predictions", f"{total:,}")
    columns[1].metric("Predicted fraud", f"{fraud:,}")
    columns[2].metric("Predicted legitimate", f"{legit:,}")
    columns[3].metric("Predicted fraud share", f"{fraud / total:.2%}" if total else "—")
    st.caption(f"Source: Stage 2 prediction registry · {time_range}. Prediction classes are model outputs, not verified outcomes.")
else:
    st.info(registry.get("error", "Prediction registry is unavailable; counts are not inferred from another dataset."))

left, right = st.columns([1, 1.5], gap="large")
with left:
    render_section_heading("Model selection", "Read from the shared active-model resolver; the dashboard does not load the estimator.")
    if model.get("available"):
        st.metric("Resolved model", model["model_version"])
        st.caption(f"Threshold {model['threshold']:.2f} · {model['feature_count']} ordered features")
        st.caption(f"Resolver source: {model['source']}")
    else:
        st.warning(model.get("error", "Active model resolver is unavailable."))
    st.markdown(status_pill("FastAPI reachable" if api_ok else "FastAPI unavailable",
                            "healthy" if api_ok else "offline"), unsafe_allow_html=True)
    if api_ok and api_health:
        st.caption(f"Live /health response, briefly cached · API model {api_health.get('model_version')} · threshold {api_health.get('threshold')}")
    elif api_error:
        st.caption("API reachability does not indicate Kafka or Spark processing health.")

with right:
    render_section_heading("Recent recorded activity", "Newest rows from the existing API and streaming report files; capped at 100 records.")
    if history:
        frame = pd.DataFrame(history)
        cols = [name for name in ("timestamp", "transaction_id", "model_version", "prediction", "fraud_probability", "risk_level", "source") if name in frame]
        st.dataframe(frame[cols].head(8), use_container_width=True, hide_index=True)
        sources = sorted(frame["source"].dropna().unique()) if "source" in frame else []
        st.caption("Sources: " + (" · ".join(sources) if sources else "not recorded"))
    else:
        st.markdown("<div class='empty-state'><strong>No accessible prediction history</strong><br/>No prediction activity is displayed until records exist in the configured files or registry.</div>", unsafe_allow_html=True)

render_section_heading("Pipeline evidence", "Historical integration reports are separate from current service health.")
if evidence:
    passed = evidence.get("status") == "PASS"
    st.markdown(status_pill(f"Phase 12 historical evidence · {evidence.get('status')}", "historical" if passed else "warning"), unsafe_allow_html=True)
    st.caption(f"Completed {evidence.get('completed_at_utc', 'time unavailable')} · {evidence.get('_report_path', 'report path unavailable')}")
else:
    st.info("No readable Phase 12 integration report is available.")
st.markdown("<div class='architecture-strip'><span>FastAPI + XGBoost</span><b>→</b><span>Kafka + Spark</span><b>→</b><span>Prediction registry and monitoring</span><b>→</b><span>Human-reviewed lifecycle workflows</span></div>", unsafe_allow_html=True)
st.caption("This academic prototype uses synthetic PaySim data. Saved evaluation scores and demo candidates are not production evidence.")
