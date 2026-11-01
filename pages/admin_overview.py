"""Compact, read-only landing page for the administrator dashboard."""
from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from services.data_service import (
    PROJECT_ROOT,
    load_active_model_snapshot,
    prediction_registry_summary,
    recent_stage1_predictions,
)
from src.orchestrator.drift_adapter import get_latest_drift_summary
from ui.components import cached_api_health, configure_page, render_footer, render_hero, render_section_heading, status_pill


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _pill(label: str, tone: str) -> None:
    st.markdown(status_pill(label, tone), unsafe_allow_html=True)


configure_page("Admin Overview", "◈")
render_hero("Fraud Detection Dashboard", "Administrative overview",
            "A concise view of the active model, API reachability, recent predictions and saved monitoring alerts.")

model = load_active_model_snapshot(PROJECT_ROOT)
api_ok, api_health, _api_error = cached_api_health()
registry = prediction_registry_summary()
recent_rows = recent_stage1_predictions(limit=5)
monitoring = get_latest_drift_summary()

render_section_heading("System at a glance")
model_col, api_col, predictions_col, alerts_col = st.columns(4)

with model_col:
    with st.container(border=True):
        st.markdown("**Model availability**")
        if model.get("available") is True:
            _pill("Available", "healthy")
            st.metric("Active model", model.get("model_version", "Unknown"))
            st.caption(f"Resolver source: {model.get('source', 'Unavailable')}")
        else:
            _pill("Unavailable", "unavailable")
            st.caption(model.get("error", "The active model could not be resolved."))

with api_col:
    with st.container(border=True):
        st.markdown("**API status**")
        if api_ok and isinstance(api_health, dict):
            _pill("Healthy", "healthy")
            st.caption(f"FastAPI responded · model {api_health.get('model_version', 'Unknown')}")
        else:
            _pill("Unavailable", "unavailable")
            st.caption("FastAPI health response is unavailable.")
        st.caption("Health response is cached for up to 15 seconds; exact check time is not recorded.")

with predictions_col:
    with st.container(border=True):
        st.markdown("**Recent predictions**")
        if recent_rows is None:
            _pill("Unavailable", "unavailable")
            st.caption("The prediction registry could not be read.")
        elif not recent_rows and registry.get("available") is True and registry.get("total") == 0:
            _pill("No recent activity", "warning")
            st.metric("Processed in last 5 minutes", 0)
            st.caption("The registry contains no saved predictions.")
        elif not recent_rows:
            _pill("Unknown", "unavailable")
            st.metric("Processed in last 5 minutes", "Unknown")
            st.caption("No recent timestamped records are available to determine activity.")
        else:
            now = datetime.now(timezone.utc)
            parsed_rows = [(_parse_timestamp(row.get("prediction_timestamp") or row.get("registered_at")), row)
                           for row in recent_rows]
            latest = max((stamp for stamp, _ in parsed_rows if stamp is not None), default=None)
            recent_count = sum(stamp is not None and 0 <= (now - stamp).total_seconds() <= 300
                               for stamp, _ in parsed_rows)
            if latest is None:
                _pill("Unknown", "unavailable")
                st.metric("Processed in last 5 minutes", "Unknown")
                st.caption("No valid prediction timestamp is available.")
            else:
                if recent_count:
                    _pill("Recent activity", "healthy")
                else:
                    _pill("No recent activity", "warning")
                st.metric("Of latest five · last 5 minutes", recent_count)
                st.caption(f"Latest saved prediction: {latest.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}")
        if registry.get("available") is True:
            st.caption(f"Registered predictions total: {registry.get('total', 0):,}")
        else:
            st.caption("All-time registry total is unavailable.")

with alerts_col:
    with st.container(border=True):
        st.markdown("**Monitoring alerts**")
        if monitoring.get("status") != "SAVED_REPORT":
            _pill("Unavailable", "unavailable")
            st.caption("No readable saved monitoring report is available.")
        else:
            signals: list[str] = []
            if monitoring.get("drift_decision") is True:
                signals.append("Feature drift detected")
            warnings = monitoring.get("warnings", [])
            if isinstance(warnings, list):
                signals.extend(str(item) for item in warnings if str(item).strip())
            if signals:
                _pill("Warning", "warning")
                st.metric("Signals in latest report", len(signals))
                st.caption(signals[0])
            elif monitoring.get("drift_decision") is False:
                _pill("Healthy", "healthy")
                st.caption("No alert signals in the latest saved report.")
            else:
                _pill("Unknown", "unavailable")
                st.caption("The saved report does not contain a drift decision.")
            st.caption(f"Report time: {monitoring.get('report_timestamp') or 'Unavailable'}")

st.caption("Prediction labels are model outputs, not verified fraud outcomes. Monitoring status is based on the latest saved report.")
render_footer()
