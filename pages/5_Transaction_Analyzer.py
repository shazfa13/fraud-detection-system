"""Explicit, API-backed transaction preview and prediction."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from services.api_client import APIClientError, PREDICTION_FIELDS, get_health, predict_transaction, preview_features
from services.data_service import PROJECT_ROOT, load_active_model_snapshot, load_transaction_types
from ui.components import configure_page, render_footer, render_hero, render_section_heading

configure_page("Transaction Analyzer", "⌕")
render_hero("Transaction Analyzer", "Existing FastAPI inference",
            "Enter the API's seven raw transaction-time fields. The backend derives the ordered 34-feature vector using its shared feature builder.")
active = load_active_model_snapshot(PROJECT_ROOT)
if active.get("available"):
    st.caption(f"Resolver: {active['model_version']} · threshold {active['threshold']:.2f} · {active['feature_count']} features. The API process must be reachable to preview or predict.")
else:
    st.warning(active.get("error", "The active model resolver could not be read."))
st.warning("Submitting a prediction calls /predict, updates the API's process-local account history, and writes to its configured prediction log. Preview only calls /state/preview and does not consume the transaction.")

types = load_transaction_types(PROJECT_ROOT)
with st.form("transaction_analyzer"):
    a, b = st.columns(2)
    with a:
        step_text = st.text_input("Step", placeholder="Required integer step")
        if types:
            tx_type = st.selectbox("Transaction type", types)
        else:
            tx_type = st.text_input("Transaction type", placeholder="Required transaction type")
        amount_text = st.text_input("Amount", placeholder="Required finite number")
        origin = st.text_input("Origin account", placeholder="Required account identifier")
    with b:
        origin_balance_text = st.text_input("Origin balance before", placeholder="Required finite number")
        destination = st.text_input("Destination account", placeholder="Required account identifier")
        destination_balance_text = st.text_input("Destination balance before", placeholder="Required finite number")
    action = st.radio("Action", ("Preview only", "Submit prediction"), horizontal=True,
                      help="A prediction is sent only when Submit prediction is selected and this form is submitted.")
    submitted = st.form_submit_button("Run selected action", type="primary")

if submitted:
    st.session_state.pop("analyzer_preview", None)
    st.session_state.pop("analyzer_prediction", None)
    st.session_state.pop("analyzer_service_health", None)
    st.session_state.pop("analyzer_model_changed", None)
    try:
        payload = {"step": int(step_text), "type": tx_type, "amount": float(amount_text),
                   "nameOrig": origin, "oldbalanceOrg": float(origin_balance_text),
                   "nameDest": destination, "oldbalanceDest": float(destination_balance_text)}
        if tuple(payload) != PREDICTION_FIELDS:
            raise APIClientError("Transaction input order does not match the supported API schema.")
        if action == "Preview only":
            preview = preview_features(payload)
            if active.get("available") and preview["feature_names"] != active["feature_names"]:
                raise APIClientError("The API feature order differs from the local shared resolver contract; no result was stored.")
            st.session_state["analyzer_preview"] = preview
            st.session_state.pop("analyzer_prediction", None)
        else:
            preview = preview_features(payload)
            if active.get("available") and preview["feature_names"] != active["feature_names"]:
                raise APIClientError("The API feature order differs from the local shared resolver contract; prediction was not submitted.")
            service_health = get_health()
            result = predict_transaction(payload)
            st.session_state["analyzer_preview"] = preview
            st.session_state["analyzer_prediction"] = result
            st.session_state["analyzer_service_health"] = service_health
            st.session_state["analyzer_model_changed"] = result["model_version"] != service_health["model_version"]
    except (ValueError, APIClientError) as exc:
        st.error(f"Input or API request failed: {exc}")

preview = st.session_state.get("analyzer_preview")
if preview:
    render_section_heading("API-generated model inputs", "Exact feature order returned by /state/preview; step and target labels are not model features.")
    st.dataframe(pd.DataFrame([{name: preview["features"][name] for name in preview["feature_names"]}]), use_container_width=True, hide_index=True)
    st.caption("Feature schema was checked against the API response (34 names and matching ordered values).")

prediction = st.session_state.get("analyzer_prediction")
if prediction:
    render_section_heading("Prediction response", "Returned by the existing /predict route; this is a model output, not verified ground truth.")
    cols = st.columns(5)
    for col, label, value in zip(cols, ("Predicted class", "Fraud probability", "Risk score", "Risk level", "Model"),
                                 (prediction["prediction"], f"{float(prediction['fraud_probability']):.2%}", f"{float(prediction['risk_score']):.2f}", prediction["risk_level"], prediction["model_version"])):
        col.metric(label, value)
    result_health = st.session_state.get("analyzer_service_health") or {}
    if st.session_state.get("analyzer_model_changed"):
        st.warning("The API model version changed between its health check and prediction. The response is preserved, but the earlier threshold is not attributed to this prediction.")
        threshold_label = "unavailable due to API model change"
    else:
        threshold_label = result_health.get("threshold", "unavailable")
    st.caption(f"Threshold reported by the API /health check immediately before prediction: {threshold_label} · API request fields: {', '.join(PREDICTION_FIELDS)}")

with st.expander("Input and feature contract"):
    st.write("Transaction-time request fields:", list(PREDICTION_FIELDS))
    st.write("The server builds the model features in metadata order. `step` is required context for time/history calculations; `isFraud` and `event_id` are not supplied as model inputs.")
    st.write("No batch input is enabled; every prediction is an explicit single-record request.")
render_footer()
