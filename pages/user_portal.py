"""Simple, role-protected transaction scoring portal."""
from __future__ import annotations

from uuid import uuid4

import pandas as pd
import streamlit as st

from services.api_client import APIClientError, analyze_transaction, get_health
from services.data_service import PROJECT_ROOT, load_transaction_types
from services.user_explanation import explain_preview_safely
from ui.components import configure_page, render_footer, render_hero, render_section_heading


configure_page("Transaction Risk Check", "◈", required_role="user")
render_hero(
    "Transaction risk check",
    "Simple transaction analysis",
    "Enter the transaction details below to see the active model's estimate and the signals behind it.",
)

types = load_transaction_types(PROJECT_ROOT)
if not types:
    st.error("The supported transaction categories could not be loaded from the project configuration.")
    st.stop()

with st.form("user_transaction_form"):
    st.markdown("#### Transaction Details")
    st.caption("Provide the transaction time, type and amount. The transaction type options come from the model's saved configuration.")
    time_col, type_col, amount_col = st.columns(3, gap="large")
    with time_col:
        step = st.number_input(
            "Transaction time step", min_value=1, value=None, step=1, key="user_step",
            help="The dataset represents each step as one simulated hour. This helps the API calculate the transaction's time-related features.",
        )
    with type_col:
        tx_type = st.selectbox(
            "Transaction type", types, index=None, placeholder="Select a transaction type", key="user_type",
            help="Choose the kind of transaction being made. Only categories supported by the existing model are shown.",
        )
    with amount_col:
        amount = st.number_input(
            "Transaction amount", min_value=0.0, value=None, step=10.0, format="%.2f", key="user_amount",
            help="Enter the amount involved in this transaction. Unusual amounts are still accepted for the model to assess.",
        )

    st.markdown("#### Sender Information")
    st.caption("The account reference lets the API look up earlier transactions supplied to this system. Use synthetic IDs for this demonstration; the API cannot access bank history automatically.")
    sender_col, sender_balance_col = st.columns(2, gap="large")
    with sender_col:
        sender = st.text_input(
            "Sender account reference", placeholder="For example, C123…", key="user_sender",
            help="Used to find earlier sender activity stored by this API. Use synthetic IDs for this demo.",
        )
    with sender_balance_col:
        sender_balance = st.number_input(
            "Sender's balance before this transaction", min_value=0.0, value=None,
            step=50.0, format="%.2f", key="user_sender_balance",
            help="Enter the sender's account balance before this transaction. Enter 0 explicitly if the balance is zero.",
        )

    st.markdown("#### Receiver Information")
    st.caption("The receiver's reference is used for the same purpose: finding earlier activity already stored by this system.")
    destination_col, destination_balance_col = st.columns(2, gap="large")
    with destination_col:
        destination = st.text_input(
            "Receiver account reference", placeholder="For example, C987…", key="user_destination",
            help="Used to find earlier receiver activity stored by this API. Use synthetic IDs for this demo.",
        )
    with destination_balance_col:
        destination_balance = st.number_input(
            "Receiver's balance before this transaction", min_value=0.0, value=None,
            step=50.0, format="%.2f", key="user_destination_balance",
            help="Enter the receiver's balance before this transaction. Enter 0 explicitly if the balance is zero.",
        )

    st.caption("Some transaction patterns and balance-related values are calculated automatically from the information you provide. You don't need to enter these technical values yourself.")
    st.caption("Account-history features use earlier transactions stored by this API, including events explicitly supplied through its history warmup route. The current transaction is analyzed before it is added to that history.")
    submitted = st.form_submit_button("Analyze Transaction", type="primary", use_container_width=True)

if submitted:
    st.session_state.pop("user_result", None)
    st.session_state.pop("user_explanation", None)
    st.session_state.pop("user_explanation_error", None)
    missing = []
    if step is None:
        missing.append("transaction time step")
    if tx_type is None:
        missing.append("transaction type")
    if amount is None:
        missing.append("transaction amount")
    if sender_balance is None:
        missing.append("sender's balance before the transaction")
    if destination_balance is None:
        missing.append("receiver's balance before the transaction")
    if not sender.strip():
        missing.append("sender account reference")
    if not destination.strip():
        missing.append("receiver account reference")
    if missing:
        st.error("Enter the required transaction details: " + ", ".join(missing) + ".")
    else:
        payload = {
            "step": int(step), "type": tx_type, "amount": float(amount), "nameOrig": sender.strip(),
            "oldbalanceOrg": float(sender_balance), "nameDest": destination.strip(),
            "oldbalanceDest": float(destination_balance),
        }
        try:
            health_before = get_health()
            pending = st.session_state.get("user_analysis_pending")
            idempotency_key = pending["key"] if pending and pending["payload"] == payload else str(uuid4())
            st.session_state["user_analysis_pending"] = {"key": idempotency_key, "payload": payload}
            analysis = analyze_transaction(payload, idempotency_key=idempotency_key)
            health_after = get_health()
            if analysis["feature_names"] != list(analysis["features"]):
                raise APIClientError("The API analysis returned values outside the ordered feature contract.")
            if analysis["model_version"] != health_before["model_version"] or analysis["model_version"] != health_after["model_version"]:
                raise APIClientError("The active API model changed during analysis. Please submit the transaction again.")
            if float(health_before["threshold"]) != float(health_after["threshold"]):
                raise APIClientError("The API threshold changed during analysis. Please submit the transaction again.")
            st.session_state["user_result"] = {
                "prediction": analysis["prediction"], "probability": analysis["fraud_probability"],
                "risk_score": analysis["risk_score"], "risk_level": analysis["risk_level"],
                "threshold": health_after["threshold"], "model_version": analysis["model_version"],
                "history_status": analysis["history_status"],
            }
            st.session_state.pop("user_analysis_pending", None)
            explanation, explanation_error = explain_preview_safely(
                analysis["feature_names"], analysis["features"], analysis["model_version"],
                float(analysis["fraud_probability"]), int(analysis["prediction"]), float(health_after["threshold"]),
            )
            if explanation is not None:
                st.session_state["user_explanation"] = explanation
            if explanation_error:
                st.session_state["user_explanation_error"] = explanation_error
        except (APIClientError, ValueError) as exc:
            st.error(f"The transaction could not be analyzed: {exc}")

result = st.session_state.get("user_result")
if result:
    fraud = bool(result["prediction"])
    label = "Classified as potentially fraudulent" if fraud else "Classified as non-fraudulent"
    style = "result-fraud" if fraud else "result-clear"
    st.markdown(
        f"<div class='result-card {style}'><div class='hero-kicker'>Transaction analysis result</div>"
        f"<h2>{label}</h2><p>Estimated fraud probability</p><strong>{result['probability']:.2%}</strong></div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "The fraud probability represents how strongly the model considers this transaction to resemble fraudulent activity based on the information provided."
    )
    st.write(f"**Risk category:** {result['risk_level'].title()} (risk score {float(result['risk_score']):.2f} out of 100).")
    st.caption(
        f"The predicted classification uses the active {result['model_version']} model's configured threshold of {result['threshold']:.2%}. "
        "The risk category is a separate range-based summary, so it may not match the predicted classification."
    )
    history_status = result.get("history_status", {})
    cold_accounts = [role for role in ("sender", "destination")
                     if history_status.get(role, {}).get("state") == "no_prior_transactions"]
    if cold_accounts:
        labels = " and ".join("receiver" if role == "destination" else role for role in cold_accounts)
        st.warning(f"No earlier transactions are stored for this {labels} account reference. The API uses only stored history; it cannot tell whether the real account has no history or history that was never supplied.")
    st.info("This is a machine-learning estimate, not a definitive determination of fraud. A low estimated probability does not guarantee that a transaction is safe.")

    render_section_heading(
        "What influenced this estimate?",
        "The model highlights input patterns that moved its estimate up or down. These are model signals, not causes of real-world fraud.",
    )
    explanation = st.session_state.get("user_explanation")
    if isinstance(explanation, pd.DataFrame):
        st.dataframe(explanation, use_container_width=True, hide_index=True)
        st.caption("A signal marked as raising estimated risk moved the model toward a higher fraud estimate; one that lowers estimated risk moved it the other way.")
        with st.expander("Technical explanation details"):
            st.write("The contribution values are SHAP contributions on the model's raw log-odds scale. They are not probability percentage points or proof of a cause.")
    else:
        st.warning("A plain-language explanation could not be generated for this result. The prediction remains available. "
                   + st.session_state.get("user_explanation_error", "Explanation output was unavailable."))

with st.expander("Optional technical details"):
    st.write("**Original transaction fields sent to the API:** `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `nameDest`, and `oldbalanceDest`.")
    st.write("**Automatically prepared by the API:** time-of-day, amount and balance patterns, supported transaction-type encoding, and prior account-history summaries. The API returns the exact ordered feature vector used for analysis.")
    st.write("**Fields deliberately not requested:** post-transaction balances are not used by this model's feature pipeline; the API accepts them only for raw PaySim row compatibility. They were excluded by the project's leakage audit. `isFraud` is ground truth and is not accepted as a prediction input.")
    st.write("**Account history:** account references and the time step let the API retrieve earlier stored events. History not supplied to this system is not available to the model.")
    st.write("**Active contract:** 34 model features in metadata order.")
    from services.data_service import load_active_model_snapshot
    contract = load_active_model_snapshot(PROJECT_ROOT)
    if contract.get("available"):
        st.code("\n".join(contract["feature_names"]), language="text")
    else:
        st.caption("The local feature list is unavailable. The API still validates its response against the ordered 34-feature contract.")

render_footer()
