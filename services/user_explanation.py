"""Read-only local SHAP explanation for a feature row returned by API preview."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from services.counterfactual_explainability import positive_class_shap, prepare_features
from services.data_service import PROJECT_ROOT
from src.retraining.active_model import resolve_active_model


FEATURE_LABELS = {
    "hour_of_day": "Hour of day",
    "amount": "Transaction amount",
    "log_amount": "Transaction amount scale",
    "is_large_amount": "Large amount indicator",
    "amount_to_sender_balance_ratio": "Amount vs. sender balance",
    "amount_ge_sender_balance": "Amount compared with sender balance",
    "sender_remaining_if_debited": "Sender balance after amount (derived estimate)",
    "amount_to_dest_balance_ratio": "Amount vs. destination balance",
    "sender_balance_before": "Sender balance before transaction",
    "sender_balance_zero_before": "Sender balance availability",
    "dest_balance_before": "Destination balance before transaction",
    "dest_balance_zero_before": "Destination balance availability",
    "sender_prev_txn_count": "Sender's earlier transaction count",
    "sender_prev_total_amount": "Sender's earlier transaction total",
    "sender_prev_mean_amount": "Sender's earlier average amount",
    "sender_prev_max_amount": "Sender's earlier largest amount",
    "dest_prev_txn_count": "Destination's earlier transaction count",
    "dest_prev_total_amount": "Destination's earlier transaction total",
    "dest_prev_mean_amount": "Destination's earlier average amount",
    "dest_prev_max_amount": "Destination's earlier largest amount",
    "amount_vs_sender_hist_mean": "Amount vs. sender's earlier average",
    "amount_vs_dest_hist_mean": "Amount vs. destination's earlier average",
    "sender_hours_since_prev_txn": "Time since sender's earlier transaction",
    "dest_hours_since_prev_txn": "Time since destination's earlier transaction",
    "dest_txn_count_last_1h": "Destination transactions in the past hour",
    "dest_txn_count_last_24h": "Destination transactions in the past day",
    "dest_txn_count_last_168h": "Destination transactions in the past week",
    "dest_amount_sum_last_24h": "Destination amount total in the past day",
    "dest_amount_sum_last_168h": "Destination amount total in the past week",
    "type_CASH_IN": "Cash-in transaction type",
    "type_CASH_OUT": "Cash-out transaction type",
    "type_DEBIT": "Debit transaction type",
    "type_PAYMENT": "Payment transaction type",
    "type_TRANSFER": "Transfer transaction type",
}


@st.cache_resource(show_spinner=False)
def _load_model(model_path: str, model_mtime_ns: int, features: tuple[str, ...]):
    del model_mtime_ns
    from xgboost import XGBClassifier

    estimator = XGBClassifier()
    estimator.load_model(model_path)
    names = estimator.get_booster().feature_names
    if estimator.n_features_in_ != 34 or (names is not None and tuple(names) != features):
        raise ValueError("Active model does not match the ordered 34-feature input contract.")
    return estimator


@st.cache_resource(show_spinner="Preparing explanation…")
def _load_explainer(model_path: str, model_mtime_ns: int, features: tuple[str, ...]):
    import shap

    model = _load_model(model_path, model_mtime_ns, features)
    return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output="raw")


def explain_preview(feature_names: list[str], feature_values: dict, expected_model_version: str,
                    expected_probability: float, expected_prediction: int, expected_threshold: float) -> pd.DataFrame:
    """Return concise positive/negative feature contributions without persisting data."""
    bundle = resolve_active_model(project_root=PROJECT_ROOT)
    if bundle.model_version != expected_model_version:
        raise ValueError("The active model changed between the API preview and local explanation.")
    features = tuple(bundle.feature_names)
    if feature_names != list(features) or list(feature_values) != feature_names:
        raise ValueError("The preview does not match the active model's exact feature order.")
    frame = prepare_features(pd.DataFrame([{name: feature_values[name] for name in features}], columns=features), features)
    model_path = bundle.model_path
    model = _load_model(str(model_path), model_path.stat().st_mtime_ns, features)
    if abs(bundle.threshold - expected_threshold) > 1e-12:
        raise ValueError("The local active-model threshold differs from the API result threshold.")
    local_probability = float(model.predict_proba(frame)[0, 1])
    if abs(local_probability - expected_probability) > 1e-6:
        raise ValueError("The preview row does not reproduce the API probability; explanation was withheld.")
    if int(local_probability >= bundle.threshold) != expected_prediction:
        raise ValueError("The local active model's threshold decision differs from the API result.")
    explainer = _load_explainer(str(model_path), model_path.stat().st_mtime_ns, features)
    values, _ = positive_class_shap(explainer(frame), frame.shape)
    result = pd.DataFrame({
        "Feature": [FEATURE_LABELS.get(name, name) for name in features],
        "Contribution": values[0],
    })
    result = result.sort_values("Contribution", key=lambda column: column.abs(), ascending=False).head(6)
    result["Direction"] = np.select(
        [result["Contribution"] > 0, result["Contribution"] < 0],
        ["Raises estimated risk", "Lowers estimated risk"],
        default="No material contribution",
    )
    result["Contribution (log-odds)"] = result["Contribution"].map(lambda value: f"{value:+.4f}")
    return result[["Feature", "Direction", "Contribution (log-odds)"]].reset_index(drop=True)


def explain_preview_safely(feature_names: list[str], feature_values: dict, expected_model_version: str,
                           expected_probability: float, expected_prediction: int,
                           expected_threshold: float) -> tuple[pd.DataFrame | None, str | None]:
    """Keep explanation failures separate from an already valid prediction result."""
    try:
        return explain_preview(feature_names, feature_values, expected_model_version,
                               expected_probability, expected_prediction, expected_threshold), None
    except Exception as exc:
        return None, str(exc)
