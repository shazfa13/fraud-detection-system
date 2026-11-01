"""Local SHAP and counterfactual analysis for one stored transaction."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

from services.counterfactual_explainability import (
    DATASET_PATH, find_counterfactual, lookup_transaction, positive_class_shap,
)
from services.data_service import PROJECT_ROOT
from src.retraining.active_model import resolve_active_model
from ui.components import configure_page, render_hero, render_section_heading


configure_page("Counterfactual Explainability", "◉")
render_hero("Counterfactual Explainability", "Local model analysis",
            "Explore the active model’s contribution patterns and a bounded, hypothetical decision-flip search.")


@st.cache_resource(show_spinner="Loading active model…")
def load_model(model_path: str, model_mtime_ns: int, expected_features: tuple[str, ...]):
    del model_mtime_ns  # included in the cache key so artifact replacement refreshes the resource
    estimator = XGBClassifier()
    estimator.load_model(model_path)
    actual = estimator.get_booster().feature_names
    if estimator.n_features_in_ != len(expected_features) or (actual is not None and tuple(actual) != expected_features):
        raise ValueError("Loaded model feature names/order differ from the active 34-feature contract.")
    return estimator


@st.cache_resource(show_spinner=False)
def load_explainer(model_path: str, model_mtime_ns: int, expected_features: tuple[str, ...]):
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is unavailable in this Python environment. Install the project dependency `shap`.") from exc
    model = load_model(model_path, model_mtime_ns, expected_features)
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output="raw")
    return explainer


def explain_row(model, explainer, frame: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    explanation = explainer(frame)
    values, base = positive_class_shap(explanation, frame.shape)
    margin = float(np.asarray(model.predict(frame, output_margin=True)).reshape(-1)[0])
    if not np.isfinite(margin):
        raise ValueError("Active model returned a non-finite raw margin.")
    contributions = pd.DataFrame({"Feature": frame.columns, "Value": frame.iloc[0].to_numpy(),
                                  "SHAP contribution (raw margin)": values[0]})
    return contributions, base, margin


with st.form("counterfactual_lookup"):
    transaction_id = st.text_input("Transaction ID", placeholder="Enter an exact event_id from the processed dataset")
    submitted = st.form_submit_button("Analyze Transaction", type="primary")

if submitted:
    st.session_state.pop("counterfactual_analysis", None)
    try:
        bundle = resolve_active_model(project_root=PROJECT_ROOT)
        features = tuple(bundle.feature_names)
        model_path = Path(bundle.model_path)
        model_mtime_ns = model_path.stat().st_mtime_ns
        model = load_model(str(model_path), model_mtime_ns, features)
        frame, details = lookup_transaction(transaction_id, features)
        probability = float(model.predict_proba(frame)[0, 1])
        prediction = int(probability >= bundle.threshold)
        analysis = {"frame": frame, "details": details, "features": features,
                    "model": model, "model_version": bundle.model_version,
                    "threshold": bundle.threshold, "source": bundle.source,
                    "prediction": prediction, "probability": probability}
        try:
            explainer = load_explainer(str(model_path), model_mtime_ns, features)
            analysis["shap"] = explain_row(model, explainer, frame)
        except Exception as exc:
            analysis["shap_error"] = str(exc)
        try:
            config = json.loads((PROJECT_ROOT / "reports" / "feature_engineering_summary.json").read_text(encoding="utf-8"))
            params = config["frozen_parameters_fitted_on_training_period"]
            analysis["counterfactual"] = find_counterfactual(
                model, frame, features, bundle.threshold,
                large_threshold=float(params["large_amount_threshold"]),
                tolerance=float(params["tolerance"]), max_candidates=60)
        except Exception as exc:
            analysis["counterfactual_error"] = str(exc)
        st.session_state["counterfactual_analysis"] = analysis
    except LookupError as exc:
        st.error(str(exc))
    except (ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
        st.error(f"Could not analyze this transaction: {exc}")
    except Exception as exc:
        st.error(f"Analysis failed safely: {exc}")

analysis = st.session_state.get("counterfactual_analysis")
if analysis:
    detail, model = analysis["details"], analysis["model"]
    st.success(f"Found exact transaction **{detail['transaction_id']}** in the existing V2 processed Parquet dataset.")
    render_section_heading("Transaction details", "Stored model inputs from the matching event_id; the target label is excluded from this analysis.")
    st.dataframe(pd.DataFrame([{"Transaction ID": detail["transaction_id"], "Step": detail["step"]}]),
                 use_container_width=True, hide_index=True)
    with st.expander("Show the exact 34 model input features"):
        st.dataframe(analysis["frame"].T.rename(columns={0: "Feature value"}), use_container_width=True)
    render_section_heading("Current model decision", "Freshly generated with the shared active model bundle. This is not a stored historical prediction.")
    columns = st.columns(4)
    columns[0].metric("Transaction ID", str(detail["transaction_id"]))
    columns[1].metric("Predicted class", "Fraud" if analysis["prediction"] else "Legitimate")
    columns[2].metric("Fraud probability", f"{analysis['probability']:.2%}")
    columns[3].metric("Active threshold", f"{analysis['threshold']:.2%}")
    st.caption(f"Shared model resolver: {analysis['model_version']} · {analysis['source']} · exact 34-feature ordering.")
    st.info("Prediction history files contain output summaries rather than the raw 34 features. This result is freshly computed from the exact matching transaction row.")

    render_section_heading("Why was this transaction classified this way?", "SHAP contributions explain how input features move the model’s raw score relative to its baseline; they are not proof of cause or criminal activity.")
    if "shap" not in analysis:
        st.warning(f"SHAP explanation is unavailable for this row: {analysis.get('shap_error', 'unsupported output')} The prediction and counterfactual section remain available.")
    else:
        contributions, base, margin = analysis["shap"]
        st.caption(f"Raw-margin relationship: baseline {base:.4f} + feature contributions = model output {margin:.4f}. Positive contributions push toward fraud; negative contributions push away.")
        ranked = contributions.assign(Absolute=lambda frame: frame["SHAP contribution (raw margin)"].abs()).sort_values("Absolute", ascending=False).head(12).sort_values("SHAP contribution (raw margin)")
        fig, ax = plt.subplots(figsize=(10, 5.5))
        colors = ["#f07178" if value > 0 else "#43c6ac" for value in ranked["SHAP contribution (raw margin)"]]
        ax.barh(ranked["Feature"], ranked["SHAP contribution (raw margin)"], color=colors)
        ax.axvline(0, color="#9aa4b2", linewidth=.8)
        ax.set_xlabel("SHAP contribution on model raw-margin scale")
        ax.set_title("Most influential features for this transaction")
        ax.grid(axis="x", alpha=.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        toward = contributions[contributions["SHAP contribution (raw margin)"] > 0].nlargest(5, "SHAP contribution (raw margin)")
        away = contributions[contributions["SHAP contribution (raw margin)"] < 0].nsmallest(5, "SHAP contribution (raw margin)")
        positive, negative = st.columns(2)
        with positive:
            st.markdown("**Pushing toward fraud**")
            st.dataframe(toward[["Feature", "Value", "SHAP contribution (raw margin)"]], hide_index=True, use_container_width=True)
        with negative:
            st.markdown("**Pushing away from fraud**")
            st.dataframe(away[["Feature", "Value", "SHAP contribution (raw margin)"]], hide_index=True, use_container_width=True)
        st.caption("Viva explanation: each bar shows how one feature shifts this model score from its baseline. The chart describes model behavior, not what caused the real-world transaction.")

    render_section_heading("What minimal changes would change this prediction?", "A bounded search tests amount changes with consistent derived features and category changes; candidates are hypothetical model inputs.")
    if analysis.get("counterfactual_error"):
        st.warning(f"Counterfactual search could not complete: {analysis['counterfactual_error']}")
    elif not analysis.get("counterfactual"):
        st.info("No prediction-flipping candidate was found within the bounded search. The original transaction was not changed.")
    else:
        cf = analysis["counterfactual"]
        st.success("A minimal-change candidate was found by the bounded search and rechecked with the active model and threshold.")
        left, right = st.columns(2)
        left.metric("Original prediction", "Fraud" if analysis["prediction"] else "Legitimate", f"{analysis['probability']:.2%} probability")
        right.metric("Counterfactual prediction", "Fraud" if cf["prediction"] else "Legitimate", f"{cf['fraud_probability']:.2%} probability")
        st.caption(cf["description"] + " The search is approximate and does not guarantee a globally smallest change; alternatives may not represent a realistic real-world transaction.")
        differences = []
        for name in analysis["features"]:
            before, after = analysis["frame"].at[0, name], cf["features"].at[0, name]
            if not (pd.isna(before) and pd.isna(after)) and before != after:
                differences.append({"Feature": name, "Original": before, "Counterfactual": after})
        st.dataframe(pd.DataFrame(differences), use_container_width=True, hide_index=True)
        st.caption("This is a model sensitivity demonstration, not advice for evading fraud detection. No stored record was modified.")
else:
    st.info("Enter an exact transaction ID to begin. Lookup uses the existing processed V2 Parquet dataset and never substitutes another row.")

st.caption(f"Lookup source: {DATASET_PATH.relative_to(PROJECT_ROOT)} · Dataset labels and IDs are never included in model inputs.")
