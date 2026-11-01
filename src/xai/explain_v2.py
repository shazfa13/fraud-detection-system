"""Standalone SHAP explanations for the frozen Model V2 artifact.

Run from the repository root with ``python -m src.xai.explain_v2``. SHAP values
are explained on XGBoost's raw-margin (log-odds) scale; they describe model
behavior relative to the TreeExplainer baseline and do not establish causality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "xgboost_model_v2.json"
METADATA_PATH = ROOT / "models" / "model_v2_metadata.json"
THRESHOLD_PATH = ROOT / "models" / "model_v2_threshold.json"
DATASET_PATH = ROOT / "data" / "processed" / "v2_retraining_dataset.parquet"
OUTPUT_DIR = ROOT / "reports" / "xai_v2"
AUDIT_COLUMNS = {"event_id", "step", "isFraud"}


def validate_feature_contract(feature_names: list[str], columns: list[str],
                              model_feature_names: list[str] | None = None) -> list[str]:
    """Validate and return V2's exact ordered feature list against model and data."""
    if len(feature_names) != 34 or len(set(feature_names)) != 34:
        raise ValueError("V2 metadata must contain exactly 34 unique feature names.")
    forbidden = AUDIT_COLUMNS.intersection(feature_names)
    if forbidden:
        raise ValueError(f"Audit/target columns cannot be model features: {sorted(forbidden)}")
    missing = [name for name in feature_names if name not in columns]
    if missing:
        raise ValueError(f"Dataset is missing required V2 feature columns: {missing}")
    dataset_order = [name for name in columns if name in set(feature_names)]
    if dataset_order != feature_names:
        raise ValueError("Dataset V2 feature columns are reordered relative to the model contract.")
    if model_feature_names is not None and model_feature_names != feature_names:
        raise ValueError("Loaded model feature names/order do not match V2 metadata.")
    return list(feature_names)


def _prepare_feature_frame(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return ordered numeric model inputs, retaining meaningful missing values as NaN.

    The V2 dataset uses nulls for history-dependent values that do not exist yet. XGBoost
    natively handles these as missing values; coercing them to zero would change their meaning.
    Invalid non-null text and infinities are rejected with the offending feature name.
    """
    if list(frame.columns) != features:
        raise ValueError("Model input columns must exactly match the ordered V2 feature contract.")
    validate_feature_contract(features, list(frame.columns), features)
    numeric = frame.copy()
    for name in features:
        original = numeric[name]
        try:
            converted = pd.to_numeric(original, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"V2 feature {name!r} contains a non-numeric value.") from exc
        if (original.notna() & converted.isna()).any():
            raise ValueError(f"V2 feature {name!r} contains a non-numeric value that became missing.")
        values = converted.to_numpy(dtype=np.float64, na_value=np.nan)
        if np.isinf(values).any():
            raise ValueError(f"V2 feature {name!r} contains an infinite value.")
        # Keep float dtype for nullable history-derived fields; NaN is the trained model's
        # missing-value representation and is intentionally not imputed.
        numeric[name] = converted.astype(np.float64)
    return numeric.loc[:, features]


def _read_config() -> tuple[dict, list[str], float]:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    threshold_info = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    features = metadata.get("feature_names")
    threshold = threshold_info.get("selected_threshold")
    if (metadata.get("model_version", "").upper() != "V2"
            or metadata.get("feature_count") != 34 or not isinstance(features, list)):
        raise ValueError("V2 metadata is missing the expected version or 34-feature contract.")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("V2 threshold must be a number in [0, 1].")
    return metadata, validate_feature_contract(features, features), float(threshold)


def _load_model(features: list[str]) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    names = model.get_booster().feature_names
    validate_feature_contract(features, features, list(names) if names is not None else None)
    if model.n_features_in_ != len(features):
        raise ValueError(f"Loaded V2 model expects {model.n_features_in_} features, not 34.")
    return model


def _parquet_file():
    import pyarrow.parquet as pq
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"V2 dataset not found: {DATASET_PATH}")
    return pq.ParquetFile(DATASET_PATH)


def _read_positions(pf: Any, features: list[str], positions: np.ndarray) -> pd.DataFrame:
    """Read only requested rows and feature columns, loading just touched row groups."""
    names = pf.schema.names
    validate_feature_contract(features, names)
    offsets = np.cumsum([0, *(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))])
    pieces: list[pd.DataFrame] = []
    for group in np.unique(np.searchsorted(offsets[1:], positions, side="right")):
        group_positions = positions[np.searchsorted(offsets[1:], positions, side="right") == group]
        table = pf.read_row_group(int(group), columns=features)
        local = group_positions - offsets[group]
        pieces.append(table.take(local.tolist()).to_pandas())
    indexed = {int(pos): row for group, frame in zip(
        np.unique(np.searchsorted(offsets[1:], positions, side="right")), pieces)
               for pos, row in zip(positions[np.searchsorted(offsets[1:], positions, side="right") == group],
                                   frame.to_dict("records"))}
    frame = pd.DataFrame([indexed[int(pos)] for pos in positions], columns=features)
    return _prepare_feature_frame(frame, features)


def sample_features(pf: Any, features: list[str], sample_size: int, seed: int) -> pd.DataFrame:
    n = pf.metadata.num_rows
    if sample_size < 1 or sample_size > n:
        raise ValueError(f"sample-size must be between 1 and {n:,} dataset rows.")
    positions = np.sort(np.random.default_rng(seed).choice(n, size=sample_size, replace=False))
    return _read_positions(pf, features, positions)


def select_local_row(pf: Any, features: list[str], row_index: int | None,
                     transaction_id: str | None) -> tuple[pd.DataFrame, dict]:
    n = pf.metadata.num_rows
    if (row_index is None) == (transaction_id is None):
        raise ValueError("Select exactly one of --row-index or --transaction-id.")
    if row_index is not None:
        if row_index < 0 or row_index >= n:
            raise ValueError(f"row-index {row_index} is out of range (0 to {n - 1}).")
        group = int(np.searchsorted(np.cumsum([pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]), row_index, side="right"))
        start = sum(pf.metadata.row_group(i).num_rows for i in range(group))
        table = pf.read_row_group(group, columns=[*features, "event_id", "isFraud"])
        row = table.slice(row_index - start, 1).to_pydict()
        ix = int(row_index)
        selected_id = row["event_id"][0]
        label = row["isFraud"][0]
        x = _prepare_feature_frame(pd.DataFrame([{f: row[f][0] for f in features}], columns=features), features)
        return x, {"row_index": ix, "transaction_id": selected_id, "ground_truth_isFraud": label}
    # IDs are audited one row group at a time; feature data is read only for a match.
    for group in range(pf.num_row_groups):
        ids = pf.read_row_group(group, columns=["event_id"]).column("event_id").to_pylist()
        matches = [i for i, value in enumerate(ids) if str(value) == str(transaction_id)]
        if len(matches) > 1:
            raise ValueError(f"Transaction ID {transaction_id!r} is duplicated in the dataset.")
        if matches:
            i = matches[0]
            table = pf.read_row_group(group, columns=[*features, "event_id", "isFraud"]).slice(i, 1).to_pydict()
            ix = sum(pf.metadata.row_group(j).num_rows for j in range(group)) + i
            x = _prepare_feature_frame(pd.DataFrame([{f: table[f][0] for f in features}], columns=features), features)
            return x, {"row_index": ix, "transaction_id": table["event_id"][0],
                       "ground_truth_isFraud": table["isFraud"][0]}
    raise ValueError(f"Transaction ID {transaction_id!r} was not found in the dataset.")


def _shap_values(model: XGBClassifier, frame: pd.DataFrame):
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is required. Install project dependencies with: python -m pip install shap") from exc
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output="raw")
    explanation = explainer(frame)
    values = np.asarray(explanation.values)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.shape != frame.shape or not np.isfinite(values).all():
        raise ValueError(f"Unexpected SHAP output shape/values: {values.shape}; expected finite {frame.shape}.")
    return shap, explainer, explanation, values


def _global(model: XGBClassifier, frame: pd.DataFrame) -> list[str]:
    frame = _prepare_feature_frame(frame, list(frame.columns))
    shap, _, explanation, values = _shap_values(model, frame)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    importance = pd.DataFrame({"feature": frame.columns,
                               "mean_absolute_shap_value_raw_margin": np.abs(values).mean(axis=0)})
    importance.sort_values("mean_absolute_shap_value_raw_margin", ascending=False, kind="stable").to_csv(
        OUTPUT_DIR / "global_shap_importance.csv", index=False)
    import matplotlib.pyplot as plt
    plt.figure()
    shap.plots.bar(explanation, max_display=34, show=False)
    plt.title("V2 global feature importance (mean |SHAP|, raw margin)")
    plt.xlabel("Mean absolute SHAP value (raw margin / log-odds)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "global_shap_importance.png", dpi=160, bbox_inches="tight")
    plt.close()
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=34, show=False)
    plt.title("V2 global SHAP values (raw margin / log-odds)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "global_shap_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close()
    return ["global_shap_importance.csv", "global_shap_importance.png", "global_shap_beeswarm.png"]


def _local(model: XGBClassifier, frame: pd.DataFrame, audit: dict, threshold: float) -> tuple[list[str], dict]:
    frame = _prepare_feature_frame(frame, list(frame.columns))
    shap, _, explanation, values = _shap_values(model, frame)
    prediction_probability = float(model.predict_proba(frame)[:, 1][0])
    prediction_class = int(prediction_probability >= threshold)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    local = pd.DataFrame({"feature": frame.columns, "feature_value": frame.iloc[0].to_numpy(),
                          "shap_value_raw_margin": values[0]})
    local.to_csv(OUTPUT_DIR / "local_shap_explanation.csv", index=False)
    import matplotlib.pyplot as plt
    plt.figure()
    shap.plots.waterfall(explanation[0], max_display=34, show=False)
    plt.title("V2 transaction explanation (raw margin / log-odds)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "local_shap_waterfall.png", dpi=160, bbox_inches="tight")
    plt.close()
    details = {**audit, "predicted_fraud_probability": prediction_probability,
               "predicted_class_at_existing_threshold": prediction_class}
    return ["local_shap_explanation.csv", "local_shap_waterfall.png"], details


def run(mode: str = "both", sample_size: int = 1000, row_index: int | None = 0,
        transaction_id: str | None = None, seed: int = 42) -> dict:
    if mode not in {"global", "local", "both"}:
        raise ValueError("mode must be global, local, or both.")
    metadata, features, threshold = _read_config()
    model = _load_model(features)
    pf = _parquet_file()
    validate_feature_contract(features, pf.schema.names, model.get_booster().feature_names)
    outputs: list[str] = []
    sampled = 0
    audit = None
    prediction = None
    if mode in {"global", "both"}:
        global_frame = sample_features(pf, features, sample_size, seed)
        sampled = len(global_frame)
        outputs.extend(_global(model, global_frame))
    if mode in {"local", "both"}:
        local_frame, audit = select_local_row(pf, features, row_index, transaction_id)
        local_outputs, prediction = _local(model, local_frame, audit, threshold)
        outputs.extend(local_outputs)
    summary = {"model_version": metadata["model_version"], "model_artifact_path": str(MODEL_PATH.relative_to(ROOT)),
               "model_artifact_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
               "feature_count": len(features), "feature_order": features,
               "dataset_path": str(DATASET_PATH.relative_to(ROOT)), "rows_sampled": sampled,
               "random_seed": seed if mode in {"global", "both"} else None,
               "selected_transaction": audit, "classification_threshold": threshold,
               "prediction": prediction, "explanation_output_scale": "raw margin (log-odds)",
               "generated_output_paths": [str(OUTPUT_DIR / p) for p in [*outputs, "xai_run_summary.json"]],
               "limitations": ["SHAP contributions describe model output relative to a baseline; they do not prove causation."]}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "xai_run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Model {metadata['model_version']} loaded; threshold={threshold:.2f}; features={len(features)}")
    print(f"SHAP output scale: raw margin (log-odds). Outputs: {OUTPUT_DIR}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["global", "local", "both"], default="both")
    parser.add_argument("--sample-size", type=int, default=1000)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--row-index", type=int, help="Zero-based dataset row (default: 0).")
    selector.add_argument("--transaction-id", help="Match the dataset's unique event_id.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        row_index = args.row_index if args.transaction_id is not None or args.row_index is not None else 0
        run(args.mode, args.sample_size, row_index, args.transaction_id, args.seed)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
