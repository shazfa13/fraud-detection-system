"""Read-only lookup and bounded local explanations for the active fraud model."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from services.data_service import PROJECT_ROOT

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "v2_retraining_dataset.parquet"
AUDIT_COLUMNS = {"event_id", "step", "isFraud"}
TYPE_FEATURES = tuple(f"type_{name}" for name in ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"))


def prepare_features(frame: pd.DataFrame, feature_names: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Validate the ordered model contract and preserve nullable history values as NaN."""
    features = list(feature_names)
    if len(features) != 34 or len(set(features)) != 34 or AUDIT_COLUMNS.intersection(features):
        raise ValueError("Active model must use the frozen ordered 34-feature contract without audit fields.")
    if list(frame.columns) != features:
        raise ValueError("Transaction feature names or ordering do not match the active model contract.")
    numeric = frame.copy()
    for name in features:
        try:
            converted = pd.to_numeric(numeric[name], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature {name!r} is not numeric.") from exc
        values = converted.to_numpy(dtype=np.float64, na_value=np.nan)
        if np.isinf(values).any():
            raise ValueError(f"Feature {name!r} contains an infinite value.")
        numeric[name] = converted.astype(np.float64)
    return numeric.loc[:, features]


def lookup_transaction(transaction_id: str, feature_names: list[str] | tuple[str, ...],
                       dataset_path: str | Path = DATASET_PATH) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Find one exact event_id in the existing V2 feature dataset; never substitutes another row."""
    identifier = str(transaction_id).strip()
    if not identifier:
        raise ValueError("Enter a transaction ID.")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("PyArrow is required to read the existing transaction dataset.") from exc
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Transaction dataset is missing: {path}")
    parquet = pq.ParquetFile(path)
    names = parquet.schema.names
    missing = [name for name in ("event_id", "step", *feature_names) if name not in names]
    if missing:
        raise ValueError(f"Transaction dataset is missing required columns: {missing}")
    matches: list[tuple[int, int]] = []
    for group in range(parquet.num_row_groups):
        ids = parquet.read_row_group(group, columns=["event_id"]).column("event_id").to_pylist()
        matches.extend((group, offset) for offset, value in enumerate(ids) if str(value) == identifier)
        if len(matches) > 1:
            raise ValueError(f"Transaction ID {identifier!r} is duplicated; lookup is ambiguous.")
    if not matches:
        raise LookupError(f"Transaction ID {identifier!r} was not found in the existing dataset.")
    group, offset = matches[0]
    table = parquet.read_row_group(group, columns=["event_id", "step", *feature_names]).slice(offset, 1).to_pydict()
    values = {name: table[name][0] for name in feature_names}
    frame = prepare_features(pd.DataFrame([values], columns=list(feature_names)), feature_names)
    details = {"transaction_id": table["event_id"][0], "step": table["step"][0]}
    return frame, details


def positive_class_shap(explanation: Any, expected_shape: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Normalize supported binary SHAP layouts to one finite positive-class matrix."""
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        if values.shape[:2] != expected_shape or values.shape[2] not in (1, 2):
            raise ValueError(f"Unsupported binary SHAP output shape {values.shape}.")
        values = values[:, :, -1]
    if values.shape != expected_shape or not np.isfinite(values).all():
        raise ValueError(f"Unsupported or non-finite SHAP output shape {values.shape}; expected {expected_shape}.")
    base = np.asarray(explanation.base_values)
    if base.ndim == 0:
        base_value = float(base)
    elif base.ndim == 1:
        base_value = float(base[-1])
    else:
        base_value = float(base.reshape(-1, base.shape[-1])[0, -1])
    if not math.isfinite(base_value):
        raise ValueError("SHAP returned a non-finite base value.")
    return values, base_value


def _safe_divide(numerator: float, denominator: float) -> float:
    return float("nan") if pd.isna(denominator) or denominator <= 0 else numerator / denominator


def _amount_candidate(original: pd.DataFrame, amount: float, *, large_threshold: float,
                      tolerance: float) -> pd.DataFrame:
    """Update amount and its exact deterministic derived features from the API feature builder."""
    candidate = original.copy()
    sender = float(candidate.at[0, "sender_balance_before"])
    destination = float(candidate.at[0, "dest_balance_before"])
    candidate.at[0, "amount"] = amount
    candidate.at[0, "log_amount"] = math.log1p(max(amount, 0.0))
    candidate.at[0, "is_large_amount"] = int(amount >= large_threshold)
    candidate.at[0, "amount_to_sender_balance_ratio"] = _safe_divide(amount, sender)
    candidate.at[0, "amount_ge_sender_balance"] = int(amount >= sender - tolerance and sender > 0)
    candidate.at[0, "sender_remaining_if_debited"] = sender - amount
    candidate.at[0, "amount_to_dest_balance_ratio"] = _safe_divide(amount, destination)
    for role in ("sender", "dest"):
        mean = candidate.at[0, f"{role}_prev_mean_amount"]
        candidate.at[0, f"amount_vs_{role}_hist_mean"] = _safe_divide(amount, float(mean) if not pd.isna(mean) else float("nan"))
    return candidate


def find_counterfactual(model: Any, original: pd.DataFrame, feature_names: list[str] | tuple[str, ...],
                        threshold: float, *, large_threshold: float, tolerance: float,
                        max_candidates: int = 60) -> dict[str, Any] | None:
    """Search a bounded amount/category neighborhood and verify every candidate with the active model."""
    features = list(feature_names)
    source = prepare_features(original, features)
    probability = float(model.predict_proba(source)[0, 1])
    predicted = int(probability >= threshold)
    original_amount = float(source.at[0, "amount"])
    if not math.isfinite(original_amount) or original_amount < 0:
        raise ValueError("Original amount must be finite and non-negative for counterfactual search.")
    candidates: list[tuple[float, str, pd.DataFrame, str]] = []
    # Relative changes are ordered by size. Values remain non-negative and engineered amount
    # fields are recomputed from the frozen API formulas; we cap the scan for responsiveness.
    fractions = (0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
    for fraction in fractions:
        for direction in (-1, 1):
            amount = max(0.0, original_amount * (1.0 + direction * fraction))
            if amount == original_amount:
                continue
            changed = _amount_candidate(source, amount, large_threshold=large_threshold, tolerance=tolerance)
            candidates.append((abs(amount - original_amount) / max(abs(original_amount), 1.0), "amount", changed,
                               f"Amount adjusted by {amount - original_amount:+,.2f}; deterministic amount-derived features recomputed."))
    present_type = next((name for name in TYPE_FEATURES if name in features and source.at[0, name] == 1), None)
    if present_type:
        for name in TYPE_FEATURES:
            if name not in features or name == present_type:
                continue
            changed = source.copy()
            for field in TYPE_FEATURES:
                if field in features:
                    changed.at[0, field] = int(field == name)
            candidates.append((1.0, "transaction type", changed,
                               f"Transaction type category switched from {present_type.removeprefix('type_')} to {name.removeprefix('type_')}."))
    candidates.sort(key=lambda item: (item[0], item[1]))
    for _, _, candidate, description in candidates[:max_candidates]:
        candidate = prepare_features(candidate, features)
        candidate_probability = float(model.predict_proba(candidate)[0, 1])
        candidate_class = int(candidate_probability >= threshold)
        if candidate_class != predicted:
            # Final validation is an independent call through the same ordered inference path.
            verified_probability = float(model.predict_proba(candidate.loc[:, features])[0, 1])
            verified_class = int(verified_probability >= threshold)
            if verified_class == predicted:
                continue
            return {"features": candidate, "prediction": verified_class,
                    "fraud_probability": verified_probability, "original_prediction": predicted,
                    "original_probability": probability, "description": description}
    return None
