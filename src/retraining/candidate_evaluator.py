"""Read-only evaluation of a staged candidate against the active model.

Evaluation uses one supplied held-out dataset and each bundle's already-selected
threshold. It never trains, changes model artifacts, or promotes/rolls back models.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from src.retraining.model_activator import validate_bundle

METRIC_NAMES = frozenset({"precision", "recall", "f1", "pr_auc"})
AUDIT_AND_TARGET_COLUMNS = frozenset({"event_id", "step", "isFraud"})


class CandidateEvaluationError(ValueError):
    """Raised when an evaluation cannot be performed or its report cannot be written."""


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return whether two resolved model directories are equal or nested."""
    left_resolved, right_resolved = left.resolve(), right.resolve()
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


@dataclass(frozen=True)
class AcceptancePolicy:
    """Explicit minimum candidate-minus-active metric deltas.

    There is deliberately no default policy. For example, callers may configure
    {"f1": 0.0, "recall": 0.0}; the evaluator does not choose these limits itself.
    """

    minimum_deltas: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.minimum_deltas:
            raise ValueError("AcceptancePolicy must specify at least one metric delta.")
        for name, value in self.minimum_deltas.items():
            if name not in METRIC_NAMES:
                raise ValueError(f"Unsupported acceptance metric: {name}")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"Minimum delta for {name} must be finite and numeric.")


def _canonical_features() -> list[str]:
    """Load the frozen V1 feature contract without importing the serving API."""
    root = Path(__file__).resolve().parents[2]
    path = root / "models" / "model_v1_metadata.json"
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateEvaluationError(f"Cannot read canonical feature metadata: {exc}") from exc
    features = metadata.get("feature_names") if isinstance(metadata, dict) else None
    if (
        not isinstance(features, list)
        or metadata.get("feature_count") != 34
        or len(features) != 34
        or any(not isinstance(name, str) or not name for name in features)
        or len(set(features)) != 34
        or AUDIT_AND_TARGET_COLUMNS.intersection(features)
    ):
        raise CandidateEvaluationError("Canonical model contract must be 34 ordered features without audit/target fields.")
    return features


def _load_xgboost_model(path: Path) -> Any:
    """Load an XGBoost JSON model lazily; tests can inject a dummy loader."""
    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model(str(path))
    return model


def _validated_probabilities(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities after validating output shape and range."""
    try:
        values = np.asarray(model.predict_proba(features), dtype=float)
    except Exception as exc:
        raise CandidateEvaluationError(f"Model predict_proba failed: {exc}") from exc
    if values.shape != (len(features), 2):
        raise CandidateEvaluationError(
            f"predict_proba must return shape ({len(features)}, 2); got {values.shape}."
        )
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise CandidateEvaluationError("Model probabilities must be finite and in [0, 1].")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-5):
        raise CandidateEvaluationError("Each predict_proba row must sum to 1.")
    return values[:, 1]


def _validate_test_data(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Validate held-out records and return ordered model inputs and shared labels."""
    required = {"event_id", "step", "isFraud", *features}
    missing = required - set(frame.columns)
    if missing:
        raise CandidateEvaluationError(f"Held-out data is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise CandidateEvaluationError("Held-out data is empty.")
    if not frame["event_id"].is_unique:
        raise CandidateEvaluationError("Held-out event_id values must be unique.")
    if not pd.api.types.is_numeric_dtype(frame["step"]):
        raise CandidateEvaluationError("Held-out step values must be numeric.")
    if frame["isFraud"].isna().any() or not frame["isFraud"].isin([0, 1]).all():
        raise CandidateEvaluationError("Held-out isFraud labels must be present and contain only 0/1.")
    if frame[features].isna().any().any():
        raise CandidateEvaluationError("Held-out feature values cannot be missing.")
    if not all(pd.api.types.is_numeric_dtype(frame[name]) for name in features):
        raise CandidateEvaluationError("All held-out model features must be numeric.")
    if not np.isfinite(frame.loc[:, features].to_numpy(dtype=float)).all():
        raise CandidateEvaluationError("Held-out feature values must be finite.")
    # Explicit selection excludes event_id, step, isFraud, and any extra audit fields.
    return frame.loc[:, features].copy(), frame["isFraud"].to_numpy(dtype=int, copy=True)


def _data_fingerprint(frame: pd.DataFrame, features: list[str]) -> str:
    """Hash the ordered input records, including shared labels, for report traceability."""
    ordered = frame.loc[:, ["event_id", "step"] + features + ["isFraud"]]
    hashed_rows = pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes()
    return hashlib.sha256(hashed_rows).hexdigest()


def _write_report(report: dict[str, Any], output_path: Path) -> None:
    """Atomically create a new JSON report without replacing an existing file."""
    if output_path.exists():
        raise CandidateEvaluationError(f"Refusing to overwrite existing evaluation report: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{output_path.name}.",
            suffix=".tmp", dir=output_path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, allow_nan=False)
        # On Windows, rename refuses to replace a report that appeared concurrently.
        os.rename(temporary, output_path)
        temporary = None
    except (OSError, TypeError, ValueError) as exc:
        raise CandidateEvaluationError(f"Could not write evaluation report: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def evaluate_candidate(
    candidate_dir: str | Path,
    active_dir: str | Path,
    held_out_df: pd.DataFrame,
    report_path: str | Path,
    *,
    policy: AcceptancePolicy | None = None,
    model_loader: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Compare candidate and active bundles on identical held-out records.

    Each model uses its own validated stored threshold. This function never tunes
    thresholds: callers must select them using training/validation data beforehand.
    Evaluation failure is returned and recorded in the report; file conflicts or a
    report-write failure raise CandidateEvaluationError. No activator is invoked.
    """
    candidate = Path(candidate_dir)
    active = Path(active_dir)
    output = Path(report_path)
    if _paths_overlap(candidate, active):
        raise CandidateEvaluationError(
            "Candidate and active directories must be separate and non-nested."
        )
    protected_dirs = (candidate, active, active.parent / "backup")
    for model_dir in protected_dirs:
        resolved = model_dir.resolve()
        if output.resolve() == resolved or resolved in output.resolve().parents:
            raise CandidateEvaluationError("Evaluation report cannot be written inside a model artifact directory.")
    if output.exists():
        raise CandidateEvaluationError(f"Refusing to overwrite existing evaluation report: {output}")

    report: dict[str, Any] = {
        "evaluation_status": "FAILED",
        "decision_status": "EVALUATION_FAILED",
        "evaluation_completed": False,
        "policy": None if policy is None else dict(policy.minimum_deltas),
        "error": None,
        "data": None,
        "models": None,
        "metrics": None,
        "candidate_minus_active": None,
        "policy_results": None,
    }
    try:
        features = _canonical_features()
        active_bundle = validate_bundle(active, features)
        candidate_bundle = validate_bundle(candidate, features)
        for label, directory in (("active", active), ("candidate", candidate)):
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            if metadata.get("target") != "isFraud":
                raise CandidateEvaluationError(f"{label} metadata target must be isFraud.")

        model_features, labels = _validate_test_data(held_out_df, features)
        fingerprint = _data_fingerprint(held_out_df, features)
        loader = model_loader or _load_xgboost_model
        active_model = loader(active / "model.json")
        candidate_model = loader(candidate / "model.json")
        active_probabilities = _validated_probabilities(active_model, model_features)
        candidate_probabilities = _validated_probabilities(candidate_model, model_features)

        # Keep metric definitions aligned with Phase 11: thresholded metrics use each
        # bundle's saved threshold; PR-AUC uses probabilities without test tuning.
        from src.retraining.v2_evaluator import _metrics

        active_metrics = _metrics(labels, active_probabilities, active_bundle.threshold)
        candidate_metrics = _metrics(labels, candidate_probabilities, candidate_bundle.threshold)
        deltas = {
            name: float(candidate_metrics[name] - active_metrics[name])
            for name in METRIC_NAMES
        }
        if policy is None:
            decision = "PENDING_POLICY"
            policy_results = None
        else:
            policy_results = {
                name: {
                    "observed_delta": deltas[name],
                    "minimum_delta": float(minimum),
                    "passed": deltas[name] >= float(minimum),
                }
                for name, minimum in policy.minimum_deltas.items()
            }
            decision = "ACCEPTED" if all(item["passed"] for item in policy_results.values()) else "REJECTED"

        report.update({
            "evaluation_status": "COMPLETED",
            "decision_status": decision,
            "evaluation_completed": True,
            "error": None,
            "data": {
                "rows": len(held_out_df),
                "feature_count": len(features),
                "feature_names": features,
                "event_id_sha256": hashlib.sha256(
                    pd.util.hash_pandas_object(held_out_df["event_id"], index=False).to_numpy().tobytes()
                ).hexdigest(),
                "records_sha256": fingerprint,
            },
            "models": {
                "active": {
                    "model_version": active_bundle.model_version,
                    "threshold": active_bundle.threshold,
                    "artifact_sha256": dict(active_bundle.sha256),
                },
                "candidate": {
                    "model_version": candidate_bundle.model_version,
                    "threshold": candidate_bundle.threshold,
                    "artifact_sha256": dict(candidate_bundle.sha256),
                },
            },
            "metrics": {"active": active_metrics, "candidate": candidate_metrics},
            "candidate_minus_active": deltas,
            "policy_results": policy_results,
        })
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"

    _write_report(report, output)
    return report
