"""Isolated, read-only evaluation of an orchestrator candidate on its held-out test split.

Evaluation produces a recommendation only. It never changes model artifacts or
active-model state, and defaults to REVIEW_REQUIRED because this project has no
validated production acceptance thresholds.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.orchestrator.config import (
    RUNS_ROOT,
    OrchestratorPathError,
    resolve_run_dir,
    resolve_run_path,
    validate_run_id,
)
from src.retraining.active_model import resolve_active_model
from src.retraining.cycle_pipeline import FINAL_FEATURES
from src.retraining.model_activator import validate_bundle


TARGET = "isFraud"
AUDIT_COLUMNS = ("event_id", "step")
EXPECTED_COLUMNS = [*AUDIT_COLUMNS, *FINAL_FEATURES, TARGET]
METRIC_NAMES = ("precision", "recall", "f1", "pr_auc", "roc_auc")
CONFUSION_NAMES = ("tp", "fp", "tn", "fn")
_DATASET_FILES = ("candidate_dataset.parquet", "dataset_metadata.json", "dataset_validation.json", "split_summary.json")
_CANDIDATE_FILES = ("model.json", "metadata.json", "threshold.json", "training_report.json")
_VALIDATION_POLICY = "orchestrator-candidate-evaluation-v1"


class CandidateEvaluationError(ValueError):
    """Raised when evaluation cannot be safely initialized or output already exists."""


@dataclass(frozen=True)
class EvaluationCriteria:
    """Explicit demo criteria; no default values imply a production policy.

    ``minimum_deltas`` must contain all five reported scalar metrics. The two
    count caps bound allowed increases in false positives and false negatives.
    Values must be configured and validated by the caller for their use case.
    """

    minimum_deltas: Mapping[str, float]
    max_false_positive_increase: int
    max_false_negative_increase: int
    description: str

    def __post_init__(self) -> None:
        if set(self.minimum_deltas) != set(METRIC_NAMES):
            raise ValueError(f"minimum_deltas must specify exactly: {', '.join(METRIC_NAMES)}")
        for name, value in self.minimum_deltas.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"Minimum delta for {name} must be finite and numeric.")
        for name, value in (("max_false_positive_increase", self.max_false_positive_increase),
                            ("max_false_negative_increase", self.max_false_negative_increase)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("Criteria description must explain the configured evaluation policy.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_fingerprint(frame: pd.DataFrame) -> str:
    rows = pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
    return hashlib.sha256(rows).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateEvaluationError(f"{label} is missing or invalid: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CandidateEvaluationError(f"{label} must be a JSON object: {path}")
    return value


def _validate_run(run_id: str | None, run_root: str | Path) -> tuple[str, Path]:
    if run_id is None:
        raise CandidateEvaluationError("A valid orchestrator run_id is required.")
    try:
        safe_id = validate_run_id(run_id)
        run_dir = resolve_run_dir(safe_id, run_root)
    except (OrchestratorPathError, TypeError) as exc:
        raise CandidateEvaluationError(f"Invalid run ID or run root: {exc}") from exc
    if not run_dir.is_dir():
        raise CandidateEvaluationError(f"Orchestrator run does not exist: {safe_id}")
    record = _read_json(resolve_run_path(safe_id, "run.json", run_root=run_root), "Run record")
    if record.get("run_id") != safe_id:
        raise CandidateEvaluationError("Run record identity does not match the requested run.")
    return safe_id, run_dir


def _load_inputs(run_id: str, run_root: str | Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    dataset_dir = resolve_run_path(run_id, "dataset", run_root=run_root)
    candidate_dir = resolve_run_path(run_id, "candidate", run_root=run_root)
    dataset_paths = {name: resolve_run_path(run_id, "dataset", name, run_root=run_root) for name in _DATASET_FILES}
    candidate_paths = {name: resolve_run_path(run_id, "candidate", name, run_root=run_root) for name in _CANDIDATE_FILES}
    if (not dataset_dir.is_dir() or dataset_dir.is_symlink()
            or any(not p.is_file() or p.is_symlink() for p in dataset_paths.values())):
        raise CandidateEvaluationError("Stage 4 dataset preparation is missing or incomplete.")
    if (not candidate_dir.is_dir() or candidate_dir.is_symlink()
            or any(not p.is_file() or p.is_symlink() for p in candidate_paths.values())):
        raise CandidateEvaluationError("Stage 5 candidate training bundle is missing or incomplete.")

    dataset_meta = _read_json(dataset_paths["dataset_metadata.json"], "Stage 4 dataset metadata")
    dataset_validation = _read_json(dataset_paths["dataset_validation.json"], "Stage 4 dataset validation")
    split = _read_json(dataset_paths["split_summary.json"], "Stage 4 split summary")
    candidate_meta = _read_json(candidate_paths["metadata.json"], "Candidate metadata")
    threshold_data = _read_json(candidate_paths["threshold.json"], "Candidate threshold")
    training_report = _read_json(candidate_paths["training_report.json"], "Candidate training report")

    for label, record in (("dataset metadata", dataset_meta), ("dataset validation", dataset_validation),
                          ("split summary", split), ("candidate metadata", candidate_meta),
                          ("training report", training_report)):
        if record.get("run_id", run_id) != run_id:
            raise CandidateEvaluationError(f"{label} does not belong to this run.")
    if dataset_validation.get("passed") is not True:
        raise CandidateEvaluationError("Stage 4 dataset validation was not successful.")
    dataset_eligibility = dataset_meta.get("eligibility")
    if not isinstance(dataset_eligibility, Mapping) or dataset_eligibility.get("decision_status") != "ELIGIBLE":
        raise CandidateEvaluationError("Stage 4 dataset does not record an ELIGIBLE provenance snapshot.")
    provenance = dataset_meta.get("label_provenance")
    if not isinstance(provenance, Mapping):
        raise CandidateEvaluationError("Stage 4 dataset is missing its label provenance summary.")
    source = provenance.get("source")
    provenance_text = " ".join(str(provenance.get(key, "")) for key in
                                ("source", "label_provenance", "label_type", "ground_truth_label_type", "dataset")).casefold()
    if (provenance.get("source_type") != "trusted_verified"
            or provenance.get("synthetic") is not False
            or provenance.get("matching_key") != "event_id"
            or provenance.get("transaction_event_id_mapping_verified") is not True
            or not isinstance(source, str) or not source.strip()
            or not isinstance(provenance.get("label_provenance"), str)
            or not provenance.get("label_provenance", "").strip()
            or provenance.get("unmatched_label_count") != 0
            or provenance.get("conflict_count") != 0
            or "paysim" in provenance_text or "simulat" in provenance_text or "synthetic" in provenance_text):
        raise CandidateEvaluationError("Stage 4 provenance is unknown, simulated, unmatched, conflicting, or not event_id-verified.")
    if training_report.get("status") != "CANDIDATE_TRAINED_NOT_PROMOTED":
        raise CandidateEvaluationError("Stage 5 training report does not confirm a completed, unpromoted candidate.")
    if candidate_meta.get("dataset_fingerprint") != dataset_meta.get("dataset_fingerprint_sha256"):
        raise CandidateEvaluationError("Candidate metadata dataset fingerprint differs from Stage 4 metadata.")
    if candidate_meta.get("label_provenance_summary") != provenance:
        raise CandidateEvaluationError("Candidate training provenance differs from Stage 4 label provenance.")
    if (candidate_meta.get("feature_count") != len(FINAL_FEATURES)
            or candidate_meta.get("feature_names") != list(FINAL_FEATURES)
            or candidate_meta.get("target") != TARGET):
        raise CandidateEvaluationError("Candidate metadata does not match the canonical 34-feature contract.")
    if candidate_meta.get("training_data_policy") != (
        "train rows only for fit; validation rows for early stopping and threshold/model selection; test rows reserved"
    ):
        raise CandidateEvaluationError("Candidate training metadata does not establish held-out test isolation.")
    test_report = training_report.get("test")
    if (not isinstance(test_report, Mapping)
            or test_report.get("accessed_for_fit_or_selection") is not False
            or test_report.get("status") != "RESERVED_FOR_LATER_EVALUATION"):
        raise CandidateEvaluationError("Stage 5 report does not confirm the test split remained reserved.")
    if candidate_meta.get("test_metrics") != {
        "status": "NOT_EVALUATED_RESERVED_FOR_LATER_EVALUATION",
        "rows": test_report.get("rows"),
    }:
        raise CandidateEvaluationError("Candidate metadata indicates test data was evaluated or has inconsistent reserved status.")
    selection_method = threshold_data.get("selection_method")
    if not isinstance(selection_method, str) or "validation" not in selection_method.casefold():
        raise CandidateEvaluationError("Candidate threshold was not recorded as validation-selected.")
    candidate_threshold = threshold_data.get("selected_threshold")
    if (isinstance(candidate_threshold, bool) or not isinstance(candidate_threshold, (int, float))
            or not math.isfinite(float(candidate_threshold)) or not 0 < float(candidate_threshold) < 1):
        raise CandidateEvaluationError("Candidate threshold is missing or invalid.")
    if dataset_meta.get("run_id") != run_id or not isinstance(dataset_meta.get("dataset_fingerprint_sha256"), str):
        raise CandidateEvaluationError("Stage 4 dataset identity/fingerprint is invalid.")

    try:
        frame = pd.read_parquet(dataset_paths["candidate_dataset.parquet"])
    except Exception as exc:
        raise CandidateEvaluationError(f"Stage 4 candidate dataset cannot be read: {exc}") from exc
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise CandidateEvaluationError("Stage 4 dataset schema differs from the exact 34-feature contract.")
    fingerprint = _dataset_fingerprint(frame)
    checks = dataset_validation.get("checks")
    if (fingerprint != dataset_meta.get("dataset_fingerprint_sha256")
            or fingerprint != candidate_meta.get("dataset_fingerprint")
            or not isinstance(checks, Mapping) or fingerprint != checks.get("dataset_sha256")):
        raise CandidateEvaluationError("Stage 4/candidate dataset fingerprint validation failed.")
    if dataset_meta.get("feature_names") != list(FINAL_FEATURES) or dataset_meta.get("feature_count") != 34:
        raise CandidateEvaluationError("Stage 4 metadata does not match the 34-feature model contract.")
    if dataset_meta.get("row_count") != len(frame) or checks.get("row_count") != len(frame):
        raise CandidateEvaluationError("Stage 4 dataset row count differs from its saved validation metadata.")
    if not frame["event_id"].is_unique or frame[TARGET].isna().any() or not frame[TARGET].isin([0, 1]).all():
        raise CandidateEvaluationError("Stage 4 event IDs or ground-truth labels are invalid.")
    if (not pd.api.types.is_numeric_dtype(frame["step"])
            or not np.isfinite(frame["step"].to_numpy(dtype=float)).all()
            or not frame["step"].is_monotonic_increasing):
        raise CandidateEvaluationError("Stage 4 step values must be finite, numeric, and chronologically ordered.")
    if not all(pd.api.types.is_numeric_dtype(frame[name]) for name in FINAL_FEATURES):
        raise CandidateEvaluationError("Stage 4 model features must all be numeric.")
    feature_values = frame.loc[:, FINAL_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        raise CandidateEvaluationError("Stage 4 model features must be finite and non-missing.")

    if not all(split.get(key) is True for key in ("step_ranges_disjoint", "event_id_sets_disjoint")):
        raise CandidateEvaluationError("Stage 4 split report does not confirm disjoint partitions.")
    train, validation, test = (split.get(name) for name in ("train", "validation", "test"))
    if not all(isinstance(item, Mapping) for item in (train, validation, test)):
        raise CandidateEvaluationError("Stage 4 split summary is missing a partition.")
    bounds = (train.get("start_step"), train.get("end_step"), validation.get("start_step"),
              validation.get("end_step"), test.get("start_step"), test.get("end_step"))
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in bounds):
        raise CandidateEvaluationError("Stage 4 split boundaries are missing or invalid.")
    if not (bounds[0] <= bounds[1] < bounds[2] <= bounds[3] < bounds[4] <= bounds[5]):
        raise CandidateEvaluationError("Stage 4 split boundaries are not chronological and disjoint.")
    partition_rows: dict[str, pd.DataFrame] = {
        "train": frame.loc[frame["step"].between(bounds[0], bounds[1])],
        "validation": frame.loc[frame["step"].between(bounds[2], bounds[3])],
        "test": frame.loc[frame["step"].between(bounds[4], bounds[5])],
    }
    for name, partition in partition_rows.items():
        expected_rows = split[name].get("rows")
        if (isinstance(expected_rows, bool) or not isinstance(expected_rows, Integral)
                or expected_rows <= 0 or len(partition) != expected_rows):
            raise CandidateEvaluationError(f"Stage 4 {name} partition is empty or inconsistent with its saved row count.")
    if sum(len(partition) for partition in partition_rows.values()) != len(frame):
        raise CandidateEvaluationError("Saved train/validation/test partitions do not cover the Stage 4 dataset exactly.")
    test_rows = frame.loc[frame["step"].between(bounds[4], bounds[5])].copy().reset_index(drop=True)
    expected_test_rows = test.get("rows")
    if (isinstance(expected_test_rows, bool) or not isinstance(expected_test_rows, Integral)
            or expected_test_rows <= 0 or len(test_rows) != expected_test_rows):
        raise CandidateEvaluationError("Stage 4 test split is missing, empty, or inconsistent with its row count.")
    if not test_rows["step"].between(bounds[4], bounds[5]).all():
        raise CandidateEvaluationError("Test rows escaped the recorded Stage 4 test boundaries.")
    if (training_report.get("dataset_fingerprint") != fingerprint
            or training_report.get("candidate_version") != candidate_meta.get("model_version")):
        raise CandidateEvaluationError("Stage 5 training report does not match this candidate dataset/version.")
    if (provenance.get("matched_label_count") != len(frame)
            or provenance.get("matched_event_ids_sha256") != checks.get("event_id_sha256")
            or training_report.get("provenance") != provenance):
        raise CandidateEvaluationError("Stage 4/5 verified-label provenance does not match dataset event IDs and row counts.")
    if training_report.get("test", {}).get("rows") != len(test_rows):
        raise CandidateEvaluationError("Stage 5 reserved test row count differs from Stage 4.")
    temporal = candidate_meta.get("temporal_split")
    if not isinstance(temporal, Mapping):
        raise CandidateEvaluationError("Candidate metadata is missing its Stage 5 temporal split record.")
    expected_temporal = {
        "train_start_step": bounds[0], "train_end_step": bounds[1],
        "validation_start_step": bounds[2], "validation_end_step": bounds[3],
        "test_start_step": bounds[4], "test_end_step": bounds[5],
    }
    if any(temporal.get(key) != value for key, value in expected_temporal.items()):
        raise CandidateEvaluationError("Candidate training temporal boundaries differ from the Stage 4 split.")
    if (candidate_meta.get("training_rows") != len(partition_rows["train"])
            or candidate_meta.get("validation_rows") != len(partition_rows["validation"])
            or threshold_data.get("validation_row_count") != len(partition_rows["validation"])):
        raise CandidateEvaluationError("Stage 5 training/threshold metadata does not match the saved train and validation splits.")
    train_report = training_report.get("training")
    validation_report = training_report.get("validation")
    if (not isinstance(train_report, Mapping) or not isinstance(validation_report, Mapping)
            or train_report.get("rows") != len(partition_rows["train"])
            or train_report.get("start_step") != bounds[0]
            or train_report.get("end_step") != bounds[1]
            or validation_report.get("rows") != len(partition_rows["validation"])
            or validation_report.get("start_step") != bounds[2]
            or validation_report.get("end_step") != bounds[3]
            or test_report.get("start_step") != bounds[4]
            or test_report.get("end_step") != bounds[5]):
        raise CandidateEvaluationError("Stage 5 training report boundaries differ from Stage 4 split boundaries.")
    return test_rows, dataset_meta, candidate_meta, threshold_data, candidate_dir


def _default_model_loader(path: Path) -> Any:
    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model(str(path))
    return model


def _positive_probabilities(model: Any, features: pd.DataFrame, label: str) -> np.ndarray:
    try:
        values = np.asarray(model.predict_proba(features), dtype=float)
    except Exception as exc:
        raise CandidateEvaluationError(f"{label} model prediction failed: {exc}") from exc
    if values.shape != (len(features), 2):
        raise CandidateEvaluationError(f"{label} predict_proba returned unexpected shape {values.shape}.")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise CandidateEvaluationError(f"{label} probabilities must be finite and in [0, 1].")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-5):
        raise CandidateEvaluationError(f"{label} probability rows do not sum to one.")
    return values[:, 1]


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = probabilities >= threshold
    actual = labels == 1
    tp = int(np.count_nonzero(predicted & actual))
    fp = int(np.count_nonzero(predicted & ~actual))
    tn = int(np.count_nonzero(~predicted & ~actual))
    fn = int(np.count_nonzero(~predicted & actual))
    reasons: dict[str, str] = {}
    precision: float | None = tp / (tp + fp) if tp + fp else None
    recall: float | None = tp / (tp + fn) if tp + fn else None
    if precision is None:
        reasons["precision"] = "No positive predictions; precision is undefined."
    if recall is None:
        reasons["recall"] = "No positive ground-truth examples; recall is undefined."
    f1: float | None = None
    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    else:
        reasons["f1"] = "Precision or recall is undefined for this test split/prediction set."
    pr_auc: float | None = None
    roc_auc: float | None = None
    if np.any(actual):
        pr_auc = float(average_precision_score(labels, probabilities))
    else:
        reasons["pr_auc"] = "No positive ground-truth examples; PR-AUC is undefined."
    if np.unique(labels).size == 2:
        roc_auc = float(roc_auc_score(labels, probabilities))
    else:
        reasons["roc_auc"] = "ROC-AUC requires both ground-truth classes in the test split."
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "pr_auc": pr_auc, "roc_auc": roc_auc,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "threshold": float(threshold), "rows": int(len(labels)),
        "unavailable_metric_reasons": reasons,
    }


def _comparison(active: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for name in METRIC_NAMES:
        left, right = active[name], candidate[name]
        if left is None or right is None:
            deltas[name] = {"absolute": None, "relative": None,
                            "reason": "Relative/absolute comparison unavailable because a metric is undefined."}
            continue
        difference = float(right - left)
        deltas[name] = {
            "absolute": difference,
            "relative": (difference / abs(float(left))) if float(left) != 0 else None,
            "relative_unavailable_reason": "Active metric is zero." if float(left) == 0 else None,
        }
    confusion = {name: int(candidate[name] - active[name]) for name in CONFUSION_NAMES}
    return {"candidate_minus_active": deltas, "confusion_matrix_differences": confusion}


def _criteria_results(criteria: EvaluationCriteria | None, active: Mapping[str, Any],
                      candidate: Mapping[str, Any]) -> tuple[str, dict[str, Any] | None, list[str]]:
    if criteria is None:
        return "REVIEW_REQUIRED", None, [
            "No validated acceptance criteria are configured; metric results are informational and require human review."
        ]
    results: dict[str, Any] = {}
    unavailable: list[str] = []
    failed = False
    for name, minimum in criteria.minimum_deltas.items():
        left, right = active.get(name), candidate.get(name)
        if left is None or right is None:
            results[name] = {"passed": None, "minimum_delta": float(minimum), "observed_delta": None,
                             "reason": "Required metric is unavailable."}
            unavailable.append(name)
        else:
            delta = float(right - left)
            passed = delta >= float(minimum)
            failed = failed or not passed
            results[name] = {"passed": passed, "minimum_delta": float(minimum), "observed_delta": delta}
    fp_increase = int(candidate["fp"] - active["fp"])
    fn_increase = int(candidate["fn"] - active["fn"])
    for name, observed, maximum in (
        ("false_positive_increase", fp_increase, criteria.max_false_positive_increase),
        ("false_negative_increase", fn_increase, criteria.max_false_negative_increase),
    ):
        passed = observed <= maximum
        failed = failed or not passed
        results[name] = {"passed": passed, "observed_increase": observed, "maximum_increase": maximum}
    if unavailable:
        status = "REVIEW_REQUIRED"
    elif failed:
        status = "CANDIDATE_REJECTED"
    else:
        status = "CANDIDATE_ACCEPTED"
    return status, results, [f"Configured evaluation criteria: {criteria.description}"]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _publish_outputs(run_id: str, run_dir: Path, report: Mapping[str, Any],
                     comparison_rows: list[dict[str, Any]], validation: Mapping[str, Any]) -> dict[str, str]:
    output_dir = resolve_run_path(run_id, "evaluation", run_root=run_dir.parent)
    if output_dir.exists():
        raise CandidateEvaluationError(f"Evaluation output already exists; refusing overwrite: {output_dir}")
    stage = Path(tempfile.mkdtemp(prefix=".evaluation-stage-", dir=run_dir))
    try:
        _write_json(stage / "evaluation_report.json", report)
        pd.DataFrame(comparison_rows).to_csv(stage / "model_comparison.csv", index=False)
        _write_json(stage / "evaluation_validation.json", validation)
        os.rename(stage, output_dir)
        stage = None  # type: ignore[assignment]
    except FileExistsError as exc:
        raise CandidateEvaluationError(f"Evaluation output already exists; refusing overwrite: {output_dir}") from exc
    except Exception as exc:
        raise CandidateEvaluationError(f"Could not safely publish evaluation outputs: {type(exc).__name__}: {exc}") from exc
    finally:
        if stage is not None and Path(stage).exists():
            shutil.rmtree(stage, ignore_errors=True)
    return {name: str(output_dir / name) for name in
            ("evaluation_report.json", "model_comparison.csv", "evaluation_validation.json")}


def evaluate_candidate(run_id: str | None, *, run_root: str | Path = RUNS_ROOT,
                       criteria: EvaluationCriteria | None = None,
                       model_loader: Callable[[Path], Any] | None = None,
                       active_resolver: Callable[..., Any] = resolve_active_model) -> dict[str, Any]:
    """Evaluate the Stage 5 candidate and current active bundle on the Stage 4 test rows.

    The default result is ``REVIEW_REQUIRED``. Supplying ``EvaluationCriteria``
    enables an explicit recommendation, never administrative approval or model
    activation. For a valid run, precondition failures are written as an
    ``EVALUATION_FAILED`` report; invalid/nonexistent run IDs cannot be recorded.
    """
    safe_id, run_dir = _validate_run(run_id, run_root)
    evaluation_dir = resolve_run_path(safe_id, "evaluation", run_root=run_root)
    if evaluation_dir.exists() or evaluation_dir.is_symlink():
        raise CandidateEvaluationError(f"Evaluation already exists; refusing duplicate: {evaluation_dir}")
    timestamp = datetime.now(timezone.utc).isoformat()
    validation_checks: dict[str, Any] = {"run_id_valid": True, "run_exists": True}
    report: dict[str, Any] = {
        "run_id": safe_id,
        "evaluation_timestamp_utc": timestamp,
        "evaluation_status": "EVALUATION_FAILED",
        "decision_status": "EVALUATION_FAILED",
        "decision_is_administrative_approval": False,
        "criteria": None if criteria is None else {
            "minimum_deltas": dict(criteria.minimum_deltas),
            "max_false_positive_increase": criteria.max_false_positive_increase,
            "max_false_negative_increase": criteria.max_false_negative_increase,
            "description": criteria.description,
        },
        "failure_reason": None,
    }
    comparison_rows: list[dict[str, Any]] = []
    try:
        test_rows, dataset_meta, candidate_meta, threshold_data, candidate_dir = _load_inputs(safe_id, run_root)
        validation_checks.update({
            "stage4_dataset_valid": True,
            "stage4_fingerprint_matches": True,
            "stage5_candidate_valid": True,
            "test_partition_non_empty": True,
            "test_rows_reserved_before_evaluation": True,
        })
        active = active_resolver(project_root=Path(__file__).resolve().parents[2])
        validation_checks["active_model_resolved"] = True
        if len(active.feature_names) != 34 or list(active.feature_names) != list(FINAL_FEATURES):
            raise CandidateEvaluationError("Active model feature contract differs from the canonical 34-feature order.")
        active_hashes = dict(active.artifact_sha256)
        for name, path in (("model.json", active.model_path), ("metadata.json", active.metadata_path),
                           ("threshold.json", active.threshold_path)):
            if _sha256(Path(path)) != active_hashes.get(name):
                raise CandidateEvaluationError(f"Active model integrity changed or failed validation: {name}.")
        validation_checks["active_model_integrity_valid"] = True
        candidate_bundle = validate_bundle(candidate_dir, FINAL_FEATURES)
        validation_checks["candidate_bundle_valid"] = True
        if (candidate_bundle.feature_count != len(active.feature_names)
                or candidate_bundle.feature_count != len(FINAL_FEATURES)):
            raise CandidateEvaluationError("Candidate/active feature counts do not match.")
        if candidate_bundle.model_version != candidate_meta.get("model_version"):
            raise CandidateEvaluationError("Candidate bundle version differs from candidate metadata.")
        validation_checks["feature_contract_matches"] = True
        if float(threshold_data["selected_threshold"]) != float(candidate_bundle.threshold):
            raise CandidateEvaluationError("Candidate threshold metadata differs from its validated bundle threshold.")
        if not 0 < float(active.threshold) < 1:
            raise CandidateEvaluationError("Active model resolver returned an invalid threshold.")
        validation_checks["candidate_threshold_validated"] = True
        validation_checks["active_threshold_resolved"] = True

        # Explicitly take only the canonical model columns; audit and target columns
        # never enter either estimator. Both models share this same DataFrame/labels.
        features = test_rows.loc[:, list(FINAL_FEATURES)].copy()
        labels = test_rows[TARGET].to_numpy(dtype=int, copy=True)
        if len(labels) != len(features) or len(labels) == 0:
            raise CandidateEvaluationError("The verified Stage 4 test partition is empty or inconsistent.")
        loader = model_loader or _default_model_loader
        active_model = loader(Path(active.model_path))
        candidate_model = loader(candidate_dir / "model.json")
        active_probabilities = _positive_probabilities(active_model, features, "Active")
        candidate_probabilities = _positive_probabilities(candidate_model, features, "Candidate")
        active_metrics = _metrics(labels, active_probabilities, float(active.threshold))
        candidate_metrics = _metrics(labels, candidate_probabilities, float(candidate_bundle.threshold))
        comparison = _comparison(active_metrics, candidate_metrics)
        decision, criteria_results, reasons = _criteria_results(criteria, active_metrics, candidate_metrics)
        for name in METRIC_NAMES:
            left, right = active_metrics[name], candidate_metrics[name]
            difference = None if left is None or right is None else float(right - left)
            relative = None if left is None or right is None or float(left) == 0 else difference / abs(float(left))
            comparison_rows.append({"metric": name, "active": left, "candidate": right,
                                    "candidate_minus_active": difference, "relative_change": relative})
        for name in CONFUSION_NAMES:
            comparison_rows.append({"metric": name, "active": active_metrics[name],
                                    "candidate": candidate_metrics[name],
                                    "candidate_minus_active": comparison["confusion_matrix_differences"][name],
                                    "relative_change": None})
        validation_checks["same_test_rows_for_both_models"] = True
        validation_checks["same_34_features_for_both_models"] = True
        validation_checks["test_rows_used_only_for_evaluation"] = True
        report.update({
            "evaluation_status": "COMPLETED",
            "decision_status": decision,
            "failure_reason": None,
            "decision_reasons": reasons,
            "data": {
                "dataset_fingerprint": dataset_meta["dataset_fingerprint_sha256"],
                "test_rows": len(test_rows),
                "feature_count": len(FINAL_FEATURES),
                "feature_names": list(FINAL_FEATURES),
                "test_event_id_sha256": hashlib.sha256(
                    pd.util.hash_pandas_object(test_rows["event_id"], index=False).to_numpy().tobytes()
                ).hexdigest(),
            },
            "models": {
                "active": {"model_version": active.model_version, "threshold": float(active.threshold),
                           "artifact_sha256": active_hashes},
                "candidate": {"model_version": candidate_bundle.model_version,
                              "run_id": safe_id, "threshold": float(candidate_bundle.threshold),
                              "artifact_sha256": dict(candidate_bundle.sha256)},
            },
            "metrics": {"active": active_metrics, "candidate": candidate_metrics},
            "comparison": comparison,
            "criteria_results": criteria_results,
        })
        validation_checks["metrics_computed"] = True
        validation_checks["evaluation_completed"] = True
    except Exception as exc:
        report["failure_reason"] = f"{type(exc).__name__}: {exc}"
        validation_checks["evaluation_completed"] = False

    validation_record = {
        "run_id": safe_id,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_policy": _VALIDATION_POLICY,
        "passed": report["evaluation_status"] == "COMPLETED",
        "checks": validation_checks,
        "failure_reason": report["failure_reason"],
    }
    report["output_paths"] = _publish_outputs(safe_id, run_dir, report, comparison_rows, validation_record)
    return report


__all__ = ["CandidateEvaluationError", "EvaluationCriteria", "evaluate_candidate"]
