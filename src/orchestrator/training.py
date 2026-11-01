"""Train and persist a candidate model inside an eligible orchestrator run.

Only the Stage 4 dataset is consumed. The chronological test partition is kept
reserved: it is not passed to fitting, threshold selection, or metric functions.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.orchestrator import eligibility, label_adapter
from src.orchestrator.config import RUNS_ROOT, OrchestratorPathError, resolve_run_dir, resolve_run_path, validate_run_id
from src.retraining import v2_config
from src.retraining.cycle_pipeline import FINAL_FEATURES
from src.retraining.model_activator import validate_bundle


TARGET = "isFraud"
EXPECTED_COLUMNS = ["event_id", "step", *FINAL_FEATURES, TARGET]
DATASET_FILES = ("candidate_dataset.parquet", "dataset_metadata.json", "dataset_validation.json", "split_summary.json")
_FINGERPRINT_ALGORITHM = "pandas.util.hash_pandas_object(index=False)+sha256"


class CandidateTrainingError(ValueError):
    """Raised when training prerequisites or candidate persistence are unsafe."""


def _load_phase11_components():
    """Import the frozen Phase 11 helpers only after all eligibility gates pass."""
    from src.retraining import v2_evaluator, v2_trainer
    return v2_trainer, v2_evaluator


def _fingerprint(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateTrainingError(f"{label} is missing or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise CandidateTrainingError(f"{label} must contain a JSON object: {path}")
    return value


def _require_count(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise CandidateTrainingError(f"Dataset metadata has an invalid {field}.")
    return int(value)


def _validate_run(run_id: str | None, run_root: str | Path) -> tuple[str, Path]:
    if run_id is None:
        raise CandidateTrainingError("An existing orchestrator run_id is required.")
    try:
        safe_id = validate_run_id(run_id)
        run_dir = resolve_run_dir(safe_id, run_root)
    except (OrchestratorPathError, TypeError) as exc:
        raise CandidateTrainingError(f"Invalid run ID or run path: {exc}") from exc
    if not run_dir.is_dir():
        raise CandidateTrainingError(f"Orchestrator run directory does not exist: {run_dir}")
    record = _read_json(resolve_run_path(safe_id, "run.json", run_root=run_root), "Run record")
    if record.get("run_id") != safe_id:
        raise CandidateTrainingError("Run record identity does not match the requested run.")
    return safe_id, run_dir


def _validate_label_provenance(provenance: Any, label_summary: Mapping[str, Any], row_count: int,
                               event_id_hash: str) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise CandidateTrainingError("Candidate dataset has no verified label provenance summary.")
    source = provenance.get("source")
    if (provenance.get("source_type") != "trusted_verified"
            or provenance.get("synthetic") is not False
            or provenance.get("matching_key") != "event_id"
            or provenance.get("transaction_event_id_mapping_verified") is not True
            or provenance.get("matched_event_ids_sha256") != event_id_hash
            or provenance.get("matched_label_count") != row_count
            or provenance.get("unmatched_label_count") != 0
            or provenance.get("conflict_count") != 0
            or not isinstance(source, str) or not source.strip()
            or not isinstance(provenance.get("label_provenance"), str)
            or not provenance.get("label_provenance", "").strip()):
        raise CandidateTrainingError("Candidate dataset provenance is incomplete or inconsistent with verified labels.")
    forbidden_text = " ".join(str(provenance.get(key, "")) for key in (
        "source", "label_provenance", "label_type", "ground_truth_label_type", "phase9_label_type", "dataset"
    )).casefold()
    if (provenance.get("ground_truth_label_type") == "SIMULATED_DELAYED_GROUND_TRUTH"
            or provenance.get("phase9_label_type") == "SIMULATED_DELAYED_GROUND_TRUTH"
            or "paysim" in forbidden_text or "simulat" in forbidden_text or "synthetic" in forbidden_text):
        raise CandidateTrainingError("Simulated PaySim/synthetic labels cannot be used for candidate training.")

    if label_summary.get("status") != "AVAILABLE" or label_summary.get("strict_retraining_ready") is not True:
        raise CandidateTrainingError("Current label adapter does not confirm strict genuine-label readiness.")
    if label_summary.get("matching_key") != "event_id":
        raise CandidateTrainingError("Current label summary does not confirm lifecycle-required event_id matching.")
    config = label_summary.get("trusted_source_configuration")
    if (not isinstance(config, Mapping) or source not in config.get("configured_sources", [])
            or config.get("authorization_token_configured") is not True):
        raise CandidateTrainingError("Trusted source is not currently configured and authorized.")
    genuine = _require_count(label_summary.get("genuinely_verified_labels"), "genuinely_verified_labels")
    if genuine < row_count:
        raise CandidateTrainingError("Current genuinely verified label count is below the candidate row count.")
    for key in ("unmatched_labels", "conflict_count", "unresolved_or_rejected_count"):
        if _require_count(label_summary.get(key), key) != 0:
            raise CandidateTrainingError(f"Current label summary has non-zero {key}.")
    return dict(provenance)


def _load_candidate_dataset(run_id: str, run_root: str | Path) -> tuple[Path, pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    dataset_dir = resolve_run_path(run_id, "dataset", run_root=run_root)
    paths = {name: resolve_run_path(run_id, "dataset", name, run_root=run_root) for name in DATASET_FILES}
    if not dataset_dir.is_dir() or any(not path.is_file() or path.is_symlink() for path in paths.values()):
        raise CandidateTrainingError("A complete Stage 4 candidate dataset is required before training.")
    metadata = _read_json(paths["dataset_metadata.json"], "Dataset metadata")
    validation = _read_json(paths["dataset_validation.json"], "Dataset validation")
    split = _read_json(paths["split_summary.json"], "Split summary")
    if metadata.get("run_id") != run_id or validation.get("run_id") != run_id or split.get("run_id") != run_id:
        raise CandidateTrainingError("Dataset artifacts do not belong to the requested run.")
    if validation.get("passed") is not True:
        raise CandidateTrainingError("Stage 4 dataset validation did not pass.")
    if (metadata.get("feature_count") != len(FINAL_FEATURES)
            or metadata.get("feature_names") != list(FINAL_FEATURES)
            or metadata.get("target_column") != TARGET
            or metadata.get("audit_columns") != ["event_id", "step"]):
        raise CandidateTrainingError("Dataset metadata does not match the frozen 34-feature contract.")
    checks = validation.get("checks")
    if not isinstance(checks, dict) or any(value is not True for key, value in checks.items() if isinstance(value, bool)):
        raise CandidateTrainingError("Dataset validation report contains a failed schema or integrity check.")
    if not all(split.get(key) is True for key in ("step_ranges_disjoint", "event_id_sets_disjoint")):
        raise CandidateTrainingError("Dataset split summary does not confirm disjoint chronological partitions.")
    try:
        frame = pd.read_parquet(paths["candidate_dataset.parquet"])
    except Exception as exc:
        raise CandidateTrainingError(f"Candidate parquet cannot be read: {exc}") from exc
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise CandidateTrainingError("Candidate parquet does not match the exact 34-feature schema/order.")
    if len(frame) != metadata.get("row_count") or len(frame) != checks.get("row_count"):
        raise CandidateTrainingError("Candidate parquet row count differs from its metadata.")
    fingerprint = _fingerprint(frame)
    if (fingerprint != metadata.get("dataset_fingerprint_sha256")
            or fingerprint != checks.get("dataset_sha256")):
        raise CandidateTrainingError("Candidate dataset fingerprint does not match its metadata/validation report.")
    if not frame["event_id"].is_unique or frame["isFraud"].isna().any() or not frame["isFraud"].isin([0, 1]).all():
        raise CandidateTrainingError("Candidate transaction IDs or targets failed revalidation.")
    if not pd.api.types.is_numeric_dtype(frame["step"]) or not frame["step"].is_monotonic_increasing:
        raise CandidateTrainingError("Candidate dataset must be chronologically sorted with numeric steps.")
    for feature in FINAL_FEATURES:
        if not pd.api.types.is_numeric_dtype(frame[feature]):
            raise CandidateTrainingError(f"Candidate feature {feature!r} is not numeric.")
        values = frame[feature].to_numpy(dtype="float64", na_value=np.nan)
        if np.isinf(values).any():
            raise CandidateTrainingError(f"Candidate feature {feature!r} contains infinite values.")
    return dataset_dir, frame, metadata, validation, split


def _make_train_validation(frame: pd.DataFrame, split: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    train_info, validation_info, test_info = split.get("train"), split.get("validation"), split.get("test")
    if not all(isinstance(item, Mapping) for item in (train_info, validation_info, test_info)):
        raise CandidateTrainingError("Dataset split summary is missing train/validation/test boundaries.")
    train_start = train_info.get("start_step")
    train_end = train_info.get("end_step")
    validation_start = validation_info.get("start_step")
    validation_end = validation_info.get("end_step")
    test_start = test_info.get("start_step")
    test_end = test_info.get("end_step")
    if not (train_start <= train_end < validation_start <= validation_end < test_start <= test_end):
        raise CandidateTrainingError("Dataset split boundaries are missing, invalid, or not chronological.")
    train = frame.loc[frame["step"].between(train_start, train_end)].copy().reset_index(drop=True)
    validation = frame.loc[frame["step"].between(validation_start, validation_end)].copy().reset_index(drop=True)
    expected_train = _require_count(train_info.get("rows"), "train rows")
    expected_validation = _require_count(validation_info.get("rows"), "validation rows")
    expected_test = _require_count(test_info.get("rows"), "test rows")
    if len(train) != expected_train or len(validation) != expected_validation:
        raise CandidateTrainingError("Train/validation row counts do not match the saved split summary.")
    if len(frame) != expected_train + expected_validation + expected_test:
        raise CandidateTrainingError("Train/validation/test counts do not sum to the dataset row count.")
    if train.empty or validation.empty or expected_test < 1:
        raise CandidateTrainingError("Every chronological split must contain rows; test remains reserved.")
    if train["step"].max() >= validation["step"].min():
        raise CandidateTrainingError("Train and validation step ranges overlap.")
    boundaries = {
        "train_start_step": int(train["step"].min()),
        "train_end_step": int(train["step"].max()),
        "validation_start_step": int(validation["step"].min()),
        "validation_end_step": int(validation["step"].max()),
        "test_start_step": int(test_start),
        "test_end_step": int(test_end),
        "train_end_step": int(train["step"].max()),
    }
    return train, validation, boundaries


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def train_candidate(*, run_id: str | None, run_root: str | Path = RUNS_ROOT) -> dict[str, Any]:
    """Revalidate Stage 4 inputs, train via Phase 11, and save a new candidate bundle.

    The test partition is never passed to Phase 11 training, threshold selection,
    prediction, or metrics; only its saved boundaries/counts are recorded.
    """
    safe_id, run_dir = _validate_run(run_id, run_root)
    dataset_dir, frame, dataset_metadata, dataset_validation, split = _load_candidate_dataset(safe_id, run_root)
    dataset_eligibility = dataset_metadata.get("eligibility")
    if not isinstance(dataset_eligibility, Mapping) or dataset_eligibility.get("decision_status") != "ELIGIBLE":
        raise CandidateTrainingError("Stage 4 dataset does not record an ELIGIBLE decision.")

    current = eligibility.evaluate_retraining_eligibility()
    if not isinstance(current, Mapping) or current.get("decision_status") != "ELIGIBLE" or current.get("eligible") is not True:
        raise CandidateTrainingError("Fresh eligibility revalidation is not ELIGIBLE; candidate training is blocked.")
    for key in ("active_model_version", "drift_status", "rule_version"):
        if dataset_eligibility.get(key) != current.get(key):
            raise CandidateTrainingError(f"Stage 4 eligibility snapshot is stale or mismatched for {key}.")

    if dataset_metadata.get("dataset_fingerprint_sha256") != _fingerprint(frame):
        raise CandidateTrainingError("Dataset fingerprint changed after Stage 4 preparation.")
    label_summary = label_adapter.get_label_readiness_summary()
    event_id_hash = hashlib.sha256(
        pd.util.hash_pandas_object(frame["event_id"], index=False).to_numpy().tobytes()
    ).hexdigest()
    provenance = _validate_label_provenance(
        dataset_metadata.get("label_provenance"), label_summary, len(frame), event_id_hash
    )

    candidate_dir = resolve_run_path(safe_id, "candidate", run_root=run_root)
    if candidate_dir.exists():
        raise CandidateTrainingError(f"Candidate directory already exists; refusing reuse: {candidate_dir}")
    if dataset_dir.parent != run_dir or candidate_dir.parent != run_dir:
        raise CandidateTrainingError("Dataset/candidate paths must be direct children of this run directory.")

    train_df, validation_df, boundaries = _make_train_validation(frame, split)
    v2_trainer, v2_evaluator = _load_phase11_components()
    # The test rows are intentionally not passed into Phase 11 validation,
    # fitting, threshold selection, prediction, or metric functions.
    try:
        v2_trainer.validate_v2_dataset(pd.concat([train_df, validation_df], ignore_index=True))
    except Exception as exc:
        raise CandidateTrainingError(f"Phase 11 dataset contract validation failed: {exc}") from exc

    try:
        v2_metadata = json.loads(v2_config.METADATA_V2_PATH.read_text(encoding="utf-8"))
        base_parameters = v2_metadata["xgboost_parameters"]
        if not isinstance(base_parameters, dict):
            raise ValueError("xgboost_parameters must be an object")
        scale = v2_trainer.compute_training_scale_pos_weight(train_df[TARGET])
        parameters = v2_trainer.build_v2_params(base_parameters, scale, v2_config.RANDOM_SEED)
    except Exception as exc:
        raise CandidateTrainingError(f"Could not prepare Phase 11 training parameters from the frozen V2 metadata: {exc}") from exc

    try:
        model, elapsed = v2_trainer.train_v2(train_df, validation_df, parameters)
        threshold_table = v2_evaluator.threshold_grid_search(model, validation_df)
        selected_threshold, threshold_row = v2_evaluator.select_threshold(threshold_table)
        validation_metrics = v2_evaluator.evaluate_split(model, validation_df, selected_threshold)
    except Exception as exc:
        raise CandidateTrainingError(f"Phase 11 candidate training/validation threshold selection failed: {exc}") from exc
    if not math.isfinite(float(selected_threshold)) or not 0 < float(selected_threshold) < 1:
        raise CandidateTrainingError("Validation-selected candidate threshold must be strictly between 0 and 1.")

    candidate_version = f"CANDIDATE-{safe_id}"
    temporal_split = {
        "strategy": split.get("strategy"),
        **boundaries,
    }
    metadata = {
        "model_name": "Isolated orchestrator candidate",
        "model_version": candidate_version,
        "model_type": "XGBClassifier (binary:logistic)",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_count": len(FINAL_FEATURES),
        "feature_names": list(FINAL_FEATURES),
        "target": TARGET,
        "dataset_fingerprint": dataset_metadata["dataset_fingerprint_sha256"],
        "dataset_fingerprint_algorithm": _FINGERPRINT_ALGORITHM,
        "label_provenance_summary": provenance,
        "temporal_split": temporal_split,
        "training_configuration": parameters,
        "training_seconds": float(elapsed),
        "training_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_metrics": {"status": "NOT_EVALUATED_RESERVED_FOR_LATER_EVALUATION",
                         "rows": split["test"]["rows"]},
        "training_data_policy": "train rows only for fit; validation rows for early stopping and threshold/model selection; test rows reserved",
    }
    threshold_artifact = {
        "selected_threshold": float(selected_threshold),
        "selection_method": v2_config.THRESHOLD_SELECTION_OBJECTIVE,
        "threshold_grid_step": v2_config.THRESHOLD_GRID_STEP,
        "validation_row_count": len(validation_df),
        "validation_selection_row": {key: (float(value) if isinstance(value, (int, float, np.number)) else value)
                                      for key, value in threshold_row.items()},
        "validation_metrics_at_threshold": validation_metrics,
    }
    training_report = {
        "run_id": safe_id,
        "status": "CANDIDATE_TRAINED_NOT_PROMOTED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_version": candidate_version,
        "dataset_fingerprint": dataset_metadata["dataset_fingerprint_sha256"],
        "feature_count": len(FINAL_FEATURES),
        "training": {"rows": len(train_df), "start_step": boundaries["train_start_step"],
                     "end_step": boundaries["train_end_step"], "seconds": float(elapsed)},
        "validation": {"rows": len(validation_df), "start_step": boundaries["validation_start_step"],
                       "end_step": boundaries["validation_end_step"],
                       "selected_threshold": float(selected_threshold),
                       "metrics": validation_metrics},
        "test": {"rows": split["test"]["rows"], "start_step": boundaries["test_start_step"],
                 "end_step": boundaries["test_end_step"], "accessed_for_fit_or_selection": False,
                 "status": "RESERVED_FOR_LATER_EVALUATION"},
        "model_training_parameters": parameters,
        "provenance": provenance,
        "active_model_version_at_training": current.get("active_model_version"),
    }

    if candidate_dir.exists():
        raise CandidateTrainingError(f"Candidate directory appeared before persistence; refusing overwrite: {candidate_dir}")
    stage_dir = Path(tempfile.mkdtemp(prefix=".candidate-stage-", dir=run_dir))
    try:
        model.save_model(str(stage_dir / "model.json"))
        _write_json(stage_dir / "metadata.json", metadata)
        _write_json(stage_dir / "threshold.json", threshold_artifact)
        _write_json(stage_dir / "training_report.json", training_report)
        try:
            bundle = validate_bundle(stage_dir, FINAL_FEATURES)
        except Exception as exc:
            raise CandidateTrainingError(f"Candidate bundle validation failed: {exc}") from exc
        # Publish the complete directory on the same volume. On Windows, rename
        # fails if candidate_dir already exists; the pre-check also gives a clear
        # error before training when a previous candidate is present.
        try:
            os.rename(stage_dir, candidate_dir)
        except FileExistsError as exc:
            raise CandidateTrainingError(f"Candidate directory already exists; refusing overwrite: {candidate_dir}") from exc
        stage_dir = None  # type: ignore[assignment]
        hashes = {name: hashlib.sha256((candidate_dir / name).read_bytes()).hexdigest()
                  for name in ("model.json", "metadata.json", "threshold.json")}
        return {
            "status": "CANDIDATE_TRAINED_NOT_PROMOTED",
            "run_id": safe_id,
            "candidate_version": candidate_version,
            "candidate_dir": str(candidate_dir),
            "artifacts": {name: str(candidate_dir / name) for name in (
                "model.json", "metadata.json", "threshold.json", "training_report.json"
            )},
            "artifact_sha256": hashes,
            "bundle_validation": {"model_version": bundle.model_version,
                                  "feature_count": bundle.feature_count,
                                  "threshold": bundle.threshold},
            "dataset_fingerprint": dataset_metadata["dataset_fingerprint_sha256"],
            "training_rows": len(train_df),
            "validation_rows": len(validation_df),
            "test_rows_reserved": split["test"]["rows"],
            "selected_threshold": float(selected_threshold),
        }
    except CandidateTrainingError:
        raise
    except Exception as exc:
        raise CandidateTrainingError(f"Candidate artifact persistence failed: {type(exc).__name__}: {exc}") from exc
    finally:
        if stage_dir is not None and Path(stage_dir).exists():
            shutil.rmtree(stage_dir, ignore_errors=True)


__all__ = ["CandidateTrainingError", "train_candidate"]
