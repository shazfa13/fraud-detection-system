"""Create-only, isolated candidate dataset preparation for an eligible run.

This stage accepts a caller-supplied in-memory frame only after fresh eligibility,
verified-label, provenance, and transaction identity checks. It never reads the
Phase 10 dataset or writes outside the run's ``dataset`` directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.orchestrator import eligibility, label_adapter
from src.orchestrator.config import (
    RUNS_ROOT,
    OrchestratorPathError,
    resolve_run_dir,
    resolve_run_path,
    validate_run_id,
)
from src.retraining.cycle_pipeline import DEFAULT_CYCLE_CONFIG, FINAL_FEATURES


TARGET_COLUMN = "isFraud"
AUDIT_COLUMNS = ("event_id", "step")
EXPECTED_COLUMNS = [*AUDIT_COLUMNS, *FINAL_FEATURES, TARGET_COLUMN]
SPLIT_STRATEGY = "chronological whole-step groups using cycle_pipeline CycleConfig fractions"


class CandidateDatasetError(ValueError):
    """Raised when candidate data or provenance fails a safety check."""


def _event_id_fingerprint(values: pd.Series) -> str:
    hashed = pd.util.hash_pandas_object(values, index=False).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()


def _dataset_fingerprint(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()


def _validate_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not isinstance(frame, pd.DataFrame):
        raise CandidateDatasetError("candidate_frame must be a pandas DataFrame.")
    if len(FINAL_FEATURES) != 34 or len(set(FINAL_FEATURES)) != 34:
        raise CandidateDatasetError("The canonical model contract must contain exactly 34 unique features.")
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise CandidateDatasetError("Candidate dataset columns/order must be [event_id, step] + the exact 34 features + isFraud.")
    if frame.empty:
        raise CandidateDatasetError("Candidate dataset must contain at least one row.")
    result = frame.copy(deep=True)
    if result["event_id"].isna().any() or not result["event_id"].is_unique:
        raise CandidateDatasetError("event_id values must be present and unique.")
    if not pd.api.types.is_numeric_dtype(result["event_id"]):
        raise CandidateDatasetError("event_id must be numeric under the frozen Phase 3/10 identifier contract.")
    event_ids = result["event_id"].to_numpy(dtype="float64")
    if not np.isfinite(event_ids).all() or not np.equal(event_ids, np.floor(event_ids)).all():
        raise CandidateDatasetError("event_id values must be finite integers.")
    if not pd.api.types.is_numeric_dtype(result["step"]):
        raise CandidateDatasetError("step must be numeric for chronological splitting.")
    steps = result["step"].to_numpy(dtype="float64")
    if not np.isfinite(steps).all():
        raise CandidateDatasetError("step values must be finite.")
    if result[TARGET_COLUMN].isna().any() or not result[TARGET_COLUMN].isin([0, 1]).all():
        raise CandidateDatasetError("isFraud labels must be non-null and binary 0/1.")
    for feature in FINAL_FEATURES:
        if not pd.api.types.is_numeric_dtype(result[feature]):
            raise CandidateDatasetError(f"Feature {feature!r} must be numeric.")
        values = result[feature].to_numpy(dtype="float64", na_value=np.nan)
        if np.isinf(values).any():
            raise CandidateDatasetError(f"Feature {feature!r} contains infinite values.")
    result = result.sort_values(["step", "event_id"], kind="stable").reset_index(drop=True)
    validation = {
        "schema_exact": list(result.columns) == EXPECTED_COLUMNS,
        "feature_count": len(FINAL_FEATURES),
        "feature_names_and_order": list(result.columns[2:-1]) == list(FINAL_FEATURES),
        "audit_columns_separate_from_features": not set(AUDIT_COLUMNS + (TARGET_COLUMN,)).intersection(FINAL_FEATURES),
        "target_separate_from_features": TARGET_COLUMN not in FINAL_FEATURES,
        "event_id_unique_and_non_null": bool(result["event_id"].is_unique and result["event_id"].notna().all()),
        "target_binary_and_non_null": bool(result[TARGET_COLUMN].notna().all() and result[TARGET_COLUMN].isin([0, 1]).all()),
        "features_numeric_and_no_infinity": True,
        "chronologically_sorted": bool(result["step"].is_monotonic_increasing),
        "row_count": int(len(result)),
        "event_id_sha256": _event_id_fingerprint(result["event_id"]),
        "dataset_sha256": _dataset_fingerprint(result),
    }
    return result, validation


def _validate_provenance(provenance: Mapping[str, Any], label_summary: Mapping[str, Any],
                         frame: pd.DataFrame) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise CandidateDatasetError("Verified label provenance is required.")
    source_type = provenance.get("source_type")
    source = provenance.get("source")
    label_provenance = provenance.get("label_provenance")
    if source_type != "trusted_verified" or provenance.get("source_verified") is not True:
        raise CandidateDatasetError("Labels must come from a genuinely verified trusted_verified source.")
    if provenance.get("synthetic") is not False:
        raise CandidateDatasetError("Provenance must explicitly establish that labels are non-synthetic.")
    if not isinstance(source, str) or not source.strip() or not isinstance(label_provenance, str) or not label_provenance.strip():
        raise CandidateDatasetError("Label source and label_provenance must be present and non-empty.")
    type_text = " ".join(str(provenance.get(key, "")) for key in (
        "source", "label_provenance", "label_type", "ground_truth_label_type",
        "phase9_label_type", "dataset"
    )).casefold()
    if (provenance.get("ground_truth_label_type") == "SIMULATED_DELAYED_GROUND_TRUTH"
            or provenance.get("label_type") == "SIMULATED_DELAYED_GROUND_TRUTH"
            or provenance.get("phase9_label_type") == "SIMULATED_DELAYED_GROUND_TRUTH"
            or provenance.get("synthetic") is True
            or "paysim" in type_text or "simulat" in type_text or "synthetic" in type_text):
        raise CandidateDatasetError("Simulated PaySim/synthetic labels cannot be used as verified ground truth.")

    matching_key = provenance.get("matching_key")
    if matching_key != "event_id" or provenance.get("transaction_event_id_mapping_verified") is not True:
        raise CandidateDatasetError("A verified transaction_id-to-event_id mapping is required.")
    if provenance.get("matched_event_ids_sha256") != _event_id_fingerprint(frame["event_id"]):
        raise CandidateDatasetError("Provenance event-ID fingerprint does not match the candidate rows.")
    matched_count = provenance.get("matched_label_count")
    unmatched_count = provenance.get("unmatched_label_count")
    conflict_count = provenance.get("conflict_count")
    if isinstance(matched_count, bool) or not isinstance(matched_count, Integral) or matched_count != len(frame):
        raise CandidateDatasetError("Matched-label count must equal the candidate row count.")
    if isinstance(unmatched_count, bool) or not isinstance(unmatched_count, Integral) or unmatched_count != 0:
        raise CandidateDatasetError("Unmatched labels block candidate dataset creation.")
    if isinstance(conflict_count, bool) or not isinstance(conflict_count, Integral) or conflict_count != 0:
        raise CandidateDatasetError("Label or prediction conflicts block candidate dataset creation.")

    if label_summary.get("status") != "AVAILABLE" or label_summary.get("strict_retraining_ready") is not True:
        raise CandidateDatasetError("Current label adapter does not confirm strict retraining readiness.")
    if label_summary.get("matching_key") != "event_id":
        raise CandidateDatasetError("Current label store does not confirm lifecycle-required event_id matching.")
    sources = label_summary.get("trusted_source_configuration", {}).get("configured_sources", [])
    if source not in sources:
        raise CandidateDatasetError("Provenance source is not present in the configured trusted-source allowlist.")
    if label_summary.get("trusted_source_configuration", {}).get("authorization_token_configured") is not True:
        raise CandidateDatasetError("Trusted-source authorization is not currently configured.")
    verified_count = label_summary.get("genuinely_verified_labels")
    if isinstance(verified_count, bool) or not isinstance(verified_count, Integral) or verified_count < len(frame):
        raise CandidateDatasetError("Current verified-label count is insufficient for the candidate rows.")
    for field in ("unmatched_labels", "conflict_count", "unresolved_or_rejected_count"):
        value = label_summary.get(field)
        if isinstance(value, bool) or not isinstance(value, Integral) or value != 0:
            raise CandidateDatasetError(f"Current label summary requires a known zero {field} count.")
    return {
        "source": source,
        "source_type": source_type,
        "label_provenance": label_provenance,
        "synthetic": False,
        "matching_key": matching_key,
        "transaction_event_id_mapping_verified": True,
        "matched_label_count": int(matched_count),
        "unmatched_label_count": 0,
        "conflict_count": 0,
        "matched_event_ids_sha256": provenance["matched_event_ids_sha256"],
    }


def _chronological_split(frame: pd.DataFrame) -> dict[str, Any]:
    """Split on whole step groups using CycleConfig's 60/20/20 fractions."""
    cycle = DEFAULT_CYCLE_CONFIG
    train_fraction = cycle.recent_train_fraction
    validation_fraction = cycle.recent_validation_fraction
    unique_steps, counts = np.unique(frame["step"].to_numpy(), return_counts=True)
    if len(unique_steps) < 3:
        raise CandidateDatasetError("At least three distinct step values are required for non-empty chronological splits.")
    cumulative = np.cumsum(counts)
    total = len(frame)
    train_index = min(int(np.searchsorted(cumulative, total * train_fraction, side="left")), len(unique_steps) - 3)
    validation_index = max(train_index + 1,
                           min(int(np.searchsorted(cumulative, total * (train_fraction + validation_fraction), side="left")),
                               len(unique_steps) - 2))
    train_end = unique_steps[train_index]
    validation_end = unique_steps[validation_index]
    train = frame.loc[frame["step"] <= train_end].copy().reset_index(drop=True)
    validation = frame.loc[(frame["step"] > train_end) & (frame["step"] <= validation_end)].copy().reset_index(drop=True)
    test = frame.loc[frame["step"] > validation_end].copy().reset_index(drop=True)
    if min(len(train), len(validation), len(test)) == 0:
        raise CandidateDatasetError("Chronological whole-step split produced an empty partition.")
    if not (train["step"].max() < validation["step"].min()
            and validation["step"].max() < test["step"].min()):
        raise CandidateDatasetError("Train, validation, and test step ranges overlap.")
    id_sets = [set(part["event_id"]) for part in (train, validation, test)]
    if any(id_sets[left] & id_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise CandidateDatasetError("event_id overlap exists between chronological partitions.")
    return {
        "strategy": SPLIT_STRATEGY,
        "fractions": {"train": train_fraction, "validation": validation_fraction,
                      "test": cycle.recent_test_fraction},
        "train": {"rows": len(train), "start_step": train["step"].min().item(), "end_step": train["step"].max().item()},
        "validation": {"rows": len(validation), "start_step": validation["step"].min().item(), "end_step": validation["step"].max().item()},
        "test": {"rows": len(test), "start_step": test["step"].min().item(), "end_step": test["step"].max().item()},
        "step_ranges_disjoint": True,
        "event_id_sets_disjoint": True,
    }


def _write_create_only_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(dict(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise CandidateDatasetError(f"Refusing to overwrite existing dataset output: {path}") from exc
    except (OSError, TypeError, ValueError) as exc:
        raise CandidateDatasetError(f"Could not safely write dataset metadata {path}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_create_only_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        frame.to_parquet(temporary, index=False)
        with temporary.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise CandidateDatasetError(f"Refusing to overwrite existing dataset output: {path}") from exc
    except (OSError, ImportError, ValueError) as exc:
        raise CandidateDatasetError(f"Could not safely write candidate parquet {path}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_candidate_dataset(*, run_id: str | None = None,
                              prior_eligibility_decision: Mapping[str, Any] | None,
                              candidate_frame: pd.DataFrame,
                              label_provenance: Mapping[str, Any] | None,
                              run_root: str | Path = RUNS_ROOT) -> dict[str, Any]:
    """Validate current eligibility/provenance, then save isolated candidate data.

    This function never derives features or labels from repository datasets. The
    in-memory frame and provenance must be supplied by a future verified source.
    """
    if run_id is None:
        raise CandidateDatasetError("An existing orchestrator run_id is required.")
    try:
        run_id = validate_run_id(run_id)
        run_dir = resolve_run_dir(run_id, run_root)
    except (OrchestratorPathError, TypeError) as exc:
        raise CandidateDatasetError(f"Invalid run ID or run path: {exc}") from exc
    if not run_dir.is_dir():
        raise CandidateDatasetError(f"Orchestrator run directory does not exist: {run_dir}")
    run_record_path = resolve_run_path(run_id, "run.json", run_root=run_root)
    try:
        run_record = json.loads(run_record_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateDatasetError("Existing run.json is unavailable or invalid.") from exc
    if not isinstance(run_record, dict) or run_record.get("run_id") != run_id:
        raise CandidateDatasetError("Run record identity does not match the requested run.")

    if not isinstance(prior_eligibility_decision, Mapping):
        raise CandidateDatasetError("A prior eligibility decision is required.")
    if prior_eligibility_decision.get("decision_status") != "ELIGIBLE":
        raise CandidateDatasetError("Prior eligibility decision is not ELIGIBLE.")
    if prior_eligibility_decision.get("eligible") is not True:
        raise CandidateDatasetError("Prior decision does not contain an affirmative eligibility result.")

    current_decision = eligibility.evaluate_retraining_eligibility()
    if (not isinstance(current_decision, Mapping)
            or current_decision.get("decision_status") != "ELIGIBLE"
            or current_decision.get("eligible") is not True):
        raise CandidateDatasetError("Fresh eligibility revalidation is not ELIGIBLE; candidate dataset creation is blocked.")
    for key in ("active_model_version", "drift_status", "rule_version"):
        if prior_eligibility_decision.get(key) != current_decision.get(key):
            raise CandidateDatasetError(f"Prior eligibility decision is stale or mismatched for {key}.")

    frame, validation = _validate_frame(candidate_frame)
    label_summary = label_adapter.get_label_readiness_summary()
    provenance = _validate_provenance(label_provenance, label_summary, frame)
    split_summary = _chronological_split(frame)

    dataset_dir = resolve_run_path(run_id, "dataset", run_root=run_root)
    outputs = {
        "candidate_dataset": dataset_dir / "candidate_dataset.parquet",
        "dataset_metadata": dataset_dir / "dataset_metadata.json",
        "dataset_validation": dataset_dir / "dataset_validation.json",
        "split_summary": dataset_dir / "split_summary.json",
    }
    if dataset_dir.exists() or any(path.exists() for path in outputs.values()):
        raise CandidateDatasetError(f"Dataset output already exists; refusing reuse: {dataset_dir}")
    dataset_dir.mkdir(exist_ok=False)

    metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint_sha256": validation["dataset_sha256"],
        "row_count": validation["row_count"],
        "audit_columns": list(AUDIT_COLUMNS),
        "feature_count": len(FINAL_FEATURES),
        "feature_names": list(FINAL_FEATURES),
        "target_column": TARGET_COLUMN,
        "label_provenance": provenance,
        "eligibility": {
            "decision_status": current_decision["decision_status"],
            "active_model_version": current_decision.get("active_model_version"),
            "drift_status": current_decision.get("drift_status"),
            "rule_version": current_decision.get("rule_version"),
            "evaluated_at_utc": current_decision.get("evaluated_at_utc"),
        },
    }
    validation_record = {
        "run_id": run_id,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": validation,
        "passed": all(value is True for key, value in validation.items()
                       if isinstance(value, bool)),
    }
    try:
        _write_create_only_parquet(outputs["candidate_dataset"], frame)
        _write_create_only_json(outputs["dataset_metadata"], metadata)
        _write_create_only_json(outputs["dataset_validation"], validation_record)
        _write_create_only_json(outputs["split_summary"], {"run_id": run_id, **split_summary})
    except CandidateDatasetError:
        raise
    return {
        "status": "DATASET_CREATED",
        "run_id": run_id,
        "dataset_dir": str(dataset_dir),
        "outputs": {name: str(path) for name, path in outputs.items()},
        "row_count": len(frame),
        "feature_count": len(FINAL_FEATURES),
        "split_summary": split_summary,
    }


__all__ = ["CandidateDatasetError", "prepare_candidate_dataset"]
