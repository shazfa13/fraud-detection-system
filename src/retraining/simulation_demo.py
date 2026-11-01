"""Run an isolated, PaySim-only V2 retraining demonstration.

This script deliberately bypasses the trusted-label retraining worker. PaySim's
``isFraud`` values are simulated labels, so this run is educational evidence
only and must never be treated as real-world validation or promoted.

Run from the project root with::

    python -m src.retraining.simulation_demo

All generated files go into a new ``reports/simulation_demo/<run-id>/`` folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.retraining import v2_config, v2_evaluator, v2_trainer


PROJECT_ROOT = v2_config.PROJECT_ROOT
DEFAULT_DATASET = v2_config.V2_DATASET_PATH
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reports" / "simulation_demo"
EXPECTED_SPLIT_BOUNDS = {
    "train": (1, 422),
    "validation": (423, 565),
    "test": (566, 743),
}
TARGET = v2_trainer.TARGET


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading the file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_input_dataset(frame: pd.DataFrame) -> None:
    """Check the frozen feature contract, labels, and chronological key."""
    try:
        v2_trainer.validate_v2_dataset(frame)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Input dataset does not satisfy the exact V2 contract: {exc}") from exc

    if frame.empty:
        raise ValueError("Input V2 dataset is empty.")
    if "step" not in frame or not pd.api.types.is_numeric_dtype(frame["step"]):
        raise ValueError("Input V2 dataset requires numeric step values.")
    steps = frame["step"].to_numpy(dtype=float)
    if not np.isfinite(steps).all() or not np.equal(steps, np.floor(steps)).all():
        raise ValueError("Input V2 dataset step values must be finite integers.")


def _partition_frame(split: dict[str, Any], name: str) -> pd.DataFrame:
    key = {"train": "train_df", "validation": "val_df", "test": "test_df"}[name]
    frame = split.get(key)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"V2 temporal split is missing a non-empty {name} DataFrame ({key}).")
    return frame


def audit_split(split: dict[str, Any], input_rows: int) -> dict[str, Any]:
    """Require the documented boundaries, full coverage, class presence, and ID separation."""
    frames = {name: _partition_frame(split, name) for name in EXPECTED_SPLIT_BOUNDS}
    actual_bounds: dict[str, tuple[int, int]] = {}
    for name, frame in frames.items():
        actual_bounds[name] = (int(frame["step"].min()), int(frame["step"].max()))
        if actual_bounds[name] != EXPECTED_SPLIT_BOUNDS[name]:
            expected = EXPECTED_SPLIT_BOUNDS[name]
            raise ValueError(
                f"Unexpected {name} step range {actual_bounds[name]}; "
                f"expected {expected[0]}-{expected[1]} from the Phase 11 split."
            )
        if frame[TARGET].nunique() != 2:
            raise ValueError(f"The {name} split must contain both simulated classes (0 and 1).")
        if not frame["event_id"].is_unique:
            raise ValueError(f"The {name} split contains duplicate event_id values.")

    ids = {name: pd.Index(frame["event_id"]) for name, frame in frames.items()}
    overlap_counts = {
        "train_validation": int(ids["train"].intersection(ids["validation"]).size),
        "train_test": int(ids["train"].intersection(ids["test"]).size),
        "validation_test": int(ids["validation"].intersection(ids["test"]).size),
    }
    if any(overlap_counts.values()):
        raise ValueError(f"event_id values cross candidate partitions: {overlap_counts}")

    assigned_rows = sum(len(frame) for frame in frames.values())
    if assigned_rows > input_rows:
        raise ValueError("The temporal split contains more rows than its input dataset.")
    excluded_rows = input_rows - assigned_rows
    if excluded_rows:
        raise ValueError(f"The V2 temporal split left {excluded_rows} input rows unassigned.")

    summaries: dict[str, Any] = {}
    for name, frame in frames.items():
        fraud_count = int(frame[TARGET].sum())
        rows = int(len(frame))
        summaries[name] = {
            "start_step": actual_bounds[name][0],
            "end_step": actual_bounds[name][1],
            "rows": rows,
            "fraud_count": fraud_count,
            "legitimate_count": rows - fraud_count,
            "fraud_rate": fraud_count / rows,
        }

    return {
        "partitions": summaries,
        "event_id_overlap_counts": overlap_counts,
        "input_rows": int(input_rows),
        "assigned_rows": int(assigned_rows),
        "excluded_from_candidate_partitions": int(excluded_rows),
    }


def _new_run_directory(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_root / run_id
    run_dir.mkdir(exist_ok=False)
    return run_dir


def run_demo(dataset_path: Path = DEFAULT_DATASET) -> dict[str, Any]:
    """Train and evaluate one simulation-only candidate; never writes active artifacts."""
    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Simulation input dataset was not found: {dataset_path}")

    try:
        frame = pd.read_parquet(dataset_path)
    except Exception as exc:
        raise RuntimeError(f"Could not read simulation input Parquet {dataset_path}: {exc}") from exc
    validate_input_dataset(frame)

    try:
        split = v2_trainer.build_v2_temporal_split(frame)
    except Exception as exc:
        raise RuntimeError(f"Could not build the documented V2 chronological split: {exc}") from exc
    split_audit = audit_split(split, len(frame))
    train_df, validation_df, test_df = (
        split["train_df"], split["val_df"], split["test_df"]
    )

    metadata_path = v2_config.METADATA_V1_PATH
    if not metadata_path.is_file():
        raise FileNotFoundError(f"V1 training metadata required by Phase 11 was not found: {metadata_path}")
    v1_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    base_params = v1_metadata.get("xgboost_parameters")
    if not isinstance(base_params, dict) or not base_params:
        raise ValueError(f"V1 metadata has no usable xgboost_parameters: {metadata_path}")
    v1_split = v1_metadata.get("train_validation_test_strategy", {})
    if (not all(key in v1_split for key in ("train_end_step", "validation_end_step"))
            or "validation_rows" not in v1_metadata):
        raise ValueError(f"V1 metadata is missing the documented validation-gap boundaries: {metadata_path}")

    scale_pos_weight = v2_trainer.compute_training_scale_pos_weight(train_df[TARGET])
    params = v2_trainer.build_v2_params(
        base_params, scale_pos_weight, v2_config.RANDOM_SEED
    )

    run_dir = _new_run_directory(DEFAULT_OUTPUT_ROOT)
    report_path = run_dir / "audit_report.json"
    model_path = run_dir / "candidate_model.json"
    dataset_hash = sha256_file(dataset_path)
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        model, training_seconds = v2_trainer.train_v2(train_df, validation_df, params)
        model.save_model(str(model_path))
        model_hash = sha256_file(model_path)

        validation_table = v2_evaluator.threshold_grid_search(model, validation_df)
        selected_threshold, selected_validation_row = v2_evaluator.select_threshold(validation_table)

        # The test split is evaluated exactly once, after model fitting and threshold selection.
        test_metrics = v2_evaluator.evaluate_split(model, test_df, selected_threshold)
    except Exception as exc:
        raise RuntimeError(f"Simulation-only candidate training/evaluation failed: {exc}") from exc

    confusion_matrix = [
        [test_metrics["tn"], test_metrics["fp"]],
        [test_metrics["fn"], test_metrics["tp"]],
    ]
    report = {
        "report_version": 1,
        "status": "simulation_only_candidate_trained_and_evaluated",
        "simulation_only": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_started_at_utc": started_at,
        "training_completion_status": "completed",
        "data_source": {
            "name": "PaySim synthetic financial transaction data",
            "label_source": "Existing isFraud labels from the frozen PaySim dataset; simulated labels, not real-world verified outcomes.",
            "dataset_path": str(dataset_path),
            "dataset_sha256": dataset_hash,
            "rows": int(len(frame)),
            "target": TARGET,
            "feature_count": len(v2_trainer.FINAL_FEATURES),
            "feature_names": list(v2_trainer.FINAL_FEATURES),
        },
        "historical_reuse_warning": (
            "These PaySim records were already used in the project's V1/V2 training, "
            "validation, or evaluation experiments. The test partition was held out from "
            "this demonstration candidate, but was used in earlier project experiments; "
            "this is not a fresh independent evaluation or real-world performance evidence."
        ),
        "split": {
            **split_audit,
            "upstream_phase10_validation_gap": {
                "start_step": int(v1_split["train_end_step"]) + 1,
                "end_step": int(v1_split["validation_end_step"]),
                "rows": int(v1_metadata.get("validation_rows", 0)),
                "reason": "Phase 10 excluded the previous V1 validation window when assembling its V2 training dataset.",
                "note": "This V1 validation window was excluded upstream when the Phase 10 V2 dataset was built; it is not part of the demonstration input.",
            },
        },
        "candidate": {
            "model_type": "XGBClassifier binary:logistic",
            "artifact_path": str(model_path),
            "artifact_sha256": model_hash,
            "configuration": params,
            "random_seed": int(v2_config.RANDOM_SEED),
            "training_seconds": float(training_seconds),
        },
        "threshold_selection": {
            "method": "maximum F1 on the validation split only; tie-break by precision, then lower threshold",
            "selected_threshold": float(selected_threshold),
            "validation_metrics_at_selected_threshold": {
                key: float(selected_validation_row[key])
                for key in ("precision", "recall", "f1")
            },
            "test_used_for_threshold_selection": False,
        },
        "test_evaluation": {
            "evaluation_count": 1,
            "test_used_for_training": False,
            "test_used_for_threshold_selection": False,
            "metrics": {
                key: test_metrics[key]
                for key in ("precision", "recall", "f1", "pr_auc")
            },
            "confusion_matrix_labels": ["legitimate", "fraud"],
            "confusion_matrix": confusion_matrix,
            "counts": {
                key: test_metrics[key]
                for key in ("rows", "fraud_count", "tn", "fp", "fn", "tp")
            },
        },
        "deployment": {
            "active_model_modified": False,
            "promotion_attempted": False,
            "real_retraining_worker_used": False,
        },
        "audit_report_path": str(report_path),
    }
    with report_path.open("x", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, allow_nan=False)
        report_file.write("\n")
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="V2 Parquet input (default: data/processed/v2_retraining_dataset.parquet)")
    args = parser.parse_args()
    try:
        report = run_demo(dataset_path=args.dataset)
    except Exception as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps({
        "status": report["status"],
        "simulation_only": report["simulation_only"],
        "candidate_artifact": report["candidate"]["artifact_path"],
        "audit_report": report["report_path"],
        "test_metrics": report["test_evaluation"]["metrics"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
