"""Bounded Stage 6 candidate-training demo on historical PaySim data.

This workflow does not invoke Stage 4 eligibility and explicitly marks its
artifacts ineligible for genuine promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from src.retraining import v2_config
from src.retraining.active_model import resolve_active_model
from src.retraining.cycle_pipeline import CycleConfig, save_candidate_bundle
from src.retraining.model_activator import validate_bundle
from src.retraining.stage4_workflow import TARGET, validate_feature_contract
from src.retraining.v2_evaluator import _metrics, select_threshold, threshold_grid_search
from src.retraining.v2_trainer import train_v2

PROJECT_ROOT = v2_config.PROJECT_ROOT
DEFAULT_DATASET = v2_config.V2_DATASET_PATH
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "models" / "candidates" / "stage6_demo"
SEED = 20261025
DEFAULT_SAMPLE_LIMITS = {"train": 60000, "validation": 15000, "evaluation": 15000}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _boundaries() -> dict[str, int]:
    metadata = json.loads(v2_config.METADATA_V2_PATH.read_text(encoding="utf-8"))
    split = metadata.get("train_validation_test_strategy", {})
    result = {key: int(split[key]) for key in ("train_end_step", "validation_end_step", "test_end_step")}
    if not (0 < result["train_end_step"] < result["validation_end_step"] < result["test_end_step"]):
        raise ValueError("V2 temporal split boundaries are invalid.")
    return result


def _masks(steps: np.ndarray, bounds: dict[str, int]) -> dict[str, np.ndarray]:
    return {
        "train": steps <= bounds["train_end_step"],
        "validation": (steps > bounds["train_end_step"]) & (steps <= bounds["validation_end_step"]),
        "evaluation": (steps > bounds["validation_end_step"]) & (steps <= bounds["test_end_step"]),
    }


def _dataset(path: Path, features: list[str]) -> ds.Dataset:
    if not path.is_file():
        raise FileNotFoundError(f"Demonstration dataset does not exist: {path}")
    dataset = ds.dataset(path, format="parquet")
    expected = ["event_id", "step", *features, TARGET]
    if dataset.schema.names != expected:
        raise ValueError("Dataset columns/order differ from the exact V2 contract.")
    if len(features) != 34 or len(set(features)) != 34 or {"event_id", "step", TARGET}.intersection(features):
        raise ValueError("Model input contract must contain 34 ordered features and exclude audit/target fields.")
    for name in features:
        dtype = dataset.schema.field(name).type
        if not (pa.types.is_integer(dtype) or pa.types.is_floating(dtype)):
            raise ValueError(f"Feature {name!r} is not numeric.")
    for name in ("step", TARGET):
        if not pa.types.is_integer(dataset.schema.field(name).type):
            raise ValueError(f"{name} must be an integer parquet column.")
    return dataset


def _inspect(dataset: ds.Dataset, bounds: dict[str, int]) -> dict[str, Any]:
    """Validate labels and reject infinities in bounded Arrow batches."""
    features = [name for name in dataset.schema.names if name not in {"event_id", "step", TARGET}]
    counts = {name: {0: 0, 1: 0, "rows": 0, "minimum_step": None, "maximum_step": None}
              for name in ("train", "validation", "evaluation")}
    for batch in dataset.scanner(columns=dataset.schema.names, batch_size=65536, use_threads=True).to_batches():
        frame = batch.to_pandas()
        if frame["event_id"].isna().any() or (frame["event_id"].astype(str).str.len() == 0).any():
            raise ValueError("Dataset event_id values must be present and non-empty.")
        labels = frame[TARGET]
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError("Dataset target must be non-null binary isFraud.")
        for feature in features:
            values = frame[feature].to_numpy(dtype=np.float64, na_value=np.nan)
            if np.isinf(values).any():
                raise ValueError(f"Feature {feature!r} contains an infinite value.")
        steps = frame["step"].to_numpy(dtype=np.int64)
        y = labels.to_numpy(dtype=np.int64)
        for name, mask in _masks(steps, bounds).items():
            if not mask.any():
                continue
            summary = counts[name]
            summary["rows"] += int(mask.sum())
            summary[0] += int((y[mask] == 0).sum())
            summary[1] += int((y[mask] == 1).sum())
            low, high = int(steps[mask].min()), int(steps[mask].max())
            summary["minimum_step"] = low if summary["minimum_step"] is None else min(summary["minimum_step"], low)
            summary["maximum_step"] = high if summary["maximum_step"] is None else max(summary["maximum_step"], high)
    for name, summary in counts.items():
        if summary[0] == 0 or summary[1] == 0:
            raise ValueError(f"{name} split has no examples from one of the two classes.")
    return counts


def _quotas(class_counts: dict[int, int], limit: int) -> dict[int, int]:
    total = sum(class_counts.values())
    if limit < 4 or min(class_counts.values()) <= 0:
        raise ValueError("Every split needs both classes and a sample limit >= 4.")
    size = min(total, limit)
    result = {label: min(class_counts[label], max(1, int(size * class_counts[label] / total)))
              for label in (0, 1)}
    while sum(result.values()) > size:
        candidates = [label for label in (0, 1) if result[label] > 1]
        label = max(candidates, key=lambda item: result[item] / class_counts[item])
        result[label] -= 1
    while sum(result.values()) < size:
        candidates = [label for label in (0, 1) if result[label] < class_counts[label]]
        if not candidates:
            break
        label = max(candidates, key=lambda item: size * class_counts[item] / total - result[item])
        result[label] += 1
    return result


def _sample(dataset: ds.Dataset, bounds: dict[str, int], limits: dict[str, int],
            seed: int, features: list[str]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Use deterministic per-split/per-class uniform reservoir samples."""
    populations = _inspect(dataset, bounds)
    quotas = {name: _quotas({0: populations[name][0], 1: populations[name][1]}, limits[name])
              for name in populations}
    rng = np.random.default_rng(seed)
    reservoirs: dict[tuple[str, int], tuple[np.ndarray, pd.DataFrame]] = {}
    columns = ["event_id", "step", *features, TARGET]
    for batch in dataset.scanner(columns=columns, batch_size=65536, use_threads=True).to_batches():
        frame = batch.to_pandas()
        for split, mask in _masks(frame["step"].to_numpy(dtype=np.int64), bounds).items():
            selected = frame.loc[mask]
            for label in (0, 1):
                rows = selected.loc[selected[TARGET] == label]
                if rows.empty:
                    continue
                key = (split, label)
                scores = rng.random(len(rows))
                if key in reservoirs:
                    old_scores, old_rows = reservoirs[key]
                    scores = np.concatenate((old_scores, scores))
                    rows = pd.concat((old_rows, rows), ignore_index=True)
                quota = quotas[split][label]
                if len(rows) > quota:
                    indexes = np.argpartition(scores, quota - 1)[:quota]
                    scores, rows = scores[indexes], rows.iloc[indexes]
                reservoirs[key] = (scores, rows.reset_index(drop=True))
    result = {}
    for split in populations:
        frame = pd.concat([reservoirs[(split, label)][1] for label in (0, 1)], ignore_index=True)
        frame = frame.sort_values(["step", "event_id"], kind="stable").reset_index(drop=True)
        if not frame.event_id.is_unique:
            raise ValueError(f"{split} sample contains duplicate source transactions.")
        result[split] = frame[columns]
    id_sets = [set(frame.event_id.astype(str)) for frame in result.values()]
    if any(id_sets[i] & id_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Transaction identity overlaps between chronological splits.")
    info = {"seed": seed, "method": "seeded stratified uniform reservoir without replacement",
        "sample_limits": limits, "population": populations,
        "sampled": {name: {"rows": len(frame), "fraud_count": int(frame[TARGET].sum()),
            "legitimate_count": int((frame[TARGET] == 0).sum()),
            "minimum_step": int(frame.step.min()), "maximum_step": int(frame.step.max())}
            for name, frame in result.items()}}
    return result, info


def preflight(*, dataset_path: str | Path = DEFAULT_DATASET,
              output_root: str | Path = DEFAULT_OUTPUT_ROOT,
              sample_limits: dict[str, int] | None = None) -> dict[str, Any]:
    """Read-only data/schema/split/dependency/output checks; never trains or writes."""
    features = validate_feature_contract()
    path, output = Path(dataset_path), Path(output_root)
    dataset = _dataset(path, features)
    bounds = _boundaries()
    coverage = _inspect(dataset, bounds)
    selected_limits = sample_limits or DEFAULT_SAMPLE_LIMITS
    if (set(selected_limits) != {"train", "validation", "evaluation"}
            or any(int(value) < 4 for value in selected_limits.values())):
        raise ValueError("sample_limits must specify train, validation, and evaluation.")
    if output.exists() and (not output.is_dir() or not os.access(output, os.W_OK)):
        raise OSError(f"Output root must be a writable directory: {output}")
    if not output.is_dir():
        ancestor = output.parent
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        if not ancestor.is_dir() or not os.access(ancestor, os.W_OK):
            raise OSError(f"Output root has no existing writable ancestor: {output}")
    return {"stage": "Stage 6 demonstration preflight", "status": "preflight_passed",
        "training_started": False, "dataset_path": str(path.resolve()),
        "dataset_rows": int(dataset.count_rows()), "schema": dataset.schema.names,
        "feature_count": len(features), "feature_names": features, "target": TARGET,
        "boundaries": bounds, "class_coverage_by_split": coverage,
        "sample_limits": selected_limits,
        "output_root": str(output.resolve()),
        "dataset_origin": "existing historical PaySim data with simulated labels"}


def run_demo(*, dataset_path: str | Path = DEFAULT_DATASET,
             output_root: str | Path = DEFAULT_OUTPUT_ROOT,
             sample_limits: dict[str, int] | None = None,
             seed: int = SEED,
             training_params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Train and evaluate one independent demo candidate; never promote."""
    limits = dict(sample_limits or DEFAULT_SAMPLE_LIMITS)
    if set(limits) != {"train", "validation", "evaluation"} or any(int(v) < 4 for v in limits.values()):
        raise ValueError("sample_limits must specify each split with at least four rows.")
    pre = preflight(dataset_path=dataset_path, output_root=output_root, sample_limits=limits)
    features = pre["feature_names"]
    dataset = _dataset(Path(dataset_path), features)
    splits, sample_info = _sample(dataset, pre["boundaries"], limits, seed, features)
    for frame in splits.values():
        matrix = frame[features].to_numpy(dtype=np.float64, na_value=np.nan)
        if np.isinf(matrix).any() or frame[TARGET].isna().any() or not frame[TARGET].isin([0, 1]).all():
            raise ValueError("Sampled rows contain invalid target or non-finite feature values.")

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    candidate_dir = output / run_id
    candidate_dir.mkdir(exist_ok=False)

    active = resolve_active_model(project_root=PROJECT_ROOT)
    source_hash = _sha256_file(Path(dataset_path))
    ids = {name: frame.event_id.astype(str).tolist() for name, frame in splits.items()}
    data_fingerprint = _sha256_json({"source_sha256": source_hash, "seed": seed,
                                    "sampled_event_ids": ids, "sampling": sample_info["method"]})
    contract_fingerprint = _sha256_json(features)
    v2_metadata = json.loads(v2_config.METADATA_V2_PATH.read_text(encoding="utf-8"))
    params = dict(training_params or v2_metadata["xgboost_parameters"])
    positives = int(splits["train"][TARGET].sum())
    negatives = len(splits["train"]) - positives
    configured_jobs = int(params.get("n_jobs", 2))
    params.update({"scale_pos_weight": negatives / positives, "random_state": seed,
                   "n_estimators": min(int(params.get("n_estimators", 120)), 120),
                   "early_stopping_rounds": min(int(params.get("early_stopping_rounds", 20)), 20),
                   "n_jobs": 2 if configured_jobs <= 0 else min(configured_jobs, 2)})

    started = time.perf_counter()
    model, _trainer_seconds = train_v2(splits["train"], splits["validation"], params)
    training_seconds = time.perf_counter() - started
    threshold, threshold_metrics = select_threshold(threshold_grid_search(model, splits["validation"]))
    evaluation = splits["evaluation"]
    candidate_probabilities = np.asarray(model.predict_proba(evaluation[features]))[:, 1]
    candidate_metrics = _metrics(evaluation[TARGET].to_numpy(dtype=int), candidate_probabilities, threshold)

    comparison: dict[str, Any]
    if active.model_version.upper() == "V2" and list(active.feature_names) == features:
        from xgboost import XGBClassifier
        active_model = XGBClassifier()
        active_model.load_model(str(active.model_path))
        active_probabilities = np.asarray(active_model.predict_proba(evaluation[features]))[:, 1]
        comparison = {"status": "same_rows_same_features",
            "active_model_version": active.model_version, "active_threshold": active.threshold,
            "active_metrics": _metrics(evaluation[TARGET].to_numpy(dtype=int), active_probabilities, active.threshold),
            "candidate_metrics": candidate_metrics,
            "limitation": "Historical PaySim held-out metrics measure this sample only, not real-world adaptation."}
    else:
        comparison = {"status": "not_comparable",
            "reason": "Active model is not V2 with the same ordered feature contract."}

    sampled = sample_info["sampled"]
    split_record = {key: {name: value for name, value in item.items()}
                    for key, item in sampled.items()}
    model_version = f"STAGE6-DEMO-{run_id}"
    metadata = {"model_name": "Stage 6 demonstration candidate", "model_version": model_version,
        "model_type": "XGBClassifier (binary:logistic)", "target": TARGET,
        "feature_count": 34, "feature_names": features, "selected_classification_threshold": threshold,
        "training_mode": "demonstration", "eligible_for_genuine_promotion": False,
        "data_origin": "historical_paysim_synthetic_labels", "dataset_fingerprint": data_fingerprint,
        "feature_contract_fingerprint": contract_fingerprint,
        "label_provenance_summary": {"source": str(Path(dataset_path).resolve()),
            "label_origin": "historical PaySim simulated labels", "trusted_verified": False,
            "genuine_adaptive_eligibility": False},
        "temporal_split": {"method": "V2 metadata chronological windows, deterministic stratified reservoir subset",
            "train_end_step": pre["boundaries"]["train_end_step"],
            "validation_end_step": pre["boundaries"]["validation_end_step"],
            "test_end_step": pre["boundaries"]["test_end_step"], "sampled_counts": split_record},
        "training_configuration": {"seed": seed, "sample_limits": limits,
            "sampling_method": sample_info["method"], "hyperparameters": params,
            "threshold_selection": "maximum F1 on validation data only"},
        "training_seconds": training_seconds, "test_metrics": candidate_metrics,
        "evaluation_metrics": {"candidate": candidate_metrics}}
    threshold_record = {"selected_threshold": threshold,
        "selection_objective": "maximum F1 on validation split only",
        "validation_metrics": threshold_metrics}
    artifact_hashes = save_candidate_bundle(model, metadata, threshold_record,
        CycleConfig(candidate_output_dir=candidate_dir), active_dir=PROJECT_ROOT / "models" / "active")
    report = {"stage": "Stage 6 controlled retraining demonstration",
        "status": "candidate_trained_not_promoted", "training_started": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_dir": str(candidate_dir.resolve()), "model_version": model_version,
        "training_mode": "demonstration", "eligible_for_genuine_promotion": False,
        "dataset": {"path": str(Path(dataset_path).resolve()), "source_sha256": source_hash,
            "dataset_fingerprint": data_fingerprint, "source_rows": pre["dataset_rows"],
            "origin": metadata["data_origin"]},
        "feature_contract": {"feature_count": len(features), "feature_names": features,
            "sha256": contract_fingerprint, "audit_fields_excluded": ["event_id", "step", "isFraud"]},
        "boundaries": pre["boundaries"], "sampling": sample_info,
        "training_seconds": training_seconds, "hyperparameters": params,
        "threshold_selection": threshold_record, "candidate_evaluation_metrics": candidate_metrics,
        "same_rows_v2_comparison": comparison, "artifact_sha256": artifact_hashes,
        "promotion": {"status": "not_attempted",
            "reason": "Stage 6 demonstration artifacts are not eligible for genuine promotion."}}
    report_path = candidate_dir / "training_report.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    validated = validate_bundle(candidate_dir, features)
    report["artifact_sha256"] = dict(validated.sha256)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preflight", action="store_true", help="run read-only checks; default without --train")
    parser.add_argument("--train", action="store_true", help="train a distinct demonstration candidate")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-rows", type=int, default=DEFAULT_SAMPLE_LIMITS["train"])
    parser.add_argument("--validation-rows", type=int, default=DEFAULT_SAMPLE_LIMITS["validation"])
    parser.add_argument("--evaluation-rows", type=int, default=DEFAULT_SAMPLE_LIMITS["evaluation"])
    args = parser.parse_args()
    try:
        args.output_root.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        parser.error("--output-root must be inside the project directory.")
    limits = {"train": args.train_rows, "validation": args.validation_rows, "evaluation": args.evaluation_rows}
    result = run_demo(dataset_path=args.dataset, output_root=args.output_root, sample_limits=limits, seed=args.seed) \
        if args.train else preflight(dataset_path=args.dataset, output_root=args.output_root, sample_limits=limits)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
