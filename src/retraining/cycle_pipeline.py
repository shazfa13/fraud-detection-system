"""Optional configurable helpers for repeatable fraud-model retraining cycles.

The historical Phase 10/11 workflow remains available unchanged. These helpers
take explicit windows, preserve the frozen feature order, split chronologically,
select thresholds from validation data only, and write candidates separately.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.retraining import config as phase10_config
from src.retraining import v2_config
from src.retraining.model_activator import ARTIFACT_NAMES, validate_bundle

_feature_metadata = json.loads(
    (v2_config.PROJECT_ROOT / "models" / "model_v1_metadata.json").read_text(encoding="utf-8")
)
FINAL_FEATURES = _feature_metadata.get("feature_names")
if (_feature_metadata.get("feature_count") != 34 or not isinstance(FINAL_FEATURES, list)
        or len(FINAL_FEATURES) != 34 or len(set(FINAL_FEATURES)) != 34):
    raise ValueError("V1 metadata does not define the frozen 34-feature contract.")


@dataclass(frozen=True)
class StepWindow:
    """Inclusive transaction-step bounds."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("StepWindow requires 0 <= start <= end.")


@dataclass(frozen=True)
class CycleConfig:
    """Data windows and output destination for one explicit retraining cycle."""

    old_window: StepWindow = StepWindow(*phase10_config.EXPECTED_OLD_TRAIN_STEP_RANGE)
    validation_gap: StepWindow = StepWindow(*phase10_config.EXPECTED_VALIDATION_GAP)
    recent_window: StepWindow = StepWindow(*phase10_config.EXPECTED_RECENT_STEP_RANGE)
    recent_train_fraction: float = v2_config.RECENT_TRAIN_EXTENSION_FRACTION
    recent_validation_fraction: float = v2_config.RECENT_VALIDATION_FRACTION
    recent_test_fraction: float = v2_config.RECENT_TEST_FRACTION
    dataset_output_path: Path | None = None
    candidate_output_dir: Path = v2_config.PROJECT_ROOT / "models" / "candidate"

    def __post_init__(self) -> None:
        if self.old_window.end >= self.validation_gap.start:
            raise ValueError("Old window must end before the validation gap.")
        if self.validation_gap.end >= self.recent_window.start:
            raise ValueError("Recent window must start after the validation gap.")
        fractions = (self.recent_train_fraction, self.recent_validation_fraction,
                     self.recent_test_fraction)
        if any(value <= 0 or value >= 1 for value in fractions):
            raise ValueError("Split fractions must be between 0 and 1.")
        if not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
            raise ValueError("Train, validation, and test fractions must sum to 1.")


DEFAULT_CYCLE_CONFIG = CycleConfig()


def _columns() -> list[str]:
    """Return the exact audit + feature + target schema."""
    if (len(FINAL_FEATURES) != 34 or len(set(FINAL_FEATURES)) != 34
            or any(name in FINAL_FEATURES for name in ("event_id", "step", "isFraud"))):
        raise ValueError("The frozen 34-feature contract is invalid or includes audit/target fields.")
    return ["event_id", "step"] + list(FINAL_FEATURES) + [phase10_config.TARGET_COLUMN]


def _select(frame: pd.DataFrame, window: StepWindow, label: str) -> pd.DataFrame:
    """Select one configured window and validate schema, labels, and uniqueness."""
    columns = _columns()
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} input is missing columns: {sorted(missing)}")
    if not pd.api.types.is_numeric_dtype(frame["step"]):
        raise ValueError(f"{label} step column must be numeric.")
    result = frame.loc[frame["step"].between(window.start, window.end), columns].copy()
    if result.empty:
        raise ValueError(f"{label} window {window.start}-{window.end} has no rows.")
    if not result["event_id"].is_unique:
        raise ValueError(f"{label} window contains duplicate event_id values.")
    target = phase10_config.TARGET_COLUMN
    if result[target].isna().any() or not result[target].isin([0, 1]).all():
        raise ValueError(f"{label} labels must be non-null and contain only 0/1.")
    if not all(pd.api.types.is_numeric_dtype(result[name]) for name in FINAL_FEATURES):
        raise ValueError(f"{label} features must be numeric.")
    return result


def build_cycle_dataset(
    old_df: pd.DataFrame,
    recent_df: pd.DataFrame,
    cycle: CycleConfig = DEFAULT_CYCLE_CONFIG,
    *,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Filter configured old/recent windows and combine without leakage or overwrite.

    Gap rows and rows outside both training windows are omitted. Duplicate records
    and overlapping old/recent IDs or steps are rejected. An optional parquet
    destination must be new and is installed from a temporary sibling file.
    """
    old = _select(old_df, cycle.old_window, "Old")
    recent = _select(recent_df, cycle.recent_window, "Recent")
    if set(old.event_id) & set(recent.event_id) or set(old.step) & set(recent.step):
        raise ValueError("Old and recent windows overlap by event_id or step.")
    combined = pd.concat([old, recent], ignore_index=True)
    combined = combined.sort_values(["step", "event_id"], kind="stable").reset_index(drop=True)
    if not combined.event_id.is_unique or combined.duplicated(["event_id", "step"]).any():
        raise ValueError("Combined cycle dataset contains duplicate transaction records.")

    target_value = output_path if output_path is not None else cycle.dataset_output_path
    if target_value is not None:
        target = Path(target_value)
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing cycle dataset: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix=f".{target.name}.", suffix=".tmp",
                                             dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
            combined.to_parquet(temporary, index=False)
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return combined


def _cutoffs(recent: pd.DataFrame, train_fraction: float,
             validation_fraction: float) -> tuple[int, int]:
    """Choose cumulative-row-share boundaries only between whole step groups."""
    counts = recent.groupby("step", sort=True).size().cumsum()
    def cutoff(fraction: float) -> int:
        position = np.searchsorted(counts.to_numpy(), len(recent) * fraction, side="left")
        return int(counts.index[position])
    first, second = cutoff(train_fraction), cutoff(train_fraction + validation_fraction)
    if second <= first:
        raise ValueError("Recent chronological split produced empty or overlapping step ranges.")
    return first, second


def split_cycle_dataset(frame: pd.DataFrame,
                        cycle: CycleConfig = DEFAULT_CYCLE_CONFIG) -> dict[str, Any]:
    """Return ordered train/validation/test frames and step boundaries.

    Any row in the validation gap or outside both configured windows is rejected;
    no rows are silently discarded at this stage.
    """
    if list(frame.columns) != _columns():
        raise ValueError("Cycle dataset columns do not match [event_id, step] + 34 features + isFraud.")
    if not frame.event_id.is_unique:
        raise ValueError("event_id must be unique before splitting.")
    if frame.isFraud.isna().any() or not frame.isFraud.isin([0, 1]).all():
        raise ValueError("isFraud must be non-null and contain only 0/1.")
    ordered = frame.sort_values(["step", "event_id"], kind="stable").reset_index(drop=True)
    old_mask = ordered.step.between(cycle.old_window.start, cycle.old_window.end)
    recent_mask = ordered.step.between(cycle.recent_window.start, cycle.recent_window.end)
    if not (old_mask | recent_mask).all():
        outside = ordered.loc[~(old_mask | recent_mask), "step"].unique().tolist()
        raise ValueError(f"Rows outside configured windows or inside validation gap: {outside[:10]}")
    old, recent = ordered.loc[old_mask], ordered.loc[recent_mask]
    if old.empty or recent.empty:
        raise ValueError("Both old and recent windows must contain data.")
    if set(old.event_id) & set(recent.event_id) or set(old.step) & set(recent.step):
        raise ValueError("Old and recent windows overlap by event_id or step.")
    train_end, validation_end = _cutoffs(
        recent, cycle.recent_train_fraction, cycle.recent_validation_fraction
    )
    train_recent = recent[recent.step <= train_end]
    validation = recent[(recent.step > train_end) & (recent.step <= validation_end)].reset_index(drop=True)
    test = recent[recent.step > validation_end].reset_index(drop=True)
    train = pd.concat([old, train_recent], ignore_index=True).sort_values(
        ["step", "event_id"], kind="stable").reset_index(drop=True)
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Chronological split produced an empty split.")
    if not (train.step.max() < validation.step.min() < test.step.min()
            and validation.step.max() < test.step.min()):
        raise ValueError("Train, validation, and test step ranges must be disjoint.")
    ids = [set(part.event_id) for part in (train, validation, test)]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("event_id overlap detected between splits.")
    return {
        "train_df": train, "val_df": validation, "test_df": test,
        "train_start_step": int(train.step.min()), "train_end_step": int(train.step.max()),
        "validation_start_step": int(validation.step.min()),
        "validation_end_step": int(validation.step.max()),
        "test_start_step": int(test.step.min()), "test_end_step": int(test.step.max()),
        "old_max_step": int(old.step.max()),
    }


def select_validation_threshold(model: Any,
                                validation_df: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    """Use the existing evaluator to tune a threshold on validation data only."""
    from src.retraining.v2_evaluator import select_threshold, threshold_grid_search
    return select_threshold(threshold_grid_search(model, validation_df))


def save_candidate_bundle(model: Any, metadata: dict[str, Any],
                          threshold_info: dict[str, Any],
                          cycle: CycleConfig = DEFAULT_CYCLE_CONFIG, *,
                          active_dir: str | Path | None = None) -> dict[str, str]:
    """Save and validate a candidate bundle without writing into active.

    Existing candidate artifact files are never overwritten. Files are staged and
    validated before per-file atomic replacement. The caller supplies metadata and
    a threshold already selected from validation data.
    """
    destination = Path(cycle.candidate_output_dir)
    active = Path(active_dir) if active_dir is not None else v2_config.ACTIVE_MODEL_DIR
    dest_resolved, active_resolved = destination.resolve(), active.resolve()
    if (dest_resolved == active_resolved
            or active_resolved in dest_resolved.parents
            or dest_resolved in active_resolved.parents):
        raise ValueError("Candidate output must be separate from the active directory.")
    if metadata.get("feature_count") != 34 or metadata.get("feature_names") != FINAL_FEATURES:
        raise ValueError("Candidate metadata must match the exact ordered 34-feature contract.")
    if not isinstance(metadata.get("model_version"), str) or not metadata["model_version"].strip():
        raise ValueError("Candidate metadata must include a non-empty model_version.")
    if metadata.get("target") != "isFraud":
        raise ValueError("Candidate metadata target must be isFraud.")
    if not isinstance(threshold_info, dict):
        raise ValueError("threshold_info must be a JSON object.")
    threshold = threshold_info.get("selected_threshold")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold)) or not 0 <= float(threshold) <= 1):
        raise ValueError("Candidate selected_threshold must be finite and in [0, 1].")
    if ("selected_classification_threshold" in metadata
            and float(metadata["selected_classification_threshold"]) != float(threshold)):
        raise ValueError("Candidate metadata threshold conflicts with threshold_info.")

    destination_created = not destination.exists()
    destination.mkdir(parents=True, exist_ok=True)
    conflicts = [name for name in ARTIFACT_NAMES if (destination / name).exists()]
    if conflicts:
        if destination_created and not any(destination.iterdir()):
            destination.rmdir()
        raise FileExistsError(f"Refusing to overwrite candidate artifacts: {conflicts}")
    stage = Path(tempfile.mkdtemp(prefix=".candidate-stage-", dir=destination.parent))
    created: list[Path] = []
    try:
        model.save_model(str(stage / "model.json"))
        (stage / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        (stage / "threshold.json").write_text(
            json.dumps(threshold_info, indent=2, allow_nan=False), encoding="utf-8")
        validate_bundle(stage, FINAL_FEATURES)
        for name in ARTIFACT_NAMES:
            target = destination / name
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite candidate artifact: {target}")
            os.replace(stage / name, target)
            created.append(target)
        return validate_bundle(destination, FINAL_FEATURES).sha256
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        if destination_created and destination.is_dir() and not any(destination.iterdir()):
            destination.rmdir()
        raise
    finally:
        import shutil
        shutil.rmtree(stage, ignore_errors=True)
