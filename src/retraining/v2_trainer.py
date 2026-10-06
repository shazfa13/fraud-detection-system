"""Dataset validation, chronological splitting, and Model V2 training."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.drift.drift_detector import FINAL_FEATURES
from src.retraining import v2_config as config

TARGET = "isFraud"
AUDIT_COLUMNS = ["event_id", "step"]
EXPECTED_COLUMNS = AUDIT_COLUMNS + FINAL_FEATURES + [TARGET]


def validate_v2_dataset(df: pd.DataFrame) -> None:
    """Raise a clear error if the Phase 10 contract is not exactly preserved."""
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError("Phase 10 dataset columns do not match the exact V2 contract")
    if len(FINAL_FEATURES) != 34:
        raise ValueError("The frozen feature contract must contain exactly 34 features")
    if not df["event_id"].is_unique:
        raise ValueError("event_id must be unique before splitting")
    if df[TARGET].isna().any() or not df[TARGET].isin([0, 1]).all():
        raise ValueError("isFraud must be non-null and contain only 0/1")
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in FINAL_FEATURES):
        raise ValueError("All model features must be numeric")


def _step_cutoffs(recent: pd.DataFrame) -> tuple[int, int]:
    counts = recent.groupby("step", sort=True).size().cumsum()
    total = len(recent)

    def cutoff(fraction: float) -> int:
        target = total * fraction
        return int(counts.index[np.searchsorted(counts.to_numpy(), target, side="left")])

    first = cutoff(config.RECENT_TRAIN_EXTENSION_FRACTION)
    second = cutoff(config.RECENT_TRAIN_EXTENSION_FRACTION + config.RECENT_VALIDATION_FRACTION)
    if second <= first:
        raise ValueError("Recent chronological split produced empty or overlapping step ranges")
    return first, second


def build_v2_temporal_split(df: pd.DataFrame) -> dict[str, Any]:
    """Return OLD + recent-extension training, then step-disjoint validation/test."""
    validate_v2_dataset(df)
    ordered = df.sort_values(["step", "event_id"], kind="stable").reset_index(drop=True)
    old_max = int(ordered.loc[ordered["step"] < 379, "step"].max())
    recent = ordered[ordered["step"] > old_max].copy()
    if recent.empty:
        raise ValueError("The Phase 10 dataset has no recent rows after the old training window")
    train_recent_end, validation_end = _step_cutoffs(recent)
    old = ordered[ordered["step"] <= old_max]
    train_recent = recent[recent["step"] <= train_recent_end]
    validation = recent[(recent["step"] > train_recent_end) & (recent["step"] <= validation_end)]
    test = recent[recent["step"] > validation_end]
    train = pd.concat([old, train_recent], ignore_index=True).sort_values(["step", "event_id"], kind="stable").reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Chronological split produced an empty split")
    return {
        "train_df": train, "val_df": validation, "test_df": test,
        "v2_train_start_step": int(train["step"].min()), "v2_train_end_step": int(train["step"].max()),
        "v2_validation_start_step": int(validation["step"].min()), "v2_validation_end_step": int(validation["step"].max()),
        "v2_test_start_step": int(test["step"].min()), "v2_test_end_step": int(test["step"].max()),
        "old_max_step": old_max,
    }


def compute_training_scale_pos_weight(labels: pd.Series) -> float:
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0:
        raise ValueError("Training split contains no positive examples")
    return negatives / positives


def build_v2_params(v1_params: dict[str, Any], scale_pos_weight: float, random_seed: int) -> dict[str, Any]:
    params = dict(v1_params)
    params["scale_pos_weight"] = float(scale_pos_weight)
    params["random_state"] = random_seed
    params["early_stopping_rounds"] = config.EARLY_STOPPING_ROUNDS
    return params


def train_v2(train_df: pd.DataFrame, val_df: pd.DataFrame, params: dict[str, Any]) -> tuple[XGBClassifier, float]:
    model_params = dict(params)
    model_params.setdefault("eval_metric", "aucpr")
    model_params.setdefault("objective", "binary:logistic")
    model = XGBClassifier(**model_params)
    started = perf_counter()
    model.fit(
        train_df[FINAL_FEATURES], train_df[TARGET],
        eval_set=[(train_df[FINAL_FEATURES], train_df[TARGET]), (val_df[FINAL_FEATURES], val_df[TARGET])],
        verbose=False,
    )
    return model, perf_counter() - started
