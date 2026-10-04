"""
Phase 10 — Build the V2 retraining dataset (OLD training data + Phase 9 recent labelled data).
====================================================================================================

Reuses, unchanged:
  * `src.drift.drift_detector.resolve_windows` / `generate_feature_dataset` — the SAME continuous
    chronological replay through `src.api.app`'s feature pipeline that Phase 8/9 already use, to
    (re)build OLD training-period features when `data/processed/featured_transactions.parquet`
    (Phase 3's own output) is not available locally.
  * `src.labels.label_collector.build_ground_truth_table` / `match_labels` — the SAME
    event_id-based, `validate="one_to_one"` deterministic label-matching Phase 9 established, used
    here to attach `isFraud` to the OLD training period exactly as Phase 9 attached it to the
    recent period.

Phase 10 does NOT reimplement feature engineering or label matching - it only decides WHICH two
already-validated datasets to put side by side, validates each, and concatenates them
deterministically. It never writes to any Phase 1-9 artifact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.retraining import config
from src.drift.drift_detector import resolve_windows, generate_feature_dataset, FINAL_FEATURES
from src.labels.label_collector import build_ground_truth_table, match_labels


def load_or_build_old_training_data(raw_df_sorted: pd.DataFrame, windows: dict) -> tuple[pd.DataFrame, str]:
    """
    Returns (old_df, source_description). old_df has exactly [event_id, step] + the 34 Model V1
    features + [isFraud], for the reference (OLD training) window only.
    """
    if config.PROCESSED_DATA_PATH.is_file():
        proc = pd.read_parquet(config.PROCESSED_DATA_PATH)
        if {"event_id", "step"}.issubset(proc.columns) and set(FINAL_FEATURES).issubset(proc.columns):
            ref_max = windows["reference_window"]["max_step"]
            old = proc[proc["step"] <= ref_max][["event_id", "step"] + FINAL_FEATURES].reset_index(drop=True)
            ground_truth = build_ground_truth_table(raw_df_sorted)
            old, _ = match_labels(old, ground_truth)
            return old[["event_id", "step"] + FINAL_FEATURES + ["isFraud"]], f"Phase 3 processed dataset ({config.PROCESSED_DATA_PATH.name}), filtered to the reference window"

    feat_df = generate_feature_dataset(raw_df_sorted, windows)
    feat_df["event_id"] = np.arange(len(feat_df), dtype="int64")   # Phase 3's own event_id definition
    old = feat_df[feat_df["window"] == "reference"][["event_id", "step"] + FINAL_FEATURES].reset_index(drop=True)
    ground_truth = build_ground_truth_table(raw_df_sorted)
    old, _ = match_labels(old, ground_truth)
    return old[["event_id", "step"] + FINAL_FEATURES + ["isFraud"]], "generated fresh via the existing Phase 3/5/8/9 pipeline (data/processed/featured_transactions.parquet not found locally - git-ignored)"


def load_recent_labelled_data(path=config.RECENT_LABELLED_DATA_PATH) -> pd.DataFrame:
    """Loads Phase 9's OWN output AS-IS, selecting only [event_id, step] + 34 features + [isFraud].
    Audit-only columns (fraud_probability, predicted_label, risk_score, risk_level) are explicitly
    dropped here - they must never become Model V2 features."""
    df = pd.read_parquet(path)
    missing = ({"event_id", "step", config.TARGET_COLUMN} | set(FINAL_FEATURES)) - set(df.columns)
    if missing:
        raise ValueError(f"Phase 9 recent labelled dataset is missing expected column(s): {missing}")
    return df[["event_id", "step"] + FINAL_FEATURES + [config.TARGET_COLUMN]].copy()


def validate_dataset(df: pd.DataFrame, name: str, expected_step_range: tuple[int, int]) -> dict:
    checks = {
        f"[{name}] 34 features present, exact names": list(df.columns[2:2 + len(FINAL_FEATURES)]) == FINAL_FEATURES,
        f"[{name}] target '{config.TARGET_COLUMN}' present": config.TARGET_COLUMN in df.columns,
        f"[{name}] target not among the 34 features": config.TARGET_COLUMN not in FINAL_FEATURES,
        f"[{name}] no duplicate event_id": bool(df["event_id"].is_unique),
        f"[{name}] no null target values": bool(df[config.TARGET_COLUMN].notna().all()),
        f"[{name}] target values only 0/1": bool(df[config.TARGET_COLUMN].isin([0, 1]).all()),
        f"[{name}] step range within expected bounds": bool(df["step"].min() >= expected_step_range[0] and df["step"].max() <= expected_step_range[1]),
        f"[{name}] feature dtypes numeric": bool(all(pd.api.types.is_numeric_dtype(df[c]) for c in FINAL_FEATURES)),
        f"[{name}] no audit-only columns present": not any(c in df.columns for c in config.AUDIT_ONLY_COLUMNS),
        f"[{name}] no nameOrig/nameDest columns present": not any(c in df.columns for c in ("nameOrig", "nameDest")),
    }
    return checks


def validate_no_overlap(old_df: pd.DataFrame, recent_df: pd.DataFrame) -> dict:
    overlap_ids = set(old_df["event_id"]) & set(recent_df["event_id"])
    overlap_steps = set(old_df["step"].unique()) & set(recent_df["step"].unique())
    return {"no event_id overlap between old and recent": len(overlap_ids) == 0,
           "no step overlap between old and recent": len(overlap_steps) == 0,
           "overlap_event_id_count": len(overlap_ids), "overlap_step_count": len(overlap_steps)}


def validate_gap_excluded(old_df: pd.DataFrame, recent_df: pd.DataFrame, gap: tuple[int, int]) -> dict:
    gap_lo, gap_hi = gap
    return {"old dataset contains no step inside the validation gap": bool(((old_df["step"] < gap_lo) | (old_df["step"] > gap_hi)).all()),
           "recent dataset contains no step inside the validation gap": bool(((recent_df["step"] < gap_lo) | (recent_df["step"] > gap_hi)).all()),
           "old dataset entirely precedes the validation gap": bool(old_df["step"].max() < gap_lo),
           "recent dataset entirely follows the validation gap": bool(recent_df["step"].min() > gap_hi)}


def combine_datasets(old_df: pd.DataFrame, recent_df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic concatenation: OLD then RECENT, re-sorted by (step, event_id). No shuffling,
    no sampling - the two inputs are already each internally chronological and non-overlapping."""
    combined = pd.concat([old_df, recent_df], ignore_index=True)
    combined = combined.sort_values(["step", "event_id"], kind="stable").reset_index(drop=True)
    return combined


def class_distribution(df: pd.DataFrame) -> dict:
    fraud = int((df[config.TARGET_COLUMN] == 1).sum())
    legit = int((df[config.TARGET_COLUMN] == 0).sum())
    total = fraud + legit
    return {"total_rows": total, "fraud_count": fraud, "legitimate_count": legit, "fraud_rate": (fraud / total) if total else 0.0}
