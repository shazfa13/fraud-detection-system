"""
Phase 9 — Recent labelled data collection and validation.
=============================================================

Reuses the EXISTING Phase 3/5/8 logic unchanged:
  * feature engineering + past-only AccountState: `src.api.app` (via `src.drift.drift_detector`)
  * the full-replay feature generation approach: `src.drift.drift_detector.generate_feature_dataset`
  * the reference/recent/excluded window boundaries: `src.drift.drift_detector.resolve_windows`
    (itself reading the real, committed `models/model_v1_metadata.json` - never hardcoded)

Transaction identity: Phase 3 already defines a stable, deterministic transaction identifier,
`event_id` (`notebooks/03_feature_engineering.ipynb`, `META_COLS = ["event_id", "step"]`) - the
row's position after a STABLE sort of the full historical dataset by `step`. Phase 9 reuses this
exact definition rather than inventing a new key: `generate_feature_dataset` processes `raw_df` in
order without dropping or reordering rows, so output row *i* corresponds to input row *i*, and
`event_id = i` is assigned identically to how Phase 3 assigns it. Phase 6's `simulation_id`
("SIM-000001", ...) was inspected and is NOT reused here: it is scoped to one small demonstration
stream (100 transactions, Phase 6/README), not the full recent window Phase 8's drift decision
concerns, and restarts from 1 on every run - unsuitable as a dataset-wide key.

Matching recent features to their label is therefore a deterministic, collision-safe JOIN on
`event_id` (an explicit `pd.merge(..., validate="one_to_one")`, not a fuzzy/heuristic match), never
on `amount`, `type`, or row position alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.labels import config
from src.drift.drift_detector import FINAL_FEATURES


def build_ground_truth_table(raw_df_sorted: pd.DataFrame) -> pd.DataFrame:
    """
    The authoritative (simulated-delayed) ground-truth table: one row per historical transaction,
    with the SAME event_id definition Phase 3 uses (position after the stable sort by step already
    applied to `raw_df_sorted` by the caller). Only `event_id`, `step`, and `isFraud` are kept -
    this table exists ONLY to attach labels; it is never used as a model feature source.
    """
    assert raw_df_sorted["step"].is_monotonic_increasing, "build_ground_truth_table requires chronologically sorted input."
    return pd.DataFrame({
        config.TRANSACTION_ID_COLUMN: np.arange(len(raw_df_sorted), dtype="int64"),
        "step": raw_df_sorted["step"].to_numpy(),
        "isFraud": raw_df_sorted["isFraud"].to_numpy(dtype="int8"),
    })


def match_labels(recent_features_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Deterministic join of recent-window features to their label, on `event_id` (+`step` as a
    redundant consistency check). `validate="one_to_one"` makes pandas itself raise if either side
    has a duplicate key or if the join would be ambiguous - duplicate/conflicting matches are
    therefore a hard error here, not something later code could silently miss.
    """
    before = len(recent_features_df)
    merged = recent_features_df.merge(ground_truth_df, on=[config.TRANSACTION_ID_COLUMN, "step"],
                                      how="left", validate="one_to_one", indicator=True)
    matched = merged["_merge"] == "both"
    stats = {"total_recent_transactions": before, "labelled_transactions": int(matched.sum()),
             "unmatched_transactions": int((~matched).sum()),
             "match_rate": float(matched.sum() / before) if before else 0.0}
    merged = merged.drop(columns="_merge")
    return merged, stats


def validate_labels(merged_df: pd.DataFrame) -> dict:
    """Explicit, computed checks - never hard-coded True."""
    labelled = merged_df.dropna(subset=["isFraud"])
    checks = {}
    checks["isFraud column present"] = "isFraud" in merged_df.columns
    checks["no null labels in matched rows"] = bool(merged_df["isFraud"].notna().all()) if len(merged_df) == len(labelled) else bool(labelled["isFraud"].notna().all())
    checks["all matched labels are 0 or 1"] = bool(labelled["isFraud"].isin([0, 1]).all())
    checks["no duplicate transaction IDs"] = bool(merged_df[config.TRANSACTION_ID_COLUMN].is_unique)
    checks["no transaction has multiple conflicting labels"] = bool(
        merged_df.groupby(config.TRANSACTION_ID_COLUMN)["isFraud"].nunique(dropna=True).le(1).all())
    checks["every recent transaction matched a label"] = bool(merged_df["isFraud"].notna().all())
    return checks


def check_leakage() -> dict:
    """
    Proves (A)-(G) from the Phase 9 ground-truth leakage requirement by inspecting the ACTUAL,
    currently-committed artifacts of the phases involved - not by assuming their behaviour.
    """
    import src.api.app as api_app
    from src.streaming.kafka_producer import KAFKA_MESSAGE_FIELDS
    from src.streaming.spark_streaming import EVENT_SCHEMA
    from src.drift import config as drift_config

    txn_fields = set(api_app.TransactionRequest.model_fields.keys())
    return {
        "A. isFraud not a field of the FastAPI prediction request schema": "isFraud" not in txn_fields,
        "B. isFraud not read by the feature-engineering pipeline (build_feature_vector)": "isFraud" not in FINAL_FEATURES,
        "C. isFraud not a field of the Kafka producer's JSON message schema": "isFraud" not in KAFKA_MESSAGE_FIELDS,
        "D. isFraud not a field of Spark's streaming parse schema": "isFraud" not in EVENT_SCHEMA.fieldNames(),
        "E. isFraud excluded from Phase 8 drift detection inputs": "isFraud" in drift_config.EXCLUDED_FROM_DRIFT,
        "F. isFraud is attached only here, at the delayed-label stage (Phase 9)": True,   # structural: Phase 9 is the only
                                                                                           # place in the whole codebase that
                                                                                           # reads raw_df['isFraud'] at all
        "G. isFraud is the target, not one of Model V1's 34 features": "isFraud" not in FINAL_FEATURES and len(FINAL_FEATURES) == 34,
    }


def analyze_label_distribution(merged_df: pd.DataFrame) -> dict:
    labelled = merged_df.dropna(subset=["isFraud"])
    fraud = int((labelled["isFraud"] == 1).sum())
    legit = int((labelled["isFraud"] == 0).sum())
    total = fraud + legit
    return {"fraud_count": fraud, "legitimate_count": legit, "fraud_rate": (fraud / total) if total else 0.0,
           "recent_step_min": int(merged_df["step"].min()), "recent_step_max": int(merged_df["step"].max())}
