"""
Phase 8 — Drift detection: reference distribution vs. recent distribution.
==============================================================================

Reuses the EXISTING Phase 3/5 feature pipeline (imported, unchanged, from `src.api.app`) to build
both the reference and recent feature datasets with exactly the same formulas, frozen constants,
and past-only `AccountState` logic Model V1 was trained and already audited with. Phase 8 does not
reimplement feature engineering, does not retrain anything, and does not use `isFraud`.

Pipeline this module implements:

    raw transactions (chronological, FULL continuous replay - see generate_feature_dataset)
        -> same 34-feature pipeline as Model V1 (src.api.app, unchanged)
        -> split into REFERENCE (training period) and RECENT (test period) by step
        -> per-feature PSI (all eligible features) + KS test (continuous only) /
           two-proportion z-test (binary only)
        -> Benjamini-Hochberg FDR correction across all tested features
        -> feature-level drift decision + documented overall drift rule
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.drift import config
from src.api.app import TransactionRequest, build_feature_vector, FINAL_FEATURES, AccountState


# -------------------------------------------------------------------------------------------
# Window resolution — mirrors src/streaming/kafka_producer.resolve_test_period's approach
# (Phase 4's recorded split is authoritative; a documented fallback if it was not recorded).
# -------------------------------------------------------------------------------------------
def resolve_windows(metadata_path: Path = config.METADATA_PATH) -> dict:
    with open(metadata_path, encoding="utf-8") as fh:
        metadata = json.load(fh)
    strategy = metadata["train_validation_test_strategy"]
    train_end_step = strategy["train_end_step"]
    validation_end_step = strategy["validation_end_step"]
    return {
        "train_end_step": train_end_step, "validation_end_step": validation_end_step,
        "reference_window": {"description": "Training period (the data Model V1 was actually fitted on)",
                             "min_step": 1, "max_step": train_end_step},
        "recent_window": {"description": "Test period (transactions Model V1 never trained on; same period Phase 6/7 use as 'live')",
                          "min_step": validation_end_step + 1, "max_step": None},
        "excluded_window": {"description": "Validation period - deliberately excluded from both windows so they never overlap",
                            "min_step": train_end_step + 1, "max_step": validation_end_step},
    }


# -------------------------------------------------------------------------------------------
# Feature dataset generation — ONE continuous chronological replay (matches how Phase 3 actually
# built Model V1's training data: state is never reset at an arbitrary window boundary, exactly as
# a real deployed system would carry state forward). Rows are tagged by window after the fact.
# -------------------------------------------------------------------------------------------
def generate_feature_dataset(raw_df: pd.DataFrame, windows: dict) -> pd.DataFrame:
    """
    Replay `raw_df` (already sorted by step) through the EXISTING Phase 3/5 feature pipeline, one
    transaction at a time, with a single fresh AccountState for the whole replay. Returns one row
    per transaction: its 34 Model V1 features, its step, and a 'window' label
    ('reference' / 'excluded' / 'recent') from `windows`. isFraud is never read from `raw_df` here.
    """
    assert raw_df["step"].is_monotonic_increasing, "generate_feature_dataset requires chronological input."
    state = AccountState()   # one fresh, isolated state for this replay - does not touch src.api.app's own singleton
    import src.api.app as app_module
    original_state = app_module.account_state
    app_module.account_state = state   # build_feature_vector reads the module-level singleton; swap it for this replay
    try:
        rows = []
        ref_max, rec_min = windows["reference_window"]["max_step"], windows["recent_window"]["min_step"]
        for r in raw_df.itertuples(index=False):
            txn = TransactionRequest(step=int(r.step), type=str(r.type), amount=float(r.amount),
                                     nameOrig=str(r.nameOrig), oldbalanceOrg=float(r.oldbalanceOrg),
                                     nameDest=str(r.nameDest), oldbalanceDest=float(r.oldbalanceDest))
            feats = build_feature_vector(txn).iloc[0].to_dict()
            feats["step"] = int(r.step)
            feats["window"] = "reference" if r.step <= ref_max else ("recent" if r.step >= rec_min else "excluded")
            rows.append(feats)
        return pd.DataFrame(rows)
    finally:
        app_module.account_state = original_state   # never leave the shared singleton swapped out


# -------------------------------------------------------------------------------------------
# Feature typing
# -------------------------------------------------------------------------------------------
def classify_feature_type(reference: pd.Series, recent: pd.Series) -> str:
    """'binary' if every non-NaN value in BOTH windows is in {0, 1}, else 'continuous'."""
    combined = pd.concat([reference.dropna(), recent.dropna()])
    if combined.empty:
        return "continuous"
    return "binary" if set(np.unique(combined.to_numpy())).issubset({0, 1}) else "continuous"


# -------------------------------------------------------------------------------------------
# PSI — safe against empty bins, zero proportions, NaN, inf, constant features, tiny samples.
# -------------------------------------------------------------------------------------------
def calculate_psi(reference: np.ndarray, recent: np.ndarray, feature_type: str,
                  bins: int = config.PSI_BINS, epsilon: float = config.PSI_EPSILON) -> float | None:
    ref = reference[np.isfinite(reference)]
    rec = recent[np.isfinite(recent)]
    if len(ref) < config.MIN_SAMPLE_SIZE or len(rec) < config.MIN_SAMPLE_SIZE:
        return None

    if feature_type == "binary":
        edges = np.array([-0.5, 0.5, 1.5])   # exact bins: {0}, {1}
    else:
        if np.nanstd(ref) == 0:              # constant reference feature - PSI undefined by quantiles
            return 0.0 if np.nanstd(rec) == 0 and np.nanmean(ref) == np.nanmean(rec) else None
        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.quantile(ref, quantiles))
        if len(edges) < 2:
            return None
        edges[0], edges[-1] = -np.inf, np.inf   # catch any recent values outside the reference range

    ref_counts, _ = np.histogram(ref, bins=edges)
    rec_counts, _ = np.histogram(rec, bins=edges)
    ref_prop = np.clip(ref_counts / ref_counts.sum(), epsilon, None)
    rec_prop = np.clip(rec_counts / rec_counts.sum(), epsilon, None)
    return float(np.sum((rec_prop - ref_prop) * np.log(rec_prop / ref_prop)))


# -------------------------------------------------------------------------------------------
# KS test (continuous only) / two-proportion z-test (binary only) - a feature is tested with the
# method appropriate to its type, never both treated identically.
# -------------------------------------------------------------------------------------------
def calculate_ks(reference: np.ndarray, recent: np.ndarray) -> tuple[float, float] | None:
    ref = reference[np.isfinite(reference)]
    rec = recent[np.isfinite(recent)]
    if len(ref) < config.MIN_SAMPLE_SIZE or len(rec) < config.MIN_SAMPLE_SIZE:
        return None
    result = stats.ks_2samp(ref, rec)
    return float(result.statistic), float(result.pvalue)


def calculate_proportion_test(reference: np.ndarray, recent: np.ndarray) -> tuple[float, float] | None:
    """Two-proportion z-test for a binary feature (its proportion of 1s, reference vs recent)."""
    ref = reference[np.isfinite(reference)]
    rec = recent[np.isfinite(recent)]
    if len(ref) < config.MIN_SAMPLE_SIZE or len(rec) < config.MIN_SAMPLE_SIZE:
        return None
    p1, n1 = ref.mean(), len(ref)
    p2, n2 = rec.mean(), len(rec)
    p_pool = (ref.sum() + rec.sum()) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p_value)


# -------------------------------------------------------------------------------------------
# Benjamini-Hochberg FDR correction (implemented directly - no new heavy dependency).
# -------------------------------------------------------------------------------------------
def benjamini_hochberg(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype="float64")
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adjusted_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n)
    adjusted[order] = np.clip(adjusted_sorted, 0, 1)
    return adjusted.tolist()


# -------------------------------------------------------------------------------------------
# Feature-level drift table
# -------------------------------------------------------------------------------------------
def detect_feature_drift(reference_df: pd.DataFrame, recent_df: pd.DataFrame,
                         features: list[str] = FINAL_FEATURES) -> pd.DataFrame:
    rows = []
    p_value_index = []   # positions of rows that got a real p-value, for BH correction afterwards
    p_values = []
    for feature in features:
        if feature in config.EXCLUDED_FROM_DRIFT:
            continue
        ref = reference_df[feature].to_numpy(dtype="float64")
        rec = recent_df[feature].to_numpy(dtype="float64")
        ftype = classify_feature_type(reference_df[feature], recent_df[feature])
        ref_finite, rec_finite = ref[np.isfinite(ref)], rec[np.isfinite(rec)]

        row = {"feature_name": feature, "feature_type": ftype,
              "reference_count": int(len(ref_finite)), "recent_count": int(len(rec_finite)),
              "reference_nan_rate": float(np.mean(~np.isfinite(ref))) if len(ref) else np.nan,
              "recent_nan_rate": float(np.mean(~np.isfinite(rec))) if len(rec) else np.nan,
              "reference_mean": float(np.mean(ref_finite)) if len(ref_finite) else np.nan,
              "recent_mean": float(np.mean(rec_finite)) if len(rec_finite) else np.nan,
              "reference_std": float(np.std(ref_finite)) if len(ref_finite) else np.nan,
              "recent_std": float(np.std(rec_finite)) if len(rec_finite) else np.nan,
              "psi": None, "ks_statistic": None, "ks_p_value": None,
              "proportion_test_statistic": None, "proportion_test_p_value": None,
              "adjusted_p_value": None,
              "drift_detected": False, "drift_reason": ""}

        if len(ref_finite) < config.MIN_SAMPLE_SIZE or len(rec_finite) < config.MIN_SAMPLE_SIZE:
            row["drift_reason"] = f"insufficient_sample (< {config.MIN_SAMPLE_SIZE} non-NaN observations in a window)"
            rows.append(row)
            continue

        row["psi"] = calculate_psi(ref, rec, ftype)
        test = calculate_ks(ref, rec) if ftype == "continuous" else calculate_proportion_test(ref, rec)
        if test is not None:
            stat, pval = test
            if ftype == "continuous":
                row["ks_statistic"], row["ks_p_value"] = stat, pval
            else:
                row["proportion_test_statistic"], row["proportion_test_p_value"] = stat, pval
            p_value_index.append(len(rows))
            p_values.append(pval)
        rows.append(row)

    adjusted = benjamini_hochberg(p_values)
    for idx, adj_p in zip(p_value_index, adjusted):
        rows[idx]["adjusted_p_value"] = adj_p

    for row in rows:
        if row["drift_reason"].startswith("insufficient_sample"):
            continue
        reasons = []
        if row["psi"] is not None and row["psi"] >= config.PSI_SIGNIFICANT_THRESHOLD:
            reasons.append(f"PSI significant ({row['psi']:.4f} >= {config.PSI_SIGNIFICANT_THRESHOLD})")
        if row["adjusted_p_value"] is not None and row["adjusted_p_value"] < config.KS_ALPHA:
            test_name = "KS" if row["feature_type"] == "continuous" else "proportion"
            reasons.append(f"{test_name} test significant after BH correction (adj_p={row['adjusted_p_value']:.4g} < {config.KS_ALPHA})")
        row["drift_detected"] = bool(reasons)
        row["drift_reason"] = "; ".join(reasons) if reasons else "no meaningful evidence of drift"

    return pd.DataFrame(rows)


# -------------------------------------------------------------------------------------------
# Overall drift decision
# -------------------------------------------------------------------------------------------
def detect_overall_drift(feature_results: pd.DataFrame, rate_threshold: float = config.OVERALL_DRIFT_RATE_THRESHOLD) -> dict:
    tested = feature_results[~feature_results["drift_reason"].str.startswith("insufficient_sample")]
    n_tested = len(tested)
    n_drifted = int(tested["drift_detected"].sum())
    drift_rate = (n_drifted / n_tested) if n_tested else 0.0
    strongest = (tested[tested["drift_detected"]].assign(_psi=lambda d: d["psi"].fillna(0))
                .sort_values("_psi", ascending=False).drop(columns="_psi").head(5)["feature_name"].tolist())
    return {
        "features_tested": n_tested, "features_drifted": n_drifted,
        "features_insufficient_sample": int(len(feature_results) - n_tested),
        "drift_rate": drift_rate, "drift_rate_threshold": rate_threshold,
        "overall_drift_detected": bool(n_tested > 0 and drift_rate >= rate_threshold),
        "strongest_drifted_features": strongest,
        "decision_rule": f"overall_drift_detected = (features_tested > 0) AND (features_drifted / features_tested >= {rate_threshold})",
    }


# -------------------------------------------------------------------------------------------
# Leakage / contract validation — never hard-coded; computed from the actual inputs.
# -------------------------------------------------------------------------------------------
def validate_drift_inputs(reference_df: pd.DataFrame, recent_df: pd.DataFrame, features: list[str] = FINAL_FEATURES) -> dict:
    checks = {}
    checks["isFraud not a drift feature"] = "isFraud" not in features
    checks["isFraud not present in reference dataset"] = "isFraud" not in reference_df.columns
    checks["isFraud not present in recent dataset"] = "isFraud" not in recent_df.columns
    checks["reference feature columns present"] = all(f in reference_df.columns for f in features)
    checks["recent feature columns present"] = all(f in recent_df.columns for f in features)
    checks["feature count == 34"] = len(features) == 34
    checks["no unexpected target column"] = not ({"isFraud"} & set(reference_df.columns) | {"isFraud"} & set(recent_df.columns))
    checks["reference window precedes recent window"] = bool(reference_df["step"].max() < recent_df["step"].min())
    return checks
