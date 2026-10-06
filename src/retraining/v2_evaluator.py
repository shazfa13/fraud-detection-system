"""Threshold selection, metrics, comparison, and Phase 11 decision logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from src.drift.drift_detector import FINAL_FEATURES

TARGET = "isFraud"


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None,
        "accuracy": float(accuracy_score(y_true, predicted)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "rows": int(len(y_true)), "fraud_count": int(y_true.sum()),
        "fraud_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "threshold": float(threshold),
    }


def evaluate_split(model, df: pd.DataFrame, threshold: float) -> dict:
    probabilities = model.predict_proba(df[FINAL_FEATURES])[:, 1]
    return _metrics(df[TARGET].to_numpy(dtype=int), probabilities, threshold)


def threshold_grid_search(model, validation_df: pd.DataFrame) -> pd.DataFrame:
    y_true = validation_df[TARGET].to_numpy(dtype=int)
    probabilities = model.predict_proba(validation_df[FINAL_FEATURES])[:, 1]
    thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)
    rows = []
    for threshold in thresholds:
        predicted = (probabilities >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "precision": float(precision_score(y_true, predicted, zero_division=0)),
            "recall": float(recall_score(y_true, predicted, zero_division=0)),
            "f1": float(f1_score(y_true, predicted, zero_division=0)),
        })
    return pd.DataFrame(rows)


def select_threshold(table: pd.DataFrame) -> tuple[float, dict]:
    best = table.sort_values(["f1", "precision", "threshold"], ascending=[False, False, True], kind="stable").iloc[0]
    return float(best["threshold"]), best.to_dict()


def compare_v1_v2_on_same_test(model_v1, threshold_v1: float, model_v2, threshold_v2: float, test_df: pd.DataFrame) -> dict:
    return {
        "v1_on_v2_test_split": evaluate_split(model_v1, test_df, threshold_v1),
        "v2_on_v2_test_split": evaluate_split(model_v2, test_df, threshold_v2),
    }


def apply_decision_rule(v1: dict, v2: dict) -> dict:
    criteria = {
        "v2_f1_not_lower_than_v1": v2["f1"] >= v1["f1"],
        "v2_pr_auc_not_lower_than_v1": v2["pr_auc"] >= v1["pr_auc"],
        "v2_recall_not_lower_than_v1": v2["recall"] >= v1["recall"],
    }
    decision = "V2_CANDIDATE_FOR_DEPLOYMENT" if all(criteria.values()) else "KEEP_V1"
    reasons = [f"{name}: {'PASS' if passed else 'FAIL'}" for name, passed in criteria.items()]
    return {"decision": decision, "criteria": criteria, "reasons": reasons}
