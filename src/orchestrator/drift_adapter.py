"""Read-only adapter for the latest persisted monitoring report."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from src.monitoring.config import MONITOR_DB_PATH


def get_latest_drift_summary(*, database_path=None) -> dict[str, Any]:
    """Read the latest saved monitor report without importing serving code.

    The query mirrors ``monitor.get_latest_report``. That module imports the
    drift detector, which imports the serving API and loads its estimator at
    module import time, so this adapter reads the same report table directly
    using SQLite read-only mode instead.
    """
    try:
        path = Path(database_path) if database_path is not None else MONITOR_DB_PATH
        if not path.is_file():
            report = None
        else:
            uri = f"file:{path.resolve().as_posix()}?mode=ro"
            with closing(sqlite3.connect(uri, uri=True, timeout=3)) as db:
                db.execute("PRAGMA query_only=ON")
                row = db.execute(
                    "SELECT report_json FROM monitor_runs ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
            report = json.loads(row[0]) if row else None
    except Exception as exc:
        return {
            "status": "UNAVAILABLE",
            "report_exists": None,
            "error": f"{type(exc).__name__}: {exc}",
            "warnings": [],
        }
    if not isinstance(report, dict):
        return {
            "status": "NO_SAVED_REPORT",
            "report_exists": False,
            "report_timestamp": None,
            "report_run_id": None,
            "reference_window": None,
            "monitoring_window": None,
            "drift_decision": None,
            "drift_status": None,
            "drifted_features": [],
            "feature_scores": [],
            "warnings": ["No saved monitoring report is available."],
        }

    drift = report.get("drift") if isinstance(report.get("drift"), dict) else {}
    feature_evidence = drift.get("feature_evidence")
    scores: list[dict[str, Any]] = []
    if isinstance(feature_evidence, list):
        for item in feature_evidence:
            if isinstance(item, dict):
                scores.append({key: item[key] for key in (
                    "feature_name", "psi", "adjusted_p_value", "drift_detected", "drift_reason"
                ) if key in item})

    warnings: list[str] = []
    for message in report.get("monitoring_errors", []) if isinstance(report.get("monitoring_errors"), list) else []:
        warnings.append(str(message))
    for section in (drift, report.get("performance")):
        if isinstance(section, dict):
            reason = section.get("reason")
            if reason:
                warnings.append(str(reason))
            limitation = section.get("baseline_comparability")
            if limitation:
                warnings.append(str(limitation))
    basis = report.get("monitoring_basis")
    if basis == "synthetic_only":
        warnings.append("Monitoring evidence is marked synthetic_only.")

    return {
        "status": "SAVED_REPORT",
        "report_exists": True,
        "report_timestamp": report.get("generated_at"),
        "report_run_id": report.get("run_id"),
        "reference_window": report.get("reference", {}).get("reference_window") if isinstance(report.get("reference"), dict) else None,
        "monitoring_window": report.get("drift_window"),
        "drift_decision": drift.get("overall_drift_detected"),
        "drift_status": report.get("drift_status"),
        "drifted_features": drift.get("drifted_features", []) if isinstance(drift.get("drifted_features", []), list) else [],
        "feature_scores": scores,
        "monitoring_basis": basis,
        "warnings": list(dict.fromkeys(warnings)),
    }


__all__ = ["get_latest_drift_summary"]
