"""Read-only Stage 3 evaluation of feature drift and eligible delayed-label performance."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.drift import config as drift_config
from src.drift.drift_detector import detect_feature_drift, detect_overall_drift
from src.labels import config as labels_config
from src.monitoring import config
from src.streaming import config as streaming_config
from src.retraining.active_model import resolve_active_model


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return str(value)


def active_model_source() -> str:
    try:
        return resolve_active_model().source
    except Exception:
        # get_active_model below fails closed; this function remains useful in reports.
        return "invalid shared active-model configuration"


def get_active_model() -> str:
    value = resolve_active_model().model_version.upper()
    if value not in {"V1", "V2"}:
        raise ValueError(f"Stage 3 has no registered reference configuration for active model {value!r}.")
    return value


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _reference_sample(model_version: str, *, path: Path | None = None,
                      sample_size: int = config.REFERENCE_SAMPLE_SIZE,
                      seed: int = config.REFERENCE_RANDOM_SEED) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reservoir-sample the model's chronological training window in bounded Arrow batches."""
    source = path or config.REFERENCE_DATA_PATHS[model_version]
    metadata = _json(config.MODEL_METADATA_PATHS[model_version])
    features = metadata.get("feature_names")
    if not isinstance(features, list) or len(features) != 34 or {"isFraud", "step", "event_id"} & set(features):
        raise ValueError("Model metadata does not define the expected safe 34-feature contract")
    train_end = int(metadata["train_validation_test_strategy"]["train_end_step"])
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to scan the reference parquet in bounded batches") from exc
    dataset = ds.dataset(source, format="parquet")
    required = set(features) | {"step"}
    if not required.issubset(dataset.schema.names):
        raise ValueError("Reference parquet does not contain the active model feature schema and step")
    scanner = dataset.scanner(columns=["step", *features],
                              filter=ds.field("step") <= train_end,
                              batch_size=65536, use_threads=True)
    rng = np.random.default_rng(seed)
    reservoir: list[dict[str, float]] = []
    # Keep a small, independent reservoir of rows containing finite values for
    # every feature. The ordinary row reservoir can miss rare history features
    # even when the source has enough observations to test them.
    coverage_target = max(30, drift_config.MIN_SAMPLE_SIZE)
    feature_reservoirs: dict[str, list[dict[str, float]]] = {name: [] for name in features}
    feature_seen = {name: 0 for name in features}
    seen = 0
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        values = frame.loc[:, features].to_numpy(dtype=np.float64, copy=False)
        steps = frame["step"].to_numpy()
        # Vectorized Algorithm R samples rows first, then the existing detector
        # filters finite values independently per feature. Complete-case filtering
        # here selected a small, biased cohort (2,070 rows in the current reference).
        upper = seen + np.arange(1, len(frame) + 1, dtype=np.int64)
        slots = np.floor(rng.random(len(frame)) * upper).astype(np.int64)
        first_fill = max(0, min(sample_size - seen, len(frame)))
        slots[:first_fill] = np.arange(seen, seen + first_fill, dtype=np.int64)
        for row_idx in np.flatnonzero(slots < sample_size):
            slot = int(slots[row_idx])
            row = {name: (float(values[row_idx, col]) if np.isfinite(values[row_idx, col]) else np.nan)
                   for col, name in enumerate(features)}
            row["step"] = int(steps[row_idx])
            if slot < len(reservoir):
                reservoir[slot] = row
            else:
                reservoir.append(row)
        for col, name in enumerate(features):
            finite_rows = np.flatnonzero(np.isfinite(values[:, col]))
            old_seen = feature_seen[name]
            feature_seen[name] += len(finite_rows)
            bucket = feature_reservoirs[name]
            if len(finite_rows):
                upper = old_seen + np.arange(1, len(finite_rows) + 1, dtype=np.int64)
                targets = rng.integers(0, upper)
                fill = max(0, min(coverage_target - old_seen, len(finite_rows)))
                targets[:fill] = np.arange(old_seen, old_seen + fill, dtype=np.int64)
                for local_idx in np.flatnonzero(targets < coverage_target):
                    row_idx = int(finite_rows[local_idx])
                    slot = int(targets[local_idx])
                    row = {feature: (float(values[row_idx, j]) if np.isfinite(values[row_idx, j]) else np.nan)
                           for j, feature in enumerate(features)}
                    row["step"] = int(steps[row_idx])
                    if slot < len(bucket):
                        bucket[slot] = row
                    else:
                        bucket.append(row)
        seen += len(frame)
    if not reservoir:
        raise ValueError("No finite feature rows were available in the reference training window")
    # Union targeted finite rows with the uniform row sample, deduplicating by
    # source step/feature vector. This preserves row-level missingness better
    # than independently sampling each feature while guaranteeing coverage for
    # sufficiently common sparse features. Targeted inclusion mildly biases
    # marginal missingness; full-window finite/missing counts below are exact.
    combined = {json.dumps(row, sort_keys=True, allow_nan=True): row for row in reservoir}
    for bucket in feature_reservoirs.values():
        for row in bucket:
            combined.setdefault(json.dumps(row, sort_keys=True, allow_nan=True), row)
    reference = pd.DataFrame(list(combined.values()))
    per_feature = {name: {"finite_count": int(np.isfinite(reference[name]).sum()),
                          "missing_count": int(len(reference) - np.isfinite(reference[name]).sum()),
                          "source_finite_count": int(feature_seen[name]),
                          "source_missing_count": int(seen - feature_seen[name])}
                   for name in features}
    details = {"reference_source": str(source), "reference_version": f"{model_version}-train-step-1-{train_end}",
               "reference_window": {"min_step": int(reference.step.min()), "max_step": train_end},
               "reference_rows_scanned": seen, "reference_sample_count": len(reference),
               "reference_sampling": "deterministic uniform row reservoir plus per-feature finite-row coverage reservoirs",
               "reference_seed": seed, "reference_sample_target": sample_size,
               "reference_per_feature_coverage_target": coverage_target,
               "reference_per_feature_counts": per_feature,
               "reference_sampling_limitation": "Targeted row inclusion improves sparse-feature test coverage but modestly overrepresents rows containing sparse history values; feature correlations are retained at row level.",
               "features": features}
    return reference, details


def _recent_features(path: Path, model_version: str, limit: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = _json(config.MODEL_METADATA_PATHS[model_version])
    features = metadata["feature_names"]
    rows: dict[str, dict[str, Any]] = {}
    malformed = 0
    data_origins: dict[str, int] = {}
    if not path.is_file():
        return pd.DataFrame(columns=features), {"records_read": 0, "malformed_records": 0,
            "window_start": None, "window_end": None, "data_origin_counts": {}}
    # Bound memory and work by reading only the tail of the append-only Stage 1 log.
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - 32 * 1024 * 1024))
        if size > 32 * 1024 * 1024:
            stream.readline()
        lines = stream.read().decode("utf-8", errors="replace").splitlines()[-limit:]
    for line in lines:
        try:
            item = json.loads(line)
            vector = item.get("feature_values")
            if item.get("model_version") != model_version or not isinstance(vector, dict):
                continue
            values = {}
            for name in features:
                raw_value = vector[name]
                if raw_value is None:
                    values[name] = np.nan
                elif isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                    raise ValueError(f"invalid feature value for {name}")
                else:
                    numeric = float(raw_value)
                    values[name] = numeric if math.isfinite(numeric) else np.nan
            values.update({"transaction_id": str(item["transaction_id"]),
                           "processing_timestamp_utc": str(item["processing_timestamp_utc"]),
                           "_data_origin": str(item.get("data_origin", "unspecified"))})
            rows[values["transaction_id"]] = values
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            malformed += 1
    records = sorted(rows.values(), key=lambda r: r["processing_timestamp_utc"])
    data_origins = {}
    for record in records:
        origin = record["_data_origin"]
        data_origins[origin] = data_origins.get(origin, 0) + 1
    frame = pd.DataFrame([{f: row[f] for f in features} for row in records])
    return frame, {"records_read": len(lines), "malformed_records": malformed,
                   "window_start": records[0]["processing_timestamp_utc"] if records else None,
                   "window_end": records[-1]["processing_timestamp_utc"] if records else None,
                   "data_origin_counts": data_origins}


def _open_monitor_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE IF NOT EXISTS monitor_runs(run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
            model_version TEXT NOT NULL, status TEXT NOT NULL, report_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS monitor_alerts(alert_id TEXT PRIMARY KEY, alert_key TEXT UNIQUE NOT NULL,
            alert_type TEXT NOT NULL, model_version TEXT NOT NULL, window_start TEXT, window_end TEXT,
            created_at TEXT NOT NULL, evidence_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS retraining_requests(request_id TEXT PRIMARY KEY, request_key TEXT UNIQUE NOT NULL,
            status TEXT NOT NULL, model_version TEXT NOT NULL, trigger_reason TEXT NOT NULL,
            alert_ids_json TEXT NOT NULL, evidence_json TEXT NOT NULL, created_at TEXT NOT NULL);
    """)
    return db


def _record_result(report: dict[str, Any], database_path: Path) -> None:
    with closing(_open_monitor_db(database_path)) as db:
      with db:
        run_id = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
        db.execute("INSERT OR IGNORE INTO monitor_runs VALUES(?,?,?,?,?)",
                   (run_id, report["generated_at"], report["model_version"], report["overall_status"],
                    json.dumps(report, sort_keys=True, allow_nan=False)))
        created_alerts: list[str] = []
        for alert in report["alerts"]:
            key = f"{alert['type']}|{report['model_version']}|{alert.get('window_start')}|{alert.get('window_end')}"
            alert_id = hashlib.sha256(key.encode()).hexdigest()
            cursor = db.execute("INSERT OR IGNORE INTO monitor_alerts VALUES(?,?,?,?,?,?,?,?)",
                (alert_id, key, alert["type"], report["model_version"], alert.get("window_start"),
                 alert.get("window_end"), report["generated_at"], json.dumps(alert, sort_keys=True)))
            alert["alert_id"] = alert_id
            alert["newly_recorded"] = bool(cursor.rowcount)
            if cursor.rowcount:
                created_alerts.append(alert_id)
        if report["retraining_request"]["requested"] and created_alerts:
            request_key = hashlib.sha256("|".join(sorted(created_alerts)).encode()).hexdigest()
            db.execute("INSERT OR IGNORE INTO retraining_requests VALUES(?,?,?,?,?,?,?,?)",
                       (hashlib.sha256(request_key.encode()).hexdigest(), request_key, "requested",
                        report["model_version"], report["retraining_request"]["reason"],
                        json.dumps(created_alerts), json.dumps(report["retraining_request"], sort_keys=True),
                        report["generated_at"]))
        db.execute("UPDATE monitor_runs SET report_json=? WHERE run_id=?",
                   (json.dumps(report, sort_keys=True, allow_nan=False), run_id))


def _performance(model_version: str, *, database_path: Path | None = None) -> dict[str, Any]:
    metadata = _json(config.MODEL_METADATA_PATHS[model_version])
    threshold = float(_json(config.MODEL_THRESHOLD_PATHS[model_version])["selected_threshold"])
    baseline = metadata.get("evaluation_metrics", {}).get(config.BASELINE_SPLIT)
    if not isinstance(baseline, dict) or float(baseline.get("threshold", -1)) != threshold:
        raise ValueError(f"No matching {model_version} held-out baseline at stored threshold {threshold}")
    trusted_sources = labels_config.trusted_label_sources()
    token_configured = bool(labels_config.trusted_label_token())
    result: dict[str, Any] = {"status": "insufficient_data", "model_version": model_version,
        "threshold": threshold, "evaluation_window": {"start": None, "end": None},
        "label_source_breakdown": {}, "eligible_labeled_transactions": 0,
        "synthetic_labels_excluded": 0, "excluded_or_pending_records": 0,
        "duplicate_submission_records": 0, "conflicting_label_transactions": 0,
        "metrics": None, "confusion_matrix": None,
        "baseline": {"source": str(config.BASELINE_REPORT_PATHS[model_version]),
                     "dataset": metadata.get("dataset_info", {}).get("source_file"),
                     "split": config.BASELINE_SPLIT, "threshold": threshold,
                     "metrics": {key: baseline.get(key) for key in
                                 ("precision", "recall", "f1", "tp", "fp", "tn", "fn", "rows")}},
        "label_basis": "verified_labels" if trusted_sources and token_configured else "unavailable",
        "generated_at": utc_now(), "reason": "No trusted genuine label source and authorization are configured."}
    stage2_path = database_path or labels_config.STAGE2_DB_PATH
    if not stage2_path.is_file():
        result["reason"] = "Stage 2 database is unavailable."
        return result
    with closing(sqlite3.connect(f"file:{stage2_path.resolve().as_posix()}?mode=ro", uri=True)) as db:
      with db:
        db.row_factory = sqlite3.Row
        result["synthetic_labels_excluded"] = db.execute(
            "SELECT COUNT(*) FROM labels WHERE source_type='synthetic_test'").fetchone()[0]
        result["excluded_or_pending_records"] = db.execute(
            "SELECT COUNT(*) FROM predictions WHERE label_status='pending'").fetchone()[0]
        sources = db.execute("SELECT source_type,label_source,COUNT(*) n FROM labels GROUP BY 1,2").fetchall()
        result["label_source_breakdown"] = {f"{r['source_type']}:{r['label_source']}": r["n"] for r in sources}
        result["duplicate_submission_records"] = db.execute(
            "SELECT (SELECT COUNT(*) FROM label_submissions WHERE disposition='duplicate_identical') + "
            "(SELECT COUNT(*) FROM submission_replays WHERE disposition='duplicate_submission')").fetchone()[0]
        result["conflicting_label_transactions"] = db.execute(
            "SELECT COUNT(DISTINCT transaction_id) FROM label_submissions WHERE disposition='conflict'").fetchone()[0]
        if not (trusted_sources and token_configured):
            return result
        placeholders = ",".join("?" for _ in trusted_sources)
        eligible = db.execute(f"""SELECT p.transaction_id,p.prediction,p.model_version,p.prediction_timestamp,
                  l.outcome,l.label_source,l.confirmed_at
            FROM predictions p JOIN labels l USING(transaction_id)
            WHERE p.model_version=? AND p.data_origin NOT LIKE 'synthetic%'
              AND l.source_type='trusted_verified' AND l.status='resolved'
              AND l.label_source IN ({placeholders})
              AND NOT EXISTS (SELECT 1 FROM label_submissions s WHERE s.transaction_id=p.transaction_id AND s.disposition='conflict')
              AND NOT EXISTS (SELECT 1 FROM prediction_conflicts c WHERE c.transaction_id=p.transaction_id)
            ORDER BY l.confirmed_at""", (model_version, *sorted(trusted_sources))).fetchall()
    result["eligible_labeled_transactions"] = len(eligible)
    result["label_basis"] = "verified_labels"
    if eligible:
        result["evaluation_window"] = {"start": eligible[0]["confirmed_at"], "end": eligible[-1]["confirmed_at"]}
    positives = sum(int(row["outcome"]) for row in eligible)
    result["eligible_fraud_labels"] = positives
    if len(eligible) < config.MIN_ELIGIBLE_LABELS or positives < config.MIN_ELIGIBLE_FRAUD_LABELS:
        result["reason"] = (f"Requires at least {config.MIN_ELIGIBLE_LABELS} eligible labels and "
                             f"{config.MIN_ELIGIBLE_FRAUD_LABELS} fraud labels; found {len(eligible)} and {positives}.")
        return result
    tp = sum(row["prediction"] == 1 and row["outcome"] == 1 for row in eligible)
    fp = sum(row["prediction"] == 1 and row["outcome"] == 0 for row in eligible)
    tn = sum(row["prediction"] == 0 and row["outcome"] == 0 for row in eligible)
    fn = sum(row["prediction"] == 0 and row["outcome"] == 1 for row in eligible)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn), "precision": precision,
               "recall": recall, "f1": f1, "rows": len(eligible)}
    result["metrics"], result["confusion_matrix"] = metrics, {k: metrics[k] for k in ("tp", "fp", "tn", "fn")}
    result["baseline_comparability"] = "PaySim synthetic held-out baseline vs verified stream labels; population comparability is unproven."
    degraded = [name for name in config.PERFORMANCE_METRICS
                if float(baseline[name]) - metrics[name] >= config.PERFORMANCE_DROP_THRESHOLD]
    result["baseline_delta"] = {name: metrics[name] - float(baseline[name]) for name in config.PERFORMANCE_METRICS}
    result["status"] = "degraded" if degraded else "healthy"
    result["degraded_metrics"] = degraded
    result["reason"] = "Provisional absolute drop threshold applied; confirm cohort comparability before action."
    return result


def run_monitor(*, stage1_path: Path | None = None, reference_path: Path | None = None,
                stage2_path: Path | None = None, monitor_db_path: Path | None = None,
                model_version: str | None = None, recent_limit: int = config.RECENT_RECORD_LIMIT,
                min_current: int = config.MIN_CURRENT_WINDOW) -> dict[str, Any]:
    model = model_version or get_active_model()
    if model not in {"V1", "V2"}:
        raise ValueError("model_version must be V1 or V2")
    generated = utc_now()
    metadata = _json(config.MODEL_METADATA_PATHS[model])
    features = metadata["feature_names"]
    recent, window = _recent_features(stage1_path or streaming_config.STAGE1_OUTPUT_PATH, model, recent_limit)
    reference_details: dict[str, Any] = {"reference_source": str(reference_path or config.REFERENCE_DATA_PATHS[model]),
        "reference_version": f"{model}-train-step-1-{metadata['train_validation_test_strategy']['train_end_step']}",
        "reference_window": {"min_step": 1,
                             "max_step": metadata["train_validation_test_strategy"]["train_end_step"]},
        "reference_rows_scanned": None, "reference_sample_count": None,
        "reference_sample_target": config.REFERENCE_SAMPLE_SIZE,
        "sampling_method": "deterministic-seed reservoir sample; per-feature missing values retained",
        "reference_seed": config.REFERENCE_RANDOM_SEED, "features": features}
    drift_status = "insufficient_data"
    drift: dict[str, Any] = {"features_tested": 0, "features_drifted": 0, "drift_rate": None,
        "overall_drift_detected": False, "reason": None,
        "detector": "PSI + KS/two-proportion z-test + Benjamini-Hochberg",
        "configuration": {"psi_bins": drift_config.PSI_BINS,
            "psi_significant_threshold": drift_config.PSI_SIGNIFICANT_THRESHOLD,
            "ks_alpha": drift_config.KS_ALPHA,
            "overall_drift_rate_threshold": drift_config.OVERALL_DRIFT_RATE_THRESHOLD,
            "minimum_feature_sample": drift_config.MIN_SAMPLE_SIZE,
            "minimum_current_window": min_current}}
    errors: list[str] = []
    data_origins = window.get("data_origin_counts", {})
    synthetic_only = bool(len(recent) and data_origins and
                          sum(data_origins.values()) == len(recent) and
                          all(name.lower().startswith("synthetic") for name in data_origins))
    if len(recent) < min_current:
        drift["reason"] = f"Need {min_current} valid current vectors; found {len(recent)}."
    else:
        try:
            reference, reference_details = _reference_sample(model, path=reference_path)
            feature_results = detect_feature_drift(reference, recent, features)
            drift_summary = detect_overall_drift(feature_results)
            drift.update(drift_summary)
            drift["reason"] = "Existing PSI + KS/proportion tests with Benjamini-Hochberg correction."
            drift["drifted_features"] = feature_results.loc[feature_results.drift_detected, "feature_name"].tolist()
            evidence_columns = ("feature_name", "feature_type", "reference_count", "recent_count",
                "reference_nan_rate", "recent_nan_rate", "psi", "ks_p_value",
                "proportion_test_p_value", "adjusted_p_value", "drift_detected", "drift_reason")
            drift["feature_evidence"] = [
                {name: _json_safe(row[name]) for name in evidence_columns}
                for row in feature_results.loc[:, evidence_columns].to_dict("records")]
            drift_status = ("insufficient_data" if drift["features_tested"] == 0 else
                            "drift" if drift["overall_drift_detected"] else "no_drift")
        except Exception as exc:
            errors.append(f"Drift evaluation: {type(exc).__name__}: {exc}")
            drift["reason"] = "Reference loading or drift evaluation failed."
    try:
        performance = _performance(model, database_path=stage2_path)
    except Exception as exc:
        errors.append(f"Performance evaluation: {type(exc).__name__}: {exc}")
        performance = {"status": "insufficient_data", "model_version": model,
                       "eligible_labeled_transactions": 0, "metrics": None,
                       "reason": "Label-store evaluation failed.", "generated_at": utc_now()}
    alerts = []
    if drift_status == "drift":
        alerts.append({"type": "feature_drift", "window_start": window["window_start"],
                       "window_end": window["window_end"], "evidence": drift})
    if performance["status"] == "degraded":
        alerts.append({"type": "performance_degradation", "window_start": performance["evaluation_window"]["start"],
                       "window_end": performance["evaluation_window"]["end"], "evidence": performance})
    status = "degraded" if performance["status"] == "degraded" else (
        "insufficient_data" if performance["status"] == "insufficient_data" or drift_status == "insufficient_data" else "healthy")
    reasons = [a["type"] for a in alerts]
    request_allowed = bool(alerts) and not synthetic_only
    request_reason = (", ".join(reasons) if request_allowed else
                      "Synthetic-only monitoring window; no retraining request authorized." if alerts and synthetic_only else
                      "No actionable drift or performance alert.")
    report = {"generated_at": generated, "model_version": model,
        "threshold": float(_json(config.MODEL_THRESHOLD_PATHS[model])["selected_threshold"]),
        "active_model_source": active_model_source(),
        "overall_status": status, "drift_status": drift_status, "drift_window": window,
        "monitoring_basis": "synthetic_only" if synthetic_only else "non_synthetic_or_mixed",
        "reference": reference_details, "drift": drift, "performance": performance,
        "monitoring_errors": errors, "alerts": alerts, "retraining_request": {"requested": request_allowed,
            "status": "requested" if request_allowed else "not_requested",
            "reason": request_reason,
            "note": "Recorded decision only; no training or model activation is performed."}}
    _record_result(report, monitor_db_path or config.MONITOR_DB_PATH)
    return report


def get_latest_report(database_path: Path | None = None) -> dict[str, Any] | None:
    path = database_path or config.MONITOR_DB_PATH
    if not path.is_file():
        return None
    with closing(sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)) as db:
        row = db.execute("SELECT report_json FROM monitor_runs ORDER BY created_at DESC LIMIT 1").fetchone()
    return json.loads(row[0]) if row else None
