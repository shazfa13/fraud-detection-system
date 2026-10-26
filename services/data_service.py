"""Read-only, size-bounded access to the project's existing reports and logs."""
from __future__ import annotations

import csv
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT_FILE_BYTES = 4 * 1024 * 1024
MAX_HISTORY_ROWS = 500

API_LOGS = (
    ("reports/model_v1_predictions_log.jsonl", "API log · V1 historical"),
    ("reports/model_v2_predictions_log.jsonl", "API log · V2 historical"),
)
STREAM_CSVS = (
    ("reports/phase7_streaming_predictions.csv", "Phase 7 streaming CSV"),
    ("reports/phase12_kafka_test_predictions.csv", "Phase 12 Kafka test CSV"),
)


def project_path(project_root: str | Path, relative_path: str | Path) -> Path | None:
    """Resolve a project-relative path and refuse traversal outside the project root."""
    root = Path(project_root).resolve()
    relative = Path(relative_path)
    if relative.is_absolute():
        return None
    candidate = (root / relative).resolve()
    return candidate if candidate == root or root in candidate.parents else None


def project_relative_path(project_root: str | Path, source_path: str | Path) -> str | None:
    """Convert a report's absolute or relative path to a safe project-relative path."""
    root = Path(project_root).resolve()
    path = Path(source_path)
    candidate = path.resolve() if path.is_absolute() else (root / path).resolve()
    if candidate == root or root in candidate.parents:
        return candidate.relative_to(root).as_posix()
    return None


def read_json_report(project_root: str | Path, relative_path: str | Path) -> tuple[Any | None, str | None]:
    """Read one bounded JSON report without raising for missing or malformed data."""
    path = project_path(project_root, relative_path)
    if path is None:
        return None, "Path is outside the project directory."
    try:
        if not path.is_file():
            return None, "File not found."
        if path.stat().st_size > MAX_TEXT_FILE_BYTES:
            return None, "File exceeds the dashboard read limit."
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream), None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"Could not read JSON: {exc}"


def _read_jsonl(path: Path, source: str, *, limit: int) -> list[dict[str, Any]]:
    try:
        size = path.stat().st_size
        if size > MAX_TEXT_FILE_BYTES:
            with path.open("rb") as stream:
                stream.seek(size - MAX_TEXT_FILE_BYTES)
                stream.readline()  # discard a potentially partial first record
                content = stream.read().decode("utf-8", errors="replace")
        else:
            content = path.read_text(encoding="utf-8")
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in content.splitlines()[-limit:]:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        record = _normalize_record(item, source)
        if record:
            records.append(record)
    return records


def _normalize_record(item: dict[str, Any], source: str) -> dict[str, Any] | None:
    """Normalize only fields present in the source; absent transaction details stay absent."""
    try:
        probability = float(item["fraud_probability"])
        prediction = int(item.get("prediction", item.get("predicted_label")))
        score = float(item["risk_score"])
    except (KeyError, TypeError, ValueError):
        return None
    if prediction not in (0, 1) or not 0 <= probability <= 1 or not 0 <= score <= 100:
        return None
    timestamp = item.get("timestamp_utc", item.get("processing_timestamp_utc", item.get("prediction_timestamp", "")))
    return {
        "timestamp": timestamp,
        "transaction_id": item.get("transaction_id", "—"),
        "step": item.get("step"),
        "type": item.get("type", "—"),
        "amount": item.get("amount"),
        "prediction": prediction,
        "fraud_probability": probability,
        "risk_score": score,
        "risk_level": str(item.get("risk_level", "Unknown")),
        "model_version": str(item.get("model_version", "Unknown")),
        "source": source,
    }


def _read_csv(path: Path, source: str, *, limit: int) -> list[dict[str, Any]]:
    try:
        if path.stat().st_size > MAX_TEXT_FILE_BYTES:
            return []
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            return [record for row in reader if (record := _normalize_record(row, source))][:limit]
    except (OSError, UnicodeError, csv.Error):
        return []


def load_prediction_history(project_root: str | Path = PROJECT_ROOT,
                            *, limit: int = MAX_HISTORY_ROWS) -> list[dict[str, Any]]:
    """Read known API and streaming history files; never scan data or model files."""
    root = Path(project_root).resolve()
    records: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    configured_log = os.environ.get("FRAUD_PREDICTION_LOG_PATH", "").strip()
    if configured_log:
        configured_path = Path(configured_log).expanduser()
        if not configured_path.is_absolute():
            configured_path = root / configured_path
        if configured_path.is_file():
            resolved = configured_path.resolve()
            seen_paths.add(resolved)
            records.extend(_read_jsonl(resolved, "Configured API log", limit=limit))

    for relative, source in API_LOGS:
        path = project_path(root, relative)
        if path and path.is_file() and path.resolve() not in seen_paths:
            seen_paths.add(path.resolve())
            records.extend(_read_jsonl(path, source, limit=limit))

    csv_sources = list(STREAM_CSVS)
    reports_dir = project_path(root, "reports")
    if reports_dir and reports_dir.is_dir():
        try:
            csv_sources.extend(
                (str(path.relative_to(root)), "Phase 12 Kafka–Spark output")
                for path in sorted(reports_dir.glob("phase12_v2_predictions_*.csv"))
            )
        except OSError:
            pass
    for relative, source in csv_sources:
        path = project_path(root, relative)
        if path and path.is_file():
            records.extend(_read_csv(path, source, limit=limit))

    records.sort(key=lambda row: str(row.get("timestamp", "")), reverse=True)
    return records[:max(0, min(int(limit), MAX_HISTORY_ROWS))]


def load_latest_phase12_evidence(project_root: str | Path = PROJECT_ROOT) -> dict[str, Any] | None:
    """Return the newest valid, explicitly passing Phase 12 evidence report."""
    root = Path(project_root).resolve()
    directory = project_path(root, "reports")
    if directory is None or not directory.is_dir():
        return None
    try:
        candidates = sorted(directory.glob("phase12_v2_integration_evidence_*.json"), reverse=True)
    except OSError:
        return None
    for path in candidates:
        try:
            if path.stat().st_size > MAX_TEXT_FILE_BYTES:
                continue
            with path.open("r", encoding="utf-8") as stream:
                report = json.load(stream)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(report, dict) and report.get("phase") == 12 and report.get("status") in {"PASS", "FAIL"}:
            report["_report_path"] = str(path.relative_to(root))
            return report
    return None


def read_project_csv(project_root: str | Path, relative_path: str | Path,
                     *, limit: int = 500) -> list[dict[str, str]]:
    """Read a bounded project-relative CSV, returning an empty list on errors."""
    path = project_path(project_root, relative_path)
    if path is None:
        return []
    try:
        if not path.is_file() or path.stat().st_size > MAX_TEXT_FILE_BYTES:
            return []
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))[:max(0, limit)]
    except (OSError, UnicodeError, csv.Error):
        return []


def read_project_jsonl(project_root: str | Path, relative_path: str | Path,
                      *, limit: int = 500) -> list[dict[str, Any]]:
    """Read the newest bounded raw JSONL objects from a safe project-relative path."""
    path = project_path(project_root, relative_path)
    if path is None:
        return []
    try:
        size = path.stat().st_size
        if size > MAX_TEXT_FILE_BYTES:
            with path.open("rb") as stream:
                stream.seek(size - MAX_TEXT_FILE_BYTES)
                stream.readline()
                content = stream.read().decode("utf-8", errors="replace")
        else:
            content = path.read_text(encoding="utf-8")
    except OSError:
        return []
    rows = []
    for line in content.splitlines()[-max(0, limit):]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def read_stage1_status(project_root: str | Path = PROJECT_ROOT) -> dict[str, Any] | None:
    report, error = read_json_report(project_root, "data/stage1_streaming/stream_status.json")
    return report if error is None and isinstance(report, dict) else None


def load_stage2_label_stats() -> dict[str, Any]:
    """Return Stage 2 registry counts without creating or modifying its database."""
    from src.labels.store import get_stats
    try:
        return {"store_available": True, **get_stats()}
    except OSError:
        return {"store_available": False, "error": "Stage 2 SQLite store is unavailable."}
    except Exception:
        return {"store_available": False, "error": "Stage 2 SQLite store could not be read."}


def prediction_registry_summary(database_path: str | Path | None = None) -> dict[str, Any]:
    """Summarize registered predictions through a strictly read-only SQLite connection."""
    if database_path is None:
        from src.labels.config import STAGE2_DB_PATH
        database_path = STAGE2_DB_PATH
    path = Path(database_path)
    if not path.is_file():
        return {"available": False, "error": "Prediction registry database is not available."}
    try:
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=3)) as db:
            db.execute("PRAGMA query_only=ON")
            row = db.execute("""SELECT COUNT(*),
                COALESCE(SUM(CASE WHEN prediction=1 THEN 1 ELSE 0 END),0),
                COALESCE(SUM(CASE WHEN prediction=0 THEN 1 ELSE 0 END),0),
                MIN(registered_at), MAX(registered_at)
                FROM predictions""").fetchone()
        return {"available": True, "total": int(row[0]), "predicted_fraud": int(row[1]),
                "predicted_legitimate": int(row[2]), "first_recorded_at": row[3],
                "last_recorded_at": row[4], "source": str(path)}
    except (sqlite3.Error, OSError):
        return {"available": False, "error": "Prediction registry could not be read safely."}


def stage2_label_overview(database_path: str | Path | None = None) -> dict[str, Any]:
    """Read label provenance and submission dispositions without exposing credentials."""
    if database_path is None:
        from src.labels.config import STAGE2_DB_PATH
        database_path = STAGE2_DB_PATH
    path = Path(database_path)
    if not path.is_file():
        return {"available": False, "error": "Stage 2 label database is not available."}
    try:
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=3)) as db:
            db.execute("PRAGMA query_only=ON")
            labels = db.execute("SELECT source_type,label_source,status,COUNT(*) FROM labels GROUP BY source_type,label_source,status").fetchall()
            submissions = db.execute("SELECT disposition,COUNT(*) FROM label_submissions GROUP BY disposition").fetchall()
        return {"available": True,
                "labels": [{"source_type": r[0], "label_source": r[1], "status": r[2], "count": int(r[3])} for r in labels],
                "submissions": [{"disposition": r[0], "count": int(r[1])} for r in submissions]}
    except (sqlite3.Error, OSError):
        return {"available": False, "error": "Stage 2 label provenance could not be read safely."}


def load_stage3_monitor_report() -> dict[str, Any] | None:
    """Read the most recent persisted Stage 3 report without touching Stage 2 storage."""
    try:
        from src.monitoring.monitor import get_latest_report
        report = get_latest_report()
        return report if isinstance(report, dict) else None
    except Exception:
        return None


def load_active_model_snapshot(project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    """Return the shared resolver's selected bundle without loading an estimator."""
    try:
        from src.retraining.active_model import resolve_active_model
        bundle = resolve_active_model(project_root=project_root)
        root = Path(project_root).resolve()
        return {"available": True, "model_version": bundle.model_version,
                "threshold": bundle.threshold, "feature_names": list(bundle.feature_names),
                "feature_count": len(bundle.feature_names), "source": bundle.source,
                "model_path": project_relative_path(root, bundle.model_path),
                "manifest_present": (root / "models" / "active_model_manifest.json").is_file()}
    except Exception as exc:
        return {"available": False, "error": f"Active model configuration is unavailable: {exc}"}


def load_latest_stage6_demo_report(project_root: str | Path = PROJECT_ROOT) -> tuple[dict[str, Any] | None, str | None]:
    """Read the newest saved Stage 6 demo report; never loads a candidate model."""
    root = Path(project_root).resolve()
    directory = project_path(root, "models/candidates/stage6_demo")
    if directory is None or not directory.is_dir():
        return None, "No Stage 6 demonstration candidate report is available."
    try:
        candidates = sorted(directory.glob("*/training_report.json"),
                            key=lambda path: path.stat().st_mtime, reverse=True)
    except OSError:
        return None, "Stage 6 demonstration reports could not be listed."
    for candidate in candidates:
        relative = project_relative_path(root, candidate)
        if relative:
            report, _ = read_json_report(root, relative)
            if isinstance(report, dict):
                report["_report_path"] = relative
                return report, None
    return None, "No readable Stage 6 demonstration report was found."


def load_transaction_types(project_root: str | Path = PROJECT_ROOT) -> list[str]:
    """Load transaction type categories from the saved feature-engineering report."""
    report, _ = read_json_report(project_root, "reports/feature_engineering_summary.json")
    try:
        categories = report["frozen_parameters_fitted_on_training_period"]["type_categories"]
    except (TypeError, KeyError):
        return []
    if not isinstance(categories, list):
        return []
    return [item for item in categories if isinstance(item, str) and item]


def local_artifact_status(project_root: str | Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    """Summarize known model paths and lifecycle-directory contents without loading models."""
    root = Path(project_root).resolve()
    paths = (
        ("API V1 model", "models/xgboost_model_v1.json", "file"),
        ("API V2 model", "models/xgboost_model_v2.json", "file"),
        ("V1 metadata", "models/model_v1_metadata.json", "file"),
        ("V2 metadata", "models/model_v2_metadata.json", "file"),
        ("V2 threshold", "models/model_v2_threshold.json", "file"),
        ("Lifecycle active model", "models/active/model.json", "file"),
        ("Lifecycle active metadata", "models/active/metadata.json", "file"),
        ("Lifecycle active threshold", "models/active/threshold.json", "file"),
    )
    result = []
    for label, relative, _ in paths:
        path = project_path(root, relative)
        result.append({"label": label, "path": relative, "available": bool(path and path.is_file())})
    for name in ("candidate", "backup"):
        path = project_path(root, Path("models") / name)
        try:
            files = sorted(child.name for child in path.iterdir() if child.is_file()) if path and path.is_dir() else []
            exists = bool(path and path.is_dir())
        except OSError:
            files, exists = [], False
        result.append({"label": f"Lifecycle {name}", "path": f"models/{name}",
                       "available": exists, "files": files, "empty": exists and not files})
    return result
