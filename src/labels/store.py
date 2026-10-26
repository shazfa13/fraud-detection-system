"""SQLite-backed Stage 2 prediction registry and delayed-label audit store."""
from __future__ import annotations

import json
import hmac
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.labels import config


class LabelValidationError(ValueError):
    pass


class UnauthorizedLabelSource(LabelValidationError):
    pass


class PredictionConflictError(ValueError):
    pass


class SubmissionIdConflict(ValueError):
    pass


class _ClosingConnection(sqlite3.Connection):
    """SQLite's built-in context manager commits but does not close the connection."""
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(database_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(database_path or config.STAGE2_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10, isolation_level="IMMEDIATE", factory=_ClosingConnection)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            transaction_id TEXT PRIMARY KEY,
            prediction INTEGER NOT NULL CHECK(prediction IN (0,1)),
            fraud_probability REAL NOT NULL CHECK(fraud_probability >= 0 AND fraud_probability <= 1),
            risk_score REAL,
            risk_level TEXT,
            prediction_timestamp TEXT NOT NULL,
            model_version TEXT NOT NULL,
            data_origin TEXT NOT NULL,
            kafka_topic TEXT,
            kafka_partition INTEGER,
            kafka_offset INTEGER,
            label_status TEXT NOT NULL DEFAULT 'pending' CHECK(label_status IN ('pending','resolved')),
            registered_at TEXT NOT NULL,
            UNIQUE(kafka_topic, kafka_partition, kafka_offset)
        );
        CREATE TABLE IF NOT EXISTS labels (
            transaction_id TEXT PRIMARY KEY,
            outcome INTEGER NOT NULL CHECK(outcome IN (0,1)),
            label_source TEXT NOT NULL,
            source_type TEXT NOT NULL CHECK(source_type IN ('trusted_verified','untrusted','synthetic_test')),
            confirmed_at TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('pending','resolved','rejected')),
            submission_id TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS label_submissions (
            submission_id TEXT PRIMARY KEY,
            transaction_id TEXT NOT NULL,
            outcome INTEGER NOT NULL CHECK(outcome IN (0,1)),
            label_source TEXT NOT NULL,
            source_type TEXT NOT NULL CHECK(source_type IN ('trusted_verified','untrusted','synthetic_test')),
            confirmed_at TEXT NOT NULL,
            disposition TEXT NOT NULL,
            submitted_at TEXT NOT NULL,
            duplicate_of TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_label_submissions_disposition ON label_submissions(disposition);
        CREATE INDEX IF NOT EXISTS idx_labels_status_source ON labels(status, source_type);
        CREATE TABLE IF NOT EXISTS prediction_conflicts (
            conflict_id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL,
            stored_record TEXT NOT NULL,
            incoming_record TEXT NOT NULL,
            observed_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_prediction_conflict_identity
            ON prediction_conflicts(transaction_id, incoming_record);
        CREATE TABLE IF NOT EXISTS submission_replays (
            replay_id INTEGER PRIMARY KEY AUTOINCREMENT,
            submission_id TEXT NOT NULL,
            disposition TEXT NOT NULL,
            observed_at TEXT NOT NULL
        );
    """)
    return connection


def initialize(database_path: str | Path | None = None) -> None:
    with _connect(database_path):
        pass


def _prediction_record(record: dict[str, Any], *, topic: str | None = None) -> dict[str, Any]:
    try:
        transaction_id = record["transaction_id"]
        prediction = record.get("prediction", record.get("predicted_label"))
        raw_probability = record["fraud_probability"]
        if isinstance(raw_probability, bool) or not isinstance(raw_probability, (int, float)):
            raise ValueError("fraud_probability must be numeric")
        probability = float(raw_probability)
        model_version = record["model_version"]
        timestamp = record.get("processing_timestamp_utc", record.get("prediction_timestamp", record.get("timestamp_utc")))
        partition, offset = record.get("kafka_partition"), record.get("kafka_offset")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"prediction record is missing required Stage 1 fields: {exc}") from exc
    if not isinstance(transaction_id, str) or not transaction_id.strip():
        raise ValueError("transaction_id must be a non-empty string")
    if isinstance(prediction, bool) or not isinstance(prediction, int) or prediction not in (0, 1):
        raise ValueError("prediction must be 0 or 1")
    if not 0 <= probability <= 1:
        raise ValueError("fraud_probability must be between 0 and 1")
    if not isinstance(timestamp, str) or not timestamp.strip() or not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("prediction timestamp and model_version are required")
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed_timestamp.tzinfo is None:
            raise ValueError("timezone required")
    except ValueError as exc:
        raise ValueError("prediction timestamp must be timezone-aware ISO-8601") from exc
    risk_score = record.get("risk_score")
    if risk_score is not None and (isinstance(risk_score, bool) or not isinstance(risk_score, (int, float)) or not 0 <= risk_score <= 100):
        raise ValueError("risk_score must be a finite number between 0 and 100")
    risk_level = record.get("risk_level")
    if risk_level is not None and (not isinstance(risk_level, str) or risk_level not in {"LOW", "MEDIUM", "HIGH"}):
        raise ValueError("risk_level must be LOW, MEDIUM, or HIGH")
    if (partition is None) != (offset is None):
        raise ValueError("Kafka partition and offset must both be present or both be absent")
    if partition is not None and (isinstance(partition, bool) or not isinstance(partition, int) or partition < 0):
        raise ValueError("kafka_partition must be an integer >= 0")
    if offset is not None and (isinstance(offset, bool) or not isinstance(offset, int) or offset < 0):
        raise ValueError("kafka_offset must be an integer >= 0")
    data_origin = record.get("data_origin", "unspecified")
    if not isinstance(data_origin, str) or not data_origin.strip():
        raise ValueError("data_origin must be a non-empty string")
    return {"transaction_id": transaction_id, "prediction": int(prediction), "fraud_probability": probability,
            "risk_score": risk_score, "risk_level": risk_level,
            "prediction_timestamp": timestamp, "model_version": model_version,
            "data_origin": data_origin,
            "kafka_topic": topic or record.get("kafka_topic") or "fraud-transactions",
            "kafka_partition": partition, "kafka_offset": offset}


def register_prediction(record: dict[str, Any], database_path: str | Path | None = None,
                        *, topic: str | None = None) -> bool:
    """Register once. Returns False for an identical replay; never rewrites an original prediction."""
    item = _prediction_record(record, topic=topic)
    with _connect(database_path) as db:
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute("SELECT * FROM predictions WHERE transaction_id=?", (item["transaction_id"],)).fetchone()
        if existing:
            stable_fields = ("prediction", "fraud_probability", "risk_score", "risk_level", "model_version",
                             "data_origin", "kafka_topic", "kafka_partition", "kafka_offset")
            if any(existing[field] != item[field] for field in stable_fields):
                db.execute("INSERT OR IGNORE INTO prediction_conflicts(transaction_id,stored_record,incoming_record,observed_at) VALUES(?,?,?,?)",
                           (item["transaction_id"], json.dumps(dict(existing)), json.dumps(item), _now()))
                db.commit()
                raise PredictionConflictError(f"different prediction already registered for {item['transaction_id']}")
            return False
        try:
            db.execute("""INSERT INTO predictions(transaction_id,prediction,fraud_probability,risk_score,risk_level,
                prediction_timestamp,model_version,data_origin,kafka_topic,kafka_partition,kafka_offset,registered_at)
                VALUES(:transaction_id,:prediction,:fraud_probability,:risk_score,:risk_level,:prediction_timestamp,
                :model_version,:data_origin,:kafka_topic,:kafka_partition,:kafka_offset,:registered_at)""",
                {**item, "registered_at": _now()})
        except sqlite3.IntegrityError as exc:
            stored = db.execute("""SELECT * FROM predictions WHERE kafka_topic=? AND kafka_partition=? AND kafka_offset=?""",
                                (item["kafka_topic"], item["kafka_partition"], item["kafka_offset"])).fetchone()
            if stored:
                db.execute("INSERT OR IGNORE INTO prediction_conflicts(transaction_id,stored_record,incoming_record,observed_at) VALUES(?,?,?,?)",
                           (item["transaction_id"], json.dumps(dict(stored)), json.dumps(item), _now()))
                db.commit()
            raise PredictionConflictError(f"prediction uniqueness conflict for {item['transaction_id']}: {exc}") from exc
        label = db.execute("SELECT * FROM labels WHERE transaction_id=?", (item["transaction_id"],)).fetchone()
        if label and label["status"] == "pending":
            if item["data_origin"].startswith("synthetic") and label["source_type"] == "trusted_verified":
                db.execute("UPDATE labels SET status='rejected' WHERE transaction_id=?", (item["transaction_id"],))
                db.execute("UPDATE label_submissions SET disposition='rejected_synthetic_origin' WHERE submission_id=?",
                           (label["submission_id"],))
            else:
                db.execute("UPDATE labels SET status='resolved' WHERE transaction_id=?", (item["transaction_id"],))
                db.execute("UPDATE label_submissions SET disposition=? WHERE submission_id=?",
                           (_accepted_disposition(label["source_type"]), label["submission_id"]))
        resolved = db.execute("SELECT 1 FROM labels WHERE transaction_id=? AND status='resolved'", (item["transaction_id"],)).fetchone()
        if resolved:
            db.execute("UPDATE predictions SET label_status='resolved' WHERE transaction_id=?", (item["transaction_id"],))
    return True


def _accepted_disposition(source_type: str) -> str:
    return {"trusted_verified": "accepted_genuine", "synthetic_test": "accepted_synthetic",
            "untrusted": "accepted_untrusted"}[source_type]


def _audit_rejected_submission(*, transaction_id: str, outcome: str, label_source: str,
                               source_type: str, confirmed_at: str, submission_id: str,
                               disposition: str, database_path: str | Path | None) -> None:
    with _connect(database_path) as db:
        db.execute("BEGIN IMMEDIATE")
        exists = db.execute("SELECT 1 FROM label_submissions WHERE submission_id=?", (submission_id,)).fetchone()
        if exists:
            db.execute("INSERT INTO submission_replays(submission_id,disposition,observed_at) VALUES(?,?,?)",
                       (submission_id, "rejected_submission_id_reuse", _now()))
        else:
            db.execute("INSERT INTO label_submissions VALUES(?,?,?,?,?,?,?,?,?)",
                       (submission_id, transaction_id, int(outcome == "fraud"), label_source, source_type,
                        confirmed_at, disposition, _now(), None))


def submit_label(*, transaction_id: str, outcome: str, label_source: str, source_type: str,
                 confirmed_at: str, submission_id: str, database_path: str | Path | None = None,
                 authorization_token: str | None = None) -> dict[str, Any]:
    """Submit delayed truth with immutable first-label semantics and a complete submission audit."""
    if not isinstance(transaction_id, str) or not transaction_id.strip():
        raise LabelValidationError("transaction_id must be a non-empty string")
    if not isinstance(outcome, str) or outcome not in {"fraud", "legitimate"}:
        raise LabelValidationError("outcome must be 'fraud' or 'legitimate'")
    if not isinstance(source_type, str) or source_type not in {"trusted_verified", "untrusted", "synthetic_test"}:
        raise LabelValidationError("source_type must be trusted_verified, untrusted, or synthetic_test")
    if not isinstance(label_source, str) or not label_source.strip() or len(label_source) > 200:
        raise LabelValidationError("label_source must be a non-empty string of at most 200 characters")
    if not isinstance(submission_id, str) or not submission_id.strip() or len(submission_id) < 8 or len(submission_id) > 100:
        raise LabelValidationError("submission_id must be between 8 and 100 characters")
    try:
        confirmed = datetime.fromisoformat(confirmed_at.replace("Z", "+00:00"))
        if confirmed.tzinfo is None:
            raise ValueError("timezone required")
    except (AttributeError, ValueError) as exc:
        raise LabelValidationError("confirmed_at must be an ISO-8601 timestamp with timezone") from exc
    normalized_time = confirmed.astimezone(timezone.utc).isoformat()
    if source_type == "trusted_verified":
        trusted_token = config.trusted_label_token()
        if label_source not in config.trusted_label_sources():
            _audit_rejected_submission(transaction_id=transaction_id, outcome=outcome, label_source=label_source,
                source_type=source_type, confirmed_at=normalized_time, submission_id=submission_id,
                disposition="rejected_untrusted_source", database_path=database_path)
            raise UnauthorizedLabelSource("label_source is not in FRAUD_TRUSTED_LABEL_SOURCES")
        if not trusted_token or not authorization_token or not hmac.compare_digest(trusted_token, authorization_token):
            _audit_rejected_submission(transaction_id=transaction_id, outcome=outcome, label_source=label_source,
                source_type=source_type, confirmed_at=normalized_time, submission_id=submission_id,
                disposition="rejected_missing_authorization", database_path=database_path)
            raise UnauthorizedLabelSource("valid trusted-source authorization is required")
    target = int(outcome == "fraud")
    with _connect(database_path) as db:
        db.execute("BEGIN IMMEDIATE")
        previous_submission = db.execute("SELECT * FROM label_submissions WHERE submission_id=?", (submission_id,)).fetchone()
        submission_values = (transaction_id, target, label_source, source_type, normalized_time)
        if previous_submission:
            prior_values = tuple(previous_submission[key] for key in ("transaction_id", "outcome", "label_source", "source_type", "confirmed_at"))
            if prior_values != submission_values:
                db.execute("INSERT INTO submission_replays(submission_id,disposition,observed_at) VALUES(?,?,?)",
                           (submission_id, "submission_id_conflict", _now()))
                db.commit()
                raise SubmissionIdConflict("submission_id was already used for a different label payload")
            db.execute("INSERT INTO submission_replays(submission_id,disposition,observed_at) VALUES(?,?,?)",
                       (submission_id, "duplicate_submission", _now()))
            return {"submission_id": submission_id, "transaction_id": transaction_id,
                    "disposition": "duplicate_submission", "idempotent": True}
        prediction = db.execute("SELECT data_origin FROM predictions WHERE transaction_id=?", (transaction_id,)).fetchone()
        if prediction and prediction["data_origin"].startswith("synthetic") and source_type == "trusted_verified":
            db.execute("INSERT INTO label_submissions VALUES(?,?,?,?,?,?,?,?,?)",
                       (submission_id, transaction_id, target, label_source, source_type, normalized_time,
                        "rejected_synthetic_origin", _now(), None))
            db.commit()
            raise UnauthorizedLabelSource("synthetic demo transactions cannot receive trusted_verified labels")
        prior_label = db.execute("SELECT * FROM labels WHERE transaction_id=?", (transaction_id,)).fetchone()
        if prior_label:
            disposition = "duplicate_identical" if prior_label["outcome"] == target else "conflict"
            db.execute("INSERT INTO label_submissions VALUES(?,?,?,?,?,?,?,?,?)",
                       (submission_id, transaction_id, target, label_source, source_type, normalized_time,
                        disposition, _now(), prior_label["submission_id"]))
            return {"submission_id": submission_id, "transaction_id": transaction_id,
                    "disposition": disposition, "idempotent": disposition == "duplicate_identical"}
        status = "resolved" if prediction else "pending"
        disposition = (_accepted_disposition(source_type) if prediction else "pending_prediction")
        db.execute("INSERT INTO labels VALUES(?,?,?,?,?,?,?,?)",
                   (transaction_id, target, label_source, source_type, normalized_time, status, submission_id, _now()))
        db.execute("INSERT INTO label_submissions VALUES(?,?,?,?,?,?,?,?,?)",
                   (submission_id, transaction_id, target, label_source, source_type, normalized_time,
                    disposition, _now(), None))
        if prediction:
            db.execute("UPDATE predictions SET label_status='resolved' WHERE transaction_id=?", (transaction_id,))
        return {"submission_id": submission_id, "transaction_id": transaction_id,
                "disposition": disposition, "idempotent": False}


def get_prediction(transaction_id: str, database_path: str | Path | None = None) -> dict[str, Any] | None:
    path = Path(database_path or config.STAGE2_DB_PATH)
    if not path.exists():
        return None
    with sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True, factory=_ClosingConnection) as db:
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT * FROM predictions WHERE transaction_id=?", (transaction_id,)).fetchone()
        return dict(row) if row else None


def get_label(transaction_id: str, database_path: str | Path | None = None) -> dict[str, Any] | None:
    path = Path(database_path or config.STAGE2_DB_PATH)
    if not path.exists():
        return None
    with sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True, factory=_ClosingConnection) as db:
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT transaction_id,outcome,label_source,source_type,confirmed_at,status,submission_id FROM labels WHERE transaction_id=?", (transaction_id,)).fetchone()
        if not row:
            return None
        return {**dict(row), "outcome": "fraud" if row["outcome"] == 1 else "legitimate"}


def get_submissions(transaction_id: str, database_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Return the recorded label-submission audit trail for one stable transaction ID."""
    path = Path(database_path or config.STAGE2_DB_PATH)
    if not path.exists():
        return []
    with sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True, factory=_ClosingConnection) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("""SELECT submission_id,transaction_id,outcome,label_source,source_type,
            confirmed_at,disposition,submitted_at,duplicate_of FROM label_submissions
            WHERE transaction_id=? ORDER BY submitted_at,submission_id""", (transaction_id,)).fetchall()
        return [{**dict(row), "outcome": "fraud" if row["outcome"] == 1 else "legitimate"} for row in rows]


def get_stats(database_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(database_path or config.STAGE2_DB_PATH)
    counts = {"predictions_registered": 0, "pending_labels": 0, "confirmed_genuine_labels": 0,
              "synthetic_test_labels": 0, "duplicate_submissions": 0, "conflicting_labels": 0,
              "unmatched_pending_submissions": 0, "prediction_conflicts": 0}
    trusted_sources = config.trusted_label_sources()
    trusted_enabled = bool(trusted_sources and config.trusted_label_token())
    if not path.exists():
        return {**counts, "genuine_source_configured": trusted_enabled,
                "genuine_labels_available": False}
    with sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True, factory=_ClosingConnection) as db:
        db.row_factory = sqlite3.Row
        counts["predictions_registered"] = db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        counts["pending_labels"] = db.execute("SELECT COUNT(*) FROM predictions WHERE label_status='pending'").fetchone()[0]
        counts["unmatched_pending_submissions"] = db.execute("SELECT COUNT(*) FROM labels WHERE status='pending'").fetchone()[0]
        counts["synthetic_test_labels"] = db.execute("SELECT COUNT(*) FROM labels WHERE source_type='synthetic_test' AND status='resolved'").fetchone()[0]
        counts["duplicate_submissions"] = (db.execute("SELECT COUNT(*) FROM label_submissions WHERE disposition='duplicate_identical'").fetchone()[0]
                                                + db.execute("SELECT COUNT(*) FROM submission_replays WHERE disposition='duplicate_submission'").fetchone()[0])
        counts["conflicting_labels"] = db.execute("SELECT COUNT(*) FROM label_submissions WHERE disposition='conflict'").fetchone()[0]
        counts["prediction_conflicts"] = db.execute("SELECT COUNT(*) FROM prediction_conflicts").fetchone()[0]
        genuine = 0
        if trusted_enabled:
            placeholders = ",".join("?" for _ in trusted_sources)
            genuine = db.execute(f"""SELECT COUNT(*) FROM labels l JOIN predictions p USING(transaction_id)
                WHERE l.source_type='trusted_verified' AND l.status='resolved'
                AND p.data_origin NOT LIKE 'synthetic%' AND l.label_source IN ({placeholders})""",
                tuple(sorted(trusted_sources))).fetchone()[0]
        counts["confirmed_genuine_labels"] = genuine
    return {**counts, "genuine_source_configured": trusted_enabled,
            "genuine_labels_available": genuine > 0 if trusted_enabled else False}


def replay_prediction_jsonl(jsonl_path: str | Path, database_path: str | Path | None = None) -> dict[str, int]:
    """Idempotently import existing Stage 1 JSONL without editing the source file."""
    counts = {"registered": 0, "duplicates": 0, "conflicts": 0, "errors": 0}
    with Path(jsonl_path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                inserted = register_prediction(record, database_path)
                counts["registered" if inserted else "duplicates"] += 1
            except PredictionConflictError:
                counts["conflicts"] += 1
            except (json.JSONDecodeError, ValueError, sqlite3.Error):
                counts["errors"] += 1
    return counts
