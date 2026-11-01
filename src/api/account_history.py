"""Persistent, strictly prior account activity for API transaction scoring."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "account_history.sqlite3"


class IdempotencyConflict(ValueError):
    """An idempotency key was reused for different transaction data."""


class AccountHistoryUnavailable(RuntimeError):
    """Persistent account history could not safely complete a database operation."""


class AccountHistoryStore:
    """SQLite history store; inference and history insertion share one write transaction."""

    def __init__(self, database_path: str | Path | None = None):
        configured = os.environ.get("FRAUD_ACCOUNT_HISTORY_DB_PATH", "").strip()
        self.database_path = Path(database_path or configured or DEFAULT_DATABASE_PATH).expanduser()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.database_path, timeout=10)) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS account_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                idempotency_key TEXT UNIQUE,
                request_fingerprint TEXT NOT NULL,
                name_orig TEXT NOT NULL,
                name_dest TEXT NOT NULL,
                step INTEGER NOT NULL,
                amount REAL NOT NULL,
                created_at_utc TEXT NOT NULL,
                event_kind TEXT NOT NULL CHECK(event_kind IN ('prediction','warmup')),
                prediction INTEGER CHECK(prediction IN (0,1)),
                fraud_probability REAL CHECK(fraud_probability >= 0 AND fraud_probability <= 1),
                risk_score REAL,
                risk_level TEXT,
                model_version TEXT,
                feature_values_json TEXT,
                history_status_json TEXT
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_account_transactions_sender_step ON account_transactions(name_orig, step, id)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_account_transactions_dest_step ON account_transactions(name_dest, step, id)")
            db.commit()

    @staticmethod
    def _fingerprint(transaction: dict[str, Any]) -> str:
        encoded = json.dumps(transaction, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _role_features(rows: list[sqlite3.Row], prefix: str, step: int) -> dict[str, float | int]:
        steps = [int(row["step"]) for row in rows]
        amounts = [float(row["amount"]) for row in rows]
        count = len(rows)
        total = sum(amounts)
        result: dict[str, float | int] = {
            f"{prefix}_prev_txn_count": count,
            f"{prefix}_prev_total_amount": total,
            f"{prefix}_prev_mean_amount": total / count if count else float("nan"),
            f"{prefix}_prev_max_amount": max(amounts) if count else float("nan"),
            f"{prefix}_hours_since_prev_txn": step - max(steps) if count else float("nan"),
        }
        for window in (1, 24, 168):
            selected = [amount for row_step, amount in zip(steps, amounts) if row_step > step - window]
            result[f"{prefix}_txn_count_last_{window}h"] = len(selected)
            if prefix == "dest" and window in (24, 168):
                result[f"dest_amount_sum_last_{window}h"] = sum(selected)
        return result

    @classmethod
    def _history_features(cls, db: sqlite3.Connection, transaction: dict[str, Any]) -> tuple[dict, dict]:
        step = int(transaction["step"])
        sender_rows = db.execute("""SELECT step, amount FROM account_transactions
            WHERE name_orig=? AND step<=? ORDER BY step, id""",
            (transaction["nameOrig"], step)).fetchall()
        destination_rows = db.execute("""SELECT step, amount FROM account_transactions
            WHERE name_dest=? AND step<=? ORDER BY step, id""",
            (transaction["nameDest"], step)).fetchall()
        features = cls._role_features(sender_rows, "sender", step)
        features.update(cls._role_features(destination_rows, "dest", step))
        amount = float(transaction["amount"])
        sender_mean = features["sender_prev_mean_amount"]
        destination_mean = features["dest_prev_mean_amount"]
        features["amount_vs_sender_hist_mean"] = amount / sender_mean if sender_mean > 0 else float("nan")
        features["amount_vs_dest_hist_mean"] = amount / destination_mean if destination_mean > 0 else float("nan")
        status = {
            "sender": {"prior_transaction_count": len(sender_rows),
                       "state": "history_available" if sender_rows else "no_prior_transactions"},
            "destination": {"prior_transaction_count": len(destination_rows),
                            "state": "history_available" if destination_rows else "no_prior_transactions"},
        }
        return features, status

    @staticmethod
    def _cached(db: sqlite3.Connection, key: str, fingerprint: str) -> dict | None:
        row = db.execute("""SELECT request_fingerprint, event_kind, prediction, fraud_probability, risk_score,
                risk_level, model_version, feature_values_json, history_status_json
            FROM account_transactions WHERE idempotency_key=?""", (key,)).fetchone()
        if row is None:
            return None
        if row["request_fingerprint"] != fingerprint:
            raise IdempotencyConflict("This idempotency key was already used for different transaction data.")
        if row["event_kind"] != "prediction":
            raise IdempotencyConflict("This idempotency key belongs to a stored history event, not a prediction.")
        return {
            "result": {"prediction": row["prediction"], "fraud_probability": row["fraud_probability"],
                       "risk_score": row["risk_score"], "risk_level": row["risk_level"],
                       "model_version": row["model_version"]},
            "features": json.loads(row["feature_values_json"]),
            "history_status": json.loads(row["history_status_json"]),
            "replayed": True,
        }

    def process(self, transaction: dict[str, Any], *, idempotency_key: str | None,
                feature_builder: Callable[[dict], tuple[pd.DataFrame, dict]],
                predictor: Callable[[pd.DataFrame], dict]) -> dict:
        """Score against stored prior events, then append this event atomically."""
        key = idempotency_key.strip() if isinstance(idempotency_key, str) else None
        if key == "":
            key = None
        if key is not None and len(key) > 200:
            raise ValueError("Idempotency-Key must be at most 200 characters.")
        storage_key = f"prediction:{key}" if key is not None else None
        fingerprint = self._fingerprint(transaction)
        with closing(sqlite3.connect(self.database_path, timeout=10, isolation_level=None)) as db:
            db.row_factory = sqlite3.Row
            db.execute("BEGIN IMMEDIATE")
            try:
                if storage_key is not None:
                    cached = self._cached(db, storage_key, fingerprint)
                    if cached is not None:
                        db.commit()
                        return cached
                history, history_status = self._history_features(db, transaction)
                model_features, feature_values = feature_builder(history)
                result = predictor(model_features)
                db.execute("""INSERT INTO account_transactions(
                    event_id,idempotency_key,request_fingerprint,name_orig,name_dest,step,amount,
                    created_at_utc,event_kind,prediction,fraud_probability,risk_score,risk_level,model_version,
                    feature_values_json,history_status_json)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (str(uuid.uuid4()), storage_key, fingerprint, transaction["nameOrig"], transaction["nameDest"],
                     int(transaction["step"]), float(transaction["amount"]),
                     datetime.now(timezone.utc).isoformat(), "prediction", int(result["prediction"]),
                     float(result["fraud_probability"]), float(result["risk_score"]),
                     str(result["risk_level"]), str(result["model_version"]),
                     json.dumps(feature_values, allow_nan=False),
                     json.dumps(history_status, allow_nan=False)))
                db.commit()
                return {"result": result, "features": feature_values,
                        "history_status": history_status, "replayed": False}
            except sqlite3.Error as exc:
                db.rollback()
                raise AccountHistoryUnavailable("Persistent account history is unavailable.") from exc
            except Exception:
                db.rollback()
                raise

    def warmup(self, transactions: list[dict[str, Any]], *, batch_key: str | None = None) -> int:
        """Persist explicitly supplied prior events without inventing prediction labels."""
        key = batch_key.strip() if isinstance(batch_key, str) else None
        if key == "":
            key = None
        if key is not None and len(key) > 180:
            raise ValueError("Idempotency-Key must be at most 180 characters for a history warmup.")
        inserted = 0
        with closing(sqlite3.connect(self.database_path, timeout=10, isolation_level=None)) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                for index, transaction in enumerate(transactions):
                    event_key = f"warmup:{key}:{index}" if key is not None else None
                    fingerprint = self._fingerprint(transaction)
                    if event_key is not None:
                        existing = db.execute("SELECT request_fingerprint FROM account_transactions WHERE idempotency_key=?",
                                              (event_key,)).fetchone()
                        if existing:
                            if existing[0] != fingerprint:
                                raise IdempotencyConflict("This history warmup key was already used for different transaction data.")
                            continue
                    db.execute("""INSERT INTO account_transactions(
                        event_id,idempotency_key,request_fingerprint,name_orig,name_dest,step,amount,
                        created_at_utc,event_kind)
                        VALUES(?,?,?,?,?,?,?,?,?)""",
                        (str(uuid.uuid4()), event_key, fingerprint, transaction["nameOrig"], transaction["nameDest"],
                         int(transaction["step"]), float(transaction["amount"]),
                         datetime.now(timezone.utc).isoformat(), "warmup"))
                    inserted += 1
                db.commit()
                return inserted
            except sqlite3.Error as exc:
                db.rollback()
                raise AccountHistoryUnavailable("Persistent account history is unavailable.") from exc
            except Exception:
                db.rollback()
                raise

    def preview(self, transaction: dict[str, Any]) -> tuple[dict, dict]:
        """Read genuine earlier history without recording the pending transaction."""
        if not self.database_path.is_file():
            raise AccountHistoryUnavailable("Persistent account history database is missing.")
        uri = f"file:{self.database_path.resolve().as_posix()}?mode=ro"
        try:
            with closing(sqlite3.connect(uri, uri=True, timeout=3)) as db:
                db.row_factory = sqlite3.Row
                return self._history_features(db, transaction)
        except sqlite3.Error as exc:
            raise AccountHistoryUnavailable("Persistent account history is unavailable.") from exc
