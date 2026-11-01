"""Persistent prior-event history used by API scoring."""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from fastapi.testclient import TestClient
from src.api import app
from src.api.account_history import AccountHistoryStore, IdempotencyConflict


class PersistentAccountHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary.name) / "account_history.sqlite3"
        self.original_store = app.account_history_store
        self.original_state = app.account_state
        self.original_log_path = app.LOG_PATH
        app.account_history_store = AccountHistoryStore(self.database)
        app.account_state = app.AccountState()
        app.LOG_PATH = Path(self.temporary.name) / "predictions.jsonl"

    def tearDown(self):
        app.account_history_store = self.original_store
        app.account_state = self.original_state
        app.LOG_PATH = self.original_log_path
        self.temporary.cleanup()

    @staticmethod
    def transaction(transaction_id: str, *, step: int, amount: float,
                    sender: str = "S1", destination: str = "D1", tx_type: str = "TRANSFER"):
        return app.TransactionRequest(
            step=step, type=tx_type, amount=amount, nameOrig=sender,
            oldbalanceOrg=10000.0, nameDest=destination, oldbalanceDest=500.0,
        )

    def analyze(self, transaction, key: str):
        return app.analyze_transaction(transaction, idempotency_key=key)

    def test_cold_start_has_explicit_status_and_does_not_include_current_event(self):
        transaction = self.transaction("cold", step=10, amount=40.0)
        result = self.analyze(transaction, "cold-start")
        self.assertEqual(result["history_status"]["sender"]["state"], "no_prior_transactions")
        self.assertEqual(result["history_status"]["destination"]["state"], "no_prior_transactions")
        self.assertEqual(result["features"]["sender_prev_txn_count"], 0.0)
        self.assertEqual(result["features"]["dest_prev_txn_count"], 0.0)
        self.assertIsNone(result["features"]["sender_prev_mean_amount"])
        self.assertIsNone(result["features"]["dest_prev_mean_amount"])

    def test_history_persists_when_api_store_is_recreated(self):
        first = self.transaction("first", step=10, amount=100.0, sender="S1", destination="D1")
        self.analyze(first, "first-event")

        # A new store instance models a restarted API process using the same configured SQLite file.
        app.account_history_store = AccountHistoryStore(self.database)
        second = self.transaction("second", step=11, amount=50.0, sender="S1", destination="D2")
        result = self.analyze(second, "second-event")
        features = result["features"]
        self.assertEqual(features["sender_prev_txn_count"], 1.0)
        self.assertEqual(features["sender_prev_total_amount"], 100.0)
        self.assertEqual(features["sender_prev_mean_amount"], 100.0)
        self.assertEqual(features["sender_prev_max_amount"], 100.0)
        self.assertEqual(features["sender_hours_since_prev_txn"], 1.0)
        self.assertEqual(features["amount_vs_sender_hist_mean"], 0.5)
        self.assertEqual(result["history_status"]["sender"]["prior_transaction_count"], 1)

    def test_account_history_is_separated_by_exact_sender_and_destination_ids(self):
        self.analyze(self.transaction("known", step=3, amount=80.0, sender="S-A", destination="D-A"), "known")
        result = self.analyze(
            self.transaction("unseen", step=4, amount=20.0, sender="S-B", destination="D-B"), "unseen")
        self.assertEqual(result["history_status"]["sender"]["state"], "no_prior_transactions")
        self.assertEqual(result["history_status"]["destination"]["state"], "no_prior_transactions")
        self.assertEqual(result["features"]["sender_prev_txn_count"], 0.0)
        self.assertEqual(result["features"]["dest_prev_txn_count"], 0.0)

    def test_prior_destination_history_and_window_aggregates_are_calculated(self):
        self.analyze(self.transaction("one", step=10, amount=100.0, sender="S1", destination="D1"), "one")
        self.analyze(self.transaction("two", step=11, amount=50.0, sender="S1", destination="D2"), "two")
        result = self.analyze(self.transaction("three", step=12, amount=25.0, sender="S2", destination="D1"), "three")
        features = result["features"]
        self.assertEqual(features["dest_prev_txn_count"], 1.0)
        self.assertEqual(features["dest_prev_total_amount"], 100.0)
        self.assertEqual(features["dest_txn_count_last_24h"], 1.0)
        self.assertEqual(features["dest_amount_sum_last_24h"], 100.0)
        self.assertEqual(features["amount_vs_dest_hist_mean"], 0.25)

    def test_idempotency_replay_does_not_add_the_transaction_twice(self):
        transaction = self.transaction("repeat", step=1, amount=10.0, sender="S", destination="D")
        first = self.analyze(transaction, "same-request")
        replay = self.analyze(transaction, "same-request")
        self.assertEqual(first["features"], replay["features"])
        with closing(sqlite3.connect(self.database)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM account_transactions").fetchone()[0], 1)
        with self.assertRaises(IdempotencyConflict):
            self.analyze(self.transaction("different", step=1, amount=11.0, sender="S", destination="D"),
                         "same-request")

    def test_explicit_warmup_history_is_persistent_and_carries_no_fraud_label(self):
        prior = self.transaction("warmup", step=5, amount=35.0, sender="S", destination="D")
        added = app.account_history_store.warmup([{
            "step": prior.step, "type": prior.type, "amount": prior.amount,
            "nameOrig": prior.nameOrig, "oldbalanceOrg": prior.oldbalanceOrg,
            "nameDest": prior.nameDest, "oldbalanceDest": prior.oldbalanceDest,
        }], batch_key="prior-batch")
        self.assertEqual(added, 1)
        self.assertEqual(app.account_history_store.warmup([{
            "step": prior.step, "type": prior.type, "amount": prior.amount,
            "nameOrig": prior.nameOrig, "oldbalanceOrg": prior.oldbalanceOrg,
            "nameDest": prior.nameDest, "oldbalanceDest": prior.oldbalanceDest,
        }], batch_key="prior-batch"), 0)
        result = self.analyze(self.transaction("after-warmup", step=6, amount=20.0, sender="S", destination="D2"), "after")
        self.assertEqual(result["features"]["sender_prev_txn_count"], 1.0)
        with closing(sqlite3.connect(self.database)) as db:
            row = db.execute("SELECT event_kind,prediction FROM account_transactions WHERE event_kind='warmup'").fetchone()
        self.assertEqual(row, ("warmup", None))

    def test_predict_http_contract_stays_unchanged_and_is_idempotent(self):
        client = TestClient(app.app)
        payload = {"step": 1, "type": "PAYMENT", "amount": 10.0, "nameOrig": "HTTP-S",
                   "oldbalanceOrg": 20.0, "nameDest": "HTTP-D", "oldbalanceDest": 0.0}
        first = client.post("/predict", json=payload, headers={"Idempotency-Key": "http-once"})
        second = client.post("/predict", json=payload, headers={"Idempotency-Key": "http-once"})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(set(first.json()), {"prediction", "fraud_probability", "risk_score", "risk_level", "model_version"})
        self.assertEqual(first.json(), second.json())
        with closing(sqlite3.connect(self.database)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM account_transactions").fetchone()[0], 1)

    def test_analyze_http_returns_exact_feature_row_for_user_explanation(self):
        client = TestClient(app.app)
        payload = {"step": 2, "type": "TRANSFER", "amount": 25.0, "nameOrig": "ANALYZE-S",
                   "oldbalanceOrg": 100.0, "nameDest": "ANALYZE-D", "oldbalanceDest": 0.0}
        response = client.post("/analyze", json=payload, headers={"Idempotency-Key": "analyze-once"})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["model_version"], "V2")
        self.assertEqual(len(body["feature_names"]), 34)
        self.assertEqual(list(body["features"]), body["feature_names"])
        self.assertEqual(body["features"]["sender_prev_txn_count"], 0.0)
        self.assertEqual(body["history_status"]["sender"]["state"], "no_prior_transactions")


if __name__ == "__main__":
    unittest.main()
