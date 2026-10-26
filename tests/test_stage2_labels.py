"""Stage 2 persistence and delayed-label lifecycle tests using isolated SQLite files."""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError
from fastapi.testclient import TestClient

from src.labels import api, config as label_config, store
from src.labels.api import LabelSubmission


class Stage2LabelStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.db = Path(self.temporary.name) / "labels.sqlite3"
        store.initialize(self.db)
        self.prediction = {"transaction_id": "TX-0001", "prediction": 1, "fraud_probability": 0.73,
                           "risk_score": 73.0, "risk_level": "HIGH", "processing_timestamp_utc": "2026-10-25T10:00:00+00:00",
                           "model_version": "V2", "data_origin": "synthetic_paysim_demo",
                           "kafka_partition": 0, "kafka_offset": 13}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def label(self, *, txn: str = "TX-0001", outcome: str = "fraud", source_type: str = "synthetic_test",
              source: str = "manual_demo_review", submission: str = "submission-0001") -> dict:
        return store.submit_label(transaction_id=txn, outcome=outcome, label_source=source,
                                  source_type=source_type, confirmed_at="2026-10-25T10:05:00Z",
                                  submission_id=submission, database_path=self.db)

    def test_prediction_register_retrieve_and_database_reopen(self) -> None:
        self.assertTrue(store.register_prediction(self.prediction, self.db))
        store.initialize(self.db)
        result = store.get_prediction("TX-0001", self.db)
        self.assertEqual(result["prediction"], 1)
        self.assertEqual(result["kafka_partition"], 0)
        self.assertEqual(result["kafka_offset"], 13)

    def test_kafka_offset_cannot_be_reassigned_to_another_transaction(self) -> None:
        store.register_prediction(self.prediction, self.db)
        with self.assertRaises(store.PredictionConflictError):
            store.register_prediction({**self.prediction, "transaction_id": "TX-OTHER"}, self.db)
        self.assertEqual(store.get_stats(self.db)["predictions_registered"], 1)
        self.assertEqual(store.get_stats(self.db)["prediction_conflicts"], 1)

    def test_delayed_label_matches_by_transaction_id_and_preserves_prediction(self) -> None:
        store.register_prediction(self.prediction, self.db)
        result = self.label()
        self.assertEqual(result["disposition"], "accepted_synthetic")
        self.assertEqual(store.get_label("TX-0001", self.db)["outcome"], "fraud")
        original = store.get_prediction("TX-0001", self.db)
        self.assertEqual((original["prediction"], original["fraud_probability"], original["model_version"]),
                         (1, 0.73, "V2"))
        self.assertEqual(original["label_status"], "resolved")

    def test_early_label_remains_pending_then_matches_when_prediction_arrives(self) -> None:
        result = self.label(txn="EARLY-1", submission="early-label-0001")
        self.assertEqual(result["disposition"], "pending_prediction")
        self.assertEqual(store.get_stats(self.db)["unmatched_pending_submissions"], 1)
        prediction = {**self.prediction, "transaction_id": "EARLY-1", "kafka_offset": 14}
        store.register_prediction(prediction, self.db)
        self.assertEqual(store.get_label("EARLY-1", self.db)["status"], "resolved")
        self.assertEqual(store.get_prediction("EARLY-1", self.db)["label_status"], "resolved")

    def test_duplicate_submission_id_and_identical_outcome_are_idempotent(self) -> None:
        store.register_prediction(self.prediction, self.db)
        self.label()
        retry = self.label()
        same_outcome = self.label(submission="submission-0002")
        self.assertEqual(retry["disposition"], "duplicate_submission")
        self.assertTrue(retry["idempotent"])
        self.assertEqual(same_outcome["disposition"], "duplicate_identical")
        self.assertEqual(store.get_stats(self.db)["duplicate_submissions"], 2)
        db = sqlite3.connect(self.db)
        try:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM labels WHERE transaction_id='TX-0001'").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM label_submissions WHERE submission_id='submission-0001'").fetchone()[0], 1)
        finally:
            db.close()

    def test_same_submission_id_with_different_outcome_is_rejected(self) -> None:
        store.register_prediction(self.prediction, self.db)
        self.label(outcome="legitimate")
        with self.assertRaises(store.SubmissionIdConflict):
            self.label(outcome="fraud")
        self.assertEqual(store.get_label("TX-0001", self.db)["outcome"], "legitimate")
        self.assertEqual(store.get_stats(self.db)["conflicting_labels"], 0)

    def test_same_submission_id_with_different_transaction_id_is_rejected(self) -> None:
        self.label(txn="EARLY-1")
        with self.assertRaises(store.SubmissionIdConflict):
            self.label(txn="EARLY-2")
        self.assertEqual(store.get_label("EARLY-1", self.db)["outcome"], "fraud")
        self.assertIsNone(store.get_label("EARLY-2", self.db))

    def test_equivalent_timezone_timestamps_are_idempotent_after_utc_normalization(self) -> None:
        payload = {"transaction_id": "EARLY-TZ-1", "outcome": "legitimate", "label_source": "manual_test",
                   "source_type": "synthetic_test", "submission_id": "timezone-submit-01",
                   "database_path": self.db}
        first = store.submit_label(**payload, confirmed_at="2026-10-25T10:05:00Z")
        retry = store.submit_label(**payload, confirmed_at="2026-10-25T15:35:00+05:30")
        self.assertEqual(first["disposition"], "pending_prediction")
        self.assertEqual(retry["disposition"], "duplicate_submission")
        self.assertTrue(retry["idempotent"])
        self.assertEqual(store.get_label("EARLY-TZ-1", self.db)["confirmed_at"], "2026-10-25T10:05:00+00:00")

    def test_conflict_is_audited_and_does_not_replace_first_label(self) -> None:
        store.register_prediction(self.prediction, self.db)
        self.label()
        result = self.label(outcome="legitimate", submission="submission-0002")
        self.assertEqual(result["disposition"], "conflict")
        self.assertEqual(store.get_label("TX-0001", self.db)["outcome"], "fraud")
        self.assertEqual(store.get_stats(self.db)["conflicting_labels"], 1)

    def test_bad_outcomes_sources_and_malformed_timestamps_rejected(self) -> None:
        for kwargs in ({"outcome": "maybe"}, {"source_type": "unknown"}):
            payload = {"transaction_id": "bad", "outcome": "fraud", "label_source": "manual",
                       "source_type": "untrusted", "confirmed_at": "not-a-date", "submission_id": "bad-submit-01"}
            payload.update(kwargs)
            with self.assertRaises(store.LabelValidationError):
                store.submit_label(**payload, database_path=self.db)
        with self.assertRaises(store.UnauthorizedLabelSource):
            self.label(source_type="trusted_verified", source="unconfigured-authority")
        with self.assertRaises(ValidationError):
            LabelSubmission.model_validate({"transaction_id": "x", "outcome": "unknown", "label_source": "manual",
                "source_type": "synthetic_test", "confirmed_at": "2026-10-25T10:00:00", "submission_id": "short"})

    def test_synthetic_test_and_untrusted_or_unmatched_labels_are_not_genuine(self) -> None:
        store.register_prediction(self.prediction, self.db)
        self.label()
        self.label(txn="UNTRUSTED-1", source_type="untrusted", source="unverified_upload",
                   submission="untrusted-0001")
        self.assertEqual(store.get_stats(self.db)["confirmed_genuine_labels"], 0)
        self.assertFalse(store.get_stats(self.db)["genuine_labels_available"])

    def test_trusted_source_counts_only_authorized_non_synthetic_prediction(self) -> None:
        with patch.dict(os.environ, {"FRAUD_TRUSTED_LABEL_SOURCES": "audit_partner",
                                    "FRAUD_TRUSTED_LABEL_TOKEN": "unit-test-trusted-token-0123456789"}):
            store.register_prediction({**self.prediction, "transaction_id": "GENUINE-1", "data_origin": "unknown"}, self.db)
            store.register_prediction({**self.prediction, "kafka_offset": 14}, self.db)
            store.submit_label(transaction_id="GENUINE-1", outcome="legitimate", label_source="audit_partner",
                               source_type="trusted_verified", confirmed_at="2026-10-25T10:05:00Z",
                               submission_id="verified-0001", database_path=self.db,
                               authorization_token="unit-test-trusted-token-0123456789")
            stats = store.get_stats(self.db)
            self.assertEqual(stats["confirmed_genuine_labels"], 1)
            self.assertTrue(stats["genuine_labels_available"])
            with self.assertRaises(store.UnauthorizedLabelSource):
                store.submit_label(transaction_id="TX-0001", outcome="fraud", label_source="audit_partner",
                                   source_type="trusted_verified", confirmed_at="2026-10-25T10:05:00Z",
                                   submission_id="verified-0002", database_path=self.db,
                                   authorization_token="unit-test-trusted-token-0123456789")

    def test_prediction_replay_deduplicates_records_without_editing_source(self) -> None:
        jsonl = Path(self.temporary.name) / "predictions.jsonl"
        line = json.dumps(self.prediction) + "\n"
        jsonl.write_text(line + line, encoding="utf-8")
        counts = store.replay_prediction_jsonl(jsonl, self.db)
        self.assertEqual(counts, {"registered": 1, "duplicates": 1, "conflicts": 0, "errors": 0})
        self.assertEqual(jsonl.read_text(encoding="utf-8"), line + line)
        self.assertEqual(store.get_stats(self.db)["predictions_registered"], 1)

    def test_replay_conflict_is_reported_without_replacing_first_prediction(self) -> None:
        jsonl = Path(self.temporary.name) / "conflict.jsonl"
        jsonl.write_text(json.dumps(self.prediction) + "\n" +
                         json.dumps({**self.prediction, "fraud_probability": 0.2, "risk_score": 20.0}) + "\n",
                         encoding="utf-8")
        counts = store.replay_prediction_jsonl(jsonl, self.db)
        self.assertEqual(counts, {"registered": 1, "duplicates": 0, "conflicts": 1, "errors": 0})
        self.assertEqual(store.get_prediction("TX-0001", self.db)["fraud_probability"], 0.73)
        self.assertEqual(store.get_stats(self.db)["prediction_conflicts"], 1)

    def test_local_api_accepts_synthetic_test_label_and_rejects_invalid_outcome(self) -> None:
        store.register_prediction(self.prediction, self.db)
        with patch.object(label_config, "STAGE2_DB_PATH", self.db):
            with TestClient(api.app) as client:
                payload = {"transaction_id": "TX-0001", "outcome": "legitimate", "label_source": "manual_test",
                           "source_type": "synthetic_test", "confirmed_at": "2026-10-25T10:05:00Z",
                           "submission_id": "api-submission-01"}
                accepted = client.post("/labels", json=payload)
                retry = client.post("/labels", json={**payload, "confirmed_at": "2026-10-25T15:35:00+05:30"})
                rejected = client.post("/labels", json={**payload, "outcome": "unknown",
                                                           "submission_id": "api-submission-02"})
        self.assertEqual(accepted.status_code, 201)
        self.assertEqual(accepted.json()["disposition"], "accepted_synthetic")
        self.assertEqual(retry.status_code, 201)
        self.assertEqual(retry.json()["disposition"], "duplicate_submission")
        self.assertTrue(retry.json()["idempotent"])
        self.assertEqual(store.get_label("TX-0001", self.db)["outcome"], "legitimate")
        self.assertEqual(rejected.status_code, 422)


if __name__ == "__main__":
    unittest.main()
