"""Isolated tests for the manually triggered monitoring-request worker."""
from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.retraining import request_worker
from src.retraining.v2_trainer import FINAL_FEATURES


class RetrainingRequestWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.database = self.root / "monitor.sqlite3"
        self.work_root = self.root / "worker-runs"
        self.evaluation_data = self.root / "held-out.parquet"
        self.evaluation_data.write_bytes(b"test placeholder")
        self.provenance = self.root / "evaluation-provenance.json"
        self.provenance.write_text(json.dumps({
            "source": "trusted fixture", "label_provenance": "verified fixture",
            "synthetic": False,
        }), encoding="utf-8")
        with closing(sqlite3.connect(self.database)) as db:
            db.execute("""CREATE TABLE retraining_requests(
                request_id TEXT PRIMARY KEY, request_key TEXT UNIQUE NOT NULL,
                status TEXT NOT NULL, model_version TEXT NOT NULL, trigger_reason TEXT NOT NULL,
                alert_ids_json TEXT NOT NULL, evidence_json TEXT NOT NULL, created_at TEXT NOT NULL)""")
            db.commit()

    def tearDown(self):
        self.temp.cleanup()

    def _insert(self, request_id="request-1", reason="feature_drift", *, status="requested",
                version="V2"):
        with closing(sqlite3.connect(self.database)) as db:
            db.execute("INSERT INTO retraining_requests VALUES(?,?,?,?,?,?,?,?)", (
                request_id, f"key-{request_id}", status, version, reason,
                json.dumps([f"alert-{request_id}"]), json.dumps({"requested": True}),
                "2026-10-27T00:00:00+00:00"))
            db.commit()

    def _row(self, request_id="request-1"):
        with closing(sqlite3.connect(self.database)) as db:
            return db.execute("SELECT status,evidence_json FROM retraining_requests WHERE request_id=?",
                              (request_id,)).fetchone()

    def _run(self, *, stage4=None, stage5=None, limit=1):
        with patch("src.retraining.request_worker.pd.read_parquet",
                   return_value=pd.DataFrame({"event_id": ["E1"], "transaction_id": ["T1"],
                       "step": [900], "isFraud": [0], **{feature: [0.0] for feature in FINAL_FEATURES}})):
            return request_worker.process_requests(database_path=self.database,
                work_root=self.work_root, evaluation_data=self.evaluation_data,
                evaluation_provenance=self.provenance, trusted_evaluation=True, limit=limit,
                stage4_runner=stage4 or self._stage4_success,
                stage5_evaluator=stage5 or self._stage5_success)

    @staticmethod
    def _stage4_success(**kwargs):
        if not kwargs["train_candidate"]:
            return {"status": "eligible_training_not_requested"}
        Path(kwargs["candidate_dir"]).mkdir(parents=True)
        return {"status": "candidate_trained_not_promoted", "dataset_fingerprint": "train-fingerprint"}

    @staticmethod
    def _stage5_success(**kwargs):
        assert len(kwargs["evaluation_provenance"]["evaluation_fingerprint"]) == 64
        return {"status": "candidate_approved_pending_authorization", "reason": None}

    def test_no_pending_requests(self):
        self.assertEqual(request_worker.inspect_requests(self.database), [])
        self.assertEqual(self._run(), [])

    def test_inspection_is_read_only(self):
        self._insert()
        items = request_worker.inspect_requests(self.database)
        self.assertEqual(items[0]["status"], "requested")
        self.assertEqual(self._row()[0], "requested")

    def test_supported_request_trains_and_evaluates_without_promotion(self):
        self._insert(reason="feature_drift, performance_degradation")
        active_model = self.root / "active-v2-model.json"
        active_model.write_bytes(b"unchanged active model")
        before = active_model.read_bytes()
        result = self._run()
        self.assertEqual(result[0]["status"], "evaluation_approved_pending_authorization")
        self.assertEqual(result[0]["alert_types"], ["feature_drift", "performance_degradation"])
        self.assertIn("not attempted", result[0]["deployment"])
        self.assertEqual(active_model.read_bytes(), before)
        self.assertEqual(self._row()[0], "evaluation_approved_pending_authorization")

    def test_synthetic_request_with_insufficient_trusted_labels_is_blocked(self):
        self._insert(reason="feature_drift (synthetic-only evidence; validation required before deployment)")
        calls = []
        def stage4(**kwargs):
            calls.append(kwargs)
            if not kwargs["train_candidate"]:
                return {"status": "insufficient_eligible_labels",
                        "explanation": "Requires 100 eligible labels and 10 fraud labels.",
                        "eligibility": {"eligible": 0, "synthetic_excluded": 1}}
            self.fail("Training must not occur without eligible labels")
        result = self._run(stage4=stage4, stage5=lambda **kwargs: self.fail("Stage 5 must not run"))
        self.assertEqual(result[0]["status"], "blocked_eligibility")
        self.assertEqual([call["train_candidate"] for call in calls], [False])
        self.assertEqual(self._row()[0], "blocked_eligibility")

    def test_unsupported_alert_type_is_blocked_without_training(self):
        self._insert(reason="informational_notice")
        result = self._run(stage4=lambda **kwargs: self.fail("Unsupported alert must not train"))
        self.assertEqual(result[0]["status"], "blocked_unsupported_alert")

    def test_duplicate_request_is_not_processed_twice(self):
        self._insert()
        self.assertEqual(len(self._run()), 1)
        self.assertEqual(self._run(), [])

    def test_two_workers_cannot_claim_the_same_request(self):
        self._insert()
        with ThreadPoolExecutor(max_workers=2) as pool:
            claims = list(pool.map(lambda _: request_worker._claim_next(self.database), range(2)))
        self.assertEqual(sum(claim is not None for claim in claims), 1)
        rows = request_worker.inspect_requests(self.database)
        self.assertEqual(rows[0]["status"], "processing")
        self.assertTrue(rows[0]["needs_manual_review"])

    def test_training_failure_is_recorded_and_not_retried(self):
        self._insert()
        calls = []
        def stage4(**kwargs):
            calls.append(kwargs)
            if not kwargs["train_candidate"]:
                return {"status": "eligible_training_not_requested"}
            raise RuntimeError("training fixture failure")
        result = self._run(stage4=stage4)
        self.assertEqual(result[0]["status"], "failed_training")
        self.assertIn("training fixture failure", result[0]["reason"])
        self.assertEqual(self._row()[0], "failed_training")
        self.assertEqual(self._run(stage4=stage4), [])
        self.assertEqual([call["train_candidate"] for call in calls], [False, True])

    def test_evaluation_failure_is_recorded_without_deployment(self):
        self._insert()
        def stage5(**kwargs):
            raise RuntimeError("evaluation fixture failure")
        result = self._run(stage5=stage5)
        self.assertEqual(result[0]["status"], "failed_evaluation")
        self.assertIn("evaluation fixture failure", result[0]["reason"])
        self.assertEqual(self._row()[0], "failed_evaluation")

    def test_interrupted_claim_is_visible_and_not_reprocessed(self):
        self._insert()
        self.assertIsNotNone(request_worker._claim_next(self.database))
        visible = request_worker.inspect_requests(self.database)
        self.assertEqual(visible[0]["status"], "processing")
        self.assertTrue(visible[0]["needs_manual_review"])
        self.assertEqual(self._run(), [])

    def test_untrusted_evaluation_cannot_claim_or_train(self):
        self._insert()
        with self.assertRaisesRegex(ValueError, "trusted_evaluation=True"):
            request_worker.process_requests(database_path=self.database, work_root=self.work_root,
                evaluation_data=self.evaluation_data, evaluation_provenance=self.provenance,
                trusted_evaluation=False)
        self.assertEqual(self._row()[0], "requested")

    def test_missing_evaluation_inputs_do_not_claim_request(self):
        self._insert()
        with self.assertRaisesRegex(FileNotFoundError, "Stage 5 evaluation data"):
            request_worker.process_requests(database_path=self.database,
                work_root=self.work_root, evaluation_data=self.root / "missing.parquet",
                evaluation_provenance=self.provenance, trusted_evaluation=True)
        self.assertEqual(self._row()[0], "requested")

    def test_synthetic_evaluation_cannot_claim_or_train(self):
        self._insert()
        self.provenance.write_text(json.dumps({"synthetic": True}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Synthetic evaluation"):
            self._run()
        self.assertEqual(self._row()[0], "requested")


if __name__ == "__main__":
    unittest.main()
