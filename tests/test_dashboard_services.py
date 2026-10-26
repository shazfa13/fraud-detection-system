"""Focused offline tests for safe dashboard report and API helpers."""
from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock, patch

from services import api_client, data_service


class DashboardDataServiceTests(unittest.TestCase):
    def test_prediction_registry_summary_uses_read_only_counts_and_range(self) -> None:
        import sqlite3
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "labels.sqlite3"
            with closing(sqlite3.connect(path)) as db:
                db.execute("CREATE TABLE predictions(transaction_id TEXT, prediction INTEGER, registered_at TEXT)")
                db.executemany("INSERT INTO predictions VALUES(?,?,?)", [
                    ("a", 1, "2026-01-01T00:00:00+00:00"),
                    ("b", 0, "2026-01-02T00:00:00+00:00"),
                ])
                db.commit()
            before = path.stat().st_size
            result = data_service.prediction_registry_summary(path)
            self.assertEqual((result["total"], result["predicted_fraud"], result["predicted_legitimate"]), (2, 1, 1))
            self.assertEqual(result["first_recorded_at"], "2026-01-01T00:00:00+00:00")
            self.assertEqual(result["last_recorded_at"], "2026-01-02T00:00:00+00:00")
            self.assertEqual(path.stat().st_size, before)

    def test_missing_prediction_database_is_not_created(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "missing.sqlite3"
            result = data_service.prediction_registry_summary(path)
            self.assertFalse(result["available"])
            self.assertFalse(path.exists())

    def test_stage2_label_overview_reads_provenance_without_exposing_values(self) -> None:
        import sqlite3
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "labels.sqlite3"
            with closing(sqlite3.connect(path)) as db:
                db.execute("CREATE TABLE labels(source_type TEXT,label_source TEXT,status TEXT)")
                db.execute("CREATE TABLE label_submissions(disposition TEXT)")
                db.execute("INSERT INTO labels VALUES('synthetic_test','manual_test','resolved')")
                db.execute("INSERT INTO label_submissions VALUES('duplicate_identical')")
                db.commit()
            result = data_service.stage2_label_overview(path)
            self.assertTrue(result["available"])
            self.assertEqual(result["labels"][0]["count"], 1)
            self.assertEqual(result["submissions"][0]["disposition"], "duplicate_identical")
            self.assertNotIn("token", str(result).lower())

    def test_stage6_demo_loader_reads_existing_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "models/candidates/stage6_demo/c1/training_report.json"
            report_path.parent.mkdir(parents=True)
            report_path.write_text(json.dumps({"status": "candidate_trained_not_promoted",
                                              "eligible_for_genuine_promotion": False}), encoding="utf-8")
            result, error = data_service.load_latest_stage6_demo_report(temporary)
            self.assertIsNone(error)
            self.assertFalse(result["eligible_for_genuine_promotion"])
            self.assertEqual(result["_report_path"], "models/candidates/stage6_demo/c1/training_report.json")

    def test_history_normalizes_real_fields_without_inventing_transaction_details(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_dir = root / "reports"
            report_dir.mkdir()
            (report_dir / "model_v2_predictions_log.jsonl").write_text(
                json.dumps({"timestamp_utc": "2026-01-01T00:00:00Z", "model_version": "V2",
                            "fraud_probability": 0.6, "predicted_label": 1, "risk_score": 60,
                            "risk_level": "HIGH"}) + "\n", encoding="utf-8")
            rows = data_service.load_prediction_history(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["prediction"], 1)
            self.assertEqual(rows[0]["transaction_id"], "—")
            self.assertNotIn("isFraud", rows[0])

    def test_project_paths_reject_traversal_and_allow_internal_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertIsNone(data_service.project_path(root, "../outside.json"))
            nested = root / "reports" / "result.csv"
            nested.parent.mkdir()
            nested.touch()
            self.assertEqual(data_service.project_relative_path(root, nested), "reports/result.csv")

    def test_stage3_latest_report_is_read_from_monitor_store(self) -> None:
        import sqlite3
        from src.monitoring import config as monitor_config
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "monitor.sqlite3"
            with closing(sqlite3.connect(path)) as db:
                db.execute("CREATE TABLE monitor_runs(run_id TEXT,created_at TEXT,model_version TEXT,status TEXT,report_json TEXT)")
                db.execute("INSERT INTO monitor_runs VALUES(?,?,?,?,?)",
                    ("r1", "2026-10-25T00:00:00+00:00", "V2", "insufficient_data",
                     json.dumps({"model_version": "V2", "overall_status": "insufficient_data"})))
                db.commit()
            with patch.object(monitor_config, "MONITOR_DB_PATH", path):
                report = data_service.load_stage3_monitor_report()
            self.assertEqual(report["model_version"], "V2")
            self.assertEqual(report["overall_status"], "insufficient_data")

    def test_latest_phase12_report_is_read_only_and_requires_known_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reports = Path(temporary) / "reports"
            reports.mkdir()
            report = reports / "phase12_v2_integration_evidence_20260101.json"
            report.write_text(json.dumps({"phase": 12, "status": "PASS", "completed_at_utc": "now"}), encoding="utf-8")
            before = report.read_bytes()
            result = data_service.load_latest_phase12_evidence(temporary)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(report.read_bytes(), before)


class DashboardAPIClientTests(unittest.TestCase):
    def test_prediction_sends_exact_supported_request_fields(self) -> None:
        payload = {"step": 1, "type": "PAYMENT", "amount": 10.0, "nameOrig": "C1",
                   "oldbalanceOrg": 20.0, "nameDest": "C2", "oldbalanceDest": 0.0}
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {"prediction": 0, "fraud_probability": 0.2, "risk_score": 20.0,
                                      "risk_level": "LOW", "model_version": "V2"}
        with patch.object(api_client.requests, "request", return_value=response) as request:
            result = api_client.predict_transaction(payload, base_url="http://example.test")
        self.assertEqual(result["model_version"], "V2")
        self.assertEqual(request.call_args.kwargs["json"], payload)

    def test_target_and_audit_features_are_rejected_before_network_call(self) -> None:
        payload = {"step": 1, "type": "PAYMENT", "amount": 10.0, "nameOrig": "C1",
                   "oldbalanceOrg": 20.0, "nameDest": "C2", "oldbalanceDest": 0.0, "isFraud": 0}
        with patch.object(api_client.requests, "request") as request:
            with self.assertRaises(api_client.APIClientError):
                api_client.predict_transaction(payload)
        request.assert_not_called()

    def test_preview_requires_ordered_34_feature_names(self) -> None:
        payload = {"step": 1, "type": "PAYMENT", "amount": 10.0, "nameOrig": "C1",
                   "oldbalanceOrg": 20.0, "nameDest": "C2", "oldbalanceDest": 0.0}
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {"feature_names": ["amount"], "features": {"amount": 10}}
        with patch.object(api_client.requests, "request", return_value=response):
            with self.assertRaises(api_client.APIClientError):
                api_client.preview_features(payload)


if __name__ == "__main__":
    unittest.main()
