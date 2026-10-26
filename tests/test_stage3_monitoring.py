"""Deterministic Stage 3 monitor tests using only temporary JSONL/parquet/SQLite files."""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.monitoring import config, monitor


class Stage3MonitoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.metadata = json.loads(config.MODEL_METADATA_PATHS["V2"].read_text(encoding="utf-8"))
        self.features = self.metadata["feature_names"]

    def tearDown(self):
        self.temp.cleanup()

    def _reference(self, rows=800):
        rng = np.random.default_rng(4001)
        frame = pd.DataFrame(rng.normal(0, 1, size=(rows, len(self.features))), columns=self.features)
        frame["step"] = np.arange(1, rows + 1)
        frame["event_id"] = np.arange(rows)
        frame["isFraud"] = 0
        path = self.root / "reference.parquet"
        frame.to_parquet(path, index=False)
        return path, frame

    def _stream(self, frame, *, data_origin="live"):
        path = self.root / "predictions.jsonl"
        with path.open("w", encoding="utf-8") as out:
            for i, row in frame.iterrows():
                out.write(json.dumps({"transaction_id": f"TX-{i}", "model_version": "V2",
                    "processing_timestamp_utc": f"2026-10-25T00:{i // 60:02d}:{i % 60:02d}+00:00",
                    "data_origin": data_origin,
                    "feature_values": {name: float(row[name]) for name in self.features}}) + "\n")
        return path

    def _empty_stage2(self):
        path = self.root / "stage2.sqlite3"
        with closing(sqlite3.connect(path)) as db:
            db.executescript("""
                CREATE TABLE predictions(transaction_id TEXT, prediction INTEGER, model_version TEXT,
                    prediction_timestamp TEXT, data_origin TEXT, label_status TEXT);
                CREATE TABLE labels(transaction_id TEXT, outcome INTEGER, label_source TEXT, source_type TEXT,
                    confirmed_at TEXT, status TEXT, submission_id TEXT);
                CREATE TABLE label_submissions(transaction_id TEXT, disposition TEXT);
                CREATE TABLE submission_replays(disposition TEXT);
                CREATE TABLE prediction_conflicts(transaction_id TEXT);
            """)
            db.commit()
        return path

    def test_reference_draw_no_drift_and_insufficient_sample(self):
        reference_path, reference = self._reference()
        with patch.dict(os.environ, {"FRAUD_ACTIVE_MODEL_VERSION": "v2"}):
            sampled, details = monitor._reference_sample("V2", path=reference_path, sample_size=400)
        self.assertGreaterEqual(len(sampled), 400)
        self.assertEqual(details["reference_sample_target"], 400)
        self.assertEqual(details["reference_version"], "V2-train-step-1-422")
        feature_sample = reference.loc[:99, self.features]
        results = monitor.detect_feature_drift(sampled, feature_sample, self.features)
        overall = monitor.detect_overall_drift(results)
        self.assertFalse(overall["overall_drift_detected"])
        self.assertEqual(overall["features_tested"], 34)
        stream_path = self._stream(feature_sample.iloc[:10])
        report = monitor.run_monitor(stage1_path=stream_path, reference_path=reference_path,
            stage2_path=self._empty_stage2(), monitor_db_path=self.root / "monitor.sqlite3",
            model_version="V2", min_current=30)
        self.assertEqual(report["drift_status"], "insufficient_data")
        self.assertIn("Need 30", report["drift"]["reason"])

    def test_reference_reservoir_keeps_rows_with_per_feature_missing_values(self):
        rng = np.random.default_rng(513)
        frame = pd.DataFrame(rng.normal(size=(100, len(self.features))), columns=self.features)
        for i in range(len(frame)):
            frame.loc[i, self.features[i % len(self.features)]] = np.nan
        frame["step"] = np.arange(1, 101)
        path = self.root / "partially_missing.parquet"
        frame.to_parquet(path, index=False)
        sampled, details = monitor._reference_sample("V2", path=path, sample_size=100)
        self.assertEqual(details["reference_rows_scanned"], 100)
        self.assertEqual(details["reference_sample_count"], 100)
        self.assertEqual(len(sampled), 100)
        self.assertTrue(sampled[self.features].isna().any().any())
        self.assertTrue((sampled[self.features].count() >= 97).all())

    def test_reference_sampling_adds_sparse_feature_coverage_deterministically(self):
        rng = np.random.default_rng(77)
        frame = pd.DataFrame(rng.normal(size=(400, len(self.features))), columns=self.features)
        sparse = ["sender_prev_mean_amount", "sender_prev_max_amount",
                  "amount_vs_sender_hist_mean", "sender_hours_since_prev_txn"]
        for name in sparse:
            frame.loc[:, name] = np.nan
            frame.loc[rng.choice(len(frame), size=80, replace=False), name] = rng.normal(size=80)
        frame["step"] = np.arange(1, len(frame) + 1)
        path = self.root / "sparse_reference.parquet"
        frame.to_parquet(path, index=False)
        first, details = monitor._reference_sample("V2", path=path, sample_size=100, seed=3025)
        second, _ = monitor._reference_sample("V2", path=path, sample_size=100, seed=3025)
        pd.testing.assert_frame_equal(first, second)
        self.assertGreaterEqual(min(first[name].count() for name in sparse), 30)
        self.assertEqual(details["reference_sampling"],
                         "deterministic uniform row reservoir plus per-feature finite-row coverage reservoirs")
        for name in sparse:
            self.assertEqual(details["reference_per_feature_counts"][name]["source_finite_count"], 80)
            self.assertGreaterEqual(details["reference_per_feature_counts"][name]["finite_count"], 30)

    def test_insufficient_source_coverage_is_explicit(self):
        frame = pd.DataFrame(np.nan, index=range(100), columns=self.features)
        frame[self.features[0]] = np.arange(100, dtype=float)
        frame.loc[0, self.features[1]] = 4.0
        frame["step"] = np.arange(1, 101)
        path = self.root / "insufficient_reference.parquet"
        frame.to_parquet(path, index=False)
        sampled, details = monitor._reference_sample("V2", path=path, sample_size=50)
        result = monitor.detect_feature_drift(sampled, sampled, self.features)
        sparse_row = result.loc[result.feature_name == self.features[1]].iloc[0]
        self.assertEqual(details["reference_per_feature_counts"][self.features[1]]["source_finite_count"], 1)
        self.assertTrue(sparse_row.drift_reason.startswith("insufficient_sample"))

    def test_malformed_feature_rows_are_skipped_and_drift_alone_is_not_performance_alert(self):
        reference_path, reference = self._reference()
        current = reference.loc[:49, self.features].copy()
        for name in self.features[:8]:
            current[name] += 10.0
        stream_path = self._stream(current)
        with stream_path.open("a", encoding="utf-8") as out:
            out.write(json.dumps({"transaction_id": "BAD", "model_version": "V2",
                "processing_timestamp_utc": "2026-10-25T02:00:00+00:00",
                "feature_values": {self.features[0]: "not-a-number"}}) + "\n")
        report = monitor.run_monitor(stage1_path=stream_path, reference_path=reference_path,
            stage2_path=self._empty_stage2(), monitor_db_path=self.root / "monitor.sqlite3",
            model_version="V2", min_current=30)
        self.assertEqual(report["drift_status"], "drift")
        self.assertEqual(report["performance"]["status"], "insufficient_data")
        self.assertEqual([item["type"] for item in report["alerts"]], ["feature_drift"])

    def test_synthetic_only_drift_does_not_create_retraining_request(self):
        reference_path, reference = self._reference()
        current = reference.loc[:99, self.features].copy()
        for name in self.features[:8]:
            current[name] += 10.0
        report_path = self.root / "monitor.sqlite3"
        report = monitor.run_monitor(stage1_path=self._stream(current, data_origin="synthetic_paysim_demo"),
            reference_path=reference_path, stage2_path=self._empty_stage2(),
            monitor_db_path=report_path, model_version="V2", min_current=30)
        self.assertEqual(report["monitoring_basis"], "synthetic_only")
        self.assertEqual(report["drift_status"], "drift")
        self.assertFalse(report["retraining_request"]["requested"])
        with closing(sqlite3.connect(report_path)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM monitor_alerts").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM retraining_requests").fetchone()[0], 0)

    def test_reference_read_error_is_reported_without_crashing_monitor(self):
        _, reference = self._reference()
        stream_path = self._stream(reference.loc[:39, self.features])
        report = monitor.run_monitor(stage1_path=stream_path, reference_path=self.root / "missing.parquet",
            stage2_path=self._empty_stage2(), monitor_db_path=self.root / "monitor.sqlite3",
            model_version="V2", min_current=30)
        self.assertEqual(report["drift_status"], "insufficient_data")
        self.assertTrue(report["monitoring_errors"])

    def test_known_shift_alerts_and_repeated_window_deduplicates(self):
        reference_path, reference = self._reference()
        current = reference.loc[:99, self.features].copy()
        for name in self.features[:8]:
            current[name] += 10.0
        stream_path = self._stream(current)
        report_path = self.root / "monitor.sqlite3"
        stage2 = self._empty_stage2()
        reports = []
        for _ in range(2):
            reports.append(monitor.run_monitor(stage1_path=stream_path, reference_path=reference_path,
                stage2_path=stage2, monitor_db_path=report_path, model_version="V2", min_current=30))
        self.assertEqual(reports[0]["drift_status"], "drift")
        self.assertTrue(any(a["type"] == "feature_drift" for a in reports[0]["alerts"]))
        evidence = reports[0]["drift"]["feature_evidence"]
        self.assertEqual(len(evidence), 34)
        self.assertTrue(any("PSI" in item["drift_reason"] or "after BH correction" in item["drift_reason"]
                            for item in evidence if item["drift_detected"]))
        self.assertTrue(reports[0]["alerts"][0]["newly_recorded"])
        self.assertFalse(reports[1]["alerts"][0]["newly_recorded"])
        with closing(sqlite3.connect(report_path)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM monitor_alerts").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM retraining_requests").fetchone()[0], 1)

    def _stage2_with_labels(self, *, synthetic=False, conflicts=False, positives=20):
        path = self._empty_stage2()
        with closing(sqlite3.connect(path)) as db:
            for i in range(102):
                outcome = int(i < positives)
                prediction = 1 - outcome  # deliberately degraded: every prediction is wrong
                source_type = "synthetic_test" if synthetic and i == 0 else "trusted_verified"
                label_source = "synthetic" if source_type == "synthetic_test" else "trusted_feed"
                db.execute("INSERT INTO predictions VALUES(?,?,?,?,?,?)",
                    (f"P{i}", prediction, "V2", f"2026-10-25T00:{i:02d}:00+00:00", "live", "resolved"))
                db.execute("INSERT INTO labels VALUES(?,?,?,?,?,?,?)",
                    (f"P{i}", outcome, label_source, source_type, f"2026-10-25T01:{i:02d}:00+00:00", "resolved", f"S{i}"))
                if conflicts and i == 1:
                    db.execute("INSERT INTO label_submissions VALUES(?,?)", (f"P{i}", "conflict"))
            db.commit()
        return path

    def test_performance_counts_excludes_synthetic_and_conflicting_rows(self):
        path = self._stage2_with_labels(synthetic=True, conflicts=True)
        env = {"FRAUD_TRUSTED_LABEL_SOURCES": "trusted_feed", "FRAUD_TRUSTED_LABEL_TOKEN": "x" * 40}
        with patch.dict(os.environ, env):
            result = monitor._performance("V2", database_path=path)
        self.assertEqual(result["eligible_labeled_transactions"], 100)
        self.assertEqual(result["synthetic_labels_excluded"], 1)
        self.assertEqual(result["confusion_matrix"], {"tp": 0, "fp": 82, "tn": 0, "fn": 18})
        self.assertEqual(result["status"], "degraded")
        self.assertEqual(result["threshold"], 0.53)

    def test_performance_confusion_matrix_and_metrics(self):
        path = self._empty_stage2()
        with closing(sqlite3.connect(path)) as db:
            for i in range(120):
                outcome = int(i < 20)
                prediction = int(i < 15 or 20 <= i < 25)
                db.execute("INSERT INTO predictions VALUES(?,?,?,?,?,?)",
                    (f"M{i}", prediction, "V2", f"2026-10-25T00:{i:02d}:00+00:00", "live", "resolved"))
                db.execute("INSERT INTO labels VALUES(?,?,?,?,?,?,?)",
                    (f"M{i}", outcome, "trusted_feed", "trusted_verified",
                     f"2026-10-25T01:{i:02d}:00+00:00", "resolved", f"MS{i}"))
            db.execute("INSERT INTO label_submissions VALUES(?,?)", ("M0", "duplicate_identical"))
            db.execute("INSERT INTO submission_replays VALUES(?)", ("duplicate_submission",))
            db.commit()
        env = {"FRAUD_TRUSTED_LABEL_SOURCES": "trusted_feed", "FRAUD_TRUSTED_LABEL_TOKEN": "x" * 40}
        with patch.dict(os.environ, env):
            result = monitor._performance("V2", database_path=path)
        self.assertEqual(result["eligible_labeled_transactions"], 120)
        self.assertEqual(result["duplicate_submission_records"], 2)
        self.assertEqual(result["confusion_matrix"], {"tp": 15, "fp": 5, "tn": 95, "fn": 5})
        self.assertEqual(result["metrics"]["precision"], 0.75)
        self.assertEqual(result["metrics"]["recall"], 0.75)
        self.assertEqual(result["metrics"]["f1"], 0.75)

    def test_zero_trusted_labels_is_insufficient_and_synthetic_never_trusted(self):
        path = self._stage2_with_labels(synthetic=True)
        env = {"FRAUD_TRUSTED_LABEL_SOURCES": "trusted_feed", "FRAUD_TRUSTED_LABEL_TOKEN": "x" * 40}
        with closing(sqlite3.connect(path)) as db:
            db.execute("DELETE FROM labels WHERE source_type='trusted_verified'")
            db.commit()
        with patch.dict(os.environ, env):
            result = monitor._performance("V2", database_path=path)
        self.assertEqual(result["status"], "insufficient_data")
        self.assertEqual(result["eligible_labeled_transactions"], 0)
        self.assertEqual(result["synthetic_labels_excluded"], 1)
        self.assertIsNone(result["metrics"])


if __name__ == "__main__":
    unittest.main()
