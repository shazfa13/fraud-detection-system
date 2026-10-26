"""Candidate comparison tests with temporary bundles and dummy probabilities."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.retraining.candidate_evaluator import AcceptancePolicy, evaluate_candidate


ROOT = Path(__file__).resolve().parents[1]
FEATURES = json.loads((ROOT / "models" / "model_v1_metadata.json").read_text(encoding="utf-8"))["feature_names"]


def bundle(directory: Path, version: str, probabilities: list[float], *, features=None, threshold=.5) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payloads = {
        "model.json": {"probabilities": probabilities},
        "metadata.json": {
            "model_name": "dummy", "model_version": version, "model_type": "dummy",
            "feature_count": 34, "feature_names": FEATURES if features is None else features,
            "target": "isFraud", "selected_classification_threshold": threshold,
        },
        "threshold.json": {"selected_threshold": threshold},
    }
    for name, value in payloads.items():
        (directory / name).write_text(json.dumps(value), encoding="utf-8")


def hashes(directory: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.iterdir() if p.is_file()}


class DummyModel:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.seen: pd.DataFrame | None = None

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        self.seen = features.copy()
        return np.column_stack((1 - self.probabilities, self.probabilities))


class CandidateEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.active, self.candidate, self.backup = (
            self.root / "active", self.root / "candidate", self.root / "backup"
        )
        bundle(self.active, "V2", [.2, .7, .6, .9], threshold=.5)
        bundle(self.candidate, "CANDIDATE", [.1, .9, .2, .8], threshold=.5)
        self.backup.mkdir()
        (self.backup / "sentinel.txt").write_text("keep", encoding="utf-8")
        self.data = self._data()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @staticmethod
    def _data() -> pd.DataFrame:
        rows = []
        for index, label in enumerate([0, 1, 0, 1]):
            row = {"event_id": f"evt-{index}", "step": 700 + index, "isFraud": label}
            row.update({name: float(index + column) for column, name in enumerate(FEATURES)})
            row["fraud_probability"] = .9  # audit-only; must never be an input feature
            rows.append(row)
        return pd.DataFrame(rows)

    def _evaluate(self, report_path: Path, *, policy=None, loader=None, data=None):
        if loader is None:
            loader = lambda path: DummyModel(json.loads(path.read_text(encoding="utf-8"))["probabilities"])
        return evaluate_candidate(
            self.candidate, self.active, self.data if data is None else data,
            report_path, policy=policy, model_loader=loader,
        )

    def test_both_models_receive_identical_ordered_records_and_labels(self):
        loaded = {}
        def loader(path):
            model = DummyModel(json.loads(path.read_text(encoding="utf-8"))["probabilities"])
            loaded[path.parent.name] = model
            return model
        with tempfile.TemporaryDirectory() as reports:
            report = self._evaluate(Path(reports) / "evaluation.json", loader=loader)
        self.assertTrue(report["evaluation_completed"])
        self.assertEqual(report["data"]["rows"], 4)
        self.assertEqual(report["models"]["active"]["model_version"], "V2")
        self.assertEqual(loaded["active"].seen.columns.tolist(), FEATURES)
        self.assertEqual(loaded["candidate"].seen.columns.tolist(), FEATURES)
        pd.testing.assert_frame_equal(loaded["active"].seen, loaded["candidate"].seen)
        self.assertNotIn("event_id", loaded["active"].seen.columns)
        self.assertNotIn("step", loaded["active"].seen.columns)
        self.assertNotIn("isFraud", loaded["active"].seen.columns)
        self.assertNotIn("fraud_probability", loaded["active"].seen.columns)

    def test_metrics_and_confusion_counts_use_saved_thresholds(self):
        with tempfile.TemporaryDirectory() as reports:
            result = self._evaluate(Path(reports) / "metrics.json")
        active = result["metrics"]["active"]
        candidate = result["metrics"]["candidate"]
        self.assertEqual((active["tn"], active["fp"], active["fn"], active["tp"]), (1, 1, 0, 2))
        self.assertAlmostEqual(active["precision"], 2 / 3)
        self.assertEqual((candidate["tn"], candidate["fp"], candidate["fn"], candidate["tp"]), (2, 0, 0, 2))
        self.assertEqual(candidate["f1"], 1.0)
        self.assertEqual(candidate["pr_auc"], 1.0)

    def test_incompatible_feature_order_fails_and_is_reported(self):
        reordered = FEATURES.copy(); reordered[0], reordered[1] = reordered[1], reordered[0]
        bundle(self.candidate, "CANDIDATE", [.1, .9, .2, .8], features=reordered)
        with tempfile.TemporaryDirectory() as reports:
            path = Path(reports) / "failed.json"
            result = self._evaluate(path)
            self.assertEqual(result["decision_status"], "EVALUATION_FAILED")
            self.assertIn("feature names/order", result["error"])
            self.assertEqual(json.loads(path.read_text())["evaluation_status"], "FAILED")

    def test_threshold_metadata_mismatch_fails_before_prediction(self):
        metadata_path = self.candidate / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["selected_classification_threshold"] = .25
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        called = []
        with tempfile.TemporaryDirectory() as reports:
            result = self._evaluate(Path(reports) / "threshold-mismatch.json",
                                    loader=lambda path: called.append(path))
        self.assertEqual(result["decision_status"], "EVALUATION_FAILED")
        self.assertEqual(called, [])

    def test_missing_labels_and_invalid_probabilities_fail_safely(self):
        bad_data = self.data.copy(); bad_data.loc[0, "isFraud"] = np.nan
        with tempfile.TemporaryDirectory() as reports:
            result = self._evaluate(Path(reports) / "labels.json", data=bad_data)
            self.assertEqual(result["decision_status"], "EVALUATION_FAILED")
            self.assertIn("labels", result["error"])
            result = self._evaluate(
                Path(reports) / "probabilities.json",
                loader=lambda path: DummyModel([.1, 1.2, .2, .8]),
            )
            self.assertEqual(result["decision_status"], "EVALUATION_FAILED")
            self.assertIn("[0, 1]", result["error"])

    def test_malformed_model_artifact_fails_and_report_is_written(self):
        (self.candidate / "model.json").write_text("{broken", encoding="utf-8")
        with tempfile.TemporaryDirectory() as reports:
            path = Path(reports) / "malformed.json"
            result = self._evaluate(path)
            self.assertTrue(path.is_file())
        self.assertEqual(result["evaluation_status"], "FAILED")

    def test_no_policy_returns_pending_policy_not_approval(self):
        with tempfile.TemporaryDirectory() as reports:
            result = self._evaluate(Path(reports) / "pending.json")
        self.assertEqual(result["evaluation_status"], "COMPLETED")
        self.assertEqual(result["decision_status"], "PENDING_POLICY")
        self.assertIsNone(result["policy_results"])

    def test_explicit_policy_can_accept_or_reject(self):
        with tempfile.TemporaryDirectory() as reports:
            accepted = self._evaluate(
                Path(reports) / "accepted.json",
                policy=AcceptancePolicy({"f1": .1, "recall": 0.0, "pr_auc": 0.0}),
            )
            rejected = self._evaluate(
                Path(reports) / "rejected.json",
                policy=AcceptancePolicy({"f1": .5}),
            )
        self.assertEqual(accepted["decision_status"], "ACCEPTED")
        self.assertEqual(rejected["decision_status"], "REJECTED")

    def test_report_written_and_all_model_and_backup_files_unchanged(self):
        before = {"active": hashes(self.active), "candidate": hashes(self.candidate),
                  "backup": hashes(self.backup)}
        with tempfile.TemporaryDirectory() as reports:
            path = Path(reports) / "nested" / "evaluation.json"
            with patch("src.retraining.model_activator.promote_candidate") as promote, \
                    patch("src.retraining.model_activator.rollback") as rollback:
                returned = self._evaluate(path)
                promote.assert_not_called()
                rollback.assert_not_called()
            written = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(written, returned)
        self.assertEqual(written["data"]["feature_count"], 34)
        self.assertEqual(before["active"], hashes(self.active))
        self.assertEqual(before["candidate"], hashes(self.candidate))
        self.assertEqual(before["backup"], hashes(self.backup))

    def test_report_path_inside_model_directory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be written inside"):
            evaluate_candidate(self.candidate, self.active, self.data,
                               self.active / "evaluation.json", model_loader=lambda _: None)

    def test_nested_candidate_and_active_directories_are_rejected(self):
        nested = self.active / "nested-candidate"
        nested.mkdir()
        with tempfile.TemporaryDirectory() as reports:
            with self.assertRaisesRegex(ValueError, "non-nested"):
                evaluate_candidate(nested, self.active, self.data,
                                   Path(reports) / "report.json", model_loader=lambda _: None)


if __name__ == "__main__":
    unittest.main(verbosity=2)

