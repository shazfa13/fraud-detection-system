from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.retraining.active_model import ActiveModelConfigurationError, resolve_active_model
from src.retraining.candidate_evaluator import _data_fingerprint
from src.retraining.stage5_workflow import evaluate_stage5, promote, rollback, validate_candidate
from src.api import app as serving_api
from src.monitoring.monitor import get_active_model


ROOT = Path(__file__).resolve().parents[1]
FEATURES = json.loads((ROOT / "models" / "model_v1_metadata.json").read_text(encoding="utf-8"))["feature_names"]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Stage5WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="stage5-fixture-")
        self.root = Path(self.temp.name)
        (self.root / "models").mkdir()
        shutil.copyfile(ROOT / "models" / "model_v1_metadata.json", self.root / "models" / "model_v1_metadata.json")
        for version in ("v1", "v2"):
            source = ROOT / "models"
            (self.root / "models" / f"xgboost_model_{version}.json").write_text('{"fixture":true}')
            metadata = json.loads((source / f"model_{version}_metadata.json").read_text(encoding="utf-8"))
            metadata["feature_names"] = FEATURES
            metadata["feature_count"] = 34
            metadata["selected_classification_threshold"] = 0.53 if version == "v2" else 0.5
            (self.root / "models" / f"model_{version}_metadata.json").write_text(json.dumps(metadata))
            threshold = json.loads((source / f"model_{version}_threshold.json").read_text(encoding="utf-8"))
            threshold["selected_threshold"] = 0.53 if version == "v2" else 0.5
            (self.root / "models" / f"model_{version}_threshold.json").write_text(json.dumps(threshold))
        self.candidate = self.root / "models" / "candidates" / "fixture-candidate"
        self.candidate.mkdir(parents=True)
        shutil.copyfile(ROOT / "models" / "xgboost_model_v2.json", self.candidate / "model.json")
        candidate_meta = json.loads((self.root / "models" / "model_v2_metadata.json").read_text())
        candidate_meta.update({"model_version": "fixture-candidate", "target": "isFraud",
            "dataset_fingerprint": "train-fingerprint", "label_provenance_summary": {"source_type": "trusted_verified"},
            "temporal_split": {"train_end_step": 600, "validation_end_step": 650,
                               "test_start_step": 651, "test_end_step": 700},
            "training_configuration": {"max_depth": 2},
            "training_seconds": 1.0, "test_metrics": {"f1": 0.8}})
        candidate_meta.pop("selected_classification_threshold", None)
        (self.candidate / "metadata.json").write_text(json.dumps(candidate_meta))
        (self.candidate / "threshold.json").write_text('{"selected_threshold":0.5}')

    def tearDown(self):
        self.temp.cleanup()

    def test_v2_default_and_manifest_resolve_shared_bundle(self):
        active = resolve_active_model(project_root=self.root, environ={})
        self.assertEqual((active.model_version, active.threshold), ("V2", 0.53))
        manifest = self.root / "models" / "active_model_manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "model_version": "V1",
            "artifacts": {"model": "models/xgboost_model_v1.json", "metadata": "models/model_v1_metadata.json",
                          "threshold": "models/model_v1_threshold.json"},
            "artifact_sha256": {"model.json": _hash(self.root / "models" / "xgboost_model_v1.json"),
                                "metadata.json": _hash(self.root / "models" / "model_v1_metadata.json"),
                                "threshold.json": _hash(self.root / "models" / "model_v1_threshold.json")}}))
        self.assertEqual(resolve_active_model(project_root=self.root).model_version, "V1")

    def test_serving_and_monitor_use_shared_active_resolver(self):
        resolved = resolve_active_model()
        self.assertEqual(serving_api.MODEL_VERSION, resolved.model_version)
        self.assertEqual(serving_api.CLASSIFICATION_THRESHOLD, resolved.threshold)
        self.assertEqual(get_active_model(), resolved.model_version)
        self.assertEqual(serving_api.health()["model_version"], resolved.model_version)
        self.assertEqual(serving_api.health()["threshold"], resolved.threshold)
        with self.assertRaises(Exception):
            serving_api.TransactionRequest.model_validate({"isFraud": 1})

    def test_bad_manifest_fails_closed(self):
        (self.root / "models" / "active_model_manifest.json").write_text("{")
        with self.assertRaises(ActiveModelConfigurationError):
            resolve_active_model(project_root=self.root)

    def test_manifest_artifact_hash_mismatch_fails_closed(self):
        manifest = self.root / "models" / "active_model_manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "model_version": "V2",
            "artifacts": {"model": "models/xgboost_model_v2.json", "metadata": "models/model_v2_metadata.json",
                          "threshold": "models/model_v2_threshold.json"}, "artifact_sha256": {}}))
        with self.assertRaises(ActiveModelConfigurationError):
            resolve_active_model(project_root=self.root)

    def test_candidate_provenance_and_hash_validation(self):
        checked = validate_candidate(self.candidate, project_root=self.root, model_loader=lambda path: object())
        self.assertEqual(checked["status"], "validated")
        self.assertEqual(len(checked["feature_names"]), 34)
        (self.candidate / "threshold.json").write_text('{"selected_threshold":1.0}')
        with self.assertRaises(ValueError):
            validate_candidate(self.candidate, project_root=self.root, model_loader=lambda path: object())

    def test_candidate_feature_order_and_recorded_hashes_are_enforced(self):
        metadata_path = self.candidate / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["feature_names"][0], metadata["feature_names"][1] = metadata["feature_names"][1], metadata["feature_names"][0]
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "feature names/order"):
            validate_candidate(self.candidate, project_root=self.root, model_loader=lambda path: object())
        metadata["feature_names"] = FEATURES
        metadata["artifact_sha256"] = {"model.json": "wrong"}
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "hashes"):
            validate_candidate(self.candidate, project_root=self.root, model_loader=lambda path: object())

    def test_missing_candidate_reported(self):
        self.assertEqual(validate_candidate(self.root / "missing", project_root=self.root)["status"], "candidate_unavailable")

    def _evaluation_frame(self):
        rows = []
        for index in range(100):
            row = {"event_id": f"event-{index}", "transaction_id": f"transaction-{index}",
                   "step": 701 + index, "isFraud": int(index >= 90)}
            row.update({feature: float(index % 7) for feature in FEATURES})
            rows.append(row)
        return pd.DataFrame(rows)

    def test_synthetic_evaluation_is_blocked_and_reported(self):
        frame = self._evaluation_frame()
        path = self.root / "blocked-evaluation.json"
        result = evaluate_stage5(self.candidate, frame, path,
            evaluation_provenance={"source": "fixture", "label_provenance": "synthetic",
                "candidate_training_fingerprint": "train-fingerprint", "evaluation_fingerprint": "fixture",
                "synthetic": True}, trusted_evaluation=True,
            model_loader=lambda model_path: object(), project_root=self.root)
        self.assertEqual(result["status"], "evaluation_blocked")
        self.assertTrue(path.is_file())

    def test_stage5_metrics_reject_regression_and_include_operating_counts(self):
        from src.retraining.candidate_evaluator import _data_fingerprint

        frame = self._evaluation_frame()
        candidate_probabilities = np.zeros(len(frame))
        active_probabilities = np.asarray([.1 if label == 0 else .9 for label in frame.isFraud])
        provenance = {"source": "authorized-heldout", "label_provenance": "trusted_verified",
            "candidate_training_fingerprint": "train-fingerprint", "synthetic": False}
        provenance["evaluation_fingerprint"] = _data_fingerprint(frame, FEATURES)
        output = self.root / "regressed-evaluation.json"

        class Model:
            def __init__(self, probs): self.probs = probs
            def predict_proba(self, X): return np.column_stack((1-self.probs, self.probs))

        def loader(path):
            return Model(candidate_probabilities if path.parent.name == self.candidate.name else active_probabilities)

        result = evaluate_stage5(self.candidate, frame, output, evaluation_provenance=provenance,
            trusted_evaluation=True, model_loader=loader, project_root=self.root)
        self.assertEqual(result["status"], "candidate_rejected")
        self.assertIn("fp", result["evaluation"]["metrics"]["active"])
        self.assertIn("fn", result["evaluation"]["metrics"]["candidate"])

    def test_authorization_binding_dry_run_promotion_and_rollback(self):
        report_path = self.root / "evaluation.json"
        checked = validate_candidate(self.candidate, project_root=self.root, model_loader=lambda path: object())
        report_path.write_text(json.dumps({"status": "candidate_approved_pending_authorization", "candidate": checked}))
        approval = {"candidate_version": checked["model_version"], "candidate_artifact_sha256": checked["artifact_sha256"],
            "evaluation_report_sha256": _hash(report_path), "dataset_fingerprint": checked["dataset_fingerprint"],
            "approved_by": "fixture reviewer", "approved_at": "2026-10-25T12:00:00+00:00",
            "approval_statement": "I approve promotion of this exact candidate and evaluation report."}
        approval_path = self.root / "approval.json"
        approval_path.write_text(json.dumps(approval))
        prior_bytes = (self.root / "models" / "model_v2_metadata.json").read_bytes()
        dry = promote(candidate_dir=self.candidate, evaluation_report=report_path, approval_file=approval_path,
                      project_root=self.root, dry_run=True)
        self.assertEqual(dry["status"], "promotion_ready")
        self.assertFalse((self.root / "models" / "active_model_manifest.json").exists())
        self.assertEqual((self.root / "models" / "model_v2_metadata.json").read_bytes(), prior_bytes)
        self.assertEqual(promote(candidate_dir=self.candidate, evaluation_report=report_path,
            approval_file=approval_path, project_root=self.root, dry_run=False)["status"], "promoted")
        self.assertEqual(resolve_active_model(project_root=self.root).model_version, "fixture-candidate")
        repeated = promote(candidate_dir=self.candidate, evaluation_report=report_path,
            approval_file=approval_path, project_root=self.root, dry_run=False)
        self.assertEqual(repeated["status"], "promotion_blocked")
        self.assertEqual(rollback(project_root=self.root, dry_run=True)["status"], "rollback_ready")
        self.assertEqual(rollback(project_root=self.root, dry_run=False)["status"], "rolled_back")
        self.assertEqual(resolve_active_model(project_root=self.root).model_version, "V2")
        self.assertEqual(rollback(project_root=self.root, dry_run=False)["status"], "rollback_blocked")

    def test_failed_manifest_activation_restores_prior_bundle(self):
        checked = validate_candidate(self.candidate, project_root=self.root, model_loader=lambda path: object())
        report_path = self.root / "activation-evaluation.json"
        report_path.write_text(json.dumps({"status": "candidate_approved_pending_authorization", "candidate": checked}))
        approval_path = self.root / "activation-approval.json"
        approval_path.write_text(json.dumps({"candidate_version": checked["model_version"],
            "candidate_artifact_sha256": checked["artifact_sha256"],
            "evaluation_report_sha256": _hash(report_path), "dataset_fingerprint": checked["dataset_fingerprint"],
            "approved_by": "fixture reviewer", "approved_at": "2026-10-25T12:00:00+00:00",
            "approval_statement": "I approve promotion of this exact candidate and evaluation report."}))
        atomic_json = __import__("src.retraining.stage5_workflow", fromlist=["_atomic_json"])._atomic_json
        failed_once = False

        def corrupt_active_manifest(path, value):
            nonlocal failed_once
            atomic_json(path, value)
            if path.name == "active_model_manifest.json" and not failed_once:
                failed_once = True
                path.write_text("{invalid", encoding="utf-8")

        with patch("src.retraining.stage5_workflow._atomic_json", side_effect=corrupt_active_manifest):
            result = promote(candidate_dir=self.candidate, evaluation_report=report_path,
                approval_file=approval_path, project_root=self.root, dry_run=False)
        self.assertEqual(result["status"], "activation_failed")
        self.assertEqual(resolve_active_model(project_root=self.root).model_version, "V2")

    def test_wrong_approval_is_rejected_without_activation(self):
        report = self.root / "evaluation.json"
        report.write_text('{"status":"candidate_approved_pending_authorization"}')
        approval = self.root / "approval.json"
        approval.write_text('{"candidate_version":"wrong"}')
        result = promote(candidate_dir=self.candidate, evaluation_report=report, approval_file=approval,
                         project_root=self.root, dry_run=False)
        self.assertEqual(result["status"], "promotion_blocked")
        self.assertFalse((self.root / "models" / "active_model_manifest.json").exists())

    def test_stale_evaluation_report_binding_is_rejected(self):
        checked = validate_candidate(self.candidate, project_root=self.root, model_loader=lambda path: object())
        report = self.root / "stale-report.json"
        report.write_text(json.dumps({"status": "candidate_approved_pending_authorization", "candidate": checked}))
        approval = self.root / "stale-approval.json"
        approval.write_text(json.dumps({"candidate_version": checked["model_version"],
            "candidate_artifact_sha256": checked["artifact_sha256"], "evaluation_report_sha256": _hash(report),
            "dataset_fingerprint": checked["dataset_fingerprint"], "approved_by": "reviewer",
            "approved_at": "2026-10-25T12:00:00+00:00",
            "approval_statement": "I approve promotion of this exact candidate and evaluation report."}))
        report.write_text(json.dumps({"status": "candidate_approved_pending_authorization", "candidate": checked,
                                      "changed_after_approval": True}))
        result = promote(candidate_dir=self.candidate, evaluation_report=report, approval_file=approval,
                         project_root=self.root, dry_run=False)
        self.assertEqual(result["status"], "promotion_blocked")
        self.assertFalse((self.root / "models" / "active_model_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
