from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.retraining.active_model import resolve_active_model
from src.retraining.stage4_workflow import validate_feature_contract
from src.retraining.stage5_workflow import promote, validate_candidate
from src.retraining.stage6_demo import preflight, run_demo


ROOT = Path(__file__).resolve().parents[1]
FEATURES = validate_feature_contract()


def make_dataset(path: Path) -> None:
    rows = []
    for split, start, count in (("train", 100, 120), ("validation", 430, 60), ("evaluation", 580, 60)):
        for index in range(count):
            row = {"event_id": f"{split}-{index:04d}", "step": start + index // 10,
                   "isFraud": index % 2}
            row.update({name: float((index + column) % 17) for column, name in enumerate(FEATURES)})
            rows.append(row)
    pd.DataFrame(rows, columns=["event_id", "step", *FEATURES, "isFraud"]).to_parquet(path, index=False)


class Stage6DemoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="stage6-demo-test-")
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "fixture.parquet"
        make_dataset(self.dataset)
        self.output = self.root / "candidate-output"

    def tearDown(self):
        self.temporary.cleanup()

    def test_preflight_is_read_only_and_validates_contract(self):
        before = resolve_active_model()
        result = preflight(dataset_path=self.dataset, output_root=self.output,
                           sample_limits={"train": 80, "validation": 30, "evaluation": 30})
        self.assertEqual(result["status"], "preflight_passed")
        self.assertFalse(result["training_started"])
        self.assertFalse(self.output.exists())
        after = resolve_active_model()
        self.assertEqual((before.model_version, before.threshold, dict(before.artifact_sha256)),
                         (after.model_version, after.threshold, dict(after.artifact_sha256)))

    def test_bad_dataset_and_wrong_feature_order_rejected(self):
        with self.assertRaises(FileNotFoundError):
            preflight(dataset_path=self.root / "missing.parquet", output_root=self.output)
        bad = self.root / "bad.parquet"
        frame = pd.read_parquet(self.dataset)
        frame.loc[0, "isFraud"] = 2
        frame.to_parquet(bad, index=False)
        with self.assertRaisesRegex(ValueError, "binary"):
            preflight(dataset_path=bad, output_root=self.output)
        reordered = self.root / "reordered.parquet"
        frame = pd.read_parquet(self.dataset)
        columns = frame.columns.tolist()
        columns[2], columns[3] = columns[3], columns[2]
        frame[columns].to_parquet(reordered, index=False)
        with self.assertRaisesRegex(ValueError, "columns/order"):
            preflight(dataset_path=reordered, output_root=self.output)
        leaked = self.root / "leaked.parquet"
        frame = pd.read_parquet(self.dataset)
        frame["target_copy"] = frame["isFraud"]
        frame.to_parquet(leaked, index=False)
        with self.assertRaisesRegex(ValueError, "columns/order"):
            preflight(dataset_path=leaked, output_root=self.output)

    def test_actual_training_is_unique_and_stage5_refuses_demo_candidate(self):
        params = {"n_estimators": 6, "max_depth": 2, "learning_rate": 0.1,
                  "subsample": 1.0, "colsample_bytree": 1.0, "min_child_weight": 1,
                  "reg_lambda": 1.0, "gamma": 0.0, "objective": "binary:logistic",
                  "eval_metric": "aucpr", "tree_method": "hist", "n_jobs": 1,
                  "early_stopping_rounds": 2}
        limits = {"train": 80, "validation": 30, "evaluation": 30}
        first = run_demo(dataset_path=self.dataset, output_root=self.output, sample_limits=limits,
                         seed=1234, training_params=params)
        second = run_demo(dataset_path=self.dataset, output_root=self.output, sample_limits=limits,
                          seed=1234, training_params=params)
        first_dir, second_dir = Path(first["candidate_dir"]), Path(second["candidate_dir"])
        self.assertNotEqual(first_dir, second_dir)
        self.assertTrue((first_dir / "model.json").is_file())
        self.assertTrue((first_dir / "training_report.json").is_file())
        metadata = json.loads((first_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["training_mode"], "demonstration")
        self.assertIs(metadata["eligible_for_genuine_promotion"], False)
        self.assertEqual(first["status"], "candidate_trained_not_promoted")
        self.assertIn("f1", first["candidate_evaluation_metrics"])
        self.assertEqual(len(first["artifact_sha256"]), 3)
        self.assertEqual(validate_candidate(first_dir)["status"], "validated")
        blocked = promote(candidate_dir=first_dir, evaluation_report=self.root / "missing-report.json",
                          approval_file=self.root / "missing-approval.json", dry_run=False)
        self.assertEqual(blocked["status"], "promotion_blocked")
        self.assertIn("ineligible", blocked["reason"])
        active = resolve_active_model()
        self.assertEqual((active.model_version, active.threshold), ("V2", 0.53))


if __name__ == "__main__":
    unittest.main(verbosity=2)
