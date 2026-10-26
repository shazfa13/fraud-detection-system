"""Offline lifecycle coordinator tests; all artifact paths are temporary."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.retraining.lifecycle_coordinator import (
    CycleRequest,
    CycleStatus,
    LabelReadiness,
    LifecycleDependencies,
    run_cycle,
)
from src.retraining.candidate_evaluator import AcceptancePolicy, evaluate_candidate
from src.retraining.cycle_pipeline import (
    FINAL_FEATURES, CycleConfig, StepWindow, build_cycle_dataset, save_candidate_bundle,
    split_cycle_dataset,
)
from src.retraining.model_activator import promote_candidate, rollback, validate_bundle


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CoordinatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.active = self.root / "active"
        self.candidate = self.root / "candidate"
        self.backup = self.root / "backup"
        self.active.mkdir()
        self.candidate.mkdir()
        self.backup.mkdir()
        for folder, version in ((self.active, "V2"), (self.candidate, "V3-candidate")):
            for name in ("model.json", "metadata.json", "threshold.json"):
                (folder / name).write_text(f"{version}:{name}", encoding="utf-8")
        self.active_hashes = {p.name: digest(p) for p in self.active.iterdir()}
        self.report = self.root / "reports" / "cycle.json"
        self.labels = LabelReadiness(
            source_name="verified test fixture", source_verified=True,
            matched_label_count=100, unmatched_record_count=0,
        )
        self.calls: list[str] = []

    def tearDown(self) -> None:
        self.temp.cleanup()

    def request(self, **changes: Any) -> CycleRequest:
        values = dict(
            drift_detected=True, label_readiness=self.labels,
            minimum_genuine_labels=20, candidate_dir=self.candidate,
            active_dir=self.active, backup_dir=self.backup,
            report_path=self.report, acceptance_policy_configured=True,
        )
        values.update(changes)
        return CycleRequest(**values)

    def dependencies(self, *, decision: str = "ACCEPTED", complete: bool = True,
                     train_error: bool = False, promote_error: bool = False) -> LifecycleDependencies:
        def build(context: Mapping[str, Any]) -> str:
            self.calls.append("build")
            return "built"

        def train(built: Any, context: Mapping[str, Any]) -> str:
            self.calls.append("train")
            if train_error:
                raise RuntimeError("training stub failed")
            return "trained"

        def evaluate(built: Any, trained: Any, context: Mapping[str, Any]) -> Mapping[str, Any]:
            self.calls.append("evaluate")
            return {"decision_status": decision, "evaluation_completed": complete}

        def validate(directory: Path) -> dict[str, str]:
            self.calls.append("validate")
            return {p.name: digest(p) for p in directory.iterdir()}

        def promote(context: Mapping[str, Any]) -> str:
            self.calls.append("promote")
            if promote_error:
                raise RuntimeError("promotion stub failed")
            return "promoted in isolated stub"

        return LifecycleDependencies(build, train, evaluate, validate, promote)

    def assert_active_unchanged(self) -> None:
        self.assertEqual(self.active_hashes, {p.name: digest(p) for p in self.active.iterdir()})

    def test_no_drift_skips_every_stage(self) -> None:
        result = run_cycle(self.request(drift_detected=False), self.dependencies())
        self.assertEqual(result.status, CycleStatus.DRIFT_NOT_DETECTED)
        self.assertEqual(self.calls, [])
        self.assertTrue(self.report.is_file())
        self.assert_active_unchanged()

    def test_simulated_or_unmatched_labels_wait(self) -> None:
        for labels in (
            LabelReadiness("Phase 9 PaySim", False, 100, 0),
            LabelReadiness("unmatched source", True, 100, 1),
            None,
        ):
            with self.subTest(labels=labels):
                self.report.unlink(missing_ok=True)
                result = run_cycle(self.request(label_readiness=labels), self.dependencies())
                self.assertEqual(result.status, CycleStatus.WAITING_FOR_LABELS)
                self.assertEqual(self.calls, [])
        self.assert_active_unchanged()

    def test_insufficient_labels_do_not_build_candidate(self) -> None:
        labels = LabelReadiness("verified", True, 19, 0)
        result = run_cycle(self.request(label_readiness=labels), self.dependencies())
        self.assertEqual(result.status, CycleStatus.INSUFFICIENT_LABELS)
        self.assertEqual(self.calls, [])

    def test_candidate_build_failure_stops(self) -> None:
        deps = self.dependencies()
        def fail(context: Mapping[str, Any]) -> None:
            self.calls.append("build")
            raise RuntimeError("build failed")
        deps = LifecycleDependencies(fail, deps.train_candidate, deps.evaluate_candidate,
                                     deps.validate_candidate, deps.promote_candidate)
        result = run_cycle(self.request(), deps)
        self.assertEqual(result.status, CycleStatus.CANDIDATE_BUILD_FAILED)
        self.assertEqual(self.calls, ["build"])

    def test_candidate_training_failure_stops(self) -> None:
        result = run_cycle(self.request(), self.dependencies(train_error=True))
        self.assertEqual(result.status, CycleStatus.CANDIDATE_TRAINING_FAILED)
        self.assertNotIn("evaluate", self.calls)
        self.assertNotIn("promote", self.calls)

    def test_pending_policy_and_rejection_do_not_promote(self) -> None:
        for decision, expected in (("PENDING_POLICY", CycleStatus.EVALUATION_PENDING_POLICY),
                                   ("REJECTED", CycleStatus.CANDIDATE_REJECTED)):
            self.report.unlink(missing_ok=True)
            self.calls.clear()
            result = run_cycle(self.request(), self.dependencies(decision=decision))
            self.assertEqual(result.status, expected)
            self.assertNotIn("promote", self.calls)
        self.assert_active_unchanged()

    def test_accepted_candidate_validates_and_invokes_promoter_once(self) -> None:
        result = run_cycle(self.request(), self.dependencies())
        self.assertEqual(result.status, CycleStatus.PROMOTION_SUCCEEDED)
        self.assertEqual(self.calls.count("promote"), 1)
        self.assertEqual(self.calls.count("validate"), 1)
        self.assert_active_unchanged()

    def test_promotion_failure_is_reported_and_active_unchanged(self) -> None:
        result = run_cycle(self.request(), self.dependencies(promote_error=True))
        self.assertEqual(result.status, CycleStatus.PROMOTION_FAILED)
        self.assertTrue(result.report["errors"])
        self.assert_active_unchanged()

    def test_report_is_safe_and_never_overwritten(self) -> None:
        result = run_cycle(self.request(drift_detected=False))
        stored = json.loads(self.report.read_text(encoding="utf-8"))
        self.assertEqual(stored["cycle_status"], result.status.value)
        original = self.report.read_bytes()
        with self.assertRaises(Exception):
            run_cycle(self.request(drift_detected=False))
        self.assertEqual(self.report.read_bytes(), original)

    def test_existing_report_conflict_stops_before_injected_work(self) -> None:
        self.report.parent.mkdir(parents=True, exist_ok=True)
        self.report.write_text('{"keep": true}', encoding="utf-8")
        with self.assertRaises(Exception):
            run_cycle(self.request(), self.dependencies())
        self.assertEqual(self.calls, [])
        self.assertEqual(self.report.read_text(encoding="utf-8"), '{"keep": true}')
        self.assert_active_unchanged()

    def test_overlapping_model_or_report_paths_stop_before_injected_work(self) -> None:
        with self.assertRaises(Exception):
            run_cycle(self.request(candidate_dir=self.active / "nested-candidate"),
                      self.dependencies())
        with self.assertRaises(Exception):
            run_cycle(self.request(report_path=self.candidate / "cycle-report.json"),
                      self.dependencies())
        self.assertEqual(self.calls, [])

    def test_integrated_build_evaluation_promotion_and_rollback(self) -> None:
        """Connect existing pipeline/evaluator/activator APIs using temp artifacts."""
        for path in self.candidate.iterdir():
            path.unlink()
        threshold = 0.5
        metadata = {
            "model_version": "V2", "feature_count": 34,
            "feature_names": FINAL_FEATURES, "target": "isFraud",
            "selected_classification_threshold": threshold,
        }
        (self.active / "model.json").write_text('{"dummy": true}', encoding="utf-8")
        (self.active / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (self.active / "threshold.json").write_text(
            json.dumps({"selected_threshold": threshold}), encoding="utf-8")
        original_active = {p.name: digest(p) for p in self.active.iterdir()}

        class DummyModel:
            def save_model(self, path: str) -> None:
                Path(path).write_text('{"dummy_candidate": true}', encoding="utf-8")

        cycle = CycleConfig(StepWindow(1, 2), StepWindow(3, 4), StepWindow(5, 14),
                            candidate_output_dir=self.candidate)

        def frame(steps: list[int], first_id: int) -> pd.DataFrame:
            rows = []
            for offset, step in enumerate(steps):
                row = {"event_id": first_id + offset, "step": step, "isFraud": offset % 2}
                row.update({name: float(offset + index) for index, name in enumerate(FINAL_FEATURES)})
                rows.append(row)
            return pd.DataFrame(rows, columns=["event_id", "step"] + FINAL_FEATURES + ["isFraud"])

        old, recent = frame([1, 2], 0), frame(list(range(5, 15)), 10)
        held_out = frame([30, 31, 32, 33], 30)
        cycle_report = self.root / "reports" / "cycle.json"
        evaluation_report = self.root / "reports" / "evaluation.json"
        calls: list[str] = []

        def build(context: Mapping[str, Any]) -> dict[str, Any]:
            calls.append("build")
            return split_cycle_dataset(build_cycle_dataset(old, recent, cycle), cycle)

        def train(split: Any, context: Mapping[str, Any]) -> Path:
            calls.append("train")
            candidate_metadata = {
                "model_version": "CYCLE-TEST", "feature_count": 34,
                "feature_names": FINAL_FEATURES, "target": "isFraud",
                "selected_classification_threshold": threshold,
            }
            save_candidate_bundle(DummyModel(), candidate_metadata,
                                 {"selected_threshold": threshold}, cycle,
                                 active_dir=self.active)
            return self.candidate

        class Probabilities:
            def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
                positive = np.asarray([.1, .9, .2, .8])
                return np.column_stack((1 - positive, positive))

        def evaluate(built: Any, trained: Any, context: Mapping[str, Any]) -> Mapping[str, Any]:
            calls.append("evaluate")
            return evaluate_candidate(
                self.candidate, self.active, held_out, evaluation_report,
                policy=AcceptancePolicy({"f1": 0.0, "recall": 0.0}),
                model_loader=lambda path: Probabilities(),
            )

        dependencies = LifecycleDependencies(
            build, train, evaluate,
            lambda directory: validate_bundle(directory, FINAL_FEATURES),
            lambda context: promote_candidate(self.candidate, self.active, self.backup),
        )
        result = run_cycle(self.request(report_path=cycle_report), dependencies)
        self.assertEqual(result.status, CycleStatus.PROMOTION_SUCCEEDED)
        self.assertEqual(calls, ["build", "train", "evaluate"])
        self.assertEqual(validate_bundle(self.active, FINAL_FEATURES).model_version, "CYCLE-TEST")
        self.assertEqual({p.name: digest(p) for p in self.backup.iterdir()}, original_active)

        rollback(self.active, self.backup)
        self.assertEqual({p.name: digest(p) for p in self.active.iterdir()}, original_active)
        self.assertEqual(validate_bundle(self.active, FINAL_FEATURES).model_version, "V2")


if __name__ == "__main__":
    unittest.main()
