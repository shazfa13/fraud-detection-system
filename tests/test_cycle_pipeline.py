"""Small temporary-data tests for cycle-configurable retraining helpers."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.retraining import config as phase10_config
from src.retraining import v2_config
from src.retraining.cycle_pipeline import (
    DEFAULT_CYCLE_CONFIG, FINAL_FEATURES, CycleConfig, StepWindow, build_cycle_dataset,
    save_candidate_bundle, select_validation_threshold, split_cycle_dataset,
)


def make_frame(steps: list[int], first_id: int = 0, audit: bool = False) -> pd.DataFrame:
    rows = []
    for i, step in enumerate(steps):
        row = {"event_id": first_id + i, "step": step, "isFraud": i % 2}
        row.update({name: float(i + j) for j, name in enumerate(FINAL_FEATURES)})
        if audit:
            row.update({"fraud_probability": .4, "predicted_label": 0,
                        "risk_score": 40.0, "risk_level": "MEDIUM"})
        rows.append(row)
    frame = pd.DataFrame(rows)
    columns = ["event_id", "step"] + FINAL_FEATURES + ["isFraud"]
    if audit:
        columns += ["fraud_probability", "predicted_label", "risk_score", "risk_level"]
    return frame[columns]


class DummyModel:
    def save_model(self, path: str) -> None:
        Path(path).write_text(json.dumps({"learner": "dummy"}), encoding="utf-8")


class CyclePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cycle = CycleConfig(StepWindow(1, 2), StepWindow(3, 4), StepWindow(5, 14))

    def test_defaults_match_original_v2_workflow(self):
        self.assertEqual((DEFAULT_CYCLE_CONFIG.old_window.start, DEFAULT_CYCLE_CONFIG.old_window.end),
                         phase10_config.EXPECTED_OLD_TRAIN_STEP_RANGE)
        self.assertEqual((DEFAULT_CYCLE_CONFIG.validation_gap.start, DEFAULT_CYCLE_CONFIG.validation_gap.end),
                         phase10_config.EXPECTED_VALIDATION_GAP)
        self.assertEqual((DEFAULT_CYCLE_CONFIG.recent_window.start, DEFAULT_CYCLE_CONFIG.recent_window.end),
                         phase10_config.EXPECTED_RECENT_STEP_RANGE)
        self.assertEqual(DEFAULT_CYCLE_CONFIG.recent_train_fraction, v2_config.RECENT_TRAIN_EXTENSION_FRACTION)
        self.assertEqual(DEFAULT_CYCLE_CONFIG.candidate_output_dir,
                         v2_config.PROJECT_ROOT / "models" / "candidate")

    def test_configurable_windows_exclude_gap_and_drop_audit_columns(self):
        old = make_frame([1, 2])
        recent = make_frame([3, *range(5, 15)], 10, audit=True)
        built = build_cycle_dataset(old, recent, self.cycle)
        self.assertEqual(set(built.step), {1, 2, *range(5, 15)})
        self.assertEqual(list(built.columns), ["event_id", "step"] + FINAL_FEATURES + ["isFraud"])
        self.assertFalse(built.step.isin([3, 4]).any())

    def test_chronological_split_is_step_disjoint(self):
        split = split_cycle_dataset(make_frame([1, 2, *range(5, 15)]), self.cycle)
        train, val, test = split["train_df"], split["val_df"], split["test_df"]
        self.assertLess(train.step.max(), val.step.min())
        self.assertLess(val.step.max(), test.step.min())
        self.assertFalse(set(train.event_id) & set(val.event_id))
        self.assertFalse(set(val.event_id) & set(test.event_id))

    def test_split_rejects_gap_rows(self):
        with self.assertRaisesRegex(ValueError, "outside configured windows"):
            split_cycle_dataset(make_frame([1, 2, 3, *range(5, 15)]), self.cycle)

    def test_contract_order_and_target_separation(self):
        frame = make_frame([1, 2, *range(5, 15)])
        reordered = frame[["event_id", "step"] + list(reversed(FINAL_FEATURES)) + ["isFraud"]]
        with self.assertRaisesRegex(ValueError, "columns do not match"):
            split_cycle_dataset(reordered, self.cycle)
        self.assertNotIn("event_id", FINAL_FEATURES)
        self.assertNotIn("step", FINAL_FEATURES)
        self.assertNotIn("isFraud", FINAL_FEATURES)

    def test_duplicate_and_cross_window_overlap_rejected(self):
        old = make_frame([1, 2])
        recent = make_frame(list(range(5, 15)), 10)
        repeated = pd.concat([old, old.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate event_id"):
            build_cycle_dataset(repeated, recent, self.cycle)
        recent.loc[recent.index[0], "event_id"] = old.event_id.iloc[0]
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_cycle_dataset(old, recent, self.cycle)

    def test_candidate_writer_isolated_from_active(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); active = root / "active"; candidate = root / "candidate"
            active.mkdir(); candidate.mkdir()
            cycle = CycleConfig(StepWindow(1, 2), StepWindow(3, 4), StepWindow(5, 14),
                                candidate_output_dir=candidate)
            metadata = {"model_version": "CYCLE-A", "feature_count": 34,
                        "feature_names": FINAL_FEATURES, "target": "isFraud",
                        "selected_classification_threshold": .42}
            hashes = save_candidate_bundle(DummyModel(), metadata,
                                           {"selected_threshold": .42}, cycle, active_dir=active)
            self.assertEqual(set(hashes), {"model.json", "metadata.json", "threshold.json"})
            self.assertEqual({p.name for p in candidate.iterdir()}, set(hashes))
            self.assertEqual(list(active.iterdir()), [])
            active_cycle = CycleConfig(StepWindow(1, 2), StepWindow(3, 4), StepWindow(5, 14),
                                       candidate_output_dir=active)
            with self.assertRaisesRegex(ValueError, "must be separate from the active"):
                save_candidate_bundle(DummyModel(), metadata, {"selected_threshold": .42},
                                      active_cycle, active_dir=active)

    def test_candidate_writer_refuses_existing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); active = root / "active"; candidate = root / "candidate"
            active.mkdir(); candidate.mkdir()
            sentinel = candidate / "model.json"
            sentinel.write_text('{"keep": true}', encoding="utf-8")
            cycle = CycleConfig(StepWindow(1, 2), StepWindow(3, 4), StepWindow(5, 14),
                                candidate_output_dir=candidate)
            metadata = {"model_version": "CYCLE-B", "feature_count": 34,
                        "feature_names": FINAL_FEATURES, "target": "isFraud"}
            with self.assertRaises(FileExistsError):
                save_candidate_bundle(DummyModel(), metadata, {"selected_threshold": .5},
                                      cycle, active_dir=active)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), '{"keep": true}')
            self.assertEqual(len(list(candidate.iterdir())), 1)

    def test_threshold_selection_uses_validation_only(self):
        class Recorder:
            seen = None
            def predict_proba(self, features):
                self.seen = features.copy()
                probabilities = np.linspace(.1, .9, len(features))
                return np.column_stack((1 - probabilities, probabilities))
        validation = make_frame([5, 6, 7, 8]); model = Recorder()
        threshold, _ = select_validation_threshold(model, validation)
        self.assertTrue(.01 <= threshold <= .99)
        self.assertEqual(list(model.seen.columns), FINAL_FEATURES)
        self.assertEqual(model.seen["amount"].tolist(), validation["amount"].tolist())
        self.assertNotIn("isFraud", model.seen.columns)
        self.assertNotIn("step", model.seen.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)

