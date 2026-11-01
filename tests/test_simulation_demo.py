"""Lightweight checks for the isolated simulation-demo safety gates."""

import unittest

import pandas as pd

from src.drift.drift_detector import FINAL_FEATURES
from src.retraining.simulation_demo import audit_split, validate_input_dataset
from src.retraining.v2_trainer import EXPECTED_COLUMNS


def _partition(event_ids, steps, labels):
    frame = pd.DataFrame({"event_id": event_ids, "step": steps, "isFraud": labels})
    for feature in FINAL_FEATURES:
        frame[feature] = 0.0
    return frame[EXPECTED_COLUMNS]


def _valid_split():
    return {
        "train_df": _partition([1, 2], [1, 422], [0, 1]),
        "val_df": _partition([3, 4], [423, 565], [0, 1]),
        "test_df": _partition([5, 6], [566, 743], [0, 1]),
    }


class SimulationDemoSafetyTests(unittest.TestCase):
    def test_input_contract_accepts_numeric_features_and_binary_target(self):
        frame = pd.concat(list(_valid_split().values()), ignore_index=True)
        frame = frame[EXPECTED_COLUMNS]

        validate_input_dataset(frame)


    def test_split_audit_reports_disjoint_ids_and_no_exclusions(self):
        audit = audit_split(_valid_split(), input_rows=6)

        self.assertEqual(audit["event_id_overlap_counts"], {
            "train_validation": 0,
            "train_test": 0,
            "validation_test": 0,
        })
        self.assertEqual(audit["input_rows"], audit["assigned_rows"])
        self.assertEqual(audit["assigned_rows"], 6)
        self.assertEqual(audit["excluded_from_candidate_partitions"], 0)
        self.assertEqual(audit["partitions"]["test"]["fraud_count"], 1)


    def test_split_audit_rejects_unexpected_boundaries(self):
        split = _valid_split()
        split["test_df"].loc[0, "step"] = 565

        with self.assertRaisesRegex(ValueError, "Unexpected test step range"):
            audit_split(split, input_rows=6)


    def test_split_audit_rejects_cross_partition_event_id(self):
        split = _valid_split()
        split["val_df"].loc[0, "event_id"] = 1

        with self.assertRaisesRegex(ValueError, "event_id values cross candidate partitions"):
            audit_split(split, input_rows=6)


    def test_input_contract_rejects_non_binary_target(self):
        frame = pd.concat(list(_valid_split().values()), ignore_index=True)[EXPECTED_COLUMNS]
        frame.loc[0, "isFraud"] = 2

        with self.assertRaisesRegex(ValueError, "isFraud must be non-null and contain only 0/1"):
            validate_input_dataset(frame)


if __name__ == "__main__":
    unittest.main()
