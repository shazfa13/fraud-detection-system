"""Isolated tests for bounded Stage 3 validation sample selection."""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

from src.streaming.stage1_producer import select_validation_transactions, write_validation_sample
from src.drift.drift_detector import generate_feature_dataset
from src.api import app


class BoundedValidationProducerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw.csv"
        self.metadata = self.root / "metadata.json"
        pd.DataFrame([
            {"step": step, "type": "PAYMENT", "amount": float(step), "nameOrig": f"S{step}",
             "oldbalanceOrg": 100.0, "newbalanceOrig": 100.0-step, "nameDest": f"D{step}",
             "oldbalanceDest": 10.0, "newbalanceDest": 10.0+step, "isFraud": 0}
            for step in (2, 4, 4, 5, 6)
        ]).to_csv(self.raw, index=False)
        self.metadata.write_text(json.dumps({"model_version": "V2", "train_validation_test_strategy": {
            "validation_end_step": 3}}), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_selects_distinct_monotonic_heldout_source_rows(self):
        rows, report = select_validation_transactions(3, raw_data_path=self.raw, metadata_path=self.metadata)
        self.assertEqual(rows.step.tolist(), [4, 5, 6])
        self.assertTrue(rows.step.is_monotonic_increasing)
        self.assertEqual(report["validation_end_step"], 3)
        self.assertEqual(report["selected_row_count"], 3)
        self.assertEqual(report["distinct_transaction_count"], 3)
        self.assertEqual(report["duplicate_source_count"], 0)

    def test_fails_safely_when_bounded_sample_cannot_be_constructed(self):
        with self.assertRaises(RuntimeError):
            select_validation_transactions(4, raw_data_path=self.raw, metadata_path=self.metadata)

    def test_offline_full_history_replay_matches_stream_feature_builder(self):
        source = pd.read_csv(self.raw).sort_values("step", kind="stable")
        windows = {"reference_window": {"max_step": 3}, "recent_window": {"min_step": 4}}
        source_features = generate_feature_dataset(source, windows)
        original_state = app.account_state
        app.account_state = app.AccountState()
        try:
            stream_rows = []
            for row in source.itertuples(index=False):
                txn = app.TransactionRequest(step=int(row.step), type=str(row.type), amount=float(row.amount),
                    nameOrig=str(row.nameOrig), oldbalanceOrg=float(row.oldbalanceOrg),
                    nameDest=str(row.nameDest), oldbalanceDest=float(row.oldbalanceDest))
                stream_rows.append(app.build_feature_vector(txn).iloc[0].to_dict())
        finally:
            app.account_state = original_state
        stream_features = pd.DataFrame(stream_rows, columns=app.FINAL_FEATURES)
        self.assertEqual(list(stream_features.columns), app.FINAL_FEATURES)
        np.testing.assert_allclose(source_features[app.FINAL_FEATURES].to_numpy(dtype=float),
                                   stream_features.to_numpy(dtype=float), equal_nan=True)

    def test_validation_file_is_bounded_offline_artifact(self):
        output = self.root / "sample.jsonl"
        report = write_validation_sample(2, output, raw_data_path=self.raw, metadata_path=self.metadata)
        events = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(events), 2)
        self.assertTrue(all(row["data_origin"] == "synthetic_validation_offline" for row in events))
        self.assertEqual(report["selected_row_count"], 2)
        self.assertNotIn("isFraud", events[0])
        with self.assertRaises(FileExistsError):
            write_validation_sample(2, output, raw_data_path=self.raw, metadata_path=self.metadata)


if __name__ == "__main__":
    unittest.main()
