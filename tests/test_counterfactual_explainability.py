"""Focused tests for transaction lookup and bounded local explanations."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from services.counterfactual_explainability import (
    find_counterfactual, lookup_transaction, positive_class_shap,
)
from src.retraining.active_model import resolve_active_model


class FakeModel:
    def predict_proba(self, frame):
        # A controlled decision boundary used to prove candidate re-evaluation.
        probability = 0.9 if float(frame.iloc[0]["amount"]) >= 101 else 0.1
        return np.asarray([[1 - probability, probability]])


class CounterfactualExplainabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = resolve_active_model()
        cls.features = list(cls.bundle.feature_names)

    def _fixture(self, directory: str) -> Path:
        values = {name: 1.0 for name in self.features}
        values.update({"amount": 100.0, "sender_balance_before": 500.0,
                       "dest_balance_before": 200.0, "sender_prev_mean_amount": 50.0,
                       "dest_prev_mean_amount": 25.0, "type_TRANSFER": 1.0})
        row = {"event_id": "known-id", "step": 12, **values, "isFraud": 0}
        path = Path(directory) / "transactions.parquet"
        pq.write_table(pa.Table.from_pylist([row]), path)
        return path

    def test_exact_lookup_and_unknown_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._fixture(directory)
            frame, details = lookup_transaction("known-id", self.features, path)
            self.assertEqual(details["transaction_id"], "known-id")
            self.assertEqual(details["step"], 12)
            self.assertEqual(list(frame.columns), self.features)
            self.assertNotIn("isFraud", frame.columns)
            with self.assertRaisesRegex(LookupError, "not found"):
                lookup_transaction("missing-id", self.features, path)

    def test_binary_shap_single_and_two_class_shapes(self):
        one_output = SimpleNamespace(values=np.ones((1, 34)), base_values=np.asarray([0.25]))
        values, base = positive_class_shap(one_output, (1, 34))
        self.assertEqual(values.shape, (1, 34))
        self.assertEqual(base, 0.25)
        two_output = SimpleNamespace(values=np.stack([np.zeros((1, 34)), np.ones((1, 34))], axis=-1),
                                     base_values=np.asarray([[0.1, 0.2]]))
        values, base = positive_class_shap(two_output, (1, 34))
        self.assertTrue(np.all(values == 1))
        self.assertEqual(base, 0.2)

    def test_counterfactual_flips_at_threshold_and_preserves_original(self):
        original = pd.DataFrame([{name: 1.0 for name in self.features}], columns=self.features)
        original.loc[0, ["amount", "sender_balance_before", "dest_balance_before",
                         "sender_prev_mean_amount", "dest_prev_mean_amount", "type_TRANSFER"]] = [100, 500, 200, 50, 25, 1]
        original_before = original.copy(deep=True)
        found = find_counterfactual(FakeModel(), original, self.features, 0.53,
                                    large_threshold=1_000_000, tolerance=0.01)
        self.assertIsNotNone(found)
        self.assertEqual(found["original_prediction"], 0)
        self.assertEqual(found["prediction"], 1)
        self.assertGreaterEqual(found["fraud_probability"], 0.53)
        self.assertEqual(found["features"].iloc[0]["amount_to_sender_balance_ratio"],
                         found["features"].iloc[0]["amount"] / 500)
        pd.testing.assert_frame_equal(original, original_before)


if __name__ == "__main__":
    unittest.main()
