import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.xai import explain_v2 as xai


FEATURES = json.loads(xai.METADATA_PATH.read_text(encoding="utf-8"))["feature_names"]


class FakePlots:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


class FakeModel:
    def predict_proba(self, frame):
        return np.array([[0.4, 0.6]])


class FakeTreeExplainer:
    def __init__(self, _model, **_kwargs):
        pass

    def __call__(self, frame):
        return FakeExplanation(np.ones(frame.shape, dtype=float))


class FakeExplanation:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, _index):
        return self


class XaiV2Tests(unittest.TestCase):
    def _dataset(self, directory):
        rows = []
        for i in range(3):
            rows.append({"event_id": f"evt-{i}", "step": i + 1,
                         **{name: float(i + j) for j, name in enumerate(FEATURES)}, "isFraud": i % 2})
        path = Path(directory) / "small.parquet"
        pd.DataFrame(rows, columns=["event_id", "step", *FEATURES, "isFraud"]).to_parquet(path, index=False)
        return pq.ParquetFile(path)

    def test_contract_exact_order_and_excludes_audit_fields(self):
        self.assertEqual(xai.validate_feature_contract(FEATURES, ["event_id", "step", *FEATURES, "isFraud"]), FEATURES)
        for forbidden in ("event_id", "step", "isFraud"):
            with self.assertRaisesRegex(ValueError, "cannot be model features"):
                xai.validate_feature_contract([forbidden, *FEATURES[1:]], [forbidden, *FEATURES[1:]])
        with self.assertRaisesRegex(ValueError, "missing"):
            xai.validate_feature_contract(FEATURES, FEATURES[:-1])
        with self.assertRaisesRegex(ValueError, "reordered"):
            xai.validate_feature_contract(FEATURES, list(reversed(FEATURES)))
        with self.assertRaisesRegex(ValueError, "model feature names/order"):
            xai.validate_feature_contract(FEATURES, FEATURES, list(reversed(FEATURES)))

    def test_sampling_and_local_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            pf = self._dataset(tmp)
            first = xai.sample_features(pf, FEATURES, 2, 42)
            second = xai.sample_features(pf, FEATURES, 2, 42)
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(list(first.columns), FEATURES)
            row, audit = xai.select_local_row(pf, FEATURES, 1, None)
            self.assertEqual(list(row.columns), FEATURES)
            self.assertEqual(audit["transaction_id"], "evt-1")
            row_by_id, audit_by_id = xai.select_local_row(pf, FEATURES, None, "evt-2")
            self.assertEqual(audit_by_id["row_index"], 2)
            self.assertEqual(row_by_id.iloc[0].tolist(), [float(2 + j) for j in range(34)])
            for args in ((pf, FEATURES, 5, None), (pf, FEATURES, None, "missing")):
                with self.assertRaisesRegex(ValueError, "out of range|not found"):
                    xai.select_local_row(*args)
            pf.close()

    def test_object_features_convert_to_numeric_and_keep_missing_as_nan(self):
        frame = pd.DataFrame([{name: "1.25" for name in FEATURES}], columns=FEATURES)
        nullable = ["amount_to_dest_balance_ratio", "sender_prev_mean_amount",
                    "sender_prev_max_amount", "dest_prev_mean_amount", "dest_prev_max_amount",
                    "amount_vs_sender_hist_mean", "amount_vs_dest_hist_mean",
                    "sender_hours_since_prev_txn", "dest_hours_since_prev_txn"]
        for name in nullable:
            frame[name] = None
        frame.loc[0, nullable[0]] = "2.5"
        numeric = xai._prepare_feature_frame(frame, FEATURES)
        self.assertEqual(list(numeric.columns), FEATURES)
        self.assertTrue(all(pd.api.types.is_numeric_dtype(numeric[name]) for name in FEATURES))
        self.assertEqual(numeric.loc[0, nullable[0]], 2.5)
        self.assertTrue(numeric.loc[0, nullable[1:]].isna().all())

    def test_invalid_object_feature_values_fail_clearly(self):
        frame = pd.DataFrame([{name: 1.0 for name in FEATURES}], columns=FEATURES)
        feature = "sender_prev_mean_amount"
        frame[feature] = frame[feature].astype(object)
        frame.loc[0, feature] = "not-a-number"
        with self.assertRaisesRegex(ValueError, feature):
            xai._prepare_feature_frame(frame, FEATURES)
        frame.loc[0, feature] = float("inf")
        with self.assertRaisesRegex(ValueError, "infinite"):
            xai._prepare_feature_frame(frame, FEATURES)

    def test_local_explanation_normalizes_object_values_before_prediction_and_shap(self):
        nullable = ["amount_to_dest_balance_ratio", "sender_prev_mean_amount",
                    "sender_prev_max_amount", "dest_prev_mean_amount", "dest_prev_max_amount",
                    "amount_vs_sender_hist_mean", "amount_vs_dest_hist_mean",
                    "sender_hours_since_prev_txn", "dest_hours_since_prev_txn"]
        raw = pd.DataFrame([{name: "3.5" for name in FEATURES}], columns=FEATURES)
        for name in nullable:
            raw[name] = None
        values = np.ones((1, 34), dtype=float)
        explanation = FakeExplanation(values)
        fake_shap = SimpleNamespace(plots=FakePlots())

        def checked_shap(_model, model_frame):
            self.assertEqual(list(model_frame.columns), FEATURES)
            self.assertTrue(all(pd.api.types.is_numeric_dtype(model_frame[name]) for name in FEATURES))
            self.assertTrue(model_frame[nullable].isna().all().all())
            return fake_shap, None, explanation, values

        with tempfile.TemporaryDirectory() as tmp, patch.object(xai, "OUTPUT_DIR", Path(tmp)), \
                patch.object(xai, "_shap_values", side_effect=checked_shap):
            xai._local(FakeModel(), raw,
                       {"row_index": 0, "transaction_id": "evt-0", "ground_truth_isFraud": 0}, 0.53)

    def test_selected_local_row_has_numeric_dtype_for_pyarrow_nulls(self):
        nullable = ["amount_to_dest_balance_ratio", "sender_prev_mean_amount",
                    "sender_prev_max_amount", "dest_prev_mean_amount", "dest_prev_max_amount",
                    "amount_vs_sender_hist_mean", "amount_vs_dest_hist_mean",
                    "sender_hours_since_prev_txn", "dest_hours_since_prev_txn"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "local.parquet"
            rows = [{"event_id": "evt-0", "step": 1,
                     **{name: (None if name in nullable else float(i)) for i, name in enumerate(FEATURES)},
                     "isFraud": 0}]
            pd.DataFrame(rows, columns=["event_id", "step", *FEATURES, "isFraud"]).to_parquet(path, index=False)
            pf = pq.ParquetFile(path)
            frame, _ = xai.select_local_row(pf, FEATURES, 0, None)
            self.assertTrue(all(pd.api.types.is_numeric_dtype(frame[name]) for name in FEATURES))
            self.assertTrue(frame[nullable].isna().all().all())
            pf.close()

    def test_global_csv_and_local_report_generation_with_mocked_shap(self):
        values = np.arange(68, dtype=float).reshape(2, 34) / 10
        explanation = FakeExplanation(values)
        fake_shap = SimpleNamespace(plots=FakePlots())
        frame = pd.DataFrame(np.ones((2, 34)), columns=FEATURES)
        with tempfile.TemporaryDirectory() as tmp, patch.object(xai, "OUTPUT_DIR", Path(tmp)), \
                patch.object(xai, "_shap_values", return_value=(fake_shap, None, explanation, values)):
            outputs = xai._global(FakeModel(), frame)
            self.assertIn("global_shap_importance.csv", outputs)
            importance = pd.read_csv(Path(tmp) / "global_shap_importance.csv")
            self.assertEqual(len(importance), 34)
            self.assertTrue(importance["mean_absolute_shap_value_raw_margin"].map(np.isfinite).all())
            local_outputs, result = xai._local(FakeModel(), frame.iloc[[0]],
                {"row_index": 0, "transaction_id": "evt-0", "ground_truth_isFraud": 0}, 0.53)
            self.assertIn("local_shap_waterfall.png", local_outputs)
            self.assertEqual(result["predicted_class_at_existing_threshold"], 1)
            report = pd.read_csv(Path(tmp) / "local_shap_explanation.csv")
            self.assertEqual(list(report.columns), ["feature", "feature_value", "shap_value_raw_margin"])
            self.assertTrue(np.isfinite(report["shap_value_raw_margin"]).all())

    def test_shap_values_are_shape_checked_and_finite(self):
        fake_shap_module = SimpleNamespace(TreeExplainer=FakeTreeExplainer)
        frame = pd.DataFrame(np.ones((2, 34)), columns=FEATURES)
        with patch.dict(sys.modules, {"shap": fake_shap_module}):
            _, _, _, values = xai._shap_values(FakeModel(), frame)
            self.assertEqual(values.shape, frame.shape)
            self.assertTrue(np.isfinite(values).all())

            class BadExplainer(FakeTreeExplainer):
                def __call__(self, _frame):
                    return FakeExplanation(np.full((2, 34), np.nan))

            fake_shap_module.TreeExplainer = BadExplainer
            with self.assertRaisesRegex(ValueError, "finite"):
                xai._shap_values(FakeModel(), frame)

    def test_configured_threshold_and_artifact_files_are_read_only(self):
        before = (xai.MODEL_PATH.read_bytes(), xai.METADATA_PATH.read_bytes(), xai.THRESHOLD_PATH.read_bytes())
        _, features, threshold = xai._read_config()
        self.assertEqual(len(features), 34)
        self.assertEqual(threshold, 0.53)
        self.assertEqual(before, (xai.MODEL_PATH.read_bytes(), xai.METADATA_PATH.read_bytes(), xai.THRESHOLD_PATH.read_bytes()))


if __name__ == "__main__":
    unittest.main()
