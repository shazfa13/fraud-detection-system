"""Focused tests for active-model SHAP explanations and non-fatal explanation failure."""
from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.api import app as fraud_api
from src.api.account_history import AccountHistoryStore
from services.user_explanation import explain_preview, explain_preview_safely


class UserExplanationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.original_store = fraud_api.account_history_store
        fraud_api.account_history_store = AccountHistoryStore(Path(self.temporary.name) / "history.sqlite3")

    def tearDown(self):
        fraud_api.account_history_store = self.original_store
        self.temporary.cleanup()

    def test_shap_explanation_uses_exact_v2_api_preview_vector(self):
        from src.api import app as fraud_api

        fraud_api.account_state.reset()
        transaction = fraud_api.TransactionRequest(
            step=9, type="TRANSFER", amount=127.5, nameOrig="EXPLANATION-TEST-SENDER",
            oldbalanceOrg=500.0, nameDest="EXPLANATION-TEST-DEST", oldbalanceDest=30.0,
        )
        try:
            analysis = fraud_api.analyze_transaction(transaction, idempotency_key="explanation-test")
            prediction = analysis
            self.assertEqual(prediction["model_version"], "V2")
            self.assertEqual(len(analysis["feature_names"]), 34)
            self.assertEqual(list(analysis["features"]), analysis["feature_names"])
            explanation = explain_preview(
                analysis["feature_names"], analysis["features"], prediction["model_version"],
                prediction["fraud_probability"], prediction["prediction"],
                fraud_api.CLASSIFICATION_THRESHOLD,
            )
        finally:
            fraud_api.account_state.reset()

        self.assertEqual(len(explanation), 6)
        self.assertEqual(list(explanation.columns), ["Feature", "Direction", "Contribution (log-odds)"])
        self.assertTrue(set(explanation["Direction"]).issubset({
            "Raises estimated risk", "Lowers estimated risk", "No material contribution",
        }))
        self.assertTrue(explanation["Contribution (log-odds)"].str.match(r"[+-][0-9]+\.[0-9]{4}").all())
        self.assertTrue((explanation["Feature"].str.len() > 0).all())

    def test_missing_shap_failure_is_returned_without_raising(self):
        with patch("services.user_explanation.explain_preview",
                   side_effect=ModuleNotFoundError("No module named 'shap'")):
            result, error = explain_preview_safely([], {}, "V2", 0.25, 0, 0.53)
        self.assertIsNone(result)
        self.assertEqual(error, "No module named 'shap'")


if __name__ == "__main__":
    unittest.main()
