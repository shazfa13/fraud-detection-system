"""Contract checks for the existing API feature builder used by the user portal."""
from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from streamlit.testing.v1 import AppTest

from src.api import app as fraud_api
from src.retraining.active_model import resolve_active_model
from services.api_client import APIClientError, FORBIDDEN_FIELDS, _validate_payload
from src.api.account_history import AccountHistoryStore


class UserPortalFeatureContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.original_store = fraud_api.account_history_store
        self.original_state = fraud_api.account_state
        fraud_api.account_history_store = AccountHistoryStore(Path(self.temporary.name) / "history.sqlite3")
        fraud_api.account_state = fraud_api.AccountState()

    def tearDown(self):
        fraud_api.account_history_store = self.original_store
        fraud_api.account_state = self.original_state
        self.temporary.cleanup()

    def test_api_builder_returns_active_ordered_34_feature_row_without_targets_or_audit_fields(self):
        bundle = resolve_active_model(project_root=fraud_api.PROJECT_ROOT)
        transaction = fraud_api.TransactionRequest(
            step=7, type="TRANSFER", amount=125.0, nameOrig="USER-DEMO-SENDER",
            oldbalanceOrg=500.0, nameDest="USER-DEMO-DEST", oldbalanceDest=50.0,
        )
        fraud_api.account_state.reset()
        try:
            features = fraud_api.build_feature_vector(transaction, update_state=False)
        finally:
            fraud_api.account_state.reset()
        self.assertEqual(list(features.columns), list(bundle.feature_names))
        self.assertEqual(features.shape, (1, 34))
        self.assertFalse((set(features.columns) & (FORBIDDEN_FIELDS | {"event_id", "step"})))
        self.assertEqual(float(fraud_api.CLASSIFICATION_THRESHOLD), bundle.threshold)

    def test_user_api_prediction_matches_model_probability_and_threshold(self):
        transaction = fraud_api.TransactionRequest(
            step=15, type="TRANSFER", amount=125000.0, nameOrig="USER-API-SENDER",
            oldbalanceOrg=500000.0, nameDest="USER-API-DEST", oldbalanceDest=1000.0,
        )
        fraud_api.account_state.reset()
        try:
            row = fraud_api.build_feature_vector(transaction, update_state=False)
            direct_probability = float(fraud_api.model.predict_proba(row)[0, 1])
            fraud_api.account_state.reset()
            result = fraud_api.predict_transaction(transaction)
            self.assertAlmostEqual(result["fraud_probability"], direct_probability, places=7)
            self.assertEqual(result["prediction"], int(direct_probability >= fraud_api.CLASSIFICATION_THRESHOLD))
        finally:
            fraud_api.account_state.reset()

    def test_identical_cold_start_inputs_are_deterministic(self):
        transaction = fraud_api.TransactionRequest(
            step=15, type="TRANSFER", amount=125000.0, nameOrig="IDENTICAL-SENDER",
            oldbalanceOrg=500000.0, nameDest="IDENTICAL-DEST", oldbalanceDest=1000.0,
        )
        try:
            outputs = []
            for _ in range(2):
                fraud_api.account_state.reset()
                outputs.append(fraud_api.predict_transaction(transaction))
            self.assertEqual(outputs[0], outputs[1])
        finally:
            fraud_api.account_state.reset()

    def test_distinct_raw_inputs_change_the_actual_model_feature_row(self):
        first = fraud_api.TransactionRequest(
            step=1, type="PAYMENT", amount=25.0, nameOrig="DIFF-A", oldbalanceOrg=100.0,
            nameDest="DIFF-B", oldbalanceDest=0.0,
        )
        second = fraud_api.TransactionRequest(
            step=19, type="TRANSFER", amount=250000.0, nameOrig="DIFF-C", oldbalanceOrg=300000.0,
            nameDest="DIFF-D", oldbalanceDest=50000.0,
        )
        fraud_api.account_state.reset()
        try:
            first_row = fraud_api.build_feature_vector(first, update_state=False)
            second_row = fraud_api.build_feature_vector(second, update_state=False)
        finally:
            fraud_api.account_state.reset()
        self.assertEqual(list(first_row.columns), list(second_row.columns))
        self.assertEqual(first_row.shape, (1, 34))
        self.assertFalse(first_row.equals(second_row))
        self.assertEqual(float(second_row.loc[0, "amount"]), 250000.0)
        self.assertEqual(float(second_row.loc[0, "type_TRANSFER"]), 1.0)

    def test_client_rejects_missing_transaction_values_and_target_input(self):
        payload = {"step": 1, "type": "TRANSFER", "amount": 1.0, "nameOrig": "A",
                   "oldbalanceOrg": 2.0, "nameDest": "B", "oldbalanceDest": 0.0}
        with self.assertRaises(APIClientError):
            _validate_payload({key: value for key, value in payload.items() if key != "amount"})
        with self.assertRaises(APIClientError):
            _validate_payload({**payload, "isFraud": 1})

    def test_user_form_uses_friendly_raw_inputs_and_preserves_explanations(self):
        dashboard = Path(__file__).parents[1] / "dashboard.py"
        app = AppTest.from_file(str(dashboard), default_timeout=30)
        app.session_state["authenticated_role"] = "user"
        app.session_state["authenticated_username"] = "form-contract-test"
        app.run()

        self.assertFalse(app.exception)
        labels = [widget.label for widget in app.number_input] + [widget.label for widget in app.selectbox] + [widget.label for widget in app.text_input]
        for label in (
            "Transaction time step", "Transaction type", "Transaction amount",
            "Sender's balance before this transaction", "Receiver's balance before this transaction",
            "Sender account reference", "Receiver account reference",
        ):
            self.assertIn(label, labels)
        page_text = " ".join(str(element.value) for element in app.markdown)
        page_text += " " + " ".join(str(element.value) for element in app.caption)
        page_source = (Path(__file__).parents[1] / "pages" / "user_portal.py").read_text(encoding="utf-8")
        page_text += " " + page_source
        expander_labels = [element.label for element in app.expander]
        for phrase in (
            "transaction details below to see the active model",
            "calculated automatically", "history warmup route",
            "not a definitive determination of fraud",
        ):
            self.assertIn(phrase.lower(), page_text.lower())
        self.assertIn("Optional technical details", expander_labels)
        self.assertNotIn("newbalanceOrig", page_text + " ".join(expander_labels))
        self.assertNotIn("newbalanceDest", page_text + " ".join(expander_labels))
        widget_help = " ".join(str(widget.help) for widget in app.number_input)
        self.assertIn("simulated hour", widget_help)


if __name__ == "__main__":
    unittest.main()
