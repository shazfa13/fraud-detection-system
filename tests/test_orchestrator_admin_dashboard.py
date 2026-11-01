"""Isolated checks for the optional lifecycle panel in the existing admin page."""
from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from services import data_service
from src.orchestrator import approval, config, overview, promotion
from src.orchestrator.eligibility import evaluate_summaries
from ui import components


def _overview(*, runs=None, labels=None):
    return {
        "active_model": {
            "status": "VALID", "model_version": "V2", "model_artifact_path": "models/xgboost_model_v2.json",
            "manifest_present": False, "integrity": {"status": "VALIDATED_BY_SHARED_RESOLVER"},
        },
        "latest_drift": {
            "status": "NO_SAVED_REPORT", "report_timestamp": None, "drift_status": None,
            "drifted_features": [], "feature_scores": [], "warnings": [],
        },
        "label_readiness": labels or {
            "status": "AVAILABLE", "total_labels": 3, "genuinely_verified_labels": 0,
            "unmatched_labels": 1, "conflict_count": 0, "unresolved_or_rejected_count": 0,
            "matching_key": "transaction_id", "simulated_labels": 3,
            "phase9_label_type": "SIMULATED_DELAYED_GROUND_TRUTH",
            "source_type_counts": {"synthetic_test": 3}, "readiness_reasons": ["Simulated labels are not genuine."],
        },
        "run_history": {"status": "AVAILABLE", "runs": list(runs or [])},
    }


def _eligibility():
    return {"decision_status": "BLOCKED", "eligible": False,
            "reasons": ["No saved monitoring report is available."], "warnings": []}


class OrchestratorAdminDashboardTests(unittest.TestCase):
    def _admin_page(self):
        app = AppTest.from_file(str(Path(__file__).parents[1] / "dashboard.py"), default_timeout=30)
        app.session_state["authenticated_role"] = "admin"
        app.session_state["authenticated_username"] = "isolated-test-admin"
        return app

    def _open_orchestrator(self, app):
        app.run()
        app.switch_page("pages/12_Model_Orchestrator.py").run()

    def test_admin_page_renders_concise_overview_and_dedicated_navigation(self):
        app = self._admin_page()
        with patch.object(components, "cached_api_health", return_value=(True, {"model_version": "V2"}, None)), \
             patch.object(data_service, "prediction_registry_summary", return_value={
                 "available": True, "total": 12, "predicted_fraud": 2, "predicted_legitimate": 10,
             }), \
             patch.object(data_service, "recent_stage1_predictions", return_value=[{
                 "transaction_id": "TX-1", "prediction_timestamp": "2020-01-01T00:00:00Z",
                 "registered_at": "2020-01-01T00:00:00Z",
             }]), \
             patch.object(data_service, "load_active_model_snapshot", return_value={
                 "available": True, "model_version": "V2", "source": "shared resolver",
             }), \
             patch.object(data_service, "load_prediction_history", return_value=[{
                 "transaction_id": "TX-1", "amount": 42.5, "prediction": 1, "fraud_probability": 0.91,
             }]), \
             patch("src.orchestrator.drift_adapter.get_latest_drift_summary", return_value={
                 "status": "SAVED_REPORT", "drift_decision": False, "warnings": [],
             }), \
             patch.object(approval, "record_admin_decision") as record_decision, \
             patch.object(promotion, "promote_approved_candidate") as promote:
            app.run()
        self.assertFalse(app.exception)
        rendered = " ".join(str(item.value) for item in app.markdown)
        self.assertIn("Fraud Detection Dashboard", rendered)
        self.assertIn("Recent predictions", rendered)
        self.assertEqual([item.value for item in app.metric], ["V2", "0"])
        self.assertIn("Registered predictions total: 12", " ".join(str(item.value) for item in app.caption))
        self.assertIn("Admin Overview", [item.label for item in app.page_link])
        self.assertIn("Pipeline Evidence", [item.label for item in app.page_link])
        self.assertNotIn("Retraining Lab", [item.label for item in app.page_link])
        self.assertEqual([item.label for item in app.page_link], [
            "Admin Overview", "Transaction Analyzer", "Transaction Explorer", "Counterfactual Explainability",
            "Pipeline Evidence", "Drift & Alerts", "Delayed Labels", "System Health",
            "Model Performance", "Model Orchestrator", "System Architecture", "Reports & Documentation",
        ])
        self.assertNotIn("Fraud Model Lifecycle / Retraining Orchestrator", rendered)
        record_decision.assert_not_called()
        promote.assert_not_called()

    def test_existing_run_id_is_displayed_without_starting_work(self):
        run_id = "20261030T000000Z_" + "a" * 32
        app = self._admin_page()
        with patch.object(components, "cached_api_health", return_value=(False, None, "offline")), \
             patch.object(data_service, "prediction_registry_summary", return_value={"available": False}), \
             patch.object(data_service, "load_active_model_snapshot", return_value={"available": False}), \
             patch.object(data_service, "load_prediction_history", return_value=[]), \
             patch("src.orchestrator.drift_adapter.get_latest_drift_summary", return_value={
                 "status": "NO_SAVED_REPORT", "warnings": [],
             }), \
             patch.object(approval, "get_admin_decision", return_value=None), \
             patch.object(approval, "record_admin_decision") as record_decision, \
             patch.object(promotion, "promote_approved_candidate") as promote:
            app.run()
        self.assertFalse(app.exception)
        self.assertNotIn(run_id, " ".join(str(item.value) for item in app.markdown))
        self.assertIn("Model Orchestrator", [item.label for item in app.page_link])
        record_decision.assert_not_called()
        promote.assert_not_called()

    def _candidate_fixture(self, temporary):
        run_id = "20261030T000000Z_" + "e" * 32
        root = Path(temporary) / "orchestrator_runs"
        report_dir = root / run_id / "evaluation"
        report_dir.mkdir(parents=True)
        (report_dir / "evaluation_report.json").write_text(json.dumps({
            "run_id": run_id, "evaluation_status": "COMPLETED", "decision_status": "REVIEW_REQUIRED",
            "models": {"candidate": {"model_version": "CANDIDATE-TEST"}, "active": {"model_version": "V2"}},
            "metrics": {"candidate": {"f1": 0.8}, "active": {"f1": 0.7}},
        }), encoding="utf-8")
        data = _overview(runs=[{"run_id": run_id, "workflow_status": "INITIALIZED", "current_stage": "run_created"}])
        return run_id, root, data

    def test_approval_action_requires_reason_and_confirmation_then_calls_stage7(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_id, root, data = self._candidate_fixture(temporary)
            app = self._admin_page()
            review = {"review_token": "fresh-review-token", "evaluation_status": "REVIEW_REQUIRED",
                      "candidate_metrics": {"f1": 0.8}, "active_model_metrics": {"f1": 0.7}}
            with patch.object(config, "RUNS_ROOT", root), \
                 patch.object(components, "cached_api_health", return_value=(False, None, "offline")), \
                 patch.object(overview, "get_orchestrator_overview", return_value=data), \
                 patch("src.orchestrator.eligibility.evaluate_retraining_eligibility", return_value=_eligibility()), \
                 patch.object(approval, "get_admin_decision", return_value=None), \
                 patch.object(approval, "get_approval_review", return_value=review), \
                 patch.object(approval, "record_admin_decision", return_value={"decision": "REJECTED"}) as record:
                self._open_orchestrator(app)
                app.button(key=f"model_orchestrator_review_{run_id}").click().run()
                form_key = f"model_orchestrator_decision_form_{run_id}"
                app.radio(key=f"model_orchestrator_decision_{run_id}").set_value("REJECTED")
                app.button(key=f"FormSubmitter:{form_key}-Record approval or rejection").click().run()
                record.assert_not_called()
                app.text_area(key=f"model_orchestrator_decision_reason_{run_id}").set_value("Reject the isolated candidate after review.")
                app.checkbox(key=f"model_orchestrator_decision_confirm_{run_id}").check()
                app.button(key=f"FormSubmitter:{form_key}-Record approval or rejection").click().run()
                record.assert_called_once()
                self.assertEqual(record.call_args.args[1], "REJECTED")
                self.assertEqual(record.call_args.args[2], "Reject the isolated candidate after review.")
                self.assertNotIn("identity", record.call_args.kwargs)
                self.assertNotIn("role", record.call_args.kwargs)
            self.assertFalse(app.exception)

    def test_promotion_action_requires_separate_confirmation_and_calls_stage8(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_id, root, data = self._candidate_fixture(temporary)
            app = self._admin_page()
            approved = {"decision": "APPROVED", "administrator": {"identity": "fixture-admin", "role": "admin"}}
            review = {"promotion_ready": True, "promotion_token": "fresh-promotion-token",
                      "candidate_version": "CANDIDATE-TEST", "active_model_version": "V2"}
            with patch.object(config, "RUNS_ROOT", root), \
                 patch.object(components, "cached_api_health", return_value=(False, None, "offline")), \
                 patch.object(overview, "get_orchestrator_overview", return_value=data), \
                 patch("src.orchestrator.eligibility.evaluate_retraining_eligibility", return_value=_eligibility()), \
                 patch.object(approval, "get_admin_decision", return_value=approved), \
                 patch.object(promotion, "get_promotion_review", return_value=review), \
                 patch.object(promotion, "promote_approved_candidate", return_value={"status": "PROMOTED"}) as promote:
                self._open_orchestrator(app)
                review_key = f"model_orchestrator_promotion_review_{run_id}"
                app.button(key=review_key).click().run()
                form_key = f"model_orchestrator_promotion_form_{run_id}"
                app.text_area(key=f"model_orchestrator_promotion_reason_{run_id}").set_value("Promote after isolated review.")
                app.button(key=f"FormSubmitter:{form_key}-Promote approved candidate").click().run()
                promote.assert_not_called()
                app.checkbox(key=f"model_orchestrator_promotion_confirm_{run_id}").check()
                app.button(key=f"FormSubmitter:{form_key}-Promote approved candidate").click().run()
                promote.assert_called_once()
                self.assertEqual(promote.call_args.kwargs["review_token"], "fresh-promotion-token")
                self.assertNotIn("identity", promote.call_args.kwargs)
                self.assertNotIn("role", promote.call_args.kwargs)
            self.assertFalse(app.exception)

    def test_non_admin_dashboard_route_has_no_lifecycle_panel(self):
        app = AppTest.from_file(str(Path(__file__).parents[1] / "dashboard.py"), default_timeout=30)
        app.session_state["authenticated_role"] = "user"
        app.session_state["authenticated_username"] = "isolated-test-user"
        with patch.object(components, "cached_api_health", return_value=(False, None, "offline")):
            app.run()
        self.assertFalse(app.exception)
        rendered = " ".join(str(item.value) for item in app.markdown)
        self.assertNotIn("Fraud Model Lifecycle / Retraining Orchestrator", rendered)

    def test_unknown_or_simulated_label_evidence_never_grants_eligibility(self):
        active = {"status": "VALID", "model_version": "V2",
                  "integrity": {"status": "VALIDATED_BY_SHARED_RESOLVER"}}
        drift = {"status": "SAVED_REPORT", "report_exists": True,
                 "drift_decision": True, "drift_status": "drift"}
        for label_patch in (
            {"status": "AVAILABLE", "strict_retraining_ready": False, "genuinely_verified_labels": 0,
             "simulated_labels": 100, "matching_key": "event_id",
             "trusted_source_configuration": {"configured_sources": ["fixture"], "authorization_token_configured": True},
             "conflict_count": 0, "unmatched_labels": 0, "unresolved_or_rejected_count": 0, "readiness_reasons": []},
            {"status": "UNKNOWN", "strict_retraining_ready": True, "genuinely_verified_labels": 100,
             "matching_key": "event_id", "trusted_source_configuration": {"configured_sources": ["fixture"], "authorization_token_configured": True},
             "conflict_count": 0, "unmatched_labels": 0, "unresolved_or_rejected_count": 0, "readiness_reasons": []},
        ):
            result = evaluate_summaries(active, drift, label_patch)
            self.assertEqual(result["decision_status"], "BLOCKED")
            self.assertFalse(result["eligible"])

    def test_stage8_promotion_lock_rejects_a_second_orchestrator_promoter(self):
        from src.orchestrator.promotion import PromotionError, _promotion_lock

        with self.subTest("temporary lock only"):
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with _promotion_lock(root, "20261030T000000Z_" + "b" * 32, "test-admin"):
                    with self.assertRaises(PromotionError):
                        with _promotion_lock(root, "20261030T000001Z_" + "c" * 32, "test-admin"):
                            self.fail("Second promoter unexpectedly acquired the lock.")


if __name__ == "__main__":
    unittest.main()
