"""UI checks for the admin-only Model Orchestrator Streamlit page."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.orchestrator import approval, config, overview, promotion, run_store, training
from ui import components


def _overview(*, runs=None, drift=None):
    return {
        "active_model": {
            "status": "VALID", "model_version": "V2", "model_artifact_path": "models/xgboost_model_v2.json",
            "manifest_present": False, "integrity": {"status": "VALIDATED_BY_SHARED_RESOLVER"},
        },
        "latest_drift": drift or {"status": "NO_SAVED_REPORT", "report_exists": False,
                                   "drift_status": None, "warnings": []},
        "label_readiness": {
            "status": "AVAILABLE", "total_labels": 14, "genuinely_verified_labels": 0,
            "unmatched_labels": 14, "conflict_count": 0, "unresolved_or_rejected_count": 0,
            "matching_key": "transaction_id", "strict_retraining_ready": False,
            "simulated_labels": 14, "phase9_label_type": "SIMULATED_DELAYED_GROUND_TRUTH",
            "source_type_counts": {"synthetic_test": 14},
            "readiness_reasons": ["Simulated labels are not verified ground truth."],
        },
        "run_history": {"status": "AVAILABLE", "runs": list(runs or [])},
    }


def _eligibility(status="BLOCKED"):
    return {
        "decision_status": status, "eligible": status == "ELIGIBLE",
        "active_model_version": "V2", "drift_status": "DRIFT_DETECTED",
        "rule_version": "orchestrator-eligibility-v1",
        "configuration": {"minimum_genuine_labels": 100},
        "reasons": ["No verified event_id-matched labels are available."], "warnings": [],
    }


class ModelOrchestratorPageTests(unittest.TestCase):
    def _admin_app(self):
        app = AppTest.from_file(str(Path(__file__).parents[1] / "dashboard.py"), default_timeout=30)
        app.session_state["authenticated_role"] = "admin"
        app.session_state["authenticated_username"] = "isolated-model-admin"
        return app

    def _open_orchestrator(self, app):
        app.run()
        self.assertIn("Model Orchestrator", [item.label for item in app.page_link])
        app.switch_page("pages/12_Model_Orchestrator.py").run()

    def test_admin_route_shows_blocked_simulated_labels_and_no_run_action(self):
        app = self._admin_app()
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "runs"
            with patch.object(config, "RUNS_ROOT", run_root), \
             patch.object(components, "cached_api_health", return_value=(False, None, "offline")), \
             patch.object(overview, "get_orchestrator_overview", return_value=_overview()), \
             patch("src.orchestrator.eligibility.evaluate_retraining_eligibility", return_value=_eligibility()), \
             patch.object(promotion, "promote_approved_candidate") as promote, \
             patch.object(run_store, "initialize_run") as initialize_run, \
             patch.object(training, "train_candidate") as train_candidate:
                self._open_orchestrator(app)
                self.assertFalse(app.exception)
                text = " ".join(str(item.value) for item in app.markdown)
                warnings = " ".join(str(item.value) for item in app.warning)
                self.assertIn("BLOCKED", text)
                self.assertIn("simulated", (text + warnings).casefold())
                self.assertIn("Genuine label availability", text)
                self.assertIn("Unavailable: current genuine-label eligibility is BLOCKED", " ".join(str(item.value) for item in app.caption))
                self.assertIn("No real candidate run is available", " ".join(str(item.value) for item in app.info))
                self.assertIn("No orchestrator runs", text)
                self.assertFalse(any("Data drift detected" in str(item.value) for item in app.error))
                starts = [button for button in app.button if button.label == "Start candidate retraining"]
                self.assertEqual(len(starts), 1)
                self.assertTrue(starts[0].disabled)
                app.button(key="model_orchestrator_refresh").click().run()
                self.assertFalse(run_root.exists(), "Page refresh unexpectedly created a run root.")
                promote.assert_not_called()
                initialize_run.assert_not_called()
                train_candidate.assert_not_called()

    def test_retraining_lab_evidence_is_available_from_orchestrator(self):
        app = self._admin_app()
        label_audit = {
            "labels_seen": 9, "eligible_label_count": 0, "eligible_fraud_count": 0,
            "minimum_verified_labels": 100, "minimum_verified_fraud_labels": 10,
            "label_source_breakdown": {"synthetic_test": 9},
            "label_state_breakdown": {"verified": 9}, "exclusions": {"untrusted_source": 9},
            "trusted_source_authorization_configured": False,
        }
        demo = {"status": "candidate_trained_not_promoted", "model_version": "DEMO-X",
                "generated_at_utc": "2026-10-30T00:00:00Z", "_report_path": "models/candidates/demo/report.json",
                "candidate_evaluation_metrics": {"precision": .8, "recall": .7, "f1": .74, "threshold": .4, "rows": 25},
                "same_rows_v2_comparison": {"active_metrics": {"precision": .5, "recall": .4, "f1": .44}},
                "eligible_for_genuine_promotion": False, "promotion": {"status": "not_attempted"}}
        with patch.object(components, "cached_api_health", return_value=(False, None, "offline")), \
             patch.object(overview, "get_orchestrator_overview", return_value=_overview()), \
             patch("src.orchestrator.eligibility.evaluate_retraining_eligibility", return_value=_eligibility()), \
             patch("src.retraining.stage4_workflow.validate_feature_contract", return_value=["feature"]), \
             patch("src.retraining.stage4_workflow.collect_eligible_labels", return_value=([], label_audit)), \
             patch("services.data_service.load_latest_stage6_demo_report", return_value=(demo, None)):
            self._open_orchestrator(app)
        self.assertFalse(app.exception)
        rendered = " ".join(str(item.value) for item in app.markdown)
        self.assertIn("Saved demonstration candidate", rendered)
        self.assertIn("Activation and rollback inventory", " ".join(str(item.label) for item in app.expander))
        self.assertTrue(any("untrusted_source" in str(item.value) for item in app.json))
        self.assertTrue(any("DEMO-X" in str(item.value) for item in app.caption))

    def test_saved_drift_alert_only_appears_for_saved_detected_drift(self):
        app = self._admin_app()
        drift = {"status": "SAVED_REPORT", "report_exists": True, "drift_decision": True,
                 "drift_status": "drift", "report_timestamp": "2026-10-30T00:00:00Z",
                 "report_run_id": "saved-1", "reference_window": "reference-A",
                 "monitoring_window": "monitor-B", "drifted_features": ["amount"],
                 "feature_scores": [{"feature_name": "amount", "psi": 0.31}], "warnings": []}
        with patch.object(components, "cached_api_health", return_value=(False, None, "offline")), \
             patch.object(overview, "get_orchestrator_overview", return_value=_overview(drift=drift)), \
             patch("src.orchestrator.eligibility.evaluate_retraining_eligibility", return_value=_eligibility()):
            self._open_orchestrator(app)
        self.assertFalse(app.exception)
        errors = " ".join(str(item.value) for item in app.error)
        self.assertIn("Data drift detected", errors)
        self.assertIn("amount", " ".join(str(item.value) for item in app.markdown))

    def test_simulation_is_click_only_and_independent_of_blocked_eligibility(self):
        app = self._admin_app()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = {
                "drift_report": {"drift_detected": True, "features_drifted": 3, "features_tested": 4, "drift_rate": .75},
                "evaluation_report": {"metrics": {
                    "V2": {"precision": .4, "recall": .3, "f1": .34, "pr_auc": .2, "roc_auc": .6, "fp": 7, "fn": 8},
                    "candidate": {"precision": .8, "recall": .7, "f1": .75, "pr_auc": .9, "roc_auc": .95, "fp": 2, "fn": 3}}},
                "recommendation_report": {"recommendation": "CANDIDATE_NOT_RECOMMENDED", "reasons": ["Test reason from report."]},
                "final_summary": {"active_model": "V2", "v2_hashes_unchanged": {"model": True, "metadata": True, "threshold": True},
                                  "real_orchestrator_runs_unchanged": True, "automatic_promotion": "NO"},
            }
            paths = {}
            for key, value in reports.items():
                path = root / f"{key}.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                paths[key] = str(path)
            candidate = root / "candidate.json"
            candidate.write_bytes(b"candidate")
            comparison = root / "comparison.csv"
            comparison.write_text("model\nV2\n", encoding="utf-8")
            result = {"run_id": "sim-test", "artifact_paths": {**paths, "candidate_model": str(candidate), "comparison_csv": str(comparison)}}
            with patch.object(components, "cached_api_health", return_value=(False, None, "offline")), \
                 patch.object(overview, "get_orchestrator_overview", return_value=_overview()), \
                 patch("src.orchestrator.eligibility.evaluate_retraining_eligibility", return_value=_eligibility()), \
                 patch("src.retraining.drift_simulation_workflow.run_simulation", return_value=result) as run_simulation:
                self._open_orchestrator(app)
                self.assertFalse(app.exception)
                self.assertTrue(next(button for button in app.button if button.label == "Start candidate retraining").disabled)
                self.assertTrue(any(button.label == "Run Simulation Demonstration" and not button.disabled for button in app.button))
                run_simulation.assert_not_called()
                app.button(key="model_orchestrator_refresh").click().run()
                run_simulation.assert_not_called()
                app.button(key="model_orchestrator_run_simulation").click().run()
                run_simulation.assert_called_once_with()
                rendered = " ".join(str(item.value) for item in app.markdown)
                self.assertEqual(app.session_state.get("model_orchestrator_simulation_result"), result)
                captioned = " ".join(str(item.value) for item in app.caption)
                self.assertIn("Test reason from report.", captioned)
                self.assertIn("Candidate is not recommended", " ".join(str(item.value) for item in app.warning))
                self.assertTrue(any("0.8" in str(item.value) for item in app.dataframe))
                self.assertIn(str(candidate), " ".join(str(item.value) for item in app.code))
                reports["recommendation_report"]["recommendation"] = "CANDIDATE_RECOMMENDED_FOR_REVIEW"
                reports["recommendation_report"]["reasons"] = ["Positive report result."]
                Path(paths["recommendation_report"]).write_text(json.dumps(reports["recommendation_report"]), encoding="utf-8")
                app.button(key="model_orchestrator_refresh").click().run()
                self.assertTrue(any("CANDIDATE_RECOMMENDED_FOR_REVIEW" in str(item.value) for item in app.success))
                candidate.unlink()
                app.button(key="model_orchestrator_refresh").click().run()
                self.assertNotIn(str(candidate), " ".join(str(item.value) for item in app.code))
                app.button(key="model_orchestrator_refresh").click().run()
                run_simulation.assert_called_once_with()
                self.assertFalse(app.exception)

    def test_non_admin_page_does_not_show_simulation_button(self):
        app = AppTest.from_file(str(Path(__file__).parents[1] / "pages/12_Model_Orchestrator.py"), default_timeout=30)
        app.session_state["authenticated_role"] = "user"
        app.session_state["authenticated_username"] = "isolated-user"
        app.run()
        self.assertFalse(any(button.label == "Run Simulation Demonstration" for button in app.button))

    def test_non_admin_navigation_does_not_register_orchestrator(self):
        app = AppTest.from_file(str(Path(__file__).parents[1] / "dashboard.py"), default_timeout=30)
        app.session_state["authenticated_role"] = "user"
        app.session_state["authenticated_username"] = "isolated-user"
        with patch.object(components, "cached_api_health", return_value=(False, None, "offline")):
            app.run()
        self.assertFalse(app.exception)
        self.assertNotIn("Model Orchestrator", [item.label for item in app.page_link])

    def _run_fixture(self, root: Path):
        run_id = "20261030T000000Z_" + "f" * 32
        run_dir = root / run_id
        (run_dir / "evaluation").mkdir(parents=True)
        (run_dir / "run.json").write_text(json.dumps({
            "run_id": run_id, "created_at_utc": "2026-10-30T00:00:00+00:00",
            "workflow_status": "CANDIDATE_READY", "current_stage": "evaluation_completed",
        }), encoding="utf-8")
        (run_dir / "evaluation" / "evaluation_report.json").write_text(json.dumps({
            "run_id": run_id, "evaluation_status": "COMPLETED", "decision_status": "REVIEW_REQUIRED",
            "models": {"active": {"model_version": "V2"}, "candidate": {"model_version": "CANDIDATE-TEST"}},
            "metrics": {"active": {"precision": 0.8, "recall": 0.7, "f1": 0.75},
                        "candidate": {"precision": 0.9, "recall": 0.8, "f1": 0.84}},
            "comparison": {"f1_delta": 0.09}, "decision_reasons": ["Review required."],
        }), encoding="utf-8")
        return run_id

    def test_candidate_report_renders_and_approval_needs_reason_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            run_id = self._run_fixture(root)
            data = _overview(runs=[{"run_id": run_id, "created_at_utc": "2026-10-30T00:00:00+00:00",
                                    "workflow_status": "CANDIDATE_READY", "current_stage": "evaluation_completed"}])
            app = self._admin_app()
            review = {"review_token": "current-stage7-review", "evaluation_status": "REVIEW_REQUIRED"}
            with patch.object(config, "RUNS_ROOT", root), \
                 patch.object(components, "cached_api_health", return_value=(False, None, "offline")), \
                 patch.object(overview, "get_orchestrator_overview", return_value=data), \
                 patch("src.orchestrator.eligibility.evaluate_retraining_eligibility", return_value=_eligibility()), \
                 patch.object(approval, "get_admin_decision", return_value=None), \
                 patch.object(approval, "get_approval_review", return_value=review), \
                 patch.object(approval, "record_admin_decision", return_value={"decision": "APPROVED"}) as record:
                self._open_orchestrator(app)
                text = " ".join(str(item.value) for item in app.markdown)
                self.assertIn("CANDIDATE-TEST", text)
                app.button(key=f"model_orchestrator_review_{run_id}").click().run()
                form_key = f"model_orchestrator_decision_form_{run_id}"
                app.radio(key=f"model_orchestrator_decision_{run_id}").set_value("APPROVED")
                app.button(key=f"FormSubmitter:{form_key}-Record approval or rejection").click().run()
                record.assert_not_called()
                app.text_area(key=f"model_orchestrator_decision_reason_{run_id}").set_value("Approved after report review.")
                app.checkbox(key=f"model_orchestrator_decision_confirm_{run_id}").check()
                app.button(key=f"FormSubmitter:{form_key}-Record approval or rejection").click().run()
                record.assert_called_once()
                self.assertEqual(record.call_args.kwargs["review_token"], "current-stage7-review")
                self.assertNotIn("identity", record.call_args.kwargs)
            self.assertFalse(app.exception)

    def test_promotion_is_separate_and_requires_reason_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            run_id = self._run_fixture(root)
            data = _overview(runs=[{"run_id": run_id, "created_at_utc": "2026-10-30T00:00:00+00:00",
                                    "workflow_status": "CANDIDATE_READY", "current_stage": "evaluation_completed"}])
            app = self._admin_app()
            approved = {"decision": "APPROVED", "reason": "reviewed"}
            ready = {"promotion_ready": True, "promotion_token": "current-stage8-token",
                     "active_model_version": "V2", "candidate_version": "CANDIDATE-TEST"}
            with patch.object(config, "RUNS_ROOT", root), \
                 patch.object(components, "cached_api_health", return_value=(False, None, "offline")), \
                 patch.object(overview, "get_orchestrator_overview", return_value=data), \
                 patch("src.orchestrator.eligibility.evaluate_retraining_eligibility", return_value=_eligibility()), \
                 patch.object(approval, "get_admin_decision", return_value=approved), \
                 patch.object(promotion, "get_promotion_review", return_value=ready), \
                 patch.object(promotion, "promote_approved_candidate", return_value={"status": "PROMOTED"}) as promote:
                self._open_orchestrator(app)
                app.button(key=f"model_orchestrator_promotion_review_{run_id}").click().run()
                form_key = f"model_orchestrator_promotion_form_{run_id}"
                app.text_area(key=f"model_orchestrator_promotion_reason_{run_id}").set_value("Promote after separate review.")
                app.button(key=f"FormSubmitter:{form_key}-Promote approved candidate").click().run()
                promote.assert_not_called()
                app.checkbox(key=f"model_orchestrator_promotion_confirm_{run_id}").check()
                app.button(key=f"FormSubmitter:{form_key}-Promote approved candidate").click().run()
                promote.assert_called_once()
                self.assertEqual(promote.call_args.kwargs["review_token"], "current-stage8-token")
                self.assertEqual(promote.call_args.kwargs["promotion_reason"], "Promote after separate review.")
            self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
