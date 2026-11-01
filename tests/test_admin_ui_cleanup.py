"""Focused UI checks for consolidated navigation and independent health views."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from services import data_service
from src.orchestrator import drift_adapter
from ui import components


class AdminUICleanupTests(unittest.TestCase):
    def _admin_app(self) -> AppTest:
        app = AppTest.from_file(str(Path(__file__).parents[1] / "dashboard.py"), default_timeout=30)
        app.session_state["authenticated_role"] = "admin"
        app.session_state["authenticated_username"] = "cleanup-test-admin"
        return app

    def _open_page(self, app: AppTest, page: str) -> None:
        with patch.object(components, "cached_api_health", return_value=(False, None, "offline")), \
             patch.object(data_service, "prediction_registry_summary", return_value={"available": False}), \
             patch.object(data_service, "recent_stage1_predictions", return_value=[]), \
             patch.object(data_service, "load_active_model_snapshot", return_value={"available": False, "error": "unavailable"}), \
             patch.object(drift_adapter, "get_latest_drift_summary", return_value={"status": "NO_SAVED_REPORT", "warnings": []}):
            app.run()
            app.switch_page(page).run()

    def test_pipeline_evidence_focuses_on_activity_not_label_or_drift_details(self):
        app = self._admin_app()
        with patch.object(data_service, "read_stage1_status", return_value=None), \
             patch.object(data_service, "recent_stage1_predictions", return_value=[]), \
             patch.object(data_service, "load_stage3_monitor_report", return_value=None), \
             patch.object(data_service, "read_project_jsonl", return_value=[]), \
             patch.object(data_service, "load_latest_phase12_evidence", return_value=None):
            self._open_page(app, "pages/2_Live_Monitoring.py")
        self.assertFalse(app.exception)
        text = " ".join(str(item.value) for item in app.markdown)
        self.assertIn("Recent transaction activity", text)
        self.assertIn("Streaming process status", text)
        self.assertNotIn("Pending labels", text)
        self.assertNotIn("Per-feature detector output", text)

    def test_system_health_reports_independent_unavailable_and_unchecked_signals(self):
        app = self._admin_app()
        with patch.object(data_service, "read_stage1_status", return_value=None), \
             patch.object(data_service, "recent_stage1_predictions", return_value=None), \
             patch.object(data_service, "load_latest_phase12_evidence", return_value=None):
            self._open_page(app, "pages/4_System_Health.py")
        self.assertFalse(app.exception)
        text = " ".join(str(item.value) for item in list(app.markdown) + list(app.caption))
        for section in ("API health", "Model resolver", "Artifact availability", "Streaming health"):
            self.assertIn(section, text)
        self.assertIn("Kafka broker: Not checked", text)
        self.assertIn("Spark query: Not directly checked", text)
        self.assertNotIn("All systems healthy", text)


if __name__ == "__main__":
    unittest.main()
