"""Credential serialization, SQLite user accounts, and Streamlit login-flow tests."""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from ui import auth
from ui.auth_diagnostics import run_isolated_self_test


ADMIN_PASSWORD = "temporary-admin-test-password"
USER_PASSWORD = "temporary-user-test-password"


class UIAuthTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary.name) / "users.sqlite3"
        self.admin_hash = auth.password_hash(ADMIN_PASSWORD, salt=b"0123456789abcdef")
        self.environment = {
            "FRAUD_ADMIN_USERNAME": "administrator",
            "FRAUD_ADMIN_PASSWORD_HASH": self.admin_hash,
            "FRAUD_USER_DB_PATH": str(self.db_path),
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_canonical_generator_parser_and_verifier_round_trip(self):
        encoded = auth.password_hash("temporary password", salt=b"0123456789abcdef")
        self.assertEqual(encoded.count(":"), 3)
        self.assertNotIn("$", encoded)
        self.assertTrue(auth.validate_password_hash(encoded))
        self.assertTrue(auth._verify("temporary password", encoded))
        self.assertFalse(auth._verify("incorrect password", encoded))

    def test_explicit_legacy_hash_is_accepted_for_existing_admin_configuration(self):
        old_format = self.admin_hash.replace(":", "$")
        with patch.dict(os.environ, {**self.environment, "FRAUD_ADMIN_PASSWORD_HASH": old_format}, clear=False):
            self.assertTrue(auth.validate_password_hash(old_format))
            self.assertTrue(auth.authenticate("admin", "administrator", ADMIN_PASSWORD))

    def test_malformed_and_unbounded_hashes_fail_closed(self):
        for malformed in (
            "not-a-hash",
            "pbkdf2_sha256:99999:" + "0" * 32 + ":" + "0" * 64,
            "pbkdf2_sha256:99999999:" + "0" * 32 + ":" + "0" * 64,
            "pbkdf2_sha256:310000:bad:bad",
        ):
            self.assertFalse(auth.validate_password_hash(malformed))
            self.assertFalse(auth._verify(ADMIN_PASSWORD, malformed))
        with patch.dict(os.environ, {**self.environment, "FRAUD_ADMIN_PASSWORD_HASH": "broken"}, clear=False):
            self.assertFalse(auth.authenticate("admin", "administrator", ADMIN_PASSWORD))
            self.assertIn("not a supported", auth.credential_configuration_error("admin"))

    def test_admin_environment_login_and_wrong_password(self):
        with patch.dict(os.environ, self.environment, clear=False):
            self.assertTrue(auth.authenticate("admin", "administrator", ADMIN_PASSWORD))
            self.assertFalse(auth.authenticate("admin", "administrator", "wrong password"))
            self.assertFalse(auth.authenticate("admin", "Administrator", ADMIN_PASSWORD))

    def test_missing_admin_config_fails_closed_with_safe_status(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(auth.credentials_configured("admin"))
            self.assertIn("FRAUD_ADMIN_USERNAME", auth.credential_configuration_error("admin"))
            self.assertFalse(auth.authenticate("admin", "administrator", ADMIN_PASSWORD))
        with patch.dict(os.environ, {"FRAUD_ADMIN_USERNAME": "administrator"}, clear=True):
            self.assertIn("FRAUD_ADMIN_PASSWORD_HASH", auth.credential_configuration_error("admin"))

    def test_registration_success_hash_storage_and_password_login(self):
        username = auth.register_user("Analyst_1", USER_PASSWORD, USER_PASSWORD, db_path=self.db_path)
        self.assertEqual(username, "Analyst_1")
        with patch.dict(os.environ, self.environment, clear=False):
            self.assertTrue(auth.authenticate("user", "analyst_1", USER_PASSWORD))
            self.assertFalse(auth.authenticate("user", "Analyst_1", "wrong password"))
            self.assertFalse(auth.authenticate("admin", "Analyst_1", USER_PASSWORD))
        with closing(sqlite3.connect(self.db_path)) as connection:
            row = connection.execute("SELECT username, username_key, password_hash FROM users").fetchone()
            columns = {item[1] for item in connection.execute("PRAGMA table_info(users)")}
        self.assertEqual(row[0:2], ("Analyst_1", "analyst_1"))
        self.assertNotEqual(row[2], USER_PASSWORD)
        self.assertTrue(auth.validate_password_hash(row[2]))
        self.assertNotIn("role", columns)

    def test_duplicate_registration_is_case_insensitive(self):
        auth.register_user("Analyst", USER_PASSWORD, USER_PASSWORD, db_path=self.db_path)
        with self.assertRaisesRegex(auth.RegistrationError, "already registered"):
            auth.register_user("analyst", USER_PASSWORD, USER_PASSWORD, db_path=self.db_path)

    def test_confirmation_mismatch_and_invalid_inputs_are_rejected(self):
        with self.assertRaisesRegex(auth.RegistrationError, "do not match"):
            auth.register_user("Analyst", USER_PASSWORD, USER_PASSWORD + "x", db_path=self.db_path)
        with self.assertRaises(auth.RegistrationError):
            auth.register_user("a", USER_PASSWORD, USER_PASSWORD, db_path=self.db_path)
        with self.assertRaises(auth.RegistrationError):
            auth.register_user("Analyst", "short", "short", db_path=self.db_path)
        self.assertFalse(self.db_path.exists())

    def test_user_persists_across_fresh_database_connections(self):
        auth.register_user("PersistentUser", USER_PASSWORD, USER_PASSWORD, db_path=self.db_path)
        with patch.dict(os.environ, self.environment, clear=False):
            self.assertTrue(auth.authenticate("user", "persistentuser", USER_PASSWORD))
        self.assertTrue(self.db_path.is_file())

    def test_registration_api_has_no_role_parameter_for_admin_escalation(self):
        with self.assertRaises(TypeError):
            auth.register_user("NormalUser", USER_PASSWORD, USER_PASSWORD, role="admin", db_path=self.db_path)

    def test_login_and_logout_session_state(self):
        state = {}
        with patch.object(auth.st, "session_state", state):
            auth.login("user", "Analyst")
            self.assertEqual(auth.current_role(), "user")
            auth.logout()
            self.assertIsNone(auth.current_role())
        self.assertNotIn("authenticated_username", state)

    def test_role_guard_blocks_user_from_admin_pages(self):
        with patch.object(auth.st, "session_state", {"authenticated_role": "user"}), \
             patch.object(auth.st, "error") as error, patch.object(auth.st, "switch_page") as switch_page:
            self.assertFalse(auth.require_role("admin"))
            error.assert_called_once()
            switch_page.assert_called_once_with("pages/0_Welcome.py")

    def test_environment_diagnostic_has_only_presence_and_format_flags(self):
        environment = {
            **self.environment,
            "FRAUD_USER_USERNAME": "obsolete-user-variable",
            "FRAUD_USER_PASSWORD_HASH": "not-used-for-registration",
        }
        with patch.dict(os.environ, environment, clear=False):
            status = auth.environment_diagnostic()
        self.assertTrue(status["admin_username_present"])
        self.assertTrue(status["admin_hash_valid"])
        self.assertTrue(status["user_username_present"])
        self.assertFalse(status["user_hash_valid"])
        self.assertNotIn("administrator", str(status))
        self.assertNotIn(self.admin_hash, str(status))

    def test_isolated_self_test_passes_without_using_real_credentials(self):
        results = run_isolated_self_test()
        self.assertTrue(results)
        self.assertTrue(all(results.values()), results)

    def test_streamlit_forms_route_signup_and_both_login_roles(self):
        dashboard = Path(__file__).parents[1] / "dashboard.py"
        with patch.dict(os.environ, self.environment, clear=False):
            # Registration route creates a DB user and sends them to the User Login form.
            app = AppTest.from_file(str(dashboard), default_timeout=30).run()
            app.button(key="welcome_signup").click().run()
            app.text_input[0].set_value("FormUser")
            app.text_input[1].set_value(USER_PASSWORD)
            app.text_input[2].set_value(USER_PASSWORD)
            app.button[0].click().run()
            self.assertTrue(any("account created" in element.value.lower() for element in app.success))
            app.text_input[0].set_value("FormUser")
            app.text_input[1].set_value(USER_PASSWORD)
            app.button[0].click().run()
            self.assertEqual(app.session_state.get("authenticated_role"), "user")
            self.assertIn("Analyze Transaction", [button.label for button in app.button])

            app_admin = AppTest.from_file(str(dashboard), default_timeout=30).run()
            app_admin.button(key="welcome_admin_login").click().run()
            app_admin.text_input[0].set_value("administrator")
            app_admin.text_input[1].set_value(ADMIN_PASSWORD)
            app_admin.button[0].click().run()
            self.assertEqual(app_admin.session_state.get("authenticated_role"), "admin")
        self.assertIn("Admin Overview", [link.label for link in app_admin.get("page_link")])


if __name__ == "__main__":
    unittest.main()
