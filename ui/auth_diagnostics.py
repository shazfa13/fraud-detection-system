"""Print secret-safe credential status and run isolated User/Admin auth self-tests."""
from __future__ import annotations

import os
import secrets
import tempfile
from pathlib import Path
from unittest.mock import patch

from ui import auth


def run_isolated_self_test() -> dict[str, bool]:
    """Exercise canonical round-trip and both role paths using only temporary credentials."""
    temporary_password = secrets.token_urlsafe(24)
    encoded = auth.password_hash(temporary_password)
    results = {
        "generated_hash_valid": auth.validate_password_hash(encoded),
        "generated_hash_is_powershell_safe": "$" not in encoded,
        "correct_password_accepted": auth._verify(temporary_password, encoded),
        "incorrect_password_rejected": not auth._verify(temporary_password + "wrong", encoded),
        "malformed_hash_rejected": not auth.validate_password_hash("not-a-password-hash"),
        "generated_hash_is_valid_after_environment_assignment": False,
    }
    with tempfile.TemporaryDirectory(prefix="fraud-auth-check-") as temporary:
        user_db = Path(temporary) / "users.sqlite3"
        env_name = "FRAUD_AUTH_SELF_TEST_HASH"
        previous_hash = os.environ.get(env_name)
        os.environ[env_name] = encoded
        results["generated_hash_is_valid_after_environment_assignment"] = auth.validate_password_hash(os.environ[env_name])
        if previous_hash is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous_hash
        with patch.dict(os.environ, {
            "FRAUD_ADMIN_USERNAME": "temporary-admin",
            "FRAUD_ADMIN_PASSWORD_HASH": encoded,
            "FRAUD_USER_DB_PATH": str(user_db),
        }, clear=False):
            results["admin_role_authenticates"] = auth.authenticate("admin", "temporary-admin", temporary_password)
            results["admin_role_rejects_wrong_password"] = not auth.authenticate("admin", "temporary-admin", temporary_password + "wrong")
            auth.register_user("temporary-user", temporary_password, temporary_password, db_path=user_db)
            results["user_role_authenticates"] = auth.authenticate("user", "temporary-user", temporary_password)
            results["user_role_rejects_wrong_password"] = not auth.authenticate("user", "temporary-user", temporary_password + "wrong")
            results["admin_account_is_not_a_user"] = not auth.authenticate("user", "temporary-admin", temporary_password)
            results["user_account_is_not_an_admin"] = not auth.authenticate("admin", "temporary-user", temporary_password)
    return results


def main() -> None:
    print("Credential configuration (values are never displayed):")
    for name, value in auth.environment_diagnostic().items():
        print(f"  {name}: {value}")
    print("Isolated temporary self-test:")
    for name, passed in run_isolated_self_test().items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
