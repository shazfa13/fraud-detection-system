"""Session auth, one configured administrator, and SQLite user accounts.

Canonical password hash format (PowerShell-safe):
    pbkdf2_sha256:<iterations>:<32 lowercase salt hex chars>:<64 lowercase digest hex chars>

The previous dollar-delimited format is parsed explicitly for existing Admin hashes. New hashes
are always serialized in the canonical colon-delimited format.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st


ROLES = ("user", "admin")
_MIN_ITERATIONS = 100_000
_MAX_ITERATIONS = 2_000_000
_CANONICAL_HASH = re.compile(r"^pbkdf2_sha256:([0-9]{5,7}):([0-9a-f]{32}):([0-9a-f]{64})$")
_LEGACY_HASH = re.compile(r"^pbkdf2_sha256\$([0-9]{5,7})\$([0-9a-f]{32})\$([0-9a-f]{64})$")
_USERNAME = re.compile(r"^[A-Za-z0-9_.-]{3,64}$")
_PASSWORD_MIN_LENGTH = 12
_PASSWORD_MAX_LENGTH = 1024
USER_DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "user_accounts.sqlite3"


def _admin_setting(name: str) -> str:
    """Read an Admin setting from the process environment or Streamlit secrets."""
    value = os.environ.get(f"FRAUD_ADMIN_{name}", "").strip()
    if value:
        return value
    try:
        return str(st.secrets.get(f"FRAUD_ADMIN_{name}", "")).strip()
    except Exception:
        # Streamlit raises when no secrets file/configuration is available.
        return ""


class RegistrationError(ValueError):
    """Safe validation or duplicate-account error suitable for the UI."""


class AuthStorageError(RuntimeError):
    """User database unavailable or unreadable; contains no credential material."""


def password_hash(password: str, *, salt: bytes | None = None, iterations: int = 310_000) -> str:
    """Return a salted PBKDF2-SHA256 hash in the canonical, PowerShell-safe format."""
    if not isinstance(password, str):
        raise TypeError("Password must be text.")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not _MIN_ITERATIONS <= iterations <= _MAX_ITERATIONS:
        raise ValueError(f"PBKDF2 iterations must be between {_MIN_ITERATIONS} and {_MAX_ITERATIONS}.")
    if salt is not None and (not isinstance(salt, bytes) or len(salt) != 16):
        raise ValueError("PBKDF2 salt must be exactly 16 bytes.")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256:{iterations}:{salt.hex()}:{digest.hex()}"


def _decode_password_hash(encoded: str) -> tuple[int, bytes, str, str] | None:
    """Parse a canonical or explicitly supported legacy hash with bounded work factors."""
    if not isinstance(encoded, str):
        return None
    match = _CANONICAL_HASH.fullmatch(encoded)
    hash_format = "canonical"
    if match is None:
        match = _LEGACY_HASH.fullmatch(encoded)
        hash_format = "legacy"
    if match is None:
        return None
    iterations_text, salt_hex, expected = match.groups()
    iterations = int(iterations_text)
    if not _MIN_ITERATIONS <= iterations <= _MAX_ITERATIONS:
        return None
    return iterations, bytes.fromhex(salt_hex), expected, hash_format


def validate_password_hash(encoded: str) -> bool:
    """Whether a hash matches one of the supported bounded formats."""
    return _decode_password_hash(encoded) is not None


def _verify(password: str, encoded: str) -> bool:
    parsed = _decode_password_hash(encoded)
    if parsed is None or not isinstance(password, str):
        return False
    iterations, salt, expected, _ = parsed
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual.hex(), expected)


def user_database_path() -> Path:
    """Stable local user-account database; an environment override supports isolated deployments/tests."""
    configured = os.environ.get("FRAUD_USER_DB_PATH", "").strip()
    return Path(configured).expanduser() if configured else USER_DATABASE_PATH


def _initialize_user_db(connection: sqlite3.Connection) -> None:
    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT NOT NULL,
            username_key TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at_utc TEXT NOT NULL
        )
    """)
    connection.commit()


def _connect_user_db(path: Path, *, create: bool) -> sqlite3.Connection | None:
    if not create and not path.is_file():
        return None
    try:
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(path), timeout=5.0)
        connection.row_factory = sqlite3.Row
        if create:
            _initialize_user_db(connection)
        return connection
    except sqlite3.Error as exc:
        raise AuthStorageError("The user account database is unavailable.") from exc


def register_user(username: str, password: str, confirmation: str, *, db_path: str | Path | None = None) -> str:
    """Create a normal user account; role is intentionally absent from inputs and schema."""
    if not isinstance(username, str) or not username.strip():
        raise RegistrationError("Enter a username.")
    username = username.strip()
    if not _USERNAME.fullmatch(username):
        raise RegistrationError("Use 3–64 letters, numbers, dots, underscores, or hyphens for the username.")
    if not isinstance(password, str) or not _PASSWORD_MIN_LENGTH <= len(password) <= _PASSWORD_MAX_LENGTH:
        raise RegistrationError(f"Choose a password between {_PASSWORD_MIN_LENGTH} and {_PASSWORD_MAX_LENGTH} characters.")
    if password != confirmation:
        raise RegistrationError("The password and confirmation do not match.")

    path = Path(db_path) if db_path is not None else user_database_path()
    encoded = password_hash(password)
    try:
        connection = _connect_user_db(path, create=True)
        assert connection is not None
        with connection:
            connection.execute(
                "INSERT INTO users (username, username_key, password_hash, created_at_utc) VALUES (?, ?, ?, ?)",
                (username, username.casefold(), encoded, datetime.now(timezone.utc).isoformat()),
            )
        return username
    except sqlite3.IntegrityError as exc:
        raise RegistrationError("That username is already registered. Try signing in instead.") from exc
    except sqlite3.Error as exc:
        raise AuthStorageError("The user account could not be saved to the local database.") from exc
    finally:
        if "connection" in locals() and connection is not None:
            connection.close()


def _authenticate_admin(username: str, password: str) -> bool:
    expected_user = _admin_setting("USERNAME")
    encoded = _admin_setting("PASSWORD_HASH")
    if not expected_user or not validate_password_hash(encoded):
        return False
    user_ok = hmac.compare_digest(username.strip().encode("utf-8"), expected_user.encode("utf-8"))
    password_ok = _verify(password, encoded)
    return user_ok and password_ok


def _authenticate_user(username: str, password: str) -> bool:
    path = user_database_path()
    try:
        connection = _connect_user_db(path, create=False)
        if connection is None:
            return False
        try:
            row = connection.execute(
                "SELECT username, password_hash FROM users WHERE username_key = ?",
                (username.strip().casefold(),),
            ).fetchone()
        finally:
            connection.close()
    except (AuthStorageError, sqlite3.Error):
        return False
    return bool(row and _verify(password, row["password_hash"]))


def credential_configuration_error(role: str) -> str | None:
    """Safe role setup diagnostic; normal User accounts are managed in SQLite."""
    if role == "user":
        return None
    if role != "admin":
        return "Unknown login role."
    if not _admin_setting("USERNAME"):
        return "FRAUD_ADMIN_USERNAME is missing from Streamlit secrets and the process environment."
    encoded = _admin_setting("PASSWORD_HASH")
    if not encoded:
        return "FRAUD_ADMIN_PASSWORD_HASH is missing from Streamlit secrets and the process environment."
    if not validate_password_hash(encoded):
        return ("FRAUD_ADMIN_PASSWORD_HASH is not a supported PBKDF2-SHA256 hash. "
                "Generate a new canonical hash and assign it using the documented PowerShell command.")
    return None


def credentials_configured(role: str) -> bool:
    return credential_configuration_error(role) is None


def authenticate(role: str, username: str, password: str) -> bool:
    """Authenticate against the single configured Admin or a registered User account."""
    if role not in ROLES or not isinstance(username, str) or not isinstance(password, str):
        return False
    if role == "admin":
        return _authenticate_admin(username, password)
    return _authenticate_user(username, password)


def environment_diagnostic() -> dict[str, bool | str]:
    """Secret-safe configuration report for `python -m ui.auth_diagnostics`."""
    result: dict[str, bool | str] = {}
    for role, prefix in (("admin", "FRAUD_ADMIN"), ("user", "FRAUD_USER")):
        if role == "admin":
            username = _admin_setting("USERNAME")
            encoded = _admin_setting("PASSWORD_HASH")
        else:
            username = os.environ.get(f"{prefix}_USERNAME", "")
            encoded = os.environ.get(f"{prefix}_PASSWORD_HASH", "")
        parsed = _decode_password_hash(encoded.strip())
        result[f"{role}_username_present"] = bool(username.strip())
        result[f"{role}_hash_present"] = bool(encoded.strip())
        result[f"{role}_hash_valid"] = parsed is not None
        result[f"{role}_hash_canonical_valid"] = bool(parsed and parsed[3] == "canonical")
        result[f"{role}_hash_format"] = parsed[3] if parsed else "invalid or absent"
    result["user_environment_credentials_used"] = False
    result["user_database_exists"] = user_database_path().is_file()
    return result


def login(role: str, username: str) -> None:
    st.session_state["authenticated_role"] = role
    st.session_state["authenticated_username"] = username.strip()
    st.session_state.pop("login_role", None)


def logout() -> None:
    for key in list(st.session_state):
        if key.startswith(("user_", "analyzer_", "counterfactual_")) or key in {
            "authenticated_role", "authenticated_username", "login_role", "signup_requested"
        }:
            st.session_state.pop(key, None)


def require_role(required_role: str) -> bool:
    if current_role() == required_role:
        return True
    st.error("Please sign in with the required account to access this page.")
    st.switch_page("pages/0_Welcome.py")
    return False


def current_role() -> str | None:
    role = st.session_state.get("authenticated_role")
    return role if role in ROLES else None
