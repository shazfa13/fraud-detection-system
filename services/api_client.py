"""Small timeout-bounded client for the project's existing FastAPI routes."""
from __future__ import annotations

import math
import os
from typing import Any, Mapping

import requests

PREDICTION_FIELDS = (
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "nameDest", "oldbalanceDest",
)
FORBIDDEN_FIELDS = frozenset({"isFraud", "fraud_probability", "predicted_label", "risk_score", "risk_level"})
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 3.0


class APIClientError(RuntimeError):
    """A safe, user-displayable API request or response error."""


def api_base_url() -> str:
    """Return the configured FastAPI base URL (environment override supported)."""
    return os.environ.get("FRAUD_API_BASE_URL", DEFAULT_API_BASE_URL).strip().rstrip("/")


def _request_json(method: str, path: str, *, payload: Mapping[str, Any] | None = None,
                  base_url: str | None = None, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Perform a bounded request and normalize network/HTTP/JSON failures."""
    url = f"{(base_url or api_base_url()).rstrip('/')}{path}"
    try:
        response = requests.request(method, url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise APIClientError(f"Could not reach the FastAPI service at {url}: {exc}") from exc
    try:
        body = response.json()
    except ValueError as exc:
        raise APIClientError(f"The API returned a non-JSON response (HTTP {response.status_code}).") from exc
    if not response.ok:
        detail = body.get("detail", "Request rejected") if isinstance(body, dict) else "Request rejected"
        raise APIClientError(f"API returned HTTP {response.status_code}: {detail}")
    if not isinstance(body, dict):
        raise APIClientError("The API response must be a JSON object.")
    return body


def get_health(*, base_url: str | None = None, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Fetch the existing live `/health` response and check its basic schema."""
    body = _request_json("GET", "/health", base_url=base_url, timeout=timeout)
    if not isinstance(body.get("status"), str) or not isinstance(body.get("model_loaded"), bool):
        raise APIClientError("The API /health response has an unexpected schema.")
    if body.get("feature_count") != 34:
        raise APIClientError("The API does not report the expected 34-feature contract.")
    threshold = body.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        raise APIClientError("The API /health threshold is missing or invalid.")
    return body


def _validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Enforce the exact seven required transaction fields used by this UI."""
    if not isinstance(payload, Mapping):
        raise APIClientError("Prediction input must be a JSON object.")
    forbidden = FORBIDDEN_FIELDS.intersection(payload)
    if forbidden:
        raise APIClientError(f"Prediction input cannot contain target or audit fields: {sorted(forbidden)}")
    unknown = set(payload) - set(PREDICTION_FIELDS)
    missing = set(PREDICTION_FIELDS) - set(payload)
    if unknown or missing:
        raise APIClientError(f"Invalid prediction fields; missing={sorted(missing)}, unexpected={sorted(unknown)}")
    values = dict(payload)
    for field in ("step",):
        if isinstance(values[field], bool) or not isinstance(values[field], int) or values[field] < 1:
            raise APIClientError("Step must be a whole number greater than or equal to 1.")
    for field in ("amount", "oldbalanceOrg", "oldbalanceDest"):
        value = values[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise APIClientError(f"{field} must be a finite number greater than or equal to 0.")
    for field in ("type", "nameOrig", "nameDest"):
        if not isinstance(values[field], str) or not values[field].strip():
            raise APIClientError(f"{field} must be a non-empty string.")
    return values


def predict_transaction(payload: Mapping[str, Any], *, base_url: str | None = None,
                        timeout: float = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Submit only transaction-time request fields to the existing `/predict` route."""
    body = _request_json("POST", "/predict", payload=_validate_payload(payload),
                         base_url=base_url, timeout=timeout)
    required = {"prediction", "fraud_probability", "risk_score", "risk_level", "model_version"}
    if not required.issubset(body):
        raise APIClientError("The API prediction response is missing required fields.")
    try:
        probability = float(body["fraud_probability"])
        score = float(body["risk_score"])
        prediction = int(body["prediction"])
    except (TypeError, ValueError) as exc:
        raise APIClientError("The API prediction response contains invalid numeric values.") from exc
    if prediction not in (0, 1) or not 0.0 <= probability <= 1.0 or not 0.0 <= score <= 100.0:
        raise APIClientError("The API prediction response is outside the expected ranges.")
    return body


def preview_features(payload: Mapping[str, Any], *, base_url: str | None = None,
                     timeout: float = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Call the existing non-consuming `/state/preview` endpoint."""
    body = _request_json("POST", "/state/preview", payload=_validate_payload(payload),
                         base_url=base_url, timeout=timeout)
    names, features = body.get("feature_names"), body.get("features")
    if not isinstance(names, list) or len(names) != 34 or not isinstance(features, dict):
        raise APIClientError("The API preview response does not match the expected 34-feature schema.")
    if any(key in names for key in FORBIDDEN_FIELDS | {"event_id", "step"}):
        raise APIClientError("API preview included an audit field or target in the model feature contract.")
    if list(features) != names:
        raise APIClientError("API preview feature values do not match the reported feature order.")
    return body
