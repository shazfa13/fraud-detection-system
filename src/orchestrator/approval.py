"""Read-only candidate review and create-only administrator decision records.

An approval is an auditable decision only. This module never promotes, activates,
or reloads a candidate model.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.orchestrator.config import (
    RUNS_ROOT,
    OrchestratorPathError,
    resolve_run_dir,
    resolve_run_path,
    validate_run_id,
)
from src.retraining.cycle_pipeline import FINAL_FEATURES
from src.retraining.model_activator import validate_bundle


DECISION_SCHEMA_VERSION = "orchestrator-admin-decision-v1"
APPROVAL_FILENAME = "decision.json"
_ALLOWED_DECISIONS = frozenset({"APPROVED", "REJECTED"})
_ALLOWED_EVALUATION_STATUSES = frozenset({"CANDIDATE_ACCEPTED", "CANDIDATE_REJECTED", "REVIEW_REQUIRED"})


class ApprovalError(ValueError):
    """Raised when a review or decision cannot safely be completed."""


class ApprovalAuthorizationError(ApprovalError):
    """Raised when the current Streamlit authentication context is not an admin."""


class ApprovalReviewError(ApprovalError):
    """Raised when candidate/evaluation evidence is missing or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ApprovalReviewError(f"{label} is missing or invalid: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ApprovalReviewError(f"{label} must be a JSON object: {path}")
    return value


def _run_directory(run_id: str | None, run_root: str | Path) -> tuple[str, Path]:
    if run_id is None:
        raise ApprovalReviewError("A valid orchestrator run_id is required.")
    try:
        safe_id = validate_run_id(run_id)
        run_dir = resolve_run_dir(safe_id, run_root)
    except (OrchestratorPathError, TypeError) as exc:
        raise ApprovalReviewError(f"Invalid run ID or run root: {exc}") from exc
    lexical_run_dir = Path(run_root).expanduser().resolve() / safe_id
    if lexical_run_dir.is_symlink() or not run_dir.is_dir():
        raise ApprovalReviewError(f"Orchestrator run does not exist or is not a safe directory: {safe_id}")
    run_record_path = resolve_run_path(safe_id, "run.json", run_root=run_root)
    run_record = _read_json(run_record_path, "Run record")
    if run_record.get("run_id") != safe_id:
        raise ApprovalReviewError("Run record identity does not match the requested run.")
    return safe_id, run_dir


def _decision_path(run_id: str, run_root: str | Path) -> Path:
    return resolve_run_path(run_id, "approval", APPROVAL_FILENAME, run_root=run_root)


def _read_decision_file(run_id: str, run_root: str | Path) -> dict[str, Any] | None:
    run_dir = resolve_run_dir(run_id, run_root)
    if (run_dir / "approval").is_symlink():
        raise ApprovalError("Approval path is a symlink; refusing redirected audit reads.")
    path = _decision_path(run_id, run_root)
    if not path.exists():
        return None
    decision = _read_json(path, "Administrator decision")
    if (decision.get("schema_version") != DECISION_SCHEMA_VERSION
            or decision.get("run_id") != run_id
            or decision.get("decision") not in _ALLOWED_DECISIONS):
        raise ApprovalError("Existing administrator decision has an unsupported schema or identity.")
    return decision


def get_admin_decision(run_id: str, *, run_root: str | Path = RUNS_ROOT) -> dict[str, Any] | None:
    """Return the existing immutable decision, or ``None`` if no decision exists."""
    safe_id, _ = _run_directory(run_id, run_root)
    return _read_decision_file(safe_id, run_root)


def get_approval_review(run_id: str, *, run_root: str | Path = RUNS_ROOT) -> dict[str, Any]:
    """Validate Stage 4–6 evidence and return a read-only, hash-bound review summary."""
    safe_id, run_dir = _run_directory(run_id, run_root)
    if any((run_dir / folder).is_symlink() for folder in ("dataset", "candidate", "evaluation", "approval")):
        raise ApprovalReviewError("A run artifact directory is a symlink; refusing to review redirected files.")
    paths = {
        "dataset_metadata": resolve_run_path(safe_id, "dataset", "dataset_metadata.json", run_root=run_root),
        "dataset_validation": resolve_run_path(safe_id, "dataset", "dataset_validation.json", run_root=run_root),
        "dataset_parquet": resolve_run_path(safe_id, "dataset", "candidate_dataset.parquet", run_root=run_root),
        "candidate_model": resolve_run_path(safe_id, "candidate", "model.json", run_root=run_root),
        "candidate_metadata": resolve_run_path(safe_id, "candidate", "metadata.json", run_root=run_root),
        "candidate_threshold": resolve_run_path(safe_id, "candidate", "threshold.json", run_root=run_root),
        "candidate_training_report": resolve_run_path(safe_id, "candidate", "training_report.json", run_root=run_root),
        "evaluation_report": resolve_run_path(safe_id, "evaluation", "evaluation_report.json", run_root=run_root),
        "evaluation_validation": resolve_run_path(safe_id, "evaluation", "evaluation_validation.json", run_root=run_root),
        "model_comparison": resolve_run_path(safe_id, "evaluation", "model_comparison.csv", run_root=run_root),
        "run_record": resolve_run_path(safe_id, "run.json", run_root=run_root),
    }
    if any(not path.is_file() or path.is_symlink() for path in paths.values()):
        missing = [name for name, path in paths.items() if not path.is_file() or path.is_symlink()]
        raise ApprovalReviewError(f"Required Stage 4–6 review artifacts are missing or unsafe: {', '.join(missing)}")
    for name in ("dataset_metadata", "dataset_validation", "dataset_parquet", "candidate_model",
                 "candidate_metadata", "candidate_threshold", "candidate_training_report",
                 "evaluation_report", "evaluation_validation", "model_comparison"):
        if paths[name].parent.is_symlink():
            raise ApprovalReviewError(f"Review artifact directory is a symlink: {paths[name].parent}")

    dataset_meta = _read_json(paths["dataset_metadata"], "Stage 4 dataset metadata")
    dataset_validation = _read_json(paths["dataset_validation"], "Stage 4 dataset validation")
    candidate_meta = _read_json(paths["candidate_metadata"], "Candidate metadata")
    candidate_threshold = _read_json(paths["candidate_threshold"], "Candidate threshold")
    training_report = _read_json(paths["candidate_training_report"], "Candidate training report")
    evaluation_report = _read_json(paths["evaluation_report"], "Stage 6 evaluation report")
    evaluation_validation = _read_json(paths["evaluation_validation"], "Stage 6 evaluation validation")
    run_record = _read_json(paths["run_record"], "Run record")
    if run_record.get("run_id") != safe_id:
        raise ApprovalReviewError("Run record identity is inconsistent.")
    if dataset_meta.get("run_id") != safe_id or dataset_validation.get("run_id") != safe_id:
        raise ApprovalReviewError("Stage 4 artifacts do not belong to the requested run.")
    if dataset_validation.get("passed") is not True:
        raise ApprovalReviewError("Stage 4 dataset validation did not pass.")
    if training_report.get("run_id") != safe_id or training_report.get("status") != "CANDIDATE_TRAINED_NOT_PROMOTED":
        raise ApprovalReviewError("Stage 5 report does not confirm this run's completed, unpromoted candidate.")
    if evaluation_report.get("run_id") != safe_id:
        raise ApprovalReviewError("Stage 6 evaluation report does not belong to the requested run.")
    if evaluation_report.get("evaluation_status") != "COMPLETED":
        raise ApprovalReviewError("Stage 6 evaluation did not complete successfully.")
    evaluation_status = evaluation_report.get("decision_status")
    if evaluation_status not in _ALLOWED_EVALUATION_STATUSES:
        raise ApprovalReviewError(f"Unsupported Stage 6 recommendation: {evaluation_status!r}.")
    if evaluation_validation.get("run_id") != safe_id or evaluation_validation.get("passed") is not True:
        raise ApprovalReviewError("Stage 6 evaluation validation is missing, failed, or belongs to another run.")

    try:
        dataset_frame = pd.read_parquet(paths["dataset_parquet"])
    except Exception as exc:
        raise ApprovalReviewError(f"Stage 4 candidate dataset cannot be re-read: {type(exc).__name__}: {exc}") from exc
    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(dataset_frame, index=False).to_numpy().tobytes()
    ).hexdigest()
    expected_fingerprint = dataset_meta.get("dataset_fingerprint_sha256")
    if (not isinstance(expected_fingerprint, str) or fingerprint != expected_fingerprint
            or candidate_meta.get("dataset_fingerprint") != expected_fingerprint
            or evaluation_report.get("data", {}).get("dataset_fingerprint") != expected_fingerprint):
        raise ApprovalReviewError("Dataset fingerprint differs across Stage 4, Stage 5, or Stage 6 artifacts.")
    if (dataset_meta.get("feature_count") != 34 or dataset_meta.get("feature_names") != list(FINAL_FEATURES)
            or candidate_meta.get("feature_count") != 34
            or candidate_meta.get("feature_names") != list(FINAL_FEATURES)):
        raise ApprovalReviewError("Dataset/candidate feature contract is inconsistent with the 34-feature model contract.")
    eligibility = dataset_meta.get("eligibility")
    provenance = dataset_meta.get("label_provenance")
    if (not isinstance(eligibility, Mapping) or eligibility.get("decision_status") != "ELIGIBLE"
            or not isinstance(provenance, Mapping)
            or provenance.get("source_type") != "trusted_verified"
            or provenance.get("synthetic") is not False
            or provenance.get("matching_key") != "event_id"
            or provenance.get("transaction_event_id_mapping_verified") is not True
            or not isinstance(provenance.get("source"), str) or not provenance.get("source", "").strip()
            or not isinstance(provenance.get("label_provenance"), str)
            or not provenance.get("label_provenance", "").strip()
            or "paysim" in str(provenance).casefold()
            or "simulat" in str(provenance).casefold()):
        raise ApprovalReviewError("Stage 4 eligibility or genuinely verified event_id provenance is invalid.")
    provenance_validation = dataset_validation.get("checks")
    if (not isinstance(provenance_validation, Mapping)
            or provenance.get("matched_label_count") != dataset_meta.get("row_count")
            or provenance.get("unmatched_label_count") != 0
            or provenance.get("conflict_count") != 0
            or provenance.get("matched_event_ids_sha256") != provenance_validation.get("event_id_sha256")):
        raise ApprovalReviewError("Stage 4 label provenance counts or event-ID mapping do not match dataset validation.")
    if candidate_meta.get("label_provenance_summary") != provenance:
        raise ApprovalReviewError("Candidate metadata label provenance differs from Stage 4 provenance.")
    evaluation_data = evaluation_report.get("data")
    if (not isinstance(evaluation_data, Mapping)
            or evaluation_data.get("feature_count") != 34
            or evaluation_data.get("feature_names") != list(FINAL_FEATURES)):
        raise ApprovalReviewError("Stage 6 report feature contract is inconsistent.")
    if candidate_meta.get("target") != "isFraud" or evaluation_report.get("decision_is_administrative_approval") is not False:
        raise ApprovalReviewError("Candidate target or Stage 6 recommendation semantics are invalid.")
    if (isinstance(evaluation_data.get("test_rows"), bool)
            or not isinstance(evaluation_data.get("test_rows"), int)
            or evaluation_data.get("test_rows", 0) <= 0):
        raise ApprovalReviewError("Stage 6 report does not contain a non-empty held-out test count.")
    timestamp = evaluation_report.get("evaluation_timestamp_utc")
    try:
        parsed_timestamp = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ApprovalReviewError("Stage 6 evaluation timestamp is missing or invalid.") from exc
    if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() is None:
        raise ApprovalReviewError("Stage 6 evaluation timestamp must include a timezone.")

    candidate_dir = resolve_run_path(safe_id, "candidate", run_root=run_root)
    try:
        bundle = validate_bundle(candidate_dir, FINAL_FEATURES)
    except Exception as exc:
        raise ApprovalReviewError(f"Candidate bundle validation failed: {type(exc).__name__}: {exc}") from exc
    if bundle.model_version != candidate_meta.get("model_version"):
        raise ApprovalReviewError("Candidate bundle version differs from its metadata.")
    if (candidate_threshold.get("selected_threshold") != bundle.threshold
            or candidate_threshold.get("selection_method") is None
            or "validation" not in str(candidate_threshold.get("selection_method")).casefold()):
        raise ApprovalReviewError("Candidate threshold is invalid or was not selected using validation data.")

    live_candidate_hashes = dict(bundle.sha256)
    report_candidate = evaluation_report.get("models", {}).get("candidate")
    report_active = evaluation_report.get("models", {}).get("active")
    if not isinstance(report_candidate, Mapping) or not isinstance(report_active, Mapping):
        raise ApprovalReviewError("Evaluation report is missing candidate or active model identity.")
    if (report_candidate.get("model_version") != bundle.model_version
            or report_candidate.get("run_id") != safe_id
            or report_candidate.get("artifact_sha256") != live_candidate_hashes
            or report_candidate.get("threshold") != bundle.threshold):
        raise ApprovalReviewError("Candidate version or artifact hashes differ from the Stage 6 evaluation report.")
    active_threshold = report_active.get("threshold")
    if (not isinstance(report_active.get("model_version"), str) or not report_active.get("model_version")
            or not isinstance(report_active.get("artifact_sha256"), Mapping)
            or isinstance(active_threshold, bool) or not isinstance(active_threshold, (int, float))
            or not 0 < float(active_threshold) < 1):
        raise ApprovalReviewError("Stage 6 active-model identity, threshold, or artifact hashes are invalid.")
    metrics = evaluation_report.get("metrics")
    if not isinstance(metrics, Mapping) or not all(
        isinstance(metrics.get(key), Mapping) for key in ("active", "candidate")
    ):
        raise ApprovalReviewError("Stage 6 report metrics are missing or malformed.")
    required_metrics = {"precision", "recall", "f1", "pr_auc", "roc_auc", "tp", "fp", "tn", "fn", "threshold"}
    if any(not required_metrics.issubset(metrics[key]) for key in ("active", "candidate")):
        raise ApprovalReviewError("Stage 6 report metrics are incomplete.")

    evaluation_hashes = {
        "evaluation_report_sha256": _sha256(paths["evaluation_report"]),
        "evaluation_validation_sha256": _sha256(paths["evaluation_validation"]),
        "model_comparison_sha256": _sha256(paths["model_comparison"]),
    }
    candidate_hashes = {
        "candidate_metadata_sha256": _sha256(paths["candidate_metadata"]),
        "candidate_model_sha256": _sha256(paths["candidate_model"]),
        "candidate_threshold_sha256": _sha256(paths["candidate_threshold"]),
        "candidate_training_report_sha256": _sha256(paths["candidate_training_report"]),
        "candidate_bundle_sha256": live_candidate_hashes,
    }
    run_record_hash = _sha256(paths["run_record"])
    review_binding = {
        "run_id": safe_id,
        "run_record_sha256": run_record_hash,
        "dataset_fingerprint": expected_fingerprint,
        "dataset_metadata_sha256": _sha256(paths["dataset_metadata"]),
        "dataset_validation_sha256": _sha256(paths["dataset_validation"]),
        **candidate_hashes,
        **evaluation_hashes,
    }
    review_token = hashlib.sha256(json.dumps(review_binding, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    existing_decision = _read_decision_file(safe_id, run_root)
    return {
        "run_id": safe_id,
        "candidate_version": bundle.model_version,
        "active_model_version_at_evaluation": report_active["model_version"],
        "evaluation_timestamp_utc": timestamp,
        "evaluation_status": evaluation_status,
        "evaluation_completed": True,
        "candidate_metrics": dict(metrics["candidate"]),
        "active_model_metrics": dict(metrics["active"]),
        "metric_differences": evaluation_report.get("comparison"),
        "evaluation_reasons": evaluation_report.get("decision_reasons", []),
        "criteria": evaluation_report.get("criteria"),
        "criteria_results": evaluation_report.get("criteria_results"),
        "candidate_hashes": candidate_hashes,
        "evaluation_hashes": evaluation_hashes,
        "dataset_fingerprint": expected_fingerprint,
        "review_token": review_token,
        "decision_exists": existing_decision is not None,
        "existing_decision": existing_decision,
        "review_binding": review_binding,
        "run_directory": str(run_dir),
    }


def _authenticated_admin_context() -> tuple[str, str]:
    """Obtain identity only from the existing server-side Streamlit auth context."""
    try:
        from ui import auth

        role = auth.current_role()
        username = auth.st.session_state.get("authenticated_username")
    except Exception as exc:
        raise ApprovalAuthorizationError("The existing authenticated Streamlit context is unavailable.") from exc
    if role != "admin":
        raise ApprovalAuthorizationError("Only an authenticated administrator may record a decision.")
    if not isinstance(username, str) or not username.strip():
        raise ApprovalAuthorizationError("Authenticated administrator identity is unavailable.")
    return username.strip(), role


def _write_decision_create_only(run_id: str, run_root: str | Path, record: Mapping[str, Any]) -> Path:
    run_dir = resolve_run_dir(run_id, run_root)
    lexical_approval_dir = run_dir / "approval"
    if lexical_approval_dir.is_symlink():
        raise ApprovalError("Approval path is a symlink; refusing redirected audit writes.")
    approval_dir = resolve_run_path(run_id, "approval", run_root=run_root)
    decision_path = _decision_path(run_id, run_root)
    if approval_dir.exists() or approval_dir.is_symlink() or decision_path.exists() or decision_path.is_symlink():
        raise ApprovalError("An approval directory or decision already exists; decisions are immutable per run.")
    created_directory = False
    temporary: Path | None = None
    try:
        approval_dir.mkdir(exist_ok=False)
        created_directory = True
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=approval_dir,
            prefix=".decision.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(dict(record), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and cannot replace a pre-existing file.
        os.link(temporary, decision_path)
        return decision_path
    except FileExistsError as exc:
        raise ApprovalError("A decision already exists; conflicting or duplicate decisions are rejected.") from exc
    except (OSError, TypeError, ValueError) as exc:
        raise ApprovalError(f"Could not safely create the administrator decision: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if created_directory and approval_dir.is_dir() and not any(approval_dir.iterdir()):
            approval_dir.rmdir()


def record_admin_decision(
    run_id: str,
    decision: str,
    reason: str,
    *,
    review_token: str,
    run_root: str | Path = RUNS_ROOT,
) -> dict[str, Any]:
    """Record one immutable admin decision bound to the reviewed artifact snapshot.

    Identity/role are intentionally not function parameters. Authorization is read
    from the existing authenticated Streamlit session; a role or username supplied
    by a caller cannot grant access. The caller must pass the token from its latest
    ``get_approval_review`` response, and all bound artifacts are re-read before
    create-only persistence.
    """
    admin_identity, admin_role = _authenticated_admin_context()
    if decision not in _ALLOWED_DECISIONS:
        raise ApprovalError("Decision must be exactly APPROVED or REJECTED.")
    if not isinstance(reason, str) or not reason.strip():
        raise ApprovalError("A non-empty administrator reason/comment is required.")
    if not isinstance(review_token, str) or not review_token.strip():
        raise ApprovalError("A current review_token from get_approval_review is required.")

    safe_id, _ = _run_directory(run_id, run_root)
    if _read_decision_file(safe_id, run_root) is not None:
        raise ApprovalError("A decision already exists; duplicate or conflicting decisions are rejected.")
    review = get_approval_review(safe_id, run_root=run_root)
    if review["decision_exists"]:
        raise ApprovalError("A decision already exists; duplicate or conflicting decisions are rejected.")
    if not hmac.compare_digest(review["review_token"], review_token):
        raise ApprovalError("Review artifacts changed or the review token is stale; obtain a fresh review.")
    if decision == "APPROVED" and review["evaluation_status"] == "CANDIDATE_REJECTED":
        raise ApprovalError("The evaluation recommends rejection; approval override is disabled for this service.")

    record = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "run_id": safe_id,
        "decision": decision,
        "reason": reason.strip(),
        "administrator": {"identity": admin_identity, "role": admin_role},
        "decided_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_version": review["candidate_version"],
        "active_model_version_at_evaluation": review["active_model_version_at_evaluation"],
        "candidate_metadata_sha256": review["candidate_hashes"]["candidate_metadata_sha256"],
        "candidate_model_sha256": review["candidate_hashes"]["candidate_model_sha256"],
        "candidate_threshold_sha256": review["candidate_hashes"]["candidate_threshold_sha256"],
        "candidate_bundle_sha256": review["candidate_hashes"]["candidate_bundle_sha256"],
        "evaluation_report_sha256": review["evaluation_hashes"]["evaluation_report_sha256"],
        "evaluation_validation_sha256": review["evaluation_hashes"]["evaluation_validation_sha256"],
        "model_comparison_sha256": review["evaluation_hashes"]["model_comparison_sha256"],
        "dataset_fingerprint": review["dataset_fingerprint"],
        "evaluation_status_at_decision": review["evaluation_status"],
        "review_token": review["review_token"],
        "promotion_performed": False,
    }
    path = _write_decision_create_only(safe_id, run_root, record)
    return {**record, "decision_path": str(path)}


__all__ = [
    "APPROVAL_FILENAME",
    "DECISION_SCHEMA_VERSION",
    "ApprovalAuthorizationError",
    "ApprovalError",
    "ApprovalReviewError",
    "get_admin_decision",
    "get_approval_review",
    "record_admin_decision",
]
