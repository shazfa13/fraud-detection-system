"""Explicit, hash-verified promotion through the shared active-model manifest.

This isolated adapter never copies candidate files into V1/V2 locations and never
invokes the older Stage 5 workflow or bundle-directory activator.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from src.orchestrator import approval
from src.orchestrator.config import (
    PROJECT_ROOT,
    RUNS_ROOT,
    OrchestratorPathError,
    resolve_run_dir,
    resolve_run_path,
    validate_run_id,
    validate_run_root,
)
from src.retraining.active_model import resolve_active_model
from src.retraining.cycle_pipeline import FINAL_FEATURES
from src.retraining.model_activator import validate_bundle


MANIFEST_SCHEMA_VERSION = 1
PROMOTION_SCHEMA_VERSION = "orchestrator-promotion-v1"
MANIFEST_NAME = "active_model_manifest.json"
PROMOTION_LOCK_NAME = ".promotion.lock"


class PromotionError(ValueError):
    """Raised when promotion preconditions or safe activation fail."""


class PromotionAuthorizationError(PromotionError):
    """Raised when the trusted current Streamlit context is not an administrator."""


class PromotionRecoveryError(PromotionError):
    """Raised when a failed activation could not be restored to its previous state."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        raise PromotionError(f"{label} is missing or invalid: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PromotionError(f"{label} must be a JSON object: {path}")
    return value


def _validated_paths(run_id: str | None, run_root: str | Path,
                     project_root: str | Path) -> tuple[str, Path, Path, Path]:
    if run_id is None:
        raise PromotionError("A valid orchestrator run_id is required.")
    try:
        safe_id = validate_run_id(run_id)
        root = validate_run_root(run_root)
        run_dir = resolve_run_dir(safe_id, root)
        project = Path(project_root).expanduser().resolve()
    except (OrchestratorPathError, TypeError, OSError) as exc:
        raise PromotionError(f"Invalid run ID or path configuration: {exc}") from exc
    if (root / safe_id).is_symlink() or not run_dir.is_dir():
        raise PromotionError(f"Orchestrator run does not exist or is unsafe: {safe_id}")
    if not run_dir.is_relative_to(root):
        raise PromotionError("Run directory escaped the configured orchestrator root.")
    record_path = resolve_run_path(safe_id, "run.json", run_root=root)
    if record_path.is_symlink():
        raise PromotionError("Run record path cannot be a symlink.")
    run_record = _read_json(record_path, "Run record")
    if run_record.get("run_id") != safe_id:
        raise PromotionError("Run record identity differs from the requested run.")
    manifest = project / "models" / MANIFEST_NAME
    if not project.is_dir() or not (project / "models").is_dir() or (project / "models").is_symlink():
        raise PromotionError("Project model directory is missing or unsafe.")
    candidate = resolve_run_path(safe_id, "candidate", run_root=root)
    if not candidate.is_relative_to(project):
        raise PromotionError("Candidate artifacts must be inside the project root for manifest resolution.")
    return safe_id, root, run_dir, manifest


def _manifest_snapshot(manifest_path: Path) -> tuple[bytes | None, dict[str, Any] | None, str | None]:
    if manifest_path.is_symlink():
        raise PromotionError("Active-model manifest cannot be a symlink.")
    if not manifest_path.exists():
        return None, None, None
    if not manifest_path.is_file():
        raise PromotionError("Active-model manifest path is not a regular file.")
    try:
        raw = manifest_path.read_bytes()
        parsed = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PromotionError(f"Existing active-model manifest is invalid: {exc}") from exc
    if not isinstance(parsed, dict):
        raise PromotionError("Existing active-model manifest must contain a JSON object.")
    return raw, parsed, _sha256_bytes(raw)


def _resolve_active(project_root: Path):
    try:
        return resolve_active_model(project_root=project_root)
    except Exception as exc:
        raise PromotionError(f"Active-model resolver failed validation: {type(exc).__name__}: {exc}") from exc


def _candidate_manifest(candidate_dir: Path, project_root: Path, bundle: Any) -> dict[str, Any]:
    resolved_candidate = candidate_dir.resolve()
    if not resolved_candidate.is_relative_to(project_root):
        raise PromotionError("Candidate bundle cannot be referenced from the project-root manifest.")
    files = {"model": resolved_candidate / "model.json",
             "metadata": resolved_candidate / "metadata.json",
             "threshold": resolved_candidate / "threshold.json"}
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "model_version": bundle.model_version,
        "artifacts": {name: path.relative_to(project_root).as_posix() for name, path in files.items()},
        "artifact_sha256": dict(bundle.sha256),
        "activated_at": _now(),
    }


def _validate_candidate_manifest(project_root: Path, manifest: Mapping[str, Any]) -> Any:
    models_dir = project_root / "models"
    validation_path = models_dir / f".orchestrator-manifest-validation-{uuid.uuid4().hex}.json"
    try:
        _write_json_create_only(validation_path, manifest)
        try:
            return resolve_active_model(project_root=project_root, manifest_path=validation_path)
        except Exception as exc:
            raise PromotionError(f"Candidate manifest failed shared resolver validation: {type(exc).__name__}: {exc}") from exc
    finally:
        validation_path.unlink(missing_ok=True)


def _auth_admin() -> tuple[str, str]:
    try:
        identity, role = approval._authenticated_admin_context()
    except Exception as exc:
        raise PromotionAuthorizationError(f"Authenticated administrator context is unavailable: {exc}") from exc
    if role != "admin" or not isinstance(identity, str) or not identity.strip():
        raise PromotionAuthorizationError("Promotion requires an authenticated administrator.")
    return identity.strip(), role


def _approval_evidence(run_id: str, run_root: Path, project_root: Path) -> dict[str, Any]:
    try:
        review = approval.get_approval_review(run_id, run_root=run_root)
        decision = approval.get_admin_decision(run_id, run_root=run_root)
    except Exception as exc:
        raise PromotionError(f"Stage 4–7 approval review failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(decision, Mapping):
        raise PromotionError("A Stage 7 administrator decision is required.")
    if decision.get("schema_version") != approval.DECISION_SCHEMA_VERSION or decision.get("run_id") != run_id:
        raise PromotionError("Administrator decision schema or run identity is invalid.")
    if decision.get("decision") != "APPROVED":
        raise PromotionError("Only an explicit Stage 7 APPROVED decision can authorize promotion.")
    admin = decision.get("administrator")
    if (not isinstance(admin, Mapping) or admin.get("role") != "admin"
            or not isinstance(admin.get("identity"), str) or not admin.get("identity", "").strip()):
        raise PromotionError("Approval record does not identify a valid administrator.")
    if not isinstance(decision.get("reason"), str) or not decision.get("reason", "").strip():
        raise PromotionError("Approval record is missing its administrator reason.")
    try:
        approved_at = datetime.fromisoformat(str(decision.get("decided_at_utc", "")).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PromotionError("Approval timestamp is invalid.") from exc
    if approved_at.tzinfo is None or approved_at.utcoffset() is None:
        raise PromotionError("Approval timestamp must be timezone-aware.")
    if decision.get("promotion_performed") is not False:
        raise PromotionError("Stage 7 approval record has an invalid promotion state.")
    if decision.get("review_token") != review.get("review_token"):
        raise PromotionError("Candidate/evaluation artifacts changed after administrator approval.")
    if review.get("decision_exists") is not True or review.get("evaluation_completed") is not True:
        raise PromotionError("Stage 7 review does not confirm a complete approved candidate evaluation.")
    if review.get("evaluation_status") == "CANDIDATE_REJECTED":
        raise PromotionError("A rejected evaluation recommendation cannot be promoted under the Stage 7 policy.")

    candidates = review.get("candidate_hashes")
    evaluations = review.get("evaluation_hashes")
    if not isinstance(candidates, Mapping) or not isinstance(evaluations, Mapping):
        raise PromotionError("Candidate or evaluation hash evidence is unavailable.")
    expected = {
        "candidate_version": review.get("candidate_version"),
        "candidate_metadata_sha256": candidates.get("candidate_metadata_sha256"),
        "candidate_model_sha256": candidates.get("candidate_model_sha256"),
        "candidate_threshold_sha256": candidates.get("candidate_threshold_sha256"),
        "candidate_bundle_sha256": candidates.get("candidate_bundle_sha256"),
        "evaluation_report_sha256": evaluations.get("evaluation_report_sha256"),
        "evaluation_validation_sha256": evaluations.get("evaluation_validation_sha256"),
        "model_comparison_sha256": evaluations.get("model_comparison_sha256"),
        "dataset_fingerprint": review.get("dataset_fingerprint"),
        "evaluation_status_at_decision": review.get("evaluation_status"),
    }
    for field, current_value in expected.items():
        if decision.get(field) != current_value:
            raise PromotionError(f"Approval record does not match the current reviewed {field}.")
    approval_path = resolve_run_path(run_id, "approval", approval.APPROVAL_FILENAME, run_root=run_root)
    approval_hash = _sha256(approval_path)

    evaluation_path = resolve_run_path(run_id, "evaluation", "evaluation_report.json", run_root=run_root)
    evaluation_report = _read_json(evaluation_path, "Evaluation report")
    active_report = evaluation_report.get("models", {}).get("active")
    if not isinstance(active_report, Mapping):
        raise PromotionError("Evaluation report does not identify its active model.")
    active = _resolve_active(project_root)
    if (active_report.get("model_version") != active.model_version
            or active_report.get("artifact_sha256") != dict(active.artifact_sha256)
            or active_report.get("threshold") != active.threshold):
        raise PromotionError("Active model changed since candidate evaluation; a fresh candidate evaluation is required.")
    raw_manifest, parsed_manifest, manifest_hash = _manifest_snapshot(project_root / "models" / MANIFEST_NAME)
    candidate_dir = resolve_run_path(run_id, "candidate", run_root=run_root)
    try:
        candidate_bundle = validate_bundle(candidate_dir, FINAL_FEATURES)
    except Exception as exc:
        raise PromotionError(f"Candidate bundle validation failed: {type(exc).__name__}: {exc}") from exc
    if candidate_bundle.model_version != review.get("candidate_version"):
        raise PromotionError("Candidate version differs from its approved identity.")
    if candidate_bundle.model_version == active.model_version:
        raise PromotionError("Candidate model version is already active.")
    if dict(candidate_bundle.sha256) != candidates.get("candidate_bundle_sha256"):
        raise PromotionError("Candidate artifact hashes changed after approval.")

    return {
        "run_id": run_id,
        "candidate_version": candidate_bundle.model_version,
        "active_model_version": active.model_version,
        "candidate_dir": str(candidate_dir),
        "candidate_artifacts": {
            "model": str(candidate_dir / "model.json"),
            "metadata": str(candidate_dir / "metadata.json"),
            "threshold": str(candidate_dir / "threshold.json"),
        },
        "candidate_artifact_sha256": dict(candidate_bundle.sha256),
        "candidate_metadata_sha256": candidates["candidate_metadata_sha256"],
        "candidate_model_sha256": candidates["candidate_model_sha256"],
        "evaluation_report_sha256": evaluations["evaluation_report_sha256"],
        "evaluation_validation_sha256": evaluations["evaluation_validation_sha256"],
        "approval_record_sha256": approval_hash,
        "dataset_fingerprint": review["dataset_fingerprint"],
        "evaluation_status": review["evaluation_status"],
        "approval_decision": decision["decision"],
        "administrator": dict(admin),
        "approval_timestamp_utc": decision["decided_at_utc"],
        "feature_count": candidate_bundle.feature_count,
        "active_artifact_sha256": dict(active.artifact_sha256),
        "active_threshold": active.threshold,
        "manifest_present": raw_manifest is not None,
        "manifest_sha256": manifest_hash,
        "manifest": parsed_manifest,
        "promotion_exists": False,
        "promotion_ready": True,
        "validation_failures": [],
        "promotion_token": _promotion_token(
            run_id, decision, candidates, evaluations, approval_hash, review["review_token"],
            active, manifest_hash,
        ),
    }


def _promotion_token(run_id: str, decision: Mapping[str, Any], candidates: Mapping[str, Any],
                     evaluations: Mapping[str, Any], approval_hash: str, stage7_review_token: str,
                     active: Any, manifest_hash: str | None) -> str:
    binding = {
        "run_id": run_id,
        "approval_record_sha256": approval_hash,
        "stage7_review_token": stage7_review_token,
        "decision": decision.get("decision"),
        "candidate_hashes": dict(candidates),
        "evaluation_hashes": dict(evaluations),
        "active_model_version": active.model_version,
        "active_artifact_sha256": dict(active.artifact_sha256),
        "manifest_sha256": manifest_hash,
    }
    encoded = json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def get_promotion_review(run_id: str | None, *, run_root: str | Path = RUNS_ROOT,
                         project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    """Return a read-only promotion readiness summary and current hash-bound token."""
    try:
        safe_id, root, run_dir, manifest_path = _validated_paths(run_id, run_root, project_root)
        promotion_dir = resolve_run_path(safe_id, "promotion", run_root=root)
        if promotion_dir.exists() or promotion_dir.is_symlink():
            status = get_promotion_status(safe_id, run_root=root, project_root=project_root)
            raise PromotionError(f"A prior or incomplete promotion state exists: {status.get('status')}.")
        if (root / PROMOTION_LOCK_NAME).exists():
            raise PromotionError("Another promotion holds the orchestrator promotion lock.")
        result = _approval_evidence(safe_id, root, Path(project_root).resolve())
        if not manifest_path.parent.is_dir():
            raise PromotionError("Project models directory is unavailable.")
        return result
    except Exception as exc:
        promotion_exists = False
        try:
            valid_id = validate_run_id(run_id) if run_id is not None else None
            if valid_id is not None:
                promotion_exists = resolve_run_path(valid_id, "promotion", run_root=run_root).exists()
        except Exception:
            pass
        return {
            "run_id": run_id,
            "promotion_ready": False,
            "validation_failures": [f"{type(exc).__name__}: {exc}"],
            "promotion_exists": promotion_exists,
        }


def _write_bytes_create_only(path: Path, value: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise PromotionError(f"Refusing to overwrite promotion history: {path}") from exc
    except OSError as exc:
        raise PromotionError(f"Could not safely create promotion history {path}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    data = (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    _write_bytes_create_only(path, data)


def _atomic_replace_bytes(path: Path, value: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_replace_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    data = (json.dumps(dict(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    _atomic_replace_bytes(path, data)


@contextmanager
def _promotion_lock(run_root: Path, run_id: str, admin_identity: str) -> Iterator[Path]:
    lock_dir = run_root / PROMOTION_LOCK_NAME
    if lock_dir.is_symlink():
        raise PromotionError("Promotion lock path is a symlink.")
    try:
        lock_dir.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise PromotionError("Another promotion is in progress or a stale promotion lock requires manual recovery.") from exc
    except OSError as exc:
        raise PromotionError(f"Could not acquire promotion lock: {exc}") from exc
    owner = lock_dir / "owner.json"
    owner_created = False
    try:
        _write_json_create_only(owner, {
            "run_id": run_id,
            "administrator": admin_identity,
            "pid": os.getpid(),
            "acquired_at_utc": _now(),
        })
        owner_created = True
        yield lock_dir
    finally:
        if owner_created and owner.exists() and not owner.is_symlink():
            owner.unlink()
        try:
            lock_dir.rmdir()
        except OSError:
            # A stale or externally modified lock is preserved for investigation.
            pass


def _restore_active_state(manifest_path: Path, previous_bytes: bytes | None,
                          project_root: Path, expected_active: Any) -> None:
    if previous_bytes is None:
        manifest_path.unlink(missing_ok=True)
    else:
        _atomic_replace_bytes(manifest_path, previous_bytes)
    restored = _resolve_active(project_root)
    if (restored.model_version != expected_active.model_version
            or dict(restored.artifact_sha256) != dict(expected_active.artifact_sha256)):
        raise PromotionRecoveryError("Previous active-model resolution did not verify after recovery.")


def _record_failure(promotion_dir: Path, run_id: str, reason: str,
                    manifest_path: Path, previous_manifest_hash: str | None,
                    active_restored: bool, recovery_error: str | None = None) -> None:
    failure_path = promotion_dir / "promotion_failure.json"
    if failure_path.exists():
        return
    try:
        _write_json_create_only(failure_path, {
            "schema_version": PROMOTION_SCHEMA_VERSION,
            "run_id": run_id,
            "status": "PROMOTION_FAILED" if active_restored else "RECOVERY_REQUIRED",
            "reason": reason,
            "previous_manifest_sha256": previous_manifest_hash,
            "current_manifest_sha256": _sha256(manifest_path) if manifest_path.is_file() else None,
            "active_restored": active_restored,
            "recovery_error": recovery_error,
            "failed_at_utc": _now(),
        })
    except Exception:
        # promotion_started.json remains as durable evidence of an incomplete attempt.
        pass


def promote_approved_candidate(
    run_id: str | None,
    *,
    review_token: str,
    promotion_reason: str,
    run_root: str | Path = RUNS_ROOT,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Explicitly promote the approved run candidate through the active manifest.

    Identity is established from the existing authenticated Streamlit context.
    The function returns success only after the shared resolver verifies the new
    manifest and a create-only promotion record is durable.
    """
    admin_identity, admin_role = _auth_admin()
    if not isinstance(review_token, str) or not review_token.strip():
        raise PromotionError("A current promotion_token from get_promotion_review is required.")
    if not isinstance(promotion_reason, str) or not promotion_reason.strip():
        raise PromotionError("A non-empty promotion reason is required.")
    safe_id, root, run_dir, manifest_path = _validated_paths(run_id, run_root, project_root)
    project = Path(project_root).expanduser().resolve()

    with _promotion_lock(root, safe_id, admin_identity):
        review = _approval_evidence(safe_id, root, project)
        expected_token = review["promotion_token"]
        if not hmac.compare_digest(expected_token, review_token):
            raise PromotionError("Promotion review is stale; obtain a fresh review before promoting.")
        promotion_dir = resolve_run_path(safe_id, "promotion", run_root=root)
        if promotion_dir.exists() or promotion_dir.is_symlink():
            raise PromotionError("A promotion attempt or promotion history already exists for this run.")

        active_before = _resolve_active(project)
        previous_bytes, _, previous_manifest_hash = _manifest_snapshot(manifest_path)
        if (active_before.model_version != review["active_model_version"]
                or dict(active_before.artifact_sha256) != review["active_artifact_sha256"]
                or previous_manifest_hash != review["manifest_sha256"]):
            raise PromotionError("Active model or manifest changed after promotion review.")

        candidate_dir = Path(review["candidate_dir"]).resolve()
        candidate_bundle = validate_bundle(candidate_dir, FINAL_FEATURES)
        if dict(candidate_bundle.sha256) != review["candidate_artifact_sha256"]:
            raise PromotionError("Candidate bundle changed after promotion review.")
        candidate_manifest = _candidate_manifest(candidate_dir, project, candidate_bundle)
        resolved_candidate = _validate_candidate_manifest(project, candidate_manifest)
        if (resolved_candidate.model_version != candidate_bundle.model_version
                or dict(resolved_candidate.artifact_sha256) != dict(candidate_bundle.sha256)):
            raise PromotionError("Candidate manifest did not independently resolve to the approved artifact hashes.")

        # Re-check all bound artifacts and the current manifest immediately before
        # beginning the durable promotion transaction.
        fresh_review = _approval_evidence(safe_id, root, project)
        if not hmac.compare_digest(fresh_review["promotion_token"], expected_token):
            raise PromotionError("Approval, candidate, evaluation, or active manifest changed before activation.")
        latest_bytes, _, latest_manifest_hash = _manifest_snapshot(manifest_path)
        if latest_manifest_hash != previous_manifest_hash or latest_bytes != previous_bytes:
            raise PromotionError("Active manifest changed before activation.")

        promotion_dir.mkdir(exist_ok=False)
        previous_manifest_path = promotion_dir / "previous_manifest.json"
        try:
            if previous_bytes is not None:
                _write_bytes_create_only(previous_manifest_path, previous_bytes)
            else:
                # Preserve that there was no manifest and retain the effective legacy
                # active selection needed for a future explicit recovery/rollback.
                _write_json_create_only(previous_manifest_path, {
                    "manifest_present": False,
                    "effective_active_model": {
                        "model_version": active_before.model_version,
                        "threshold": active_before.threshold,
                        "artifact_sha256": dict(active_before.artifact_sha256),
                        "artifacts": {
                            "model": active_before.model_path.resolve().relative_to(project).as_posix(),
                            "metadata": active_before.metadata_path.resolve().relative_to(project).as_posix(),
                            "threshold": active_before.threshold_path.resolve().relative_to(project).as_posix(),
                        },
                    },
                })
            started = {
                "schema_version": PROMOTION_SCHEMA_VERSION,
                "run_id": safe_id,
                "status": "PROMOTION_IN_PROGRESS",
                "candidate_version": candidate_bundle.model_version,
                "previous_model_version": active_before.model_version,
                "previous_manifest_present": previous_bytes is not None,
                "previous_manifest_sha256": previous_manifest_hash,
                "candidate_artifact_sha256": dict(candidate_bundle.sha256),
                "evaluation_report_sha256": review["evaluation_report_sha256"],
                "approval_record_sha256": review["approval_record_sha256"],
                "started_by": admin_identity,
                "started_at_utc": _now(),
            }
            _write_json_create_only(promotion_dir / "promotion_started.json", started)
        except Exception as exc:
            _record_failure(promotion_dir, safe_id, f"Could not persist promotion start: {exc}",
                            manifest_path, previous_manifest_hash, True)
            raise PromotionError(f"Could not create complete promotion history; active model was not changed: {exc}") from exc

        new_manifest_bytes = (json.dumps(candidate_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        try:
            # Last compare before replace; the local lock serializes orchestrator
            # promoters. Other legacy writers do not honor this lock (see report).
            before_replace_bytes, _, before_replace_hash = _manifest_snapshot(manifest_path)
            if before_replace_hash != previous_manifest_hash or before_replace_bytes != previous_bytes:
                raise PromotionError("Active manifest changed immediately before atomic replacement.")
            _atomic_replace_manifest(manifest_path, candidate_manifest)
            resolved_after = _resolve_active(project)
            if (resolved_after.model_version != candidate_bundle.model_version
                    or dict(resolved_after.artifact_sha256) != dict(candidate_bundle.sha256)
                    or resolved_after.model_path.resolve() != (candidate_dir / "model.json").resolve()
                    or resolved_after.metadata_path.resolve() != (candidate_dir / "metadata.json").resolve()
                    or resolved_after.threshold_path.resolve() != (candidate_dir / "threshold.json").resolve()
                    or _sha256(manifest_path) != _sha256_bytes(new_manifest_bytes)):
                raise PromotionError("Post-activation resolver verification did not match the approved candidate.")

            record = {
                "schema_version": PROMOTION_SCHEMA_VERSION,
                "run_id": safe_id,
                "status": "PROMOTED",
                "candidate_version": candidate_bundle.model_version,
                "previous_model_version": active_before.model_version,
                "candidate_dir": str(candidate_dir),
                "candidate_artifact_sha256": dict(candidate_bundle.sha256),
                "candidate_model_sha256": candidate_bundle.sha256["model.json"],
                "candidate_metadata_sha256": candidate_bundle.sha256["metadata.json"],
                "candidate_threshold_sha256": candidate_bundle.sha256["threshold.json"],
                "dataset_fingerprint": review["dataset_fingerprint"],
                "evaluation_report_sha256": review["evaluation_report_sha256"],
                "approval_record_sha256": review["approval_record_sha256"],
                "previous_manifest_present": previous_bytes is not None,
                "previous_manifest_sha256": previous_manifest_hash,
                "previous_manifest_file_sha256": _sha256(previous_manifest_path),
                "new_manifest_sha256": _sha256(manifest_path),
                "promoting_administrator": {"identity": admin_identity, "role": admin_role},
                "promotion_reason": promotion_reason.strip(),
                "promoted_at_utc": _now(),
                "resolver_verified": True,
            }
            _write_json_create_only(promotion_dir / "promotion_record.json", record)
            return {**record, "promotion_directory": str(promotion_dir)}
        except Exception as exc:
            # Restore only if the manifest is exactly the candidate manifest this
            # operation wrote. Never clobber a different concurrent writer's state.
            current_hash = _sha256(manifest_path) if manifest_path.is_file() else None
            new_manifest_hash = _sha256_bytes(new_manifest_bytes)
            active_restored = False
            recovery_error = None
            if current_hash == new_manifest_hash:
                try:
                    _restore_active_state(manifest_path, previous_bytes, project, active_before)
                    active_restored = True
                except Exception as restore_exc:
                    recovery_error = f"{type(restore_exc).__name__}: {restore_exc}"
                    active_restored = False
            elif current_hash == previous_manifest_hash:
                try:
                    current_active = _resolve_active(project)
                    active_restored = (current_active.model_version == active_before.model_version
                                       and dict(current_active.artifact_sha256) == dict(active_before.artifact_sha256))
                except Exception as restore_exc:
                    recovery_error = f"Previous active state did not validate: {type(restore_exc).__name__}: {restore_exc}"
            elif previous_bytes is None and not manifest_path.exists():
                try:
                    current_active = _resolve_active(project)
                    active_restored = (current_active.model_version == active_before.model_version
                                       and dict(current_active.artifact_sha256) == dict(active_before.artifact_sha256))
                except Exception as restore_exc:
                    recovery_error = f"Previous legacy active state did not validate: {type(restore_exc).__name__}: {restore_exc}"
            else:
                recovery_error = "Active manifest changed concurrently; it was preserved and requires explicit review."
            _record_failure(promotion_dir, safe_id, f"{type(exc).__name__}: {exc}",
                            manifest_path, previous_manifest_hash, active_restored, recovery_error)
            if not active_restored:
                raise PromotionRecoveryError(
                    f"Promotion did not complete and previous active restoration needs attention: {recovery_error or exc}"
                ) from exc
            raise PromotionError(f"Promotion failed; previous active model remains verified: {type(exc).__name__}: {exc}") from exc


def get_promotion_status(run_id: str | None, *, run_root: str | Path = RUNS_ROOT,
                         project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    """Read the per-run promotion history without changing active model state."""
    try:
        safe_id, root, run_dir, manifest_path = _validated_paths(run_id, run_root, project_root)
        promotion_dir = resolve_run_path(safe_id, "promotion", run_root=root)
        if not promotion_dir.exists():
            return {"run_id": safe_id, "status": "NOT_PROMOTED", "promotion_exists": False}
        if promotion_dir.is_symlink() or not promotion_dir.is_dir():
            raise PromotionError("Promotion history path is unsafe.")
        record_path = promotion_dir / "promotion_record.json"
        failure_path = promotion_dir / "promotion_failure.json"
        started_path = promotion_dir / "promotion_started.json"
        if record_path.is_file() and not record_path.is_symlink():
            record = _read_json(record_path, "Promotion record")
            if record.get("run_id") != safe_id or record.get("status") != "PROMOTED":
                raise PromotionError("Promotion record schema/status is invalid.")
            if not manifest_path.is_file() or _sha256(manifest_path) != record.get("new_manifest_sha256"):
                return {"run_id": safe_id, "status": "PROMOTED_BUT_ACTIVE_MANIFEST_CHANGED",
                        "promotion_exists": True, "promotion_record": record}
            return {"run_id": safe_id, "status": "PROMOTED", "promotion_exists": True,
                    "promotion_record": record}
        if failure_path.is_file() and not failure_path.is_symlink():
            return {"run_id": safe_id, "status": _read_json(failure_path, "Promotion failure").get("status", "PROMOTION_FAILED"),
                    "promotion_exists": True,
                    "promotion_failure": _read_json(failure_path, "Promotion failure")}
        if started_path.is_file() and not started_path.is_symlink():
            started = _read_json(started_path, "Promotion start record")
            active = _resolve_active(Path(project_root).resolve())
            candidate_active = (active.model_version == started.get("candidate_version")
                                and dict(active.artifact_sha256) == started.get("candidate_artifact_sha256"))
            return {"run_id": safe_id,
                    "status": "PROMOTION_INCOMPLETE_CANDIDATE_ACTIVE" if candidate_active else "PROMOTION_INCOMPLETE",
                    "promotion_exists": True, "promotion_started": started}
        return {"run_id": safe_id, "status": "PROMOTION_CONFLICTING_HISTORY",
                "promotion_exists": True, "promotion_directory": str(promotion_dir)}
    except Exception as exc:
        return {"run_id": run_id, "status": "STATUS_UNAVAILABLE",
                "error": f"{type(exc).__name__}: {exc}"}


__all__ = [
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA_VERSION",
    "PROMOTION_SCHEMA_VERSION",
    "PromotionAuthorizationError",
    "PromotionError",
    "PromotionRecoveryError",
    "get_promotion_review",
    "get_promotion_status",
    "promote_approved_candidate",
]
