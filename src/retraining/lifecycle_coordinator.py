"""Offline orchestration for one explicitly authorized retraining cycle.

This coordinator has no background behavior. Phase 9's repository label source is
documented as simulated delayed ground truth, so it is not considered genuine for
automatic retraining. Callers must explicitly attest a verified source and matching
before the injected build/train/evaluate/promotion components can run.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping


class CycleStatus(str, Enum):
    """Terminal status for a single offline cycle attempt."""

    DRIFT_NOT_DETECTED = "DRIFT_NOT_DETECTED"
    WAITING_FOR_LABELS = "WAITING_FOR_LABELS"
    INSUFFICIENT_LABELS = "INSUFFICIENT_LABELS"
    CANDIDATE_BUILD_FAILED = "CANDIDATE_BUILD_FAILED"
    CANDIDATE_TRAINING_FAILED = "CANDIDATE_TRAINING_FAILED"
    EVALUATION_PENDING_POLICY = "EVALUATION_PENDING_POLICY"
    EVALUATION_FAILED = "EVALUATION_FAILED"
    EVALUATION_INCONCLUSIVE = "EVALUATION_INCONCLUSIVE"
    CANDIDATE_REJECTED = "CANDIDATE_REJECTED"
    CANDIDATE_ACCEPTED = "CANDIDATE_ACCEPTED"
    PROMOTION_FAILED = "PROMOTION_FAILED"
    PROMOTION_SUCCEEDED = "PROMOTION_SUCCEEDED"


@dataclass(frozen=True)
class LabelReadiness:
    """Evidence that labels are genuine and joined to transactions unambiguously.

    ``source_verified`` is an explicit caller attestation. The Phase 9 PaySim
    simulated-label artifact must set it to False. Matching must use event_id,
    with no unmatched records, before any candidate work is permitted.
    """

    source_name: str
    source_verified: bool
    matched_label_count: int
    unmatched_record_count: int
    matching_key: str = "event_id"
    provenance: str = ""

    def is_reliable(self) -> bool:
        return (
            self.source_verified
            and bool(self.source_name.strip())
            and self.matching_key == "event_id"
            and self.matched_label_count > 0
            and self.unmatched_record_count == 0
        )


@dataclass(frozen=True)
class LifecycleDependencies:
    """Injected existing pipeline/evaluator/activator operations.

    The callbacks adapt the existing Step 2–4 APIs to the coordinator's small
    context mapping. No callback is invoked until drift and labels pass gates.
    """

    build_candidate: Callable[[Mapping[str, Any]], Any]
    train_candidate: Callable[[Any, Mapping[str, Any]], Any]
    evaluate_candidate: Callable[[Any, Any, Mapping[str, Any]], Mapping[str, Any]]
    validate_candidate: Callable[[Path], Any]
    promote_candidate: Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class CycleRequest:
    """Inputs for one offline cycle attempt."""

    drift_detected: bool
    label_readiness: LabelReadiness | None
    minimum_genuine_labels: int
    candidate_dir: Path
    active_dir: Path
    backup_dir: Path
    report_path: Path
    acceptance_policy_configured: bool = False
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CycleResult:
    """Cycle outcome and concise stage report."""

    status: CycleStatus
    report: Mapping[str, Any]


class LifecycleCoordinatorError(RuntimeError):
    """Raised when a cycle report cannot be safely written."""


def _ensure_disjoint_paths(*paths: Path) -> None:
    """Reject equal or nested model/report paths before invoking any stage."""
    resolved = [path.resolve() for path in paths]
    if any(
        left == right or left in right.parents or right in left.parents
        for index, left in enumerate(resolved)
        for right in resolved[index + 1:]
    ):
        raise LifecycleCoordinatorError(
            "Candidate, active, backup, and report paths must be separate and non-nested."
        )


def _write_new_report(path: Path, report: Mapping[str, Any]) -> None:
    """Atomically create a JSON report without overwriting an existing report."""
    if path.exists():
        raise LifecycleCoordinatorError(f"Refusing to overwrite existing cycle report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication gives create-if-absent semantics on the same volume.
        # It fails safely if another writer created the report in the meantime.
        os.link(temporary, path)
    except FileExistsError as exc:
        raise LifecycleCoordinatorError(f"Refusing to overwrite existing cycle report: {path}") from exc
    except (OSError, TypeError, ValueError) as exc:
        raise LifecycleCoordinatorError(f"Could not write cycle report {path}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_cycle(request: CycleRequest, dependencies: LifecycleDependencies | None = None) -> CycleResult:
    """Run gated offline stages, stopping unless genuine labels are verified.

    Phase 9's configured PaySim labels are simulated. They should therefore be
    represented with ``source_verified=False``. This function never infers
    genuine labels from that report and never runs an unconfigured stage.
    """
    if request.minimum_genuine_labels < 1:
        raise ValueError("minimum_genuine_labels must be positive")
    _ensure_disjoint_paths(
        request.candidate_dir, request.active_dir, request.backup_dir, request.report_path
    )
    # Check report conflicts before any injected work: a repeated cycle must not
    # train, evaluate, or promote and only discover the conflict at report time.
    if request.report_path.exists():
        raise LifecycleCoordinatorError(
            f"Refusing to overwrite existing cycle report: {request.report_path}"
        )

    status = CycleStatus.WAITING_FOR_LABELS
    stages: dict[str, Any] = {"drift_detected": bool(request.drift_detected)}
    errors: list[str] = []
    labels = request.label_readiness

    if not request.drift_detected:
        status = CycleStatus.DRIFT_NOT_DETECTED
    elif labels is None or not labels.is_reliable():
        status = CycleStatus.WAITING_FOR_LABELS
        stages["labels"] = None if labels is None else asdict(labels)
        if labels is not None and not labels.source_verified:
            errors.append("Label source is not verified as genuine; simulated Phase 9 labels cannot trigger retraining.")
    elif labels.matched_label_count < request.minimum_genuine_labels:
        status = CycleStatus.INSUFFICIENT_LABELS
        stages["labels"] = asdict(labels)
    elif dependencies is None:
        status = CycleStatus.CANDIDATE_BUILD_FAILED
        errors.append("Candidate pipeline dependencies are not configured.")
    else:
        stages["labels"] = asdict(labels)
        context = {
            **dict(request.context),
            "candidate_dir": request.candidate_dir,
            "active_dir": request.active_dir,
            "backup_dir": request.backup_dir,
            "minimum_genuine_labels": request.minimum_genuine_labels,
            "label_readiness": labels,
        }
        try:
            built = dependencies.build_candidate(context)
            stages["candidate_build"] = "SUCCEEDED"
        except Exception as exc:
            status = CycleStatus.CANDIDATE_BUILD_FAILED
            errors.append(f"{type(exc).__name__}: {exc}")
            built = None

        trained: Any = None
        if status != CycleStatus.CANDIDATE_BUILD_FAILED:
            try:
                trained = dependencies.train_candidate(built, context)
                stages["candidate_training"] = "SUCCEEDED"
            except Exception as exc:
                status = CycleStatus.CANDIDATE_TRAINING_FAILED
                errors.append(f"{type(exc).__name__}: {exc}")

        evaluation: Mapping[str, Any] | None = None
        if status not in (CycleStatus.CANDIDATE_BUILD_FAILED, CycleStatus.CANDIDATE_TRAINING_FAILED):
            try:
                evaluation = dependencies.evaluate_candidate(
                    built, trained,
                    {**context, "acceptance_policy_configured": request.acceptance_policy_configured},
                )
                stages["evaluation"] = dict(evaluation)
            except Exception as exc:
                status = CycleStatus.EVALUATION_FAILED
                errors.append(f"{type(exc).__name__}: {exc}")

        if evaluation is not None:
            decision = evaluation.get("decision_status")
            complete = evaluation.get("evaluation_completed") is True
            if decision == "PENDING_POLICY" or not request.acceptance_policy_configured:
                status = CycleStatus.EVALUATION_PENDING_POLICY
            elif not complete or decision in ("EVALUATION_FAILED", "FAILED"):
                status = CycleStatus.EVALUATION_FAILED
                errors.append(str(evaluation.get("error") or "Evaluation did not complete."))
            elif decision in ("INCONCLUSIVE", "EVALUATION_INCONCLUSIVE"):
                status = CycleStatus.EVALUATION_INCONCLUSIVE
            elif decision == "REJECTED":
                status = CycleStatus.CANDIDATE_REJECTED
            elif decision != "ACCEPTED":
                status = CycleStatus.EVALUATION_INCONCLUSIVE
                errors.append(f"Unrecognized evaluation decision: {decision!r}")
            else:
                status = CycleStatus.CANDIDATE_ACCEPTED
                try:
                    validation = dependencies.validate_candidate(request.candidate_dir)
                    stages["candidate_bundle_validation"] = "SUCCEEDED"
                except Exception as exc:
                    status = CycleStatus.PROMOTION_FAILED
                    errors.append(f"Candidate bundle validation failed: {type(exc).__name__}: {exc}")
                if status == CycleStatus.CANDIDATE_ACCEPTED:
                    try:
                        promotion = dependencies.promote_candidate(context)
                        stages["promotion"] = promotion if isinstance(promotion, (str, int, float, bool, dict, list, type(None))) else repr(promotion)
                        status = CycleStatus.PROMOTION_SUCCEEDED
                    except Exception as exc:
                        status = CycleStatus.PROMOTION_FAILED
                        errors.append(f"{type(exc).__name__}: {exc}")

    report: dict[str, Any] = {
        "cycle_status": status.value,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "drift_detected": bool(request.drift_detected),
            "minimum_genuine_labels": request.minimum_genuine_labels,
            "acceptance_policy_configured": request.acceptance_policy_configured,
            "candidate_dir": str(request.candidate_dir),
            "active_dir": str(request.active_dir),
            "backup_dir": str(request.backup_dir),
        },
        "stages": stages,
        "errors": errors,
    }
    _write_new_report(request.report_path, report)
    return CycleResult(status=status, report=report)
