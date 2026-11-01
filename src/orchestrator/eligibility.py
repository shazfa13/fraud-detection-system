"""Deterministic retraining eligibility rules for the optional orchestrator."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Integral
from typing import Any, Mapping

from src.orchestrator import drift_adapter, label_adapter, model_adapter


RULE_VERSION = "orchestrator-eligibility-v1"
# Conservative academic demonstration floor, aligned with the existing Stage 3
# monitor's provisional minimum. It is not a production threshold and requires
# validation against a representative, genuinely labelled cohort.
DEFAULT_MINIMUM_GENUINE_LABELS = 100


@dataclass(frozen=True)
class EligibilityPolicy:
    """Per-evaluation label floor for the deterministic eligibility rules."""

    minimum_genuine_labels: int = DEFAULT_MINIMUM_GENUINE_LABELS
    rule_version: str = RULE_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.minimum_genuine_labels, bool) or not isinstance(self.minimum_genuine_labels, Integral):
            raise ValueError("minimum_genuine_labels must be a positive integer.")
        if self.minimum_genuine_labels < 1:
            raise ValueError("minimum_genuine_labels must be a positive integer.")
        if not isinstance(self.rule_version, str) or not self.rule_version.strip():
            raise ValueError("rule_version must be a non-empty string.")


def _adapter_reasons(summary: Mapping[str, Any], key: str) -> list[str]:
    value = summary.get(key, [])
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    return int(value)


def evaluate_summaries(active: Mapping[str, Any], drift: Mapping[str, Any],
                       labels: Mapping[str, Any], *,
                       policy: EligibilityPolicy = EligibilityPolicy(),
                       evaluated_at: str | None = None) -> dict[str, Any]:
    """Evaluate supplied adapter summaries; unknown or conflicting evidence blocks.

    This pure decision function does not read or write project state. The public
    ``evaluate_retraining_eligibility`` function obtains the summaries through
    the Stage 2 adapters and delegates here.
    """
    if not isinstance(policy, EligibilityPolicy):
        raise TypeError("policy must be an EligibilityPolicy.")
    active = active if isinstance(active, Mapping) else {}
    drift = drift if isinstance(drift, Mapping) else {}
    labels = labels if isinstance(labels, Mapping) else {}
    timestamp = evaluated_at or datetime.now(timezone.utc).isoformat()
    reasons: list[str] = []
    warnings = _adapter_reasons(drift, "warnings") + _adapter_reasons(labels, "warnings")
    active_status = active.get("status")
    active_version = active.get("model_version")
    drift_report_status = drift.get("status")
    raw_drift_status = drift.get("drift_status")
    label_summary_status = labels.get("status")
    label_ready = labels.get("strict_retraining_ready") is True

    active_integrity = active.get("integrity")
    active_integrity_status = active_integrity.get("status") if isinstance(active_integrity, Mapping) else None
    active_adapter_reasons = _adapter_reasons(active, "reasons")
    if not active_adapter_reasons and isinstance(active_integrity, Mapping) and active_integrity.get("error"):
        active_adapter_reasons = [str(active_integrity["error"])]
    adapter_reasons = {
        "active_model": active_adapter_reasons,
        "drift": _adapter_reasons(drift, "warnings"),
        "labels": _adapter_reasons(labels, "readiness_reasons"),
    }

    decision: str
    normalized_drift: str | None = None
    drift_decision_confirmed = False
    if (active_status != "VALID" or not isinstance(active_version, str) or not active_version.strip()
            or active_integrity_status != "VALIDATED_BY_SHARED_RESOLVER"):
        decision = "BLOCKED"
        reasons.append("Active model information is missing or failed resolver validation.")
        reasons.extend(adapter_reasons["active_model"])
    elif drift_report_status == "NO_SAVED_REPORT" or drift.get("report_exists") is False:
        decision = "BLOCKED"
        reasons.append("No saved monitoring report is available; drift evidence is insufficient.")
    elif drift_report_status != "SAVED_REPORT" or drift.get("report_exists") is not True:
        decision = "BLOCKED"
        reasons.append("Monitoring report availability is unknown or invalid.")
        reasons.extend(adapter_reasons["drift"])
    else:
        detected = drift.get("drift_decision")
        if detected is True:
            if raw_drift_status not in (None, "drift"):
                decision = "BLOCKED"
                reasons.append("Drift decision and drift status are inconsistent or ambiguous.")
            else:
                normalized_drift = "DRIFT_DETECTED"
                drift_decision_confirmed = True
                decision = "BLOCKED"  # changed to ELIGIBLE only after every label gate passes
                reasons.append("Feature drift is detected; verified labels and all readiness gates are required.")
        elif detected is False:
            if raw_drift_status not in (None, "no_drift"):
                decision = "BLOCKED"
                reasons.append("No-drift decision conflicts with the saved drift status.")
            else:
                normalized_drift = "NO_DRIFT"
                decision = "NOT_REQUIRED"
                reasons.append("The saved monitoring report does not detect feature drift.")
        elif raw_drift_status == "drift":
            normalized_drift = "DRIFT_DETECTED"
            decision = "BLOCKED"
            reasons.append("The report indicates drift but has no explicit boolean drift decision.")
        elif raw_drift_status == "no_drift":
            normalized_drift = "NO_DRIFT"
            decision = "BLOCKED"
            reasons.append("The report indicates no drift but has no explicit boolean drift decision.")
        else:
            decision = "BLOCKED"
            reasons.append("Drift state is unknown or ambiguous; eligibility cannot be established.")

        if decision == "BLOCKED" and normalized_drift == "DRIFT_DETECTED":
            reasons.extend(adapter_reasons["labels"])
            if label_summary_status != "AVAILABLE":
                reasons.append("Label readiness summary is unavailable.")
            verified_count = _count(labels.get("genuinely_verified_labels"))
            if verified_count is None:
                reasons.append("The genuinely verified label count is unknown or invalid.")
            elif verified_count < policy.minimum_genuine_labels:
                reasons.append(
                    f"Insufficient genuinely verified labels: {verified_count} available; "
                    f"{policy.minimum_genuine_labels} required by the configured academic demonstration floor."
                )
            if labels.get("strict_retraining_ready") is not True:
                reasons.append("The existing strict label-readiness gate is not satisfied.")
            matching_key = labels.get("matching_key")
            if matching_key != "event_id":
                reasons.append("Label matching is not confirmed on the lifecycle-required event_id key.")
            source_configuration = labels.get("trusted_source_configuration")
            if not isinstance(source_configuration, Mapping):
                reasons.append("Trusted-source configuration details are unknown.")
            else:
                if not source_configuration.get("configured_sources"):
                    reasons.append("No trusted label source is configured.")
                if source_configuration.get("authorization_token_configured") is not True:
                    reasons.append("Trusted-source authorization is not configured.")
            conflicts = _count(labels.get("conflict_count"))
            if conflicts is None:
                reasons.append("Label/prediction conflict count is unknown.")
            elif conflicts > 0:
                reasons.append(f"Label or prediction conflicts block retraining eligibility ({conflicts}).")
            unmatched = _count(labels.get("unmatched_labels"))
            if unmatched is None:
                reasons.append("Unmatched-label count is unknown.")
            elif unmatched > 0:
                reasons.append(f"Unmatched labels block retraining eligibility ({unmatched}).")
            unresolved = labels.get("unresolved_or_rejected_count")
            unresolved_count = _count(unresolved)
            if unresolved_count is None:
                reasons.append("Unresolved/rejected label count is unknown or invalid.")
            elif unresolved_count > 0:
                reasons.append(f"Unresolved or rejected labels are present ({unresolved_count}).")
            simulated = labels.get("simulated_labels")
            if verified_count == 0 and (simulated is True or (isinstance(simulated, Integral) and simulated > 0)
                                        or labels.get("phase9_label_type") == "SIMULATED_DELAYED_GROUND_TRUTH"):
                reasons.append("Only simulated labels are available; PaySim labels cannot establish genuine-label readiness.")

            no_hard_label_blocks = (
                drift_decision_confirmed
                and label_summary_status == "AVAILABLE"
                and verified_count is not None
                and verified_count >= policy.minimum_genuine_labels
                and labels.get("strict_retraining_ready") is True
                and matching_key == "event_id"
                and isinstance(source_configuration, Mapping)
                and bool(source_configuration.get("configured_sources"))
                and source_configuration.get("authorization_token_configured") is True
                and conflicts == 0
                and unmatched == 0
                and unresolved_count == 0
                and not adapter_reasons["labels"]
            )
            if no_hard_label_blocks and not any(
                "unknown" in reason.lower() or "unavailable" in reason.lower()
                for reason in adapter_reasons["labels"]
            ):
                decision = "ELIGIBLE"
                reasons.append("Drift is detected and strict genuine-label readiness meets the configured count floor.")

    # NOT_REQUIRED is distinct from an eligibility grant. Do not turn any unknown
    # or malformed state into an ELIGIBLE result through truthiness/coercion.
    eligible = decision == "ELIGIBLE"
    if not eligible and decision not in {"BLOCKED", "NOT_REQUIRED"}:
        decision = "BLOCKED"
        reasons.append("Unrecognized decision state; defaulting safely to blocked.")

    return {
        "decision_status": decision,
        "eligible": eligible,
        "active_model_version": active_version if isinstance(active_version, str) else None,
        "drift_status": normalized_drift or raw_drift_status,
        "label_readiness_status": "READY" if label_ready else (
            "BLOCKED" if label_summary_status == "AVAILABLE" else (label_summary_status or "UNKNOWN")
        ),
        "label_summary_status": label_summary_status,
        "reasons": list(dict.fromkeys(reasons)),
        "adapter_reasons": adapter_reasons,
        "warnings": list(dict.fromkeys(warnings)),
        "rule_version": policy.rule_version,
        "configuration": {
            "minimum_genuine_labels": policy.minimum_genuine_labels,
            "minimum_label_floor_basis": "academic demonstration setting; requires validation, not a production threshold",
        },
        "evaluated_at_utc": timestamp,
    }


def evaluate_retraining_eligibility(*, policy: EligibilityPolicy = EligibilityPolicy()) -> dict[str, Any]:
    """Read current summaries through Stage 2 adapters and return a decision."""
    active = model_adapter.get_active_model_summary()
    drift = drift_adapter.get_latest_drift_summary()
    labels = label_adapter.get_label_readiness_summary()
    return evaluate_summaries(active, drift, labels, policy=policy)


__all__ = [
    "DEFAULT_MINIMUM_GENUINE_LABELS",
    "RULE_VERSION",
    "EligibilityPolicy",
    "evaluate_summaries",
    "evaluate_retraining_eligibility",
]
