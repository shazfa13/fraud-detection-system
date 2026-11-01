"""Read-only summary of Stage 2 label-store provenance and readiness."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.labels import config as label_config
from src.labels.store import get_stats
from src.retraining.lifecycle_coordinator import LabelReadiness


def get_label_readiness_summary(*, database_path: str | Path | None = None,
                                phase9_report_path: str | Path | None = None) -> dict[str, Any]:
    """Read label counts/provenance without registering or changing any records."""
    db_path = Path(database_path) if database_path is not None else label_config.STAGE2_DB_PATH
    phase9_path = (Path(phase9_report_path) if phase9_report_path is not None else
                   Path(__file__).resolve().parents[2] / "reports" / "recent_label_collection_report.json")
    trusted_sources = sorted(label_config.trusted_label_sources())
    token_configured = bool(label_config.trusted_label_token())
    reasons: list[str] = []
    phase9_type: str | None = None
    try:
        phase9 = json.loads(phase9_path.read_text(encoding="utf-8"))
        if isinstance(phase9, dict):
            phase9_type = phase9.get("ground_truth_label_type")
    except (OSError, UnicodeError, json.JSONDecodeError):
        pass

    base: dict[str, Any] = {
        "status": "UNAVAILABLE",
        "database_path": str(db_path),
        "predictions_registered": None,
        "total_labels": None,
        "matched_labels": None,
        "unmatched_labels": None,
        "label_status_counts": None,
        "source_type_counts": None,
        "label_source_counts": None,
        "conflict_count": None,
        "unresolved_or_rejected_count": None,
        "trusted_source_configuration": {
            "configured_sources": trusted_sources,
            "authorization_token_configured": token_configured,
            "configuration_is_evidence_of_truth": False,
        },
        "phase9_label_type": phase9_type,
        "genuinely_verified_labels": None,
        "simulated_labels": None,
        "matching_key": "transaction_id",
        "lifecycle_required_matching_key": "event_id",
        "strict_retraining_ready": False,
        "readiness_reasons": reasons,
    }
    if not db_path.is_file():
        reasons.append("Stage 2 label database is unavailable.")
        if phase9_type == "SIMULATED_DELAYED_GROUND_TRUTH":
            reasons.append("Phase 9 PaySim labels are simulated and cannot satisfy genuine-label readiness.")
            base["simulated_labels"] = "PHASE9_PAYSIM_SIMULATED"
        return base

    try:
        stats = get_stats(db_path)
        uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=3) as db:
            db.execute("PRAGMA query_only=ON")
            total_labels = int(db.execute("SELECT COUNT(*) FROM labels").fetchone()[0])
            matched = int(db.execute(
                "SELECT COUNT(*) FROM labels l JOIN predictions p USING(transaction_id)"
            ).fetchone()[0])
            unmatched = total_labels - matched
            label_status_counts = {str(row[0]): int(row[1]) for row in db.execute(
                "SELECT status, COUNT(*) FROM labels GROUP BY status"
            )}
            source_type_counts = {str(row[0]): int(row[1]) for row in db.execute(
                "SELECT source_type, COUNT(*) FROM labels GROUP BY source_type"
            )}
            label_source_counts = {f"{row[0]}:{row[1]}": int(row[2]) for row in db.execute(
                "SELECT source_type, label_source, COUNT(*) FROM labels GROUP BY source_type,label_source"
            )}
        genuine_count = int(stats["confirmed_genuine_labels"])
        simulated_count = int(stats["synthetic_test_labels"])
        conflict_count = int(stats["conflicting_labels"] + stats["prediction_conflicts"])
        unresolved = int(label_status_counts.get("pending", 0) + label_status_counts.get("rejected", 0))
        base.update({
            "status": "AVAILABLE",
            "predictions_registered": int(stats["predictions_registered"]),
            "total_labels": total_labels,
            "matched_labels": matched,
            "unmatched_labels": unmatched,
            "label_status_counts": label_status_counts,
            "source_type_counts": source_type_counts,
            "label_source_counts": label_source_counts,
            "conflict_count": conflict_count,
            "unresolved_or_rejected_count": unresolved,
            "genuinely_verified_labels": genuine_count,
            "simulated_labels": simulated_count,
        })
    except Exception as exc:
        reasons.append(f"Label store could not be read safely: {type(exc).__name__}: {exc}")
        return base

    if not trusted_sources:
        reasons.append("No trusted label source is configured.")
    if not token_configured:
        reasons.append("Trusted-source authorization is not configured.")
    if base["genuinely_verified_labels"] == 0:
        reasons.append("No resolved, matched, non-synthetic trusted_verified labels are available.")
    if base["unmatched_labels"]:
        reasons.append("Some labels do not match a registered prediction by transaction_id.")
    if conflict_count:
        reasons.append("Label or prediction conflicts are present.")
    if phase9_type == "SIMULATED_DELAYED_GROUND_TRUTH":
        reasons.append("Phase 9 PaySim labels are simulated and cannot satisfy genuine-label readiness.")
    else:
        reasons.append("Phase 9 label provenance is unavailable or is not explicitly identified as simulated.")

    lifecycle_readiness = LabelReadiness(
        source_name=",".join(trusted_sources),
        source_verified=bool(base["genuinely_verified_labels"]),
        matched_label_count=int(base["genuinely_verified_labels"] or 0),
        unmatched_record_count=int(base["unmatched_labels"] or 0),
        matching_key="transaction_id",
        provenance="Stage 2 store trusted_verified/resolved rows; mapping to Phase 10 event_id is not established.",
    )
    if not lifecycle_readiness.is_reliable():
        reasons.append("Existing lifecycle readiness requires event_id matching; Stage 2 records use transaction_id and no event_id mapping is established.")
    ready = (
        lifecycle_readiness.is_reliable()
        and bool(trusted_sources)
        and token_configured
        and not conflict_count
        and phase9_type != "SIMULATED_DELAYED_GROUND_TRUTH"
    )
    base["strict_retraining_ready"] = ready
    base["readiness_reasons"] = list(dict.fromkeys(reasons))
    return base


__all__ = ["get_label_readiness_summary"]
