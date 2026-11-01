"""Manually process monitoring retraining requests into evaluated candidates.

This worker never promotes a candidate. A request is an investigation signal;
Stage 4's trusted-label eligibility gates remain authoritative.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.labels import config as labels_config
from src.monitoring import config as monitor_config
from src.retraining.candidate_evaluator import _data_fingerprint
from src.retraining import v2_config
from src.retraining.stage4_workflow import run_stage4
from src.retraining.stage5_workflow import evaluate_stage5
from src.retraining.v2_trainer import FINAL_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_ALERTS = frozenset({"feature_drift", "performance_degradation"})


def _connect(database_path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    path = Path(database_path).resolve()
    if readonly:
        if not path.is_file():
            raise FileNotFoundError(f"Monitoring database does not exist: {path}")
        db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=10)
        db.execute("PRAGMA query_only=ON")
        return db
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path, timeout=10, isolation_level=None)


def _alert_types(trigger_reason: str) -> list[str]:
    """Return supported alert names found in the monitoring reason string."""
    return [name for name in sorted(SUPPORTED_ALERTS)
            if name in str(trigger_reason).split(" (", 1)[0].split(", ")]


def inspect_requests(database_path: Path, *, limit: int = 20) -> list[dict[str, Any]]:
    """Read requests without claiming or changing any records."""
    if limit < 1:
        raise ValueError("limit must be positive")
    with closing(_connect(database_path, readonly=True)) as db:
        rows = db.execute("""SELECT request_id, status, model_version, trigger_reason,
                    alert_ids_json, evidence_json, created_at
                FROM retraining_requests
                WHERE status IN ('requested', 'processing')
                ORDER BY created_at, request_id LIMIT ?""", (limit,)).fetchall()
    result = []
    for row in rows:
        request_id, status, version, reason, alert_ids, evidence, created_at = row
        kinds = _alert_types(reason)
        result.append({"request_id": request_id, "status": status, "model_version": version,
            "alert_types": kinds, "supported": bool(kinds), "trigger_reason": reason,
            "alert_ids": json.loads(alert_ids), "evidence": json.loads(evidence),
            "created_at": created_at,
            "needs_manual_review": status == "processing"})
    return result


def _claim_next(database_path: Path) -> dict[str, Any] | None:
    """Atomically claim the oldest request; concurrent workers cannot claim it twice."""
    token = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    with closing(_connect(database_path)) as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("""SELECT request_id, model_version, trigger_reason, alert_ids_json,
                    evidence_json, created_at FROM retraining_requests
                WHERE status='requested' ORDER BY created_at, request_id LIMIT 1""").fetchone()
        if row is None:
            db.commit()
            return None
        request_id, version, reason, alert_ids, evidence_text, created_at = row
        evidence = json.loads(evidence_text)
        evidence["worker"] = {"claim_token": token, "claimed_at": now,
                              "stage": "claimed", "attempt": "single_attempt"}
        updated = json.dumps(evidence, sort_keys=True)
        cursor = db.execute("""UPDATE retraining_requests SET status='processing', evidence_json=?
            WHERE request_id=? AND status='requested'""", (updated, request_id))
        if cursor.rowcount != 1:
            db.rollback()
            return None
        db.commit()
    return {"request_id": request_id, "model_version": version, "trigger_reason": reason,
        "alert_ids": json.loads(alert_ids), "evidence": evidence, "evidence_text": updated,
        "created_at": created_at, "claim_token": token,
        "alert_types": _alert_types(reason)}


def _set_stage(database_path: Path, request: dict[str, Any], stage: str) -> None:
    evidence = dict(request["evidence"])
    worker = dict(evidence.get("worker", {}))
    if worker.get("claim_token") != request["claim_token"]:
        raise RuntimeError("Request claim token changed; refusing to continue.")
    worker["stage"] = stage
    worker["updated_at"] = datetime.now(timezone.utc).isoformat()
    evidence["worker"] = worker
    updated = json.dumps(evidence, sort_keys=True)
    with closing(_connect(database_path)) as db:
        db.execute("BEGIN IMMEDIATE")
        cursor = db.execute("""UPDATE retraining_requests SET evidence_json=?
            WHERE request_id=? AND status='processing' AND evidence_json=?""",
            (updated, request["request_id"], request["evidence_text"]))
        if cursor.rowcount != 1:
            db.rollback()
            raise RuntimeError("Request is no longer owned by this worker.")
        db.commit()
    request["evidence"] = evidence
    request["evidence_text"] = updated


def _finish(database_path: Path, request: dict[str, Any], status: str,
            outcome: dict[str, Any]) -> None:
    evidence = dict(request["evidence"])
    worker = dict(evidence.get("worker", {}))
    if worker.get("claim_token") != request["claim_token"]:
        raise RuntimeError("Request claim token changed; refusing to finalize.")
    worker.update({"stage": "finished", "finished_at": datetime.now(timezone.utc).isoformat(),
                  "outcome": outcome})
    evidence["worker"] = worker
    updated = json.dumps(evidence, sort_keys=True, allow_nan=False)
    with closing(_connect(database_path)) as db:
        db.execute("BEGIN IMMEDIATE")
        cursor = db.execute("""UPDATE retraining_requests SET status=?, evidence_json=?
            WHERE request_id=? AND status='processing' AND evidence_json=?""",
            (status, updated, request["request_id"], request["evidence_text"]))
        if cursor.rowcount != 1:
            db.rollback()
            raise RuntimeError("Request is no longer owned by this worker; outcome was not committed.")
        db.commit()


def _process_claimed(request: dict[str, Any], *, database_path: Path, work_root: Path,
                     held_out: pd.DataFrame, evaluation_details: dict[str, Any],
                     stage4_runner: Callable[..., dict[str, Any]],
                     stage5_evaluator: Callable[..., dict[str, Any]]) -> dict[str, Any]:
    supported = request["alert_types"]
    if not supported:
        result = {"reason": "Unsupported alert type; no training was run."}
        _finish(database_path, request, "blocked_unsupported_alert", result)
        return {"request_id": request["request_id"], "status": "blocked_unsupported_alert", **result}
    if request["model_version"] != "V2":
        result = {"reason": "Stage 4 supports V2 requests only; no training was run."}
        _finish(database_path, request, "blocked_unsupported_model", result)
        return {"request_id": request["request_id"], "status": "blocked_unsupported_model", **result}
    safe_id = hashlib.sha256(request["request_id"].encode()).hexdigest()
    request_dir = Path(work_root).resolve() / safe_id
    candidate_dir = request_dir / "candidate"
    preflight_report = request_dir / "stage4_eligibility_report.json"
    stage4_report = request_dir / "stage4_report.json"
    stage5_report = request_dir / "stage5_evaluation.json"
    if candidate_dir.exists() or preflight_report.exists() or stage4_report.exists() or stage5_report.exists():
        result = {"reason": "Request output already exists; refusing duplicate candidate processing.",
                  "output_dir": str(request_dir)}
        _finish(database_path, request, "failed_existing_output", result)
        return {"request_id": request["request_id"], "status": "failed_existing_output", **result}

    try:
        _set_stage(database_path, request, "stage4_eligibility_check")
        eligibility_report = stage4_runner(label_db=labels_config.STAGE2_DB_PATH,
            prediction_log=monitor_config.STAGE1_JSONL_PATH,
            historical_dataset=v2_config.V2_DATASET_PATH, report_path=preflight_report,
            candidate_dir=candidate_dir, train_candidate=False)
    except Exception as exc:
        result = {"reason": f"Stage 4 eligibility check failed: {type(exc).__name__}: {exc}",
                  "alert_types": supported, "eligibility_report": str(preflight_report)}
        _finish(database_path, request, "failed_eligibility_check", result)
        return {"request_id": request["request_id"], "status": "failed_eligibility_check", **result}

    if eligibility_report.get("status") == "insufficient_eligible_labels":
        result = {"reason": eligibility_report.get("explanation", "Stage 4 eligibility requirements were not met."),
                  "alert_types": supported, "eligibility_report": str(preflight_report),
                  "eligibility": eligibility_report.get("eligibility")}
        _finish(database_path, request, "blocked_eligibility", result)
        return {"request_id": request["request_id"], "status": "blocked_eligibility", **result}
    if eligibility_report.get("status") != "eligible_training_not_requested":
        result = {"reason": eligibility_report.get("explanation", "Unexpected Stage 4 eligibility status."),
                  "alert_types": supported, "stage4_status": eligibility_report.get("status"),
                  "eligibility_report": str(preflight_report)}
        _finish(database_path, request, "failed_eligibility_check", result)
        return {"request_id": request["request_id"], "status": "failed_eligibility_check", **result}
    try:
        _set_stage(database_path, request, "stage4_training_and_internal_evaluation")
        stage4 = stage4_runner(label_db=labels_config.STAGE2_DB_PATH,
            prediction_log=monitor_config.STAGE1_JSONL_PATH,
            historical_dataset=v2_config.V2_DATASET_PATH, report_path=stage4_report,
            candidate_dir=candidate_dir, train_candidate=True)
    except Exception as exc:
        result = {"reason": f"Stage 4 failed: {type(exc).__name__}: {exc}",
                  "alert_types": supported, "stage4_report": str(stage4_report),
                  "eligibility_report": str(preflight_report)}
        _finish(database_path, request, "failed_training", result)
        return {"request_id": request["request_id"], "status": "failed_training", **result}

    stage4_status = stage4.get("status")
    if stage4_status != "candidate_trained_not_promoted":
        result = {"reason": stage4.get("explanation", f"Unexpected Stage 4 status: {stage4_status}"),
                  "alert_types": supported, "stage4_status": stage4_status,
                  "stage4_report": str(stage4_report)}
        _finish(database_path, request, "failed_training", result)
        return {"request_id": request["request_id"], "status": "failed_training", **result}

    try:
        provenance = dict(evaluation_details)
        training_fingerprint = stage4.get("dataset_fingerprint")
        if not training_fingerprint:
            raise ValueError("Stage 4 report lacks the candidate training fingerprint.")
        supplied_fingerprint = provenance.get("candidate_training_fingerprint")
        if supplied_fingerprint and supplied_fingerprint != training_fingerprint:
            raise ValueError("Evaluation provenance training fingerprint does not match Stage 4.")
        provenance["candidate_training_fingerprint"] = training_fingerprint
        _set_stage(database_path, request, "stage5_candidate_evaluation")
        evaluation = stage5_evaluator(candidate_dir=candidate_dir,
            held_out=held_out, report_path=stage5_report,
            evaluation_provenance=provenance, trusted_evaluation=True,
            project_root=PROJECT_ROOT)
    except Exception as exc:
        result = {"reason": f"Stage 5 evaluation failed: {type(exc).__name__}: {exc}",
                  "alert_types": supported, "stage4_report": str(stage4_report),
                  "stage5_report": str(stage5_report), "candidate_dir": str(candidate_dir)}
        _finish(database_path, request, "failed_evaluation", result)
        return {"request_id": request["request_id"], "status": "failed_evaluation", **result}

    evaluation_status = evaluation.get("status")
    if evaluation_status == "candidate_approved_pending_authorization":
        status = "evaluation_approved_pending_authorization"
    elif evaluation_status == "candidate_rejected":
        status = "candidate_rejected"
    else:
        status = "evaluation_blocked"
    result = {"alert_types": supported, "stage4_report": str(stage4_report),
        "stage5_report": str(stage5_report), "candidate_dir": str(candidate_dir),
        "stage4_status": stage4_status, "stage5_status": evaluation_status,
        "stage5_reason": evaluation.get("reason"),
        "deployment": "not attempted; Stage 5 explicit authorization remains required"}
    _finish(database_path, request, status, result)
    return {"request_id": request["request_id"], "status": status, **result}


def process_requests(*, database_path: Path, work_root: Path, evaluation_data: Path,
                     evaluation_provenance: Path, trusted_evaluation: bool = False,
                     limit: int = 1, stage4_runner: Callable[..., dict[str, Any]] = run_stage4,
                     stage5_evaluator: Callable[..., dict[str, Any]] = evaluate_stage5) -> list[dict[str, Any]]:
    """Manually process at most ``limit`` new requests; requests are never auto-retried."""
    if limit < 1:
        raise ValueError("limit must be positive")
    if not Path(evaluation_data).is_file() or not Path(evaluation_provenance).is_file():
        raise FileNotFoundError("Stage 5 evaluation data or provenance file does not exist.")
    if not trusted_evaluation:
        raise ValueError("Pass trusted_evaluation=True only after verifying the evaluation source.")
    evaluation_details = json.loads(Path(evaluation_provenance).read_text(encoding="utf-8"))
    if not isinstance(evaluation_details, dict):
        raise ValueError("Evaluation provenance must be a JSON object.")
    if evaluation_details.get("synthetic") is True:
        raise ValueError("Synthetic evaluation data cannot authorize candidate processing.")
    if evaluation_details.get("synthetic") is not False:
        raise ValueError("Evaluation provenance must explicitly identify non-synthetic data.")
    required_provenance = ("source", "label_provenance")
    missing_provenance = [key for key in required_provenance if not evaluation_details.get(key)]
    if missing_provenance:
        raise ValueError(f"Evaluation provenance is missing: {missing_provenance}")
    held_out = pd.read_parquet(evaluation_data)
    try:
        evaluation_fingerprint = _data_fingerprint(held_out, FINAL_FEATURES)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Evaluation data does not match the Stage 5 feature/label contract: {exc}") from exc
    supplied_fingerprint = evaluation_details.get("evaluation_fingerprint")
    if supplied_fingerprint and supplied_fingerprint != evaluation_fingerprint:
        raise ValueError("Evaluation provenance fingerprint does not match the supplied held-out data.")
    evaluation_details["evaluation_fingerprint"] = evaluation_fingerprint
    results = []
    for _ in range(limit):
        request = _claim_next(database_path)
        if request is None:
            break
        results.append(_process_claimed(request, database_path=database_path,
            work_root=work_root, held_out=held_out,
            evaluation_details=evaluation_details,
            stage4_runner=stage4_runner, stage5_evaluator=stage5_evaluator))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="read and report pending requests only (default)")
    mode.add_argument("--process", action="store_true", help="claim requests and build/evaluate isolated candidates")
    parser.add_argument("--database", type=Path, default=monitor_config.MONITOR_DB_PATH)
    parser.add_argument("--work-root", type=Path,
        default=PROJECT_ROOT / "data" / "stage3" / "request_worker_runs")
    parser.add_argument("--evaluation-data", type=Path,
        help="trusted held-out parquet required with --process")
    parser.add_argument("--evaluation-provenance", type=Path,
        help="trusted evaluation provenance JSON required with --process")
    parser.add_argument("--trusted-evaluation", action="store_true",
        help="explicitly attest that evaluation provenance is trusted and non-synthetic")
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    if not args.process:
        result = inspect_requests(args.database, limit=args.limit)
        print(json.dumps({"mode": "dry_run", "requests": result}, indent=2, allow_nan=False))
        return
    if not args.evaluation_data or not args.evaluation_provenance:
        parser.error("--process requires --evaluation-data and --evaluation-provenance")
    result = process_requests(database_path=args.database, work_root=args.work_root,
        evaluation_data=args.evaluation_data, evaluation_provenance=args.evaluation_provenance,
        trusted_evaluation=args.trusted_evaluation, limit=args.limit)
    print(json.dumps({"mode": "process", "results": result}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
