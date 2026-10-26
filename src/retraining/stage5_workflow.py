"""Stage 5 candidate evaluation and explicitly authorized manifest promotion.

This module never trains a model. Reports and promotion records are append/new-file
operations; active selection is a single atomic manifest replacement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.retraining.active_model import ActiveModelConfigurationError, resolve_active_model
from src.retraining.candidate_evaluator import AcceptancePolicy, evaluate_candidate
from src.retraining.model_activator import validate_bundle

MIN_EVALUATION_ROWS = 100
MIN_FRAUD_COUNT = 10
MIN_FRAUD_RECALL = 0.80
MAX_REGRESSION = 0.02
IMPORTANT_METRICS = ("precision", "recall", "f1", "pr_auc")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically create a report without replacing an existing report."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_candidate(candidate_dir: str | Path, *, project_root: str | Path | None = None,
                       model_loader=None) -> dict[str, Any]:
    """Validate Stage 4 bundle completeness, provenance fields and hashes."""
    candidate = Path(candidate_dir).resolve()
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
    if not candidate.is_dir() or not any(candidate.iterdir()):
        return {"status": "candidate_unavailable", "candidate_dir": str(candidate)}
    features = list(resolve_active_model(project_root=root).feature_names)
    bundle = validate_bundle(candidate, features)
    if model_loader is None:
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model(str(candidate / "model.json"))
    else:
        model = model_loader(candidate / "model.json")
    metadata = _json(candidate / "metadata.json")
    threshold = _json(candidate / "threshold.json").get("selected_threshold")
    if (not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
            or not math.isfinite(float(threshold)) or not 0 < float(threshold) < 1):
        raise ValueError("Candidate threshold must be finite and strictly between 0 and 1.")
    required = ("dataset_fingerprint", "label_provenance_summary", "temporal_split",
                "training_configuration", "training_seconds", "test_metrics")
    missing = [key for key in required if key not in metadata or metadata[key] is None]
    if missing:
        raise ValueError(f"Candidate metadata is missing Stage 4 provenance fields: {missing}")
    split = metadata.get("temporal_split")
    if (not isinstance(split, dict) or "train_end_step" not in split
            or "validation_end_step" not in split or "test_end_step" not in split
            or int(split["train_end_step"]) >= int(split["validation_end_step"])
            or int(split["validation_end_step"]) > int(split["test_end_step"])):
        raise ValueError("Candidate metadata must record ordered train, validation, and test step boundaries.")
    if metadata.get("target") != "isFraud":
        raise ValueError("Candidate metadata target must be isFraud.")
    paths = {name: candidate / name for name in ("model.json", "metadata.json", "threshold.json")}
    hashes = {name: _sha256(path) for name, path in paths.items()}
    recorded = metadata.get("artifact_sha256")
    if recorded is not None and recorded != hashes:
        raise ValueError("Candidate artifact hashes do not match metadata.")
    return {"status": "validated", "candidate_dir": str(candidate), "model_version": bundle.model_version,
            "threshold": bundle.threshold, "feature_names": list(features), "artifact_sha256": hashes,
            "dataset_fingerprint": metadata["dataset_fingerprint"],
            "label_provenance_summary": metadata["label_provenance_summary"],
            "temporal_split": metadata["temporal_split"]}


def evaluate_stage5(candidate_dir: str | Path, held_out: pd.DataFrame, report_path: str | Path, *,
                    evaluation_provenance: dict[str, Any], trusted_evaluation: bool = False,
                    model_loader=None, project_root: str | Path | None = None) -> dict[str, Any]:
    """Evaluate same-row candidate/active predictions; block untrusted/small cohorts."""
    candidate = Path(candidate_dir).resolve()
    report_path = Path(report_path)
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation report: {report_path}")
    report = {"stage": "Stage 5 candidate evaluation", "generated_at": datetime.now(timezone.utc).isoformat(),
              "status": "evaluation_blocked", "candidate": None, "evaluation_provenance": evaluation_provenance,
              "acceptance_policy": {"minimum_rows": MIN_EVALUATION_ROWS, "minimum_fraud_count": MIN_FRAUD_COUNT,
                                    "minimum_fraud_recall": MIN_FRAUD_RECALL,
                                    "maximum_regression": MAX_REGRESSION, "provisional": True}}
    try:
        checked = validate_candidate(candidate, project_root=project_root, model_loader=model_loader)
        report["candidate"] = checked
        if checked["status"] != "validated":
            report["status"] = "candidate_unavailable"
        elif (not trusted_evaluation or not evaluation_provenance.get("source")
              or not evaluation_provenance.get("label_provenance")
              or not evaluation_provenance.get("candidate_training_fingerprint")
              or not evaluation_provenance.get("evaluation_fingerprint")):
            report["reason"] = "Evaluation cohort must have explicit trusted provenance; synthetic fixtures are not production evidence."
        elif "transaction_id" not in held_out or not held_out["transaction_id"].is_unique:
            report["reason"] = "Evaluation rows must have distinct source transaction_id values."
        elif evaluation_provenance.get("synthetic") is True:
            report["reason"] = "Synthetic evaluation data cannot support a candidate approval decision."
        elif evaluation_provenance.get("candidate_training_fingerprint") == evaluation_provenance.get("evaluation_fingerprint"):
            report["reason"] = "Evaluation cohort fingerprint overlaps the candidate training dataset."
        elif evaluation_provenance.get("candidate_training_fingerprint") != checked.get("dataset_fingerprint"):
            report["reason"] = "Evaluation provenance training fingerprint does not match the candidate metadata."
        elif ("step" not in held_out or not pd.api.types.is_numeric_dtype(held_out["step"])
              or not held_out["step"].notna().all()):
            report["reason"] = "Evaluation rows require finite numeric step values for temporal leakage checks."
        elif int(held_out["step"].min()) <= int(checked["temporal_split"].get("test_end_step", -1)):
            report["reason"] = "Evaluation rows overlap the candidate training, threshold-selection, or reserved test window."
        elif len(held_out) < MIN_EVALUATION_ROWS or int(held_out.get("isFraud", pd.Series(dtype=int)).sum()) < MIN_FRAUD_COUNT:
            report["reason"] = "Evaluation cohort is below provisional row/fraud-count minimums."
        else:
            active = resolve_active_model(project_root=project_root)
            with tempfile.TemporaryDirectory(prefix="stage5-active-bundle-") as temporary_dir, \
                    tempfile.TemporaryDirectory(prefix="stage5-evaluator-report-") as report_temp:
                active_dir = Path(temporary_dir)
                shutil.copyfile(active.model_path, active_dir / "model.json")
                shutil.copyfile(active.metadata_path, active_dir / "metadata.json")
                shutil.copyfile(active.threshold_path, active_dir / "threshold.json")
                base = evaluate_candidate(candidate, active_dir, held_out, Path(report_temp) / "metrics.json",
                    policy=AcceptancePolicy({name: -MAX_REGRESSION for name in IMPORTANT_METRICS}),
                    model_loader=model_loader)
            # Existing evaluator's report is the canonical detailed metric report. Extend only
            # Stage 5 status here; callers should keep this path as the report reference.
            if not base.get("evaluation_completed"):
                report.update({"status": "evaluation_blocked", "reason": base.get("error"), "evaluation": base})
            elif evaluation_provenance["evaluation_fingerprint"] != base["data"]["records_sha256"]:
                report.update({"status": "evaluation_blocked", "reason": "Evaluation provenance fingerprint does not match supplied held-out rows.",
                               "evaluation": base})
            else:
                metrics = base["metrics"]["candidate"]
                passed = (base["decision_status"] == "ACCEPTED" and metrics["recall"] >= MIN_FRAUD_RECALL)
                report.update({"status": "candidate_approved_pending_authorization" if passed else "candidate_rejected",
                    "evaluation": base, "evaluation_report": str(report_path),
                    "rows": len(held_out), "fraud_count": int(held_out["isFraud"].sum()),
                    "class_distribution": held_out["isFraud"].value_counts().to_dict(),
                    "reason": None if passed else "Acceptance criteria failed."})
    except Exception as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"
    _write_new_json(report_path, report)
    return report


def _manifest_for(bundle: dict[str, Any], root: Path) -> dict[str, Any]:
    paths = {"model": bundle.model_path, "metadata": bundle.metadata_path, "threshold": bundle.threshold_path}
    return {"schema_version": 1, "model_version": bundle.model_version,
            "artifacts": {key: path.resolve().relative_to(root).as_posix() for key, path in paths.items()},
            "artifact_sha256": dict(bundle.artifact_sha256), "activated_at": datetime.now(timezone.utc).isoformat()}


def promote(*, candidate_dir: str | Path, evaluation_report: str | Path, approval_file: str | Path,
            project_root: str | Path | None = None, dry_run: bool = True) -> dict[str, Any]:
    """Promote only with approval bound to exact candidate hashes and report bytes."""
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
    manifest_path = root / "models" / "active_model_manifest.json"
    result = {"status": "promotion_blocked", "dry_run": dry_run}
    try:
        candidate = validate_candidate(candidate_dir, project_root=root)
        if candidate["status"] != "validated":
            raise ValueError("Candidate is unavailable.")
        candidate_metadata = _json(Path(candidate_dir) / "metadata.json")
        if (candidate_metadata.get("training_mode") == "demonstration"
                or candidate_metadata.get("eligible_for_genuine_promotion") is False):
            raise ValueError("Demonstration candidates are ineligible for genuine production promotion.")
        report_path, approval_path = Path(evaluation_report).resolve(), Path(approval_file).resolve()
        report = _json(report_path); approval = _json(approval_path)
        if report.get("status") != "candidate_approved_pending_authorization":
            raise ValueError("Evaluation report is not approved pending authorization.")
        expected = {"candidate_version": candidate["model_version"],
                    "candidate_artifact_sha256": candidate["artifact_sha256"],
                    "evaluation_report_sha256": _sha256(report_path),
                    "dataset_fingerprint": candidate["dataset_fingerprint"]}
        for key, value in expected.items():
            if approval.get(key) != value:
                raise ValueError(f"Approval does not match {key}.")
        if not approval.get("approved_by") or approval.get("approval_statement") != "I approve promotion of this exact candidate and evaluation report.":
            raise ValueError("Approval requires approver identity and the exact approval statement.")
        approved_at = datetime.fromisoformat(str(approval.get("approved_at", "")).replace("Z", "+00:00"))
        if approved_at.tzinfo is None or approved_at.utcoffset() is None:
            raise ValueError("Approval timestamp must be timezone-aware ISO-8601.")
        report_candidate = report.get("candidate")
        if (not isinstance(report_candidate, dict)
                or report_candidate.get("model_version") != candidate["model_version"]
                or report_candidate.get("artifact_sha256") != candidate["artifact_sha256"]
                or report_candidate.get("dataset_fingerprint") != candidate["dataset_fingerprint"]):
            raise ValueError("Evaluation report candidate identity/fingerprint does not match current artifacts.")
        current = resolve_active_model(project_root=root)
        if current.model_version == candidate["model_version"] and dict(current.artifact_sha256) == candidate["artifact_sha256"]:
            raise ValueError("Candidate is already the active model; repeated promotion is rejected.")
        approval_hash = _sha256(approval_path)
        if any(_json(record).get("approval_file_sha256") == approval_hash
               for record in (root / "models" / "promotion_history").glob("*/promotion_record.json")):
            raise ValueError("This exact approval has already been used for a promotion attempt.")
        previous_manifest = _manifest_for(current, root)
        new_manifest = {"schema_version": 1, "model_version": candidate["model_version"],
            "artifacts": {"model": (Path(candidate_dir).resolve()/"model.json").relative_to(root).as_posix(),
                          "metadata": (Path(candidate_dir).resolve()/"metadata.json").relative_to(root).as_posix(),
                          "threshold": (Path(candidate_dir).resolve()/"threshold.json").relative_to(root).as_posix()},
            "artifact_sha256": candidate["artifact_sha256"], "activated_at": datetime.now(timezone.utc).isoformat()}
        _validate_temp_manifest(root, new_manifest)
        result.update({"status": "promotion_ready", "previous_model_version": current.model_version,
                       "candidate_model_version": candidate["model_version"], "candidate_artifact_sha256": candidate["artifact_sha256"],
                       "evaluation_report_sha256": expected["evaluation_report_sha256"]})
        if dry_run:
            return result
        backup_dir = root / "models" / "promotion_history" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_dir.mkdir(parents=True, exist_ok=False)
        backup_path = backup_dir / "previous_manifest.json"
        _atomic_json(backup_path, previous_manifest)
        try:
            _atomic_json(manifest_path, new_manifest)
            resolved = resolve_active_model(project_root=root)
            if resolved.model_version != candidate["model_version"] or resolved.artifact_sha256 != candidate["artifact_sha256"]:
                raise ValueError("Post-activation resolution did not match the candidate.")
        except Exception as activation_error:
            _atomic_json(manifest_path, previous_manifest)
            failed_record = {**result, "status": "activation_failed", "dry_run": False,
                "backup_manifest": str(backup_path), "approval_file": str(approval_path),
                "approval_file_sha256": approval_hash, "evaluation_report": str(report_path),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "reason": f"{type(activation_error).__name__}: {activation_error}"}
            _atomic_json(backup_dir / "promotion_record.json", failed_record)
            return failed_record
        record = {**result, "status": "promoted", "dry_run": False, "backup_manifest": str(backup_path),
                  "approval_file": str(approval_path), "evaluation_report": str(report_path),
                  "approval_file_sha256": approval_hash,
                  "promoted_manifest_sha256": _sha256(manifest_path), "completed_at": datetime.now(timezone.utc).isoformat()}
        _atomic_json(backup_dir / "promotion_record.json", record)
        return record
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
        return result


def _validate_temp_manifest(root: Path, value: dict[str, Any]) -> None:
    """Validate proposed resolution using a temporary manifest that is always removed."""
    path = root / "models" / f".stage5-validation-{os.getpid()}.json"
    try:
        _atomic_json(path, value)
        resolve_active_model(project_root=root, manifest_path=path)
    finally:
        path.unlink(missing_ok=True)


def rollback(*, project_root: str | Path | None = None, dry_run: bool = True) -> dict[str, Any]:
    """Restore the latest promoted manifest once, retaining both audit references."""
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
    manifest_path = root / "models" / "active_model_manifest.json"
    result = {"status": "rollback_blocked", "dry_run": dry_run}
    try:
        records = sorted((root / "models" / "promotion_history").glob("*/promotion_record.json"), reverse=True)
        record_path = next((p for p in records if _json(p).get("status") == "promoted" and not _json(p).get("rolled_back_at")), None)
        if record_path is None or not manifest_path.is_file():
            raise ValueError("No unrolled promotion record and active manifest were found.")
        record = _json(record_path)
        if _sha256(manifest_path) != record.get("promoted_manifest_sha256"):
            raise ValueError("Active manifest changed after promotion; refusing stale rollback.")
        backup = Path(record["backup_manifest"])
        prior = _json(backup)
        resolved_before = resolve_active_model(project_root=root)
        _validate_temp_manifest(root, prior)
        result.update({"status": "rollback_ready", "current_model_version": resolved_before.model_version,
                       "restore_model_version": prior["model_version"]})
        if dry_run:
            return result
        _atomic_json(manifest_path, prior)
        restored = resolve_active_model(project_root=root)
        if restored.model_version != prior["model_version"]:
            raise ValueError("Restored active model failed validation.")
        record["rolled_back_at"] = datetime.now(timezone.utc).isoformat()
        record["rollback_model_version"] = restored.model_version
        _atomic_json(record_path, record)
        result["status"] = "rolled_back"
        return result
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status", help="read-only candidate inventory; never trains or promotes")
    status.add_argument("--candidate", type=Path, default=Path("models/candidates"))
    ev = sub.add_parser("evaluate")
    ev.add_argument("--candidate", type=Path, required=True)
    ev.add_argument("--data", type=Path, required=True)
    ev.add_argument("--provenance", type=Path, required=True)
    ev.add_argument("--report", type=Path, required=True)
    ev.add_argument("--trusted-evaluation", action="store_true")
    pr = sub.add_parser("promote")
    pr.add_argument("--candidate", type=Path, required=True)
    pr.add_argument("--evaluation-report", type=Path, required=True)
    pr.add_argument("--approval", type=Path, required=True)
    pr.add_argument("--execute", action="store_true", help="execute only after all bound authorization checks")
    rb = sub.add_parser("rollback")
    rb.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "status":
        candidate = args.candidate
        result = {"stage": "Stage 5", "status": "candidate_unavailable",
                  "candidate_root": str(candidate.resolve()), "training_started": False,
                  "candidate_directories": [str(path) for path in candidate.iterdir() if path.is_dir()]
                  if candidate.is_dir() else [],
                  "active_model": resolve_active_model().model_version,
                  "threshold": resolve_active_model().threshold}
        if result["candidate_directories"]:
            result["status"] = "candidate_artifacts_present_not_evaluated"
    elif args.command == "evaluate":
        result = evaluate_stage5(args.candidate, pd.read_parquet(args.data), args.report,
                                 evaluation_provenance=_json(args.provenance), trusted_evaluation=args.trusted_evaluation)
    elif args.command == "promote":
        result = promote(candidate_dir=args.candidate, evaluation_report=args.evaluation_report,
                         approval_file=args.approval, dry_run=not args.execute)
    else:
        result = rollback(dry_run=not args.execute)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
