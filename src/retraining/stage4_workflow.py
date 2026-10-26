"""Stage 4: build eligible retraining data and train an isolated candidate only on authorization.

Stage 2 is opened read-only. Synthetic demo labels are not ground truth for this
workflow. Historical V2 training rows are retained as the existing baseline; new
rows must have resolved, allowlisted trusted labels and a non-synthetic V2
prediction with an exact Stage 1 feature vector.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.labels import config as labels_config
from src.monitoring import config as monitor_config
from src.retraining import v2_config
from src.retraining.cycle_pipeline import save_candidate_bundle
from src.retraining.v2_evaluator import evaluate_split, select_threshold, threshold_grid_search
from src.retraining.v2_trainer import FINAL_FEATURES, TARGET, train_v2

MIN_VERIFIED_LABELS = 100
MIN_VERIFIED_FRAUD_LABELS = 10
RECENT_SPLIT_FRACTIONS = (0.60, 0.20, 0.20)
AUDIT_COLUMNS = ("event_id", "step")
FORBIDDEN_FEATURES = {"step", "event_id", "isFraud", "nameOrig", "nameDest",
                      "transaction_id", "fraud_probability", "predicted_label",
                      "risk_score", "risk_level", "data_origin", "label_source",
                      "source_type", "confirmed_at", "submission_id"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def validate_feature_contract(metadata_path: Path | None = None) -> list[str]:
    """Return V2 feature order after validating it against the frozen V1 contract."""
    path = metadata_path or monitor_config.MODEL_METADATA_PATHS["V2"]
    metadata = json.loads(path.read_text(encoding="utf-8"))
    features = metadata.get("feature_names")
    if (not isinstance(features, list) or len(features) != 34
            or any(not isinstance(name, str) or not name for name in features)
            or len(set(features)) != 34 or FORBIDDEN_FEATURES.intersection(features)):
        raise ValueError("V2 metadata must define 34 distinct safe model features in order")
    v1 = json.loads(v2_config.METADATA_V1_PATH.read_text(encoding="utf-8"))
    if features != v1.get("feature_names"):
        raise ValueError("V2 ordered feature contract differs from the frozen V1/V2 contract")
    if metadata.get("target") != TARGET:
        raise ValueError("V2 target must remain isFraud")
    return features


def _readonly_db(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"Stage 2 label database does not exist: {path}")
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    db = sqlite3.connect(uri, uri=True, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    return db


def _label_rows(database_path: Path) -> list[dict[str, Any]]:
    with closing(_readonly_db(database_path)) as db:
        return [dict(row) for row in db.execute("""
            SELECT l.transaction_id,l.outcome,l.label_source,l.source_type,l.confirmed_at,
                   l.status,l.submission_id,
                   p.model_version,p.data_origin,p.prediction_timestamp,
                   s.disposition AS submission_disposition,
                   EXISTS(SELECT 1 FROM label_submissions cs
                          WHERE cs.transaction_id=l.transaction_id AND cs.disposition='conflict') AS label_conflict,
                   EXISTS(SELECT 1 FROM prediction_conflicts pc
                          WHERE pc.transaction_id=l.transaction_id) AS prediction_conflict
            FROM labels l
            LEFT JOIN predictions p ON p.transaction_id=l.transaction_id
            LEFT JOIN label_submissions s ON s.submission_id=l.submission_id
            ORDER BY l.confirmed_at,l.transaction_id
        """).fetchall()]


def _logged_vectors(path: Path, transaction_ids: set[str], features: list[str]) -> tuple[dict[str, dict], dict[str, int]]:
    found: dict[str, dict] = {}
    counts: dict[str, int] = {}
    if not transaction_ids or not path.is_file():
        return found, counts
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            txn = item.get("transaction_id")
            if txn not in transaction_ids or item.get("model_version") != "V2":
                continue
            counts[txn] = counts.get(txn, 0) + 1
            found.setdefault(txn, item)
    return found, counts


def _vector_values(item: dict, features: list[str]) -> dict[str, float]:
    vector = item.get("feature_values")
    if not isinstance(vector, dict) or list(vector) != features:
        raise ValueError("missing feature vector or feature order differs from V2 metadata")
    result: dict[str, float] = {}
    for name in features:
        value = vector[name]
        if value is None:
            result[name] = np.nan
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"feature {name} is not numeric")
        elif not math.isfinite(float(value)):
            raise ValueError(f"feature {name} is non-finite")
        else:
            result[name] = float(value)
    return result


def collect_eligible_labels(*, database_path: Path, prediction_log_path: Path,
                            features: list[str], trusted_sources: set[str],
                            token_configured: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collect one strictly eligible Stage 2 label per transaction, without DB writes."""
    labels = _label_rows(database_path)
    vectors, vector_counts = _logged_vectors(prediction_log_path,
        {row["transaction_id"] for row in labels}, features)
    rows: list[dict[str, Any]] = []
    excluded = {name: 0 for name in ("synthetic_label", "untrusted_or_unauthorized", "pending_or_rejected",
        "unmatched_prediction", "synthetic_prediction", "non_v2_prediction", "label_conflict",
        "prediction_conflict", "submission_not_accepted", "missing_or_duplicate_feature_vector",
        "invalid_feature_vector", "heldout_boundary_violation")}
    source_breakdown: dict[str, int] = {}
    state_breakdown: dict[str, int] = {}
    trusted_verified_resolved = 0
    allowlisted_verified_resolved = 0
    labels_by_txn: set[str] = set()
    metadata = json.loads(monitor_config.MODEL_METADATA_PATHS["V2"].read_text(encoding="utf-8"))
    heldout_after = int(metadata["train_validation_test_strategy"]["validation_end_step"])
    for label in labels:
        txn = str(label["transaction_id"])
        source_key = f"{label['source_type']}:{label['label_source']}:{label['status']}"
        source_breakdown[source_key] = source_breakdown.get(source_key, 0) + 1
        state_key = f"{label['source_type']}:{label['status']}"
        state_breakdown[state_key] = state_breakdown.get(state_key, 0) + 1
        if label["source_type"] == "trusted_verified" and label["status"] == "resolved":
            trusted_verified_resolved += 1
            if label["label_source"] in trusted_sources:
                allowlisted_verified_resolved += 1
        reasons = []
        if label["source_type"] == "synthetic_test": reasons.append("synthetic_label")
        if label["source_type"] != "trusted_verified" or label["label_source"] not in trusted_sources or not token_configured:
            reasons.append("untrusted_or_unauthorized")
        if label["status"] != "resolved": reasons.append("pending_or_rejected")
        if label["prediction_timestamp"] is None: reasons.append("unmatched_prediction")
        if label["data_origin"] and str(label["data_origin"]).lower().startswith("synthetic"):
            reasons.append("synthetic_prediction")
        if label["model_version"] != "V2": reasons.append("non_v2_prediction")
        if label["label_conflict"]: reasons.append("label_conflict")
        if label["prediction_conflict"]: reasons.append("prediction_conflict")
        if label["submission_disposition"] != "accepted_genuine": reasons.append("submission_not_accepted")
        item = vectors.get(txn)
        if item is None or vector_counts.get(txn, 0) != 1:
            reasons.append("missing_or_duplicate_feature_vector")
        if item is not None:
            if str(item.get("data_origin", "")).lower().startswith("synthetic"):
                reasons.append("synthetic_prediction")
            try:
                values = _vector_values(item, features)
                step = item.get("step")
                if isinstance(step, bool) or not isinstance(step, int) or step <= heldout_after:
                    reasons.append("heldout_boundary_violation")
            except (TypeError, ValueError):
                values = None
                reasons.append("invalid_feature_vector")
        else:
            values = None
        if reasons:
            for reason in set(reasons): excluded[reason] += 1
            continue
        if txn in labels_by_txn:
            excluded["label_conflict"] += 1
            continue
        labels_by_txn.add(txn)
        assert values is not None and item is not None
        rows.append({"event_id": f"live:{txn}", "transaction_id": txn, "step": int(item["step"]),
                     **values, TARGET: int(label["outcome"]), "label_source": label["label_source"],
                     "source_type": label["source_type"], "confirmed_at": label["confirmed_at"],
                     "submission_id": label["submission_id"], "data_origin": label["data_origin"]})
    frame = pd.DataFrame(rows)
    if rows:
        frame = frame.sort_values(["step", "event_id"], kind="stable").reset_index(drop=True)
    return frame, {"labels_seen": len(labels), "eligible_label_count": len(frame),
        "eligible_fraud_count": int(frame[TARGET].sum()) if len(frame) else 0,
        "eligible_legitimate_count": int((frame[TARGET] == 0).sum()) if len(frame) else 0,
        "label_source_breakdown": source_breakdown, "label_state_breakdown": state_breakdown,
        "resolved_trusted_verified_labels_seen": trusted_verified_resolved,
        "resolved_allowlisted_trusted_labels_seen": allowlisted_verified_resolved,
        "exclusions": excluded,
        "trusted_source_allowlist": sorted(trusted_sources), "trusted_source_authorization_configured": bool(trusted_sources and token_configured),
        "minimum_verified_labels": MIN_VERIFIED_LABELS,
        "minimum_verified_fraud_labels": MIN_VERIFIED_FRAUD_LABELS,
        "heldout_boundary_exclusive_step": heldout_after,
        "feature_vector_log": str(prediction_log_path)}


def validate_example_frame(frame: pd.DataFrame, features: list[str]) -> None:
    forbidden = FORBIDDEN_FEATURES.intersection(features)
    if forbidden:
        raise ValueError(f"Forbidden/audit columns in model feature contract: {sorted(forbidden)}")
    required = {"event_id", "step", TARGET, *features}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if not frame.event_id.is_unique:
        raise ValueError("Duplicate transaction identity in supervised examples")
    if frame[TARGET].isna().any() or not frame[TARGET].isin([0, 1]).all():
        raise ValueError("Supervised target must be present and binary")
    if not pd.api.types.is_numeric_dtype(frame.step):
        raise ValueError("step must be numeric for chronological ordering")
    for feature in features:
        if not pd.api.types.is_numeric_dtype(frame[feature]):
            raise ValueError(f"Feature {feature} must be numeric")
        values = frame[feature].to_numpy(dtype=np.float64)
        if np.isinf(values).any():
            raise ValueError(f"Feature {feature} contains infinite values")


def chronological_split(historical: pd.DataFrame, recent: pd.DataFrame,
                        features: list[str]) -> dict[str, Any]:
    """Keep existing V2 history in train; divide new labels by whole step groups."""
    if recent.empty:
        raise ValueError("No eligible recent labels are available for splitting")
    if set(historical.event_id.astype(str)) & set(recent.event_id.astype(str)):
        raise ValueError("Duplicate transaction identities cross historical and recent data")
    if historical.step.max() >= recent.step.min():
        raise ValueError("New verified labels must chronologically follow historical V2 training data")
    recent = recent.sort_values(["step", "event_id"], kind="stable").reset_index(drop=True)
    counts = recent.groupby("step", sort=True).size().cumsum()
    cuts = []
    for fraction in RECENT_SPLIT_FRACTIONS[:2]:
        target = len(recent) * (sum(RECENT_SPLIT_FRACTIONS[:len(cuts)+1]))
        position = min(int(np.searchsorted(counts.to_numpy(), target, side="left")), len(counts)-1)
        cuts.append(int(counts.index[position]))
    train_recent = recent[recent.step <= cuts[0]]
    validation = recent[(recent.step > cuts[0]) & (recent.step <= cuts[1])]
    test = recent[recent.step > cuts[1]]
    if min(len(train_recent), len(validation), len(test)) == 0:
        raise ValueError("Temporal split produced an empty recent train/validation/test split")
    train = pd.concat([historical, train_recent], ignore_index=True).sort_values(
        ["step", "event_id"], kind="stable").reset_index(drop=True)
    parts = [train, validation.reset_index(drop=True), test.reset_index(drop=True)]
    id_sets = [set(part.event_id.astype(str)) for part in parts]
    if any(id_sets[i] & id_sets[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError("Transaction identity crosses chronological splits")
    if not (train.step.max() < parts[1].step.min() <= parts[1].step.max() < parts[2].step.min()):
        raise ValueError("Chronological split step ranges overlap")
    for name, part in zip(("train", "validation", "test"), parts):
        if part[TARGET].nunique() < 2:
            raise ValueError(f"{name} split must contain both legitimate and fraud examples")
        if name != "train" and int(part[TARGET].sum()) < 1:
            raise ValueError(f"{name} split must contain at least one verified fraud example")
    return {"train_df":parts[0],"validation_df":parts[1],"test_df":parts[2],
        "train_start_step":int(parts[0].step.min()),"train_end_step":int(parts[0].step.max()),
        "validation_start_step":int(parts[1].step.min()),"validation_end_step":int(parts[1].step.max()),
        "test_start_step":int(parts[2].step.min()),"test_end_step":int(parts[2].step.max()),
        "recent_split_fractions":list(RECENT_SPLIT_FRACTIONS)}


def _historical_source_summary(dataset_path: Path, train_end: int) -> dict[str, Any]:
    import pyarrow.dataset as ds
    dataset = ds.dataset(dataset_path, format="parquet")
    required = {"event_id", "step", TARGET, *FINAL_FEATURES}
    if not required.issubset(dataset.schema.names):
        raise ValueError(f"Historical V2 parquet is missing columns: {sorted(required-set(dataset.schema.names))}")
    scanner = dataset.scanner(columns=["step"], filter=ds.field("step") <= train_end,
                              batch_size=65536, use_threads=True)
    count = sum(len(batch) for batch in scanner.to_batches())
    return {"path":str(dataset_path),"training_window_max_step":train_end,
            "eligible_historical_training_rows":count,"source_sha256":_sha256_file(dataset_path)}


def _load_historical(dataset_path: Path, train_end: int, features: list[str]) -> pd.DataFrame:
    import pyarrow.dataset as ds
    dataset=ds.dataset(dataset_path,format="parquet")
    frame=dataset.to_table(columns=["event_id","step",*features,TARGET],
        filter=ds.field("step")<=train_end).to_pandas()
    frame["event_id"]="historical:"+frame.event_id.astype(str)
    validate_example_frame(frame,features)
    return frame[["event_id","step",*features,TARGET]]


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path=Path(path)
    if path.exists(): raise FileExistsError(f"Refusing to overwrite Stage 4 report: {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",dir=path.parent,
                                     prefix=f".{path.name}.",suffix=".tmp",delete=False) as out:
        temporary=Path(out.name)
        json.dump(report,out,indent=2,allow_nan=False)
    try: os.replace(temporary,path)
    except Exception:
        temporary.unlink(missing_ok=True); raise


def run_stage4(*, label_db: Path | None = None, prediction_log: Path | None = None,
               historical_dataset: Path | None = None, report_path: Path,
               candidate_dir: Path | None = None, train_candidate: bool = False) -> dict[str, Any]:
    """Inspect eligibility; optionally train only after every explicit gate passes."""
    features=validate_feature_contract()
    v2meta=json.loads(monitor_config.MODEL_METADATA_PATHS["V2"].read_text(encoding="utf-8"))
    train_end=int(v2meta["train_validation_test_strategy"]["train_end_step"])
    history_path=historical_dataset or v2_config.V2_DATASET_PATH
    hist_summary=_historical_source_summary(history_path,train_end)
    eligible, eligibility = collect_eligible_labels(database_path=label_db or labels_config.STAGE2_DB_PATH,
        prediction_log_path=prediction_log or monitor_config.STAGE1_JSONL_PATH,features=features,
        trusted_sources=labels_config.trusted_label_sources(),token_configured=bool(labels_config.trusted_label_token()))
    historical_hash=hist_summary["source_sha256"]
    fingerprint=_sha256_json({"historical_source_sha256":historical_hash,"historical_step_max":train_end,
        "eligible_live_labels":[{"transaction_id":r["transaction_id"],"submission_id":r["submission_id"],
            "outcome":int(r[TARGET]),"step":int(r["step"]),"feature_values":{f:(None if pd.isna(r[f]) else float(r[f])) for f in features}}
            for r in eligible.to_dict("records")],"feature_names":features})
    report={"generated_at":datetime.now(timezone.utc).isoformat(),"stage":"Stage 4 candidate retraining",
        "status":"insufficient_eligible_labels","training_started":False,"candidate_artifacts":None,
        "target":TARGET,"feature_count":len(features),"feature_names":features,
        "historical_training_source":hist_summary,"eligibility":eligibility,
        "dataset_fingerprint":fingerprint,"dataset_counts":{"historical_training_rows":hist_summary["eligible_historical_training_rows"],
            "eligible_new_verified_labels":int(len(eligible)),"combined_rows":None},
        "eligibility_policy":{"minimum_new_verified_labels":MIN_VERIFIED_LABELS,
            "minimum_new_verified_fraud_labels":MIN_VERIFIED_FRAUD_LABELS,
            "validation_test_require_both_classes":True,"new_labels_must_be_after_v2_validation_end_step":eligibility["heldout_boundary_exclusive_step"],
            "historical_v2_train_rows_are_baseline_only":True},
        "split":None,"explanation":None}
    enough = (len(eligible)>=MIN_VERIFIED_LABELS and int(eligible[TARGET].sum())>=MIN_VERIFIED_FRAUD_LABELS)
    if not enough:
        report["explanation"]=(f"Requires at least {MIN_VERIFIED_LABELS} eligible verified labels and "
            f"{MIN_VERIFIED_FRAUD_LABELS} verified fraud labels; found {len(eligible)} and "
            f"{int(eligible[TARGET].sum()) if len(eligible) else 0}. No training was run.")
        _write_report(report_path,report); return report
    if not train_candidate:
        report["status"]="eligible_training_not_requested"
        report["explanation"]="Eligibility passed, but candidate training requires explicit --train-candidate opt-in."
        _write_report(report_path,report); return report
    historical=_load_historical(history_path,train_end,features)
    validate_example_frame(eligible[["event_id","step",*features,TARGET]],features)
    try:
        split=chronological_split(historical,eligible[["event_id","step",*features,TARGET]],features)
    except ValueError as exc:
        report["explanation"]=f"Eligibility/split gate failed: {exc}. No training was run."
        _write_report(report_path,report); return report
    report["dataset_counts"]["combined_rows"]=len(historical)+len(eligible)
    report["split"]={k:(int(v) if isinstance(v,(np.integer,int)) else len(v) if isinstance(v,pd.DataFrame) else v)
        for k,v in split.items() if not isinstance(v,pd.DataFrame)}
    train_df,val_df,test_df=(split[k] for k in ("train_df","validation_df","test_df"))
    params=dict(v2meta.get("xgboost_parameters",{}))
    model,elapsed=train_v2(train_df,val_df,params)
    threshold,threshold_info=select_threshold(threshold_grid_search(model,val_df))
    metrics=evaluate_split(model,test_df,threshold)
    destination=candidate_dir or (v2_config.PROJECT_ROOT/"models"/"candidates"/
        f"stage4-{datetime.now(timezone.utc):%Y%m%dT%H%M%S-%fZ}")
    candidate_metadata={"model_name":"Stage 4 isolated V2 candidate","model_version":destination.name,
        "feature_count":len(features),"feature_names":features,"target":TARGET,
        "dataset_fingerprint":fingerprint,"label_provenance_summary":eligibility,
        "temporal_split":report["split"],"training_configuration":params,
        "training_seconds":elapsed,"test_metrics":metrics}
    artifacts=save_candidate_bundle(model,candidate_metadata,{"selected_threshold":threshold,
        "selection_objective":"maximum F1 on chronological validation labels","validation_metrics":threshold_info},
        active_dir=v2_config.PROJECT_ROOT/"models"/"active",
        cycle=__import__("src.retraining.cycle_pipeline",fromlist=["CycleConfig"]).CycleConfig(
            candidate_output_dir=destination))
    report.update({"status":"candidate_trained_not_promoted","training_started":True,
        "candidate_artifacts":{"directory":str(destination),"sha256":artifacts},
        "training_seconds":elapsed,"selected_threshold":threshold,"test_metrics":metrics,
        "dataset_counts":{"historical_training_rows":len(historical),"eligible_new_verified_labels":len(eligible),
                           "combined_rows":len(historical)+len(eligible)},
        "explanation":"Candidate artifacts were written separately; no promotion or deployment occurred."})
    _write_report(report_path,report)
    return report


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True,help="new report path; existing files are never overwritten")
    parser.add_argument("--label-db",type=Path,default=labels_config.STAGE2_DB_PATH)
    parser.add_argument("--prediction-log",type=Path,default=monitor_config.STAGE1_JSONL_PATH)
    parser.add_argument("--historical-dataset",type=Path,default=v2_config.V2_DATASET_PATH)
    parser.add_argument("--candidate-dir",type=Path)
    parser.add_argument("--train-candidate",action="store_true",help="explicitly permit training if all gates pass")
    args=parser.parse_args()
    report=run_stage4(label_db=args.label_db,prediction_log=args.prediction_log,
        historical_dataset=args.historical_dataset,report_path=args.report,
        candidate_dir=args.candidate_dir,train_candidate=args.train_candidate)
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=="__main__": main()
