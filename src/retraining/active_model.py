"""Shared, fail-closed active-model configuration for FastAPI and Spark.

If no activation manifest exists, the historical environment setting is honored
and V2 remains the default. Once a manifest exists it is authoritative: malformed
or incompatible manifests raise instead of silently selecting a different model.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class ActiveModelConfigurationError(ValueError):
    """Active model configuration or bundle failed integrity validation."""


@dataclass(frozen=True)
class ActiveModelBundle:
    model_version: str
    threshold: float
    model_path: Path
    metadata_path: Path
    threshold_path: Path
    feature_names: tuple[str, ...]
    artifact_sha256: Mapping[str, str]
    source: str


def _sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    try:
        value=json.loads(path.read_text(encoding="utf-8"))
    except (OSError,UnicodeError,json.JSONDecodeError) as exc:
        raise ActiveModelConfigurationError(f"Cannot read active model JSON {path}: {exc}") from exc
    if not isinstance(value,dict):
        raise ActiveModelConfigurationError(f"Active model JSON must be an object: {path}")
    return value


def _canonical_features(root: Path) -> list[str]:
    metadata=_read_json(root/"models"/"model_v1_metadata.json")
    features=metadata.get("feature_names")
    if (metadata.get("feature_count")!=34 or not isinstance(features,list) or len(features)!=34
            or len(set(features))!=34 or {"step","event_id","isFraud"}.intersection(features)):
        raise ActiveModelConfigurationError("Frozen model contract must contain 34 ordered safe features.")
    return features


def _bundle(root: Path, model_path: Path, metadata_path: Path, threshold_path: Path,
            *, source: str, recorded_hashes: Mapping[str,str] | None=None,
            required_version: str | None=None) -> ActiveModelBundle:
    for path in (model_path,metadata_path,threshold_path):
        if not path.is_file():
            raise ActiveModelConfigurationError(f"Required model artifact is missing: {path}")
    metadata=_read_json(metadata_path); threshold_data=_read_json(threshold_path)
    version=metadata.get("model_version")
    features=metadata.get("feature_names")
    canonical=_canonical_features(root)
    if not isinstance(version,str) or not version.strip() or (required_version and version!=required_version):
        raise ActiveModelConfigurationError("Active manifest version does not match bundle metadata.")
    if metadata.get("feature_count")!=34 or features!=canonical:
        raise ActiveModelConfigurationError("Active bundle feature order differs from the frozen 34-feature contract.")
    threshold=threshold_data.get("selected_threshold")
    if isinstance(threshold,bool) or not isinstance(threshold,(int,float)) or not math.isfinite(float(threshold)) or not 0<float(threshold)<1:
        raise ActiveModelConfigurationError("Active threshold must be finite and strictly between 0 and 1.")
    recorded_meta_threshold=metadata.get("selected_classification_threshold")
    if recorded_meta_threshold is not None and float(recorded_meta_threshold)!=float(threshold):
        raise ActiveModelConfigurationError("Metadata and threshold artifact disagree.")
    paths={"model.json":model_path,"metadata.json":metadata_path,"threshold.json":threshold_path}
    hashes={name:_sha256(path) for name,path in paths.items()}
    if recorded_hashes is not None:
        if set(recorded_hashes)!=set(hashes) or any(recorded_hashes[name]!=hashes[name] for name in hashes):
            raise ActiveModelConfigurationError("Active manifest artifact hash mismatch.")
    try:
        model_json=json.loads(model_path.read_text(encoding="utf-8"))
    except (OSError,UnicodeError,json.JSONDecodeError) as exc:
        raise ActiveModelConfigurationError(f"Model artifact is not valid JSON: {exc}") from exc
    if not isinstance(model_json,(dict,list)):
        raise ActiveModelConfigurationError("Model artifact must be a JSON object or array.")
    return ActiveModelBundle(version,float(threshold),model_path,metadata_path,threshold_path,
                             tuple(features),hashes,source)


def resolve_active_model(*, project_root: str | Path | None=None,
                         manifest_path: str | Path | None=None,
                         environ: Mapping[str,str] | None=None) -> ActiveModelBundle:
    root=Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    root=root.resolve(); manifest=Path(manifest_path) if manifest_path is not None else root/"models"/"active_model_manifest.json"
    if manifest.exists():
        data=_read_json(manifest)
        if data.get("schema_version")!=1 or not isinstance(data.get("artifacts"),dict):
            raise ActiveModelConfigurationError("Active model manifest schema is invalid.")
        artifacts=data["artifacts"]
        resolved={}
        for key in ("model","metadata","threshold"):
            rel=artifacts.get(key)
            if not isinstance(rel,str) or not rel:
                raise ActiveModelConfigurationError(f"Manifest artifact path {key!r} is required.")
            path=(root/rel).resolve()
            if root not in path.parents:
                raise ActiveModelConfigurationError("Manifest artifact paths must stay inside the project root.")
            resolved[key]=path
        return _bundle(root,resolved["model"],resolved["metadata"],resolved["threshold"],
            source=str(manifest),recorded_hashes=data.get("artifact_sha256"),
            required_version=data.get("model_version"))
    env=os.environ if environ is None else environ
    version=str(env.get("FRAUD_ACTIVE_MODEL_VERSION","v2")).strip().lower()
    if version not in {"v1","v2"}:
        raise ActiveModelConfigurationError("FRAUD_ACTIVE_MODEL_VERSION must be exactly v1 or v2 when no manifest exists.")
    model_root=root/"models"
    return _bundle(root,model_root/f"xgboost_model_{version}.json",
        model_root/f"model_{version}_metadata.json",model_root/f"model_{version}_threshold.json",
        source="legacy V2 default or FRAUD_ACTIVE_MODEL_VERSION environment",required_version=version.upper())
