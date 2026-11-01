"""Read-only adapter for the shared active-model resolver."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.retraining.active_model import resolve_active_model


def get_active_model_summary(*, project_root: str | Path | None = None) -> dict[str, Any]:
    """Resolve and summarize the active model without loading an estimator."""
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[2]
    manifest_path = root / "models" / "active_model_manifest.json"
    manifest_present = manifest_path.is_file()
    manifest: dict[str, Any] | None = None
    try:
        bundle = resolve_active_model(project_root=root)
        if manifest_present:
            parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                manifest = {
                    "schema_version": parsed.get("schema_version"),
                    "model_version": parsed.get("model_version"),
                    "artifacts": parsed.get("artifacts"),
                    "artifact_sha256": parsed.get("artifact_sha256"),
                    "activated_at": parsed.get("activated_at"),
                    "path": str(manifest_path),
                }
        return {
            "status": "VALID",
            "model_version": bundle.model_version,
            "model_artifact_path": str(bundle.model_path),
            "metadata_path": str(bundle.metadata_path),
            "threshold_path": str(bundle.threshold_path),
            "threshold": bundle.threshold,
            "manifest_present": manifest_present,
            "manifest": manifest,
            "selection_source": bundle.source,
            "feature_count": len(bundle.feature_names),
            "integrity": {
                "status": "VALIDATED_BY_SHARED_RESOLVER",
                "artifact_sha256": dict(bundle.artifact_sha256),
            },
        }
    except Exception as exc:
        return {
            "status": "UNAVAILABLE",
            "model_version": None,
            "model_artifact_path": None,
            "metadata_path": None,
            "threshold_path": None,
            "manifest_present": manifest_present,
            "manifest": manifest,
            "selection_source": None,
            "integrity": {"status": "FAILED_OR_UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"},
        }


__all__ = ["get_active_model_summary"]
