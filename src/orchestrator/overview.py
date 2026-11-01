"""Read-only combined overview of active model, drift, labels, and run records."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.orchestrator.config import RUNS_ROOT, resolve_run_dir, validate_run_id
from src.orchestrator.drift_adapter import get_latest_drift_summary
from src.orchestrator.label_adapter import get_label_readiness_summary
from src.orchestrator.model_adapter import get_active_model_summary


def get_orchestrator_overview(*, run_root: str | Path = RUNS_ROOT) -> dict[str, Any]:
    """Collect existing state using read-only interfaces; perform no lifecycle work."""
    root = Path(run_root).expanduser().resolve()
    history: list[dict[str, Any]] = []
    history_status = "AVAILABLE"
    if root.is_dir():
        try:
            for entry in root.iterdir():
                if not entry.is_dir() or entry.is_symlink():
                    continue
                try:
                    run_id = validate_run_id(entry.name)
                    run_dir = resolve_run_dir(run_id, root)
                    record_path = run_dir / "run.json"
                    if not record_path.is_file() or record_path.is_symlink():
                        continue
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                    if isinstance(record, dict) and record.get("run_id") == run_id:
                        history.append(record)
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
            history.sort(key=lambda record: str(record.get("created_at_utc", "")), reverse=True)
        except OSError as exc:
            history_status = f"UNAVAILABLE: {type(exc).__name__}: {exc}"
    else:
        history_status = "NO_RUN_ROOT"

    return {
        "active_model": get_active_model_summary(),
        "latest_drift": get_latest_drift_summary(),
        "label_readiness": get_label_readiness_summary(),
        "run_history": {"status": history_status, "runs": history},
    }


__all__ = ["get_orchestrator_overview"]
