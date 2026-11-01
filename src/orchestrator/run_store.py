"""Create-only initialization and JSON records for isolated orchestrator runs."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.orchestrator.config import (
    RUNS_ROOT,
    OrchestratorPathError,
    resolve_run_dir,
    resolve_run_path,
    validate_run_id,
    validate_run_root,
    generate_run_id,
)


class RunStoreError(RuntimeError):
    """Raised when a run cannot be safely initialized or recorded."""


def _write_create_only_json(path: Path, record: Mapping[str, Any]) -> None:
    """Atomically publish JSON without replacing any existing record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(dict(record), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Creating a hard link publishes the complete temp file atomically and
        # fails if a record with this name already exists.
        os.link(temporary, path)
    except FileExistsError as exc:
        raise RunStoreError(f"Run record already exists; refusing to overwrite: {path}") from exc
    except (OSError, TypeError, ValueError) as exc:
        raise RunStoreError(f"Could not safely persist run record {path}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def initialize_run(*, run_id: str | None = None,
                   run_root: str | Path = RUNS_ROOT,
                   metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Create a new unique run directory and its initial create-only record."""
    selected_id = generate_run_id() if run_id is None else validate_run_id(run_id)
    root = validate_run_root(run_root)
    run_dir = resolve_run_dir(selected_id, root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise RunStoreError(f"Run directory already exists; refusing reuse: {run_dir}") from exc
    except OSError as exc:
        raise RunStoreError(f"Could not create run directory {run_dir}: {exc}") from exc

    record: dict[str, Any] = {
        "run_id": selected_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_status": "INITIALIZED",
        "current_stage": "run_created",
        "metadata": dict(metadata or {}),
    }
    try:
        record_path = resolve_run_path(selected_id, "run.json", run_root=root)
        _write_create_only_json(record_path, record)
    except (OrchestratorPathError, RunStoreError) as exc:
        raise RunStoreError(f"Run directory created but initial record could not be persisted: {exc}") from exc
    return {**record, "run_dir": str(run_dir), "record_path": str(record_path)}


def persist_run_record(run_id: str, record: Mapping[str, Any], *,
                       run_root: str | Path = RUNS_ROOT) -> Path:
    """Create a run record at ``run.json``; an existing record is never replaced."""
    validate_run_id(run_id)
    run_dir = resolve_run_dir(run_id, run_root)
    if not run_dir.is_dir():
        raise RunStoreError(f"Run directory does not exist: {run_dir}")
    if record.get("run_id") != run_id:
        raise RunStoreError("Run record run_id must match the target run directory.")
    path = resolve_run_path(run_id, "run.json", run_root=run_root)
    _write_create_only_json(path, record)
    return path


__all__ = ["RunStoreError", "initialize_run", "persist_run_record"]
