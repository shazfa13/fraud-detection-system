"""Isolated paths and identifiers for optional orchestrator runs.

This module deliberately defines its own output root. It does not import or
change the Phase 10/11 configurations, and it rejects roots that overlap their
fixed V2 artifacts or dataset.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "data" / "orchestrator_runs"

# Explicitly protected Phase 10/11 outputs. Keep this list local so this new
# package does not require edits to any existing configuration file.
PROTECTED_OUTPUTS = (
    PROJECT_ROOT / "data" / "processed" / "v2_retraining_dataset.parquet",
    PROJECT_ROOT / "reports" / "v2_retraining_dataset_report.json",
    PROJECT_ROOT / "models" / "xgboost_model_v2.json",
    PROJECT_ROOT / "models" / "model_v2_metadata.json",
    PROJECT_ROOT / "models" / "model_v2_threshold.json",
    PROJECT_ROOT / "reports" / "model_v2_training_report.json",
    PROJECT_ROOT / "reports" / "model_v1_vs_v2_comparison.json",
    PROJECT_ROOT / "reports" / "model_v1_vs_v2_comparison.csv",
)

_RUN_ID_PATTERN = re.compile(r"\A\d{8}T\d{6}Z_[0-9a-f]{32}\Z")


class OrchestratorPathError(ValueError):
    """Raised when a run identifier or output path is unsafe."""


def generate_run_id() -> str:
    """Return a UTC timestamp plus random UUID, unique for practical purposes."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid4().hex}"


def validate_run_id(run_id: str) -> str:
    """Validate and return a generated-format run ID; reject path syntax."""
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise OrchestratorPathError("run_id must match YYYYMMDDTHHMMSSZ_<32 lowercase hex characters>.")
    return run_id


def validate_run_root(run_root: str | Path = RUNS_ROOT) -> Path:
    """Resolve an explicit run root and reject overlap with protected outputs."""
    root = Path(run_root).expanduser().resolve()
    if root == Path(root.anchor):
        raise OrchestratorPathError("The orchestrator run root cannot be a filesystem root.")
    for protected_path in PROTECTED_OUTPUTS:
        protected = protected_path.resolve()
        if root == protected or root in protected.parents or protected in root.parents:
            raise OrchestratorPathError(
                f"Run root overlaps a protected Phase 10/11 output: {protected_path.relative_to(PROJECT_ROOT)}"
            )
    return root


def resolve_run_dir(run_id: str, run_root: str | Path = RUNS_ROOT) -> Path:
    """Resolve a run directory and prove it remains a direct child of run_root."""
    safe_id = validate_run_id(run_id)
    root = validate_run_root(run_root)
    run_dir = (root / safe_id).resolve()
    if run_dir.parent != root or run_dir == root:
        raise OrchestratorPathError("Resolved run directory must be a direct child of the run root.")
    return run_dir


def resolve_run_path(run_id: str, *parts: str | Path,
                     run_root: str | Path = RUNS_ROOT) -> Path:
    """Resolve an output below a run directory, rejecting traversal/absolute parts."""
    run_dir = resolve_run_dir(run_id, run_root)
    output = run_dir
    for part in parts:
        relative = Path(part)
        if relative.is_absolute() or any(piece == ".." for piece in relative.parts):
            raise OrchestratorPathError("Run output path components must be relative and cannot traverse upward.")
        output = output / relative
    resolved = output.resolve()
    if resolved != run_dir and run_dir not in resolved.parents:
        raise OrchestratorPathError("Resolved output path escapes its run directory.")
    return resolved
