"""Optional, isolated model lifecycle orchestration package."""

from src.orchestrator.config import RUNS_ROOT, generate_run_id

__all__ = ["RUNS_ROOT", "generate_run_id"]
