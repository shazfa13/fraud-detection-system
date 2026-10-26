"""Safely promote and roll back pre-trained model artifact bundles.

The module only manages model.json, metadata.json, and threshold.json. It does
not train models or connect to the API, Kafka, Spark, or live traffic. Individual
file replacements are atomic; a three-file operation is not globally atomic to
concurrent readers.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ARTIFACT_NAMES = ("model.json", "metadata.json", "threshold.json")
EXPECTED_FEATURE_COUNT = 34


class ModelActivationError(ValueError):
    """Raised for invalid bundles or unsafe lifecycle operations."""


@dataclass(frozen=True)
class OperationResult:
    """Result metadata and SHA-256 hashes for files involved in an operation."""

    operation: str
    dry_run: bool
    model_version: str
    feature_count: int
    threshold: float
    sha256: Mapping[str, str]


def _hash(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    """Read a JSON artifact and normalize read or parse errors."""
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelActivationError(f"Unreadable JSON artifact {path}: {exc}") from exc


def _canonical_features() -> list[str]:
    """Read the frozen ordered feature contract from the original V1 metadata."""
    path = Path(__file__).resolve().parents[2] / "models" / "model_v1_metadata.json"
    metadata = _json(path)
    if not isinstance(metadata, dict):
        raise ModelActivationError("Canonical V1 metadata must be a JSON object.")
    features = metadata.get("feature_names")
    if (
        metadata.get("feature_count") != EXPECTED_FEATURE_COUNT
        or not isinstance(features, list)
        or len(features) != EXPECTED_FEATURE_COUNT
        or any(not isinstance(item, str) or not item for item in features)
        or len(set(features)) != EXPECTED_FEATURE_COUNT
    ):
        raise ModelActivationError("Canonical V1 metadata does not define the expected 34-feature contract.")
    return features


def _validate_bundle(
    directory: Path,
    expected_features: Sequence[str] | None = None,
) -> OperationResult:
    """Validate artifact readability, required metadata, feature order, and threshold."""
    if not directory.is_dir():
        raise ModelActivationError(f"Artifact directory missing: {directory}")

    paths = {name: directory / name for name in ARTIFACT_NAMES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ModelActivationError("Missing artifacts: " + ", ".join(missing))

    model_data = _json(paths["model.json"])
    if not isinstance(model_data, (dict, list)):
        raise ModelActivationError("model.json must contain a JSON object or array.")

    metadata = _json(paths["metadata.json"])
    threshold_data = _json(paths["threshold.json"])
    if not isinstance(metadata, dict) or not isinstance(threshold_data, dict):
        raise ModelActivationError("Metadata and threshold artifacts must be JSON objects.")

    version = metadata.get("model_version")
    features = metadata.get("feature_names")
    feature_count = metadata.get("feature_count")
    if not isinstance(version, str) or not version.strip():
        raise ModelActivationError("metadata.model_version must be a non-empty string.")
    if (
        not isinstance(features, list)
        or len(features) != EXPECTED_FEATURE_COUNT
        or any(not isinstance(item, str) or not item for item in features)
        or len(set(features)) != EXPECTED_FEATURE_COUNT
        or feature_count != EXPECTED_FEATURE_COUNT
    ):
        raise ModelActivationError(
            "Metadata must define exactly 34 unique ordered feature_names and feature_count=34."
        )
    if expected_features is not None and features != list(expected_features):
        raise ModelActivationError("Model feature names/order differ from the frozen contract.")

    threshold_value = threshold_data.get("selected_threshold")
    if isinstance(threshold_value, bool) or not isinstance(threshold_value, (int, float)):
        raise ModelActivationError("threshold.selected_threshold must be numeric.")
    threshold = float(threshold_value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ModelActivationError("selected_threshold must be finite and in [0, 1].")

    metadata_threshold = metadata.get("selected_classification_threshold")
    if metadata_threshold is not None and (
        isinstance(metadata_threshold, bool)
        or not isinstance(metadata_threshold, (int, float))
        or not math.isfinite(float(metadata_threshold))
        or float(metadata_threshold) != threshold
    ):
        raise ModelActivationError("Metadata threshold conflicts with threshold.json.")

    return OperationResult(
        operation="validate",
        dry_run=True,
        model_version=version,
        feature_count=feature_count,
        threshold=threshold,
        sha256={name: _hash(path) for name, path in paths.items()},
    )


def validate_bundle(
    directory: str | Path,
    expected_features: Sequence[str] | None = None,
) -> OperationResult:
    """Validate a bundle without writing or replacing any files."""
    return _validate_bundle(Path(directory), expected_features)


def _make_stage(parent: Path) -> Path:
    """Create an operation-owned temporary staging directory."""
    return Path(tempfile.mkdtemp(prefix=".model-activation-", dir=parent))


def _paths_overlap(*paths: Path) -> bool:
    """Return whether any resolved managed directories are equal or nested."""
    resolved = [path.resolve() for path in paths]
    return any(
        left == right or left in right.parents or right in left.parents
        for index, left in enumerate(resolved)
        for right in resolved[index + 1:]
    )


def _copy_bundle(source: Path, destination: Path) -> None:
    """Copy only the three managed artifacts into a staging directory."""
    for name in ARTIFACT_NAMES:
        shutil.copyfile(source / name, destination / name)


def _atomic_copy(source: Path, destination: Path) -> None:
    """Atomically replace one destination file via a temporary sibling file."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _cleanup_stages(*stages: Path | None) -> None:
    """Remove only staging directories created by this module."""
    for stage in stages:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def _operation_hashes(
    **bundles: Mapping[str, str],
) -> dict[str, str]:
    """Return namespaced hashes for all bundles involved in a completed operation."""
    return {
        f"{bundle_name}/{artifact_name}": digest
        for bundle_name, hashes in bundles.items()
        for artifact_name, digest in hashes.items()
    }


def promote_candidate(
    candidate_dir: str | Path,
    active_dir: str | Path,
    backup_dir: str | Path,
    *,
    dry_run: bool = False,
) -> OperationResult:
    """Validate and promote a candidate while preserving current active as rollback backup.

    The backup must be absent or empty. Existing backup content is never overwritten.
    Validation and staging complete before active artifacts are replaced. If promotion
    fails partway through, active is restored from its staging copy where possible.
    """
    candidate = Path(candidate_dir)
    active = Path(active_dir)
    backup = Path(backup_dir)

    if not active.is_dir() or not candidate.is_dir():
        raise ModelActivationError("Active and candidate directories must already exist.")
    if _paths_overlap(candidate, active, backup):
        raise ModelActivationError(
            "Candidate, active, and backup directories must be separate and non-nested."
        )

    features = _canonical_features()
    previous = _validate_bundle(active, features)
    proposed = _validate_bundle(candidate, features)
    if backup.exists() and (not backup.is_dir() or any(backup.iterdir())):
        raise ModelActivationError(
            "Backup is not empty; refusing to overwrite the existing rollback point."
        )
    if dry_run:
        return OperationResult(
            "promote", True, proposed.model_version, proposed.feature_count,
            proposed.threshold, _operation_hashes(candidate=proposed.sha256),
        )

    backup_created = not backup.exists()
    if backup_created:
        backup.mkdir(parents=True)

    previous_stage: Path | None = None
    candidate_stage: Path | None = None
    written_backup_files: list[Path] = []
    backup_complete = False
    try:
        previous_stage = _make_stage(active.parent)
        candidate_stage = _make_stage(active.parent)
        _copy_bundle(active, previous_stage)
        _copy_bundle(candidate, candidate_stage)

        for name in ARTIFACT_NAMES:
            if _hash(previous_stage / name) != previous.sha256[name]:
                raise ModelActivationError(f"Active staging hash mismatch: {name}.")
            if _hash(candidate_stage / name) != proposed.sha256[name]:
                raise ModelActivationError(f"Candidate staging hash mismatch: {name}.")

        for name in ARTIFACT_NAMES:
            destination = backup / name
            if destination.exists():
                raise ModelActivationError(f"Backup destination conflict: {destination}")
            _atomic_copy(previous_stage / name, destination)
            written_backup_files.append(destination)
        backed_up = _validate_bundle(backup, features)
        backup_complete = True

        try:
            for name in ARTIFACT_NAMES:
                _atomic_copy(candidate_stage / name, active / name)
            promoted = _validate_bundle(active, features)
            if promoted.sha256 != proposed.sha256:
                raise ModelActivationError("Promoted artifact hashes differ from candidate.")
        except Exception as promotion_error:
            try:
                for name in ARTIFACT_NAMES:
                    _atomic_copy(previous_stage / name, active / name)
                _validate_bundle(active, features)
            except Exception as recovery_error:
                raise ModelActivationError(
                    f"Promotion failed ({promotion_error}); active recovery failed "
                    f"({recovery_error}); the previous active remains in backup."
                ) from promotion_error
            raise ModelActivationError(
                f"Promotion failed; prior active restored and backup retained: {promotion_error}"
            ) from promotion_error
    except Exception:
        if not backup_complete:
            for path in written_backup_files:
                path.unlink(missing_ok=True)
        raise
    finally:
        _cleanup_stages(previous_stage, candidate_stage)
        if backup_created and not backup_complete and backup.is_dir() and not any(backup.iterdir()):
            backup.rmdir()

    return OperationResult(
        "promote",
        False,
        proposed.model_version,
        proposed.feature_count,
        proposed.threshold,
        _operation_hashes(candidate=proposed.sha256, previous_active=previous.sha256,
                          active=promoted.sha256, backup=backed_up.sha256),
    )


def rollback(
    active_dir: str | Path,
    backup_dir: str | Path,
    *,
    dry_run: bool = False,
) -> OperationResult:
    """Restore backup to active and preserve the replaced active bundle in backup."""
    active = Path(active_dir)
    backup = Path(backup_dir)
    if _paths_overlap(active, backup):
        raise ModelActivationError("Active and backup directories must be separate and non-nested.")
    features = _canonical_features()
    current = _validate_bundle(active, features)
    previous = _validate_bundle(backup, features)

    if dry_run:
        return OperationResult(
            "rollback", True, previous.model_version, previous.feature_count,
            previous.threshold, _operation_hashes(backup=previous.sha256),
        )

    active_stage: Path | None = None
    backup_stage: Path | None = None
    try:
        active_stage = _make_stage(active.parent)
        backup_stage = _make_stage(active.parent)
        _copy_bundle(active, active_stage)
        _copy_bundle(backup, backup_stage)

        for name in ARTIFACT_NAMES:
            if _hash(active_stage / name) != current.sha256[name]:
                raise ModelActivationError(f"Active staging hash mismatch: {name}.")
            if _hash(backup_stage / name) != previous.sha256[name]:
                raise ModelActivationError(f"Backup staging hash mismatch: {name}.")

        try:
            for name in ARTIFACT_NAMES:
                _atomic_copy(backup_stage / name, active / name)
            restored = _validate_bundle(active, features)
            if restored.sha256 != previous.sha256:
                raise ModelActivationError("Restored active hashes differ from backup.")
            for name in ARTIFACT_NAMES:
                _atomic_copy(active_stage / name, backup / name)
            replaced = _validate_bundle(backup, features)
        except Exception as switch_error:
            try:
                for name in ARTIFACT_NAMES:
                    _atomic_copy(active_stage / name, active / name)
                for name in ARTIFACT_NAMES:
                    _atomic_copy(backup_stage / name, backup / name)
            except Exception as recovery_error:
                raise ModelActivationError(
                    f"Rollback failed ({switch_error}); recovery failed ({recovery_error})."
                ) from switch_error
            raise ModelActivationError(
                f"Rollback failed; active and backup restored: {switch_error}"
            ) from switch_error
    finally:
        _cleanup_stages(active_stage, backup_stage)

    return OperationResult(
        "rollback",
        False,
        previous.model_version,
        previous.feature_count,
        previous.threshold,
        _operation_hashes(backup=previous.sha256, active=restored.sha256,
                          replaced_active=replaced.sha256),
    )
