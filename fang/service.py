"""The surface the Antares apps/fang gRPC server consumes.

Kept deliberately small: everything here is called on a request path, so
nothing in this module unpickles a model bundle except prediction itself.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

BUNDLE_FILENAME = "model.joblib"
RUN_ID_PATTERN = re.compile(
    r"^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})"
    r"T(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})Z-[0-9a-f]+$"
)


@dataclass(frozen=True)
class RunInfo:
    """One trained run, described without loading it."""

    run_id: str
    created_at: str
    schema_version: int
    selected_model: str
    threshold: float
    path: Path


def _created_at_from_run_id(run_id: str) -> str:
    """ISO-8601 timestamp parsed out of the run id, or an empty string."""
    match = RUN_ID_PATTERN.match(run_id)
    if match is None:
        return ""
    parts = match.groupdict()
    return (
        f"{parts['year']}-{parts['month']}-{parts['day']}"
        f"T{parts['hour']}:{parts['minute']}:{parts['second']}Z"
    )


def _read_mapping(path: Path) -> dict[str, Any] | None:
    """Parse a JSON or YAML sidecar, or None when it is unusable.

    Returns None rather than an empty dict so a corrupt file can never be
    mistaken for a file whose fields happen to be absent.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        parsed = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError):
        return None
    return parsed if isinstance(parsed, dict) else None


def list_runs(artifact_dir: Path | str = "models") -> list[RunInfo]:
    """Every trained run under ``artifact_dir`` that is safe to serve, newest first.

    A run qualifies when it has a bundle, a run id this build recognises, and
    readable sidecars carrying a numeric threshold and schema version.
    Everything else is skipped with a warning: a hand-named directory such as
    ``models/latest`` would otherwise sort above every timestamped id and
    become the newest run, and a truncated sidecar would otherwise be served
    as ``threshold=0.0``, a legitimate value that classifies every candidate
    as positive. Returns an empty list when the directory does not exist,
    which is the state before the first model is ever trained.
    """
    base = Path(artifact_dir)
    if not base.is_dir():
        return []

    runs: list[RunInfo] = []
    for directory in sorted(base.iterdir(), reverse=True):
        if not directory.is_dir() or not (directory / BUNDLE_FILENAME).is_file():
            continue

        created_at = _created_at_from_run_id(directory.name)
        if not created_at:
            logger.warning("Skipping %s: not a run id this build recognises.", directory)
            continue

        schema = _read_mapping(directory / "feature_schema.yaml")
        threshold = _read_mapping(directory / "threshold.json")
        if schema is None or threshold is None:
            logger.warning("Skipping %s: a sidecar is missing or malformed.", directory)
            continue

        try:
            schema_version = int(schema["schema_version"])
            decision_threshold = float(threshold["threshold"])
            selected_model = str(threshold["selected_model"])
        except (KeyError, TypeError, ValueError):
            logger.warning("Skipping %s: a sidecar field is missing or not a number.", directory)
            continue

        runs.append(
            RunInfo(
                run_id=directory.name,
                created_at=created_at,
                schema_version=schema_version,
                selected_model=selected_model,
                threshold=decision_threshold,
                path=directory,
            )
        )
    return runs
