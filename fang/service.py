"""The surface the Antares apps/fang gRPC server consumes.

Kept deliberately small: everything here is called on a request path, so
nothing in this module unpickles a model bundle except prediction itself.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def list_runs(artifact_dir: Path | str = "models") -> list[RunInfo]:
    """Every trained run under ``artifact_dir``, newest first.

    A directory counts as a run when it contains ``model.joblib``; anything
    else is a partial or aborted run and is skipped. Returns an empty list
    when the directory does not exist, which is the state before the first
    model is ever trained.
    """
    base = Path(artifact_dir)
    if not base.is_dir():
        return []

    runs: list[RunInfo] = []
    for directory in sorted(base.iterdir(), reverse=True):
        if not directory.is_dir() or not (directory / BUNDLE_FILENAME).is_file():
            continue

        schema = _read_yaml(directory / "feature_schema.yaml")
        threshold = _read_json(directory / "threshold.json")

        runs.append(
            RunInfo(
                run_id=directory.name,
                created_at=_created_at_from_run_id(directory.name),
                schema_version=int(schema.get("schema_version", 0)),
                selected_model=str(threshold.get("selected_model", "")),
                threshold=float(threshold.get("threshold", 0.0)),
                path=directory,
            )
        )
    return runs
