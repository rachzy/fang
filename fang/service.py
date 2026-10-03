"""The surface the Antares apps/fang gRPC server consumes.

Kept deliberately small: everything here is called on a request path, so
nothing in this module unpickles a model bundle except prediction itself.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .errors import EmptyDatasetError
from .predict import predict_dataset
from .schema import load_schema

logger = logging.getLogger(__name__)

BUNDLE_FILENAME = "model.joblib"
RUN_ID_PATTERN = re.compile(
    r"^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})"
    r"T(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})Z-[0-9a-f]+$"
)

_UNSAFE_STAR_ID_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


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
    except (OSError, ValueError):
        return None
    try:
        parsed = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    except (ValueError, yaml.YAMLError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _as_int(value: Any) -> int:
    """Coerce to int, rejecting bools that int() would silently accept."""
    if isinstance(value, bool):
        raise TypeError("expected a number, got a boolean")
    return int(value)


def _as_finite_float(value: Any) -> float:
    """Coerce to float, rejecting bools, NaN and infinity.

    A NaN threshold makes every comparison false, so nothing is ever
    classified positive — an unusable model served as a working one.
    """
    if isinstance(value, bool):
        raise TypeError("expected a number, got a boolean")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"expected a finite number, got {number!r}")
    return number


def list_runs(artifact_dir: Path | str = "models") -> list[RunInfo]:
    """Every trained run under ``artifact_dir`` that is safe to serve, newest first.

    A run qualifies when it has a bundle, a run id this build recognises, and
    readable sidecars carrying a numeric threshold in ``[0, 1]`` (both ends
    valid) and a schema version in ``[1, 2**31)`` that equals the installed
    schema's, so a listed run is one ``load_model`` will accept and whose
    fields fit an int32. Everything else is skipped (malformed or
    out-of-range sidecars warn; sidecars that are not written yet, and
    directories whose name is not a run id, are logged at debug): a
    hand-named directory such as
    ``models/latest`` would otherwise sort above every timestamped id and
    become the newest run, and a truncated sidecar would otherwise be served
    as ``threshold=0.0``, a legitimate value that classifies every candidate
    as positive. Returns an empty list when the directory does not exist,
    which is the state before the first model is ever trained.
    """
    base = Path(artifact_dir)
    if not base.is_dir():
        return []

    installed_version = load_schema().schema_version
    runs: list[RunInfo] = []
    for directory in sorted(base.iterdir(), reverse=True):
        if not directory.is_dir() or not (directory / BUNDLE_FILENAME).is_file():
            continue

        created_at = _created_at_from_run_id(directory.name)
        if not created_at:
            logger.debug("Skipping %s: not a run id this build recognises.", directory)
            continue

        schema_path = directory / "feature_schema.yaml"
        threshold_path = directory / "threshold.json"
        if not schema_path.is_file() or not threshold_path.is_file():
            logger.debug("Skipping %s: sidecars not written yet.", directory)
            continue

        schema = _read_mapping(schema_path)
        threshold = _read_mapping(threshold_path)
        if schema is None or threshold is None:
            logger.warning("Skipping %s: a sidecar is malformed.", directory)
            continue

        try:
            schema_version = _as_int(schema["schema_version"])
            decision_threshold = _as_finite_float(threshold["threshold"])
            selected_model = threshold["selected_model"]
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            logger.warning("Skipping %s: bad sidecar field (%s).", directory, error)
            continue

        if not isinstance(selected_model, str) or not selected_model:
            logger.warning("Skipping %s: selected_model is missing or empty.", directory)
            continue

        if not 0.0 <= decision_threshold <= 1.0:
            logger.warning(
                "Skipping %s: threshold %s is outside [0, 1].", directory, decision_threshold
            )
            continue

        if not 1 <= schema_version < 2**31:
            logger.warning(
                "Skipping %s: schema_version %s is outside [1, 2**31).", directory, schema_version
            )
            continue

        if schema_version != installed_version:
            logger.warning(
                "Skipping %s: schema_version %s does not match the installed schema (%s).",
                directory,
                schema_version,
                installed_version,
            )
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


def latest_run(artifact_dir: Path | str = "models") -> RunInfo:
    """The newest trained run under ``artifact_dir``."""
    runs = list_runs(artifact_dir)
    if runs:
        return runs[0]

    base = Path(artifact_dir)
    if base.is_dir() and any(base.iterdir()):
        raise FileNotFoundError(
            f"No trained runs found under {artifact_dir}: entries exist but none "
            "was servable. Check the warning log for why each was skipped."
        )
    raise FileNotFoundError(f"No trained runs found under {artifact_dir}.")


def _safe_star_id(star_id: str) -> str:
    """Reduce a caller-supplied id to something safe as a filename stem.

    ``star_id`` reaches this module from a network request and becomes part
    of a path, so separators and traversal segments must not survive.
    """
    cleaned = _UNSAFE_STAR_ID_CHARS.sub("-", star_id).strip(".-")
    return cleaned or "unknown"


def predict_features(
    rows: list[dict[str, Any]],
    model_dir: Path | str,
    *,
    star_id: str = "unknown",
) -> list[dict[str, Any]]:
    """Score in-memory candidate rows with the model bundle in ``model_dir``.

    The rows are staged as one per-star CSV in a temporary directory so that
    :func:`fang.predict.predict_dataset` applies the same strict schema
    validation it applies to files on disk. Extra columns the schema excludes
    are tolerated and dropped; missing feature columns raise.

    ``star_id`` is returned as supplied. Only the internal staging filename
    is sanitised (see :func:`_safe_star_id`), so callers can join on it.
    """
    if not rows:
        raise EmptyDatasetError("predict_features was given no rows to score.")
    frame = pd.DataFrame(rows)

    with tempfile.TemporaryDirectory(prefix="fang-predict-") as staging:
        # load_dataset requires filenames shaped "<star_id>_YYYYMMDD.csv".
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d")
        destination = Path(staging) / f"{_safe_star_id(star_id)}_{stamp}.csv"
        frame.to_csv(destination, index=False)

        predictions = predict_dataset(model=Path(model_dir), data_dir=Path(staging))

    records = predictions.to_dict(orient="records")
    for record in records:
        # The staging filename is sanitised and internal; callers get their own id back.
        record["star_id"] = star_id
    return records
