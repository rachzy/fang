"""Run provenance and immutable artifact directories."""

from __future__ import annotations

import datetime as dt
import json
import platform
import shutil
import subprocess
import sys
import uuid
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .errors import RunIdError

TRACKED_PACKAGES = (
    "fang",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "lightgbm",
    "catboost",
    "joblib",
    "pyyaml",
    "matplotlib",
    "typer",
)


def new_run_id() -> str:
    """A sortable, unique identifier for one training or evaluation run."""
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def validate_run_id(run_id: str) -> str:
    """Return ``run_id`` if the service would list a run under that name, else raise.

    ``list_runs`` skips any directory whose name does not match its
    ``RUN_ID_PATTERN``, so a run trained under such a name would never be served.
    """
    from .service import RUN_ID_PATTERN

    valid = RUN_ID_PATTERN.match(run_id) is not None
    if valid:
        try:
            dt.datetime.strptime(run_id[:16], "%Y%m%dT%H%M%SZ")
        except ValueError:
            valid = False
    if not valid:
        raise RunIdError(
            f"Invalid run id {run_id!r}: expected <UTC YYYYMMDDTHHMMSSZ>-<lowercase hex>, "
            f"e.g. {new_run_id()}. The model service only lists runs named this way."
        )
    return run_id


def check_run_destination(base: str | Path, run_id: str, overwrite: bool = False) -> Path:
    """Fail early when ``base/run_id`` already holds a run and ``overwrite`` is off."""
    directory = Path(base) / run_id
    if directory.exists() and not overwrite and (
        not directory.is_dir() or any(directory.iterdir())
    ):
        raise RunIdError(
            f"Run directory {directory} already exists. Runs are immutable: choose a "
            "different --run-id or --artifact-dir, or pass --overwrite to replace it."
        )
    return directory


def stage_run_dir(base: str | Path, run_id: str) -> Path:
    """Create a hidden sibling of ``base/run_id`` to write a run into before publishing it.

    Its name never matches the run-id pattern, so ``list_runs`` ignores it even
    if the process dies before :func:`publish_run_dir` moves it into place.
    """
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    staging = base / f".{run_id}.partial-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    return staging


def publish_run_dir(staging: Path, destination: Path, overwrite: bool = False) -> Path:
    """Atomically rename a fully written ``staging`` directory to ``destination``.

    With ``overwrite``, an existing run is moved aside first and deleted only
    once the new one is in place.
    """
    check_run_destination(destination.parent, destination.name, overwrite=overwrite)
    replaced: Path | None = None
    if destination.exists():
        if destination.is_dir() and not any(destination.iterdir()):
            destination.rmdir()
        else:
            replaced = destination.parent / f".{destination.name}.replaced-{uuid.uuid4().hex[:8]}"
            destination.rename(replaced)
    staging.rename(destination)
    if replaced is not None:
        if replaced.is_dir():
            shutil.rmtree(replaced, ignore_errors=True)
        else:
            replaced.unlink(missing_ok=True)
    return destination


def git_commit(repo: Path | None = None) -> dict[str, Any]:
    """Current commit and worktree cleanliness, or a reason it is unavailable."""
    cwd = str(repo or Path.cwd())
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (subprocess.SubprocessError, OSError) as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}


def dependency_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:  # pragma: no cover - optional at runtime
            versions[name] = "not-installed"
    return versions


def provenance(seed: int, run_id: str | None = None, repo: Path | None = None) -> dict[str, Any]:
    """The complete provenance block stored with every run."""
    return {
        "run_id": run_id or new_run_id(),
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "seed": seed,
        "git": git_commit(repo),
        "dependencies": dependency_versions(),
    }


class _Encoder(json.JSONEncoder):
    """Serialise NumPy scalars, arrays, and paths that appear in reports."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            value = float(o)
            return value if np.isfinite(value) else None
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, dt.datetime | dt.date):
            return o.isoformat()
        return super().default(o)


def _clean(value: Any) -> Any:
    """Normalise NumPy scalars and replace non-finite floats with null.

    JSON has no NaN or Infinity, so anything non-finite becomes ``null`` rather
    than the invalid literals ``json.dumps`` would otherwise emit.
    """
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_clean(payload), indent=2, cls=_Encoder, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return path


def write_yaml(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(_clean(payload), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return path


def write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
