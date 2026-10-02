"""The service-facing surface consumed by the Antares apps/fang gRPC server."""

from __future__ import annotations

import json

import pytest
import yaml

from ..service import RunInfo, list_runs


def _write_run(base, run_id, schema_version=1, selected_model="lightgbm", threshold=0.5):
    """Create a run directory with the sidecars list_runs() reads."""
    directory = base / run_id
    directory.mkdir(parents=True)
    (directory / "model.joblib").write_bytes(b"not a real bundle")
    (directory / "feature_schema.yaml").write_text(
        yaml.safe_dump({"schema_version": schema_version})
    )
    (directory / "threshold.json").write_text(
        json.dumps({"selected_model": selected_model, "threshold": threshold})
    )
    return directory


def test_missing_artifact_dir_returns_empty_list(tmp_path):
    assert list_runs(tmp_path / "never_trained") == []


def test_lists_a_single_run(tmp_path):
    _write_run(tmp_path, "20260929T120000Z-abcd1234")
    runs = list_runs(tmp_path)

    assert len(runs) == 1
    assert isinstance(runs[0], RunInfo)
    assert runs[0].run_id == "20260929T120000Z-abcd1234"
    assert runs[0].schema_version == 1
    assert runs[0].selected_model == "lightgbm"
    assert runs[0].threshold == pytest.approx(0.5)


def test_created_at_is_derived_from_the_run_id(tmp_path):
    _write_run(tmp_path, "20260929T120000Z-abcd1234")
    assert list_runs(tmp_path)[0].created_at == "2026-09-29T12:00:00Z"


def test_runs_are_returned_newest_first(tmp_path):
    _write_run(tmp_path, "20260101T000000Z-aaaaaaaa")
    _write_run(tmp_path, "20261231T235959Z-bbbbbbbb")
    _write_run(tmp_path, "20260615T120000Z-cccccccc")

    assert [r.run_id[:8] for r in list_runs(tmp_path)] == [
        "20261231",
        "20260615",
        "20260101",
    ]


def test_directory_without_a_bundle_is_skipped(tmp_path):
    _write_run(tmp_path, "20260929T120000Z-abcd1234")
    aborted = tmp_path / "20260930T000000Z-deadbeef"
    aborted.mkdir()
    (aborted / "threshold.json").write_text("{}")

    runs = list_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].run_id == "20260929T120000Z-abcd1234"


def test_loose_files_in_the_artifact_dir_are_ignored(tmp_path):
    _write_run(tmp_path, "20260929T120000Z-abcd1234")
    (tmp_path / "notes.txt").write_text("scratch")

    assert len(list_runs(tmp_path)) == 1


def test_run_with_missing_sidecars_still_lists(tmp_path):
    """A bundle with no sidecars is listed with fallbacks, not skipped."""
    directory = tmp_path / "20260929T120000Z-abcd1234"
    directory.mkdir(parents=True)
    (directory / "model.joblib").write_bytes(b"bundle")

    run = list_runs(tmp_path)[0]
    assert run.run_id == "20260929T120000Z-abcd1234"
    assert run.selected_model == ""
