"""The service-facing surface consumed by the Antares apps/fang gRPC server."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
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


def test_run_with_missing_sidecars_is_skipped(tmp_path):
    """A bundle with no sidecars has no known threshold, so it cannot be served."""
    directory = tmp_path / "20260929T120000Z-abcd1234"
    directory.mkdir(parents=True)
    (directory / "model.joblib").write_bytes(b"bundle")

    assert list_runs(tmp_path) == []


def test_directory_that_is_not_a_run_id_is_skipped(tmp_path):
    """`models/latest` sorts above every timestamped id; it must not become newest."""
    _write_run(tmp_path, "20260929T120000Z-abcd1234")
    _write_run(tmp_path, "latest")

    runs = list_runs(tmp_path)
    assert [r.run_id for r in runs] == ["20260929T120000Z-abcd1234"]


def test_corrupt_sidecar_is_skipped_without_crashing_the_listing(tmp_path):
    """One truncated sidecar must not take the whole registry down with it."""
    _write_run(tmp_path, "20260101T000000Z-aaaaaaaa")
    broken = _write_run(tmp_path, "20261231T235959Z-bbbbbbbb")
    (broken / "threshold.json").write_text('{"threshold": 0.5, "selected')

    runs = list_runs(tmp_path)
    assert [r.run_id for r in runs] == ["20260101T000000Z-aaaaaaaa"]


def test_sidecar_that_is_not_a_mapping_is_skipped(tmp_path):
    """Valid JSON of the wrong shape must not raise AttributeError on .get."""
    broken = _write_run(tmp_path, "20260929T120000Z-abcd1234")
    (broken / "threshold.json").write_text("[]")

    assert list_runs(tmp_path) == []


def test_non_numeric_threshold_is_skipped(tmp_path):
    """A threshold that will not cast must not reach a caller as 0.0."""
    broken = _write_run(tmp_path, "20260929T120000Z-abcd1234")
    (broken / "threshold.json").write_text('{"selected_model": "lightgbm", "threshold": "high"}')

    assert list_runs(tmp_path) == []


def test_invalid_utf8_sidecar_is_skipped_without_crashing(tmp_path):
    """read_text raises UnicodeDecodeError, a ValueError that OSError does not catch."""
    _write_run(tmp_path, "20260101T000000Z-aaaaaaaa")
    broken = _write_run(tmp_path, "20261231T235959Z-bbbbbbbb")
    (broken / "threshold.json").write_bytes(b'{"threshold": \xff\xfe}')

    assert [r.run_id for r in list_runs(tmp_path)] == ["20260101T000000Z-aaaaaaaa"]


def test_infinite_schema_version_is_skipped(tmp_path):
    """json.loads accepts the bare Infinity literal; int(inf) raises OverflowError."""
    broken = _write_run(tmp_path, "20260929T120000Z-abcd1234")
    (broken / "feature_schema.yaml").write_text("schema_version: .inf")

    assert list_runs(tmp_path) == []


def test_huge_threshold_is_skipped(tmp_path):
    broken = _write_run(tmp_path, "20260929T120000Z-abcd1234")
    (broken / "threshold.json").write_text(
        '{"selected_model": "lightgbm", "threshold": 1e400}'
    )

    assert list_runs(tmp_path) == []


def test_nan_threshold_is_skipped(tmp_path):
    """A NaN threshold makes every comparison false, classifying nothing."""
    broken = _write_run(tmp_path, "20260929T120000Z-abcd1234")
    (broken / "threshold.json").write_text(
        '{"selected_model": "lightgbm", "threshold": NaN}'
    )

    assert list_runs(tmp_path) == []


def test_boolean_schema_version_is_skipped(tmp_path):
    """int(True) is 1, so a boolean would otherwise pass as a version."""
    broken = _write_run(tmp_path, "20260929T120000Z-abcd1234")
    (broken / "feature_schema.yaml").write_text("schema_version: true")

    assert list_runs(tmp_path) == []


def test_missing_or_empty_selected_model_is_skipped(tmp_path):
    """str(None) was yielding the literal string "None"."""
    null_model = _write_run(tmp_path, "20260929T120000Z-abcd1234")
    (null_model / "threshold.json").write_text(
        '{"selected_model": null, "threshold": 0.5}'
    )
    assert list_runs(tmp_path) == []

    empty_model = _write_run(tmp_path, "20260930T120000Z-abcd1234")
    (empty_model / "threshold.json").write_text(
        '{"selected_model": "", "threshold": 0.5}'
    )
    assert list_runs(tmp_path) == []


from ..errors import DataValidationError, EmptyDatasetError  # noqa: E402
from ..service import _safe_star_id, latest_run, predict_features  # noqa: E402


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory, fast_config):
    """Train one small real model and write it to an artifact directory."""
    from fang.tests.conftest import write_synthetic_dataset

    from ..data import load_dataset
    from ..training import train_model, write_training_artifacts

    data_dir = write_synthetic_dataset(tmp_path_factory.mktemp("train"))
    dataset = load_dataset(data_dir, mode="train")
    run = train_model(dataset=dataset, config=fast_config, run_evaluation=False)

    artifacts = tmp_path_factory.mktemp("artifacts")
    write_training_artifacts(run, artifacts)
    return artifacts / run.bundle.run_id


@pytest.fixture
def feature_rows(schema):
    """Two candidate rows carrying exactly the schema's feature columns."""
    return [
        {name: 1.0 for name in schema.feature_columns},
        {name: 2.0 for name in schema.feature_columns},
    ]


def test_predicts_every_row(trained_run, feature_rows):
    predictions = predict_features(feature_rows, trained_run)

    assert len(predictions) == 2
    assert all("prob_stack" in row for row in predictions)
    assert all("prediction" in row for row in predictions)


def test_predictions_carry_the_model_run_id(trained_run, feature_rows):
    predictions = predict_features(feature_rows, trained_run)
    assert predictions[0]["model_run_id"] == trained_run.name


def test_star_id_is_preserved_in_the_output(trained_run, feature_rows):
    predictions = predict_features(feature_rows, trained_run, star_id="KIC-8120608")
    assert predictions[0]["star_id"] == "KIC-8120608"


def test_excluded_columns_are_accepted_and_ignored(trained_run, feature_rows, schema):
    """Shaula emits columns no model may see; they must not break prediction."""
    rows = [dict(row) for row in feature_rows]
    for row in rows:
        row["t0"] = 131.5
        row["mes_threshold_used"] = 7.1
        row["duration_days"] = 0.12

    predictions = predict_features(rows, trained_run)
    assert len(predictions) == 2
    assert all(c not in schema.feature_columns for c in ("t0", "mes_threshold_used"))


def test_missing_feature_column_raises_naming_it(trained_run, feature_rows):
    rows = [dict(row) for row in feature_rows]
    for row in rows:
        del row["period_days"]

    with pytest.raises(DataValidationError, match="period_days"):
        predict_features(rows, trained_run)


def test_empty_rows_raise_rather_than_returning_nothing(trained_run):
    with pytest.raises(EmptyDatasetError):
        predict_features([], trained_run)


def test_latest_run_picks_the_newest(tmp_path):
    _write_run(tmp_path, "20260101T000000Z-aaaaaaaa")
    _write_run(tmp_path, "20261231T235959Z-bbbbbbbb")

    assert latest_run(tmp_path).run_id == "20261231T235959Z-bbbbbbbb"


def test_latest_run_with_no_runs_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No trained runs"):
        latest_run(tmp_path)


def test_a_real_trained_run_is_listed(trained_run):
    """The validation in list_runs must accept what write_training_artifacts produces."""
    runs = list_runs(trained_run.parent)

    assert [r.run_id for r in runs] == [trained_run.name]
    assert runs[0].selected_model
    assert runs[0].schema_version == 1
    assert 0.0 < runs[0].threshold <= 1.0


def test_latest_run_says_when_every_run_was_unservable(tmp_path):
    _write_run(tmp_path, "20260101T000000Z-aaaaaaaa", threshold=float("nan"))

    with pytest.raises(FileNotFoundError, match="none was servable"):
        latest_run(tmp_path)


def test_star_id_with_path_separators_cannot_escape_the_staging_dir(
    trained_run, feature_rows, monkeypatch
):
    """star_id arrives from a network request and must not reach a path unsanitised.

    Spies on the staged CSV path itself: the returned star_id is the caller's raw
    value, so only the path proves the sanitising.
    """
    staged: list[Path] = []
    real_to_csv = pd.DataFrame.to_csv

    def spy(self, path_or_buf=None, *args, **kwargs):
        staged.append(Path(path_or_buf))
        return real_to_csv(self, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", spy)

    predictions = predict_features(
        feature_rows, trained_run, star_id="../../escaped/KIC-1"
    )

    assert len(predictions) == 2
    assert len(staged) == 1
    destination = staged[0]
    assert destination.parent.name.startswith("fang-predict-")
    assert destination.parent.parent == Path(tempfile.gettempdir())
    assert destination.name.startswith("escaped-KIC-1_")


def test_star_id_is_returned_as_supplied(trained_run, feature_rows):
    predictions = predict_features(feature_rows, trained_run, star_id="KIC 8120608")
    assert {p["star_id"] for p in predictions} == {"KIC 8120608"}


def test_star_id_that_sanitises_to_nothing_falls_back():
    assert _safe_star_id("../..") == "unknown"


def test_ordinary_star_id_survives_sanitisation_recognisably():
    assert _safe_star_id("KIC 8120608") == "KIC-8120608"


@pytest.mark.parametrize("missing", [float("nan"), None], ids=["nan", "none"])
def test_single_star_request_with_a_missing_feature_is_scored(
    trained_run, feature_rows, schema, missing
):
    """A lone star's missing feature must not fail the request."""
    row = dict(feature_rows[0])
    row[schema.feature_columns[0]] = missing

    predictions = predict_features([row], trained_run)

    assert len(predictions) == 1
