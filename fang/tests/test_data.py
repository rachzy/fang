"""Data contract: filenames, strictness, labels, weights, and failure modes."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fang.data import (
    load_dataset,
    parse_source_filename,
    star_balanced_weights,
    validate_dataset,
)
from fang.errors import DataValidationError, EmptyDatasetError
from fang.tests.conftest import write_synthetic_dataset


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("filename", "star_id", "date"),
    [
        ("Kepler-452_20260823.csv", "Kepler-452", dt.date(2026, 8, 23)),
        ("Kepler-1704_20260823.csv", "Kepler-1704", dt.date(2026, 8, 23)),
        # Everything before the FINAL underscore is the star id.
        ("HD_209458_20240101.csv", "HD_209458", dt.date(2024, 1, 1)),
        ("TOI-700 d_19991231.csv", "TOI-700 d", dt.date(1999, 12, 31)),
    ],
)
def test_parses_star_id_before_final_underscore(filename, star_id, date):
    assert parse_source_filename(filename) == (star_id, date)


@pytest.mark.parametrize(
    "filename",
    [
        "Kepler-452.csv",  # no date
        "Kepler-452_2026823.csv",  # seven digits
        "Kepler-452_202608234.csv",  # nine digits
        "Kepler-452_20261323.csv",  # month 13
        "Kepler-452_20260230.csv",  # 30 February
        "_20260823.csv",  # empty star id
        "Kepler-452_20260823.txt",  # not a CSV
        "Kepler-452_abcdefgh.csv",  # non-numeric date
    ],
)
def test_rejects_malformed_filenames(filename):
    with pytest.raises(DataValidationError):
        parse_source_filename(filename)


def test_inverted_variant_keeps_the_base_star_id():
    assert parse_source_filename("KIC10014875_inverted_20261020.csv") == (
        "KIC10014875",
        dt.date(2026, 10, 20),
    )


def _add_inverted_copy(directory: Path, star: str, date: str) -> Path:
    source = next(directory.glob(f"{star}_*.csv"))
    frame = pd.read_csv(source)
    frame["candidate_label"] = "FALSE-POSITIVE"
    frame["fp_source"] = "inverted"
    frame["period_days"] = frame["period_days"] + 1.0
    target = directory / f"{star}_inverted_{date}.csv"
    frame.to_csv(target, index=False)
    return target


def test_inverted_file_is_grouped_with_its_star(tmp_path, schema):
    write_synthetic_dataset(tmp_path, n_stars=3, seed=1)
    _add_inverted_copy(tmp_path, "Synth-000", "20260105")

    dataset = load_dataset(tmp_path, mode="train", schema=schema)
    assert dataset.stars == ("Synth-000", "Synth-001", "Synth-002")
    assert len(dataset.files) == 4
    inverted = dataset.source_file == "Synth-000_inverted_20260105.csv"
    assert inverted.any()
    assert set(dataset.star_id[inverted]) == {"Synth-000"}
    assert (dataset.y[inverted] == 0).all()


def test_rejects_two_inverted_files_for_the_same_star(tmp_path, schema):
    write_synthetic_dataset(tmp_path, n_stars=3, seed=1)
    _add_inverted_copy(tmp_path, "Synth-000", "20260105")
    _add_inverted_copy(tmp_path, "Synth-000", "20260106")

    with pytest.raises(DataValidationError, match="multiple inverted files"):
        load_dataset(tmp_path, mode="train", schema=schema)


def test_koi_catalogue_files_are_ignored(tmp_path, schema):
    write_synthetic_dataset(tmp_path, n_stars=3, seed=1)
    (tmp_path / "Synth-000-koi.csv").write_text(
        "target,period_days,disposition\nK00001.01,3.2,FALSE POSITIVE\n"
    )

    dataset = load_dataset(tmp_path, mode="train", schema=schema)
    assert all(not p.name.endswith("-koi.csv") for p in dataset.files)
    assert "Synth-000-koi.csv" not in set(dataset.source_file)


@pytest.mark.parametrize("content", ["", "\n", "period_days,MES\n"])
def test_files_without_candidates_are_skipped(tmp_path, schema, content):
    write_synthetic_dataset(tmp_path, n_stars=3, seed=1)
    (tmp_path / "Synth-009_20260110.csv").write_text(content)

    dataset = load_dataset(tmp_path, mode="train", schema=schema)
    assert "Synth-009" not in dataset.stars
    assert dataset.summary()["n_files_without_candidates"] == 1


def test_directory_of_only_empty_files_fails(tmp_path, schema):
    (tmp_path / "Synth-000_20260101.csv").write_text("\n")
    with pytest.raises(EmptyDatasetError, match="no candidate rows"):
        load_dataset(tmp_path, mode="train", schema=schema)


def _mark_first_row_unknown(directory: Path) -> None:
    target = directory / "Synth-001_20260102.csv"
    frame = pd.read_csv(target)
    frame.loc[0, "candidate_label"] = "UNKNOWN"
    frame.to_csv(target, index=False)


def test_unknown_labels_are_dropped_for_training(tmp_path, schema):
    write_synthetic_dataset(tmp_path, n_stars=3, seed=1)
    before = len(load_dataset(tmp_path, mode="train", schema=schema))
    _mark_first_row_unknown(tmp_path)

    dataset = load_dataset(tmp_path, mode="train", schema=schema)
    assert len(dataset) == before - 1
    assert "UNKNOWN" not in set(dataset.frame["candidate_label"])
    assert len(dataset.weights) == len(dataset.y) == len(dataset.features) == len(dataset)
    assert dataset.summary()["n_unlabelled"] == 1


def test_unknown_labels_are_kept_for_prediction(tmp_path, schema):
    write_synthetic_dataset(tmp_path, n_stars=3, seed=1)
    before = len(load_dataset(tmp_path, mode="predict", schema=schema))
    _mark_first_row_unknown(tmp_path)

    dataset = load_dataset(tmp_path, mode="predict", schema=schema)
    assert len(dataset) == before
    assert dataset.y is None


def test_fp_source_is_known_but_never_a_feature(schema):
    assert "fp_source" in schema.known_columns
    assert "fp_source" not in schema.feature_columns


def test_rejects_two_files_for_the_same_star(tmp_path, schema):
    write_synthetic_dataset(tmp_path, n_stars=3, seed=1)
    source = tmp_path / "Synth-000_20260101.csv"
    source.with_name("Synth-000_20260202.csv").write_text(source.read_text())

    with pytest.raises(DataValidationError, match="multiple files"):
        load_dataset(tmp_path, mode="train", schema=schema)


# ---------------------------------------------------------------------------
# Directory-level failures
# ---------------------------------------------------------------------------
def test_empty_directory_fails_with_actionable_error(tmp_path):
    with pytest.raises(EmptyDatasetError, match="Add at least one"):
        load_dataset(tmp_path, mode="train")


def test_missing_directory_fails_with_actionable_error(tmp_path):
    with pytest.raises(EmptyDatasetError, match="not found"):
        load_dataset(tmp_path / "absent", mode="train")


# ---------------------------------------------------------------------------
# Strict schema
# ---------------------------------------------------------------------------
def test_unknown_column_is_rejected(tmp_path, schema):
    write_synthetic_dataset(tmp_path, n_stars=3, seed=2)
    target = tmp_path / "Synth-001_20260102.csv"
    frame = pd.read_csv(target)
    frame["brand_new_feature"] = 1.0
    frame.to_csv(target, index=False)

    with pytest.raises(DataValidationError, match="unknown columns"):
        load_dataset(tmp_path, mode="train", schema=schema)


def test_missing_feature_column_is_rejected(tmp_path, schema):
    write_synthetic_dataset(tmp_path, n_stars=3, seed=3)
    target = tmp_path / "Synth-002_20260103.csv"
    frame = pd.read_csv(target).drop(columns=["MES"])
    frame.to_csv(target, index=False)

    with pytest.raises(DataValidationError, match="missing required feature columns"):
        load_dataset(tmp_path, mode="train", schema=schema)


def test_train_mode_requires_label_and_status(tmp_path, schema):
    write_synthetic_dataset(tmp_path, n_stars=3, seed=4, include_supervision=False)
    with pytest.raises(DataValidationError, match="required for mode 'train'"):
        load_dataset(tmp_path, mode="train", schema=schema)


def test_predict_mode_does_not_require_supervision(tmp_path, schema):
    write_synthetic_dataset(tmp_path, n_stars=3, seed=5, include_supervision=False)
    dataset = load_dataset(tmp_path, mode="predict", schema=schema)
    assert dataset.y is None
    assert len(dataset) > 0


def test_duplicate_rows_are_rejected(tmp_path, schema):
    write_synthetic_dataset(tmp_path, n_stars=3, seed=6)
    target = tmp_path / "Synth-000_20260101.csv"
    frame = pd.read_csv(target)
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(target, index=False)

    with pytest.raises(DataValidationError, match="duplicate candidate rows"):
        load_dataset(tmp_path, mode="train", schema=schema)


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------
def test_nan_features_are_permitted(train_dir, schema):
    dataset = load_dataset(train_dir, mode="train", schema=schema)
    assert dataset.features.isna().to_numpy().any(), "fixture should contain missing values"


def test_infinite_values_are_rejected(tmp_path, schema):
    write_synthetic_dataset(tmp_path, n_stars=3, seed=8)
    target = tmp_path / "Synth-001_20260102.csv"
    frame = pd.read_csv(target)
    frame.loc[0, "snr_global"] = np.inf
    frame.to_csv(target, index=False)

    with pytest.raises(DataValidationError, match="Infinite values"):
        load_dataset(tmp_path, mode="train", schema=schema)


def test_entirely_empty_feature_is_rejected(tmp_path, schema):
    write_synthetic_dataset(tmp_path, n_stars=3, seed=9)
    for path in tmp_path.glob("*.csv"):
        frame = pd.read_csv(path)
        frame["depth_stability"] = np.nan
        frame.to_csv(path, index=False)

    with pytest.raises(DataValidationError, match="entirely empty"):
        load_dataset(tmp_path, mode="train", schema=schema)


def test_non_numeric_feature_is_rejected(tmp_path, schema):
    write_synthetic_dataset(tmp_path, n_stars=3, seed=10)
    target = tmp_path / "Synth-000_20260101.csv"
    frame = pd.read_csv(target)
    frame["MES"] = frame["MES"].astype(object)
    frame.loc[0, "MES"] = "high"
    frame.to_csv(target, index=False)

    with pytest.raises(DataValidationError, match="non-numeric"):
        load_dataset(tmp_path, mode="train", schema=schema)


@pytest.mark.parametrize("column", ["candidate_label", "detection_status"])
def test_invalid_supervision_values_are_rejected(tmp_path, schema, column):
    write_synthetic_dataset(tmp_path, n_stars=3, seed=11)
    target = tmp_path / "Synth-002_20260103.csv"
    frame = pd.read_csv(target)
    frame.loc[0, column] = "MAYBE"
    frame.to_csv(target, index=False)

    with pytest.raises(DataValidationError, match=f"Invalid {column}"):
        load_dataset(tmp_path, mode="train", schema=schema)


def test_target_mapping(train_dir, schema):
    dataset = load_dataset(train_dir, mode="train", schema=schema)
    labels = dataset.frame["candidate_label"].to_numpy()
    assert set(np.unique(dataset.y)) == {0, 1}
    assert np.array_equal(dataset.y == 1, labels == "CONFIRMED")
    assert np.array_equal(dataset.y == 0, labels == "FALSE-POSITIVE")


def test_every_row_is_labelled_regardless_of_detection_status(train_dir, schema):
    dataset = load_dataset(train_dir, mode="train", schema=schema)
    statuses = dataset.status
    assert "rejected" in statuses, "fixture should contain rejected rows"
    assert dataset.y.size == len(dataset), "every row, not just accepted ones, gets a label"


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
def test_every_star_carries_the_same_total_weight(train_dir, schema):
    dataset = load_dataset(train_dir, mode="train", schema=schema)
    totals = [dataset.weights[dataset.star_id == star].sum() for star in dataset.stars]
    assert np.allclose(totals, totals[0])
    assert pytest.approx(dataset.weights.mean()) == 1.0


def test_star_weights_normalise_within_any_subset():
    stars = np.array(["a", "a", "a", "b", "c", "c"], dtype=object)
    weights = star_balanced_weights(stars)
    assert pytest.approx(weights[stars == "a"].sum()) == weights[stars == "b"].sum()
    assert pytest.approx(weights[stars == "c"].sum()) == weights[stars == "b"].sum()
    assert pytest.approx(weights.mean()) == 1.0


def test_star_weights_recomputed_for_any_subset(train_dir, schema):
    """The equal-star property must hold for any fitting subset, not just the full set."""
    dataset = load_dataset(train_dir, mode="train", schema=schema)
    subset_stars = dataset.star_id[dataset.y == 1]
    weights = star_balanced_weights(subset_stars)
    totals = [weights[subset_stars == star].sum() for star in np.unique(subset_stars)]
    assert np.allclose(totals, totals[0])


def test_empty_weights():
    assert star_balanced_weights(np.array([], dtype=object)).shape == (0,)


# ---------------------------------------------------------------------------
# Provenance and ordering
# ---------------------------------------------------------------------------
def test_dataset_records_provenance_and_deterministic_order(train_dir, schema):
    dataset = load_dataset(train_dir, mode="train", schema=schema)
    assert len(dataset.file_checksums) == len(dataset.files)
    assert all(len(digest) == 64 for digest in dataset.file_checksums.values())

    files = [Path(f).name for f in dataset.source_file]
    assert files == sorted(files), "files must be concatenated in sorted order"
    for star in dataset.stars:
        mask = dataset.star_id == star
        assert list(dataset.row_index[mask]) == list(range(int(mask.sum())))


def test_validate_dataset_returns_summary(train_dir, schema):
    summary = validate_dataset(train_dir, mode="train", schema=schema)
    assert summary["n_stars"] == 12
    assert summary["n_features"] == len(schema.model_features)
    assert summary["n_positive"] > 0
    assert summary["n_negative"] > 0
    assert summary["status_counts"]["rejected"] > 0


def test_predict_mode_accepts_a_single_star_with_an_all_nan_feature(tmp_path, schema):
    """One star can legitimately lack a feature; imputers are fitted at training time."""
    write_synthetic_dataset(tmp_path, n_stars=1, seed=11, include_supervision=False)
    for path in tmp_path.glob("*.csv"):
        frame = pd.read_csv(path)
        frame["vshape_metric"] = np.nan
        frame.to_csv(path, index=False)

    dataset = load_dataset(tmp_path, mode="predict", schema=schema)
    assert len(dataset) > 0
