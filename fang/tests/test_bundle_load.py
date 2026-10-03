"""Loading a bundle that cannot be unpickled by this package."""

from __future__ import annotations

import pytest

from fang import training
from fang.errors import SchemaVersionError
from fang.training import ModelBundle


def _raise_missing(name):
    def fail(_source):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    return fail


def test_pre_rename_bundle_is_reported_as_needing_a_retrain(tmp_path, monkeypatch):
    """Pickles from the `src` package fail to unpickle before any version check runs."""
    (tmp_path / "model.joblib").write_bytes(b"stand-in")
    monkeypatch.setattr(training.joblib, "load", _raise_missing("src"))

    with pytest.raises(SchemaVersionError, match="retrain"):
        ModelBundle.load(tmp_path)


def test_pre_rename_submodule_is_reported_as_needing_a_retrain(tmp_path, monkeypatch):
    (tmp_path / "model.joblib").write_bytes(b"stand-in")
    monkeypatch.setattr(training.joblib, "load", _raise_missing("src.stacking"))

    with pytest.raises(SchemaVersionError, match="retrain"):
        ModelBundle.load(tmp_path)


def test_unrelated_missing_module_is_not_converted(tmp_path, monkeypatch):
    (tmp_path / "model.joblib").write_bytes(b"stand-in")
    monkeypatch.setattr(training.joblib, "load", _raise_missing("somethingelse"))

    with pytest.raises(ModuleNotFoundError, match="somethingelse"):
        ModelBundle.load(tmp_path)
