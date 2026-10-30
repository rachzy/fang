"""Packaged named configs: listing, resolution by name, and shipping in the wheel."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from importlib import resources
from pathlib import Path

import pytest
import yaml

from fang.config import available_configs, load_config
from fang.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_available_configs_start_with_default_and_include_the_recipes():
    names = available_configs()
    assert names[0] == "default"
    assert "recipe-v3" in names
    assert names[1:] == sorted(names[1:])


def test_default_name_is_the_packaged_config():
    assert load_config(name="default").data == load_config().data


def test_named_config_resolves_to_its_packaged_file():
    packaged = resources.files("fang.resources").joinpath("configs", "recipe-v3.yaml")
    expected = yaml.safe_load(packaged.read_text(encoding="utf-8"))
    assert load_config(name="recipe-v3").data == expected


def test_every_listed_config_is_valid():
    for name in available_configs():
        load_config(name=name)


@pytest.mark.parametrize("name", ["nope", "../config", "configs/recipe-v3", ".hidden"])
def test_unknown_config_name_lists_the_available_ones(name):
    with pytest.raises(ConfigError, match="Available configs: default, recipe-v3"):
        load_config(name=name)


def test_path_and_name_are_mutually_exclusive(tmp_path):
    with pytest.raises(ConfigError, match="not both"):
        load_config(tmp_path / "x.yaml", name="default")


def test_named_configs_ship_in_the_wheel_and_load_from_it(tmp_path):
    """An install from git only sees what the wheel carries, not the source tree."""
    wheel_builder = pytest.importorskip("hatchling.builders.wheel")
    builder = wheel_builder.WheelBuilder(str(REPO_ROOT))
    wheel = Path(next(iter(builder.build(directory=str(tmp_path / "dist"), versions=["standard"]))))

    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
        assert "fang/resources/config.yaml" in members
        assert "fang/resources/configs/recipe-v3.yaml" in members
        archive.extractall(tmp_path / "site")

    site = tmp_path / "site"
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]); import fang; "
        "assert fang.__file__.startswith(sys.argv[1]), fang.__file__; "
        "from fang.config import available_configs, load_config; "
        "names = available_configs(); [load_config(name=n) for n in names]; print(names)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(site)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "'recipe-v3'" in result.stdout
