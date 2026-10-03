"""Provenance metadata reflects the installed distribution."""

from __future__ import annotations

from ..artifacts import TRACKED_PACKAGES, dependency_versions


def test_this_distribution_is_tracked_under_its_real_name():
    """Renaming the distribution without updating TRACKED_PACKAGES must fail here."""
    assert "fang" in TRACKED_PACKAGES


def test_every_tracked_package_resolves_to_a_version():
    """A tracked name that does not match an installed distribution records
    "not-installed", which silently corrupts a run's provenance."""
    versions = dependency_versions()
    unresolved = [name for name in TRACKED_PACKAGES if versions[name] == "not-installed"]
    assert unresolved == []
