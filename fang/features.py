"""Features fang derives from the schema's columns before any model sees them."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

# The Sun's radius in Earth radii, to turn planet radius over sqrt(depth) into R_star.
_SOLAR_RADIUS_REARTH = 109.1


def log_rho_transit(features: pd.DataFrame) -> pd.Series:
    """Log stellar-density mismatch implied by the transit shape.

    A planet's transit fixes the host's density through P / T^3 (Seager and
    Mallen-Ornelas 2003); scaling by R_star^2, taken from the fitted radius and the
    depth, turns it into a ratio against a main-sequence star of that radius.
    Eclipsing binaries and blends give durations the star cannot host, so their
    values sit apart from planets even when every single feature looks planet-like.
    """
    depth = features["depth_mean_per_transit"].where(features["depth_mean_per_transit"] > 0)
    duration_days = features["duration_hours"].where(features["duration_hours"] > 0) / 24.0
    period = features["period_days"].where(features["period_days"] > 0)
    r_star = features["planet_radius_rearth"] / np.sqrt(depth) / _SOLAR_RADIUS_REARTH
    value = np.log10(period / duration_days**3 * r_star**2)
    return value.replace([np.inf, -np.inf], np.nan)


DERIVED_FEATURES: dict[str, tuple[tuple[str, ...], Callable[[pd.DataFrame], pd.Series]]] = {
    "log_rho_transit": (
        ("period_days", "duration_hours", "planet_radius_rearth", "depth_mean_per_transit"),
        log_rho_transit,
    ),
}


def add_derived_features(features: pd.DataFrame, names: tuple[str, ...]) -> pd.DataFrame:
    """Return ``features`` with each named derived column appended, in order."""
    out = features.copy()
    for name in names:
        _, compute = DERIVED_FEATURES[name]
        out[name] = compute(features).astype(float)
    return out
