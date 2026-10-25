"""Derived features: the transit-implied stellar density."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fang.errors import ConfigError
from fang.features import add_derived_features, log_rho_transit
from fang.schema import load_schema


def _transit(period, duration_hours, radius, depth) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period_days": [period],
            "duration_hours": [duration_hours],
            "planet_radius_rearth": [radius],
            "depth_mean_per_transit": [depth],
        }
    )


def test_earth_around_the_sun_has_the_expected_density_ratio():
    # 365.25 d, 13 h, 1 R_earth with depth (1/109.1)^2 -> R_star = 1 R_sun.
    value = log_rho_transit(_transit(365.25, 13.0, 1.0, (1 / 109.1) ** 2)).iloc[0]
    expected = np.log10(365.25 / (13.0 / 24.0) ** 3)
    assert value == pytest.approx(expected)


def test_shorter_duration_raises_the_implied_density():
    slow = log_rho_transit(_transit(10.0, 4.0, 2.0, 4e-4)).iloc[0]
    fast = log_rho_transit(_transit(10.0, 2.0, 2.0, 4e-4)).iloc[0]
    assert fast - slow == pytest.approx(3 * np.log10(2))


@pytest.mark.parametrize(
    "row", [(10.0, 0.0, 2.0, 4e-4), (10.0, 3.0, 2.0, 0.0), (10.0, 3.0, 2.0, -1e-4)]
)
def test_unphysical_inputs_become_nan(row):
    assert np.isnan(log_rho_transit(_transit(*row)).iloc[0])


def test_derived_columns_are_appended_in_order():
    frame = _transit(10.0, 3.0, 2.0, 4e-4)
    out = add_derived_features(frame, ("log_rho_transit",))
    assert list(out.columns) == [*frame.columns, "log_rho_transit"]
    assert list(frame.columns) == list(_transit(1, 1, 1, 1).columns)


def test_packaged_schema_appends_the_derived_feature():
    schema = load_schema()
    assert schema.model_features[-1] == "log_rho_transit"


def test_unknown_derived_feature_is_rejected(tmp_path):
    import yaml

    payload = load_schema().to_dict()
    payload["derived_features"] = ["not_a_feature"]
    path = tmp_path / "schema.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ConfigError, match="Unknown derived_features"):
        load_schema(path)
