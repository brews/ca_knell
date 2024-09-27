from muuttaa import apply_transformations, SegmentWeights
import numpy as np
import pytest
import xarray as xr

from ca_knell.energy.transformation import (
    make_auffhammer_degreedays_1998_2015_mean,
    make_auffhammer_degreedays_21yrmean,
)


@pytest.fixture
def basic_segment_weights():
    sw = SegmentWeights(
        weights=xr.Dataset(
            {
                "region": (["idx"], ["foobar"]),
                "weight": (["idx"], [1.0]),
                "lon": (["idx"], [1.0]),
                "lat": (["idx"], [0.0]),
            },
        )
    )
    return sw


@pytest.fixture
def tasmax_tasmin():
    # Auffhammer decimal degree transformations make strong assumptions
    # about data frequency and time window so we need to carefully set this up.
    n_t = 365 * 22  # Daily, noleap, 22 years.
    x_tasmin = np.linspace(start=15.0, stop=30.0, num=n_t, dtype=np.float32)
    x_tasmax = x_tasmin + 2
    t = xr.date_range(
        start="1995-01-01 12:00:00",
        end="2016-12-31 12:00:00",
        freq="1D",
        calendar="noleap",
    )
    out = xr.Dataset(
        {
            "tasmax": (["lon", "lat", "time"], x_tasmax.reshape(1, 1, n_t)),
            "tasmin": (["lon", "lat", "time"], x_tasmin.reshape(1, 1, n_t)),
        },
        coords={
            "lon": [1.0],
            "lat": [0.0],
            "time": t,
        },
    )
    return out


def test_make_auffhammer_degreedays_1998_2015_mean(
    basic_segment_weights, tasmax_tasmin
):
    """
    Test make_auffhammer_degreedays_1998_2015_mean transformation runs through muuttaa.apply_transformation with basic_segment_weights
    Does some basic/lazy checks for correctness in the output.
    """
    expected = xr.Dataset(
        {
            "cdd": (["region"], np.array([2141.5033227])),
            "cdd2": (["region"], np.array([205.67305628])),
            "hdd": (["region"], np.array([7.11630932])),
        },
        coords={"region": np.array(["foobar"])},
    )

    actual = apply_transformations(
        tasmax_tasmin,
        strategies=[make_auffhammer_degreedays_1998_2015_mean],
        regionalize=basic_segment_weights,
    )
    xr.testing.assert_allclose(actual, expected)


def test_make_auffhammer_degreedays_21yrmean(basic_segment_weights, tasmax_tasmin):
    """
    Test make_auffhammer_degreedays_21yrmean transformation runs through muuttaa.apply_transformation with basic_segment_weights
    Does some basic/lazy checks for correctness in the output.
    """
    expected = xr.Dataset(
        {
            "cdd": (["region", "year"], np.array([[1836.91288749, 2050.86910008]])),
            "cdd2": (["region", "year"], np.array([[176.2911911, 245.58801216]])),
            "hdd": (["region", "year"], np.array([[75.81420545, 41.20585702]])),
        },
        coords={
            "region": np.array(["foobar"]),
            "year": np.array([2005, 2006]),
        },
    )

    actual = apply_transformations(
        tasmax_tasmin,
        strategies=[make_auffhammer_degreedays_21yrmean],
        regionalize=basic_segment_weights,
    )
    xr.testing.assert_allclose(actual.dropna("year"), expected)
