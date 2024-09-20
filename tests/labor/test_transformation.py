from muuttaa import apply_transformations, SegmentWeights
import numpy as np
import pytest
import xarray as xr

from ca_knell.labor.transformation import make_tasmax_20yrmean_annual_histogram


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


def test_make_tasmax_20yrmean_annual_histogram(basic_segment_weights):
    """
    Test make_tasmax_20yrmean_annual_histogram transformation runs through muuttaa.apply_transformation with basic_segment_weights
    Does some basic/lazy checks for correctness in the output.
    """
    expected = xr.Dataset(
        {
            "histogram_tasmax": (
                ["region", "year", "tasmax_bin"],
                np.zeros((1, 22, 110)),
            )
        },
        coords={
            "region": np.array(["foobar"]),
            "year": np.arange(2000, 2022),
            "tasmax_bin": np.arange(230.5, 340.5),
        },
    )
    expected["histogram_tasmax"].loc[
        {"region": "foobar", "tasmax_bin": 274.5, "year": 2010}
    ] = 365.0
    expected["histogram_tasmax"].loc[
        {"region": "foobar", "tasmax_bin": 274.5, "year": 2011}
    ] = 365.0
    expected["histogram_tasmax"].loc[
        {"region": "foobar", "tasmax_bin": 274.5, "year": 2012}
    ] = 346.79998779

    ds_in = xr.Dataset(
        {
            "tasmax": (
                ["lon", "lat", "time"],
                np.ones((1, 1, 7666), dtype=np.float32) + 273.15,
            )
        },
        coords={
            "lon": [1.0],
            "lat": [0.0],
            "time": xr.date_range(
                "2000-01-01", "2021-01-01", freq="1D", calendar="noleap"
            ),
        },
    )

    actual = apply_transformations(
        ds_in,
        strategies=[make_tasmax_20yrmean_annual_histogram],
        regionalize=basic_segment_weights,
    )
    xr.testing.assert_allclose(actual, expected)
