from muuttaa import project
import numpy as np
import pytest
import xarray as xr

from ca_knell.labor.projection import rcspline, labor_impact_model


@pytest.fixture
def beta():
    b = np.arange(4).reshape((2, 1, 2))
    out = xr.Dataset(
        {"beta": (["risk_sector", "region", "tasmax_bin"], b)},
        coords={
            "region": np.array(["foobar"]),
            "tasmax_bin": np.array([20.5, 21.5]),
            "risk_sector": np.array(["low", "high"]),
        },
    )
    return out


@pytest.fixture
def histogram_tasmax():
    x = np.arange(4).reshape((1, 2, 2))
    x *= 10
    out = xr.Dataset(
        {"histogram_tasmax": (["region", "year", "tasmax_bin"], x)},
        coords={
            "region": np.array(["foobar"]),
            "year": np.array([2020, 2050]),
            "tasmax_bin": np.array([20.5, 21.5]),
        },
    )
    return out


def test_labor_impact_model(beta, histogram_tasmax):
    """
    Test that labor_impact_model runs through muuttaa.project with generally correct output.
    """
    expected = xr.Dataset(
        {
            "impact": (["region", "risk_sector"], np.array([[0.05479452, 0.2739726]])),
            "_effect": (
                ["region", "year", "risk_sector"],
                np.array([[[0.02739726, 0.08219178], [0.08219178, 0.35616438]]]),
            ),
        },
        coords={
            "region": np.array(["foobar"]),
            "year": np.array([2020, 2050]),
            "risk_sector": np.array(["low", "high"]),
        },
    )

    actual = project(
        histogram_tasmax,  # Transformed input climate data.
        model=labor_impact_model,
        parameters=beta,
    )

    xr.testing.assert_allclose(actual, expected)


def test_rcspline():
    """
    Basic test of rcspline spliney-ness.
    """
    da_in = xr.DataArray(range(25, 60))
    expected = xr.DataArray(
        [
            0.000e00,
            0.000e00,
            0.000e00,
            1.000e00,
            8.000e00,
            2.700e01,
            6.400e01,
            1.250e02,
            2.160e02,
            3.430e02,
            5.120e02,
            7.290e02,
            1.000e03,
            1.325e03,
            1.680e03,
            2.040e03,
            2.400e03,
            2.760e03,
            3.120e03,
            3.480e03,
            3.840e03,
            4.200e03,
            4.560e03,
            4.920e03,
            5.280e03,
            5.640e03,
            6.000e03,
            6.360e03,
            6.720e03,
            7.080e03,
            7.440e03,
            7.800e03,
            8.160e03,
            8.520e03,
            8.880e03,
        ]
    )

    actual = rcspline(da_in)
    xr.testing.assert_equal(actual, expected)
