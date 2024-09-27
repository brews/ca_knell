from muuttaa import project
import numpy as np
import pytest
import xarray as xr

from ca_knell.labor.projection import (
    rcspline,
    labor_impact_model,
    labor_impact_model_gamma,
)


@pytest.fixture
def beta():
    b = np.arange(4).reshape(2, 1, 2)
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
def gamma():
    g_m = np.arange(2 * 2, dtype="float64").reshape(2, 2)
    g_m += 0.1
    g_sampled = np.arange(2 * 2 * 2, dtype="float64").reshape(2, 2, 2)
    g_sampled += 0.1
    out = xr.Dataset(
        {
            "gamma_mean": (["risk_sector", "predname"], g_m),
            "gamma_sampled": (
                ["sample", "risk_sector", "predname"],
                g_sampled,
            ),
        },
        coords={
            "predname": ["tasmax", "tasmax_rcspline1"],
            "risk_sector": np.array(["low", "high"]),
            "sample": [0, 1],
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


@pytest.fixture
def tasmax():
    x = np.arange(2).reshape(
        (
            1,
            2,
        )
    )
    x *= 10
    out = xr.Dataset(
        {"tasmax": (["region", "year"], x)},
        coords={
            "region": np.array(["foobar"]),
            "year": np.array([2020, 2050]),
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


def test_labor_impact_model_gamma_mean(gamma, histogram_tasmax, tasmax):
    """
    Test that labor_impact_model_gamma runs through muuttaa.project.
     Checks for generally correct output using mean gamma as input.
    """
    # Build up what we expect output to be.
    ex_i = np.array([[0.23013699, 4.83287671]])
    ex_e = np.array([[[0.05890411, 1.2369863], [0.2890411, 6.06986301]]])
    expected = xr.Dataset(
        {
            "impact": (["region", "risk_sector"], ex_i),
            "_effect": (["region", "year", "risk_sector"], ex_e),
        },
        coords={
            "region": np.array(["foobar"]),
            "year": np.array([2020, 2050]),
            "risk_sector": np.array(["low", "high"]),
        },
    )

    # Combine data for input
    transformed_input = xr.merge([histogram_tasmax, tasmax])
    # Using rename_vars because model expects "gamma" variable, not "gamma_mean".
    params = gamma[["gamma_mean"]].rename_vars({"gamma_mean": "gamma"})

    actual = project(
        transformed_input,
        model=labor_impact_model_gamma,
        parameters=params,
    )

    xr.testing.assert_allclose(actual, expected)


def test_labor_impact_model_gamma_sampled(gamma, histogram_tasmax, tasmax):
    """
    Test that labor_impact_model_gamma runs through muuttaa.project.
     Checks for generally correct output using sampled gamma as input.
    """
    # Build up what we expect output to be.
    ex_i = np.array([[[0.23013699, 4.83287671], [9.43561644, 14.03835616]]])
    ex_e = np.array(
        [
            [
                [[0.05890411, 1.2369863], [2.41506849, 3.59315068]],
                [[0.2890411, 6.06986301], [11.85068493, 17.63150685]],
            ]
        ]
    )
    expected = xr.Dataset(
        {
            "impact": (["region", "sample", "risk_sector"], ex_i),
            "_effect": (["region", "year", "sample", "risk_sector"], ex_e),
        },
        coords={
            "region": np.array(["foobar"]),
            "year": np.array([2020, 2050]),
            "risk_sector": np.array(["low", "high"]),
            "sample": [0, 1],
        },
    )

    # Combine data for input
    transformed_input = xr.merge([histogram_tasmax, tasmax])
    # Using rename_vars because model expects "gamma" variable, not "gamma_sampled".
    params = gamma[["gamma_sampled"]].rename_vars({"gamma_sampled": "gamma"})

    actual = project(
        transformed_input,
        model=labor_impact_model_gamma,
        parameters=params,
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
