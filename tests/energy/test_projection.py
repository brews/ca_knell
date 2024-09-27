from muuttaa import project
import numpy as np
import pytest
import xarray as xr

from ca_knell.energy.projection import energy_impact_model


@pytest.fixture
def beta():
    out = xr.Dataset(
        {
            "electricity_logs": (
                ["region", "predictor"],
                np.array([[0.0313605, 0.00878708]]),
            ),
            "electricity_levels": (
                ["region", "predictor"],
                np.array([[0.43287864, 0.21389027]]),
            ),
            "gas_logs": (
                ["region", "predictor"],
                np.array([[-0.02961163, 0.09822069]]),
            ),
            "gas_levels": (
                ["region", "predictor"],
                np.array([[0.02047786, 0.19052568]]),
            ),
        },
        coords={
            "region": np.array(["foobar"]),
            "predictor": np.array(["CDD_WS", "HDD_WS"]),
        },
    )
    return out


@pytest.fixture
def dds():
    out = xr.Dataset(
        {
            "cdd": (["region", "year"], np.array([[2150.0, 2155.0]])),
            "cdd2": (["region", "year"], np.array([[200.0, 205.0]])),
            "hdd": (["region", "year"], np.array([[5.0, 10.0]])),
        },
        coords={
            "region": np.array(["foobar"]),
            "year": np.array([2020, 2050]),
        },
    )
    return out


def test_energy_impact_model(beta, dds):
    """
    Test that energy_impact_model runs through muuttaa.project with generally correct output.
    """
    expected = xr.Dataset(
        {
            "electricity_logs": (["region"], np.array([0.36132822])),
            "electricity_levels": (["region"], np.array([5.82092019])),
            "gas_logs": (["region"], np.array([0.61748154])),
            "gas_levels": (["region"], np.array([1.89903186])),
        },
        coords={
            "region": np.array(["foobar"]),
        },
    )

    actual = project(
        dds,  # Transformed input climate data.
        model=energy_impact_model,
        parameters=beta,
    )

    xr.testing.assert_allclose(actual, expected)
