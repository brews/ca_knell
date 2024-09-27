from muuttaa import project
import numpy as np
import xarray as xr

from ca_knell.energy.valuation import energy_valuation_model


def test_energy_valuation_model():
    """
    Test energy_valuation_model can run through muuttaa.project.
    Does basic check of output.
    """
    expected = xr.Dataset(
        {
            "electricity_consumption": (
                ["valuation", "region"],
                np.array([[1.52079445e3], [0.487434118], [1.18857381e-5]]),
            ),
            "gas_consumption": (
                ["valuation", "region"],
                np.array([[3.30823372e3], [1.06033132], [2.58554333e-5]]),
            ),
        },
        coords={
            "region": np.array(["foobar"]),
            "valuation": np.array(["damages_total", "damages_pc", "damages_pincome"]),
        },
    )

    impact = xr.Dataset(
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

    params = xr.Dataset(
        {
            "kWh_price": 0.20206,  # $/kwh
            "therm_price": 1.3473034062372766,  # $/therm
            "pop": (["region"], [3120.0]),
            "hhcount": (["region"], [1293]),
            "pci": (["region"], [41010.0]),
        },
        coords={
            "region": np.array(["foobar"]),
            "risk_sector": np.array(["low", "high"]),
        },
    )

    actual = project(impact, model=energy_valuation_model, parameters=params)

    xr.testing.assert_allclose(actual, expected)
