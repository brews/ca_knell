from muuttaa import project
import numpy as np
import xarray as xr

from ca_knell.labor.valuation import labor_valuation_model


def test_labor_valuation_model():
    """
    Test labor_valuation_model can run through muuttaa.project.
    Does basic check of output.
    """
    expected = xr.Dataset(
        {
            "damages_total": (
                ["region", "risk_sector"],
                np.array([[-1955778.04548872, -1237618.61804511]]),
            ),
            "damages_pc": (
                ["region", "risk_sector"],
                np.array([[-626.85193766, -396.67263399]]),
            ),
            "damages_pincome": (
                ["region", "risk_sector"],
                np.array([[-0.00538996, -0.00341077]]),
            ),
        },
        coords={
            "region": np.array(["foobar"]),
            "risk_sector": np.array(["low", "high"]),
        },
    )

    impact = xr.Dataset(
        {
            "impact": (["region", "risk_sector"], np.array([[1.0, 4.0]])),
        },
        coords={
            "region": np.array(["foobar"]),
            "risk_sector": np.array(["low", "high"]),
        },
    )

    params = xr.Dataset(
        {
            "total_employed": (["region"], [1596.0]),
            "pop": (["risk_sector", "region"], [[1378.0], [218.0]]),
            # Randomly selected from 2019 labor earnings income (salary + wages + self employed income)
            "wages": (["region"], [2.792691e8]),
            "total_tract_population": (["region"], [3120.0]),
            "pci": (["region"], [116300.0]),
        },
        coords={
            "region": np.array(["foobar"]),
            "risk_sector": np.array(["low", "high"]),
        },
    )

    actual = project(impact, model=labor_valuation_model, parameters=params)

    xr.testing.assert_allclose(actual, expected)
