import numpy as np
import xarray as xr

from ca_knell.energy.auffhammer2022 import auffhammer_dds


def test_auffhammer_dds():
    """
    Test auffhammer_dds gives approx result for each degree-day variable
    """
    # Making input xr.Dataset
    n = 365
    start_time = "1995-01-01 12:00:00"
    end_time = "1995-12-31 12:00:00"
    tasmax_name = "tasmax"
    tasmin_name = "tasmin"
    x_tasmax = np.linspace(start=15.0, stop=30.0, num=n) + 2
    x_tasmin = np.linspace(start=15.0, stop=30.0, num=n)
    t = xr.cftime_range(
        start=start_time,
        end=end_time,
        freq="D",
        calendar="noleap",
    )
    in_ds = xr.Dataset(
        {tasmax_name: (["time"], x_tasmax), tasmin_name: (["time"], x_tasmin)},
        coords={"time": t},
    )

    out = auffhammer_dds(
        ds=in_ds,
        tasmax_name="tasmax",
        tasmin_name="tasmin",
    )

    # Check that out has three output degree-day variables, close to what we
    # might expect.
    np.testing.assert_allclose(
        out["cdd"].data,
        np.array([1946.949221], dtype="float32"),
    )
    np.testing.assert_allclose(
        out["cdd2"].data,
        np.array([231.879036], dtype="float32"),
    )
    np.testing.assert_allclose(
        out["hdd"].data,
        np.array([73.296443], dtype="float32"),
    )
