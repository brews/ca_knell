# make_binned_tas
# %pip install impactlab-tools==0.6.0
from impactlab_tools.utils.binning import binned_statistic_1d
import numpy as np
import xarray as xr


def _no_processing(ds: xr.Dataset) -> xr.Dataset:
    return ds


def annual_bincounts_1degree(ds, bins, variable):
    """frequency of days with variable in each bin across the year"""
    # Approach from the original project, adapted from Delgado code.
    assert (
        ((bins.min() <= ds[variable]) & (ds[variable] < bins.max())).all().item()
        is True
    ), "We got some crazy temperatures! min: {:0.2f}; max: {:0.2f}".format(
        ds[variable].min().item(), ds[variable].max().item()
    )

    # I'm not sure this impactlab_tools.utils.binning function is unittested
    variable_binned = binned_statistic_1d(
        ds[variable],
        dim="time",
        bins=bins,
        statistic="count",
    )

    binned_name = f"{variable}_binned"
    res = xr.Dataset({binned_name: variable_binned})
    res[binned_name].attrs.update(
        {
            "long_name": "annual frequency of days within each bin",
            "units": "count",
            "description": "bin counts based on a 365-day year (leap days excluded)",
        }
    )
    return res


def make_annual_bincounts_tas(ds: xr.Dataset) -> xr.Dataset:
    """
    Get annualized histogram 'tas_binned' from 'tas'. Input and output in Kelvin.
    """
    # Pre regionalization
    bins = np.arange(230, 341)  # Range we get histogram count for. NOTE: in Kelvin!
    out = (
        ds[["tas"]]
        .groupby("time.year")
        .map(annual_bincounts_1degree, bins=bins, variable="tas")
    )
    return out


make_tas_binned = TransformStrategy(
    preprocess=make_annual_bincounts_tas,
    postprocess=_no_processing,
)

# TODO: Rewrite as `make_annual_tas_histogram` using cf-convention style histogram instead.


############### Manually checking make_annual_bincounts_tas() ###############

# First simple test.

n = 365 * 10
# x = (np.sin(range(n))[..., np.newaxis, np.newaxis] + 290)  # Scale because tas is in Kelvin.
x = np.ones((n, 1, 1)) * 273.15
ds = xr.Dataset(
    {"tas": (["time", "lat", "lon"], x)},
    coords={
        "time": xr.cftime_range(
            start="2000-01-01", periods=n, freq="D", calendar="noleap"
        )
    },
)
test = make_annual_bincounts_tas(ds)
assert (test["tas_binned"].sel(groups="(273, 274]") == 365).all().item()

# Second, slightly more sophisticated test.

n = 365 * 10
# x = (np.sin(range(n))[..., np.newaxis, np.newaxis] + 290)  # Scale because tas is in Kelvin.
x = np.ones((n, 1, 1)) * 273.15
x[:365, ...] += 1
ds = xr.Dataset(
    {"tas": (["time", "lat", "lon"], x)},
    coords={
        "time": xr.cftime_range(
            start="2000-01-01", periods=n, freq="D", calendar="noleap"
        )
    },
)
test = make_annual_bincounts_tas(ds)
# We should see a bump in the higher group count for the first year and a bump in the lower group count for the second year.
assert (test["tas_binned"].sel(groups="(273, 274]", year=2000) == 0).item() and (
    test["tas_binned"].sel(groups="(274, 275]", year=2000) == 365
).item()
assert (test["tas_binned"].sel(groups="(273, 274]", year=2001) == 365).item() and (
    test["tas_binned"].sel(groups="(274, 275]", year=2001) == 0
).item()
