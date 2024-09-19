"""
Logic for energy transformation and regionalization.
"""

from muuttaa import TransformationStrategy
import xarray as xr

from ca_knell.energy.auffhammer2022 import auffhammer_dds

# TODO: auffhammer_dds requires units degC, it's not clear about that. Should see if we can do some kind of unit conversion.


def _make_1998_2015_mean(ds: xr.Dataset) -> xr.Dataset:
    """
    Take mean of all variables from 1998 to 2015 (on time dim).
    Asserts this slicing will give 18 (annual) values across time.
    """
    ds_sub = ds.sel(time=slice("1998", "2015"))
    assert (
        ds_sub["time"].size == 18
    ), "'ds' should have 18 annual values when sliced from 1998 to 2015"
    return ds_sub.mean(dim="time")


# This is applied to PRISM and CMIP ensemble to create a delta for delta-adjustment.
make_auffhammer_degreedays_1998_2015_mean = TransformationStrategy(
    preprocess=auffhammer_dds, postprocess=_make_1998_2015_mean
)


def _make_21yrmean(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.rename(
        {"time": "year"}
    )  # Makes it consistent with time aggregations in other sectors. Not sure we want this or not.
    return ds.rolling(year=21, center=True).mean()


# This is the main transformation for the sectors impact model. Note, it does not do delta-adjustment of the variables.
make_auffhammer_degreedays_21yrmean = TransformationStrategy(
    preprocess=auffhammer_dds, postprocess=_make_21yrmean
)
