"""
Logic for labor transformation and regionalization.
"""

from muuttaa import TransformationStrategy
import numpy as np
import xarray as xr
from xhistogram.xarray import histogram


def _no_processing(ds: xr.Dataset) -> xr.Dataset:
    return ds


def _make_tasmax_20yrmean_annual_histogram(ds: xr.Dataset) -> xr.Dataset:
    # TODO: Needs docstr.
    # TODO: This is similar to a tas histogram transformation in mortality. Maybe generalize some logic?
    bins = np.arange(230, 341)  # Range we get histogram count for. NOTE: in Kelvin!
    tasmax_annual_histogram = (
        ds["tasmax"].groupby("time.year").map(histogram, bins=[bins], dim=["time"])
    )

    tasmax_histogram_20yr = (
        tasmax_annual_histogram.rolling(year=20, center=True).mean().to_dataset()
    )
    return tasmax_histogram_20yr.astype("float32")


make_tasmax_20yrmean_annual_histogram = TransformationStrategy(
    preprocess=_make_tasmax_20yrmean_annual_histogram,
    postprocess=_no_processing,
)
