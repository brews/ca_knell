# apply_transforms test

# %pip install muuttaa==0.1.0

import datetime
import os
from os import PathLike
from io import BufferedIOBase
import uuid
from typing import Any
# from functools import wraps
# from collections.abc import Callable, Sequence


from dask_gateway import GatewayCluster
import datatree as dt
from muuttaa import TransformationStrategy, SegmentWeights, apply_transformations
import numpy as np
import xarray as xr
from xhistogram.xarray import histogram


JUPYTER_IMAGE = os.environ.get("JUPYTER_IMAGE")
UID = str(uuid.uuid4())
START_TIME = datetime.datetime.now(datetime.timezone.utc).isoformat()
print(f"{JUPYTER_IMAGE=}\n{START_TIME=}\n{UID=}")

CMIP5_URL = "gs://impactlab-data-scratch/brews/5515562a-9488-41d8-a655-2ae1ddd62f90/cmip5_concat.zarr"
CARB_SEGMENT_WEIGHTS_URL = "gs://rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/source_data/segment_weights/California_2019_census_tracts_weighted_by_population_0p25.csv"

OUT_URL = f"{os.environ['CIL_SCRATCH_PREFIX']}/{os.environ['JUPYTERHUB_USER']}/{UID}/cmip5_transformed_test.zarr"


########### Defining here because I'm too lazy to import muuttaa

# from collections.abc import Callable, Sequence
# from dataclasses import dataclass
# from typing import Protocol, Any


# class Regionalizer(Protocol):
#     """
#     Use to regionalize gridded, n-dimentional dataset.

#     The key is that it can be loaded with invariants or state information (e.g. weights, spatial geometry) before it is used in the transformation process to regionalize gridded data.
#     """

#     def regionalize(self, ds: xr.Dataset) -> xr.Dataset:
#         ...


# # TODO: Can we still read docstr of callables through their attributes after theyve been passed to an instantiated TransformationStrategy?
# @dataclass(frozen=True)
# class TransformationStrategy:
#     """
#     Named, tranformation steps applied to input gridded data, pre/post regionalization, to create a derived variable as output.

#     These steps should be general. They may contain logic for sanity checks on inputs and outputs, calculating derived variables and climate indices, adding or checking metadata or units. Avoid including logic for cleaning, or harmonizing input data, especially if it is specific to a single project's usecase. Generally avoid using a single strategy to output multiple unrelated variables.
#     """
#     # TODO: This is setting mutable as default, use a factory or something to fix.
#     preprocess: Callable[[xr.Dataset], xr.Dataset] = _no_processing
#     postprocess: Callable[[xr.Dataset], xr.Dataset] = _no_processing


# # Use class for segment weights because we're making assumptions/enforcements about the weight data's content and interactions...
# class SegmentWeights:
#     """
#     Segment weights to regionalize regularly-gridded data
#     """

#     def __init__(self, weights: xr.Dataset):
#         target_variables = ("lat", "lon", "weight", "region")
#         missing_variables = [v for v in target_variables if v not in weights.variables]
#         if missing_variables:
#             raise ValueError(
#                 f"input weights is missing required {missing_variables} variable(s)"
#             )
#         self._data = weights

#     def regionalize(self, x: xr.Dataset) -> xr.Dataset:
#         """
#         Regionalize input gridded data
#         """
#         # TODO: See how this errors in different common scenarios. What happens on the unhappy path?
#         region_sel = x.sel(lat=self._data["lat"], lon=self._data["lon"])
#         out = (region_sel * self._data["weight"]).groupby(self._data["region"]).sum()
#         # TODO: Maybe drop lat/lon and set 'region' as dim/coord? I feel like we can do this because we're asking weights to strictly match input's lat/lon. Maybe make this a req of segment weights we're reading in?
#         return out

#     def __call__(self, x: xr.Dataset) -> xr.Dataset:
#         return self.regionalize(x)


# def _default_transform_merge(x: Sequence[xr.Dataset]) -> xr.Dataset:
#     return xr.merge(x)


# def apply_transformations(
#     gridded: xr.Dataset,
#     *,
#     strategies: Sequence[TransformationStrategy],
#     regionalize: Callable[[xr.Dataset], xr.Dataset],
#     merge_transformed: Callable[[Sequence[xr.Dataset]], xr.Dataset] | None = None,
# ) -> xr.Dataset:
#     """
#     Apply multiple regionalized transformations output to a single Dataset.
#     """
#     strategies = tuple(strategies)

#     if merge_transformed is None:
#         merge_transformed = _default_transform_merge

#     transformed = []
#     for s in strategies:
#         preprocessed = s.preprocess(gridded)
#         regionalized = regionalize(preprocessed)
#         postprocessed = s.postprocess(regionalized)
#         transformed.append(postprocessed)

#     return merge_transformed(transformed)

########### Custom stuff we need just for CARB


def open_carb_segmentweights(
    url: str | PathLike[Any] | BufferedIOBase,
) -> SegmentWeights:
    """Open SegmentWeights from CARB project weights file"""
    import pandas as pd

    sw = pd.read_csv(
        url,
        dtype={"GEOID": str},  # Otherwise ID is read as int...
    )
    sw["longitude"] = (sw["longitude"] + 180) % 360 - 180
    sw = sw.to_xarray().rename_vars(
        {"longitude": "lon", "latitude": "lat", "GEOID": "region"}
    )
    return SegmentWeights(sw)


########### Define transformations


def _no_processing(ds: xr.Dataset) -> xr.Dataset:
    return ds


def make_annual_tas(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute annual average for 'tas'.
    """
    return ds[["tas"]].groupby("time.year").mean("time")


def make_30hbartlett_climtas(ds: xr.Dataset) -> xr.Dataset:
    """
    From annaual 'tas' compute 30-year half-Bartlett kernel average.

    Output variable is "climtas". This assumes input's "tas" has "year"
    time dim.
    """
    kernel_length = 30
    w = np.arange(kernel_length)
    weight = xr.DataArray(w / w.sum(), dims=["window"])
    da = ds["tas"].rolling(year=30).construct(year="window").dot(weight)
    # TODO: What to do for NaNs? What happened in carb analysis for climtas? Check 'gs://rhg-data/climate/aggregated/NASA/NEX-GDDP-BCSD-reformatted/California_2019_census_tracts_weighted_by_population/{scenario}/{model}/tas-bartlett30/tas-bartlett30_BCSD_CA-censustract2019_{model}_{scenario}_{version}_{year}.zarr'
    return da.to_dataset(name="climtas").astype("float32")


make_climtas = TransformationStrategy(
    preprocess=make_annual_tas,
    postprocess=make_30hbartlett_climtas,
)


def _make_tas_20yrmean_annual_histogram(ds: xr.Dataset) -> xr.Dataset:
    bins = np.arange(230, 341)  # Range we get histogram count for. NOTE: in Kelvin!
    tas_annual_histogram = (
        ds["tas"].groupby("time.year").map(histogram, bins=[bins], dim=["time"])
    )

    ## Needed to rechunk to avoid dask killing workers in the next step.
    # tas_annual_histogram = tas_annual_histogram.chunk({"lat": 180, "lon": 180})
    tas_histogram_20yr = (
        tas_annual_histogram.rolling(year=20, center=True).mean().to_dataset()
    )
    return tas_histogram_20yr.astype("float32")


make_tas_20yrmean_annual_histogram = TransformationStrategy(
    preprocess=_make_tas_20yrmean_annual_histogram,
    postprocess=_no_processing,
)


################### Run stuff on single dataset.

segment_weights = open_carb_segmentweights(CARB_SEGMENT_WEIGHTS_URL)


cluster = GatewayCluster(
    worker_image=JUPYTER_IMAGE, scheduler_image=JUPYTER_IMAGE, profile="micro"
)
client = cluster.get_client()
print(client.dashboard_link)
cluster.scale(50)

cmip5 = dt.open_datatree(CMIP5_URL, engine="zarr", chunks={}).chunk(
    {"time": 365 * 20, "lat": 90, "lon": 90}
)
test_ds = cmip5["rcp45/ACCESS1-0"].ds

transformed = apply_transformations(
    test_ds,
    regionalize=segment_weights,
    strategies=[
        make_climtas,
        make_tas_20yrmean_annual_histogram,
    ],
)

transformed.chunk({"year": -1, "region": -1, "tas_bin": 30}).to_zarr(OUT_URL, mode="w")
print(OUT_URL)

cluster.scale(0)

################### Run stuff on leaves of the cmip5 datatree.


cluster.scale(50)

cmip5 = dt.open_datatree(CMIP5_URL, engine="zarr", chunks={}).chunk(
    {"time": 365 * 20, "lat": 90, "lon": 90}
)

transformed = cmip5.map_over_subtree(
    apply_transformations,
    regionalize=segment_weights,
    strategies=[
        make_climtas,
        make_tas_20yrmean_annual_histogram,
    ],
)

transformed.chunk({"year": -1, "region": -1, "tas_bin": 30}).to_zarr(OUT_URL, mode="w")
print(OUT_URL)
# gs://rhg-data-scratch/brews/f3d91aef-5b54-415b-b375-fa15dfaf0dfe/cmip5_transformed_test.zarr
# Took under 45 minutes with 50 workers. Size is 1.5 GiB.

cluster.scale(0)


cluster.shutdown()
