import datetime
import os
import uuid

from ca_knell.energy.transformation import make_auffhammer_degreedays_1998_2015_mean, make_auffhammer_degreedays_21yrmean
from ca_knell.io import open_carb_segmentweights
from dask_gateway import GatewayCluster
import datatree as dt
import geopandas as gpd
from muuttaa import apply_transformations
import xarray as xr

JUPYTER_IMAGE = os.environ.get("JUPYTER_IMAGE")
UID = str(uuid.uuid4())
START_TIME = datetime.datetime.now(datetime.timezone.utc).isoformat()
PRISM_TASMAX_URL = "gs://rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data/climate/prism-tmax-ca-20220510.zarr"
PRISM_TASMIN_URL = "gs://rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data/climate/prism-tmin-ca-20220510.zarr"
PRISM_SEGMENT_WEIGHTS_URL = "gs://rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/source_data/segment_weights/California_2019_census_tracts_weighted_by_population_0p0417.csv"
CMIP5_URI = "gs://rhg-data-scratch/brews/16c38870-f245-472c-b09c-3899a4501716/cmip5_concat.zarr"
CARB_SEGMENT_WEIGHTS_URI = "gs://rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/source_data/segment_weights/California_2019_census_tracts_weighted_by_population_0p25.csv"
CA_TRACTS_PATH = "/gcs/rhg-data/impactlab-rhg/spatial/shapefiles/source/us_census/TIGER2019/TRACT/tl_2019_06_tract"

print(f"{JUPYTER_IMAGE=}\n{START_TIME=}\n{UID=}")


# Process PRISM to output HDD, CDD, CDD2

# Read, merge PRISM tmin, tmax
prism = (
        xr.open_mfdataset([PRISM_TASMIN_URL, PRISM_TASMAX_URL], engine="zarr")
        .sel(time=slice("1998-01-01", "2015-12-31"))
        .rename_vars({"tmax": "tasmax", "tmin": "tasmin"})
)
prism_segment_weights = open_carb_segmentweights(PRISM_SEGMENT_WEIGHTS_URL)


def fuzzy_prism_regionalization(x: xr.Dataset):
    """
    Custom regionalization function for PRISM field, subset population-weight, and aggregate to California census tract regions.

    Weight and climate grid lat/lon matching needs to be fuzzy enough to compensate for offset grid without reaching into adjacent grid points.
    """
    region_sel = x.sel(
        lat=prism_segment_weights._data["lat"],
        lon=prism_segment_weights._data["lon"],
        method="nearest", tolerance=0.03,
    ) # For PRISM 2.5 degree minute grid.
    out = (region_sel * prism_segment_weights._data["weight"]).groupby(prism_segment_weights._data["region"]).sum()
    return out


segment_weights = open_carb_segmentweights(CARB_SEGMENT_WEIGHTS_URI)
cmip5 = dt.open_datatree(CMIP5_URI, engine="zarr", chunks={}).chunk(
    {"time": 365 * 20, "lat": 90, "lon": 10}  # Using small lon chunks otherwise DDs calculation will take loads of space.
)
test_ds = cmip5["rcp45/ACCESS1-0"].ds


cluster = GatewayCluster(
    worker_image=JUPYTER_IMAGE, scheduler_image=JUPYTER_IMAGE, profile="micro"
)
client = cluster.get_client()
print(client.dashboard_link)
cluster.scale(50)


# First we need to create a "delta": comparing PRISM and GCM DDS over a period.
# The difference is a delta, which we then apply to the transformed data in
# future projecting GCM runs as an additive adjustment, before impact projection.
dds_baseline = apply_transformations(
    prism.sel(time=slice("1998", "2015")),  # Time slice data early to make calculation faster.
    strategies=[make_auffhammer_degreedays_1998_2015_mean],
    regionalize=fuzzy_prism_regionalization,
)
# dds_baseline = dds_baseline.compute()

dds_gcm = apply_transformations(
    test_ds.sel(time=slice("1998", "2015")) - 273.15,  # Time slice data early to make calculation faster. Convert K to degC.
    strategies=[make_auffhammer_degreedays_1998_2015_mean],
    regionalize=segment_weights,
)
# dds_gcm = dds_gcm.compute()

dds_delta = dds_baseline - dds_gcm
dds_delta = dds_delta.compute()

transformed = apply_transformations(
    test_ds,
    strategies=[make_auffhammer_degreedays_21yrmean],
    regionalize=segment_weights,
)
# Apply delta shift to adjust DD variables before impact modeling.
# TODO: Likely need to do this for each variable? What if non-DD variables in datasets?
transformed_adj = transformed + dds_delta
transformed_adj = transformed_adj.compute()

cluster.shutdown()

# TODO: Convert PRISM data calendar to noleap
# TODO: Calculate Auffhammer-style degree-days (HDD, CDD, CDD2).
# (See cell 9 of https://gitlab.com/rhodium/impactlab-rhg/carb-cvm/data-prep/-/blob/2bb3e6c8b587befafa6f62b186dc2ad83248c3db/notebooks/make_dds.ipynb, and the code https://gitlab.com/rhodium/impactlab-rhg/carb-cvm/data-prep/-/blob/2bb3e6c8b587befafa6f62b186dc2ad83248c3db/src/dprep/climate/dds.py)

########################################################################################################################
# Plot choropleths. For diagnostics.

# Geometry to plot PRISM values in census tract polygons as Choropleths.
tracts = gpd.read_file(
    CA_TRACTS_PATH,
    dtype={"GEOID": "str"}
).rename(columns={"GEOID": "region"})

# Cleanup. Removing tracts without land.
water_tracts_mask = (tracts["ALAND"] <= 0.0)
tracts = tracts[~water_tracts_mask]

# Dividing prism_dds by 365 to get daily average values.
tract_prism = tracts.merge((dds_baseline / 365).to_dataframe(), on="region")
# tract_prism = tracts.merge(transformed_adj.sel(year="2020").to_dataframe(), on="region")
# tract_prism = tracts.merge((dds_delta / 365).to_dataframe(), on="region")   # This matches plots in the original carb notebooks.

for variable in dds_baseline.data_vars:
    tract_prism.plot(
        column=variable, legend=True, figsize=(8, 8), legend_kwds={"label": variable}
    )



########################################################################################################################

@dask.delayed
def apply_delta_shift(ds: xr.Dataset, delta: xr.Dataset) -> Delayed:
    """
    Delayed function to apply "delta-correction"/bias-correction to an annual-sum degree-days field.

    Note that 'delta' is multiplied by 365 because the input argument is a daily average, when an
    annual sum is required. This assumes the fields are from a "noleap" calendar. Output
    is also cast to float32.
    """
    # Multiplying delta by 365 because it was a daily average, need average annual sum.
    with xr.set_options(keep_attrs=True):
        out = (ds + (delta * 365)).astype("float32")

    # Passing main Dataset's root and variable attrs.
    out.attrs.update(ds.attrs.copy())
    for v in ds.data_vars:
        if v in out.data_vars:
            out[v].attrs.update(ds[v].attrs.copy())

    return out


@dask.delayed
def calc_dailyavg_delta(ds: xr.Dataset, baseline: xr.Dataset) -> Delayed:
    """Calculate a daily average delta field from a baseline and simulation

    This is used for a simple delta-shift bias correction. The baseline should be a daily average
    degree day while the ds is an annual sum degree day. Yes, I know that's a pain.
    Output is cast to float32. This assumes the data fields are from a "noleap" calendar.
    """
    with xr.set_options(keep_attrs=True):
        out = (baseline - (ds / 365)).astype("float32")

        # Passing main Dataset's root and variable attrs.
    out.attrs.update(ds.attrs.copy())
    for v in ds.data_vars:
        if v in out.data_vars:
            out[v].attrs.update(ds[v].attrs.copy())

    return out
