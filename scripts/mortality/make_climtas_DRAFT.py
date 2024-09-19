# make_climtas DRAFT

import numpy as np
import xarray as xr


###########################################################################
# To muuttaa
###########################################################################


class Registry:
    """
    Registry to collect callables that transform data and such so we can apply them en masse
    """

    def __init__(self, registered=None):
        if registered is None:
            registered = set()
        self._registered = registered

    def register(self, f):
        """
        Register f in this registry instance
        """
        self._registered.add(f)
        return f


###########################################################################
# CARB specific
###########################################################################

# transformations = Registry()  # Not sure we need this.


def make_annual_tas(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute annual average for 'tas'.
    """
    return ds[["tas"]].groupby("time.year").mean("time")


def make_30hbartlett_climtas(ds: xr.Dataset) -> xr.Dataset:
    """
    From annaual 'tas' compute 30-year half-Bartlett kernel average.

    Output variable is "climtas". This assumes input's "tas" variable is 
    annualized so if you pass in annual data, it will be a 30-year kernel. If 
    daily, it will be a 30 day kernel.
    """
    kernel_length = 30
    w = np.arange(kernel_length)
    weight = xr.DataArray(w / w.sum(), dims=["window"])
    da = ds["tas"].rolling(time=30).construct(time="window").dot(weight)
    # TODO: What to do for NaNs? What happened in carb analysis for climtas? Check 'gs://rhg-data/climate/aggregated/NASA/NEX-GDDP-BCSD-reformatted/California_2019_census_tracts_weighted_by_population/{scenario}/{model}/tas-bartlett30/tas-bartlett30_BCSD_CA-censustract2019_{model}_{scenario}_{version}_{year}.zarr'
    return da.to_dataset(name="climtas")


make_climtas = TransformationStrategy(
    preprocess=make_annual_tas,
    postprocess=make_30hbartlett_climtas,
)



############### Manually checking make_annual_tas() ###############

n = 365 * 10
x = np.sin(range(n))[..., np.newaxis, np.newaxis]
ds = xr.Dataset(
        {
            "tas": (
                ["time", "lat", "lon"], 
                x
            )
        }, 
        coords={"time": xr.cftime_range(start="2000-01-01", periods=n, freq="D", calendar="noleap")}
)
make_annual_tas(ds)


############### Manually checking make_30hbartlett_avg_tas() ###############

n = 30
# x = np.array([0] * 15 + [1] * 15)[..., np.newaxis, np.newaxis]
# x = np.ones((n, 1, 1))
x = np.sin(range(n))[..., np.newaxis, np.newaxis]
ds = xr.Dataset(
        {
            "tas": (
                ["time", "lat", "lon"], 
                x
            )
        }, 
        coords={"time": xr.cftime_range(start="2000", end=f"{2000+n}", freq="Y")}
)
make_30hbartlett_avg_tas(ds)

In [11]: ds["tas"].data[:, 0, 0] * weight.data
Out[11]: 
array([ 0.        ,  0.00193442,  0.00418068,  0.00097324, -0.0069591 ,
       -0.01102212, -0.00385401,  0.0105722 ,  0.01819509,  0.00852659,
       -0.01250623, -0.02528711, -0.01480201,  0.01255672,  0.03188162,
        0.02242372, -0.01058955, -0.03757186, -0.03107533,  0.00654636,
        0.04197449,  0.04039027, -0.00044765, -0.04474269, -0.04996294,
       -0.00760642,  0.04557821,  0.05936126,  0.01743761, -0.04424226])

In [12]: ds["tas"].data[:, 0, 0]
Out[12]: 
array([ 0.        ,  0.84147098,  0.90929743,  0.14112001, -0.7568025 ,
       -0.95892427, -0.2794155 ,  0.6569866 ,  0.98935825,  0.41211849,
       -0.54402111, -0.99999021, -0.53657292,  0.42016704,  0.99060736,
        0.65028784, -0.28790332, -0.96139749, -0.75098725,  0.14987721,
        0.91294525,  0.83665564, -0.00885131, -0.8462204 , -0.90557836,
       -0.13235175,  0.76255845,  0.95637593,  0.27090579, -0.66363388])

In [13]: weight.data
Out[13]: 
array([0.        , 0.00229885, 0.0045977 , 0.00689655, 0.0091954 ,
       0.01149425, 0.0137931 , 0.01609195, 0.0183908 , 0.02068966,
       0.02298851, 0.02528736, 0.02758621, 0.02988506, 0.03218391,
       0.03448276, 0.03678161, 0.03908046, 0.04137931, 0.04367816,
       0.04597701, 0.04827586, 0.05057471, 0.05287356, 0.05517241,
       0.05747126, 0.05977011, 0.06206897, 0.06436782, 0.06666667])

In [14]: (ds["tas"].data[:, 0, 0] * weight.data).sum()
Out[14]: np.float64(0.021863195217709103)


####################### read CARB segment weights ###########################
import pandas as pd

segment_weights = pd.read_csv(
    "gs://rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/source_data/segment_weights/California_2019_census_tracts_weighted_by_population_0p25.csv",
    dtype={"GEOID": str}  # Otherwise ID is read as int...
)
# check this to make sure it's doing what you think it is :)
segment_weights['longitude'] = (segment_weights["longitude"] + 180) % 360 - 180
segment_weights = segment_weights.to_xarray().rename_vars({'longitude': 'lon', 'latitude': 'lat', 'GEOID': 'region'})


def regionalize(ds: xr.Dataset, segment_weights: xr.Dataset) -> xr.Dataset:
    out = (
        ds.sel(
            lat=segment_weights["lat"],
            lon=segment_weights["lon"],
            # method="nearest",
            # tolerance=0.1,
        ) * segment_weights["weight"]
    ).groupby(segment_weights["region"]).sum()
    return out 


regionalize(cmip5["rcp45/ACCESS1-0/tas"].ds, segment_weights=segment_weights)

# Testing on cleaned cmip5 datatree
# gs://rhg-data-scratch/brews/237c2e81-3a70-4d9e-8b05-d30a845b106f/cmip5.zarr

# cmip5 = xr.backends.ZarrBackendEntrypoint.open_datatree("gs://impactlab-data-scratch/brews/5a0bd618-0480-4235-a036-63f0c28ae887/cmip5_concat.zarr", chunks={})


################################# What applying the transformations might look like #############################################

transformed = apply_transformations(
    downscaled_cmip5,
    regionalize=segment_weights,
    strategies=[
        make_climtas,
        make_binned_tasmax,
        make_binned_tas,
    ],
)
