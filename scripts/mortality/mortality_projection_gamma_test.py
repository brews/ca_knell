# Mortality impact projection
# This version uses gammas as input to the projection model.

%pip install muuttaa==0.1.0


import datatree
from ca_knell.io import read_csvv
from muuttaa import Projector, project
import numpy as np
import pandas as pd
import xarray as xr



TRANSFORMED_CMIP_URI = "gs://rhg-data-scratch/brews/4e1d23be-fa0b-4354-beb9-069239352015/cmip5_transformed_test.zarr"
CMIP5_SMME_WGTS_URI = "gs://impactlab-data-scratch/brews/cmip5_smme_weights.zarr"
INCOME_PATH = "/gcs/rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/output_data/income_adjusted.nc4"
CSVV_PATH = "gs://rhg-data-scratch/brews/Agespec_interaction_GMFD_POLY-4_TINV_CYA_NW_w1.csvv"  # Moved into scratch from https://gitlab.com/rhodium/impactlab-rhg/carb-cvm/beta-generation/-/blob/330bf3b949881749e6f3d13c88349be0d65bbfb8/csvvs/Agespec_interaction_GMFD_POLY-4_TINV_CYA_NW_w1.csvv
POP_PATH = "gs://rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/output_data/population_age_binned.csv"


def xr_clip_bidirectional_monotonic_increase(da, dim, lmmt=10, ummt=30):
    """Performs U Clipping of an unclipped response function for all regions
    simultaneously, centered around each region's Minimum Mortality Temperature (MMT).

    Parameters
    ----------
    da : DataArray
        xarray.DataArray of unclipped response functions
    dim : str
        Dimension name along which clipping will be applied. Clipping will be applied
        along dimensions `dim` for all other dimensions independently (e.g. a unique
        minimum point will be found for each combination of any other dimensions in the
        data).
    lmmt : int
        Lower bound of the temperature range in which a minimum mortality temperature
        will be searched for. Default is 10.
    ummt : int
        Upper bound of the temperature range in which a minimum mortality temperature
        will be searched for. Default is 30.

    Returns
    -------
    clipped : xarray DataArray of the regions response functioned centered on its mmt
    """
    # identify mmt within defined range
    min_idx = da.where(da[dim] >= lmmt).where(da[dim] <= ummt).idxmin(dim=dim)
    min_val = da.sel({dim: min_idx})

    # subtract mmt beta value
    diffed = da - min_val

    # mask data on each side of mmt and take cumulative max in each direction
    rhs = (
        diffed.where(diffed[dim] >= min_idx)
        .rolling({dim: len(diffed[dim])}, min_periods=1)
        .max()
    )
    lhs = (
        diffed.where(diffed[dim] <= min_idx)
        .sel({dim: slice(None, None, -1)})
        .rolling({dim: len(diffed[dim])}, min_periods=1)
        .max()
        .sel({dim: slice(None, None, -1)})
    )

    # combine the arrays where they've been masked
    clipped = rhs.fillna(lhs)

    # run some basic checks to make sure we haven't messed things up:

    # ensure that the min index hasn't changed (but account for the fact that the min
    # value may not be unique anymore by testing that the new min index takes on the
    # same value as the old min index takes on in the clipped data
    new_min_idx = clipped.idxmin(dim=dim)
    assert (
        clipped.sel({dim: new_min_idx}) == clipped.sel({dim: min_idx})
    ).all().item() is True, "previous min index is no longer the min value"

    assert (clipped == 0).any(
        dim=dim
    ).all().item() is True, "0 not present in array after differencing min value"
    assert (
        clipped.min(dim=dim) == 0
    ).all().item() is True, "Min value not equal to 0 for all dims"
    assert (
        clipped.where(clipped[dim] >= new_min_idx).fillna(0).diff(dim=dim) >= 0
    ).all().item() is True, (
        "not increasing weakly monotonically right of min value after clipping"
    )
    assert (
        clipped.where(clipped[dim] <= new_min_idx).fillna(0).diff(dim=dim) <= 0
    ).all().item() is True, (
        "not declining weakly monotonically left of min value after clipping"
    )

    return clipped


def build_mortality_betas(csvv, temps, loggdppc, climtas, clipped=True):
    """Creates an Xarray DataSet of mortality beta values at each temperature bin value that can be
    plotted as response functions. Caluctes these beta values by taking the dot product of:
    (predictor * covariate * gamma * age dummy), where
        predictor = {tas, tas2, tas3, tas4}:  temperature or temperature polynomial
        covariate = {loggdppc, climtas, 1}: potential interaction term
        gamma: coefficient for term
        age dummy = {0, 1}: ensures we only use the term for the respective age group


    Parameters
    ----------
    csvv : CSVV file
        File containing mortality regression output, including coefficient values, predictors,
        and covariates needed to construct betas
    temps : xarray DataArray
        Data array containing temperature bins. Temperature becomes one of the coordinates in
        the output DataSet
    loggdppc : DataArray or float
        Value(s) that will be used for the log of GDP per capita covariate. Can be a single value
        (in which temperature will be the only coordinate) or a DataArray of loggdppc values for
        each GEOID (resulting in an additional GEOID coordinate being added)
    climtas : DataArray or float
        Value(s) that will be used for the long run climate covariate. Can be a single value
        (in which temperature will be the only coordinate) or a DataArray of climtas values for
        each GEOID (resulting in an additional GEOID coordinate being added)
    clipped : boolean
        If true, applies clipping to values in response function

    Returns
    -------
    combined : xarray DataSet of beta values which define the mortality response (in deaths per 100k)
    at each binned degree for either a single region or many which contains a variable of betas for
    each age group.
    """

    assert type(climtas) == type(
        loggdppc
    )  # is True, "climtas and loggdppc should be objects of equal size"
    if type(climtas) == xr.DataArray:
        assert (
            climtas.size == loggdppc.size
        )  # is True, "climtas and loggdppc DataArray coordinates should match"

    # set dictionaries which will be used to replace respective csvv prednames and covarnames with values
    cdic = {"climtas": climtas, "loggdppc": loggdppc, "1": 1}
    tdic = {"tas": temps, "tas2": temps**2, "tas3": temps**3, "tas4": temps**4}

    # create age group dummy arrays for binned dot product
    ones = np.ones(12)
    zeros = np.zeros(12)
    age1 = np.concatenate((ones, zeros, zeros), axis=None)
    age2 = np.concatenate((zeros, ones, zeros), axis=None)
    age3 = np.concatenate((zeros, zeros, ones), axis=None)

    # takes the dot product for the equation above for each age group
    a1 = sum(
        [
            tdic[pred] * cdic[cov] * gam * age
            for pred, cov, gam, age in zip(
                csvv["prednames"], csvv["covarnames"], csvv["gamma"], age1
            )
        ]
    )
    a2 = sum(
        [
            tdic[pred] * cdic[cov] * gam * age
            for pred, cov, gam, age in zip(
                csvv["prednames"], csvv["covarnames"], csvv["gamma"], age2
            )
        ]
    )
    a3 = sum(
        [
            tdic[pred] * cdic[cov] * gam * age
            for pred, cov, gam, age in zip(
                csvv["prednames"], csvv["covarnames"], csvv["gamma"], age3
            )
        ]
    )

    # combines into single Dataset
    combined = xr.Dataset({"age1": a1, "age2": a2, "age3": a3})

    # centers around MMT and performs clipping
    if clipped:
        combined = combined.map(xr_clip_bidirectional_monotonic_increase, dim="temp")

    return combined


smme_weights = xr.open_dataset(CMIP5_SMME_WGTS_URI, engine="zarr", chunks={})["smme_weight"]
cmip5 = datatree.open_datatree(TRANSFORMED_CMIP_URI, engine="zarr", chunks={}).sel(year=[2020, 2050])
test_ds = cmip5["rcp45/ACCESS1-0"].to_dataset().sel(year=2020)

csvv = read_csvv(CSVV_PATH)
income = xr.open_dataset(INCOME_PATH).rename({"GEOID": "region"})
# Encode "region" as Object rather than <U11 so consistent with transformed climate data region.
income = income.assign_coords(region=(income["region"].astype("O")))
############### All from https://gitlab.com/rhodium/impactlab-rhg/carb-cvm/beta-generation/-/blob/330bf3b949881749e6f3d13c88349be0d65bbfb8/src/genbeta/core.py

# Unit conversion K to degree C.
# TODO: This should be part of transformation step.
test_ds = test_ds.assign_coords(tas_bin=(test_ds["tas_bin"] - 273))
test_ds["climtas"] -= 273

inc_var = 'loggdppc_residual_scaled'
# load csvv

# 1 degree C bins from -43 to 67 centered at the center to match the climate generation process 
# bins = np.arange(-42.5, 67.5, 1)
bins = test_ds["tas_bin"]
temps = test_ds["tas_bin"]

temps = xr.DataArray(bins, dims=['temp'], coords=[bins])
# # Original CARB code has climtas for gamma -> beta calculation as RCP45 SMME-weighted median.
# climtas = test_ds["climtas"].weighted(smme_weights.sel(ensemble_member="rcp45/ACCESS1-0")).quantile(q=0.5)
climtas = test_ds["climtas"]  # Don't need weighting for this case.
# combine to single dataset then drop tracts without income because nan don't work with the clipping 
covars = xr.merge([income[inc_var], climtas]).dropna(dim='region')

# `build_mortality_betas()` doesn't work on dask arrays.
# TODO: Can we get this to work with dask array-backed Dataframes?
covars = covars.load()  
temps = temps.load()

betas = build_mortality_betas(
    csvv=csvv,
    temps=temps,
    loggdppc=covars[inc_var].rename('loggdppc'),
    climtas=covars["climtas"],
) 

# Change so all our "age*" variables are a single "beta" variable with "age_cohort" coord instead.
# TODO: Can we get it so this is the natural output of `build_mortality_betas()`?
betas["beta"] = xr.concat([betas["age1"], betas["age2"], betas["age3"]], pd.Index(["age1", "age2", "age3"], name="age_cohort")).rename({"temp": "tas_bin"})
beta = betas[["beta"]]

########################### Now the actual projection #########################

pop_raw = pd.read_csv(POP_PATH, dtype={'GEOID':str}).set_index("GEOID").to_xarray().rename({'total_tract_population':'combined', 'pop_lt5':'age1', 'pop_5-64':'age2', 'pop_65+':'age3'})
# calculate pop shares 
for var in beta["age_cohort"].values:
    pop_raw[f"{var}_share"] = pop_raw[var]/pop_raw["combined"] 
pop = xr.Dataset()
pop["pop"] = xr.concat([pop_raw["age1"], pop_raw["age2"], pop_raw["age3"]], pd.Index(["age1", "age2", "age3"], name="age_cohort"))
pop["combined"] = pop_raw["combined"]
pop["share"] = xr.concat([pop_raw["age1_share"], pop_raw["age2_share"], pop_raw["age3_share"]], pd.Index(["age1", "age2", "age3"], name="age_cohort"))
pop = pop.rename({"GEOID": "region"})


def _no_processing(ds: xr.Dataset) -> xr.Dataset:
    return ds


def _tas_unit_conversion(ds: xr.Dataset) -> xr.Dataset:
    """Pop and betas assume temperature in Kelvin so convert Kelvin to C"""
    # NOTE: This conversion is not correct (should be -273.15), but this is what was in the orignal code.
    # TODO: Fix betas and pop up stream and then fix this.
    return ds.assign_coords(tas_bin=(ds["tas_bin"] - 273))


def _mortality_impact_model(ds: xr.Dataset) -> xr.Dataset:
    # dot product of betas and t_bins * census tract age-spec populations
    _effect = (ds["histogram_tas"] * ds["beta"]).sum(dim="tas_bin") * ds["share"]

    # impacts are difference of future - historical effect
    impact = _effect.sel(year=2050) - _effect.sel(year=2020)
    
    return xr.Dataset({"impact": impact, "_effect": _effect})


mortality_impact_model = Projector(
    preprocess=_tas_unit_conversion,
    project=_mortality_impact_model,
    postprocess=_no_processing,
)

########### Running on single dataset

test_ds = cmip5["rcp45/ACCESS1-0"].to_dataset()

mortality_impacts = project(
    test_ds,
    model=mortality_impact_model,
    parameters=xr.merge([pop, beta]),
)
mortality_impacts = mortality_impacts.compute()

# damages = project(
#     impacts,
#     model=valuation_spec.model,
#     parameters=valuation_spec.parameters
# )

# # TODO: GCM ensemble weighting?

########### Running on ensemble tree

mortality_impacts = cmip5.map_over_subtree(
    project,
    model=mortality_impact_model,
    parameters=xr.merge([pop, beta]),
)
mortality_impacts = mortality_impacts.compute()



# # ds.assign_coords(tas_bin=(ds["tas_bin"] - 273))


# ### Original version test #####################################################
# covpath = '/gcs/rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/output_data/'
# income = xr.open_dataset(os.path.join(covpath, 'income_adjusted.nc4'))
# climtas = xr.open_dataset(os.path.join(covpath, 'climtas.nc4'))
# inc_var = 'loggdppc_residual_scaled'
# # combine to single dataset then drop tracts without income because nan don't work with the clipping 
# covars = xr.merge([income[inc_var], climtas]).dropna(dim='GEOID')
# # <xarray.Dataset> Size: 481kB
# # Dimensions:                   (GEOID: 8016)
# # Coordinates:
# #   * GEOID                     (GEOID) <U11 353kB '06001400100' ... '06115041100'
# # Data variables:
# #     loggdppc_residual_scaled  (GEOID) float64 64kB 11.66 11.45 ... 10.68 10.46
# #     climtas                   (GEOID) float64 64kB 15.18 15.68 ... 17.52 16.2

# ########### Where did this climtas.nc4 come from? #############################
# # From https://gitlab.com/rhodium/impactlab-rhg/carb-cvm/data-prep/-/blob/2bb3e6c8b587befafa6f62b186dc2ad83248c3db/notebooks/Climtas_prep.ipynb
# import pandas as pd

# CLIMTAS_CSV_URI = "gs://rhg-data/climate/aggregated/NASA/NEX-GDDP-BCSD-reformatted/California_2019_census_tracts_weighted_by_population/tas-bartlett30_2019_SMME-median_rcp45.csv"
# climtas = pd.read_csv(CLIMTAS_CSV_URI, header=14, dtype={"GEOID": str}).set_index("GEOID").to_xarray()
# climtas = climtas - 273
# climtas = climtas.rename({'climtas_bartlett30_K':'climtas'})
########### Where did this tas-bartlett30_2019_SMME-median_rcp45.csv come from? ###
# From https://gitlab.com/rhodium/impactlab-rhg/carb-cvm/data-prep/-/blob/2bb3e6c8b587befafa6f62b186dc2ad83248c3db/notebooks/make_climtas.ipynb





# # Hack workaround to https://github.com/pydata/xarray/issues/3476 Thanks Ian and Mike.
# v = "ensemble_member"
# if gdpcir_ds.coords[v].dtype == object:
#     gdpcir_ds.coords[v] = gdpcir_ds.coords[v].astype("str")
 

# To test with polars: "gs://impactlab-data-scratch/brews/20240731_test.parquet"

 # This seems to work well for single GCM Datasets.
 # test_ds.chunk({"region": 1000, "year": -1, "tas_bin": -1})



#  smme_weights["smme_weights"][smme_weights["ensemble_member"].str.contains("rcp45/")]