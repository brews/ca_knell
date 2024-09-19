import datetime
import os
import uuid
from os import PathLike
from io import BufferedIOBase
from typing import Any

from muuttaa import (
    TransformationStrategy,
    Projector,
    SegmentWeights,
    apply_transformations,
    project,
)
import numpy as np
import pandas as pd
import xarray as xr
from xhistogram.xarray import histogram
from dask_gateway import GatewayCluster
import datatree as dt


JUPYTER_IMAGE = os.environ.get("JUPYTER_IMAGE")
UID = str(uuid.uuid4())
START_TIME = datetime.datetime.now(datetime.timezone.utc).isoformat()
print(f"{JUPYTER_IMAGE=}\n{START_TIME=}\n{UID=}")
CMIP5_URI = (
    "gs://rhg-data-scratch/brews/16c38870-f245-472c-b09c-3899a4501716/cmip5_concat.zarr"
)
POP_URI = "gs://rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/output_data/employment_risk_binned.csv"
POP2019_URI = "gs://rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/output_data/population_age_binned.csv"
PCI2019_URI = "gs://rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/output_data/PCI_2019.csv"
WAGES_URI = "gs://rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/output_data/aggregate_labor_earnings.csv"
CARB_SEGMENT_WEIGHTS_URI = "gs://rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/source_data/segment_weights/California_2019_census_tracts_weighted_by_population_0p25.csv"
GAMMA_URI = (
    "gs://rhg-data-scratch/brews/58aeb857-ff65-4dea-adb5-3182a8a595fd/gamma_labor.zarr"
)


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


def _no_processing(ds: xr.Dataset) -> xr.Dataset:
    return ds


def _labor_impact_model(ds: xr.Dataset) -> xr.Dataset:
    # dot product of betas and t_bins * census tract age-spec populations
    _effect = (ds["histogram_tasmax"] * ds["beta"]).sum(dim="tasmax_bin")

    # convert from annual minute change to daily to match CIL output
    _effect = _effect / 365

    # impacts are difference of future - historical effect
    impact = _effect.sel(year=2050) - _effect.sel(year=2020)

    # TODO: Consider doing "total_employed" not sure if it should be here. It depends on coordinate levels. From In[25] of https://gitlab.com/rhodium/impactlab-rhg/carb-cvm/impact-projections/-/blob/main/notebooks/calculate_labor_impacts.ipynb?ref_type=heads
    # scn['total_employed'] = scn['low_risk'] * pop['low_risk_share'] + scn['high_risk'] * pop['high_risk_share']

    return xr.Dataset({"impact": impact, "_effect": _effect})


# If you already have beta.
labor_impact_model = Projector(
    preprocess=_no_processing,
    project=_labor_impact_model,
    postprocess=_no_processing,
)


def rcspline(
    temps: xr.DataArray,
    k1: int | float = 27,
    k2: int | float = 37,
    k3: int | float = 39,
) -> xr.DataArray:
    """Takes in an array of temperatures and creates a corresponding RC Spline
    DataArray with 3 knots to be used in the labor beta generation stage

    Parameters
    ----------
    temps : DataArray
        Binned temps
    k1, k2, k3 : int
        knot values needed to construct masks to execute RC spline formula.
        Defaults are 27, 37, 39 to match the main model from CIL labor paper.

    Returns
    -------
        rcspline : DataArray with the same coordinates as the input temp DataArray
    """
    # create masks
    mk1 = (temps > k1) * 1
    mk2 = (temps > k2) * 1
    mk3 = (temps > k3) * 1

    out = (
        (temps - k1) ** 3 * mk1
        - (temps - k2) ** 3 * (k3 - k1) / (k3 - k2) * mk2
        + (temps - k3) ** 3 * (k2 - k1) / (k3 - k2) * mk3
    )

    return out


def _beta_from_gamma(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculates labor impact model's beta coefficients from gamma coefficients.

    Returns a copy of `ds` with new "beta" variable. Beta here is the labor response (in minutes per x)
    at each binned degree.
    """
    # Predictors.
    tasmax = ds["tasmax_bin"]
    tasmax_rcspline1 = rcspline(tasmax)

    # Coefficients.
    gamma_tasmax = ds["gamma"].sel(predname="tasmax")
    gamma_tasmax_rcspline1 = ds["gamma"].sel(predname="tasmax_rcspline1")

    # Note: Apparently CARB labor doesn't use gammas for covariates so we're not doing it here even though they're in the CSVV file.

    beta = tasmax * gamma_tasmax + tasmax_rcspline1 * gamma_tasmax_rcspline1

    # TODO: Consider adding logic to center beta coefficient relative to a given temperature or a risk shares minimum labor loss (MLL).
    # Apparently whether the response functions are centered or not does not matter for projecting impacts, but are useful for graphic display.
    # See https://gitlab.com/rhodium/impactlab-rhg/carb-cvm/beta-generation/-/blob/330bf3b949881749e6f3d13c88349be0d65bbfb8/src/genbeta/core.py#L250

    # Returns new dataset with beta added as new variable. Not modifying
    # original ds.
    return ds.assign({"beta": beta})


# If you have gamma and need to compute beta.
labor_impact_model_gamma = Projector(
    preprocess=_beta_from_gamma,  # Not sure this should be preprocess but I'm lazy.
    project=_labor_impact_model,
    postprocess=_no_processing,
)


def _labor_valuation_model(ds: xr.Dataset) -> xr.Dataset:
    # Total damages are impacts in mins/worker/day * 365 days * worker population *  year wage income / (250 days * 60 mins * 6 hours * elasticity = 0.5 )
    # negative sign is applied since minutes lost is a positive damage
    damages_total = (
        -ds["impact"]
        * 365
        * ds["pop"]
        * ds["wages"]
        / (ds["total_employed"] * 250 * 60 * 6 * 0.5)
    )

    # Damages per capita are divided by total tract population
    damages_pc = damages_total / ds["total_tract_population"]

    # Damages as share of average tract income = damages per capita / income per capita
    damages_pincome = damages_pc / ds["pci"]

    out = xr.Dataset(
        {
            "damages_total": damages_total,
            "damages_pc": damages_pc,
            "damages_pincome": damages_pincome,
        }
    )
    return out


labor_valuation_model = Projector(
    preprocess=_no_processing,
    project=_labor_valuation_model,
    postprocess=_no_processing,
)

#### Read in and parse population data, socioeconomics, parameters ###################################

# This pop was used for labor impact projection in https://gitlab.com/rhodium/impactlab-rhg/carb-cvm/impact-projections/-/blob/cb6e4abf28adfbb737354b7e585491ec738c29e4/notebooks/calculate_labor_impacts.ipynb
pop_raw = pd.read_csv(POP_URI, dtype={"GEOID": str}).set_index("GEOID").to_xarray()
pop_raw = pop_raw.rename({"GEOID": "region"})
# calculate pop shares
for var in ["low_risk", "high_risk"]:
    pop_raw[f"{var}_share"] = pop_raw[var] / pop_raw["total_employed"]

# Used for valuation.
# load 2019 labor earnings income (salary + wages + self employed income)
wages = (
    pd.read_csv(WAGES_URI, dtype={"GEOID": str})
    .set_index("GEOID")["total_labor_earnings"]
    .to_xarray()
)
wages = wages.rename({"GEOID": "region"})

# load 2019 income for valuation metric
pci2019 = (
    pd.read_csv(PCI2019_URI, dtype={"GEOID": str})
    .set_index("GEOID")
    .rename(columns={"2019": "pci_2019"})["pci_2019"]
    .to_xarray()
)
pci2019 = pci2019.rename({"GEOID": "region"})

# load 2019 population for valuation metric
pop2019 = (
    pd.read_csv(POP2019_URI, dtype={"GEOID": str})
    .set_index("GEOID")["total_tract_population"]
    .to_xarray()
)
pop2019 = pop2019.rename({"GEOID": "region"})

# Clean up coordinates and variables.
pop = xr.Dataset()
pop["pop"] = xr.concat(
    [pop_raw["low_risk"], pop_raw["high_risk"]],
    pd.Index(["low", "high"], name="risk_sector"),
)
pop["share"] = xr.concat(
    [pop_raw["low_risk_share"], pop_raw["high_risk_share"]],
    pd.Index(["low", "high"], name="risk_sector"),
)

valuation_params = xr.Dataset(
    {
        "pop": pop["pop"],
        "wages": wages,
        "total_employed": pop_raw["total_employed"],
        "total_tract_population": pop2019,
        "pci": pci2019,
    }
).dropna(dim="region")

# Read cleaned, pre-sampled gammas.
gammas = xr.open_zarr(GAMMA_URI)

impact_params = xr.Dataset(
    {
        # "gamma": gammas["gamma_mean"]
        "gamma": gammas["gamma_sampled"]
    }
)

##########################################################################
# TODO: Stopped here. Need to run everything, but also need to re-generate clean CMIP5 data for input.

segment_weights = open_carb_segmentweights(CARB_SEGMENT_WEIGHTS_URI)
cmip5 = dt.open_datatree(CMIP5_URI, engine="zarr", chunks={}).chunk(
    {"time": 365 * 20, "lat": 90, "lon": 90}
)
test_ds = cmip5["rcp45/ACCESS1-0"].ds


def _merge_impact_inputs(x):
    """
    Merge and rechunk mortality impact projection inputs and parameters before projecting.

    This is to be passed to ``muuttaa.project()`` "merge_predictors_parameters" argument.
    This custom merging is needed because parameters and transformed
    data have different regions. The gamma parameter can also have multiple
    samples. Rechunking after they've merged ensures the chunk sizes
    when we compute beta do not become overwhelming. Different regions between
    input data can also mean that chunks are offset from one another after merging
    and this takes time to correct during a projection.

    An alternative to using this custom merging function is to chunk the gamma and regions
    data before projecting and ensure that all data use the same regions.
    """
    return xr.merge(x).chunk({"sample": 1, "region": 1000})


with GatewayCluster(
    worker_image=JUPYTER_IMAGE, scheduler_image=JUPYTER_IMAGE, profile="micro"
) as cluster:
    client = cluster.get_client()
    print(client.dashboard_link)
    cluster.scale(50)

    transformed = apply_transformations(
        test_ds,
        regionalize=segment_weights,
        strategies=[make_tasmax_20yrmean_annual_histogram],
    )

    transformed = (
        transformed.assign_coords(tasmax_bin=(transformed["tasmax_bin"] - 273)).sel(
            year=slice(1990, 2098)
        )  # Years outside this have NA due to climtas rolling operations.
    )
    transformed = transformed.compute()

    labor_impacts = project(
        transformed,
        model=labor_impact_model_gamma,
        parameters=impact_params,  # TODO: Muuttaa update -> What if have not parameters?
        merge_predictors_parameters=_merge_impact_inputs,  # Important if we're using sampled gammas.
    )
    labor_impacts = labor_impacts.compute()

    labor_damages = project(
        labor_impacts,
        model=labor_valuation_model,
        parameters=valuation_params,
    )
    labor_damages = labor_damages.compute()
print(labor_damages)


# Plot "total_employed". Sanity check?
(labor_impacts["impact"] * pop["share"]).sum("risk_sector").plot.hist()
