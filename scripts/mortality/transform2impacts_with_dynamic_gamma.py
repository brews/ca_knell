%pip install muuttaa==0.1.0 metacsv


import datetime
import os
import uuid
import csv
import re
from os import PathLike
from io import BufferedIOBase
from typing import Any


from dask_gateway import GatewayCluster
import datatree as dt
import fsspec
import metacsv
from muuttaa import SegmentWeights
import numpy as np
from muuttaa import apply_transformations, project, Projector
from muuttaa import TransformationStrategy
import pandas as pd
import xarray as xr
from xhistogram.xarray import histogram


JUPYTER_IMAGE = os.environ.get("JUPYTER_IMAGE")
UID = str(uuid.uuid4())
START_TIME = datetime.datetime.now(datetime.timezone.utc).isoformat()
print(f"{JUPYTER_IMAGE=}\n{START_TIME=}\n{UID=}")
CMIP5_URI = "gs://impactlab-data-scratch/brews/c4c98753-7428-4fec-9fd8-77c38033fabf/cmip5_concat.zarr"
CARB_SEGMENT_WEIGHTS_URL = "gs://rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/source_data/segment_weights/California_2019_census_tracts_weighted_by_population_0p25.csv"
CSVV_PATH = "gs://rhg-data-scratch/brews/Agespec_interaction_GMFD_POLY-4_TINV_CYA_NW_w1.csvv"  # Moved into scratch from https://gitlab.com/rhodium/impactlab-rhg/carb-cvm/beta-generation/-/blob/330bf3b949881749e6f3d13c88349be0d65bbfb8/csvvs/Agespec_interaction_GMFD_POLY-4_TINV_CYA_NW_w1.csvv
INCOME_PATH = "/gcs/rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/output_data/income_adjusted.nc4"
BETA_PATH = "/gcs/rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/betas/clipped_mortality_betas_loggdppc_residual_scaled.nc4"
POP_URI = "gs://rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/output_data/population_age_binned.csv"
PCI2019_URI = "gs://rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/output_data/PCI_2019.csv"


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


def read_csvv(filename):
    """Interpret a CSVV file into a dictionary of the included information.

    Specific implementation is described in the two CSVV version
    readers, `read_girdin` and `csvvfile_legacy.read`.
    """
    with fsspec.open(filename, "r") as fp:
        attrs, coords, variables = metacsv.read_header(fp, parse_vars=True)

        # Clean up variables
        for variable in variables:
            vardef = variables[variable[0]]
            assert isinstance(
                vardef, dict
            ), "Variable definition '%s' malformed." % str(vardef)
            if "unit" in vardef:
                fullunit = vardef["unit"]
                if "]" in fullunit:
                    vardef["unit"] = fullunit[: fullunit.index("]")]
            else:
                print("WARNING: Missing unit for variable %s." % variable)
                vardef["unit"] = None

        data = {"attrs": attrs, "variables": variables, "coords": coords}

        # `attrs` should have "csvv-version" otherwise should be read in with
        # `csvvfile_legacy.read` - but I'm not sure what this actually is.
        csvv_version = attrs["csvv-version"]
        if csvv_version == "girdin-2017-01-10":
            return _read_girdin(data, fp)
        else:
            raise ValueError("Unknown version " + csvv_version)


def _read_girdin(data, fp):
    """Interpret a Girdin version CSVV file into a dictionary of the
    included inforation.

    A Girdin CSVV has a lists of predictor and covariate names, which
    are matched up one-for-one.  This offered more flexibility and
    clarity than the previous version of CSVV files.

    Parameters
    ----------
    data : dict
        Meta-data from the MetaCSV description.
    fp : file pointer
        File pointer to the start of the file content.

    Returns
    -------
    dict
        Dictionary with MetaCSV information and the predictor and
    covariate information.
    """
    reader = csv.reader(fp)
    variable_reading = None

    for row in reader:
        if len(row) == 0 or (len(row) == 1 and len(row[0].strip()) == 0):
            continue
        row[0] = row[0].strip()

        if row[0] in [
            "observations",
            "prednames",
            "covarnames",
            "gamma",
            "gammavcv",
            "residvcv",
        ]:
            data[row[0]] = []
            variable_reading = row[0]
        else:
            if variable_reading is None:
                print("No variable queued.")
                print(row)
            assert variable_reading is not None
            if len(row) == 1:
                row = row[0].split(",")
            if len(row) == 1:
                row = row[0].split("\t")
            if len(row) == 1:
                row = re.split(r"\s", row[0])
            data[variable_reading].append([x.strip() for x in row])

    data["observations"] = float(data["observations"][0][0])
    data["prednames"] = data["prednames"][0]
    data["covarnames"] = data["covarnames"][0]
    data["gamma"] = np.array(list(map(float, data["gamma"][0])))
    data["gammavcv"] = np.array([list(map(float, row)) for row in data["gammavcv"]])
    data["residvcv"] = np.array([list(map(float, row)) for row in data["residvcv"]])
    return data


def _no_processing(ds: xr.Dataset) -> xr.Dataset:
    return ds


def _make_annual_tas(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute annual average for 'tas'.
    """
    return ds[["tas"]].groupby("time.year").mean("time")


def _make_30hbartlett_climtas(ds: xr.Dataset) -> xr.Dataset:
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
    preprocess=_make_annual_tas,
    postprocess=_make_30hbartlett_climtas,
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

# Betas from Gammas #####################################################


def uclip(da, dim, lmmt=10, ummt=30):
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


def _add_degree_coord(da, max_degrees):
    if max_degrees < 2:
        raise ValueError("'max_degree' arg must be >= 2")

    degree_idx = list(range(1, max_degrees + 1))
    out = xr.concat(
        [da] + [da**i for i in degree_idx[1:]],
        pd.Index(degree_idx, name="degree")
    )
    return out


def _beta_from_gamma(ds: xr.Dataset) -> xr.Dataset:
    # Unpack the gamma coefs for particular covariates. Makes it easier to read.
    gamma_1 = ds["gamma"].sel(covarnames="1")
    gamma_climtas = ds["gamma"].sel(covarnames="climtas")
    gamma_loggdppc = ds["gamma"].sel(covarnames="loggdppc")

    # With annual histograms as input, use histogram bin labels ("tas_bin") as "tas".
    # Need to raise tas to powers equal to degrees in polynomial. These are added as
    # new "degree" coord on tas.
    tas = _add_degree_coord(ds["tas_bin"], max_degrees=gamma_1["degree"].size)

    beta = (
            gamma_1 * tas
            + gamma_climtas * ds["climtas"] * tas
            + gamma_loggdppc * ds["loggdppc"] * tas
    ).sum("degree") # Reduce all degrees of polynomial summing across degrees.

    beta = uclip(beta, dim="tas_bin")

    # TODO: Pre-generate multivariate normal draws as extra dim here, pass in a model parameters.
    return ds.assign({"beta": beta})


def _mortality_impact_model(ds: xr.Dataset) -> xr.Dataset:
    # dot product of betas and t_bins * census tract age-spec populations
    _effect = (ds["histogram_tas"] * ds["beta"]).sum(dim="tas_bin") * ds["share"]

    # impacts are difference of future - historical effect
    impact = _effect.sel(year=2050) - _effect.sel(year=2020)

    return xr.Dataset({"impact": impact, "_effect": _effect})


# If you have gamma and need to compute beta.
mortality_impact_model_gamma = Projector(
    preprocess=_beta_from_gamma,
    project=_mortality_impact_model,
    postprocess=_no_processing,
)

##########################################################################

segment_weights = open_carb_segmentweights(CARB_SEGMENT_WEIGHTS_URL)

# Additional params for impact projection #####################################

betas_raw = xr.open_dataset(BETA_PATH)
# Rename is temporary measure to ensure var names match - need to change in data-prep step
pop_raw = (
    pd.read_csv(POP_URI, dtype={"GEOID": str})
    .set_index("GEOID")
    .to_xarray()
    .rename(
        {
            "total_tract_population": "combined",
            "pop_lt5": "age1",
            "pop_5-64": "age2",
            "pop_65+": "age3",
        }
    )
)

# calculate pop shares
for var in betas_raw.keys():
    pop_raw[f"{var}_share"] = pop_raw[var] / pop_raw["combined"]

beta = xr.concat(
    [betas_raw["age1"], betas_raw["age2"], betas_raw["age3"]],
    pd.Index(["age1", "age2", "age3"], name="age_cohort"),
).rename({"temp": "tas_bin", "GEOID": "region"})
beta.name = "beta"
beta = beta.assign_coords(region=(beta["region"].astype("O")))

pop = xr.Dataset()
pop["pop"] = xr.concat(
    [pop_raw["age1"], pop_raw["age2"], pop_raw["age3"]],
    pd.Index(["age1", "age2", "age3"], name="age_cohort"),
)
pop["combined"] = pop_raw["combined"]
pop["share"] = xr.concat(
    [pop_raw["age1_share"], pop_raw["age2_share"], pop_raw["age3_share"]],
    pd.Index(["age1", "age2", "age3"], name="age_cohort"),
)
pop = pop.rename({"GEOID": "region"})

# Additional params for impact valuation ######################################

valuation_params = xr.Dataset()

# load 2019 income for valuation metric
pci2019 = (
    pd.read_csv(PCI2019_URI, dtype={"GEOID": str})
    .set_index("GEOID")
    .rename(columns={"2019": "pci_2019"})["pci_2019"]
    .to_xarray()
)
pci2019 = pci2019.rename({"GEOID": "region"})
# set VSL at EPA value of $7.4 M in 2006$ converted to 2019$
# this was calculated using the convert_dollars function from the data-prep repo - let's call that and do the math here in the future
vsl = 9146910
vsl2 = 9926525

# convert to impacts per person
scale = 1 / 100000
# NOTE: Beware we're multiplying big numbers by small numbers here.

valuation_params["vsl"] = 9926525
valuation_params["scale"] = scale
valuation_params["pci"] = pci2019
valuation_params["pop"] = pop["pop"]

# Run #########################################################################


cmip5 = dt.open_datatree(CMIP5_URI, engine="zarr", chunks={}).chunk(
    {"time": 365 * 20, "lat": 90, "lon": 90}
)
test_ds = cmip5["rcp45/ACCESS1-0"].ds

# Need dask cluster for these climate transformations.
with GatewayCluster(
        worker_image=JUPYTER_IMAGE, scheduler_image=JUPYTER_IMAGE, profile="micro"
) as cluster:
    client = cluster.get_client()
    print(client.dashboard_link)
    cluster.scale(50)

    transformed = apply_transformations(
        test_ds,
        regionalize=segment_weights,
        strategies=[
            make_climtas,
            make_tas_20yrmean_annual_histogram,
        ],
    )
    transformed = transformed.compute()


    # Convert units and subset data to just 2020 and 2050
    transformed = (
        transformed
        # .sel(year=[2020, 2050])
        .assign_coords(tas_bin=(transformed["tas_bin"] - 273))
        .assign(climtas=(transformed["climtas"] - 273))
    )

    csvv = read_csvv(CSVV_PATH)
    income = xr.open_dataset(INCOME_PATH).rename({"GEOID": "region"})
    # Encode "region" as Object rather than <U11 so consistent with transformed climate data region.
    income = income.assign_coords(region=(income["region"].astype("O")))

    gamma = csvv["gamma"].reshape([3, 4, 3])  # Makes it [age, degree, covarnames]. BINGO!
    # This gets us the median.
    # TODO: What if we want multiple RNG draws from multivariate normal?
    g = xr.Dataset(
        {
            "gamma": (["age_cohort", "degree", "covarnames"], gamma),
        },
        coords={
            "age_cohort": (["age_cohort"], ["age1", "age2", "age3"]),
            "degree": (["degree"], np.arange(4)+1),
            "covarnames": (["covarnames"], ["1", "climtas", "loggdppc"]),
        }
    )

    impact_gamma_params = xr.Dataset()
    impact_gamma_params["loggdppc"] = income["loggdppc_residual_scaled"]
    impact_gamma_params["gamma"] = g["gamma"]
    impact_gamma_params["share"] = pop["share"]

    def merge_damage_inputs(predictors, params):
        # combine to single dataset then drop tracts without income because nan don't work with the clipping
        return xr.merge([predictors, params]).dropna(dim="region")

    # ds = merge_damage_inputs(
    #     transformed.sel(year=slice(1990, 2098)), # Need to drop years with NA, as climtas as nans. Otherwise the merged output is empty.
    #     impact_gamma_params
    # )

    # out = _beta_from_gamma(ds)
    # print(out)
    # out = out.compute()

    mortality_impacts = project(
        transformed.sel(year=slice(1990, 2098)),
        model=mortality_impact_model_gamma,
        parameters=impact_gamma_params.dropna(dim="region"),
    )
    print(mortality_impacts)
    mortality_impacts = mortality_impacts.compute()



# # Compare with old results
# beta.mean("region").sel(age_cohort="age1").plot()
# out["beta"].mean("region").sel(year=2020, age_cohort="age1").plot()