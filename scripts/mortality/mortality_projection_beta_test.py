# Mortality projection.
# This version uses betas as inputs to the projection model.

# Adapted from https://gitlab.com/rhodium/impactlab-rhg/carb-cvm/impact-projections/-/blob/cb6e4abf28adfbb737354b7e585491ec738c29e4/notebooks/calculate_mortality_impacts.ipynb

# %pip install muuttaa==0.1.0


import datatree
from muuttaa import Projector, project
import pandas as pd
import xarray as xr


TRANSFORMED_CMIP_URI = "gs://rhg-data-scratch/brews/f3d91aef-5b54-415b-b375-fa15dfaf0dfe/cmip5_transformed_test.zarr"
BETA_PATH = "/gcs/rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/betas/clipped_mortality_betas_loggdppc_residual_scaled.nc4"
POP_URI = "gs://rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/output_data/population_age_binned.csv"
PCI2019_URI = "gs://rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/output_data/PCI_2019.csv"


# Only need "tas_histogram"
# Historical is 2010 - 2030, future is 2040 - 2060
cmip5 = datatree.open_datatree(TRANSFORMED_CMIP_URI, engine="zarr", chunks={}).sel(
    year=[2020, 2050]
)

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


########################## Now playing with my version ############################

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

######################## Do model manually ###############

test_ds = cmip5["rcp45/ACCESS1-0"].to_dataset()

# Pop and betas assume temperature in Kelvin so convert Kelvin to C.
# NOTE: Yes, this conversion isn't correct (should be -273.15), but this is what was in the orignal code.
# TODO: Fix betas and pop up stream and then fix this.
test_ds = test_ds.assign_coords(tas_bin=(test_ds["tas_bin"] - 273))

# dot product of betas and t_bins * census tract age-spec populations
_tmp = (test_ds["histogram_tas"] * beta).sum(dim="tas_bin") * pop["share"]
# impacts are difference of future - historical effect
impacts = _tmp.sel(year=2050) - _tmp.sel(year=2020)
# TODO: if impacts are future - hist, what do we call `_tmp` above?

# TODO: Check this math! ^


########################### Inputs and parameters for valuing impacts (creating damages from impacts) ###############
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


########################### Do this in muuttaa style.
def _no_processing(ds: xr.Dataset) -> xr.Dataset:
    return ds


def _tas_unit_conversion(ds: xr.Dataset) -> xr.Dataset:
    """Pop and betas assume temperature in Kelvin so convert Kelvin to C"""
    # NOTE: This conversion is not correct (should be -273.15), but this is what was in the orignal code.
    # TODO: Fix betas and pop up stream and then fix this.
    # TODO: Don't think we need this preprocessing function anymore.
    return ds.assign_coords(tas_bin=(ds["tas_bin"] - 273))


def _mortality_impact_model(ds: xr.Dataset) -> xr.Dataset:
    # dot product of betas and t_bins * census tract age-spec populations
    _effect = (ds["histogram_tas"] * ds["beta"]).sum(dim="tas_bin") * ds["share"]

    # impacts are difference of future - historical effect
    impact = _effect.sel(year=2050) - _effect.sel(year=2020)

    return xr.Dataset({"impact": impact, "_effect": _effect})


mortality_impact_model = Projector(
    preprocess=_no_processing,
    project=_mortality_impact_model,
    postprocess=_no_processing,
)


def _mortality_valuation_model(ds: xr.Dataset) -> xr.Dataset:
    # Total damages are age-spec physical impacts (deaths/100k) * age-spec population * scale * vsl
    damages_total = ds["impact"] * ds["pop"] * ds["scale"] * ds["vsl"]

    # Damages per capita = total damages / population
    damages_pc = ds["impact"] * ds["scale"] * ds["vsl"]

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


mortality_valuation_model = Projector(
    preprocess=_no_processing,
    project=_mortality_valuation_model,
    postprocess=_no_processing,
)

########### Running on single dataset

test_ds = cmip5["rcp45/ACCESS1-0"].to_dataset()
test_ds = test_ds.assign_coords(tas_bin=(test_ds["tas_bin"] - 273))

mortality_impacts = project(
    test_ds, model=mortality_impact_model, parameters=xr.merge([pop, beta])
)
mortality_impacts

damages = project(
    mortality_impacts,
    model=mortality_valuation_model,
    parameters=valuation_params,
)
damages = damages.compute()

# # TODO: GCM ensemble weighting?

########### Running on ensemble tree

mortality_impacts = cmip5.map_over_subtree(
    project,
    model=mortality_impact_model,
    parameters=xr.merge([pop, beta]),
)
mortality_impacts = mortality_impacts.compute()
