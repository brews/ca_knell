# %pip install muuttaa==0.1.0 metacsv

import datetime
import os
import uuid

from dask_gateway import GatewayCluster
import datatree as dt
from muuttaa import apply_transformations, project
import pandas as pd
import xarray as xr

from ca_knell.io import open_carb_segmentweights
from ca_knell.mortality.transformation import (
    make_climtas,
    make_tas_20yrmean_annual_histogram,
)
from ca_knell.mortality.projection import (
    mortality_impact_model,
    mortality_valuation_model,
)


JUPYTER_IMAGE = os.environ.get("JUPYTER_IMAGE")
UID = str(uuid.uuid4())
START_TIME = datetime.datetime.now(datetime.timezone.utc).isoformat()
print(f"{JUPYTER_IMAGE=}\n{START_TIME=}\n{UID=}")
CMIP5_URI = "gs://impactlab-data-scratch/brews/c4c98753-7428-4fec-9fd8-77c38033fabf/cmip5_concat.zarr"
CARB_SEGMENT_WEIGHTS_URL = "gs://rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/source_data/segment_weights/California_2019_census_tracts_weighted_by_population_0p25.csv"
BETA_PATH = "/gcs/rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/betas/clipped_mortality_betas_loggdppc_residual_scaled.nc4"
POP_URI = "gs://rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/output_data/population_age_binned.csv"
PCI2019_URI = "gs://rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/output_data/PCI_2019.csv"


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

mortality_impacts = project(
    transformed, model=mortality_impact_model, parameters=xr.merge([pop, beta])
)

mortality_damages = project(
    mortality_impacts,
    model=mortality_valuation_model,
    parameters=valuation_params,
)
mortality_damages = mortality_damages.compute()


# # With pipes  #################################################################
# cmip5 = dt.open_datatree(CMIP5_URI, engine="zarr", chunks={}).chunk(
#     {"time": 365 * 20, "lat": 90, "lon": 90}
# )
# test_ds = cmip5["rcp45/ACCESS1-0"].ds

# # Need dask cluster for these climate transformations.
# with GatewayCluster(worker_image=JUPYTER_IMAGE, scheduler_image=JUPYTER_IMAGE, profile="micro") as cluster:
#     cluster.scale(50)

#     transformed = apply_transformations(
#         test_ds,
#         regionalize=segment_weights,
#         strategies=[
#             make_climtas,
#             make_tas_20yrmean_annual_histogram,
#         ],
#     )
#     transformed = transformed.compute()

# transformed = transformed.assign_coords(tas_bin=(transformed["tas_bin"] - 273))

# mortality_damages = (
#     transformed
#         .pipe(project, model=mortality_impact_model, parameters=xr.merge([pop, beta]))
#         .pipe(project, model=mortality_valuation_model, parameters=valuation_params)
# )


# # Entire cmip5 ensemble #######################################################

# cmip5 = dt.open_datatree(CMIP5_URI, engine="zarr", chunks={}).chunk(
#     {"time": 365 * 20, "lat": 90, "lon": 90}
# )

# def _transform2damages(ds, w, impact_p, valuation_p):
#     transformed = apply_transformations(
#         ds,
#         regionalize=w,
#         strategies=[
#             make_climtas,
#             make_tas_20yrmean_annual_histogram,
#         ],
#     )

#     transformed = transformed.assign_coords(tas_bin=(transformed["tas_bin"] - 273))

#     mortality_damages = (
#         transformed
#             .pipe(project, model=mortality_impact_model, parameters=impact_p)
#             .pipe(project, model=mortality_valuation_model, parameters=valuation_p)
#     )
#     return mortality_damages


# # Need dask cluster for these climate transformations.
# with GatewayCluster(worker_image=JUPYTER_IMAGE, scheduler_image=JUPYTER_IMAGE, profile="micro") as cluster:
#     print(cluster.get_client().dashboard_link)
#     cluster.scale(50)

#     mortality_damages = cmip5.map_over_subtree(
#         _transform2damages,
#         w=segment_weights,
#         impact_p=xr.merge([pop, beta]),
#         valuation_p=valuation_params,
#     )

#     mortality_damages = mortality_damages.compute()

# print(mortality_damages)
