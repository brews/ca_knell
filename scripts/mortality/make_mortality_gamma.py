"""
Generates a Zarr Store with structured mean and random samples of mortality's gamma parameter. These are created from an input CSVV file.

This structured gamma data should be created from the CSVV before the projection system is run so the projection system can use these gammas as inputs.

These gammas are pre-created so the projection system itself can be deterministic. This also helps to ensure we can replicate outputs.
"""

import datetime
import os
import uuid

import numpy as np
import xarray as xr

from ca_knell.io import read_csvv

JUPYTER_IMAGE = os.environ.get("JUPYTER_IMAGE")
UID = str(uuid.uuid4())
START_TIME = datetime.datetime.now(datetime.timezone.utc).isoformat()
print(f"{JUPYTER_IMAGE=}\n{START_TIME=}\n{UID=}")

CSVV_URI = "gs://rhg-data-scratch/brews/Agespec_interaction_GMFD_POLY-4_TINV_CYA_NW_w1.csvv"  # Moved into scratch from https://gitlab.com/rhodium/impactlab-rhg/carb-cvm/beta-generation/-/blob/330bf3b949881749e6f3d13c88349be0d65bbfb8/csvvs/Agespec_interaction_GMFD_POLY-4_TINV_CYA_NW_w1.csvv
GAMMA_OUT_URI = f"gs://rhg-data-scratch/brews/{UID}/gamma_mortality.zarr"
SEED = 42
N_SAMPLES = 15

# NOTE: If you change these, you will likely need to change the structure of the output Dataset, also.
N_AGE_COHORT = 3
N_POLYNOMIAL_DEGREES = 4
N_COVARNAMES = 3
GAMMA_SHAPE = [N_AGE_COHORT, N_POLYNOMIAL_DEGREES, N_COVARNAMES]


csvv = read_csvv(CSVV_URI)
gamma_mean = csvv["gamma"].reshape(GAMMA_SHAPE)
# This gets us the median.

rng = np.random.default_rng(SEED)
gamma_samples_raw = rng.multivariate_normal(csvv["gamma"], csvv["gammavcv"], N_SAMPLES)

# Add additional dim for samples drawn, and reshape flat array to match structure.
samples_shape = [N_SAMPLES] + GAMMA_SHAPE
gamma_samples = gamma_samples_raw.reshape(samples_shape)

# NOTE: This has some magic coordinates that need to change if the CSVV structure changes.
g = xr.Dataset(
    {
        "gamma_mean": (["age_cohort", "degree", "covarname"], gamma_mean),
        "gamma_sampled": (
            ["sample", "age_cohort", "degree", "covarname"],
            gamma_samples,
        ),
    },
    coords={
        "age_cohort": (["age_cohort"], ["age1", "age2", "age3"]),
        "covarname": (["covarname"], ["1", "climtas", "loggdppc"]),
        "degree": (["degree"], np.arange(4) + 1),
        "sample": np.arange(N_SAMPLES),
    },
)

# TODO: Add metadata with "created_at", "uri", source CSVV path ("history"?), etc...

g.to_zarr(GAMMA_OUT_URI, mode="w")
print(GAMMA_OUT_URI)
# gs://rhg-data-scratch/brews/5e1da757-2849-4abc-841c-2e45e343c25b/gamma_mortality.zarr
