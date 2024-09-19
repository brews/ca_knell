"""
Generates a Zarr Store with structured mean and random samples of labor's gamma parameter. These are created from an input CSVV file.

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

CSVV_URI = "gs://rhg-data-scratch/brews/labor_main_uninteracted.csvv"  # Moved into scratch from https://gitlab.com/rhodium/impactlab-rhg/carb-cvm/beta-generation/-/blob/330bf3b949881749e6f3d13c88349be0d65bbfb8/csvvs/labor_main_uninteracted.csvv
GAMMA_OUT_URI = f"gs://rhg-data-scratch/brews/{UID}/gamma_labor.zarr"
SEED = 42
N_SAMPLES = 15

# NOTE: If you change these, you will likely need to change the structure of the output Dataset, also.
N_RISK_SECTORS = 2
N_PREDNAMES = 2
GAMMA_SHAPE = [N_RISK_SECTORS, N_PREDNAMES]

csvv = read_csvv(CSVV_URI)
# We're grabbing the first 4 gammas, first 2 are low risk, last 2 are high risk according to CSVV. Not doing remaining (for unemployment shares).
gamma_mean = csvv["gamma"][:4].reshape(GAMMA_SHAPE)

rng = np.random.default_rng(SEED)
# Again, remember we're only using first 4 elements of gamma and gamma variance-covariance matrix.
gamma_samples_raw = rng.multivariate_normal(
    csvv["gamma"][:4], csvv["gammavcv"][:4, :4], N_SAMPLES
)

# Add additional dim for samples drawn, and reshape flat array to match structure.
samples_shape = [N_SAMPLES] + GAMMA_SHAPE
gamma_samples = gamma_samples_raw.reshape(samples_shape)

# NOTE: This has some magic coordinates that need to change if the CSVV structure changes.
g = xr.Dataset(
    {
        "gamma_mean": (["risk_sector", "predname"], gamma_mean),
        "gamma_sampled": (["sample", "risk_sector", "predname"], gamma_samples),
    },
    coords={
        "risk_sector": (["risk_sector"], ["low", "high"]),
        "predname": (["predname"], ["tasmax", "tasmax_rcspline1"]),
        "sample": np.arange(N_SAMPLES),
    },
)

# TODO: Add metadata with "created_at", "uri", source CSVV path ("history"?), etc...

g.to_zarr(GAMMA_OUT_URI, mode="w")
print(GAMMA_OUT_URI)
# gs://rhg-data-scratch/brews/58aeb857-ff65-4dea-adb5-3182a8a595fd/gamma_labor.zarr
