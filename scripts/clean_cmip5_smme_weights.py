# Clean CMIP5 SMME weights

from typing import Literal

import pandas as pd
import xarray as xr


RAW_WEIGHTS_PATH = "https://raw.githubusercontent.com/ClimateImpactLab/impactlab-tools/master/src/impactlab_tools/assets/weights_gcp.csv"
OUT_WEIGHTS_URI = "gs://impactlab-data-scratch/brews/cmip5_smme_weights.zarr"

CMIP5_PATTERNMODELS_RCP45 = [
    f"pattern{i}" for i in [1, 2, 3, 5, 6, 27, 28, 29, 30, 31, 32]
]
CMIP5_PATTERNMODELS_RCP85 = [
    f"pattern{i}" for i in [1, 2, 3, 4, 5, 6, 28, 29, 30, 31, 32, 33]
]
CMIP5_MODELS = [
    "ACCESS1-0",
    "CNRM-CM5",
    "GFDL-ESM2G",
    "MIROC-ESM",
    "MPI-ESM-MR",
    "inmcm4",
    "BNU-ESM",
    "CSIRO-Mk3-6-0",
    "GFDL-ESM2M",
    "MIROC-ESM-CHEM",
    "MRI-CGCM3",
    "CCSM4",
    "CanESM2",
    "IPSL-CM5A-LR",
    "MIROC5",
    "NorESM1-M",
    "CESM1-BGC",
    "GFDL-CM3",
    "IPSL-CM5A-MR",
    "MPI-ESM-LR",
    "bcc-csm1-1",
]
CMIP5_RCP45_SMME_MODELS = CMIP5_MODELS + CMIP5_PATTERNMODELS_RCP45
CMIP5_RCP85_SMME_MODELS = CMIP5_MODELS + CMIP5_PATTERNMODELS_RCP85


def get_cmip5_surrogate_name(
    *,
    scenario: Literal["rcp45", "rcp85"],
    pattern: str,
    errors: Literal["raise", "ignore"] = "raise",
) -> str:
    """
    Get GCM surrogate name from a pattern model name

    Raises
    ------
    KeyError : If input does not exist or has no surrogate, unless ``errors="ignore"``.

    Examples
    --------
    >>> get_cmip5_surrogate_name(scenario="rcp45", pattern="pattern28")
    "surrogate_CanESM2_89"
    """
    # Function is from https://gitlab.com/rhodium/impactlab-rhg/carb-cvm/data-prep/-/blob/2bb3e6c8b587befafa6f62b186dc2ad83248c3db/src/dprep/climate/surrogates.py
    # Prefer this over creating custom read-only UserDict...
    pattern_map = {
        "rcp45": {
            "pattern1": "surrogate_MRI-CGCM3_01",
            "pattern2": "surrogate_GFDL-ESM2G_01",
            "pattern3": "surrogate_MRI-CGCM3_06",
            "pattern4": "surrogate_GFDL-ESM2G_06",
            "pattern5": "surrogate_MRI-CGCM3_11",
            "pattern6": "surrogate_GFDL-ESM2G_11",
            "pattern27": "surrogate_GFDL-CM3_89",
            "pattern28": "surrogate_CanESM2_89",
            "pattern29": "surrogate_GFDL-CM3_94",
            "pattern30": "surrogate_CanESM2_94",
            "pattern31": "surrogate_GFDL-CM3_99",
            "pattern32": "surrogate_CanESM2_99",
        },
        "rcp85": {
            "pattern1": "surrogate_MRI-CGCM3_01",
            "pattern2": "surrogate_GFDL-ESM2G_01",
            "pattern3": "surrogate_MRI-CGCM3_06",
            "pattern4": "surrogate_GFDL-ESM2G_06",
            "pattern5": "surrogate_MRI-CGCM3_11",
            "pattern6": "surrogate_GFDL-ESM2G_11",
            "pattern28": "surrogate_GFDL-CM3_89",
            "pattern29": "surrogate_CanESM2_89",
            "pattern30": "surrogate_GFDL-CM3_94",
            "pattern31": "surrogate_CanESM2_94",
            "pattern32": "surrogate_GFDL-CM3_99",
            "pattern33": "surrogate_CanESM2_99",
        },
    }

    # We always throw an error if scenario is bad.
    m = pattern_map[scenario]
    # Maybe ignore error if pattern is bad...
    try:
        out = m[pattern]
    except KeyError:
        if errors == "ignore":
            out = str(pattern)
        else:
            raise

    return out


all_targets = [("rcp45", m) for m in CMIP5_RCP45_SMME_MODELS] + [
    ("rcp85", m) for m in CMIP5_RCP85_SMME_MODELS
]

raw_weights = pd.read_csv(RAW_WEIGHTS_PATH, index_col=["rcp", "model"]).to_xarray()

out_ensemble_member = []
out_weight = []
for scenario, model in all_targets:
    # If it's a pattern model name, get the surrogate name.
    not_pattern_model_name = get_cmip5_surrogate_name(
        scenario=scenario, pattern=model, errors="ignore"
    )

    w = (
        raw_weights["weight"]
        .sel(rcp=scenario, model=not_pattern_model_name.lower())
        .item()
    )

    out_ensemble_member.append(f"{scenario}/{model}")
    out_weight.append(w)

new_weights = xr.DataArray(
    out_weight,
    dims=("ensemble_member",),
    coords={"ensemble_member": out_ensemble_member},
    name="smme_weight",
)
new_weights.to_zarr(OUT_WEIGHTS_URI, mode="w")
print(OUT_WEIGHTS_URI)
