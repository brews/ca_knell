"""
Logic for projecting energy impacts from transformed data.
"""

from muuttaa import Projector
import pandas as pd
import xarray as xr


def _no_processing(ds: xr.Dataset) -> xr.Dataset:
    return ds


def _energy_impact_model(ds: xr.Dataset) -> xr.Dataset:
    betas = ds["beta"]

    # convert hdd/cdd norms and t variables to F
    hdd = ds["hdd"] * (9 / 5)
    cdd = ds["cdd"] * (9 / 5)
    cdd2 = ds["cdd2"] * (9 / 5)

    # create Data Array of predictors from temp array
    climpreds = xr.concat(
        [hdd, cdd, cdd2],
        dim=pd.Index(["HDD_WS", "CDD_WS", "CDD_WS2"], name="predictor"),
    ).astype("float32")

    # TODO: Do we need the `.map()` here?
    # take dot product of covars, preds, gammas by model
    _effect = betas.map(lambda x: xr.dot(x, climpreds, dims=["predictor"]))

    # impacts are difference of future - historical effect
    impact = _effect.sel(year=2050) - _effect.sel(year=2020)

    return xr.Dataset({"impact": impact})


# If you already have beta.
energy_impact_model = Projector(
    preprocess=_no_processing,
    project=_energy_impact_model,
    postprocess=_no_processing,
)
