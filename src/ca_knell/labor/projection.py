"""
Logic for projecting labor impacts from transformed data.
"""

from muuttaa import Projector
import xarray as xr


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
