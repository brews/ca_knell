"""
Logic for valuing labor impacts into damages.
"""

from muuttaa import Projector
import xarray as xr


def _no_processing(ds: xr.Dataset) -> xr.Dataset:
    return ds


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
