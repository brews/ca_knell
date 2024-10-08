"""
Logic for valuing energy impacts into damages.
"""

from muuttaa import Projector
import pandas as pd
import xarray as xr


def _no_processing(ds: xr.Dataset) -> xr.Dataset:
    return ds


def _energy_valuation_model(ds: xr.Dataset) -> xr.Dataset:
    # Calculate electricity damages
    # total impacts = impacts in kWh/year * kWh price * n households in tract
    tot_electricity_damages = ds["electricity_levels"] * ds["kWh_price"] * ds["hhcount"]

    # impacts per capita = total impacts / n people in tract
    electricity_damages_pc = tot_electricity_damages / ds["pop"]

    # impacts as share of income = impacts per capita / tract per capita income
    electricity_damages_pincome = electricity_damages_pc / ds["pci"]

    # Combine different valuation methods for electricity model into new coordinate.
    electricity_damages = xr.concat(
        [tot_electricity_damages, electricity_damages_pc, electricity_damages_pincome],
        dim=pd.Index(
            ["damages_total", "damages_pc", "damages_pincome"], name="valuation"
        ),
    )

    # Calculate gas damages
    # total impacts = impacts in therm/year * therm price * n households in tract
    tot_gas_damages = ds["gas_levels"] * ds["therm_price"] * ds["hhcount"]

    # impacts per capita = total impacts / n people in tract
    gas_damages_pc = tot_gas_damages / ds["pop"]

    # impacts as share of income = impacts per capita / tract per capita income
    gas_damages_pincome = gas_damages_pc / ds["pci"]

    # Combine different valuation methods for gas model into new coordinate.
    gas_damages = xr.concat(
        [tot_gas_damages, gas_damages_pc, gas_damages_pincome],
        dim=pd.Index(
            ["damages_total", "damages_pc", "damages_pincome"], name="valuation"
        ),
    )

    # Combine electricity and gas damages and spit it out.
    damages = xr.Dataset(
        {
            "electricity_consumption": electricity_damages,
            "gas_consumption": gas_damages,
        }
    )
    return damages


energy_valuation_model = Projector(
    preprocess=_no_processing,
    project=_energy_valuation_model,
    postprocess=_no_processing,
)
