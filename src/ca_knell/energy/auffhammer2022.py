"""
Code to calculate Max Auffhammer-style decimal degree variables.

Specifically, this is estimating 'CDD', 'CDD2', 'HDD', from
daily maximum and minimum air temperature fields. This method is from
Max Auffhammer (2022) "Climate Adaptive Response Estimation: Short
And Long Run Impacts Of Climate Change On Residential Electricity
and Natural Gas Consumption"
"""

import numpy as np
import xarray as xr


def _degf_to_degc(f):
    return (f - 32) * (5 / 9)


def _estimate_hourly_tas(
    daily_ds, *, tasmax_name="tasmax", tasmin_name="tasmin", time_name="time"
):
    """
    Estimate hourly "tas" DataArray from Dataset of daily tasmin, tasmax with sine wave

    This replicates method used by Max Auffhammer (2022) "Climate Adaptive
    Response Estimation: Short And Long Run Impacts Of Climate Change On
    Residential Electricity and Natural Gas Consumption".
    """
    hourly = daily_ds.resample({time_name: "1h"}).pad()

    # Translated from Max's Stat code but breaking this down into functional parts.
    amplitude = (hourly[tasmax_name] - hourly[tasmin_name]) / 2
    period = (hourly[time_name].dt.hour / 24) * 2 * np.pi
    vertical_shift = (
        hourly[tasmin_name] + (hourly[tasmax_name] - hourly[tasmin_name]) / 2
    )
    return amplitude * np.cos(period) + vertical_shift


def _auffhammer_cdd(tas_da, threshold):
    """Cooling degree-days"""
    out = (tas_da - threshold).where(tas_da >= threshold, 0)
    # Re-insert original nans before returning.
    out = out.where(~tas_da.isnull(), np.nan)

    # Roughly based on xclim degree-day docs but hacked to work with
    # Auffhammer's approach. This doesn't have everything required like
    # proper "cell_methods" or bounds.
    out.attrs.update(
        {
            "long_name": f"Cooling degree days (Thour > {threshold})",
            "standard_name": "integral_of_air_temperature_excess_wrt_time",
            "units": "K days",
            "comment": (
                "Calculated from hourly air temperature data estimated from daily air temperature data "
                "following methods used by Max Auffhammer's 2018 research for California Air Resources Board."
            ),
        }
    )

    return out


def _auffhammer_hdd(tas_da, threshold):
    """Heating degree-days"""
    out = (threshold - tas_da).where(threshold >= tas_da, 0)
    # Re-insert original nans before returning.
    out = out.where(~tas_da.isnull(), np.nan)

    # Roughly based on xclim degree-day docs but hacked to work with
    # Auffhammer's approach.
    out.attrs.update(
        {
            "long_name": f"Heating degree days (Thour < {threshold})",
            "standard_name": "integral_of_air_temperature_deficit_wrt_time",
            "units": "K days",
            "comment": (
                "Calculated from hourly air temperature data estimated from daily air temperature data "
                "following methods used by Max Auffhammer's 2018 research for California Air Resources Board."
            ),
        }
    )

    return out


def auffhammer_dds(
    ds: xr.Dataset, tasmax_name: str = "tasmax", tasmin_name: str = "tasmin"
) -> xr.Dataset:
    """
    Get Dataset of annual-sum Auffhammer (2022)-style HDD, CDD, CDD2, in degC.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a daily minimum and maximum air temperature variable.
        These variables need to at least be over a CF-compliant 'time'
        coordinate variable. Both temperature variables should be in
        units degC. Though this is not enforced.
    tasmax_name : str, optional
        Name of the maximum daily air temperature variable in `ds`.
    tasmin_name : str, optional
        Name of the minimum daily air temperature variable in `ds`.

    Returns
    -------
    dds : xr.Dataset
        A dataset with the annual sum of daily degree day (DD) variables,
        'hdd', 'cdd', 'cdd2' (see Auffhammer 2022 for details). In units
        'K day' or 'degC day'.

    References
    ----------
    ..  Max Auffhammer (2022) 'Climate Adaptive Response Estimation: Short
        And Long Run Impacts Of Climate Change On Residential Electricity
        and Natural Gas Consumption'.
    """
    # Would be nice to have a more definitive citation for the original
    # Auffhammer reference but this is the best I have for now.
    b = _degf_to_degc(65)
    b2 = _degf_to_degc(80)

    hr_estimate = _estimate_hourly_tas(
        ds, tasmax_name=tasmax_name, tasmin_name=tasmin_name
    )

    # Dividing sum by 24 here accounts for us calculating a "daily" value
    # from an hourly estimate...
    with xr.set_options(keep_attrs=True):
        dds = (
            xr.Dataset(
                {
                    "cdd": _auffhammer_cdd(hr_estimate, b)
                    .groupby("time.year")
                    .sum(skipna=False),
                    "cdd2": _auffhammer_cdd(hr_estimate, b2)
                    .groupby("time.year")
                    .sum(skipna=False),
                    "hdd": _auffhammer_hdd(hr_estimate, b)
                    .groupby("time.year")
                    .sum(skipna=False),
                }
            )
            / 24
        )

    return dds
