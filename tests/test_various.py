"""
Testing for various odds and ends until this takes more structure.
"""

import xarray as xr

from ca_knell.labor.projection import rcspline


def test_rcspline():
    """
    Basic test of rcspline spliney-ness.
    """
    da_in = xr.DataArray(range(25, 60))
    expected = xr.DataArray(
        [
            0.000e00,
            0.000e00,
            0.000e00,
            1.000e00,
            8.000e00,
            2.700e01,
            6.400e01,
            1.250e02,
            2.160e02,
            3.430e02,
            5.120e02,
            7.290e02,
            1.000e03,
            1.325e03,
            1.680e03,
            2.040e03,
            2.400e03,
            2.760e03,
            3.120e03,
            3.480e03,
            3.840e03,
            4.200e03,
            4.560e03,
            4.920e03,
            5.280e03,
            5.640e03,
            6.000e03,
            6.360e03,
            6.720e03,
            7.080e03,
            7.440e03,
            7.800e03,
            8.160e03,
            8.520e03,
            8.880e03,
        ]
    )

    actual = rcspline(da_in)
    xr.testing.assert_equal(actual, expected)
