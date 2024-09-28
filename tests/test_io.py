import io
import pytest
import fsspec
import numpy as np

from ca_knell.io import read_csvv, open_carb_segmentweights


@pytest.fixture
def csvv_filebuffer():
    fl_content = """---
oneline: oneline
version: version
dependencies: dependencies
description: description
csvv-version: girdin-2017-01-10
variables:
  k1: v1
...
observations
123
prednames
a,b,c
covarnames
1,1,1
gamma
1.0,2.0,3.0
gammavcv
1.0,2.0,3.0
1.0,2.0,3.0
1.0,2.0,3.0
residvcv
123.0
"""
    return io.StringIO(fl_content)


@pytest.mark.parametrize(
    "target_attr,expected",
    [
        ("observations", 123),
        ("prednames", ["a", "b", "c"]),
        ("covarnames", ["1", "1", "1"]),
        ("residvcv", 123.0),
    ],
)
def test_read_csvv(csvv_filebuffer, target_attr, expected):
    """Test that read_csvv() gives Csvv with correct non-ndarray attributes"""
    test_url = "memory://_test_read_csvv.csvv"
    with fsspec.open(test_url, mode="w") as fl:
        fl.write(csvv_filebuffer.getvalue())

    csvv = read_csvv(test_url)

    assert csvv[target_attr] == expected


@pytest.mark.parametrize(
    "target_attr,expected",
    [
        ("gamma", np.array([1, 2, 3], dtype="float")),
        (
            "gammavcv",
            np.repeat(np.array([[1, 2, 3]], dtype="float"), repeats=3, axis=0),
        ),
    ],
)
def test_read_csvv_arrays(csvv_filebuffer, target_attr, expected):
    """Test that read_csvv() gives Csvv with correct ndarray attributes"""
    test_url = "memory://_test_read_csvv_arrays.csvv"
    with fsspec.open(test_url, mode="w") as fl:
        fl.write(csvv_filebuffer.getvalue())

    csvv = read_csvv(test_url)

    np.testing.assert_allclose(csvv[target_attr], expected)


def test_open_carb_segmentweights_runs():
    """
    Basic test that open_carb_segmentweights runs without error.
    """
    fl_content = """
longitude,latitude,GEOID,weight
2.0,0.0,999,1.0
"""
    _ = open_carb_segmentweights(io.StringIO(fl_content))
