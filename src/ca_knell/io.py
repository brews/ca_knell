import csv
import re
from os import PathLike
from io import BufferedIOBase
from typing import Any

import fsspec
import metacsv
from muuttaa import SegmentWeights
import numpy as np


def open_carb_segmentweights(
    url: str | PathLike[Any] | BufferedIOBase,
) -> SegmentWeights:
    """Open SegmentWeights from CARB project weights file"""
    import pandas as pd

    sw = pd.read_csv(
        url,
        dtype={"GEOID": str},  # Otherwise ID is read as int...
    )
    sw["longitude"] = (sw["longitude"] + 180) % 360 - 180
    sw = sw.to_xarray().rename_vars(
        {"longitude": "lon", "latitude": "lat", "GEOID": "region"}
    )
    return SegmentWeights(sw)


def read_csvv(filename):
    """Interpret a CSVV file into a dictionary of the included information.

    Specific implementation is described in the two CSVV version
    readers, `read_girdin` and `csvvfile_legacy.read`.
    """
    with fsspec.open(filename, "r") as fp:
        attrs, coords, variables = metacsv.read_header(fp, parse_vars=True)

        # Clean up variables
        for variable in variables:
            vardef = variables[variable[0]]
            assert isinstance(vardef, dict), (
                f"Variable definition {vardef} malformed."
            )
            if "unit" in vardef:
                fullunit = vardef["unit"]
                if "]" in fullunit:
                    vardef["unit"] = fullunit[: fullunit.index("]")]
            else:
                print(f"WARNING: Missing unit for variable {variable}.")
                vardef["unit"] = None

        data = {"attrs": attrs, "variables": variables, "coords": coords}

        # `attrs` should have "csvv-version" otherwise should be read in with
        # `csvvfile_legacy.read` - but I'm not sure what this actually is.
        csvv_version = attrs["csvv-version"]
        if csvv_version == "girdin-2017-01-10":
            return _read_girdin(data, fp)
        else:
            raise ValueError("Unknown version " + csvv_version)


def _read_girdin(data, fp):
    """Interpret a Girdin version CSVV file into a dictionary of the
    included inforation.

    A Girdin CSVV has a lists of predictor and covariate names, which
    are matched up one-for-one.  This offered more flexibility and
    clarity than the previous version of CSVV files.

    Parameters
    ----------
    data : dict
        Meta-data from the MetaCSV description.
    fp : file pointer
        File pointer to the start of the file content.

    Returns
    -------
    dict
        Dictionary with MetaCSV information and the predictor and
    covariate information.
    """
    reader = csv.reader(fp)
    variable_reading = None

    for row in reader:
        if len(row) == 0 or (len(row) == 1 and len(row[0].strip()) == 0):
            continue
        row[0] = row[0].strip()

        if row[0] in [
            "observations",
            "prednames",
            "covarnames",
            "gamma",
            "gammavcv",
            "residvcv",
        ]:
            data[row[0]] = []
            variable_reading = row[0]
        else:
            if variable_reading is None:
                print("No variable queued.")
                print(row)
            assert variable_reading is not None
            if len(row) == 1:
                row = row[0].split(",")
            if len(row) == 1:
                row = row[0].split("\t")
            if len(row) == 1:
                row = re.split(r"\s", row[0])
            data[variable_reading].append([x.strip() for x in row])

    data["observations"] = float(data["observations"][0][0])
    data["prednames"] = data["prednames"][0]
    data["covarnames"] = data["covarnames"][0]
    data["gamma"] = np.array(list(map(float, data["gamma"][0])))
    data["gammavcv"] = np.array([list(map(float, row)) for row in data["gammavcv"]])
    data["residvcv"] = np.array([list(map(float, row)) for row in data["residvcv"]])
    return data
