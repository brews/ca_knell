"""
Reads in and cleans ACS CA census tract per capita income for the year 2019, 5-year estimate.
"""

import datetime
import os
import uuid

import pandas as pd


JUPYTER_IMAGE = os.environ.get("JUPYTER_IMAGE")
UID = str(uuid.uuid4())
START_TIME = datetime.datetime.now(datetime.UTC).isoformat()
IN_CSV_URI = "gs://rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/source_data/per_capita_income_data/census_tract/ACSDT5Y2019.B19301_data_with_overlays_2021-08-10T135627.csv"
OUT_URI = f"{os.environ['CIL_SCRATCH_PREFIX']}/{os.environ['JUPYTERHUB_USER']}/{UID}/PCI_2019.csv"

# Need GEOID to be str, otherwise it's read in as int. Need to skip row with field descriptions.
df_raw = pd.read_csv(
    IN_CSV_URI,
    na_values=["-"],
    dtype={"GEO_ID": str},
    skiprows=[1],
)

df = df_raw
# Creating a new columns with a shortened GEOID info to only include the information at the resolution of the state level (so as to only include the 06 california code and everything after)
# Dropping the lengthier GEO_ID Column now that there is a more concise one that will match the California Tiger Shapefile's GEOID scheme
df["region"] = df["GEO_ID"].str[9:]
# Select only what we need and rename.
df = df[["region", "B19301_001E"]].rename(columns={"B19301_001E": "pci"})
df = df.set_index("region")

df.to_csv(OUT_URI)
