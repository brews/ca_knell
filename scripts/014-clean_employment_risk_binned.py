"""
Employment Binning by Risk Categorization

This notebook takes in a csv file containing employment data broken down by industry type and bins the industries by risk cetegory to return a sum of workers in each census track working in High and Low risk jobs.
The data used is ACS Table S2405 at Census Track Resolution for all tracts in the state of California.

Adapted from a notebook by Stefan Klos.
"""

import datetime
import os
import uuid

import pandas as pd


JUPYTER_IMAGE = os.environ.get("JUPYTER_IMAGE")
UID = str(uuid.uuid4())
START_TIME = datetime.datetime.now(datetime.UTC).isoformat()
IN_CSV_URI = "gs://rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/source_data/employment_data/ACSST5Y2019.S2405_data_with_overlays_2021-07-08T112937.csv"
OUT_URI = f"{os.environ['CIL_SCRATCH_PREFIX']}/{os.environ['JUPYTERHUB_USER']}/{UID}/population_age_binned.csv"


# Need GEOID to be str, otherwise it's read in as int. Need to skip row with field descriptions.
df = pd.read_csv(
    IN_CSV_URI,
    dtype={"GEO_ID": str},
    skiprows=[1],
    usecols=[
        "GEO_ID",
        "NAME",
        "S2405_C01_001E",
        "S2405_C01_002E",
        "S2405_C01_003E",
        "S2405_C01_004E",
        "S2405_C01_005E",
        "S2405_C01_006E",
        "S2405_C01_007E",
        "S2405_C01_008E",
        "S2405_C01_009E",
        "S2405_C01_010E",
        "S2405_C01_011E",
        "S2405_C01_012E",
        "S2405_C01_013E",
        "S2405_C01_014E",
    ],
)

# Creating a new columns with a shortened GEOID info to only include the information at the resolution of the state level (so as to only include the 06 california code and everything after)
# Dropping the lengthier GEO_ID Column now that there is a more concise one that will match the California Tiger Shapefile's GEOID scheme
df["region"] = df["GEO_ID"].str[9:]

df = df.drop(["GEO_ID", "NAME"], axis=1)
df = df.rename(
    columns={
        "S2405_C01_001E": "total_employed",
        "S2405_C01_002E": "farm_mine_hunt_foresty",
        "S2405_C01_003E": "construction",
        "S2405_C01_004E": "manufacturing",
        "S2405_C01_005E": "wholesale_trade",
        "S2405_C01_006E": "retail_trade",
        "S2405_C01_007E": "transport_warehouse_utility",
        "S2405_C01_008E": "information",
        "S2405_C01_009E": "finance_realestate_renting",
        "S2405_C01_010E": "science_admin_wasteManage",
        "S2405_C01_011E": "education_healthcare",
        "S2405_C01_012E": "entertainment_hotel_food",
        "S2405_C01_013E": "other",
        "S2405_C01_014E": "publicAdmin",
    }
)
df = df.set_index("region")

# Reduce sectors into risk sectors
df["low"] = (
    df["construction"]
    + df["farm_mine_hunt_foresty"]
    + df["manufacturing"]
    + df["transport_warehouse_utility"]
)
df["high"] = (
    df["wholesale_trade"]
    + df["retail_trade"]
    + df["information"]
    + df["finance_realestate_renting"]
    + df["science_admin_wasteManage"]
    + df["education_healthcare"]
    + df["entertainment_hotel_food"]
    + df["other"]
    + df["publicAdmin"]
)

# Just what is needed.
df = df[["total_employed", "high", "low"]]

# Calculate and add risk sector population shares.
for cohort in ["low", "high"]:
    df[f"{cohort}_share"] = df[cohort] / df["total_employed"]

df.to_cvs(OUT_URI)
