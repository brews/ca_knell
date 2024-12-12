"""
Take raw csv file containing population data seperated into 18 bins and rebins the data into 3 larger bins (<5, 5-64, 65+).
Input data used is American Community Survey(ACS) Table S0101 at Census Track Resolution for all tracts in the state of California.

Adapted from a notebook by Stefan Klos.
"""

import datetime
import os
import uuid
import pandas as pd


JUPYTER_IMAGE = os.environ.get("JUPYTER_IMAGE")
UID = str(uuid.uuid4())
START_TIME = datetime.datetime.now(datetime.UTC).isoformat()
IN_CSV_URI = "gs://rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/source_data/population_data/ACSST5Y2019.S0101_data_with_overlays_2021-06-22T101516.csv"
OUT_URI = f"{os.environ['CIL_SCRATCH_PREFIX']}/{os.environ['JUPYTERHUB_USER']}/{UID}/population_age_binned.csv"


# Need GEOID to be str, otherwise it's read in as int. Need to skip row with field descriptions.
df_raw = pd.read_csv(
    IN_CSV_URI,
    dtype={"GEO_ID": str},
    skiprows=[1],
    usecols=[
        "GEO_ID",
        "NAME",
        "S0101_C01_001E",
        "S0101_C01_002E",
        "S0101_C01_003E",
        "S0101_C01_004E",
        "S0101_C01_005E",
        "S0101_C01_006E",
        "S0101_C01_007E",
        "S0101_C01_008E",
        "S0101_C01_009E",
        "S0101_C01_010E",
        "S0101_C01_011E",
        "S0101_C01_012E",
        "S0101_C01_013E",
        "S0101_C01_014E",
        "S0101_C01_015E",
        "S0101_C01_016E",
        "S0101_C01_017E",
        "S0101_C01_018E",
        "S0101_C01_019E",
    ],
)

df = df_raw
# Creating a new columns with a shortened GEOID info to only include the information at the resolution of the state level (so as to only include the 06 california code and everything after)
# Dropping the lengthier GEO_ID Column now that there is a more concise one that will match the California Tiger Shapefile's GEOID scheme
df["region"] = df_raw["GEO_ID"].str[9:]

df = df.drop(["GEO_ID", "NAME"], axis=1)
df = df.rename(
    columns={
        "S0101_C01_001E": "combined",  # total_tract_population
        "S0101_C01_002E": "under_5",
        "S0101_C01_003E": "5-9",
        "S0101_C01_004E": "10-14",
        "S0101_C01_005E": "15-19",
        "S0101_C01_006E": "20-24",
        "S0101_C01_007E": "25-29",
        "S0101_C01_008E": "30-34",
        "S0101_C01_009E": "35-39",
        "S0101_C01_010E": "40-44",
        "S0101_C01_011E": "45-49",
        "S0101_C01_012E": "50-54",
        "S0101_C01_013E": "55-59",
        "S0101_C01_014E": "60-64",
        "S0101_C01_015E": "65-69",
        "S0101_C01_016E": "70-74",
        "S0101_C01_017E": "75-79",
        "S0101_C01_018E": "80-84",
        "S0101_C01_019E": "over_85",
    }
)
df = df.set_index("region")

# Aggregate into age cohorts for mortality model.
df["age1"] = df[["under_5"]].sum(axis=1)  # pop_lt5
df["age2"] = df[
    [
        "5-9",
        "10-14",
        "15-19",
        "20-24",
        "25-29",
        "30-34",
        "35-39",
        "40-44",
        "45-49",
        "50-54",
        "55-59",
        "60-64",
    ]
].sum(axis=1)  # pop_5-64
df["age3"] = df[["65-69", "70-74", "75-79", "80-84", "over_85"]].sum(axis=1)  # pop_65+

# Sanity check.
math_check = df["combined"] - (df["age1"] + df["age2"] + df["age3"])
assert (
    math_check == 0
).all(), "all regional age-cohort counts did not match the total count"

# Just what is needed.
df = df[["combined", "age1", "age2", "age3"]]

# Calculate and add age population shares.
for cohort in ["age1", "age2", "age3"]:
    df[f"{cohort}_share"] = df[cohort] / df["combined"]


# Figure 3
# tmp = (
#     df[["age1", "age2", "age3", "combined"]]
#     .rename(columns={"age1": "pop_lt5", "age2": "pop_5-64", "age3": "pop_65+", "combined": "total_tract_population"})
#     .sum()
# )
# details = {
#     'Population Bins' :['Total State Population', 'pop_lt5', 'pop_5-64','pop_65+'
#                         ],
#
#     'Population_Bin_Totals' : [tmp['total_tract_population'], tmp['pop_lt5'],
#                                tmp['pop_5-64'], tmp['pop_65+']]
#
#
# }
# # creating a Dataframe object
# pop19_sums = pd.DataFrame(details)
# pop19_sums.set_index('Population Bins', inplace=True)
#
# pop19_sums.plot.barh(color=['grey','blue','blue','blue'])


df.to_cvs(OUT_URI)
