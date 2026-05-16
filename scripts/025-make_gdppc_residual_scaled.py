"""
Downscaling county GDP per capita to tract level data using distribution of PCI

This process generates 3 additional long run income variables which we tested as an input to the mortality beta generation pipeline - the variable we settled on after testing is gdppc_residual_scaled, which distributes the residual of county GDP and Income evenly to all individuals in the given county.Inputs: BEA county level GDP, ACS Income per CapitaOutputs: Figure 2 from the paper, /2021-carb-cvm/data-prep/output_data/income_adjusted.nc4 which contains income data

Adapted from a notebook by Stefan Klos.
"""

import glob

import pandas as pd
import numpy as np
import os


def load_census_per_capita_income(input_path, resolution="tract"):
    """
        Function Purpose

    1) read in a folder of csv files and turn it into a pandas datframe object
    2) rename the columns to reflect that each column of Per Capit Income is from a different year
    3) convert the values in the income columns from objects to float64
    4) create a new GEOID column with string modifications

        Parameters
    input_path:str of location of folder containin csv files
    resolution: str of income data resolution
        "tract": default, census tract level
        "county": for loading county data

    """

    # create list of file paths to csvs
    all_files = sorted(glob.glob(input_path + "/*.csv"))

    # load all in object that is a list of dataframes
    dfs = [pd.read_csv(f, index_col="GEO_ID") for f in all_files]

    # create dataframe with just geoid and name
    df = pd.DataFrame(dfs[0]["NAME"])

    # adding and renaming B19301_001E column file by file
    for i in np.arange(0, len(dfs), 1):
        df = df.merge(dfs[i]["B19301_001E"], left_index=True, right_index=True).rename(
            columns={"B19301_001E": i + 2010}
        )

    # drop first row which is column details
    df = df.iloc[1:].copy()

    # convert strs to values
    for col in np.arange(2010, 2020, 1):
        df[col] = df[col].apply(pd.to_numeric, errors="coerce")

    # reset index to access it as a column
    df = df.reset_index()

    if resolution == "county":
        # create countyFP to match crosswalk
        df["COUNTYFP"] = df["GEO_ID"].astype(str).str[-3:]
        # isolate county name
        df["county"] = df["NAME"].str.split(" County", expand=True)[0]
    else:
        df["GEOID"] = df["GEO_ID"].astype(str).str[9:]

    return df


def convert_dollars(cpi_file, income_series, series_year, to_year=2005):
    """Returns Income Per Capita in 2005 real dollars (or any other year).
    Defaults to intake nominal PCI, but also can rebase real series.
    """

    # Load Fed GDP deflator data.
    fed = pd.read_csv(cpi_file).set_index("year")
    fed_gdpdef = fed["gdpdef"].to_dict()

    # Step 1: convert deflator to $2005
    fed["base_year"] = fed["gdpdef"] / fed_gdpdef[to_year] * 100

    # Step 2: take only the years of the input data
    def_slice = fed.loc[series_year, "base_year"]

    # Step 3: divide series pairwise by GDP deflator
    adjusted = income_series / def_slice * 100

    return adjusted


def main():
    # path to raw census pci data
    pci_path = "/gcs/rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/source_data/per_capita_income_data"
    clean_path = "/gcs/rhg-data/impactlab-rhg/client-projects/2021-carb-cvm/data-prep/output_data/"

    # prepare county data (use this to fix/generalize core)
    county_pci = load_census_per_capita_income(
        input_path=os.path.join(pci_path, "county"), resolution="county"
    )

    # Prepare LR log
    # deflate to 2005$
    cpi_file = "../data/fed_income_inflation.csv"

    for year in np.arange(2010, 2020, 1):
        county_pci[f"{year}_def"] = convert_dollars(
            cpi_file=cpi_file,
            income_series=county_pci[year],
            series_year=year,
            to_year=2005,
        )

    # average across deflated columns
    def_list = [f"{year}_def" for year in np.arange(2010, 2020, 1)]

    county_pci["pci_county"] = county_pci[def_list].mean(axis=1)

    county_pci.rename(columns={2019: "2019_county_pci"}, inplace=True)

    # load census data, merge crosswalk, merge county data
    ## county to census tract crosswalk
    cw = pd.read_csv(
        "/gcs/rhg-data/impactlab-rhg/spatial/shapefiles/source/us_census/ca_county_tract.csv",
        dtype="str",
    )

    # read in cleaned tract income
    tract_pci = pd.read_csv(
        os.path.join(clean_path, "logpci.csv"), index_col=0, dtype={"GEOID": str}
    )
    # read in raw tract income
    tract_raw = load_census_per_capita_income(
        input_path=os.path.join(pci_path, "census_tract"), resolution="tract"
    )

    # merge crosswalk
    tract_pci = tract_pci.merge(cw, how="inner", on="GEOID")
    tract_raw = tract_raw.merge(cw, how="inner", on="GEOID")

    tract_raw.rename(columns={2019: "2019_tract_pci"}, inplace=True)
    tract_pci = tract_pci.merge(
        tract_raw[["GEOID", "2019_tract_pci"]], how="left", on="GEOID"
    )

    # merge county data
    dft = tract_pci.merge(
        county_pci[["COUNTYFP", "county", "pci_county", "2019_county_pci"]],
        how="left",
        on="COUNTYFP",
    ).copy()

    # take ratio of tract income to county income
    dft["tract_county_ratio"] = dft["pci"] / dft["pci_county"]

    # Clean and calculate BEA GDPpc
    dpath = "../data"
    fbea = "bea_CA_current_GDP.csv"
    fpop = "PEPPOP2019.PEPANNRES_data_with_overlays_2021-11-19T014448.csv"
    # load bea
    bea = pd.read_csv(os.path.join(dpath, fbea), dtype={"COUNTYFP": str})

    # columns = years
    bea = bea.drop(columns=["GeoFips", "GeoName"]).set_index("COUNTYFP")
    gdp_cols = [f"{year}_gdp" for year in np.arange(2010, 2020, 1)]
    bea.columns = gdp_cols

    # convert to 2005$ and multiply by 1000s so units are in $
    for year in np.arange(2010, 2020, 1):
        bea[f"{year}_gdp_def"] = (
            convert_dollars(
                cpi_file=cpi_file,
                income_series=bea[f"{year}_gdp"],
                series_year=year,
                to_year=2005,
            )
            * 1000
        )
        # also multiply undeflated series by 1000
        bea[f"{year}_gdp"] = bea[f"{year}_gdp"] * 1000

    # load acs pop
    pop = pd.read_csv(os.path.join(dpath, fpop))[1:].copy()

    # take county code
    pop["COUNTYFP"] = pop["GEO_ID"].str[-3:]

    # pivot from long to wide
    pop = pd.pivot(
        pop, values="POP", index="COUNTYFP", columns="DATE_CODE"
    ).reset_index()

    # drop actual census data (acs data only)
    pop = pop.drop(
        columns=["4/1/2010 Census population", "4/1/2010 population estimates base"]
    ).set_index("COUNTYFP")

    # rename columns to years
    pop_cols = [f"{year}_pop" for year in np.arange(2010, 2020, 1)]
    pop.columns = pop_cols

    # convert strs to values
    for col in pop_cols:
        pop[col] = pop[col].apply(pd.to_numeric, errors="coerce")

    # merge county pop into county gdp
    gdp = bea.merge(pop, how="left", left_index=True, right_index=True)

    # calculate gdppc (bea data is in thousands of $)
    for year in np.arange(2010, 2020, 1):
        gdp[f"{year}_gdppc"] = gdp[f"{year}_gdp"] / gdp[f"{year}_pop"]
        gdp[f"{year}_gdppc_def"] = gdp[f"{year}_gdp_def"] / gdp[f"{year}_pop"]

    # 10 yr avgs
    gdp_cols = [f"{year}_gdp_def" for year in np.arange(2010, 2020, 1)]
    gdppc_cols = [f"{year}_gdppc_def" for year in np.arange(2010, 2020, 1)]

    gdp["lr_gdp_county"] = gdp[gdp_cols].mean(axis=1)
    gdp["lr_gdppc_county"] = gdp[gdppc_cols].mean(axis=1)

    # merge gdppc onto df
    dft = dft.merge(
        gdp[["lr_gdp_county", "lr_gdppc_county", "2019_gdp", "2019_gdppc", "2019_pop"]],
        how="left",
        left_on="COUNTYFP",
        right_index=True,
    )

    # Merge in tract population
    # TODO: Do we need this section?
    tract_pop = pd.read_csv(
        os.path.join(clean_path, "population_age_binned.csv"), dtype={"GEOID": str}
    ).set_index("GEOID")
    dft = dft.merge(
        tract_pop["total_tract_population"],
        how="left",
        left_on="GEOID",
        right_index=True,
    )

    # Scaling option #2
    # take the difference between county GDP and income, distribute it so that each person within a county gets an equal share of the county residual

    # calculate the total county income by multiplying by population
    dft["income_county"] = dft["pci_county"] * dft["2019_pop"]
    dft["income_county_2019"] = dft["2019_county_pci"] * dft["2019_pop"]

    # Subtract Income from GDP to get county residual
    dft["resid"] = dft["lr_gdp_county"] - dft["income_county"]
    dft["resid_2019"] = dft["2019_gdp"] - dft["income_county_2019"]

    # calc resid per capita within the county
    dft["resid_pc"] = dft["resid"] / dft["2019_pop"]
    dft["resid_2019_pc"] = dft["resid_2019"] / dft["2019_pop"]

    # add the residual per capita (same within county) to income per capita (different within county)
    dft["gdppc_residual_scaled"] = dft["pci"] + dft["resid_pc"]
    dft["loggdppc_residual_scaled"] = np.log(dft["gdppc_residual_scaled"])

    # same for 2019 current
    dft["gdppc_residual_scaled_2019"] = dft["2019_tract_pci"] + dft["resid_2019_pc"]


if __name__ == "__main__":
    main()
