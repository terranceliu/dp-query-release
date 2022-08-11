from typing import *

import numpy as np
import pandas as pd

from .data_schema import DataSchema
from .census_query_definitions import census_queries
from .geo_location import GeoLocation

def select_ppmf_geolocation(ppmf: pd.DataFrame, geolocation: GeoLocation):
    if geolocation.state_id is not None:
        ppmf = ppmf[ppmf["TABBLKST"] == geolocation.state_id]
    if geolocation.county_id is not None:
        ppmf = ppmf[ppmf["TABBLKCOU"] == geolocation.county_id]
    if geolocation.census_tract is not None:
        ppmf = ppmf[ppmf["TABTRACTCE"] == geolocation.census_tract]
    if geolocation.block is not None:
        ppmf = ppmf[ppmf["TABBLK"] == geolocation.block]
    return ppmf

def get_census_schema_and_data(ppmf) -> Tuple[DataSchema, pd.DataFrame]:
    """`get_census_schema_and_data(ppmf: pd.DataFrame) -> Tuple[DataSchema, pd.DataFrame]`
    
    Given the privacy protected microdata file (stored as a pandas dataframe),
    this function returns a DataSchema and a dataframe containing TABBLK, QAGE, 
    QSEX, CENRACE, and CENHISP columns. Additionally, it maps the values in 
    these columns to more convenient values. The DataSchema infers the set of
    values in the TABBLK column from the data, but the value sets for the
    remaining columns are fixed. In particular:
    
    - QAGE must take a value in `range(0,116)`
    - QSEX must take a value in `["Male", "Female"]`
    - CENRACE must take on one of the 63 possible values of `CensusRace`
    - CENHISP must take a value in `["HLO", "Not HLO"]`
    """
    for col in ["TABBLKST", "TABBLKCOU", "TABTRACTCE"]:
        assert np.all(
            ppmf[col] == ppmf[col].iloc[0]
        ), "All data must be from the same census tract"

    blocks = np.unique(ppmf["TABBLK"])

    schema = DataSchema()
    schema.add_column(blocks, name="TABBLK", dtype=np.dtype("int32"))
    schema.add_column(range(0, 116), name="QAGE", dtype=np.dtype("int32"))
    schema.add_column(["Male", "Female"], name="QSEX")
    schema.add_column(range(1, 64), name="CENRACE", dtype=np.dtype("int32"))
    schema.add_column(["HLO", "Not HLO"], name="CENHISP")

    data = ppmf.loc[:, ["TABBLK", "QAGE", "QSEX", "CENRACE", "CENHISP"]]

    # Convert sex id into a more user-friendly string
    data["QSEX"] = ppmf["QSEX"].map(lambda sex_id: "Male" if sex_id == 1 else "Female")
    # Convert the hispanic id into either "HLO" or "Not HLO"
    data["CENHISP"] = ppmf["CENHISP"].map(
        lambda hisp_id: "Not HLO" if hisp_id == 1 else "HLO"
    )

    return schema, data

def build_census_queries(schema: DataSchema, use_tract_queries=True, use_PCT12AN=False):
    queries = []
    for (table_cell, query) in census_queries.items():
        race_ids = None if query.races is None else [r.to_id() for r in query.races]
        ages = list(query.ages) if isinstance(query.ages, range) else query.ages

        if query.level == "Tract":
            if not use_tract_queries:
                continue
            if not query.in_2020 and not use_PCT12AN:
                continue
            allowed_value_dict = {
                "QAGE": ages,
                "QSEX": query.sexes,
                "CENRACE": race_ids,
                "CENHISP": query.HLOs,
            }
            allowed_value_dict = {k: v for k, v in allowed_value_dict.items() if v is not None}
            if len(allowed_value_dict) == 0:
                continue
            queries.append(allowed_value_dict)
        else:
            for block in schema.get_column_values("TABBLK"):
                allowed_value_dict = {
                    "TABBLK": [block],
                    "QAGE": ages,
                    "QSEX": query.sexes,
                    "CENRACE": race_ids,
                    "CENHISP": query.HLOs,
                }
                allowed_value_dict = {k: v for k, v in allowed_value_dict.items() if v is not None}
                if len(allowed_value_dict) == 0:
                    continue
                queries.append(allowed_value_dict)

    return queries