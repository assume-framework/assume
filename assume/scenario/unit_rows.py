# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""
Transformations the csv loader applies to the unit tables before the rows reach the unit classes.

They are shared by the loader and the scenario verifier, so that the verifier validates the
values a unit is really constructed with. Empty cells are replaced by 0 later on, in
``read_units`` and ``add_units``.
"""

import pandas as pd

MIXED_DEMAND_SIGNS = "min_power and max_power must both be either negative or positive"


def mixed_demand_signs(df: pd.DataFrame) -> pd.Series:
    """Rows of the demand units where one of ``min_power`` and ``max_power`` is negative and the other positive."""
    return ((df["min_power"] < 0) & (df["max_power"] > 0)) | (
        (df["min_power"] > 0) & (df["max_power"] < 0)
    )


def prepare_unit_rows(key: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the sign conventions to the power columns of a unit table.

    Demand power is stored as negative values and the charge powers of storages are negative,
    whatever sign the csv file uses.

    Args:
        key (str): The kind of table, e.g. ``"demand_units"``.
        df (pandas.DataFrame): The table as read from the csv file.

    Returns:
        pandas.DataFrame: A copy of the table with the sign conventions applied.
    """
    columns = {
        "demand_units": ("min_power", "max_power"),
        "storage_units": ("max_power_charge", "min_power_charge"),
    }.get(key, ())
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = -abs(df[column])
    return df
