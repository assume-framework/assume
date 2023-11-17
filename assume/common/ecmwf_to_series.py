# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

WEATHER_PARAMS_ECMWF = [
    "temp_air",
    "wind_meridional",
    "ghi",
    "wind_zonal",
    "wind_speed",
]


class WeatherInterface:
    def __init__(self, name, database_url):
        self.engine = create_engine(
            database_url, connect_args={"application_name": name}
        )

    def get_param(
        self,
        params: str,
        start: datetime,
        end: datetime,
        area: str = "",
    ):
        if isinstance(params, str):
            params = [params]
        params = [f"avg({p}) as {p}" for p in params]
        selection = ", ".join(params)
        query = f"SELECT time, {selection} FROM ecmwf_eu  WHERE time BETWEEN '{start.isoformat()}' AND '{end.isoformat()}'"
        if area is not None:
            query += f" AND nuts_id LIKE '{area.upper()}%%'"
        query += "group by time"
        with self.engine.begin() as connection:
            return pd.read_sql_query(query, connection, index_col="time")


if __name__ == "__main__":
    nuts_engine = create_engine(
        "postgresql://readonly:readonly@timescale.nowum.fh-aachen.de:5432/nuts"
    )

    weather_engine = create_engine(
        "postgresql://readonly:readonly@timescale.nowum.fh-aachen.de:5432/weather"
    )

    with nuts_engine.begin() as conn:
        plz_nuts = pd.read_sql_query(
            "select code, nuts3, longitude, latitude from plz", conn, index_col="code"
        )
    uri = "postgresql://readonly:readonly@timescale.nowum.fh-aachen.de:5432/weather"
    wi = WeatherInterface("test", uri)
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 3)
    wind_speed = wi.get_param("wind_speed", start, end, "DEA")
    weather_df = wi.get_param(WEATHER_PARAMS_ECMWF, start, end, "DEA")
    weather_df = wi.get_param(WEATHER_PARAMS_ECMWF, start, end, "DE")
