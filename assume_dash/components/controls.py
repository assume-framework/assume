# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Global controls shared by every tab and helpers to apply them to frames."""

from dataclasses import dataclass

import pandas as pd
from dash import Input, dcc, html

SIMULATION = "global-simulation"
INTERVAL = "global-interval"
DATES = "global-dates"

INTERVALS = {
    "raw": "Raw",
    "1h": "Hourly mean",
    "4h": "4-hourly mean",
    "1D": "Daily mean",
    "7D": "Weekly mean",
}


@dataclass(frozen=True)
class View:
    """The currently selected simulation, aggregation interval and time window."""

    simulation: str | None
    interval: str = "raw"
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None

    @classmethod
    def from_args(cls, simulation, interval, start, end) -> "View":
        return cls(
            simulation,
            interval or "raw",
            pd.Timestamp(start) if start else None,
            # the end date is inclusive up to the end of that day
            pd.Timestamp(end) + pd.Timedelta(days=1) if end else None,
        )


def view_inputs() -> list[Input]:
    """Callback inputs; unpack the callback arguments with ``View.from_args``."""
    return [
        Input(SIMULATION, "value"),
        Input(INTERVAL, "value"),
        Input(DATES, "start_date"),
        Input(DATES, "end_date"),
    ]


def global_controls(simulations: list[str]) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Simulation"),
                    dcc.Dropdown(
                        simulations,
                        simulations[0] if simulations else None,
                        clearable=False,
                        id=SIMULATION,
                    ),
                ],
                className="control control-wide",
            ),
            html.Div(
                [
                    html.Label("Time window"),
                    dcc.DatePickerRange(id=DATES, display_format="YYYY-MM-DD"),
                ],
                className="control",
            ),
            html.Div(
                [
                    html.Label("Aggregation"),
                    dcc.Dropdown(
                        [{"label": v, "value": k} for k, v in INTERVALS.items()],
                        "raw",
                        clearable=False,
                        id=INTERVAL,
                    ),
                ],
                className="control",
            ),
        ],
        className="controls",
    )


def filter_time(df: pd.DataFrame, column: str, view: View) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df
    mask = pd.Series(True, index=df.index)
    if view.start is not None:
        mask &= df[column] >= view.start
    if view.end is not None:
        mask &= df[column] < view.end
    return df[mask].sort_values(column, kind="stable")


def aggregate(
    df: pd.DataFrame,
    time_col: str,
    by: list[str],
    values: list[str],
    interval: str,
    how: str = "mean",
) -> pd.DataFrame:
    """Aggregate ``values`` per ``by`` group and time bucket (replaces ``$__interval``)."""
    if df.empty:
        return df
    if interval == "raw":
        return df[[time_col, *by, *values]].sort_values(time_col, kind="stable")
    grouper = [pd.Grouper(key=time_col, freq=interval), *by]
    return df.groupby(grouper, observed=True)[values].agg(how).reset_index()
