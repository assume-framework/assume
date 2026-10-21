# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Small reusable layout and figure building blocks."""

import dash_ag_grid as dag
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html


def graph(component_id: str, height: int = 380, **style) -> dcc.Loading:
    return dcc.Loading(
        dcc.Graph(id=component_id, style={"height": f"{height}px", **style}),
        type="dot",
    )


def graph_figure(fig: go.Figure, height: int = 380) -> dcc.Graph:
    """Graph for figures built inside a callback (dynamic layouts)."""
    return dcc.Graph(figure=fig, style={"height": f"{height}px"})


def row(*children) -> html.Div:
    return html.Div([html.Div(c) for c in children], className="row")


def section(title: str, *children) -> html.Div:
    return html.Div([html.H3(title), *children], className="section")


def empty_figure(message: str = "No data for the current selection") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        annotations=[
            {"text": message, "showarrow": False, "font": {"size": 16, "color": "grey"}}
        ],
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def style_figure(fig: go.Figure, title: str, yaxis: str | None = None) -> go.Figure:
    fig.update_layout(
        title=title,
        yaxis_title=yaxis,
        legend_title_text="",
        margin={"l": 60, "r": 20, "t": 50, "b": 40},
        hovermode="x unified",
    )
    return fig


def table(component_id: str, df: pd.DataFrame | None = None) -> dag.AgGrid:
    df = pd.DataFrame() if df is None else df
    return dag.AgGrid(
        id=component_id,
        rowData=df.to_dict("records"),
        columnDefs=[{"field": str(c), "filter": True} for c in df.columns],
        defaultColDef={"sortable": True, "resizable": True},
        dashGridOptions={"domLayout": "autoHeight"},
        className="ag-theme-alpine",
    )


def table_props(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """``(rowData, columnDefs)`` for callbacks updating a :func:`table`."""
    return (
        df.round(2).to_dict("records"),
        [{"field": str(c), "filter": True} for c in df.columns],
    )


def stat_cards(items: list[tuple[str, str]]) -> list[html.Div]:
    return [
        html.Div(
            [
                html.Div(label, className="label"),
                html.Div(value, className="value"),
            ],
            className="stat-card",
        )
        for label, value in items
    ]
