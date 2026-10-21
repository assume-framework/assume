# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Nodal tab: grid line utilisation map, flows and nodal prices (Grafana "Nodal view")."""

import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from assume_dash.components.controls import (
    SIMULATION,
    View,
    aggregate,
    filter_time,
    view_inputs,
)
from assume_dash.components.widgets import (
    empty_figure,
    graph,
    section,
    style_figure,
)
from assume_dash.tabs.base import Tab

_COORDINATE_PAIR = re.compile(
    r"(-?\d+(?:\.\d+)?(?:e-?\d+)?)\s+(-?\d+(?:\.\d+)?(?:e-?\d+)?)"
)


def parse_linestring(wkt: str) -> tuple[list[float], list[float]]:
    """Longitudes and latitudes of an ``SRID=4326;LINESTRING (x y, x y)`` string."""
    pairs = _COORDINATE_PAIR.findall(wkt or "")
    return [float(x) for x, _ in pairs], [float(y) for _, y in pairs]


class NodalTab(Tab):
    id = "nodal"
    title = "Nodal"

    def available(self) -> bool:
        return self.data.has("grid_flows") or self.data.has("lines_geo")

    def layout(self):
        return html.Div(
            [
                section(
                    "Grid information",
                    dcc.RadioItems(
                        {"mean": "Mean utilisation", "max": "Peak utilisation"},
                        "mean",
                        inline=True,
                        id=self.cid("statistic"),
                    ),
                    graph(self.cid("map"), height=550),
                    graph(self.cid("flows")),
                ),
                section(
                    "Nodal market results",
                    html.Label("Market"),
                    dcc.Dropdown(id=self.cid("market"), clearable=False),
                    graph(self.cid("prices")),
                ),
            ]
        )

    def register_callbacks(self, app: Dash) -> None:
        @app.callback(
            Output(self.cid("market"), "options"),
            Output(self.cid("market"), "value"),
            Input(SIMULATION, "value"),
        )
        def market_options(simulation):
            markets = self.data.markets(simulation)
            return markets, (markets[0] if markets else None)

        @app.callback(
            Output(self.cid("map"), "figure"),
            Output(self.cid("flows"), "figure"),
            Input(self.cid("statistic"), "value"),
            *view_inputs(),
        )
        def grid(statistic, *args):
            view = View.from_args(*args)
            flows = filter_time(
                self.data.table("grid_flows", view.simulation), "datetime", view
            )
            if flows.empty:
                return empty_figure("No grid flows available"), empty_figure()
            agg = aggregate(flows, "datetime", ["line"], ["flow"], view.interval)
            flows_fig = style_figure(
                px.line(agg, x="datetime", y="flow", color="line"),
                "Grid flows",
                "MW",
            )
            return self._map_figure(flows, statistic, view.simulation), flows_fig

        @app.callback(
            Output(self.cid("prices"), "figure"),
            Input(self.cid("market"), "value"),
            *view_inputs(),
        )
        def prices(market, *args):
            view = View.from_args(*args)
            df = self.data.table("market_meta", view.simulation)
            if df.empty or market is None or "node" not in df.columns:
                return empty_figure()
            df = filter_time(df[df["market_id"] == market], "product_start", view)
            df = df.dropna(subset=["node"])
            if df.empty:
                return empty_figure("No nodal prices")
            agg = aggregate(df, "product_start", ["node"], ["price"], view.interval)
            return style_figure(
                px.line(agg, x="product_start", y="price", color="node"),
                "Nodal prices",
                "EUR/MWh",
            )

    def _map_figure(self, flows: pd.DataFrame, statistic: str, simulation: str):
        lines = self.data.table("lines_geo", simulation)
        needed = {"line", "wkt_srid_4326", "s_nom"}
        if lines.empty or not needed <= set(lines.columns):
            return empty_figure("No grid geometry available")
        lines = lines.copy()
        lines["capacity"] = lines["s_nom"] * (
            lines["s_max_pu"] if "s_max_pu" in lines else 1
        )
        usage = (
            flows.assign(flow=flows["flow"].abs())
            .groupby("line")["flow"]
            .agg("max" if statistic == "max" else "mean")
        )
        lines["usage"] = lines["line"].map(usage).fillna(0) / lines["capacity"]
        colors = px.colors.sample_colorscale(
            "RdYlGn_r", lines["usage"].clip(0, 1).tolist()
        )
        fig = go.Figure()
        for (_, line), color in zip(lines.iterrows(), colors, strict=True):
            lon, lat = parse_linestring(line["wkt_srid_4326"])
            fig.add_trace(
                go.Scattermap(
                    lon=lon,
                    lat=lat,
                    mode="lines",
                    line={"width": 4, "color": color},
                    name=str(line["line"]),
                    hovertemplate=f"{line['line']}: {line['usage']:.0%}<extra></extra>",
                    showlegend=False,
                )
            )
        # an invisible marker trace carries the colour bar
        fig.add_trace(
            go.Scattermap(
                lon=[None],
                lat=[None],
                mode="markers",
                marker={
                    "colorscale": "RdYlGn_r",
                    "cmin": 0,
                    "cmax": 1,
                    "showscale": True,
                    "colorbar": {"title": "Utilisation", "tickformat": ".0%"},
                },
                showlegend=False,
            )
        )
        all_lon = [x for t in fig.data for x in (t.lon or []) if x is not None]
        all_lat = [y for t in fig.data for y in (t.lat or []) if y is not None]
        fig.update_layout(
            map={
                "style": "open-street-map",
                "center": {
                    "lon": sum(all_lon) / max(len(all_lon), 1),
                    "lat": sum(all_lat) / max(len(all_lat), 1),
                },
                "zoom": 6,
            },
            margin={"l": 0, "r": 0, "t": 30, "b": 0},
            title="Line utilisation",
        )
        return fig
