# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comparison tab: differences between two simulations (Grafana "Compare scenarios")."""

import pandas as pd
import plotly.express as px
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
    row,
    section,
    style_figure,
)
from assume_dash.tabs.base import Tab


def _bucket(df, time_col, by, value, interval) -> pd.DataFrame:
    # raw data may hold several rows per step (e.g. nodes); merging needs unique keys
    if interval == "raw":
        return df.groupby([time_col, *by], as_index=False)[value].mean()
    return aggregate(df, time_col, by, [value], interval)


def difference(
    base: pd.DataFrame,
    other: pd.DataFrame,
    time_col: str,
    by: list[str],
    value: str,
    interval: str,
) -> pd.DataFrame:
    """``other - base`` per time bucket and group, for buckets present in both."""
    if base.empty or other.empty:
        return pd.DataFrame()
    a = _bucket(base, time_col, by, value, interval)
    b = _bucket(other, time_col, by, value, interval)
    merged = a.merge(b, on=[time_col, *by], suffixes=("_base", "_other"))
    merged["difference"] = merged[f"{value}_other"] - merged[f"{value}_base"]
    return merged


class ComparisonTab(Tab):
    id = "comparison"
    title = "Scenario comparison"

    def available(self) -> bool:
        return len(self.data.simulations()) > 1

    def layout(self):
        return html.Div(
            [
                html.P(
                    "Base scenario is the simulation selected above; "
                    "differences are shown as comparison minus base."
                ),
                row(
                    [
                        html.Label("Compare with"),
                        dcc.Dropdown(id=self.cid("other"), clearable=False),
                    ],
                    [
                        html.Label("Market"),
                        dcc.Dropdown(id=self.cid("market"), clearable=False),
                    ],
                ),
                section(
                    "Market",
                    graph(self.cid("price")),
                    row(graph(self.cid("price-diff")), graph(self.cid("volume-diff"))),
                    row(
                        graph(self.cid("bid-price-diff")),
                        graph(self.cid("bid-volume-diff")),
                    ),
                ),
                section("Key performance indicators", graph(self.cid("kpis"))),
                section(
                    "Unit dispatch",
                    dcc.Dropdown(id=self.cid("units"), multi=True),
                    graph(self.cid("dispatch-diff")),
                ),
            ]
        )

    def register_callbacks(self, app: Dash) -> None:
        other = Input(self.cid("other"), "value")
        market = Input(self.cid("market"), "value")

        @app.callback(
            Output(self.cid("other"), "options"),
            Output(self.cid("other"), "value"),
            Output(self.cid("market"), "options"),
            Output(self.cid("market"), "value"),
            Output(self.cid("units"), "options"),
            Output(self.cid("units"), "value"),
            Input(SIMULATION, "value"),
        )
        def options(simulation):
            others = [s for s in self.data.simulations() if s != simulation]
            markets = self.data.markets(simulation)
            units = self.data.table("unit_dispatch", simulation)
            unit_ids = sorted(units["unit"].unique()) if not units.empty else []
            return (
                others,
                others[0] if others else None,
                markets,
                markets[0] if markets else None,
                unit_ids,
                unit_ids[:5],
            )

        @app.callback(
            Output(self.cid("price"), "figure"),
            Output(self.cid("price-diff"), "figure"),
            Output(self.cid("volume-diff"), "figure"),
            other,
            market,
            *view_inputs(),
        )
        def market_comparison(other_sim, market_id, *args):
            view = View.from_args(*args)
            base = self._market_meta(view.simulation, market_id, view)
            comp = self._market_meta(other_sim, market_id, view)
            price = difference(base, comp, "product_start", [], "price", view.interval)
            if price.empty:
                return (empty_figure(),) * 3
            fig = px.line(
                price.melt(
                    id_vars="product_start",
                    value_vars=["price_base", "price_other", "difference"],
                    var_name="series",
                ).replace(
                    {
                        "price_base": view.simulation,
                        "price_other": other_sim,
                        "difference": "Difference",
                    }
                ),
                x="product_start",
                y="value",
                color="series",
            )
            style_figure(fig, "Prices", "EUR/MWh")
            diffs = [
                style_figure(
                    px.line(
                        difference(base, comp, "product_start", [], col, view.interval),
                        x="product_start",
                        y="difference",
                    ),
                    title,
                )
                for col, title in (
                    ("price", "Price difference"),
                    ("supply_volume", "Supply volume difference"),
                )
            ]
            return fig, *diffs

        @app.callback(
            Output(self.cid("bid-price-diff"), "figure"),
            Output(self.cid("bid-volume-diff"), "figure"),
            other,
            market,
            *view_inputs(),
        )
        def bid_comparison(other_sim, market_id, *args):
            view = View.from_args(*args)
            figs = []
            frames = [
                self._orders(s, market_id, view) for s in (view.simulation, other_sim)
            ]
            for value, title in (
                ("accepted_price", "Accepted bid price difference"),
                ("accepted_volume", "Accepted bid volume difference"),
            ):
                diff = difference(
                    *frames, "start_time", ["bid_label"], value, view.interval
                )
                figs.append(
                    empty_figure()
                    if diff.empty
                    else style_figure(
                        px.line(
                            diff, x="start_time", y="difference", color="bid_label"
                        ),
                        title,
                    )
                )
            return figs

        @app.callback(
            Output(self.cid("kpis"), "figure"), Input(SIMULATION, "value"), other
        )
        def kpis(simulation, other_sim):
            df = self.data.table("kpis")
            df = (
                df[df["simulation"].isin([simulation, other_sim])]
                if not df.empty
                else df
            )
            if df.empty:
                return empty_figure("No KPIs available")
            df = df.groupby(["simulation", "variable"], as_index=False)["value"].mean()
            fig = px.bar(
                df, x="variable", y="value", color="simulation", barmode="group"
            )
            fig.update_layout(title="KPIs (mean over markets)")
            return fig

        @app.callback(
            Output(self.cid("dispatch-diff"), "figure"),
            other,
            Input(self.cid("units"), "value"),
            *view_inputs(),
        )
        def dispatch(other_sim, selected, *args):
            view = View.from_args(*args)
            frames = []
            for sim in (view.simulation, other_sim):
                df = self.data.table("unit_dispatch", sim)
                df = df[df["unit"].isin(selected or [])] if not df.empty else df
                frames.append(filter_time(df, "time", view))
            diff = difference(*frames, "time", ["unit"], "power", view.interval)
            if diff.empty:
                return empty_figure()
            return style_figure(
                px.line(diff, x="time", y="difference", color="unit"),
                "Unit dispatch difference",
                "MW",
            )

    def _market_meta(self, simulation, market_id, view: View) -> pd.DataFrame:
        df = self.data.table("market_meta", simulation)
        if df.empty or market_id is None:
            return pd.DataFrame()
        return filter_time(df[df["market_id"] == market_id], "product_start", view)

    def _orders(self, simulation, market_id, view: View) -> pd.DataFrame:
        df = self.data.orders(simulation)
        if df.empty or market_id is None:
            return pd.DataFrame()
        return filter_time(df[df["market_id"] == market_id], "start_time", view)
