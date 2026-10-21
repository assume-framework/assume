# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Market tab: prices, merit order, market result and bids (Grafana "ASSUME")."""

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
    row,
    section,
    style_figure,
)
from assume_dash.tabs.base import Tab


class MarketTab(Tab):
    id = "market"
    title = "Market"

    def available(self) -> bool:
        return self.data.has("market_meta")

    def layout(self):
        return html.Div(
            [
                html.Label("Market"),
                dcc.Dropdown(id=self.cid("market"), clearable=False),
                section(
                    "Market summary",
                    graph(self.cid("summary"), height=600),
                    row(
                        graph(self.cid("price-duration")),
                        graph(self.cid("by-technology")),
                    ),
                ),
                section(
                    "Merit order",
                    html.Label("Product start"),
                    dcc.Dropdown(id=self.cid("product"), clearable=False),
                    graph(self.cid("merit-order"), height=450),
                ),
                section(
                    "Accepted bids",
                    graph(self.cid("accepted-price")),
                    graph(self.cid("accepted-volume")),
                ),
            ]
        )

    def register_callbacks(self, app: Dash) -> None:
        market = Input(self.cid("market"), "value")

        @app.callback(
            Output(self.cid("market"), "options"),
            Output(self.cid("market"), "value"),
            Input(SIMULATION, "value"),
        )
        def market_options(simulation):
            markets = self.data.markets(simulation)
            return markets, (markets[0] if markets else None)

        @app.callback(
            Output(self.cid("product"), "options"),
            Output(self.cid("product"), "value"),
            market,
            *view_inputs(),
        )
        def product_options(market_id, *args):
            view = View.from_args(*args)
            orders = self._orders(view, market_id)
            starts = sorted(orders["start_time"].unique()) if not orders.empty else []
            options = [
                {"label": str(pd.Timestamp(t)), "value": str(pd.Timestamp(t))}
                for t in starts
            ]
            return options, (options[0]["value"] if options else None)

        @app.callback(
            Output(self.cid("summary"), "figure"),
            Output(self.cid("price-duration"), "figure"),
            market,
            *view_inputs(),
        )
        def summary(market_id, *args):
            view = View.from_args(*args)
            df = self.data.table("market_meta", view.simulation)
            if df.empty or market_id is None:
                return empty_figure(), empty_figure()
            df = df[df["market_id"] == market_id].copy()
            df["node"] = df["node"].fillna("system") if "node" in df else "system"
            df = filter_time(df, "product_start", view)
            return self._summary_figure(df, view), self._price_duration(df)

        @app.callback(
            Output(self.cid("by-technology"), "figure"),
            Output(self.cid("accepted-price"), "figure"),
            Output(self.cid("accepted-volume"), "figure"),
            market,
            *view_inputs(),
        )
        def results(market_id, *args):
            view = View.from_args(*args)
            orders = self._orders(view, market_id)
            if orders.empty:
                return empty_figure(), empty_figure(), empty_figure()
            by_unit = aggregate(
                orders,
                "start_time",
                ["unit_id", "technology"],
                ["accepted_volume"],
                "1h" if view.interval == "raw" else view.interval,
            )
            tech = by_unit.groupby(["start_time", "technology"], as_index=False)[
                "accepted_volume"
            ].sum()
            tech_fig = style_figure(
                px.area(tech, x="start_time", y="accepted_volume", color="technology"),
                "Market result by technology",
                "MW",
            )
            price = style_figure(
                px.line(
                    orders.sort_values("start_time"),
                    x="start_time",
                    y="accepted_price",
                    color="unit_id",
                ),
                "Accepted bid price",
                "EUR/MWh",
            )
            volume = style_figure(
                px.line(
                    orders.sort_values("start_time"),
                    x="start_time",
                    y="accepted_volume",
                    color="unit_id",
                ),
                "Accepted bid volume",
                "MW",
            )
            return tech_fig, price, volume

        @app.callback(
            Output(self.cid("merit-order"), "figure"),
            Input(self.cid("product"), "value"),
            market,
            Input(SIMULATION, "value"),
        )
        def merit_order(product, market_id, simulation):
            if product is None or market_id is None:
                return empty_figure()
            orders = self.data.orders(simulation)
            orders = orders[
                (orders["market_id"] == market_id)
                & (orders["start_time"] == pd.Timestamp(product))
            ]
            return self._merit_order_figure(orders, product)

    def _orders(self, view: View, market_id: str | None) -> pd.DataFrame:
        if market_id is None or view.simulation is None:
            return pd.DataFrame()
        orders = self.data.orders_with_technology(view.simulation)
        if orders.empty:
            return orders
        orders = orders[orders["market_id"] == market_id]
        return filter_time(orders, "start_time", view)

    @staticmethod
    def _summary_figure(df: pd.DataFrame, view: View) -> go.Figure:
        if df.empty:
            return empty_figure()
        agg = aggregate(
            df,
            "product_start",
            ["node"],
            ["price", "demand_volume", "supply_volume"],
            view.interval,
        )
        long = agg.melt(
            id_vars=["product_start", "node"],
            var_name="quantity",
            value_name="value",
        )
        fig = px.line(
            long,
            x="product_start",
            y="value",
            color="node",
            facet_row="quantity",
            category_orders={"quantity": ["price", "demand_volume", "supply_volume"]},
        )
        fig.update_yaxes(matches=None)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_layout(title="Market summary", hovermode="x unified")
        return fig

    @staticmethod
    def _price_duration(df: pd.DataFrame) -> go.Figure:
        if df.empty:
            return empty_figure()
        prices = (
            df.groupby("product_start")["price"].mean().sort_values(ascending=False)
        )
        share = (pd.RangeIndex(1, len(prices) + 1) / len(prices)).to_numpy()
        fig = px.line(x=share, y=prices.to_numpy())
        fig.update_xaxes(tickformat=".0%", hoverformat=".2%", title="Share of time")
        fig.update_yaxes(title="Price in EUR/MWh")
        fig.update_layout(title="Price duration curve")
        return fig

    @staticmethod
    def _merit_order_figure(orders: pd.DataFrame, product: str) -> go.Figure:
        if orders.empty:
            return empty_figure()
        supply = orders[orders["volume"] > 0].sort_values(["price", "unit_id"])
        demand = orders[orders["volume"] < 0].sort_values(
            ["price", "unit_id"], ascending=[False, True]
        )
        fig = go.Figure()
        for name, side, color in (
            ("Supply", supply, "royalblue"),
            ("Demand", demand, "firebrick"),
        ):
            if side.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=side["volume"].abs().cumsum(),
                    y=side["price"],
                    name=name,
                    mode="lines+markers",
                    line={"shape": "hv", "color": color},
                    customdata=side[["unit_id", "accepted_volume"]],
                    hovertemplate="%{customdata[0]}<br>price %{y:.2f}<br>"
                    "cum. volume %{x:.2f}<br>accepted %{customdata[1]:.2f}"
                    "<extra></extra>",
                )
            )
        fig.update_layout(
            title=f"Merit order at {product}",
            xaxis_title="Cumulative volume in MW",
            yaxis_title="Price in EUR/MWh",
        )
        return fig
