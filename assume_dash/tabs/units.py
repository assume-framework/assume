# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tab: dispatch, bids, specification and financials per unit group."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from assume_dash.components.controls import (
    SIMULATION,
    View,
    aggregate,
    aggregate_bids,
    filter_time,
    view_inputs,
)
from assume_dash.components.widgets import (
    empty_figure,
    graph,
    graph_figure,
    row,
    section,
    stat_cards,
    style_figure,
    table,
    table_props,
)
from assume_dash.tabs.base import Tab


class UnitsTab(Tab):
    id = "units"
    title = "Units"

    def available(self) -> bool:
        return bool(self.data.available_kinds())

    def layout(self):
        kinds = self.data.available_kinds()
        return html.Div(
            [
                html.Div(
                    row(
                        [
                            html.Label("Unit type"),
                            dcc.RadioItems(
                                kinds,
                                kinds[0] if kinds else None,
                                inline=True,
                                id=self.cid("kind"),
                            ),
                        ],
                        [
                            html.Label("Units"),
                            dcc.Checklist(
                                id=self.cid("units"),
                                inline=True,
                                className="chip-list",
                            ),
                        ],
                    ),
                    className="floating-select",
                ),
                section(
                    "Specification",
                    table(self.cid("specs")),
                ),
                section(
                    "Dispatch",
                    html.Div(id=self.cid("dispatch")),
                    html.Div(
                        id=self.cid("soc-container"), children=graph(self.cid("soc"))
                    ),
                ),
                section("Bids", html.Div(id=self.cid("bids"))),
                section(
                    "Financials",
                    html.Div(
                        id=self.cid("cards"),
                        className="row stat-row",
                    ),
                    graph(self.cid("financial")),
                ),
            ]
        )

    def register_callbacks(self, app: Dash) -> None:
        kind = Input(self.cid("kind"), "value")
        units = Input(self.cid("units"), "value")

        @app.callback(
            Output(self.cid("units"), "options"),
            Output(self.cid("units"), "value"),
            Input(SIMULATION, "value"),
            kind,
        )
        def unit_options(simulation, kind_):
            ids = self.data.unit_ids(simulation, kind_) if kind_ else []
            return ids, ids

        @app.callback(
            Output(self.cid("specs"), "rowData"),
            Output(self.cid("specs"), "columnDefs"),
            Input(SIMULATION, "value"),
            kind,
            units,
        )
        def specs(simulation, kind_, selected):
            meta = self.data.unit_meta(simulation, kind_) if kind_ else pd.DataFrame()
            if meta.empty:
                return [], []
            meta = meta[meta["unit"].isin(selected or [])].drop(
                columns=["simulation"], errors="ignore"
            )
            return table_props(meta)

        @app.callback(
            Output(self.cid("dispatch"), "children"),
            Output(self.cid("soc"), "figure"),
            Output(self.cid("soc-container"), "style"),
            units,
            *view_inputs(),
        )
        def dispatch(selected, *args):
            view = View.from_args(*args)
            actual = self._unit_dispatch(view, selected)
            market = self._market_dispatch(view, selected)
            figs = []
            if not market.empty:
                m = aggregate(
                    market,
                    "datetime",
                    ["unit_id", "market_id"],
                    ["power"],
                    view.interval,
                )
                for market_id, mg in m.groupby("market_id"):
                    fig = go.Figure()
                    for unit, g in mg.groupby("unit_id"):
                        fig.add_scatter(
                            x=g["datetime"], y=g["power"], mode="lines", name=unit
                        )
                    figs.append(
                        style_figure(fig, f"Dispatch - {market_id} market", "MW")
                    )
            if not actual.empty:
                a = aggregate(actual, "time", ["unit"], ["power"], view.interval)
                fig = go.Figure()
                for unit, g in a.groupby("unit"):
                    fig.add_scatter(x=g["time"], y=g["power"], mode="lines", name=unit)
                figs.append(style_figure(fig, "Dispatch - actual (unit level)", "MW"))
            if not figs:
                return (
                    [graph_figure(empty_figure())],
                    empty_figure(),
                    {"display": "none"},
                )

            soc_fig, soc_style = empty_figure(), {"display": "none"}
            if not actual.empty and "soc" in actual and actual["soc"].notna().any():
                s = aggregate(
                    actual.dropna(subset=["soc"]),
                    "time",
                    ["unit"],
                    ["soc"],
                    view.interval,
                )
                soc_fig = style_figure(
                    px.line(s, x="time", y="soc", color="unit"), "State of charge"
                )
                soc_style = {}
            return [graph_figure(f, 420) for f in figs], soc_fig, soc_style

        @app.callback(
            Output(self.cid("bids"), "children"),
            units,
            *view_inputs(),
        )
        def bids(selected, *args):
            view = View.from_args(*args)
            if view.simulation is None or not selected:
                return graph_figure(empty_figure())
            orders = self.data.orders(view.simulation)
            orders = filter_time(
                orders[orders["unit_id"].isin(selected)], "start_time", view
            )
            if orders.empty:
                return graph_figure(empty_figure())
            blocks = []
            for market_id, og in orders.groupby("market_id"):
                figs = []
                for title, unit_label, column in (
                    ("Bid price", "EUR/MWh", "price"),
                    ("Accepted price", "EUR/MWh", "accepted_price"),
                    ("Bid volume", "MW", "volume"),
                    ("Accepted volume", "MW", "accepted_volume"),
                ):
                    agg = aggregate_bids(og, column, view)
                    fig = px.line(agg, x="start_time", y=column, color="unit_id")
                    figs.append(
                        graph_figure(
                            style_figure(fig, f"{title} - {market_id}", unit_label)
                        )
                    )
                blocks += [row(figs[0], figs[1]), row(figs[2], figs[3])]
            return blocks

        @app.callback(
            Output(self.cid("cards"), "children"),
            Output(self.cid("financial"), "figure"),
            kind,
            units,
            *view_inputs(),
        )
        def financials(kind_, selected, *args):
            view = View.from_args(*args)
            df = self._unit_dispatch(view, selected)
            needed = {"energy_cashflow", "energy_generation_costs"}
            if df.empty or not needed <= set(df.columns):
                return [], empty_figure("No financial data")
            df = df.assign(profit=df["energy_cashflow"] - df["energy_generation_costs"])
            totals = df[["energy_cashflow", "energy_generation_costs", "profit"]].sum()
            cards = [
                ("Cashflow", f"{totals['energy_cashflow']:,.0f} EUR"),
                ("Production costs", f"{totals['energy_generation_costs']:,.0f} EUR"),
                ("Profit", f"{totals['profit']:,.0f} EUR"),
            ]
            volume = df["power"].abs().sum()
            if volume > 0:
                cards.append(
                    (
                        "Weighted average price",
                        f"{abs(df['energy_cashflow'].sum()) / volume:,.2f} EUR/MWh",
                    )
                )
            cards.append(("Dispatched volume", f"{df['power'].sum():,.1f} MWh"))
            summed = df.groupby("time", as_index=False)[
                ["energy_cashflow", "energy_generation_costs", "profit"]
            ].sum()
            summed = aggregate(
                summed,
                "time",
                [],
                ["energy_cashflow", "energy_generation_costs", "profit"],
                view.interval,
                how="sum",
            )
            long = summed.melt(id_vars="time", var_name="series")
            long["series"] = long["series"].map(
                {
                    "energy_cashflow": "Cashflow",
                    "energy_generation_costs": "Production costs",
                    "profit": "Profit",
                }
            )
            fig = style_figure(
                px.line(long, x="time", y="value", color="series"),
                "Energy-related cashflow of the selected units",
                "EUR",
            )
            return stat_cards(cards), fig

    def _unit_dispatch(self, view: View, selected: list[str] | None) -> pd.DataFrame:
        if view.simulation is None or not selected:
            return pd.DataFrame()
        df = self.data.table("unit_dispatch", view.simulation)
        if df.empty:
            return df
        return filter_time(df[df["unit"].isin(selected)], "time", view)

    def _market_dispatch(self, view: View, selected: list[str] | None) -> pd.DataFrame:
        if view.simulation is None or not selected:
            return pd.DataFrame()
        df = self.data.table("market_dispatch", view.simulation)
        if df.empty:
            return df
        return filter_time(df[df["unit_id"].isin(selected)], "datetime", view)
