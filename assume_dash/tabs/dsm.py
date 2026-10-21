# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Demand-side tab: flexible demand units (steel plants, EVs, ...) in six views.

Folds the former standalone ``dsm_dash`` draft into the shared dashboard: the
simulation, time window and aggregation come from the global controls and all
numbers are derived from the simulation output instead of dummy data.
"""

from collections.abc import Callable

import numpy as np
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
    graph_figure,
    row,
    stat_cards,
    style_figure,
    table,
    table_props,
)
from assume_dash.tabs.base import Tab

DEFAULT_SELECTED_UNITS = 8
DEMAND_KINDS = ("demand", "electric_vehicle")
POWER_COLUMNS = ("min_power", "max_power", "max_power_charge", "max_power_discharge")
HIGH_LOAD_SHARE = 0.6
LIMIT_WARNING_SHARE = 0.95

SECTIONS = {
    "overview": "System overview",
    "operations": "Operational analysis",
    "financial": "Financial analysis",
    "carriers": "Energy carriers",
    "market": "Market participation",
    "grid": "Grid",
}


class DsmTab(Tab):
    id = "dsm"
    title = "DSM"

    def available(self) -> bool:
        return any(k in DEMAND_KINDS for k in self.data.available_kinds())

    def kinds(self) -> list[str]:
        return [k for k in self.data.available_kinds() if k in DEMAND_KINDS]

    def layout(self):
        kinds = self.kinds()
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
                html.Label("View"),
                dcc.RadioItems(
                    [{"label": t, "value": k} for k, t in SECTIONS.items()],
                    "overview",
                    inline=True,
                    id=self.cid("section"),
                    className="section-nav",
                ),
                html.Div(id=self.cid("content")),
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
            return ids, ids[:DEFAULT_SELECTED_UNITS]

        @app.callback(
            Output(self.cid("content"), "children"),
            Input(self.cid("section"), "value"),
            kind,
            units,
            *view_inputs(),
        )
        def render(section, kind_, selected, *args):
            view = View.from_args(*args)
            if view.simulation is None or not selected or kind_ is None:
                return graph_figure(empty_figure("Select at least one unit"))
            builders: dict[str, Callable] = {
                "overview": self._overview,
                "operations": self._operations,
                "financial": self._financial,
                "carriers": self._carriers,
                "market": self._market,
                "grid": self._grid,
            }
            return builders[section](view, kind_, selected)

    # ---- data helpers -------------------------------------------------------
    def _meta(self, view: View, kind_: str, selected: list[str]) -> pd.DataFrame:
        meta = self.data.unit_meta(view.simulation, kind_)
        if meta.empty:
            return meta
        return meta[meta["unit"].isin(selected)]

    def _dispatch(self, view: View, selected: list[str]) -> pd.DataFrame:
        df = self.data.table("unit_dispatch", view.simulation)
        if df.empty:
            return df
        df = filter_time(df[df["unit"].isin(selected)], "time", view)
        # demand is dispatched with a negative sign; show consumption as positive
        return df.assign(load=df["power"].abs())

    def _rated_power(self, meta: pd.DataFrame) -> pd.Series:
        for column in ("max_power", "max_power_charge"):
            if column in meta and meta[column].notna().any():
                return meta.set_index("unit")[column].astype(float).abs()
        return pd.Series(dtype=float)

    # ---- sections -----------------------------------------------------------
    def _overview(self, view: View, kind_: str, selected: list[str]):
        meta = self._meta(view, kind_, selected)
        if meta.empty:
            return graph_figure(empty_figure())
        spec = table("dsm-specs")
        spec.rowData, spec.columnDefs = table_props(
            meta.drop(columns=["simulation"], errors="ignore")
        )
        blocks = [html.H3("Unit configuration & technical specifications"), spec]
        power_cols = [c for c in POWER_COLUMNS if c in meta and meta[c].notna().any()]
        if power_cols:
            fig = px.bar(
                meta.melt(id_vars="unit", value_vars=power_cols),
                x="unit",
                y="value",
                color="variable",
                barmode="group",
            )
            blocks.append(graph_figure(style_figure(fig, "Rated power", "MW")))
        if "capacity" in meta and meta["capacity"].notna().any():
            fig = px.bar(meta, x="unit", y="capacity")
            blocks.append(graph_figure(style_figure(fig, "Storage capacity", "MWh")))
        return blocks

    def _operations(self, view: View, kind_: str, selected: list[str]):
        d = self._dispatch(view, selected)
        if d.empty:
            return graph_figure(empty_figure())
        meta = self._meta(view, kind_, selected)
        rated = self._rated_power(meta)
        a = aggregate(d, "time", ["unit"], ["load"], view.interval)

        profile = px.line(a, x="time", y="load", color="unit")
        stacked = px.area(a, x="time", y="load", color="unit")
        blocks = [
            html.H3("Operational analysis"),
            graph_figure(style_figure(profile, "Operating profile", "MW"), 420),
            graph_figure(style_figure(stacked, "Dispatch (stacked)", "MW"), 420),
        ]

        figs = []
        if not rated.empty:
            mean_load = d.groupby("unit")["load"].mean()
            util = (mean_load / rated.reindex(mean_load.index)).dropna()
            util_fig = px.bar(
                util.rename("utilisation").reset_index(), x="unit", y="utilisation"
            )
            util_fig.update_yaxes(tickformat=".0%", hoverformat=".1%")
            figs.append(graph_figure(style_figure(util_fig, "Mean utilisation")))
            cap = d["unit"].map(rated)
            regime = np.select(
                [d["load"] <= 1e-6, d["load"] > HIGH_LOAD_SHARE * cap],
                ["idle", "high"],
                default="partial",
            )
            hist = px.histogram(
                d.assign(regime=regime),
                x="unit",
                color="regime",
                barmode="group",
                category_orders={"regime": ["idle", "partial", "high"]},
            )
            figs.append(
                graph_figure(style_figure(hist, "Operating regimes (time steps)"))
            )
        if figs:
            blocks.append(row(*figs))
        if "soc" in d and d["soc"].notna().any():
            soc = aggregate(
                d.dropna(subset=["soc"]), "time", ["unit"], ["soc"], view.interval
            )
            blocks.append(
                graph_figure(
                    style_figure(
                        px.line(soc, x="time", y="soc", color="unit"), "State of charge"
                    )
                )
            )
        return blocks

    def _financial(self, view: View, kind_: str, selected: list[str]):
        d = self._dispatch(view, selected)
        needed = {"energy_cashflow", "energy_generation_costs"}
        if d.empty or not needed <= set(d.columns):
            return graph_figure(empty_figure("No financial data"))
        d = d.assign(net=d["energy_cashflow"] - d["energy_generation_costs"])
        cards = stat_cards(
            [
                ("Energy cashflow", f"{d['energy_cashflow'].sum():,.2f} EUR"),
                ("Operating costs", f"{d['energy_generation_costs'].sum():,.2f} EUR"),
                ("Net cashflow", f"{d['net'].sum():,.2f} EUR"),
            ]
        )
        cumulative = d.sort_values("time").assign(
            cumulative=lambda x: x.groupby("unit")["net"].cumsum()
        )
        cumulative = aggregate(
            cumulative, "time", ["unit"], ["cumulative"], view.interval, how="last"
        )
        per_unit = (
            d.groupby("unit")[["energy_cashflow", "energy_generation_costs"]]
            .sum()
            .reset_index()
            .melt(id_vars="unit")
        )
        per_unit["variable"] = per_unit["variable"].map(
            {"energy_cashflow": "Cashflow", "energy_generation_costs": "Costs"}
        )
        return [
            html.H3("Financial analysis"),
            html.Div(cards, className="row stat-row"),
            graph_figure(
                style_figure(
                    px.line(cumulative, x="time", y="cumulative", color="unit"),
                    "Cumulative net cashflow",
                    "EUR",
                ),
                420,
            ),
            graph_figure(
                style_figure(
                    px.bar(
                        per_unit, x="unit", y="value", color="variable", barmode="group"
                    ),
                    "Cashflow and costs per unit",
                    "EUR",
                )
            ),
        ]

    def _carriers(self, view: View, kind_: str, selected: list[str]):
        d = self._dispatch(view, selected)
        if d.empty:
            return graph_figure(empty_figure())
        parts = [d.assign(carrier="electricity", energy=d["load"])]
        if "heat" in d and d["heat"].abs().sum() > 0:
            parts.append(d.assign(carrier="heat", energy=d["heat"].abs()))
        long = pd.concat(parts, ignore_index=True)[
            ["time", "unit", "carrier", "energy"]
        ]
        by_time = aggregate(
            long, "time", ["carrier"], ["energy"], view.interval, how="sum"
        )
        mix = long.groupby("carrier", as_index=False)["energy"].sum()
        mix_fig = px.pie(mix, names="carrier", values="energy")
        mix_fig.update_traces(textposition="outside", textinfo="percent+label")
        mix_fig.update_layout(title="Energy mix")

        dominant = long.loc[long.groupby(["time", "unit"])["energy"].idxmax()]
        selection = px.histogram(dominant, x="unit", color="carrier", barmode="group")

        flows = long.groupby(["carrier", "unit"], as_index=False)["energy"].sum()
        nodes = list(flows["carrier"].unique()) + list(flows["unit"].unique())
        idx = {n: i for i, n in enumerate(nodes)}
        sankey = go.Figure(
            go.Sankey(
                node={"label": nodes, "pad": 20},
                link={
                    "source": flows["carrier"].map(idx),
                    "target": flows["unit"].map(idx),
                    "value": flows["energy"],
                },
            )
        )
        sankey.update_layout(title="Energy flows: carrier to unit")
        return [
            html.H3("Energy carrier analysis"),
            graph_figure(
                style_figure(
                    px.area(by_time, x="time", y="energy", color="carrier"),
                    "Energy consumption by carrier",
                    "MWh",
                ),
                420,
            ),
            row(
                graph_figure(mix_fig),
                graph_figure(style_figure(selection, "Dominant carrier (time steps)")),
            ),
            graph_figure(sankey, 420),
        ]

    def _market(self, view: View, kind_: str, selected: list[str]):
        orders = self.data.orders(view.simulation)
        if orders.empty:
            return graph_figure(empty_figure("No bids available"))
        orders = filter_time(
            orders[orders["unit_id"].isin(selected)], "start_time", view
        )
        if orders.empty:
            return graph_figure(empty_figure())
        blocks: list = [html.H3("Market participation analysis")]
        for market_id, og in orders.groupby("market_id"):
            figs = []
            for title, unit_label, column in (
                ("Accepted price", "EUR/MWh", "accepted_price"),
                ("Accepted volume", "MW", "accepted_volume"),
            ):
                fig = px.line(
                    aggregate_bids(og, column, view),
                    x="start_time",
                    y=column,
                    color="unit_id",
                )
                figs.append(
                    graph_figure(
                        style_figure(fig, f"{title} - {market_id}", unit_label)
                    )
                )
            blocks += [html.H3(f"{market_id} market"), row(*figs)]
        return blocks

    def _grid(self, view: View, kind_: str, selected: list[str]):
        d = self._dispatch(view, selected)
        meta = self._meta(view, kind_, selected)
        rated = self._rated_power(meta)
        if d.empty:
            return graph_figure(empty_figure())
        total = d.groupby("time", as_index=False)["load"].sum().sort_values("time")
        limit = float(rated.sum()) if not rated.empty else None
        peak = total["load"].max()
        cards = [("Peak load", f"{peak:,.2f} MW")]
        if limit:
            cards += [
                ("Peak vs. rated power", f"{peak / limit:.0%}"),
                (
                    f"Time steps above {LIMIT_WARNING_SHARE:.0%} of rated power",
                    str(int((total["load"] > limit * LIMIT_WARNING_SHARE).sum())),
                ),
            ]
        orders = self.data.orders(view.simulation)
        if not orders.empty:
            orders = filter_time(
                orders[orders["unit_id"].isin(selected)], "start_time", view
            )
            for market_id, og in orders.groupby("market_id"):
                offered = og["volume"].abs().sum()
                if offered > 0:
                    share = og["accepted_volume"].abs().sum() / offered
                    cards.append((f"Accepted share - {market_id}", f"{share:.2%}"))

        profile = go.Figure()
        agg = aggregate(total, "time", [], ["load"], view.interval)
        profile.add_scatter(x=agg["time"], y=agg["load"], mode="lines", name="Load")
        if limit:
            profile.add_hline(y=limit, line_dash="dash", annotation_text="Rated power")
        duration = px.line(
            total.sort_values("load", ascending=False).reset_index(drop=True),
            y="load",
        )
        duration.update_xaxes(title="Time steps (sorted)")
        return [
            html.H3("Grid-related analysis"),
            html.Div(stat_cards(cards), className="row stat-row"),
            graph_figure(style_figure(profile, "Grid load profile", "MW"), 420),
            graph_figure(style_figure(duration, "Load duration curve", "MW")),
        ]
