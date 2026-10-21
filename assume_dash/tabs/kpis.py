# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""KPI tab: key indicators across simulations and installed capacities."""

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, html

from assume_dash.components.controls import SIMULATION
from assume_dash.components.widgets import (
    empty_figure,
    graph,
    row,
    section,
    style_figure,
    table,
    table_props,
)
from assume_dash.tabs.base import Tab


class KpiTab(Tab):
    id = "kpi"
    title = "KPIs"

    def available(self) -> bool:
        return self.data.has("kpis") or self.data.has("power_plant_meta")

    def layout(self):
        return html.Div(
            [
                section(
                    "Key performance indicators",
                    row(graph(self.cid("cost-volume")), graph(self.cid("capacity"))),
                    table(self.cid("table")),
                ),
            ]
        )

    def register_callbacks(self, app: Dash) -> None:
        @app.callback(
            Output(self.cid("cost-volume"), "figure"),
            Output(self.cid("table"), "rowData"),
            Output(self.cid("table"), "columnDefs"),
            Input(SIMULATION, "value"),
        )
        def kpis(_simulation):
            wide = self._kpi_table()
            if wide.empty:
                return empty_figure("No KPIs available"), [], []
            melt = wide.melt(
                id_vars="simulation",
                value_vars=[c for c in ("total_cost", "total_volume") if c in wide],
            )
            fig = style_figure(
                px.bar(
                    melt,
                    x="simulation",
                    y="value",
                    color="variable",
                    barmode="group",
                ),
                "Total cost and volume",
            )
            fig.update_layout(hovermode="closest")
            return (fig, *table_props(wide))

        @app.callback(
            Output(self.cid("capacity"), "figure"), Input(SIMULATION, "value")
        )
        def capacity(simulation):
            meta = self.data.unit_meta(simulation, "generation")
            if meta.empty or "technology" not in meta:
                return empty_figure()
            caps = meta.groupby("technology", as_index=False)["max_power"].sum()
            fig = px.pie(caps, names="technology", values="max_power")
            fig.update_traces(
                hovertemplate="%{label}<br>%{value:,.2f} MW (%{percent:.2%})"
                "<extra></extra>",
                texttemplate="%{percent:.2%}",
                textposition="outside",
            )
            fig.update_layout(title="Installed generation capacity")
            return fig

    def _kpi_table(self) -> pd.DataFrame:
        kpis = self.data.table("kpis")
        if kpis.empty:
            return kpis
        # KPIs are stored per market; averages are averaged, totals are summed
        grouped = kpis.groupby(["simulation", "variable"])["value"]
        values = grouped.mean().where(
            grouped.mean().index.get_level_values("variable").str.startswith("avg"),
            grouped.sum(),
        )
        return values.unstack("variable").reset_index().rename_axis(columns=None)
