# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Draft of a modular ASSUME dashboard for demand-side units.

Requirements: dash, dash-ag-grid, plotly, pandas, numpy

Each tab is built from a ``build_*_tab`` (layout) function and registers its own
callbacks. Data access is isolated in ``load_data``; real simulation outputs are
used where available, otherwise dummy data is generated. To support supply-side
units later, extend ``load_data`` and the unit-type filter.
"""

from pathlib import Path

import dash_ag_grid as dag
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html

P_OUT = Path("examples/notebooks/outputs/tutorial_10_steel_plant_Day_Ahead")
P_IN = Path("examples/notebooks/inputs/tutorial_10")

RNG = np.random.default_rng(42)
TIME = pd.date_range("2025-01-01", periods=24 * 7, freq="h")
CARRIERS = ["electricity", "natural_gas", "hydrogen"]
MARKETS = ["EOM", "Redispatch", "Reserve (aFRR)"]


# --------------------------------------------------------------------------- #
# Data layer (dummy data, replace with real loaders)
# --------------------------------------------------------------------------- #
def _hourly_pattern(base: float, amp: float) -> np.ndarray:
    h = np.arange(len(TIME))
    return base + amp * np.sin(2 * np.pi * h / 24) + RNG.normal(0, amp * 0.2, len(h))


def load_data() -> dict[str, pd.DataFrame]:
    """Return all frames used by the dashboard, keyed by name."""
    units = pd.DataFrame(
        {
            "unit": ["steel_plant_1", "steel_plant_2", "data_center_1"],
            "unit_type": "demand",
            "technology": ["steel_plant", "steel_plant", "data_center"],
            "max_power_MW": [100, 60, 30],
            "min_power_MW": [20, 10, 5],
            "ramp_up_MW_h": [40, 25, 30],
            "ramp_down_MW_h": [40, 25, 30],
            "min_operating_time_h": [4, 4, 0],
            "efficiency": [0.85, 0.82, 0.95],
            "grid_connection_MW": [110, 70, 35],
        }
    )
    if (P_IN / "demand_units.csv").exists():
        units = pd.read_csv(P_IN / "demand_units.csv")

    rows = []
    for _, u in units.iterrows():
        power = np.clip(
            _hourly_pattern(u.max_power_MW * 0.6, u.max_power_MW * 0.25),
            u.min_power_MW,
            u.max_power_MW,
        )
        price = _hourly_pattern(60, 25)
        baseline = np.full(len(TIME), u.max_power_MW * 0.65)
        flex = baseline - power
        rows.append(
            pd.DataFrame(
                {
                    "time": TIME,
                    "unit": u.unit,
                    "power_MW": power,
                    "baseline_MW": baseline,
                    "flex_MW": flex,
                    "price": price,
                    "grid_import_MW": power * 0.8 + 0.05 * u.max_power_MW,
                    "grid_export_MW": np.clip(-flex, 0, None) * 0.2,
                    "grid_limit_MW": u.grid_connection_MW,
                    "regime": np.where(power > u.max_power_MW * 0.6, "high", "low"),
                    "on": (power > u.min_power_MW * 1.05).astype(int),
                }
            )
        )
    ops = pd.concat(rows, ignore_index=True)

    # financials
    ops["energy_cost"] = ops.power_MW * ops.price
    ops["baseline_cost"] = ops.baseline_MW * ops.price
    ops["savings"] = ops.baseline_cost - ops.energy_cost
    ops["revenue"] = np.clip(ops.flex_MW, 0, None) * ops.price * 0.3
    ops["fuel_cost"] = ops.power_MW * 8
    ops["maintenance_cost"] = ops.power_MW * 1.5
    ops["net_cash_flow"] = (
        ops.revenue + ops.savings - ops.fuel_cost - ops.maintenance_cost
    )

    # energy carriers
    carriers = []
    for u in units.unit:
        d = ops[ops.unit == u]
        share = {"electricity": 0.6, "natural_gas": 0.3, "hydrogen": 0.1}
        for c in CARRIERS:
            carriers.append(
                pd.DataFrame(
                    {
                        "time": d.time.values,
                        "unit": u,
                        "carrier": c,
                        "energy_MWh": d.power_MW.values
                        * share[c]
                        * (1 + RNG.normal(0, 0.05, len(d))),
                    }
                )
            )
    carriers = pd.concat(carriers, ignore_index=True)

    # market bids
    bids = []
    for m in MARKETS:
        for u in units.unit:
            vol = np.abs(RNG.normal(20, 6, len(TIME)))
            acc = vol * RNG.uniform(0.3, 1.0, len(TIME))
            bids.append(
                pd.DataFrame(
                    {
                        "time": TIME,
                        "unit": u,
                        "market": m,
                        "bid_volume": vol,
                        "accepted_volume": acc,
                        "bid_price": _hourly_pattern(55, 20),
                        "accepted_price": _hourly_pattern(60, 20),
                        "activated_volume": acc * RNG.uniform(0.0, 1.0, len(TIME)),
                    }
                )
            )
    bids = pd.concat(bids, ignore_index=True)

    return {
        "units": units,
        "ops": ops,
        "carriers": carriers,
        "bids": bids,
    }


DATA = load_data()
UNIT_OPTIONS = list(DATA["units"]["unit"])


def unit_dropdown(component_id: str, multi: bool = True) -> dcc.Dropdown:
    return dcc.Dropdown(
        UNIT_OPTIONS,
        UNIT_OPTIONS if multi else UNIT_OPTIONS[0],
        multi=multi,
        clearable=False,
        id=component_id,
    )


def _sel(df: pd.DataFrame, units) -> pd.DataFrame:
    units = [units] if isinstance(units, str) else units
    return df[df.unit.isin(units)]


# --------------------------------------------------------------------------- #
# Tab 1: System overview
# --------------------------------------------------------------------------- #
def build_overview_tab():
    u = DATA["units"]
    return html.Div(
        [
            html.H3("Unit configuration & technical specifications"),
            dag.AgGrid(
                id="units-grid",
                rowData=u.to_dict("records"),
                columnDefs=[{"field": c} for c in u.columns],
                dashGridOptions={"domLayout": "autoHeight"},
            ),
            dcc.Graph(
                figure=px.bar(
                    u,
                    x="unit",
                    y=["min_power_MW", "max_power_MW", "grid_connection_MW"],
                    barmode="group",
                    title="Installed capacities",
                )
            ),
            dcc.Graph(
                figure=px.bar(
                    u.melt(
                        id_vars="unit",
                        value_vars=["ramp_up_MW_h", "ramp_down_MW_h"],
                    ),
                    x="unit",
                    y="value",
                    color="variable",
                    barmode="group",
                    title="Ramping constraints",
                )
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# Tab 2: Operational analysis
# --------------------------------------------------------------------------- #
def build_operations_tab():
    return html.Div(
        [
            html.H3("Operational analysis"),
            unit_dropdown("ops-units"),
            dcc.Graph(id="ops-profile"),
            dcc.Graph(id="ops-dispatch"),
            html.Div(
                [
                    dcc.Graph(id="ops-utilisation", style={"flex": 1}),
                    dcc.Graph(id="ops-regime", style={"flex": 1}),
                ],
                style={"display": "flex"},
            ),
            dcc.Graph(id="ops-flex"),
        ]
    )


@callback(
    Output("ops-profile", "figure"),
    Output("ops-dispatch", "figure"),
    Output("ops-utilisation", "figure"),
    Output("ops-regime", "figure"),
    Output("ops-flex", "figure"),
    Input("ops-units", "value"),
)
def update_ops(units):
    d = _sel(DATA["ops"], units)
    cap = DATA["units"].set_index("unit")["max_power_MW"]
    profile = px.line(d, x="time", y="power_MW", color="unit", title="Operating profile")
    dispatch = px.area(
        d, x="time", y="power_MW", color="unit", title="Dispatch (stacked)"
    )
    util = (d.groupby("unit")["power_MW"].mean() / cap[d.unit.unique()]).reset_index()
    util.columns = ["unit", "utilisation"]
    util_fig = px.bar(util, x="unit", y="utilisation", title="Mean utilisation")
    util_fig.update_yaxes(tickformat=".0%")
    regime = px.histogram(
        d, x="unit", color="regime", barmode="group", title="Operating regimes (hours)"
    )
    flex = px.bar(
        d, x="time", y="flex_MW", color="unit", title="Flexibility activation vs baseline"
    )
    return profile, dispatch, util_fig, regime, flex


# --------------------------------------------------------------------------- #
# Tab 3: Financial analysis
# --------------------------------------------------------------------------- #
def build_financial_tab():
    return html.Div(
        [
            html.H3("Financial analysis"),
            unit_dropdown("fin-units"),
            html.Div(id="fin-kpis", style={"display": "flex", "gap": "2rem"}),
            dcc.Graph(id="fin-cashflow"),
            html.Div(
                [
                    dcc.Graph(id="fin-structure", style={"flex": 1}),
                    dcc.Graph(id="fin-savings", style={"flex": 1}),
                ],
                style={"display": "flex"},
            ),
        ]
    )


@callback(
    Output("fin-kpis", "children"),
    Output("fin-cashflow", "figure"),
    Output("fin-structure", "figure"),
    Output("fin-savings", "figure"),
    Input("fin-units", "value"),
)
def update_fin(units):
    d = _sel(DATA["ops"], units)
    kpis = [
        html.Div([html.B(name), html.Div(f"{d[col].sum():,.0f} €")])
        for name, col in [
            ("Energy cost", "energy_cost"),
            ("Revenue", "revenue"),
            ("Cost savings", "savings"),
            ("Net cash flow", "net_cash_flow"),
        ]
    ]
    daily = d.set_index("time").groupby("unit").resample("D")["net_cash_flow"].sum()
    daily = daily.reset_index()
    daily["cumulative"] = daily.groupby("unit")["net_cash_flow"].cumsum()
    cash = px.line(daily, x="time", y="cumulative", color="unit", title="Cumulative cash flow")
    cost = d[["energy_cost", "fuel_cost", "maintenance_cost"]].sum().reset_index()
    cost.columns = ["component", "EUR"]
    structure = px.pie(cost, names="component", values="EUR", title="Cost structure")
    sav = d.groupby("unit")[["baseline_cost", "energy_cost", "savings"]].sum().reset_index()
    savings = px.bar(
        sav,
        x="unit",
        y=["baseline_cost", "energy_cost"],
        barmode="group",
        title="Baseline vs. optimised energy cost",
    )
    return kpis, cash, structure, savings


# --------------------------------------------------------------------------- #
# Tab 4: Energy carriers
# --------------------------------------------------------------------------- #
def build_carrier_tab():
    return html.Div(
        [
            html.H3("Energy carrier analysis"),
            unit_dropdown("car-units"),
            dcc.Graph(id="car-consumption"),
            html.Div(
                [
                    dcc.Graph(id="car-mix", style={"flex": 1}),
                    dcc.Graph(id="car-selection", style={"flex": 1}),
                ],
                style={"display": "flex"},
            ),
            dcc.Graph(id="car-sankey"),
        ]
    )


@callback(
    Output("car-consumption", "figure"),
    Output("car-mix", "figure"),
    Output("car-selection", "figure"),
    Output("car-sankey", "figure"),
    Input("car-units", "value"),
)
def update_carriers(units):
    d = _sel(DATA["carriers"], units)
    cons = px.area(
        d.groupby(["time", "carrier"], as_index=False)["energy_MWh"].sum(),
        x="time",
        y="energy_MWh",
        color="carrier",
        title="Energy consumption by carrier",
    )
    mix = px.pie(
        d.groupby("carrier", as_index=False)["energy_MWh"].sum(),
        names="carrier",
        values="energy_MWh",
        title="Energy mix",
    )
    # dominant carrier per time step as proxy for carrier selection
    dom = d.loc[d.groupby(["time", "unit"])["energy_MWh"].idxmax()]
    sel = px.histogram(
        dom, x="unit", color="carrier", barmode="group", title="Carrier selection (hours dominant)"
    )

    # Sankey: carrier -> unit
    agg = d.groupby(["carrier", "unit"], as_index=False)["energy_MWh"].sum()
    nodes = list(agg.carrier.unique()) + list(agg.unit.unique())
    idx = {n: i for i, n in enumerate(nodes)}
    sankey = go.Figure(
        go.Sankey(
            node={"label": nodes, "pad": 20},
            link={
                "source": agg.carrier.map(idx),
                "target": agg.unit.map(idx),
                "value": agg.energy_MWh,
            },
        )
    )
    sankey.update_layout(title="Energy flows: carrier → unit (MWh)")
    return cons, mix, sel, sankey


# --------------------------------------------------------------------------- #
# Tab 5: Market participation
# --------------------------------------------------------------------------- #
def build_market_tab():
    return html.Div(
        [
            html.H3("Market participation analysis"),
            unit_dropdown("mkt-units"),
            dcc.Dropdown(MARKETS, MARKETS, multi=True, clearable=False, id="mkt-markets"),
            dcc.Graph(id="mkt-prices"),
            dcc.RadioItems(
                options=["volume", "price"], value="volume", id="mkt-bid-kind", inline=True
            ),
            dcc.Graph(id="mkt-bids"),
            dcc.Graph(id="mkt-flex"),
        ]
    )


@callback(
    Output("mkt-prices", "figure"),
    Output("mkt-bids", "figure"),
    Output("mkt-flex", "figure"),
    Input("mkt-units", "value"),
    Input("mkt-markets", "value"),
    Input("mkt-bid-kind", "value"),
)
def update_market(units, markets, kind):
    d = _sel(DATA["bids"], units)
    d = d[d.market.isin(markets)]
    prices = px.line(
        d.groupby(["time", "market"], as_index=False)["accepted_price"].mean(),
        x="time",
        y="accepted_price",
        color="market",
        title="Market prices / price signals",
    )
    daily = (
        d.set_index("time")
        .groupby(["market"])
        .resample("D")[[f"bid_{kind}", f"accepted_{kind}"]]
        .mean()
        .reset_index()
    )
    bids = px.bar(
        daily.melt(id_vars=["market", "time"]),
        x="time",
        y="value",
        color="variable",
        facet_row="market",
        barmode="group",
        title=f"Submitted vs. accepted {kind} (daily mean)",
    )
    flex = d.groupby("market")[["bid_volume", "accepted_volume", "activated_volume"]].sum()
    flex = px.bar(
        flex.reset_index().melt(id_vars="market"),
        x="market",
        y="value",
        color="variable",
        barmode="group",
        title="Flexibility offered / allocated / activated (MWh)",
    )
    return prices, bids, flex


# --------------------------------------------------------------------------- #
# Tab 6: Grid
# --------------------------------------------------------------------------- #
def build_grid_tab():
    return html.Div(
        [
            html.H3("Grid-related analysis"),
            unit_dropdown("grid-units", multi=False),
            html.Div(id="grid-kpis", style={"display": "flex", "gap": "2rem"}),
            dcc.Graph(id="grid-profile"),
            html.Div(
                [
                    dcc.Graph(id="grid-duration", style={"flex": 1}),
                    dcc.Graph(id="grid-compliance", style={"flex": 1}),
                ],
                style={"display": "flex"},
            ),
        ]
    )


@callback(
    Output("grid-kpis", "children"),
    Output("grid-profile", "figure"),
    Output("grid-duration", "figure"),
    Output("grid-compliance", "figure"),
    Input("grid-units", "value"),
)
def update_grid(unit):
    d = _sel(DATA["ops"], unit)
    limit = d.grid_limit_MW.iloc[0]
    peak = d.grid_import_MW.max()
    violations = int((d.grid_import_MW > limit * 0.95).sum())
    kpis = [
        html.Div([html.B(n), html.Div(v)])
        for n, v in [
            ("Peak import", f"{peak:.1f} MW"),
            ("Connection utilisation (peak)", f"{peak / limit:.0%}"),
            ("Hours above 95% of limit", str(violations)),
        ]
    ]
    profile = go.Figure()
    profile.add_scatter(x=d.time, y=d.grid_import_MW, name="Import")
    profile.add_scatter(x=d.time, y=-d.grid_export_MW, name="Export")
    profile.add_hline(y=limit, line_dash="dash", annotation_text="Connection limit")
    profile.update_layout(title="Grid import / export profile", yaxis_title="MW")

    duration = px.line(
        d.sort_values("grid_import_MW", ascending=False).reset_index(drop=True),
        y="grid_import_MW",
        title="Load duration curve",
    )
    # dummy service provision requirement: deliver >= 70% of the offered flexibility
    b = DATA["bids"]
    b = b[(b.unit == unit)].groupby("market")[["accepted_volume", "activated_volume"]].sum()
    b["fulfilment"] = b.activated_volume / b.accepted_volume
    comp = px.bar(b.reset_index(), x="market", y="fulfilment", title="Service provision fulfilment")
    comp.add_hline(y=0.7, line_dash="dash", annotation_text="Requirement")
    comp.update_yaxes(tickformat=".0%")
    return kpis, profile, duration, comp


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
TABS = [
    ("System overview", "overview", build_overview_tab),
    ("Operational analysis", "operations", build_operations_tab),
    ("Financial analysis", "financial", build_financial_tab),
    ("Energy carriers", "carriers", build_carrier_tab),
    ("Market participation", "market", build_market_tab),
    ("Grid", "grid", build_grid_tab),
]

app = Dash(__name__, suppress_callback_exceptions=True)
app.layout = html.Div(
    [
        html.H1("ASSUME Dashboard – Demand-side units"),
        dcc.Tabs(
            id="tabs",
            value="overview",
            children=[dcc.Tab(label=label, value=value) for label, value, _ in TABS],
        ),
        html.Div(id="tab-content", style={"padding": "1rem"}),
    ]
)


@callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    return dict((v, b) for _, v, b in TABS)[tab]()


if __name__ == "__main__":
    app.run(debug=True)
