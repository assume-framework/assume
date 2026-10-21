# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Application factory wiring data, global controls and tabs together."""

import copy
from pathlib import Path

import plotly.io as pio
from dash import Dash, Input, Output, clientside_callback, dcc, html

from assume_dash.components.controls import DATES, SIMULATION, global_controls
from assume_dash.data import DashboardData, DataSource
from assume_dash.tabs import ALL_TABS, Tab

THEMES = {"light": "Light", "dark": "Dark"}

# brand colours first, then harmonised extensions so 8+ series stay distinguishable
COLORWAY = [
    "#EE6063",  # coral
    "#82CDD7",  # light blue
    "#84C69B",  # mint
    "#E2DC6D",  # yellow
    "#8A9AA5",  # slate grey (visible on light and dark surfaces)
    "#8E6BBF",  # purple
    "#F2A65A",  # orange
    "#3E6FB0",  # deep blue
    "#4C9A6A",  # dark green
    "#B8393C",  # dark coral
    "#C77DA5",  # mauve
    "#B3A62E",  # olive
    "#3E95A3",  # teal
    "#B08968",  # brown
    "#A7D86E",  # lime
    "#8C8C8C",  # grey
]


def _register_plotly_template() -> None:
    template = copy.deepcopy(pio.templates["plotly_white"])
    template.layout.colorway = COLORWAY
    template.layout.font.color = "#333333"
    # at most two decimals in tick and hover labels; date x axes are left untouched
    template.layout.yaxis.hoverformat = ",.2~f"
    template.layout.yaxis.tickformat = ",.2~f"
    pio.templates["assume"] = template
    pio.templates.default = "assume"


def _index_with_css(css: str) -> str:
    """Dash index page carrying user CSS after the packaged theme."""
    return (
        "<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}"
        f"{{%css%}}<style>{css}</style></head>"
        "<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer>"
        "</body></html>"
    )


def create_app(
    source: DataSource,
    tabs: list[type[Tab]] | None = None,
    theme: str = "light",
    custom_css: str | Path | None = None,
) -> Dash:
    """Build the dashboard for a data source; tabs without data are skipped.

    ``theme`` is the initial palette (``light`` or ``dark``). ``custom_css`` is a
    CSS file overriding the variables of ``assets/style.css``.
    """
    data = DashboardData(source)
    _register_plotly_template()
    active = [t for t in (tab(data) for tab in (tabs or ALL_TABS)) if t.available()]
    simulations = data.simulations()

    app = Dash(__name__, title="ASSUME dashboard", suppress_callback_exceptions=True)
    if custom_css:
        app.index_string = _index_with_css(Path(custom_css).read_text(encoding="utf-8"))
    layouts = {t.id: t.layout() for t in active}
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H1(["ASSUME ", html.Span("dashboard")]),
                    dcc.RadioItems(
                        id="tabs",
                        options=[{"label": t.title, "value": t.id} for t in active],
                        value=active[0].id if active else None,
                        className="nav-list",
                    ),
                    dcc.RadioItems(
                        THEMES,
                        theme,
                        inline=True,
                        id="theme",
                        className="theme-switch",
                    ),
                ],
                className="sidebar",
            ),
            html.Div(
                [
                    global_controls(simulations),
                    # all pages stay mounted so their selections survive navigation
                    html.Div(
                        [
                            html.Div(
                                layouts[t.id],
                                id=f"page-{t.id}",
                                style={"display": "none"},
                            )
                            for t in active
                        ]
                    ),
                ],
                className="main",
            ),
        ],
        className="app",
    )

    clientside_callback(
        "function(theme) {"
        " document.documentElement.setAttribute('data-theme', theme);"
        " return window.dash_clientside.no_update; }",
        Output("theme", "className"),
        Input("theme", "value"),
    )

    @app.callback(
        [Output(f"page-{t.id}", "style") for t in active], Input("tabs", "value")
    )
    def show_tab(selected):
        return [{} if t.id == selected else {"display": "none"} for t in active]

    @app.callback(
        Output(DATES, "min_date_allowed"),
        Output(DATES, "max_date_allowed"),
        Output(DATES, "start_date"),
        Output(DATES, "end_date"),
        Output(DATES, "initial_visible_month"),
        Input(SIMULATION, "value"),
    )
    def reset_dates(simulation):
        """Fit the time window to the data of the newly selected simulation."""
        bounds = data.time_bounds(simulation) if simulation else None
        if bounds is None:
            return None, None, None, None, None
        start, end = (b.date().isoformat() for b in bounds)
        return start, end, start, end, start

    for tab in active:
        tab.register_callbacks(app)
    return app
