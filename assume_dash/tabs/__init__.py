# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Registry of dashboard tabs. Add a new tab by appending its class here."""

from assume_dash.tabs.base import Tab
from assume_dash.tabs.comparison import ComparisonTab
from assume_dash.tabs.dsm import DsmTab
from assume_dash.tabs.kpis import KpiTab
from assume_dash.tabs.learning import LearningTab
from assume_dash.tabs.market import MarketTab
from assume_dash.tabs.nodal import NodalTab
from assume_dash.tabs.units import UnitsTab

ALL_TABS: list[type[Tab]] = [
    MarketTab,
    UnitsTab,
    DsmTab,
    KpiTab,
    ComparisonTab,
    LearningTab,
    NodalTab,
]

__all__ = ["ALL_TABS", "Tab"]
