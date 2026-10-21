# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Base class every dashboard tab implements."""

from abc import ABC, abstractmethod

from dash import Dash
from dash.development.base_component import Component

from assume_dash.data import DashboardData


class Tab(ABC):
    """A self-contained dashboard page: layout plus its own callbacks.

    Subclasses only depend on :class:`DashboardData` and the shared helpers in
    ``assume_dash.components``; they never touch each other.
    """

    id: str
    title: str

    def __init__(self, data: DashboardData):
        self.data = data

    def cid(self, name: str) -> str:
        """Component id namespaced by the tab, so tabs cannot collide."""
        return f"{self.id}-{name}"

    def available(self) -> bool:
        """Tabs whose tables are missing in the data source are hidden."""
        return True

    @abstractmethod
    def layout(self) -> Component:
        """Static layout; dynamic content is filled by callbacks."""

    @abstractmethod
    def register_callbacks(self, app: Dash) -> None:
        """Attach all callbacks of the tab to ``app``."""
