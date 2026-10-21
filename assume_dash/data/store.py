# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cached, typed access to the ASSUME result tables used by the dashboard."""

from collections import OrderedDict

import pandas as pd

from assume_dash.data.sources import DataSource

# unit kind -> meta table holding its static specification
UNIT_KINDS: dict[str, str] = {
    "generation": "power_plant_meta",
    "demand": "demand_meta",
    "storage": "storage_meta",
    "electric_vehicle": "electric_vehicle_meta",
}

_TIME_COLUMNS = {
    "market_meta": ["product_start", "product_end", "time"],
    "market_orders": ["start_time", "end_time"],
    "market_dispatch": ["datetime"],
    "unit_dispatch": ["time"],
    "rl_params": ["datetime"],
    "grid_flows": ["datetime"],
}


class DashboardData:
    """Facade used by all tabs; hides the backend and caches loaded tables."""

    def __init__(self, source: DataSource, cache_size: int = 24):
        self.source = source
        self._cache: OrderedDict[tuple[str, str | None], pd.DataFrame] = OrderedDict()
        self._cache_size = cache_size

    # ------------------------------------------------------------------ #
    # raw tables
    # ------------------------------------------------------------------ #
    def table(self, name: str, simulation: str | None = None) -> pd.DataFrame:
        """Return a table with parsed time columns. Never mutate the result."""
        key = (name, simulation)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        df = self.source.read_table(name, simulation)
        for col in _TIME_COLUMNS.get(name, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        self._cache[key] = df
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return df

    def has(self, name: str) -> bool:
        return self.source.has_table(name)

    # ------------------------------------------------------------------ #
    # catalogue helpers (dropdown options)
    # ------------------------------------------------------------------ #
    def simulations(self) -> list[str]:
        for name in ("market_meta", "unit_dispatch", "power_plant_meta", "rl_params"):
            df = self.table(name)
            if "simulation" in df.columns and not df.empty:
                return sorted(df["simulation"].dropna().unique())
        return []

    def markets(self, simulation: str) -> list[str]:
        df = self.table("market_meta", simulation)
        return sorted(df["market_id"].dropna().unique()) if not df.empty else []

    def available_kinds(self) -> list[str]:
        return [k for k, t in UNIT_KINDS.items() if self.has(t)]

    def unit_meta(self, simulation: str, kind: str) -> pd.DataFrame:
        df = self.table(UNIT_KINDS[kind], simulation)
        if df.empty:
            return df
        return df.rename(columns={"index": "unit"}) if "index" in df.columns else df

    def unit_ids(self, simulation: str, kind: str) -> list[str]:
        meta = self.unit_meta(simulation, kind)
        return list(meta["unit"]) if "unit" in meta.columns else []

    def time_bounds(self, simulation: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        df = self.table("market_meta", simulation)
        if df.empty:
            df = self.table("unit_dispatch", simulation)
            col = "time"
        else:
            col = "product_start"
        if df.empty:
            return None
        return df[col].min(), df[col].max()

    def rl_simulations(self) -> list[str]:
        df = self.table("rl_params")
        return sorted(df["simulation"].unique()) if not df.empty else []

    # ------------------------------------------------------------------ #
    # derived frames
    # ------------------------------------------------------------------ #
    def orders(self, simulation: str) -> pd.DataFrame:
        """Market orders with the Grafana-style ``bid_label`` (unit + bid type)."""
        df = self.table("market_orders", simulation)
        if df.empty:
            return df
        df = df.copy()
        bid_type = (
            df["bid_type"]
            if "bid_type" in df.columns
            else pd.Series(None, index=df.index)
        )
        suffix = bid_type.fillna("SB").replace("", "SB")
        df["bid_label"] = df["unit_id"].astype(str) + "_" + suffix.astype(str)
        return df

    def orders_with_technology(self, simulation: str) -> pd.DataFrame:
        orders = self.orders(simulation)
        meta = self.unit_meta(simulation, "generation")
        if orders.empty or meta.empty or "technology" not in meta.columns:
            return orders.assign(technology="unknown") if not orders.empty else orders
        tech = meta.set_index("unit")["technology"]
        return orders.assign(technology=orders["unit_id"].map(tech).fillna("other"))
