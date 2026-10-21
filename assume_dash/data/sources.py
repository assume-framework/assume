# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Backends that deliver raw ASSUME output tables as DataFrames.

The dashboard only talks to :class:`DataSource`. SQL databases (the same ones
Grafana used) and CSV export folders share the table layout written by
``assume.common.outputs.WriteOutput``.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect, text


class DataSource(ABC):
    """Read-only access to the ASSUME output tables."""

    @abstractmethod
    def tables(self) -> set[str]:
        """Names of all available tables."""

    @abstractmethod
    def read_table(self, table: str, simulation: str | None = None) -> pd.DataFrame:
        """Return a table, optionally restricted to a single simulation."""

    def has_table(self, table: str) -> bool:
        return table in self.tables()


def _normalise_index(df: pd.DataFrame) -> pd.DataFrame:
    """Meta tables store the unit id in an unnamed index column."""
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "index"})
    return df


class SqlSource(DataSource):
    """Any SQLAlchemy URL, e.g. ``postgresql://assume:assume@localhost:5432/assume``."""

    def __init__(self, url: str):
        self.engine = create_engine(url)

    def tables(self) -> set[str]:
        return set(inspect(self.engine).get_table_names())

    def read_table(self, table: str, simulation: str | None = None) -> pd.DataFrame:
        if not self.has_table(table):
            return pd.DataFrame()
        query, params = f'SELECT * FROM "{table}"', {}
        columns = {c["name"] for c in inspect(self.engine).get_columns(table)}
        if simulation is not None and "simulation" in columns:
            query += " WHERE simulation = :sim"
            params["sim"] = simulation
        with self.engine.connect() as con:
            return pd.read_sql(text(query), con, params=params)


class CsvSource(DataSource):
    """A folder of ``<table>.csv`` files, or a folder of such folders.

    Every folder holding CSVs is treated as one result set; files of the same
    name in different sub folders are concatenated (the ``simulation`` column
    keeps them apart).
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"CSV folder not found: {self.root}")
        files = list(self.root.glob("*.csv")) + list(self.root.glob("*/*.csv"))
        self._files: dict[str, list[Path]] = {}
        for f in sorted(files):
            self._files.setdefault(f.stem, []).append(f)

    def tables(self) -> set[str]:
        return set(self._files)

    def read_table(self, table: str, simulation: str | None = None) -> pd.DataFrame:
        frames = [_normalise_index(pd.read_csv(f)) for f in self._files.get(table, [])]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        if simulation is not None and "simulation" in df.columns:
            df = df[df["simulation"] == simulation]
        return df.reset_index(drop=True)
