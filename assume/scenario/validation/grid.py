# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""
Checks of the grid files ``buses.csv`` and ``lines.csv``.

They are not named in the config like the other csv files. A market refers to them through
``param_dict.network_path``, the folder (relative to the scenario folder) which contains them.
"""

from pathlib import Path

import pandas as pd

from assume.scenario.validation.config import StudyCase
from assume.scenario.validation.folder import ScenarioFolder
from assume.scenario.validation.report import Reporter
from assume.scenario.validation.schema import GRID_FILES, UNIT_FILES
from assume.scenario.validation.utils import hint, listed

# clearing mechanisms which cannot work without a grid
GRID_MECHANISMS = ("nodal_clearing", "redispatch")
LINE_REQUIRED = ("bus0", "bus1", "s_nom")
LINE_NUMERIC = ("s_nom", "s_max_pu", "s_nom_forward", "s_nom_reverse", "x", "r")
BUS_NUMERIC = ("v_nom", "x", "y")


class GridChecker:
    """Checks the grid files that the markets of the study cases refer to."""

    def __init__(self, folder: ScenarioFolder, cases: list[StudyCase]):
        self.folder = folder
        self.cases = cases

    def check(self) -> None:
        """Check the grid settings of every market, then each grid folder once."""
        # grid folder -> the zones_identifier of all markets that use it
        grids: dict[Path, set[str]] = {}
        for case in self.cases:
            markets = case.config.get("markets_config")
            if not isinstance(markets, dict):
                continue
            for market_id, params in markets.items():
                if not isinstance(params, dict):
                    continue
                log = case.reporter(section=f"markets_config.{market_id}: ")
                grid = self._check_market(params, log)
                if grid is not None:
                    directory, zones = grid
                    grids.setdefault(directory, set()).update(zones)
        for directory, zones in grids.items():
            self._check_grid(directory, zones)

    def _check_market(
        self, params: dict, log: Reporter
    ) -> tuple[Path, set[str]] | None:
        """Check the grid settings of a market; return its grid folder and zone column."""
        param_dict = params.get("param_dict")
        param_dict = param_dict if isinstance(param_dict, dict) else {}
        network = param_dict.get("network_path")
        if not isinstance(network, str):
            if params.get("market_mechanism") in GRID_MECHANISMS:
                log.error(
                    f"'{params['market_mechanism']}' needs a grid: set param_dict.network_path"
                    " to the folder with buses.csv and lines.csv"
                )
            return None
        directory = self.folder.path / network
        if not directory.is_dir():
            return None  # reported by the config checker
        zones = param_dict.get("zones_identifier")
        if not isinstance(zones, str):
            return directory, set()
        buses = self._read(directory / "buses.csv")
        if buses is not None and zones not in buses.columns:
            log.error(
                f"param_dict.zones_identifier '{zones}' is no column of buses.csv"
                f"{hint(zones, buses.columns)}"
            )
            return directory, set()
        return directory, {zones}

    def _read(self, path: Path) -> pd.DataFrame | None:
        return self.folder.read(path) if path.is_file() else None

    def _check_grid(self, directory: Path, zones: set[str]) -> None:
        """Check the files of a grid folder, which may be shared by several markets."""
        network = directory.relative_to(self.folder.path)
        for name in GRID_FILES:
            if not (directory / name).is_file():
                self.folder.reporter(self.folder.CONFIG_FILE).error(
                    f"network_path '{network}' needs '{name}'"
                )
        buses = self._read(directory / "buses.csv")
        lines = self._read(directory / "lines.csv")
        if buses is not None:
            self._check_buses(buses, self.folder.reporter("buses.csv"))
        if lines is not None:
            self._check_lines(lines, buses, self.folder.reporter("lines.csv"))
        if buses is not None:
            self._check_unit_nodes(directory, buses, zones)

    def _check_buses(self, buses: pd.DataFrame, log: Reporter) -> None:
        if not self._has_rows(buses, "bus", log):
            return
        for column in BUS_NUMERIC:
            self._check_numeric(buses, column, log)

    def _check_lines(
        self, lines: pd.DataFrame, buses: pd.DataFrame | None, log: Reporter
    ) -> None:
        if not self._has_rows(lines, "line", log):
            return
        missing = [c for c in LINE_REQUIRED if c not in lines.columns]
        for column in missing:
            log.error(f"missing column '{column}'{hint(column, lines.columns)}")
        for column in LINE_NUMERIC:
            self._check_numeric(lines, column, log, required=column == "s_nom")
        if "s_max_pu" in lines.columns:
            values = pd.to_numeric(lines["s_max_pu"], errors="coerce")
            if (values <= 0).any():
                log.warning("s_max_pu should be positive, the line cannot carry power")
        for column in ("bus0", "bus1"):
            if column not in lines.columns:
                continue
            if lines[column].isna().any():
                log.error(
                    f"'{column}' is empty for {listed(lines.index[lines[column].isna()])}"
                )
            elif buses is not None:
                unknown = lines.loc[~lines[column].isin(buses.index), column]
                if len(unknown):
                    log.error(
                        f"'{column}' refers to buses missing in buses.csv: "
                        f"{listed(unknown.unique())}{hint(unknown.iloc[0], buses.index)}"
                    )
        if not missing:
            loops = lines.index[lines["bus0"] == lines["bus1"]]
            if len(loops):
                log.warning(f"lines connect a bus with itself: {listed(loops)}")

    def _check_unit_nodes(
        self, directory: Path, buses: pd.DataFrame, zones: set[str]
    ) -> None:
        """Units are placed on a bus (or, with zones, a zone) through their ``node`` column."""
        nodes = set(buses.index)
        for zone in zones & set(buses.columns):
            nodes |= set(buses[zone].dropna())
        for key in UNIT_FILES:
            path = directory / f"{key}.csv"
            df = self.folder.read(path) if path.is_file() else None
            if df is None or "node" not in df.columns:
                continue
            unknown = df.loc[~df["node"].isin(nodes), "node"].dropna()
            if len(unknown):
                self.folder.reporter(path.name).warning(
                    f"'node' of {listed(unknown.index)} is no bus of buses.csv: "
                    f"{listed(unknown.unique())}{hint(unknown.iloc[0], nodes)}"
                )

    @staticmethod
    def _has_rows(df: pd.DataFrame, what: str, log: Reporter) -> bool:
        if df.empty:
            log.error("contains no rows")
            return False
        duplicates = df.index[df.index.duplicated()].unique()
        if len(duplicates):
            log.error(f"duplicate {what} names: {listed(duplicates)}")
        if df.index.isna().any():
            log.error(f"has rows without a {what} name in the first column")
        return True

    @staticmethod
    def _check_numeric(
        df: pd.DataFrame, column: str, log: Reporter, required: bool = False
    ) -> None:
        if column not in df.columns:
            return
        values = pd.to_numeric(df[column], errors="coerce")
        invalid = df.index[values.isna() & df[column].notna()]
        if len(invalid):
            log.error(f"'{column}' must be a number, check {listed(invalid)}")
        elif required and values.isna().any():
            log.error(f"'{column}' is empty for {listed(df.index[values.isna()])}")
        elif column.startswith("s_nom") and (values < 0).any():
            log.error(f"'{column}' must not be negative")
