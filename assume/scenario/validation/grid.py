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
    """Checks the grid files that the markets of one study case refer to."""

    def __init__(self, folder: ScenarioFolder, case: StudyCase):
        self.folder = folder
        self.case = case

    def check(self) -> None:
        markets = self.case.config.get("markets_config")
        if not isinstance(markets, dict):
            return
        for market_id, params in markets.items():
            if isinstance(params, dict):
                self._check_market(str(market_id), params)

    def _check_market(self, market_id: str, params: dict) -> None:
        log = self.case.reporter(section=f"markets_config.{market_id}: ")
        param_dict = params.get("param_dict")
        param_dict = param_dict if isinstance(param_dict, dict) else {}
        network = param_dict.get("network_path")
        if not isinstance(network, str):
            if params.get("market_mechanism") in GRID_MECHANISMS:
                log.error(
                    f"'{params['market_mechanism']}' needs a grid: set param_dict.network_path"
                    " to the folder with buses.csv and lines.csv"
                )
            return
        directory = self.folder.path / network
        if not directory.is_dir():
            return  # reported by the config checker
        buses, lines = self._read(directory, log)
        zones = param_dict.get("zones_identifier")
        if buses is not None and isinstance(zones, str) and zones not in buses.columns:
            log.error(
                f"param_dict.zones_identifier '{zones}' is no column of buses.csv"
                f"{hint(zones, buses.columns)}"
            )
            zones = None
        if self.folder.first_visit(directory):
            self._check_files(directory, buses, lines)

    def _read(
        self, directory: Path, log: Reporter
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        found = []
        for name in GRID_FILES:
            path = directory / name
            if path.is_file():
                found.append(self.folder.read(path))
            else:
                log.error(f"needs '{name}' in network_path '{directory.name}'")
                found.append(None)
        buses, lines = found
        return buses, lines

    def _check_files(
        self, directory: Path, buses: pd.DataFrame | None, lines: pd.DataFrame | None
    ) -> None:
        if buses is not None:
            self._check_buses(buses, self.folder.reporter("buses.csv"))
        if lines is not None:
            self._check_lines(lines, buses, self.folder.reporter("lines.csv"))
        if buses is not None:
            self._check_unit_nodes(directory, buses, self._zone_columns(directory))

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

    def _zone_columns(self, directory: Path) -> set[str]:
        """The ``zones_identifier`` of all markets of the study case using this grid."""
        zones = set()
        for params in self.case.config["markets_config"].values():
            param_dict = params.get("param_dict") if isinstance(params, dict) else None
            if isinstance(param_dict, dict):
                network, zone = (
                    param_dict.get("network_path"),
                    param_dict.get("zones_identifier"),
                )
                if isinstance(network, str) and isinstance(zone, str):
                    if self.folder.path / network == directory:
                        zones.add(zone)
        return zones

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
