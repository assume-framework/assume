# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""
Checks of the csv files of a scenario.

Every csv file is wrapped in a :class:`Table`. The subclasses know what their kind of file
must look like: :class:`UnitTable` (power plants, demand, storage, exchange), :class:`DsmUnitTable`,
:class:`OperatorTable` and :class:`TimeSeriesTable`. :class:`CaseFiles` bundles the tables of one
study case and checks how they fit together.
"""

import difflib
import inspect
import typing
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from assume.common.utils import load_index_file
from assume.scenario.validation.config import StudyCase
from assume.scenario.validation.folder import ScenarioFolder
from assume.scenario.validation.schema import (
    DSM_FILES,
    FILE_KEYS,
    REQUIRED_UNIT_FILES,
    TIMESERIES_FILES,
    UNIT_FILES,
)
from assume.scenario.validation.utils import (
    Annotation,
    hint,
    listed,
    quiet_logging,
)

# constructor arguments the loader fills in itself, so they are no csv columns
NON_COLUMN_PARAMS = {"self", "id", "forecaster", "bidding_strategies"}


@dataclass
class UnitSignature:
    """The csv columns a unit class accepts, derived from its ``__init__``."""

    accepted: dict[str, Annotation]  # column -> annotation (None if not annotated)
    required: list[str]

    @classmethod
    def of(cls, unit_class: type) -> "UnitSignature":
        accepted: dict[str, Annotation] = {}
        for klass in unit_class.__mro__:
            init = vars(klass).get("__init__")
            if init is None:
                continue
            try:
                hints = typing.get_type_hints(init)
            except Exception:  # unresolvable forward references
                hints = {}
            for name, param in inspect.signature(init).parameters.items():
                if param.kind not in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
                    accepted.setdefault(name, hints.get(name))
        leaf = inspect.signature(unit_class).parameters
        required = [
            name
            for name, p in leaf.items()
            if p.default is p.empty
            and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
            and name not in NON_COLUMN_PARAMS
        ]
        return cls(accepted, required)


class Table:
    """A csv file of the scenario folder."""

    def __init__(
        self,
        folder: ScenarioFolder,
        case: StudyCase,
        key: str,
        path: Path,
        df: pd.DataFrame,
    ):
        self.folder = folder
        self.case = case
        self.key = key  # the file's name in the config, e.g. "powerplant_units"
        self.path = path
        self.df = df
        self.log = folder.reporter(path.name)  # about the file itself
        self.case_log = case.reporter(path.name)  # about the file in this study case

    @property
    def index(self) -> pd.Index:
        return self.df.index

    def has_rows(self) -> bool:
        """Report an empty file or unnamed rows; False if there is nothing left to check."""
        if self.df.empty:
            report = (
                self.log.error if self.key in REQUIRED_UNIT_FILES else self.log.warning
            )
            report("contains no rows")
            return False
        if self.index.isna().any():
            self.log.error("has rows without a unit name in the first column")
        return True


class BiddingTable(Table):
    """A table of units or operators with ``bidding_<market>`` columns, which are shared by study cases."""

    UNIQUE_NAMES = True
    NEEDS_STRATEGY = True  # whether units without any strategy are suspicious

    def check(self, markets: set[str]) -> None:
        """
        Check the file.

        Args:
            markets (set[str]): The ids of all markets defined in the verified study cases.
        """
        if not self.has_rows():
            return
        if self.UNIQUE_NAMES:
            duplicates = self.index[self.index.duplicated()].unique()
            if len(duplicates):
                self.log.error(f"duplicate unit names: {listed(duplicates)}")
        self.check_columns()
        for column in self.bidding_columns:
            self._check_bidding_column(column, markets)
        if self.NEEDS_STRATEGY:
            self._check_strategies_given()

    def check_columns(self) -> None:
        """Check the columns specific to the kind of table."""

    @property
    def bidding_columns(self) -> list[str]:
        return [str(c) for c in self.df.columns if str(c).startswith("bidding_")]

    @staticmethod
    def is_free_form(column: str) -> bool:
        """Columns with names chosen by the user: forecast algorithms and the bidding strategies."""
        return column.startswith(("bidding_", "forecast_"))

    def _check_bidding_column(self, column: str, markets: set[str]) -> None:
        market = column.removeprefix("bidding_")
        if markets and market not in markets:
            self.log.warning(
                f"column '{column}' refers to market '{market}', which no study case defines"
                f"{hint(market, markets)}"
            )
        strategies = self.folder.registry.strategies
        for unit, strategy in self.df[column].dropna().items():
            if strategy not in strategies:
                learning = (
                    " (learning strategies are only available if torch is installed)"
                    if "learning" in str(strategy)
                    else ""
                )
                self.log.error(
                    f"{unit}: unknown bidding strategy '{strategy}' in '{column}'"
                    f"{hint(strategy, strategies)}{learning}"
                )

    def _check_strategies_given(self) -> None:
        columns = self.bidding_columns
        if not columns:
            self.log.warning("no 'bidding_<market>' column, the units never bid")
            return
        # DSM units span several rows (one per technology), the strategy is taken from any of them
        has_strategy = self.df[columns].notna().any(axis=1).groupby(level=0).any()
        silent = has_strategy.index[~has_strategy]
        if len(silent):
            self.log.warning(
                f"units without any bidding strategy never bid: {listed(silent)}"
            )


class UnitTable(BiddingTable):
    """Power plants, demand, storage and exchange units: columns are the unit's constructor arguments."""

    def __init__(
        self,
        folder: ScenarioFolder,
        case: StudyCase,
        key: str,
        path: Path,
        df: pd.DataFrame,
    ):
        super().__init__(folder, case, key, path, df)
        unit_class = self.folder.registry.unit_types[UNIT_FILES[self.key]]
        self.signature = UnitSignature.of(unit_class)

    def check_columns(self) -> None:
        for column in self.signature.required:
            if column not in self.df.columns:
                self.log.error(f"missing required column '{column}'")
        for column in self.df.columns:
            if self.is_free_form(str(column)):
                continue
            if column not in self.signature.accepted:
                self.log.warning(
                    f"unknown column '{column}' is ignored{hint(column, self.signature.accepted)}"
                )
            else:
                self._check_column_type(column, self.signature.accepted[column])

    def _check_column_type(self, column: str, annotation: Annotation) -> None:
        series = self.df[column].dropna()
        if series.empty:
            return
        args = {annotation, *typing.get_args(annotation)}
        if args & {int, float} and str not in args:
            bad = series[pd.to_numeric(series, errors="coerce").isna()]
            expected = "a number"
        elif bool in args and not args & {int, float, str}:
            bad = series[~series.isin([True, False, 0, 1])]
            expected = "True/False (or 1/0)"
        else:
            return
        if len(bad):
            shown = listed(f"{unit}={value!r}" for unit, value in bad.items())
            self.log.error(f"column '{column}' must be {expected}, got {shown}")


class DsmUnitTable(BiddingTable):
    """Demand side units: one row per technology of a unit, so names repeat and columns vary."""

    UNIQUE_NAMES = False

    def check_columns(self) -> None:
        for column in ("unit_type", "technology", "unit_operator"):
            if column not in self.df.columns:
                self.log.error(f"missing required column '{column}'")
        if "unit_type" in self.df.columns:
            unit_types = self.folder.registry.unit_types
            for unit_type in self.df["unit_type"].dropna().unique():
                if unit_type not in unit_types:
                    self.log.error(
                        f"unknown unit_type '{unit_type}'{hint(unit_type, unit_types)}"
                    )


class OperatorTable(BiddingTable):
    """Unit operators with portfolio strategies."""

    UNIQUE_NAMES = False
    NEEDS_STRATEGY = False

    def check_columns(self) -> None:
        for column in self.df.columns:
            if not self.is_free_form(str(column)):
                self.log.warning(f"unknown column '{column}' is ignored")


class TimeSeriesTable(Table):
    """A table indexed by time (demand, availability, fuel prices, ...), checked per study case."""

    def check(self) -> None:
        """Check that the table is numeric and covers the simulation time line of the study case."""
        log, df, index = self.case_log, self.df, self.case.index
        numeric = df.apply(pd.to_numeric, errors="coerce")
        text = [c for c in df.columns if (numeric[c].isna() & df[c].notna()).any()]
        if text:
            log.error(f"non-numeric values in: {listed(text)}")
        empty = [c for c in df.columns if df[c].isna().all()]
        if empty:
            log.warning(f"columns without any value: {listed(empty)}")

        # a single row is accepted as a constant value (e.g. fuel prices)
        if not isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
            log.error("first column must contain datetimes")
        elif index is not None and not self._covers(index):
            log.error(
                f"does not cover the simulation time line ({index[0]} to {index[-1]}, "
                f"step {index.freq}); the loader would ignore the file"
            )

    def _covers(self, index: pd.DatetimeIndex) -> bool:
        """`load_index_file` ignores (returns None) a file which cannot be mapped onto the time line."""
        try:
            with quiet_logging():
                return load_index_file(self.path, index) is not None
        except (KeyError, ValueError, TypeError):
            return False


def table_class(key: str) -> type[TimeSeriesTable | BiddingTable]:
    if key in TIMESERIES_FILES:
        return TimeSeriesTable
    if key in DSM_FILES:
        return DsmUnitTable
    if key == "unit_operators":
        return OperatorTable
    return UnitTable


class CaseFiles:
    """The csv files of one study case."""

    def __init__(self, folder: ScenarioFolder, case: StudyCase):
        self.folder = folder
        self.case = case
        self.tables: dict[str, TimeSeriesTable | BiddingTable] = {}
        self._load()

    def _load(self) -> None:
        for key in FILE_KEYS:
            ref = self.folder.resolve(self.case.config, key)
            if ref.path is None:
                continue
            if not ref.path.is_file():
                if ref.explicit:
                    self.case.reporter().error(
                        f"'{key}' points to missing file '{ref.path.name}'"
                    )
                elif key in REQUIRED_UNIT_FILES:
                    self.folder.reporter(ref.path.name).error(
                        f"required file not found in {self.folder.path}"
                    )
                continue
            is_timeseries = key in TIMESERIES_FILES
            df = (
                self.folder.read(ref.path, parse_dates=True)
                if is_timeseries
                else self.folder.read(ref.path)
            )
            if df is not None:
                self.tables[key] = table_class(key)(
                    self.folder, self.case, key, ref.path, df
                )

    def check(self, markets: set[str]) -> None:
        """
        Check all files of the study case and how they fit together.

        Args:
            markets (set[str]): The ids of all markets defined in the verified study cases.
        """
        for table in self.tables.values():
            if isinstance(table, TimeSeriesTable):
                table.check()  # depends on the time line, so once per study case
            elif self.folder.first_visit(table.path):
                table.check(markets)
        self._check_demand()
        self._check_availability()
        self._check_exchanges()
        self._check_fuel_prices()

    def _check_demand(self) -> None:
        units, series = self.tables.get("demand_units"), self.tables.get("demand_df")
        if units is None or series is None:
            return
        unknown = series.df.columns.difference(units.index)
        if len(unknown):
            series.case_log.warning(
                f"columns without a unit in demand_units: {listed(unknown)}"
            )
        missing = units.index.difference(series.df.columns)
        if len(missing):
            series.case_log.warning(
                f"no demand time series for {listed(missing)}, filled with zeros"
            )

    def _check_availability(self) -> None:
        series = self.tables.get("availability_df")
        if series is None:
            return
        units = pd.Index([])
        for key in (*UNIT_FILES, *DSM_FILES):
            if key in self.tables:
                units = units.append(self.tables[key].index)
        unknown = series.df.columns.difference(units)
        if len(unknown):
            series.case_log.warning(
                f"columns without a matching unit: {listed(unknown)}"
            )

    def _check_exchanges(self) -> None:
        units = self.tables.get("exchange_units")
        if units is None:
            return
        series = self.tables.get("exchanges_df")
        if series is None:
            self.case.reporter("exchanges_df.csv").error(
                "required by exchange_units but missing"
            )
            return
        for unit in units.index:
            for direction in ("import", "export"):
                if f"{unit}_{direction}" not in series.df.columns:
                    series.case_log.error(f"missing column '{unit}_{direction}'")

    def _check_fuel_prices(self) -> None:
        plants = self.tables.get("powerplant_units")
        if plants is None or "fuel_type" not in plants.df.columns:
            return
        prices = self.tables.get("fuel_prices_df")
        if prices is None:
            return
        known = set(prices.df.columns)
        # fuels without a price are valid (price 0), so only flag likely typos
        for fuel in sorted(set(plants.df["fuel_type"].dropna()) - known):
            close = difflib.get_close_matches(fuel, known, n=1)
            if close:
                prices.case_log.warning(
                    f"fuel_type '{fuel}' of a power plant has no price, assumed 0"
                    f" (did you mean '{close[0]}'?)"
                )
