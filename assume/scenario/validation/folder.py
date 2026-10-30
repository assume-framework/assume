# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import yaml

from assume.scenario.validation.report import Reporter, ValidationReport
from assume.scenario.validation.utils import YamlMapping, YamlValue


@dataclass(frozen=True)
class Registry:
    """What ASSUME offers and a scenario can refer to by name."""

    unit_types: dict[str, type]
    strategies: frozenset[str]
    mechanisms: frozenset[str]

    @classmethod
    def load(cls) -> "Registry":
        from assume.markets.clearing_algorithms import clearing_mechanisms
        from assume.strategies import bidding_strategies, deprecated_bidding_strategies
        from assume.units import unit_types

        return cls(
            unit_types=dict(unit_types),
            strategies=frozenset(bidding_strategies)
            | frozenset(deprecated_bidding_strategies),
            mechanisms=frozenset(clearing_mechanisms),
        )


class FileRef(NamedTuple):
    """Where the loader looks for a file of a study case."""

    path: Path | None  # None if the file is disabled with `null` in the config
    explicit: bool  # whether the config names the file


class ScenarioFolder:
    """
    A scenario folder on disk: ``config.yaml`` plus the csv files.

    It reads files the way the loader does and remembers what it already read, so that csv files
    shared between study cases are parsed (and their parse errors reported) only once.
    """

    CONFIG_FILE = "config.yaml"

    def __init__(self, path: str | Path, registry: Registry | None = None):
        self.path = Path(path)
        self.registry = registry or Registry.load()
        self.report = ValidationReport(path=str(path))
        self._csv: dict[Path, pd.DataFrame | None] = {}
        self._checked: set[Path] = set()

    def reporter(self, file: str, prefix: str = "") -> Reporter:
        return self.report.scope(file, prefix)

    def load_config(self) -> dict[object, YamlValue] | None:
        """Read ``config.yaml``; report and return None if that is not possible."""
        log = self.reporter(self.CONFIG_FILE)
        if not self.path.is_dir():
            self.report.error(str(self.path), "is not a directory")
            return None
        config_file = self.path / self.CONFIG_FILE
        if not config_file.is_file():
            log.error(f"not found in {self.path}")
            return None
        try:
            with open(config_file, encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except (yaml.YAMLError, UnicodeDecodeError) as e:
            log.error(f"cannot be parsed: {e}")
            return None
        if not isinstance(config, dict) or not config:
            log.error("must be a mapping of study case names to their configuration")
            return None
        return config

    def resolve(self, case_config: YamlMapping, name: str) -> FileRef:
        """Mirror of ``load_file``: a key in the config renames (or, if null, disables) a file."""
        if name in case_config:
            if case_config[name] is None:
                return FileRef(None, True)
            return FileRef(self.path / str(case_config[name]), True)
        return FileRef(self.path / f"{name}.csv", False)

    def read(self, path: Path, parse_dates: bool = False) -> pd.DataFrame | None:
        """Read a csv file like the loader does; None (and an error) if it is not valid."""
        if path not in self._csv:
            try:
                with warnings.catch_warnings():
                    # pandas warns about index values it cannot parse as dates; we report that ourselves
                    warnings.simplefilter("ignore", UserWarning)
                    self._csv[path] = pd.read_csv(
                        path,
                        index_col=0,
                        encoding="utf-8",
                        na_values=["n.a.", "None", "-", "none", "nan"],
                        parse_dates=parse_dates,
                    )
            except (
                pd.errors.ParserError,
                pd.errors.EmptyDataError,
                UnicodeDecodeError,
                ValueError,
            ) as e:
                self.reporter(path.name).error(f"is not a valid csv file: {e}")
                self._csv[path] = None
        return self._csv[path]

    def first_visit(self, path: Path) -> bool:
        """True the first time it is called for ``path``, to check shared files only once."""
        if path in self._checked:
            return False
        self._checked.add(path)
        return True
