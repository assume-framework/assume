# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""Checks of ``config.yaml``: study cases, time settings, ``markets_config`` and ``learning_config``."""

import typing
from dataclasses import fields

import pandas as pd

from assume.common.base import LearningConfig
from assume.common.utils import convert_to_rrule_freq
from assume.scenario.validation.folder import ScenarioFolder
from assume.scenario.validation.report import Reporter, ValidationReport
from assume.scenario.validation.schema import (
    FILE_KEYS,
    MARKET_KEYS,
    OPTIONAL_CASE_KEYS,
    PRODUCT_KEYS,
    REQUIRED_CASE_KEYS,
    REQUIRED_MARKET_KEYS,
)
from assume.scenario.validation.utils import (
    YamlMapping,
    YamlValue,
    describe_wrong_type,
    hint,
    listed,
    type_ok,
)


class StudyCase:
    """One study case of ``config.yaml``, e.g. ``base``."""

    def __init__(self, name: str, config: YamlMapping, report: ValidationReport):
        self.name = name
        self.config = config
        self.index: pd.DatetimeIndex | None = (
            None  # simulation time line, set by the checker
        )
        self._report = report

    @property
    def market_ids(self) -> set[str]:
        markets = self.config.get("markets_config")
        return {str(m) for m in markets} if isinstance(markets, dict) else set()

    def reporter(self, file: str = "config.yaml", section: str = "") -> Reporter:
        """Reporter whose messages are marked with the study case (and the config section)."""
        return self._report.scope(file, prefix=f"[{self.name}] {section}")


class ConfigChecker:
    """Checks that the study cases declare the right keys with values of the right type."""

    def __init__(self, folder: ScenarioFolder):
        self.folder = folder
        self.registry = folder.registry

    def load_cases(self, only_case: str = "") -> list[StudyCase]:
        """Read the study cases from ``config.yaml``: all of them, or only ``only_case``."""
        config = self.folder.load_config()
        if config is None:
            return []
        log = self.folder.reporter(self.folder.CONFIG_FILE)
        if only_case:
            if only_case not in config:
                log.error(
                    f"study case '{only_case}' not found{hint(only_case, config)}. "
                    f"Available: {listed(config)}"
                )
                return []
            config = {only_case: config[only_case]}
        cases = []
        for name, case_config in config.items():
            if isinstance(case_config, dict):
                cases.append(StudyCase(str(name), case_config, self.folder.report))
            else:
                log.error(
                    f"[{name}] must be a mapping, got {type(case_config).__name__}"
                )
        return cases

    def check(self, case: StudyCase) -> None:
        """Check one study case and, if its dates are valid, store its time line in ``case.index``."""
        self._check_keys(case)
        self._check_save_frequency(case)
        case.index = self._build_index(case)
        learning = case.config.get("learning_config")
        if isinstance(learning, dict):
            self._check_learning(case, learning)
        self._check_markets(case)

    # -- keys and values of the study case --------------------------------------
    def _check_keys(self, case: StudyCase) -> None:
        log, config = case.reporter(), case.config
        for key in REQUIRED_CASE_KEYS:
            if key not in config:
                log.error(f"missing required key '{key}'")
        known = {*REQUIRED_CASE_KEYS, *OPTIONAL_CASE_KEYS, *FILE_KEYS}
        for key in config:
            if key == "learning_mode":
                log.error(
                    "'learning_mode' moved into 'learning_config'. Please adjust your config"
                )
            elif key not in known:
                log.warning(f"unknown key '{key}' is ignored{hint(key, known)}")
        for key, annotation in OPTIONAL_CASE_KEYS.items():
            if key in config and not type_ok(config[key], annotation):
                log.error(describe_wrong_type(key, config[key], annotation))
        for key in FILE_KEYS:
            if key in config and not (
                config[key] is None or isinstance(config[key], str)
            ):
                log.error(f"'{key}' must be a file name or null, got {config[key]!r}")

    def _check_save_frequency(self, case: StudyCase) -> None:
        saved = case.config.get("save_frequency_hours")
        if isinstance(saved, int | float) and not isinstance(saved, bool):
            if saved < 0 or saved != int(saved):
                case.reporter().error(
                    f"'save_frequency_hours' must be null or a positive integer, got {saved!r}"
                )

    def _build_index(self, case: StudyCase) -> pd.DatetimeIndex | None:
        log, config = case.reporter(), case.config
        stamps: dict[str, pd.Timestamp] = {}
        for key in ("start_date", "end_date"):
            if key in config:
                try:
                    stamp = pd.Timestamp(config[key])
                except (ValueError, TypeError):
                    stamp = pd.NaT
                # null and "" give NaT instead of raising
                if pd.isna(stamp):
                    log.error(f"'{key}' is not a valid date: {config[key]!r}")
                else:
                    stamps[key] = stamp
        try:
            freq = pd.tseries.frequencies.to_offset(config["time_step"])
        except (KeyError, ValueError, TypeError):
            if "time_step" in config:
                log.error(
                    f"'time_step' is not a valid frequency like '1h' or '15min': {config['time_step']!r}"
                )
            return None
        if len(stamps) < 2:
            return None
        if stamps["end_date"] <= stamps["start_date"]:
            log.error("'end_date' must be after 'start_date'")
            return None
        return pd.date_range(stamps["start_date"], stamps["end_date"], freq=freq)

    # -- learning_config ---------------------------------------------------------
    def _check_learning(self, case: StudyCase, learning: YamlMapping) -> None:
        log = case.reporter(section="learning_config: ")
        hints = typing.get_type_hints(LearningConfig)
        names = {f.name for f in fields(LearningConfig)}
        for key, value in learning.items():
            if key not in names:
                log.error(f"unknown key '{key}'{hint(key, names)}")
            elif not type_ok(value, hints[key]):
                tip = ""
                if isinstance(value, str) and "e" in value.lower():
                    tip = " (yaml reads '1e-3' as text, write '1.0e-3')"
                message = describe_wrong_type(key, value, hints[key])
                log.error(f"{message}{tip}")
        load_path = learning.get("trained_policies_load_path")
        if isinstance(load_path, str) and not (self.folder.path / load_path).exists():
            log.error(
                f"trained_policies_load_path '{load_path}' does not exist in {self.folder.path}"
            )
        if learning.get("continue_learning") and not load_path:
            log.error("continue_learning needs trained_policies_load_path")

    # -- markets_config ----------------------------------------------------------
    def _check_markets(self, case: StudyCase) -> None:
        markets = case.config.get("markets_config")
        if markets is None:
            return  # reported as missing key
        if not isinstance(markets, dict):
            case.reporter().error(
                "'markets_config' must be a mapping of market ids to their settings"
            )
            return
        if not markets:
            case.reporter().error("'markets_config' must contain at least one market")
        for market_id, params in markets.items():
            self._check_market(
                params, case.reporter(section=f"markets_config.{market_id}: ")
            )

    def _check_market(self, params: YamlValue, log: Reporter) -> None:
        if not isinstance(params, dict):
            return log.error("must be a mapping of settings")
        for key in REQUIRED_MARKET_KEYS:
            if key not in params:
                log.error(f"missing required key '{key}'")
        for key, value in params.items():
            if key not in MARKET_KEYS:
                log.warning(f"unknown key '{key}' is ignored{hint(key, MARKET_KEYS)}")
            elif not type_ok(value, MARKET_KEYS[key]):
                log.error(describe_wrong_type(key, value, MARKET_KEYS[key]))

        mechanism = params.get("market_mechanism")
        mechanisms = self.registry.mechanisms
        if isinstance(mechanism, str) and mechanism not in mechanisms:
            log.error(
                f"unknown market_mechanism '{mechanism}'{hint(mechanism, mechanisms)}. "
                f"Available: {listed(sorted(mechanisms))}"
            )
        if isinstance(params.get("opening_frequency"), str):
            try:
                convert_to_rrule_freq(params["opening_frequency"])
            except (ValueError, KeyError) as e:
                log.error(f"'opening_frequency': {e}")
        if isinstance(params.get("opening_duration"), str):
            self._check_duration(params["opening_duration"], "opening_duration", log)
        for key in ("start_date", "end_date"):
            if params.get(key) is not None:
                try:
                    pd.Timestamp(params[key])
                except (ValueError, TypeError):
                    log.error(f"'{key}' is not a valid date: {params[key]!r}")
        for i, product in enumerate(params.get("products") or []):
            self._check_product(product, log.nested(f"products[{i}]: "))
        network = (params.get("param_dict") or {}).get("network_path")
        if isinstance(network, str) and not (self.folder.path / network).is_dir():
            log.error(
                f"param_dict.network_path '{network}' is not a directory in {self.folder.path}"
            )

    def _check_product(self, product: YamlValue, log: Reporter) -> None:
        if not isinstance(product, dict):
            return log.error("must be a mapping")
        for key in PRODUCT_KEYS:
            if key not in product:
                log.error(f"missing '{key}'")
        for key in set(product) - set(PRODUCT_KEYS):
            log.error(f"has unknown key '{key}'{hint(key, PRODUCT_KEYS)}")
        for key in ("duration", "first_delivery"):
            if key in product:
                self._check_duration(product[key], key, log)
        count = product.get("count")
        if "count" in product and not (isinstance(count, int) and count > 0):
            log.error(f"count must be a positive integer, got {count!r}")

    @staticmethod
    def _check_duration(value: YamlValue, key: str, log: Reporter) -> None:
        try:
            pd.Timedelta(value)
        except (ValueError, TypeError):
            log.error(
                f"'{key}' is not a valid duration like '1h' or '15min': {value!r}"
            )
