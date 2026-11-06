# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""Unit tests for the building blocks of ``assume.scenario.validation``."""

import logging
from pathlib import Path

import pytest

from assume.scenario.validation.config import ConfigChecker, StudyCase
from assume.scenario.validation.folder import FileRef, Registry, ScenarioFolder
from assume.scenario.validation.report import ValidationReport
from assume.scenario.validation.tables import UnitSignature
from assume.scenario.validation.utils import (
    hint,
    listed,
    quiet_logging,
    type_name,
    type_ok,
)
from assume.units import Demand, PowerPlant

EXAMPLE = Path("examples/inputs/example_01a")


# -- utils ----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value, annotation, expected",
    [
        (1, int, True),
        (1.5, int, True),  # int and float both mean "number"
        (1, float, True),
        ("1e-3", float, False),  # yaml reads 1e-3 as text
        (True, int, False),  # bool is no number
        (None, float | None, True),
        (None, float, False),
        ("x", str, True),
        (1, str, False),
        (True, bool, True),
        (1, bool, False),
        ({}, dict, True),
        ([], list, True),
        ([], dict, False),
        ("anything", None, True),
    ],
)
def test_type_ok(value, annotation, expected):
    assert type_ok(value, annotation) is expected


def test_type_name():
    assert type_name(float | None) == "number | null"
    assert type_name(int) == "number"
    assert type_name(str) == "str"
    assert type_name(None) == "null"


def test_hint():
    known = ["pay_as_clear", "pay_as_bid"]
    assert hint("pay_as_clr", known) == " (did you mean 'pay_as_clear'?)"
    assert hint("something_else", known) == ""


def test_listed_shortens_long_lists():
    assert listed(range(3)) == "0, 1, 2"
    assert listed(range(7)) == "0, 1, 2, 3, 4 and 2 more"


def test_quiet_logging_silences_and_restores(capsys):
    with quiet_logging():
        print("hidden")
        assert logging.root.manager.disable == logging.CRITICAL
    print("visible")
    assert logging.root.manager.disable == logging.NOTSET
    assert capsys.readouterr().out == "visible\n"


# -- report ---------------------------------------------------------------------------------
def test_report_counts_and_summary():
    report = ValidationReport(path="some/dir", cases=["base"])
    assert report.ok
    assert (
        report.summary()
        == "OK: some/dir can be loaded (1 study case(s), 0 error(s), 0 warning(s))"
    )

    report.warning("a.csv", "careful")
    assert report.ok

    report.error("b.csv", "broken")
    assert not report.ok
    assert [str(i) for i in report.errors] == ["ERROR   b.csv: broken"]
    assert [str(i) for i in report.warnings] == ["WARNING a.csv: careful"]
    assert report.summary() == "FAILED: some/dir (1 error(s), 1 warning(s))"


def test_reporter_prefixes_messages():
    report = ValidationReport(path="dir")
    log = report.scope("config.yaml", prefix="[base] ")
    log.error("bad")
    log.nested("markets_config.EOM: ").warning("odd")
    assert [str(i) for i in report.issues] == [
        "ERROR   config.yaml: [base] bad",
        "WARNING config.yaml: [base] markets_config.EOM: odd",
    ]


# -- folder ---------------------------------------------------------------------------------
def test_registry_knows_what_a_scenario_can_refer_to():
    registry = Registry.load()
    assert {"power_plant", "demand", "storage", "exchange"} <= set(registry.unit_types)
    assert "powerplant_energy_naive" in registry.strategies
    assert "pay_as_clear" in registry.mechanisms


def test_resolve_mirrors_the_loader():
    folder = ScenarioFolder(EXAMPLE)
    assert folder.resolve({}, "demand_df") == FileRef(EXAMPLE / "demand_df.csv", False)
    renamed = folder.resolve({"demand_df": "other.csv"}, "demand_df")
    assert renamed == FileRef(EXAMPLE / "other.csv", True)
    assert folder.resolve({"demand_df": None}, "demand_df") == FileRef(None, True)


def test_read_reports_an_invalid_csv_only_once(tmp_path):
    (tmp_path / "broken.csv").write_text("")
    folder = ScenarioFolder(tmp_path)
    assert folder.read(tmp_path / "broken.csv") is None
    assert folder.read(tmp_path / "broken.csv") is None
    assert len(folder.report.errors) == 1


def test_read_caches_files():
    folder = ScenarioFolder(EXAMPLE)
    first = folder.read(EXAMPLE / "demand_units.csv")
    assert folder.read(EXAMPLE / "demand_units.csv") is first


def test_first_visit():
    folder = ScenarioFolder(EXAMPLE)
    assert folder.first_visit(EXAMPLE / "demand_units.csv")
    assert not folder.first_visit(EXAMPLE / "demand_units.csv")
    assert folder.first_visit(EXAMPLE / "powerplant_units.csv")


# -- config ---------------------------------------------------------------------------------
def test_load_cases():
    checker = ConfigChecker(ScenarioFolder(EXAMPLE))
    cases = checker.load_cases()
    assert [c.name for c in cases] == [
        "base",
        "base_with_exchanges",
        "tiny",
        "dam",
        "dam_with_complex_clearing",
    ]
    assert [c.name for c in checker.load_cases("tiny")] == ["tiny"]
    assert checker.load_cases("tinyy") == []
    assert "did you mean 'tiny'" in str(checker.folder.report.errors[0])


def test_check_builds_the_time_line():
    checker = ConfigChecker(ScenarioFolder(EXAMPLE))
    (case,) = checker.load_cases("tiny")
    assert case.index is None
    checker.check(case)
    assert not checker.folder.report.issues
    assert len(case.index) == 25  # one day in hourly steps, both ends included
    assert case.market_ids == {"EOM"}


def test_study_case_marks_messages_with_its_name():
    report = ValidationReport(path="dir")
    case = StudyCase("base", {"markets_config": {"EOM": {}, 2: {}}}, report)
    assert case.market_ids == {"EOM", "2"}
    case.reporter("demand_df.csv").error("bad")
    assert str(report.issues[0]) == "ERROR   demand_df.csv: [base] bad"


# -- tables ---------------------------------------------------------------------------------
def test_unit_signature_reads_the_constructor():
    power_plant = UnitSignature.of(PowerPlant)
    assert {"technology", "max_power"} <= set(power_plant.required)
    assert not {"id", "forecaster", "bidding_strategies"} & set(power_plant.required)
    assert power_plant.accepted["efficiency"] is float
    # accepted through the base classes, too
    assert "node" in power_plant.accepted
    assert "efficiency" not in power_plant.required  # has a default

    assert {"max_power", "min_power"} <= set(UnitSignature.of(Demand).required)
