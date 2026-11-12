# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from assume.scenario.validation import verify_scenario_folder
from assume_cli.cli import cli

EXAMPLE = Path("examples/inputs/example_01a")


@pytest.fixture
def scenario(tmp_path) -> Path:
    """A copy of example_01a which can be broken by the tests."""
    target = tmp_path / "scenario"
    shutil.copytree(EXAMPLE, target)
    return target


def edit_config(scenario: Path, case: str = "tiny", **changes) -> None:
    path = scenario / "config.yaml"
    config = yaml.safe_load(path.read_text())
    config[case].update(changes)
    path.write_text(yaml.safe_dump(config))


def edit_csv(scenario: Path, name: str, old: str, new: str) -> None:
    path = scenario / name
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))


def edit_nested_config(scenario: Path, change, case: str = "tiny") -> None:
    """Apply ``change(case_config)`` to a study case of the config."""
    path = scenario / "config.yaml"
    config = yaml.safe_load(path.read_text())
    change(config[case])
    path.write_text(yaml.safe_dump(config))


def edit_df(scenario: Path, name: str, change) -> None:
    """Apply ``change(dataframe)`` to a csv file (the first column is the index)."""
    path = scenario / name
    df = pd.read_csv(path, index_col=0)
    change(df)
    df.to_csv(path)


def messages(report, severity="error") -> str:
    return "\n".join(str(i) for i in report.issues if i.severity == severity)


def test_valid_scenario_loads(scenario):
    report = verify_scenario_folder(scenario, study_case="tiny")
    assert report.ok, messages(report)
    assert not report.issues


def test_missing_directory_and_config(tmp_path):
    assert "not a directory" in messages(verify_scenario_folder(tmp_path / "nope"))
    assert "not found" in messages(verify_scenario_folder(tmp_path))


def test_invalid_yaml(scenario):
    (scenario / "config.yaml").write_text("base: [unclosed")
    assert "cannot be parsed" in messages(verify_scenario_folder(scenario))


def test_unknown_study_case(scenario):
    report = verify_scenario_folder(scenario, study_case="tinny")
    assert "did you mean 'tiny'" in messages(report)


def test_required_file_missing(scenario):
    (scenario / "powerplant_units.csv").unlink()
    report = verify_scenario_folder(scenario, dry_run=False)
    assert "powerplant_units.csv: required file not found" in messages(report)


def test_configured_file_missing(scenario):
    edit_config(scenario, storage_units="nonexistent.csv")
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert "'storage_units' points to missing file" in messages(report)


def test_config_keys_and_types(scenario):
    config = yaml.safe_load((scenario / "config.yaml").read_text())
    del config["tiny"]["time_step"]
    config["tiny"]["seeed"] = 1
    config["tiny"]["save_frequency_hours"] = "daily"
    config["tiny"]["markets_config"]["EOM"]["market_mechanism"] = "pay_as_clr"
    config["tiny"]["markets_config"]["EOM"]["maximum_bid_price"] = "high"
    (scenario / "config.yaml").write_text(yaml.safe_dump(config))

    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    errors, warnings = messages(report), messages(report, "warning")
    assert "missing required key 'time_step'" in errors
    assert "'save_frequency_hours' must be number | null" in errors
    assert (
        "unknown market_mechanism 'pay_as_clr' (did you mean 'pay_as_clear'?)" in errors
    )
    assert "'maximum_bid_price' must be number | null, got 'high'" in errors
    assert "unknown key 'seeed'" in warnings


def test_end_before_start(scenario):
    edit_config(scenario, end_date="2018-01-01 00:00")
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert "'end_date' must be after 'start_date'" in messages(report)


@pytest.mark.parametrize("empty", [None, ""])
def test_empty_date(scenario, empty):
    edit_config(scenario, start_date=empty)
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert f"'start_date' is not a valid date: {empty!r}" in messages(report)


def test_learning_config(scenario):
    edit_config(
        scenario,
        learning_config={"learning_rate": "1e-3", "learnign_mode": True},
    )
    errors = messages(
        verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    )
    assert "learning_config: 'learning_rate' must be number" in errors
    assert "write '1.0e-3'" in errors
    assert "unknown key 'learnign_mode'" in errors


def test_csv_column_type(scenario):
    edit_csv(scenario, "powerplant_units.csv", "1000.0,200.0,0.3", "1000.0,200.0,high")
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert "column 'efficiency' must be a number, got Unit 1='high'" in messages(report)


def test_csv_missing_and_unknown_columns(scenario):
    edit_csv(scenario, "powerplant_units.csv", "max_power,", "max_powr,")
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert "missing required column 'max_power'" in messages(report)
    assert "unknown column 'max_powr'" in messages(report, "warning")


def test_csv_duplicate_unit_names(scenario):
    edit_csv(scenario, "powerplant_units.csv", "Unit 2,", "Unit 1,")
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert "duplicate unit names: Unit 1" in messages(report)


def test_unknown_bidding_strategy(scenario):
    edit_csv(scenario, "demand_units.csv", "demand_energy_naive", "demand_naive")
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert "unknown bidding strategy 'demand_naive'" in messages(report)


def test_unparseable_csv(scenario):
    (scenario / "demand_units.csv").write_text("")
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert "demand_units.csv: is not a valid csv file" in messages(report)


def test_timeseries_too_short(scenario):
    lines = (scenario / "demand_df.csv").read_text().splitlines()
    (scenario / "demand_df.csv").write_text("\n".join(lines[:5]) + "\n")
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert "does not cover the simulation time line" in messages(report)


def test_missing_exchange_columns(scenario):
    edit_csv(scenario, "exchanges_df.csv", "exchange_import", "exchange_in")
    report = verify_scenario_folder(
        scenario, study_case="base_with_exchanges", dry_run=False
    )
    assert "missing column 'exchange_import'" in messages(report)


def test_dry_run_reports_what_static_checks_cannot_know(scenario):
    # a strategy which exists but does not fit the unit type is only found when the unit is built
    edit_csv(
        scenario,
        "powerplant_units.csv",
        "powerplant_energy_naive",
        "storage_energy_heuristic_flexable",
    )
    report = verify_scenario_folder(scenario, study_case="tiny")
    assert "cannot be loaded into ASSUME" in messages(report)
    assert "is not a MinMaxStrategy" in messages(report)


def test_demand_with_mixed_power_signs(scenario):
    # the loader rejects it, so the static stage reports it (once) for the unit
    edit_csv(scenario, "demand_units.csv", "1000000,0,", "-1,5,")
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert messages(report).count("min_power and max_power must both be") == 1
    assert "demand_EOM: min_power and max_power" in messages(report)


def test_dry_run_does_not_write_into_scenario(scenario):
    edit_config(scenario, save_forecasts=True)
    before = sorted(p.name for p in scenario.iterdir())
    assert verify_scenario_folder(scenario, study_case="tiny").ok
    assert sorted(p.name for p in scenario.iterdir()) == before


def test_cli_verify_exit_codes(scenario, capsys):
    with pytest.raises(SystemExit) as ok:
        cli(["--verify", str(scenario), "-c", "tiny"])
    assert ok.value.code == 0
    assert "OK:" in capsys.readouterr().out

    (scenario / "demand_units.csv").unlink()
    with pytest.raises(SystemExit) as failed:
        cli(["--verify", str(scenario), "-c", "tiny"])
    assert failed.value.code == 1
    assert "demand_units.csv: required file not found" in capsys.readouterr().out


# -- config.yaml ----------------------------------------------------------------------------
def test_config_must_be_mapping_of_cases(scenario):
    (scenario / "config.yaml").write_text("- just\n- a list\n")
    assert "must be a mapping of study case names" in messages(
        verify_scenario_folder(scenario)
    )
    (scenario / "config.yaml").write_text("base: 5\n")
    assert "[base] must be a mapping, got int" in messages(
        verify_scenario_folder(scenario)
    )


def test_learning_mode_moved(scenario):
    edit_config(scenario, learning_mode=True)
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert "'learning_mode' moved into 'learning_config'" in messages(report)


def test_invalid_time_settings(scenario):
    edit_config(scenario, time_step="often", start_date="not a date")
    errors = messages(verify_scenario_folder(scenario, "tiny", dry_run=False))
    assert "'time_step' is not a valid frequency" in errors
    assert "'start_date' is not a valid date" in errors


def test_file_key_must_be_name_or_null(scenario):
    edit_config(scenario, demand_df=5)
    errors = messages(verify_scenario_folder(scenario, "tiny", dry_run=False))
    assert "'demand_df' must be a file name or null" in errors


def test_disabled_file_is_not_checked(scenario):
    edit_config(scenario, demand_df=None)
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert report.ok, messages(report)
    assert not report.issues


def test_learning_config_paths(scenario):
    edit_config(
        scenario,
        learning_config={
            "continue_learning": True,
            "trained_policies_load_path": "nowhere",
        },
    )
    errors = messages(verify_scenario_folder(scenario, "tiny", dry_run=False))
    assert "trained_policies_load_path 'nowhere' does not exist" in errors

    edit_config(scenario, learning_config={"continue_learning": True})
    errors = messages(verify_scenario_folder(scenario, "tiny", dry_run=False))
    assert "continue_learning needs trained_policies_load_path" in errors


# -- markets_config -------------------------------------------------------------------------
def edit_market(scenario: Path, change) -> None:
    edit_nested_config(scenario, lambda case: change(case["markets_config"]["EOM"]))


def test_markets_config_structure(scenario):
    edit_nested_config(scenario, lambda case: case.update(markets_config={}))
    errors = messages(verify_scenario_folder(scenario, "tiny", dry_run=False))
    assert "'markets_config' must contain at least one market" in errors

    edit_nested_config(scenario, lambda case: case.update(markets_config=["EOM"]))
    errors = messages(verify_scenario_folder(scenario, "tiny", dry_run=False))
    assert "'markets_config' must be a mapping of market ids" in errors

    edit_nested_config(scenario, lambda case: case.update(markets_config={"EOM": 1}))
    errors = messages(verify_scenario_folder(scenario, "tiny", dry_run=False))
    assert "markets_config.EOM: must be a mapping of settings" in errors


def test_market_missing_keys_and_durations(scenario):
    def change(market):
        del market["operator"]
        market["opening_frequency"] = "5x"
        market["opening_duration"] = "soon"
        market["start_date"] = "never"

    edit_market(scenario, change)
    errors = messages(verify_scenario_folder(scenario, "tiny", dry_run=False))
    assert "markets_config.EOM: missing required key 'operator'" in errors
    assert "'opening_frequency': Frequency '5x' not supported" in errors
    assert "'opening_duration' is not a valid duration" in errors
    assert "'start_date' is not a valid date: 'never'" in errors


def test_market_products(scenario):
    def change(market):
        market["products"] = [
            {"duration": "1h", "count": 0, "first_delivery": "1h"},
            {"duration": "forever", "count": 1},
            {"duration": "1h", "count": 1, "first_delivery": "1h", "cuont": 2},
            "1h",
        ]

    edit_market(scenario, change)
    errors = messages(verify_scenario_folder(scenario, "tiny", dry_run=False))
    assert "products[0]: count must be a positive integer, got 0" in errors
    assert "products[1]: 'duration' is not a valid duration" in errors
    assert "products[1]: missing 'first_delivery'" in errors
    assert "products[2]: has unknown key 'cuont' (did you mean 'count'?)" in errors
    assert "products[3]: must be a mapping" in errors


def test_market_unknown_key_and_network_path(scenario):
    def change(market):
        market["maximum_volume"] = 5
        market["param_dict"] = {"network_path": "grid"}

    edit_market(scenario, change)
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert "unknown key 'maximum_volume'" in messages(report, "warning")
    assert "network_path 'grid' is not a directory" in messages(report)


# -- unit csv files -------------------------------------------------------------------------
def test_csv_without_rows(scenario):
    header = (scenario / "powerplant_units.csv").read_text().splitlines()[0]
    (scenario / "powerplant_units.csv").write_text(header + "\n")
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert "powerplant_units.csv: contains no rows" in messages(report)


def test_powerplant_rows_are_validated_with_the_model(scenario):
    edit_df(
        scenario,
        "powerplant_units.csv",
        lambda df: (
            df.__setitem__("efficiency", [0.3, -0.5, 0.4, 0.6]),
            df.__setitem__("min_power", [200.0, 200.0, 2000.0, 200.0]),
        ),
    )
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    errors = messages(report)
    assert (
        "Unit 2: efficiency=-0.5: Input should be greater than or equal to 0 (column 'efficiency')"
        in errors
    )
    assert (
        "Unit 3: min_power=2000.0 must be <= max_power=1000.0 (column 'min_power')"
        in errors
    )
    assert "Unit 1" not in errors and "Unit 4" not in errors


def test_powerplant_model_does_not_repeat_type_errors(scenario):
    edit_csv(scenario, "powerplant_units.csv", "1000.0,200.0,0.3", "1000.0,200.0,high")
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert messages(report).count("efficiency") == 1


def test_csv_bool_column(scenario):
    edit_df(
        scenario,
        "powerplant_units.csv",
        lambda df: df.__setitem__("partial_load_eff", ["maybe", True, False, True]),
    )
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert "column 'partial_load_eff' must be True/False (or 1/0)" in messages(report)


def test_bidding_column_of_unknown_market(scenario):
    edit_df(
        scenario,
        "demand_units.csv",
        lambda df: df.rename(columns={"bidding_EOM": "bidding_EOMM"}, inplace=True),
    )
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert "refers to market 'EOMM'" in messages(report, "warning")
    assert "(did you mean 'EOM'?)" in messages(report, "warning")


def test_units_without_bidding_strategy(scenario):
    edit_df(
        scenario,
        "powerplant_units.csv",
        lambda df: df.__setitem__(
            "bidding_EOM", [None] + ["powerplant_energy_naive"] * 3
        ),
    )
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert "never bid: Unit 1" in messages(report, "warning")

    edit_df(
        scenario,
        "powerplant_units.csv",
        lambda df: df.drop(columns="bidding_EOM", inplace=True),
    )
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert "no 'bidding_<market>' column" in messages(report, "warning")


def test_shared_csv_is_checked_once(scenario):
    edit_csv(scenario, "powerplant_units.csv", "Unit 2,", "Unit 1,")
    report = verify_scenario_folder(scenario, dry_run=False)  # all five study cases
    assert len(report.cases) == 5
    assert messages(report).count("duplicate unit names") == 1


def test_unit_operators(scenario):
    (scenario / "unit_operators.csv").write_text(
        "name,bidding_EOM,colour\nOperator 1,units_operator_direct,red\n"
        "Operator 2,no_such_strategy,blue\n"
    )
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert "unknown column 'colour' is ignored" in messages(report, "warning")
    assert "Operator 2: unknown bidding strategy 'no_such_strategy'" in messages(report)
    assert "Operator 1" not in messages(report)


def test_dsm_units(scenario):
    (scenario / "dsm.csv").write_text(
        "name,unit_type,technology,unit_operator,bidding_EOM\n"
        "A,nonsense,heat_pump,op,demand_energy_naive\n"
        "A,nonsense,boiler,op,\n"
    )
    edit_config(scenario, industrial_dsm_units="dsm.csv")
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert "unknown unit_type 'nonsense'" in messages(report)
    # one row per technology: repeated names and a strategy on only one row are fine
    assert "duplicate" not in messages(report)
    assert "never bid" not in messages(report, "warning")

    (scenario / "dsm.csv").write_text("name,technology\nA,heat_pump\n")
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert "dsm.csv: missing required column 'unit_type'" in messages(report)


# -- time series ----------------------------------------------------------------------------
def test_timeseries_values(scenario):
    edit_csv(scenario, "demand_df.csv", "2196.6", "abc")
    edit_df(scenario, "demand_df.csv", lambda df: df.__setitem__("empty", None))
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert "[tiny] non-numeric values in: demand_EOM" in messages(report)
    assert "[tiny] columns without any value: empty" in messages(report, "warning")


def test_fuel_prices_too_short(scenario):
    # several rows are a time series, which the loader drops (all prices 0) if it is too short
    (scenario / "fuel_prices_df.csv").write_text(
        "datetime,uranium\n2019-01-01 00:00,1\n2019-01-01 01:00,2\n"
    )
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert "fuel_prices_df.csv: [tiny] does not cover the simulation time line" in (
        messages(report)
    )


def test_timeseries_needs_datetimes(scenario):
    (scenario / "fuel_prices_df.csv").write_text("fuel,uranium\nfoo,1\nbar,2\n")
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert "first column must contain datetimes" in messages(report)


# -- how the files fit together -------------------------------------------------------------
def test_demand_series_and_units_must_match(scenario):
    edit_df(scenario, "demand_df.csv", lambda df: df.__setitem__("other", 1.0))
    with open(scenario / "demand_units.csv", "a") as f:
        f.write("demand_2,inflex_demand,demand_energy_naive,1000,0,eom_de\n")
    warnings = messages(
        verify_scenario_folder(scenario, study_case="tiny", dry_run=False), "warning"
    )
    assert "columns without a unit in demand_units: other" in warnings
    assert "no demand time series for demand_2, filled with zeros" in warnings


def test_availability_needs_matching_units(scenario):
    index = pd.date_range("2019-01-01", "2019-01-02", freq="1h", name="datetime")
    pd.DataFrame({"ghost": 1.0}, index=index).to_csv(scenario / "availability_df.csv")
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert report.ok, messages(report)
    assert "columns without a matching unit: ghost" in messages(report, "warning")


def test_fuel_type_typo(scenario):
    edit_csv(scenario, "powerplant_units.csv", "naive,uranium", "naive,uraniun")
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert "fuel_type 'uraniun' of a power plant has no price" in messages(
        report, "warning"
    )
    assert "(did you mean 'uranium'?)" in messages(report, "warning")


def test_fuel_without_price_is_fine(scenario):
    edit_csv(scenario, "powerplant_units.csv", "naive,uranium", "naive,sunshine")
    report = verify_scenario_folder(scenario, study_case="tiny", dry_run=False)
    assert not report.issues


def test_exchange_units_need_exchanges_df(scenario):
    (scenario / "exchanges_df.csv").unlink()
    report = verify_scenario_folder(
        scenario, study_case="base_with_exchanges", dry_run=False
    )
    assert "required by exchange_units but missing" in messages(report)


# -- shipped examples -----------------------------------------------------------------------
@pytest.mark.parametrize(
    "example",
    ["example_01a", "example_01b", "example_01e"],
)
def test_examples_pass_static_checks(example):
    report = verify_scenario_folder(Path("examples/inputs") / example, dry_run=False)
    assert report.ok, messages(report)


# -- learning examples
@pytest.mark.require_learning
@pytest.mark.parametrize(
    "example",
    ["example_02a", "example_02b", "example_02c"],
)
def test_learning_examples_pass_static_checks(example):
    report = verify_scenario_folder(Path("examples/inputs") / example, dry_run=False)
    assert report.ok, messages(report)


# -- parameter models of the other units -----------------------------------------------------
@pytest.fixture
def full_scenario(tmp_path) -> Path:
    """A copy of example_03 which has storage, demand and exchange units."""
    target = tmp_path / "scenario"
    shutil.copytree(Path("examples/inputs/example_03"), target)
    return target


def test_storage_demand_and_exchange_rows_are_validated(full_scenario):
    def break_storage(df):
        df[["efficiency_charge", "min_soc", "max_soc"]] = df[
            ["efficiency_charge", "min_soc", "max_soc"]
        ].astype(float)
        df.loc[df.index[0], "efficiency_charge"] = 1.5
        df.loc[df.index[1], ["min_soc", "max_soc"]] = [0.9, 0.1]

    def break_demand(df):
        df["elasticity_model"] = ["isoelastic"] + [None] * (len(df) - 1)
        df["num_bids"] = [3] + [None] * (len(df) - 1)

    edit_df(full_scenario, "storage_units.csv", break_storage)
    edit_df(full_scenario, "demand_units.csv", break_demand)
    edit_df(
        full_scenario,
        "exchange_units.csv",
        lambda df: df.__setitem__("price_import", ["high"] + [0] * (len(df) - 1)),
    )
    report = verify_scenario_folder(full_scenario, dry_run=False)
    errors = messages(report)
    storage = pd.read_csv(full_scenario / "storage_units.csv", index_col=0)
    demand = pd.read_csv(full_scenario / "demand_units.csv", index_col=0)
    exchange = pd.read_csv(full_scenario / "exchange_units.csv", index_col=0)
    assert (
        f"{storage.index[0]}: efficiency_charge=1.5: must be between 0 and 1 (column 'efficiency_charge')"
        in errors
    )
    assert f"{storage.index[1]}: max_soc=0.1 must be >= min_soc=0.9" in errors
    assert (
        f"{demand.index[0]}: 'elasticity' parameter must be given and negative for isoelastic demand (column 'elasticity')"
        in errors
    )
    assert (
        f"column 'price_import' must be a number, got {exchange.index[0]}='high'"
        in errors
    )
    # every problem is reported once, also the not-a-number cell which model and type check both see
    assert errors.count("price_import") == 1
    assert errors.count("efficiency_charge=1.5") == 1
    assert errors.count("max_soc=0.1") == 1
    assert errors.count("'elasticity' parameter") == 1
    # kwargs-only parameters are known columns
    assert "elasticity_model" not in messages(report, "warning")
    assert "num_bids" not in messages(report, "warning")


def test_storage_charge_power_sign_follows_the_loader(full_scenario):
    # the loader negates the charge powers, so positive values in the csv are the convention
    edit_df(
        full_scenario,
        "storage_units.csv",
        lambda df: df.__setitem__("min_power_charge", [-1.0] + [30.0] * (len(df) - 1)),
    )
    report = verify_scenario_folder(full_scenario, dry_run=False)
    errors = messages(report)
    # every row ends with min_power_charge <= 0, and rows with |max| < |min| are the only invalid ones
    storage = pd.read_csv(full_scenario / "storage_units.csv", index_col=0)
    too_small = storage.index[
        storage["max_power_charge"].abs() < storage["min_power_charge"].abs()
    ]
    assert len(too_small) > 0
    for unit in too_small:
        assert f"{unit}: max_power_charge=" in errors
    assert errors.count("max_power_charge=") == len(too_small)


def test_demand_power_sign_follows_the_loader(full_scenario):
    # positive demand power is fine for the loader, as it negates both columns
    edit_df(
        full_scenario,
        "demand_units.csv",
        lambda df: df.__setitem__("max_power", [5.0] * len(df)),
    )
    report = verify_scenario_folder(full_scenario, dry_run=False)
    assert "demand_units.csv" not in messages(report)
