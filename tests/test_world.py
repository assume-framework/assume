# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import asyncio

import numpy as np
import pandas as pd
import pytest

from assume import World
from assume.common.forecaster import DemandForecaster
from assume.scenario.loader_csv import load_scenario_folder
from assume.units.demand import Demand
from tests.utils import index, setup_simple_world


def test_world_scenario():
    world = World(database_uri=None, export_csv_path=None)
    load_scenario_folder(
        world, inputs_path="examples/inputs", scenario="example_01a", study_case="base"
    )

    assert world.db_uri is None
    assert world.export_csv_path is None

    assert isinstance(world.loop, asyncio.BaseEventLoop)
    assert world.market_operators.keys()
    assert world.markets.keys()
    assert world.unit_operators.keys()


def test_world_operators_default():
    world = setup_simple_world()
    world.add_unit_operator("test_operator")
    world.add_unit(
        id="test_unit",
        unit_type="demand",
        unit_operator_id="test_operator",
        unit_params={
            "min_power": 0,
            "max_power": -1000,
            "technology": "demand",
            "bidding_strategies": {},
        },
        forecaster=DemandForecaster(index, demand=-100),
    )
    assert "test_operator" in world.unit_operators
    assert len(world.unit_operators["test_operator"].units) == 1


def test_world_operators_by_instance():
    world = setup_simple_world()
    world.add_unit_operator("test_operator")
    world.add_unit_instance(
        operator_id="test_operator",
        unit=Demand(
            id="test_unit",
            unit_operator="test_operator",
            min_power=0,
            max_power=-1000,
            technology="demand",
            bidding_strategies={},
            forecaster=DemandForecaster(index, demand=-100),
        ),
    )
    assert "test_operator" in world.unit_operators
    assert len(world.unit_operators["test_operator"].units) == 1


def test_world_export(tmp_path):
    world = World(database_uri=None, export_csv_path=None)
    load_scenario_folder(
        world, inputs_path="examples/inputs", scenario="example_01a", study_case="base"
    )

    # Export it
    world.export(scenario_save_path=tmp_path, study_case="base")

    # Check that export folder exists
    exported_dir = tmp_path / "example_01a_base"
    assert exported_dir.exists()
    assert (exported_dir / "config.yaml").exists()
    assert (exported_dir / "powerplant_units.csv").exists()
    assert (exported_dir / "demand_units.csv").exists()
    assert (exported_dir / "demand_df.csv").exists()

    # Load it back
    world2 = World(database_uri=None, export_csv_path=None)
    load_scenario_folder(
        world2,
        inputs_path=str(tmp_path),
        scenario="example_01a_base",
        study_case="base",
    )

    assert len(world2.units) == len(world.units)
    assert len(world2.markets) == len(world.markets)


def test_world_export_forecasts(tmp_path):
    """
    Round-trip the forecast setup of the ``forecast_export`` fixture through an export.

    The fixture draws its forecast algorithms from all three sources at once:
        config.yaml         -> defaults for every unit
        unit csv            -> per-unit override of a single key (Unit 2, Unit 3)
        forecasts_df.csv    -> a given ``price_EOM`` series that supersedes the
                               calculated price forecast

    After exporting and loading again, every forecaster must end up with the same
    algorithms and the same base forecasts, i.e. ``price_EOM`` still comes from
    forecasts_df.csv and ``residual_load`` is still recalculated by its algorithm.
    """
    world = World(database_uri=None, export_csv_path=None)
    load_scenario_folder(
        world,
        inputs_path="tests/fixtures",
        scenario="forecast_export",
        study_case="base",
    )

    world.export(scenario_save_path=tmp_path, study_case="base")

    exported_dir = tmp_path / "forecast_export_base"
    assert (exported_dir / "forecasts_df.csv").exists()

    world2 = World(database_uri=None, export_csv_path=None)
    load_scenario_folder(
        world2,
        inputs_path=str(tmp_path),
        scenario="forecast_export_base",
        study_case="base",
    )

    assert world2.units.keys() == world.units.keys()

    # 1. the algorithms of each unit survive the export, no matter where they came from
    config_algorithms = {
        "price": "price_naive_forecast",
        "residual_load": "residual_load_naive_forecast",
        "preprocess_price": "price_default",
        "preprocess_residual_load": "residual_load_default",
        "update_price": "price_default",
        "update_residual_load": "residual_load_default",
    }
    expected_algorithms = {
        # no override, so purely the config defaults
        "Unit 1": config_algorithms,
        "demand_EOM": config_algorithms,
        # unit csv overrides a single key each
        "Unit 2": config_algorithms | {"price": "price_keep_given"},
        "Unit 3": config_algorithms | {"residual_load": "residual_load_keep_given"},
    }
    for unit_id, algorithms in expected_algorithms.items():
        assert world.units[unit_id].forecaster.forecast_algorithms == algorithms
        assert world2.units[unit_id].forecaster.forecast_algorithms == algorithms

    # 2. the resulting base forecasts are identical before and after the export
    for unit_id in world.units:
        forecaster, reloaded = (
            world.units[unit_id].forecaster,
            world2.units[unit_id].forecaster,
        )
        for name in ("price", "residual_load"):
            forecast = getattr(forecaster, name)
            reloaded_forecast = getattr(reloaded, name)
            assert forecast.keys() == reloaded_forecast.keys()
            for market_id, series in forecast.items():
                assert np.allclose(series, reloaded_forecast[market_id]), (
                    f"{unit_id}: {name}[{market_id}] changed during export"
                )

    # 3. forecasts_df.csv was applied wherever it provides a column ...
    given_price = pd.read_csv(
        exported_dir / "forecasts_df.csv", index_col=0, parse_dates=True
    )["price_EOM"]
    for unit_id in ("Unit 1", "Unit 3", "demand_EOM"):
        assert np.allclose(world2.units[unit_id].forecaster.price["EOM"], given_price)

    # ... but never overrules a unit that opted out of a calculated price forecast
    assert np.allclose(world2.units["Unit 2"].forecaster.price["EOM"], 50.0)

    # 4. and the algorithms were applied wherever forecasts_df.csv provides no column.
    # forecasts_df.csv holds no residual_load, so residual_load_naive_forecast runs and
    # (without renewable feed-in) reproduces the demand profile
    demand = -world2.units["demand_EOM"].forecaster.demand
    for unit_id in ("Unit 1", "Unit 2", "demand_EOM"):
        assert np.allclose(
            world2.units[unit_id].forecaster.residual_load["EOM"], demand
        )

    # residual_load_keep_given calculates nothing, so Unit 3 has no residual load at all
    assert world2.units["Unit 3"].forecaster.residual_load == {}


def test_world_pypsa_export(tmp_path):
    pytest.importorskip("pypsa")
    from collections import defaultdict
    from datetime import timedelta

    import pypsa
    from dateutil import rrule as rr

    from assume.common.market_objects import MarketConfig, MarketProduct
    from assume.scenario.loader_pypsa import load_pypsa

    network = pypsa.examples.ac_dc_meshed()
    # Downsample snapshot range to keep test execution fast
    network.snapshots = network.snapshots[:3]

    world = World(database_uri=None, export_csv_path=None)
    scenario = "world_pypsa"
    study_case = "ac_dc_meshed_test"
    market_mechanism = "complex_clearing"

    start = network.snapshots[0] - timedelta(hours=1)
    end = network.snapshots[-1]

    marketdesign = [
        MarketConfig(
            f"{market_mechanism}_market",
            rr.rrule(
                rr.HOURLY, interval=1, dtstart=start + timedelta(hours=1), until=end
            ),
            timedelta(hours=1),
            market_mechanism,
            [MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
            additional_fields=["node", "max_power", "min_power", "bid_type"],
            maximum_bid_volume=1e9,
            maximum_bid_price=1e9,
            param_dict={"log_flows": True},
        )
    ]

    default_strategies = {mc.market_id: "demand_energy_naive" for mc in marketdesign}

    bidding_strategies = {
        "power_plant": defaultdict(lambda: default_strategies),
        "demand": defaultdict(
            lambda: {mc.market_id: "demand_energy_naive" for mc in marketdesign}
        ),
        "storage": defaultdict(lambda: default_strategies),
    }

    load_pypsa(world, scenario, study_case, network, marketdesign, bidding_strategies)

    # Export it
    world.export(scenario_save_path=tmp_path, study_case="base")

    # Check that export folder exists
    exported_dir = tmp_path / f"{scenario}_{study_case}"
    assert exported_dir.exists()
    assert (exported_dir / "config.yaml").exists()
    assert (exported_dir / "powerplant_units.csv").exists()
    assert (exported_dir / "demand_units.csv").exists()

    # Load it back in a fresh world using loader_csv
    world2 = World(database_uri=None, export_csv_path=None)
    load_scenario_folder(
        world2,
        inputs_path=str(tmp_path),
        scenario=f"{scenario}_{study_case}",
        study_case="base",
    )

    assert len(world2.units) == len(world.units)
    assert len(world2.markets) == len(world.markets)
