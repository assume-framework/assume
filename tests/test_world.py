# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import asyncio

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
