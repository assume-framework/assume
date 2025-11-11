# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import asyncio

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
