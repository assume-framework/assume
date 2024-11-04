# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import asyncio

from assume import World
from assume.scenario.loader_csv import load_scenario_folder


def test_world():
    scenario = "example_01a"
    study_case = "base"
    world = World(database_uri=None, export_csv_path=None)
    load_scenario_folder(
        world, inputs_path="examples/inputs", scenario=scenario, study_case=study_case
    )

    assert world.db_uri is None
    assert world.export_csv_path is None

    assert isinstance(world.loop, asyncio.BaseEventLoop)
    assert world.market_operators.keys()
    assert world.markets.keys()
    assert world.unit_operators.keys()
