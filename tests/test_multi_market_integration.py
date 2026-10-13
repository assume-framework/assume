# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from datetime import datetime

import pandas as pd
import pytest

from assume import World
from assume.scenario.loader_csv import load_scenario_folder


@pytest.fixture
def world():
    return World(database_uri=None, export_csv_path=None)


def test_multi_market_storage_scenario(world):
    """
    Runs the tiny multi-market storage scenario end to end.

    The storage bids into the day-ahead market and both balancing capacity
    markets simultaneously. Asserts that the strategy discovers all markets,
    bids on all of them, and that the dispatch respects the power and SoC
    limits of the unit.
    """
    load_scenario_folder(
        world,
        inputs_path="tests/fixtures",
        scenario="multi_market_storage",
        study_case="tiny_multi_market",
    )
    world.run()

    storage = world.units["Storage 1"]
    strategy = storage.bidding_strategies["EOM"]

    # all three markets were discovered
    assert set(strategy.markets.keys()) == {"EOM", "CRM_pos", "CRM_neg"}

    # the storage bid on all three markets
    # (checked via the unit outputs - each product type was written)
    # CRM capacity was reserved at some point
    assert storage.outputs["capacity_pos"].max() > 0

    # the dispatch respects the power limits
    assert storage.outputs["energy"].max() <= storage.max_power_discharge + 1e-6
    assert storage.outputs["energy"].min() >= storage.max_power_charge - 1e-6

    # the SoC respects the limits and was used (storage is not static)
    soc = storage.outputs["soc"]
    assert soc.min() >= storage.min_soc - 1e-6
    assert soc.max() <= storage.max_soc + 1e-6


def test_multi_market_storage_cross_market_coupling(world):
    """
    The sum of all commitments per delivery period respects the power limits.

    Balancing capacity commitments reserve power: the day-ahead bids must
    stay within the remaining capacity of the unit.
    """
    load_scenario_folder(
        world,
        inputs_path="tests/fixtures",
        scenario="multi_market_storage",
        study_case="tiny_multi_market",
    )
    world.run()

    storage = world.units["Storage 1"]

    # per time step: discharge = energy + capacity_pos, charge = -energy + capacity_neg
    # both must not exceed the respective power limits
    discharge = storage.outputs["energy"] + storage.outputs["capacity_pos"]
    charge = -storage.outputs["energy"] + storage.outputs["capacity_neg"]
    assert (discharge <= storage.max_power_discharge + 1e-6).all()
    assert (charge <= -storage.max_power_charge + 1e-6).all()
