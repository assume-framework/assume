# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from dateutil import rrule as rr

pytest.importorskip("pypsa")
import pypsa

from assume import World
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.scenario.loader_pypsa import load_pypsa


def build_network(freq: str = "h") -> pypsa.Network:
    network = pypsa.Network()
    network.set_snapshots(pd.date_range("2024-01-01", periods=4, freq=freq))
    network.add("Bus", "bus0")
    network.add("Bus", "bus1")
    # ramp limits in PyPSA are per unit of p_nom and per snapshot
    network.add(
        "Generator",
        "ramped",
        bus="bus0",
        p_nom=100,
        p_min_pu=0.2,
        marginal_cost=10,
        carrier="gas",
        min_up_time=3,
        min_down_time=2,
        ramp_limit_up=0.5,
        ramp_limit_down=0.25,
    )
    # PyPSA default: ramp_limit_up/down are NaN, meaning "no limit"
    network.add(
        "Generator",
        "unramped",
        bus="bus1",
        p_nom=200,
        marginal_cost=20,
        carrier="wind",
        p_max_pu=pd.Series([1.0, 0.5, 0.25, 0.0], index=network.snapshots),
    )
    # the loader only picks up loads with a time-varying p_set
    network.add(
        "Load",
        "load0",
        bus="bus0",
        p_set=pd.Series([50.0, 60.0, 70.0, 40.0], index=network.snapshots),
    )
    # a load with only a static p_set is skipped by the loader
    network.add("Load", "static_load", bus="bus1", p_set=30)
    return network


def run_loader(network: pypsa.Network) -> World:
    index = network.snapshots
    marketdesign = [
        MarketConfig(
            "EOM",
            rr.rrule(rr.HOURLY, dtstart=index[0], until=index[-1]),
            timedelta(hours=1),
            "pay_as_clear",
            [MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
            additional_fields=["node"],
        )
    ]
    strategies = defaultdict(lambda: {"EOM": "powerplant_energy_naive"})
    demand_strategies = defaultdict(lambda: {"EOM": "demand_energy_naive"})
    bidding_strategies = {
        "power_plant": strategies,
        "demand": demand_strategies,
        "storage": strategies,
    }

    world = World(database_uri=None, export_csv_path=None)
    load_pypsa(world, "test", "pypsa", network, marketdesign, bidding_strategies)
    return world


def test_load_pypsa_registers_market():
    world = run_loader(build_network())

    assert list(world.markets) == ["EOM"]


def test_load_pypsa_generator_parameters():
    world = run_loader(build_network())

    unit = world.units["ramped"]
    assert unit.unit_operator == "powerplant_operator"
    assert unit.max_power == 100
    assert unit.min_power == pytest.approx(20)  # p_nom * p_min_pu
    assert unit.node == "bus0"
    assert unit.fuel_type == "gas"
    # marginal_cost is passed as fuel price with efficiency 1
    assert unit.efficiency == 1
    assert np.allclose(unit.forecaster.get_price("gas"), 10)
    # with hourly snapshots, hours equal simulation steps
    assert unit.min_operating_time == 3
    assert unit.min_down_time == 2


def test_load_pypsa_time_varying_availability():
    world = run_loader(build_network())

    unit = world.units["unramped"]
    assert list(unit.forecaster.availability) == [1.0, 0.5, 0.25, 0.0]
    # generators without a p_max_pu time series are always available
    assert np.allclose(world.units["ramped"].forecaster.availability, 1.0)


def test_load_pypsa_demand():
    world = run_loader(build_network())

    unit = world.units["load0"]
    assert unit.unit_operator == "demand_operator"
    assert unit.node == "bus0"
    # demand is negative in ASSUME
    assert unit.max_power == -70
    assert list(unit.forecaster.demand) == [-50.0, -60.0, -70.0, -40.0]


def test_load_pypsa_skips_loads_without_time_series():
    world = run_loader(build_network())

    assert "static_load" not in world.units


def test_load_pypsa_converts_ramp_limits():
    world = run_loader(build_network("h"))

    unit = world.units["ramped"]
    # 0.5 * p_nom per one-hour snapshot -> 50 MW per one-hour simulation step
    assert unit.ramp_up == pytest.approx(50)
    assert unit.ramp_down == pytest.approx(25)


def test_load_pypsa_converts_ramp_limits_sub_hourly():
    world = run_loader(build_network("15min"))

    unit = world.units["ramped"]
    # ramp limits are per snapshot, so with 15 min snapshots and 15 min
    # simulation steps the per-step limit is unchanged
    assert unit.ramp_up == pytest.approx(50)
    assert unit.ramp_down == pytest.approx(25)


def test_load_pypsa_nan_ramp_means_no_limit():
    world = run_loader(build_network("h"))

    unit = world.units["unramped"]
    assert unit.ramp_up is None
    assert unit.ramp_down is None
