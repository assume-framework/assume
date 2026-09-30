# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""End-to-end simulation integration test using ASSUME World with DMAS components.

Validates the full closed-loop simulation:
- PowerplantDmasStrategy (linked & avoided start bidding)
- StorageDmasStrategy (exclusive block bidding)
- ComplexDmasClearingRole (two-sided market clearing with linked & exclusive orders)
- Demand unit providing market demand
- Simulated across a multi-day horizon via World.run()
"""

from datetime import datetime, timedelta

from dateutil import rrule as rr

from assume import World
from assume.common.fast_pandas import FastIndex
from assume.common.forecaster import (
    DemandForecaster,
    PowerplantForecaster,
    UnitForecaster,
)
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.strategies.dmas_powerplant import EnergyOptimizationDmasStrategy
from assume.strategies.dmas_storage import StorageEnergyOptimizationDmasStrategy
from assume.strategies.naive_strategies import EnergyNaiveStrategy
from assume.units import Demand, PowerPlant, Storage


def test_dmas_world_simulation_end_to_end():
    start = datetime(2022, 1, 1, 0, 0)
    end = datetime(2022, 1, 3, 0, 0)
    index = FastIndex(start, end, freq="h")

    world = World(database_uri=None, export_csv_path=None)
    world.setup(
        start=start,
        end=end,
        save_frequency_hours=24,
        simulation_id="test_dmas_sim",
        index=index,
    )

    market_id = "dayahead"
    market_config = MarketConfig(
        market_id=market_id,
        opening_hours=rr.rrule(
            rr.HOURLY,
            interval=24,
            dtstart=start,
            until=end - timedelta(days=1),
            cache=True,
        ),
        opening_duration=timedelta(hours=1),
        market_mechanism="pay_as_clear_complex_dmas",
        market_products=[MarketProduct(timedelta(hours=1), 24, timedelta(hours=1))],
        additional_fields=["block_id", "link", "exclusive_id"],
    )

    mo_id = "market_operator"
    world.add_market_operator(id=mo_id)
    world.add_market(mo_id, market_config)

    # 1. Demand Unit
    world.add_unit_operator("demand_operator")
    demand_forecaster = DemandForecaster(index, demand=-300.0)
    demand_unit = Demand(
        id="city_demand",
        unit_operator="demand_operator",
        min_power=0.0,
        max_power=-1000.0,
        bidding_strategies={market_id: EnergyNaiveStrategy()},
        technology="demand",
        forecaster=demand_forecaster,
    )
    world.add_unit_instance("demand_operator", demand_unit)

    # 2. DMAS PowerPlant Unit
    world.add_unit_operator("pp_operator")
    pp_forecaster = PowerplantForecaster(
        index=index,
        availability=1.0,
        fuel_prices={"gas": 20.0, "co2": 30.0},
        market_prices={market_id: 65.0},
    )
    pp_unit = PowerPlant(
        id="ccgt_plant",
        unit_operator="pp_operator",
        technology="CCGT",
        fuel_type="gas",
        efficiency=0.5,
        emission_factor=0.35,
        cold_start_cost=10.0,
        min_operating_runtime=2,
        min_operating_offtime=2,
        bidding_strategies={market_id: EnergyOptimizationDmasStrategy()},
        max_power=500.0,
        min_power=100.0,
        forecaster=pp_forecaster,
    )
    world.add_unit_instance("pp_operator", pp_unit)

    # 3. DMAS Storage Unit
    world.add_unit_operator("storage_operator")
    # Provide a price spread to encourage storage charging and discharging
    storage_prices = [30.0] * 12 + [80.0] * 12 + [30.0] * 12 + [80.0] * 12
    storage_forecaster = UnitForecaster(
        index=index,
        availability=1.0,
        market_prices={market_id: storage_prices},
    )
    storage_unit = Storage(
        id="battery_storage",
        unit_operator="storage_operator",
        technology="battery",
        bidding_strategies={market_id: StorageEnergyOptimizationDmasStrategy()},
        max_power_charge=-50.0,
        max_power_discharge=50.0,
        capacity=200.0,
        initial_soc=0.5,
        efficiency_charge=0.9,
        efficiency_discharge=0.9,
        forecaster=storage_forecaster,
    )
    world.add_unit_instance("storage_operator", storage_unit)

    world.init_forecasts()
    world.run()

    # Verify units are registered and simulation finished
    assert "city_demand" in world.units
    assert "ccgt_plant" in world.units
    assert "battery_storage" in world.units

    # Verify market operator cleared the 24 market delivery hours
    market_role = world.market_operators[mo_id].roles[0]
    assert len(market_role.results) == 24

    for r in market_role.results:
        assert r["price"] > 0, "Market price should be strictly positive"
        assert r["demand_volume"] >= 300.0, "Base demand should be satisfied"
        assert r["supply_volume"] >= 300.0, "Generation must cover demand"
