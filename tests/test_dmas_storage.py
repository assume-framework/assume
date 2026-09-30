# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import pandas as pd
import pytest
from dateutil import rrule as rr

from assume.common.forecaster import UnitForecaster
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.utils import get_available_products
from assume.strategies.dmas_storage import StorageEnergyOptimizationDmasStrategy
from assume.units import Storage

from .utils import get_test_prices


@pytest.fixture
def storage_unit() -> Storage:
    index = pd.date_range("2022-01-01", periods=4, freq="h")
    forecaster = UnitForecaster(index, market_prices={"EOM": 50}, availability=1)

    return Storage(
        id="Test_Storage",
        unit_operator="TestOperator",
        technology="TestTechnology",
        bidding_strategies={"EOM": StorageEnergyOptimizationDmasStrategy()},
        forecaster=forecaster,
        max_power_charge=-100,
        max_power_discharge=100,
        capacity=1000,
        initial_soc=0.5,
        efficiency_charge=0.9,
        efficiency_discharge=0.95,
        ramp_down_charge=-50,
        ramp_down_discharge=50,
        ramp_up_charge=-60,
        ramp_up_discharge=60,
        additional_cost_charge=3,
        additional_cost_discharge=4,
    )


@pytest.fixture
def storage_day() -> Storage:
    periods = 48
    index = pd.date_range("2022-01-01", periods=periods, freq="h")

    prices = get_test_prices(periods)
    ff = UnitForecaster(index, market_prices={"EOM": prices["power"]}, availability=1)
    return Storage(
        id="Test_Storage",
        unit_operator="TestOperator",
        technology="TestTechnology",
        bidding_strategies={"EOM": StorageEnergyOptimizationDmasStrategy()},
        max_power_charge=-100,
        max_power_discharge=100,
        capacity=1000,
        initial_soc=0.5,
        efficiency_charge=0.9,
        efficiency_discharge=0.95,
        ramp_down_charge=-50,
        ramp_down_discharge=50,
        ramp_up_charge=-60,
        ramp_up_discharge=60,
        additional_cost_charge=3,
        additional_cost_discharge=4,
        forecaster=ff,
    )


def test_dmas_str_init(storage_unit):
    strategy = StorageEnergyOptimizationDmasStrategy()
    hour_count = len(storage_unit.index)

    strategy.build_model(
        storage_unit,
        datetime(2022, 1, 1),
        hour_count,
    )


def test_dmas_calc(storage_unit):
    strategy = StorageEnergyOptimizationDmasStrategy()
    hour_count = len(storage_unit.index) // 2

    mc = MarketConfig(
        market_id="EOM",
        opening_hours=rr.rrule(rr.HOURLY),
        opening_duration=timedelta(hours=1),
        market_mechanism="not needed",
        market_products=[
            MarketProduct(timedelta(hours=1), hour_count, timedelta(hours=0))
        ],
        additional_fields=["exclusive_id"],
    )
    start = storage_unit.index[0]
    products = get_available_products(mc.market_products, start)
    orderbook = strategy.calculate_bids(
        storage_unit, market_config=mc, product_tuples=products
    )
    assert orderbook
    exclusive_ids = {o["exclusive_id"] for o in orderbook}
    assert exclusive_ids


def test_dmas_day(storage_day):
    strategy = StorageEnergyOptimizationDmasStrategy()
    hour_count = len(storage_day.index) // 2
    assert hour_count == 24

    mc = MarketConfig(
        market_id="EOM",
        opening_hours=rr.rrule(rr.HOURLY),
        opening_duration=timedelta(hours=1),
        market_mechanism="not needed",
        market_products=[
            MarketProduct(timedelta(hours=1), hour_count, timedelta(hours=0))
        ],
        additional_fields=["exclusive_id"],
    )
    start = storage_day.index[0]
    products = get_available_products(mc.market_products, start)
    orderbook = strategy.calculate_bids(
        storage_day, market_config=mc, product_tuples=products
    )
    assert orderbook
    exclusive_ids = {o["exclusive_id"] for o in orderbook}
    assert exclusive_ids


def test_storage_economic_arbitrage(storage_unit):
    """
    Test that storage correctly charges in cheap hours and discharges in expensive hours,
    satisfies volume limits, and ends at half-capacity SoC.
    """
    prices = [10.0, 100.0, 10.0, 100.0]
    storage_unit.forecaster.price["EOM"] = prices

    strategy = StorageEnergyOptimizationDmasStrategy()
    hour_count = 4
    start = storage_unit.index[0]
    results = strategy.optimize(storage_unit, "EOM", start, hour_count)
    power = results["normal"]

    # Hour 0 (price 10): should be charging (grid_power < 0)
    assert power[0] < 0, f"Expected charging at hour 0, got {power[0]}"
    # Hour 1 (price 100): should be discharging (grid_power > 0)
    assert power[1] > 0, f"Expected discharging at hour 1, got {power[1]}"

    # Volume at end must be capacity / 2
    final_vol = storage_unit.outputs["volume"].iloc[3]
    assert abs(final_vol - storage_unit.capacity / 2) < 1e-3

    # Total profit must be positive
    total_profit = sum(storage_unit.outputs["profit"].iloc[:4])
    assert total_profit > 0


def test_storage_bids_structure_and_sign(storage_unit):
    """
    Test that generated bids have negative volume for charging, positive volume for discharging,
    and appropriate pricing.
    """
    prices = [10.0, 100.0, 10.0, 100.0]
    storage_unit.forecaster.price["EOM"] = prices
    strategy = StorageEnergyOptimizationDmasStrategy()
    hour_count = 4

    mc = MarketConfig(
        market_id="EOM",
        opening_hours=rr.rrule(rr.HOURLY),
        opening_duration=timedelta(hours=1),
        market_mechanism="not needed",
        market_products=[
            MarketProduct(timedelta(hours=1), hour_count, timedelta(hours=0))
        ],
        additional_fields=["exclusive_id"],
    )
    start = storage_unit.index[0]
    products = get_available_products(mc.market_products, start)
    orderbook = strategy.calculate_bids(
        storage_unit, market_config=mc, product_tuples=products
    )

    assert len(orderbook) > 0
    # There should be orders with both positive and negative volume (charging and discharging)
    vols = [o["volume"] for o in orderbook]
    assert any(v < 0 for v in vols), "Should have charging bids (negative volume)"
    assert any(v > 0 for v in vols), "Should have discharging asks (positive volume)"


def test_storage_market_clearing(storage_unit):
    """
    Test that storage exclusive orders can be cleared in ComplexDmasClearingRole.
    """
    from assume.markets.clearing_algorithms.complex_clearing_dmas import (
        ComplexDmasClearingRole,
    )

    prices = [10.0, 100.0, 10.0, 100.0]
    storage_unit.forecaster.price["EOM"] = prices
    strategy = StorageEnergyOptimizationDmasStrategy()
    hour_count = 4

    start = storage_unit.index[0]
    mc = MarketConfig(
        market_id="EOM",
        opening_hours=rr.rrule(
            rr.HOURLY,
            dtstart=start,
            until=start + timedelta(days=2),
        ),
        opening_duration=timedelta(hours=1),
        market_mechanism="not needed",
        market_products=[
            MarketProduct(timedelta(hours=1), hour_count, timedelta(hours=0))
        ],
        additional_fields=["exclusive_id", "link", "block_id"],
    )
    products = get_available_products(mc.market_products, start)
    orderbook = strategy.calculate_bids(
        storage_unit, market_config=mc, product_tuples=products
    )

    for o in orderbook:
        o["agent_addr"] = "storage_agent"
        o["block_id"] = None
        o["link"] = None

    # Add single bids for demand (to buy discharging power and sell charging power)
    end0 = start + timedelta(hours=1)
    end1 = start + timedelta(hours=2)

    demand_orders = [
        # Demand at hour 1 (when storage discharges ~70 MW)
        {
            "start_time": start + timedelta(hours=1),
            "end_time": end1,
            "volume": -70,
            "price": 200,
            "agent_addr": "dem1",
            "bid_id": "dem_h1",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
        # Supply at hour 0 (when storage charges ~100 MW)
        {
            "start_time": start,
            "end_time": end0,
            "volume": 100,
            "price": 5,
            "agent_addr": "gen1",
            "bid_id": "gen_h0",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
    ]

    all_orders = orderbook + demand_orders
    mr = ComplexDmasClearingRole(mc)
    accepted, rejected, meta, flows = mr.clear(all_orders, products)
    assert len(accepted) > 0
