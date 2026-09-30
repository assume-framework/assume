# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import pandas as pd
import pytest
from dateutil import rrule as rr

from assume.common.fast_pandas import FastIndex
from assume.common.forecaster import PowerplantForecaster
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.utils import get_available_products
from assume.strategies.dmas_powerplant import EnergyOptimizationDmasStrategy
from assume.units import PowerPlant

from .utils import get_test_prices


@pytest.fixture
def power_plant_1() -> PowerPlant:
    index = FastIndex("2022-01-01", periods=4, freq="h")
    ff = PowerplantForecaster(
        index=index,
        availability=1,
        fuel_prices={"lignite": [10, 11, 12, 13], "co2": [10, 20, 30, 30]},
        market_prices={"EOM": 50},
    )
    # Create a PowerPlant instance with some example parameters
    return PowerPlant(
        id="test_pp",
        unit_operator="test_operator",
        technology="hard coal",
        bidding_strategies={"EOM": EnergyOptimizationDmasStrategy()},
        max_power=1000,
        min_power=200,
        efficiency=0.5,
        additional_cost=10,
        fuel_type="lignite",
        emission_factor=0.5,
        forecaster=ff,
    )


@pytest.fixture
def power_plant_day(fuel_type="lignite") -> PowerPlant:
    periods = 48
    index = pd.date_range("2022-01-01", periods=periods, freq="h")

    prices = get_test_prices(periods)
    ff = PowerplantForecaster(
        index,
        availability=1,
        fuel_prices=prices,
        market_prices={"EOM": prices["power"]},
    )
    # Create a PowerPlant instance with some example parameters
    return PowerPlant(
        id="test_pp",
        unit_operator="test_operator",
        technology="hard coal",
        bidding_strategies={"EOM": EnergyOptimizationDmasStrategy()},
        max_power=1000,
        min_power=200,
        efficiency=0.5,
        additional_cost=10,
        fuel_type="lignite",
        emission_factor=0.5,
        forecaster=ff,
    )


def test_dmas_init(power_plant_1):
    strategy = EnergyOptimizationDmasStrategy()
    hour_count = len(power_plant_1.index)

    prices = get_test_prices()

    strategy.build_model(
        power_plant_1,
        datetime(2022, 1, 1),
        hour_count,
        prices["co2"],
        prices[power_plant_1.fuel_type],
        prices["power"],
        runtime=1,
        p0=300,
    )


def test_dmas_calc(power_plant_1):
    strategy = EnergyOptimizationDmasStrategy()
    hour_count = len(power_plant_1.index) // 2

    mc = MarketConfig(
        market_id="EOM",
        opening_hours=rr.rrule(rr.HOURLY),
        opening_duration=timedelta(hours=1),
        market_mechanism="not needed",
        market_products=[
            MarketProduct(timedelta(hours=1), hour_count, timedelta(hours=0))
        ],
        additional_fields=["link", "block_id"],
    )
    start = power_plant_1.index[0]
    products = get_available_products(mc.market_products, start)
    orderbook = strategy.calculate_bids(
        power_plant_1, market_config=mc, product_tuples=products
    )
    assert orderbook
    block_ids = {o["block_id"] for o in orderbook} | {-1}
    # all links should match existing block ids
    unknown = [o["link"] for o in orderbook if o["link"] not in block_ids]
    assert unknown == [], "found unknown link orders"


def test_dmas_day(power_plant_day):
    strategy = EnergyOptimizationDmasStrategy()
    hour_count = len(power_plant_day.index) // 2
    assert hour_count == 24

    mc = MarketConfig(
        market_id="EOM",
        opening_hours=rr.rrule(rr.HOURLY),
        opening_duration=timedelta(hours=1),
        market_mechanism="not needed",
        market_products=[
            MarketProduct(timedelta(hours=1), hour_count, timedelta(hours=0))
        ],
        additional_fields=["link", "block_id"],
    )
    start = power_plant_day.index[0]
    products = get_available_products(mc.market_products, start)
    orderbook = strategy.calculate_bids(
        power_plant_day, market_config=mc, product_tuples=products
    )
    assert orderbook
    block_ids = {o["block_id"] for o in orderbook} | {-1}
    # all links should match existing block ids
    unknown = [o["link"] for o in orderbook if o["link"] not in block_ids]
    assert unknown == [], "found unknown link orders"


def test_dmas_ramp_day(power_plant_day):
    """
    Test that ramping constraints are respected in the bidding behavior
    """
    power_plant_day.ramp_down = power_plant_day.max_power / 2
    power_plant_day.ramp_up = power_plant_day.max_power / 2
    strategy = EnergyOptimizationDmasStrategy()
    hour_count = len(power_plant_day.index) // 2
    assert hour_count == 24

    mc = MarketConfig(
        market_id="EOM",
        opening_hours=rr.rrule(rr.HOURLY),
        opening_duration=timedelta(hours=1),
        market_mechanism="not needed",
        market_products=[
            MarketProduct(timedelta(hours=1), hour_count, timedelta(hours=0))
        ],
        additional_fields=["link", "block_id"],
    )
    start = power_plant_day.index[0]
    products = get_available_products(mc.market_products, start)
    orderbook = strategy.calculate_bids(
        power_plant_day, market_config=mc, product_tuples=products
    )
    assert orderbook
    block_ids = {o["block_id"] for o in orderbook} | {-1}
    # all links should match existing block ids
    unknown = [o["link"] for o in orderbook if o["link"] not in block_ids]
    assert unknown == [], "found unknown link orders"


def test_dmas_prevent_start(power_plant_day):
    """
    This test makes sure, that the powerplants still bids positive marginal cost, with block bids.
    Even if the price is not well between the day.
    The market should still see this as the best option instead of turning off the powerplant
    """
    strategy = EnergyOptimizationDmasStrategy()
    hour_count = len(power_plant_day.index) // 2
    assert hour_count == 24

    # quite bad forecast here
    power_plant_day.forecaster.price["EOM"].iloc[10:11] = -10

    mc = MarketConfig(
        market_id="EOM",
        opening_hours=rr.rrule(rr.HOURLY),
        opening_duration=timedelta(hours=1),
        market_mechanism="not needed",
        market_products=[
            MarketProduct(timedelta(hours=1), hour_count, timedelta(hours=0))
        ],
        additional_fields=["link", "block_id"],
    )
    start = power_plant_day.index[0]
    products = get_available_products(mc.market_products, start)
    orderbook = strategy.calculate_bids(
        power_plant_day, market_config=mc, product_tuples=products
    )
    assert orderbook
    block_ids = {o["block_id"] for o in orderbook} | {-1}
    # all links should match existing block ids
    unknown = [o["link"] for o in orderbook if o["link"] not in block_ids]
    assert unknown == [], "found unknown link orders"


def test_dmas_prevent_start_end(power_plant_day):
    """
    The powerplant should bid negative at the end of the day to produce a prevented start.
    This should ensure, that the powerplant is on at the start of the next day
    """
    strategy = EnergyOptimizationDmasStrategy()
    hour_count = len(power_plant_day.index) // 2
    assert hour_count == 24

    # quite bad forecast here
    power_plant_day.forecaster.price["EOM"].iloc[20:24] = -10

    mc = MarketConfig(
        market_id="EOM",
        opening_hours=rr.rrule(rr.HOURLY),
        opening_duration=timedelta(hours=1),
        market_mechanism="not needed",
        market_products=[
            MarketProduct(timedelta(hours=1), hour_count, timedelta(hours=0))
        ],
        additional_fields=["link", "block_id"],
    )
    start = power_plant_day.index[0]
    products = get_available_products(mc.market_products, start)
    orderbook = strategy.calculate_bids(
        power_plant_day, market_config=mc, product_tuples=products
    )
    assert orderbook
    block_ids = {o["block_id"] for o in orderbook} | {-1}
    # all links should match existing block ids
    unknown = [o["link"] for o in orderbook if o["link"] not in block_ids]
    assert unknown == [], "found unknown link orders"


def test_dmas_prevent_start_avoided_cost_validation(power_plant_day):
    """
    Validates that when a price dip occurs before high morning prices,
    prevented_start triggers with delta > 0, and bid prices during dip hours
    are discounted to keep the plant dispatched and avoid morning startup costs.
    """
    power_plant_day.cold_start_cost = 50000.0
    hour_count = 24
    power_plant_day.forecaster.price["EOM"].iloc[:] = 50.0
    # Price dip below marginal cost at the end of Day 1
    power_plant_day.forecaster.price["EOM"].iloc[20:24] = 10.0
    # High price on Day 2
    power_plant_day.forecaster.price["EOM"].iloc[24:48] = 50.0

    strategy = EnergyOptimizationDmasStrategy()

    mc = MarketConfig(
        market_id="EOM",
        opening_hours=rr.rrule(rr.HOURLY),
        opening_duration=timedelta(hours=1),
        market_mechanism="not needed",
        market_products=[
            MarketProduct(timedelta(hours=1), hour_count, timedelta(hours=0))
        ],
        additional_fields=["link", "block_id"],
    )
    start = power_plant_day.index[0]
    products = get_available_products(mc.market_products, start)
    orderbook = strategy.calculate_bids(
        power_plant_day, market_config=mc, product_tuples=products
    )

    assert strategy.prevented_start["prevent"] is True
    assert strategy.prevented_start["delta"] > 0
    assert len(strategy.prevented_start["hours"]) > 0

    # Verify orders in orderbook
    df = pd.DataFrame(orderbook)
    assert not df.empty
    # Orders during dip hours should reflect discounted price
    dip_start_times = [
        start + timedelta(hours=int(h)) for h in strategy.prevented_start["hours"]
    ]
    dip_orders = df[df["start_time"].isin(dip_start_times)]
    assert not dip_orders.empty
    assert (dip_orders["price"] < 20.0).any()
    # Next day reduction should be stored
    assert start.date() in strategy.reduction_next_day


def test_powerplant_ramping_limits_enforced(power_plant_day):
    """Test that ramp up and ramp down constraints strictly limit rate of power change."""
    import numpy as np

    power_plant_day.ramp_up = 150.0
    power_plant_day.ramp_down = 150.0
    strategy = EnergyOptimizationDmasStrategy()
    hour_count = 24
    start = power_plant_day.index[0]
    base_price = power_plant_day.forecaster.price["EOM"]

    gen = strategy.optimize(power_plant_day, start, hour_count, base_price)
    power = np.asarray(gen)
    # When staying online, power deltas must not exceed ramp limits
    for t in range(1, len(power)):
        if power[t] > 0 and power[t - 1] > 0:
            assert power[t] - power[t - 1] <= power_plant_day.ramp_up + 1e-3
            assert power[t - 1] - power[t] <= power_plant_day.ramp_down + 1e-3


if __name__ == "__main__":
    pytest.main(["-s", __file__])
