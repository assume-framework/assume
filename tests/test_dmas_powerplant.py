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


if __name__ == "__main__":
    pytest.main(["-s", __file__])
