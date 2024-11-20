# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime

import pandas as pd
import pytest
from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd
from mango import RoleAgent, activate, addr, create_tcp_container
from mango.util.clock import ExternalClock
from mango.util.termination_detection import tasks_complete_or_sleeping

from assume.common.fast_pandas import FastIndex
from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.units_operator import UnitsOperator
from assume.common.utils import datetime2timestamp
from assume.strategies.naive_strategies import NaiveSingleBidStrategy
from assume.units.demand import Demand
from assume.units.powerplant import PowerPlant

try:
    from assume.reinforcement_learning.learning_unit_operator import RLUnitsOperator
except ImportError:
    # to suffice the type hint as best as possible
    from assume.common.units_operator import UnitsOperator as RLUnitsOperator

start = datetime(2020, 1, 1)
end = datetime(2020, 12, 2)


@pytest.fixture
async def units_operator() -> UnitsOperator:
    market_id = "EOM"
    marketconfig = MarketConfig(
        market_id=market_id,
        opening_hours=rr.rrule(rr.HOURLY, dtstart=start, until=end),
        opening_duration=rd(hours=1),
        market_mechanism="pay_as_clear",
        market_products=[MarketProduct(rd(hours=1), 1, rd(hours=1))],
    )
    clock = ExternalClock(0)
    container = create_tcp_container(addr=("0.0.0.0", 9098), clock=clock)
    units_agent = RoleAgent()
    units_role = UnitsOperator(available_markets=[marketconfig])
    units_agent.add_role(units_role)
    agent_id = container.register(units_agent)

    index = FastIndex(start=start, end=end + pd.Timedelta(hours=4), freq="1h")

    params_dict = {
        "bidding_strategies": {"EOM": NaiveSingleBidStrategy()},
        "technology": "energy",
        "unit_operator": agent_id,
        "max_power": 1000,
        "min_power": 0,
        "forecaster": NaiveForecast(index, demand=1000),
    }
    unit = Demand("testdemand", **params_dict)
    units_role.add_unit(unit)

    start_ts = datetime2timestamp(start)
    clock.set_time(start_ts)

    async with activate(container):
        yield units_role
        end_ts = datetime2timestamp(end)
        clock.set_time(end_ts)
        await tasks_complete_or_sleeping(container)


@pytest.fixture
async def rl_units_operator() -> RLUnitsOperator:
    market_id = "EOM"
    marketconfig = MarketConfig(
        market_id=market_id,
        opening_hours=rr.rrule(rr.HOURLY, dtstart=start, until=end),
        opening_duration=rd(hours=1),
        market_mechanism="pay_as_clear",
        market_products=[MarketProduct(rd(hours=1), 1, rd(hours=1))],
    )
    clock = ExternalClock(0)
    container = create_tcp_container(addr=("0.0.0.0", 9098), clock=clock)
    units_agent = RoleAgent()
    units_role = RLUnitsOperator(available_markets=[marketconfig])
    units_agent.add_role(units_role)
    agent_id = container.register(units_agent)

    index = FastIndex(start=start, end=end + pd.Timedelta(hours=4), freq="1h")

    params_dict = {
        "bidding_strategies": {"EOM": NaiveSingleBidStrategy()},
        "technology": "energy",
        "unit_operator": agent_id,
        "max_power": 1000,
        "min_power": 0,
        "forecaster": NaiveForecast(index, demand=1000),
    }
    unit = Demand("testdemand", **params_dict)
    units_role.add_unit(unit)

    start_ts = datetime2timestamp(start)
    clock.set_time(start_ts)

    units_role.context.data.update(
        {
            "train_start": start,
            "train_end": end,
            "train_freq": "24h",
        }
    )

    async with activate(container):
        yield units_role
        end_ts = datetime2timestamp(end)
        clock.set_time(end_ts)
        await tasks_complete_or_sleeping(container)


async def test_set_unit_dispatch(units_operator: UnitsOperator):
    orderbook = [
        {
            "start_time": start,
            "end_time": start + rd(hours=1),
            "volume": 500,
            "accepted_volume": 500,
            "price": 1000,
            "accepted_price": 1000,
            "agent_addr": "gen1",
            "unit_id": "testdemand",
            "only_hours": None,
        }
    ]
    marketconfig = units_operator.available_markets[0]

    assert units_operator.units["testdemand"].outputs["energy"].max() == 0

    units_operator.set_unit_dispatch(orderbook, marketconfig)
    assert units_operator.units["testdemand"].outputs["energy"].max() == 500


async def test_write_actual_dispatch(units_operator: UnitsOperator):
    units_operator.write_actual_dispatch("energy")
    assert units_operator.last_sent_dispatch["energy"] > 0
    assert units_operator.last_sent_dispatch["test"] == 0
    units_operator.write_actual_dispatch("test")
    assert units_operator.last_sent_dispatch["test"] > 0


async def test_formulate_bids(units_operator: UnitsOperator):
    marketconfig = units_operator.available_markets[0]
    from assume.common.utils import get_available_products

    products = get_available_products(marketconfig.market_products, start)
    orderbook = await units_operator.formulate_bids(marketconfig, products)
    assert len(orderbook) == 1

    assert orderbook[0]["volume"] == -1000
    assert orderbook[0]["price"] == 3000


@pytest.mark.require_learning
async def test_write_learning_params(rl_units_operator: RLUnitsOperator):
    from assume.strategies.learning_advanced_orders import RLAdvancedOrderStrategy
    from assume.strategies.learning_strategies import RLStrategy

    marketconfig = rl_units_operator.available_markets[0]
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 2)
    index = FastIndex(start=start, end=end + pd.Timedelta(hours=24), freq="1h")

    params_dict = {
        "bidding_strategies": {
            "EOM": RLAdvancedOrderStrategy(
                unit_id="testplant",
                learning_mode=True,
            )
        },
        "technology": "energy",
        "unit_operator": "test_operator",
        "max_power": 1000,
        "min_power": 0,
        "forecaster": NaiveForecast(index, powerplant=1000),
    }
    unit = PowerPlant("testplant", **params_dict)
    rl_units_operator.add_unit(unit)

    rl_units_operator.learning_mode = True
    rl_units_operator.learning_data = {"test": 1}

    rl_units_operator.context.data.update(
        {
            "learning_output_agent_addr": addr("world", "export_agent_1"),
            "learning_agent_addr": addr("world_0", "learning_agent"),
        }
    )

    from assume.common.utils import get_available_products

    products = get_available_products(marketconfig.market_products, start)
    orderbook = await rl_units_operator.formulate_bids(marketconfig, products)

    open_tasks = len(rl_units_operator.context.scheduler._scheduled_tasks)

    rl_units_operator.write_learning_to_output(
        orderbook, market_id=marketconfig.market_id
    )

    assert len(rl_units_operator.context.scheduler._scheduled_tasks) == open_tasks + 1

    rl_units_operator.units["testplant"].bidding_strategies[
        "EOM"
    ].bidding_strategies = RLStrategy(
        unit_id="testplant",
        learning_mode=True,
        observation_dimension=50,
        action_dimension=2,
    )

    rl_units_operator.learning_data = {"test": 2}

    products = get_available_products(marketconfig.market_products, start)
    orderbook = await rl_units_operator.formulate_bids(marketconfig, products)

    open_tasks = len(rl_units_operator.context.scheduler._scheduled_tasks)

    rl_units_operator.write_learning_to_output(
        orderbook, market_id=marketconfig.market_id
    )

    assert len(rl_units_operator.context.scheduler._scheduled_tasks) == open_tasks + 1


async def test_get_actual_dispatch(units_operator: UnitsOperator):
    # GIVEN the first hour happened
    # the UnitOperator does not
    clock = units_operator.context.context.clock

    last = clock.time
    clock.set_time(clock.time + 3600)
    # WHEN actual_dispatch is called
    market_dispatch, unit_dfs = units_operator.get_actual_dispatch("energy", last)
    # THEN resulting unit dispatch dataframe contains one row
    # which is for the current time - as we must know our current dispatch
    assert datetime2timestamp(unit_dfs[0]["time"][0]) == last
    assert datetime2timestamp(unit_dfs[0]["time"][1]) == clock.time
    # only 1 start and stop contained
    assert len(unit_dfs[0]["time"]) == 2
    assert len(market_dispatch) == 0

    # WHEN another hour passes
    last = clock.time
    clock.set_time(clock.time + 3600)

    # THEN resulting unit dispatch dataframe contains only one row with current dispatch
    market_dispatch, unit_dfs = units_operator.get_actual_dispatch("energy", last)
    assert datetime2timestamp(unit_dfs[0]["time"][0]) == last
    assert datetime2timestamp(unit_dfs[0]["time"][1]) == clock.time
    assert len(unit_dfs[0]["time"]) == 2
    assert len(market_dispatch) == 0

    last = clock.time
    clock.set_time(clock.time + 3600)

    market_dispatch, unit_dfs = units_operator.get_actual_dispatch("energy", last)
    assert datetime2timestamp(unit_dfs[0]["time"][0]) == last
    assert datetime2timestamp(unit_dfs[0]["time"][1]) == clock.time
    assert len(unit_dfs[0]["time"]) == 2
    assert len(market_dispatch) == 0
