# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime

import pandas as pd
import pytest
from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd
from mango import RoleAgent, activate, create_tcp_container
from mango.util.clock import ExternalClock
from mango.util.termination_detection import tasks_complete_or_sleeping

from assume.common.fast_pandas import FastIndex
from assume.common.forecaster import DemandForecaster, PowerplantForecaster
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.units_operator import UnitsOperator
from assume.common.utils import datetime2timestamp, timestamp2datetime
from assume.strategies.naive_strategies import EnergyNaiveStrategy
from assume.strategies.portfolio_strategies import (
    UnitsOperatorDirectStrategy,
)
from assume.units.demand import Demand
from assume.units.powerplant import PowerPlant

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
        "bidding_strategies": {"EOM": EnergyNaiveStrategy()},
        "technology": "energy",
        "unit_operator": agent_id,
        "max_power": -1000,
        "min_power": 0,
        "forecaster": DemandForecaster(index, market_prices={"EOM": 50}, demand=-1000),
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


async def test_write_market_dispatch(units_operator: UnitsOperator):
    units_operator.write_market_dispatch(units_operator.available_markets[0])
    assert units_operator.last_sent_market_dispatch["EOM"] > 0
    assert units_operator.last_sent_market_dispatch["test"] == 0


async def test_write_actual_dispatch(units_operator: UnitsOperator):
    sent = []
    units_operator.context.schedule_instant_message = lambda receiver_addr, content: (
        sent.append(content)
    )

    # without an output agent nothing is written
    units_operator.write_actual_dispatch([{"unit": "testdemand"}])
    assert sent == []

    units_operator.context.data["output_agent_addr"] = "output_agent"

    # an empty dispatch is not worth a message either
    units_operator.write_actual_dispatch([])
    assert sent == []

    units_operator.write_actual_dispatch([{"unit": "testdemand"}])
    assert len(sent) == 1
    # the message type is the name of the database table it is written to
    assert sent[0]["type"] == "unit_dispatch"
    assert sent[0]["data"] == [{"unit": "testdemand"}]


async def test_independent_bids_portfolio(units_operator: UnitsOperator):
    marketconfig = units_operator.available_markets[0]
    from assume.common.utils import get_available_products

    products = get_available_products(marketconfig.market_products, start)
    strategy = UnitsOperatorDirectStrategy()
    orderbook = strategy.calculate_bids(units_operator, marketconfig, products)
    assert len(orderbook) == 1

    assert orderbook[0]["volume"] == -1000
    assert orderbook[0]["price"] == 3000


async def test_get_actual_dispatch(units_operator: UnitsOperator):
    # GIVEN the first hour happened
    # the UnitOperator does not
    clock = units_operator.context.context.clock

    last = clock.time
    clock.set_time(clock.time + 3600)
    # WHEN the actual dispatch is retrieved
    unit_dfs = units_operator.get_actual_dispatch(
        timestamp2datetime(last + 1), timestamp2datetime(clock.time)
    )
    # THEN resulting unit dispatch dataframe contains one row
    # which is for the current time - as we must know our current dispatch
    assert datetime2timestamp(unit_dfs[0]["time"][0]) == clock.time
    assert len(unit_dfs[0]["time"]) == 1

    # WHEN another hour passes
    last = clock.time
    clock.set_time(clock.time + 3600)

    # THEN resulting unit dispatch dataframe contains only one row with current dispatch
    unit_dfs = units_operator.get_actual_dispatch(
        timestamp2datetime(last + 1), timestamp2datetime(clock.time)
    )
    assert datetime2timestamp(unit_dfs[0]["time"][0]) == clock.time
    assert len(unit_dfs[0]["time"]) == 1

    last = clock.time
    clock.set_time(clock.time + 3600)

    unit_dfs = units_operator.get_actual_dispatch(
        timestamp2datetime(last + 1), timestamp2datetime(clock.time)
    )
    assert datetime2timestamp(unit_dfs[0]["time"][0]) == clock.time
    assert len(unit_dfs[0]["time"]) == 1


def make_order(unit_id="testdemand", market_id="EOM", hours=1):
    return {
        "start_time": start,
        "end_time": start + rd(hours=hours),
        "volume": -1000,
        "accepted_volume": -1000,
        "price": 50,
        "accepted_price": 50,
        "agent_addr": "gen1",
        "unit_id": unit_id,
        "market_id": market_id,
        "only_hours": None,
    }


def track_rewards(units_operator: UnitsOperator, unit_id="testdemand"):
    """Replaces the unit's reward calculation by a call recorder."""
    rewarded = []
    strategy = units_operator.units[unit_id].bidding_strategies["EOM"]
    strategy.calculate_reward = lambda unit, marketconfig, orderbook: rewarded.append(
        orderbook
    )
    return rewarded


async def test_cashflow_is_booked_at_clearing(units_operator: UnitsOperator):
    marketconfig = units_operator.available_markets[0]
    units_operator.registered_markets[marketconfig.market_id] = marketconfig
    unit = units_operator.units["testdemand"]
    rewarded = track_rewards(units_operator)

    units_operator.handle_market_feedback(
        {
            "context": "clearing",
            "market_id": "EOM",
            "accepted_orders": [make_order()],
            "rejected_orders": [],
        },
        {},
    )

    # the cashflow follows from the clearing result and is booked right away
    assert unit.outputs["energy_cashflow"].at[start] == -1000 * 50
    # the reward is deferred until the delivery period has been executed
    assert rewarded == []
    assert len(units_operator.pending_orders["EOM"]) == 1


async def test_reward_is_calculated_after_delivery(units_operator: UnitsOperator):
    marketconfig = units_operator.available_markets[0]
    units_operator.registered_markets[marketconfig.market_id] = marketconfig
    rewarded = track_rewards(units_operator)

    units_operator.handle_market_feedback(
        {
            "context": "clearing",
            "market_id": "EOM",
            "accepted_orders": [make_order()],
            "rejected_orders": [],
        },
        {},
    )

    # the time step of the product has not been executed yet
    units_operator.calculate_unit_reward(start - rd(hours=1))
    assert rewarded == []
    assert len(units_operator.pending_orders["EOM"]) == 1

    # the product [start, start + 1h) covers the single time step `start`,
    # so it is complete once that step has been executed
    units_operator.calculate_unit_reward(start)
    assert len(rewarded) == 1
    assert units_operator.pending_orders["EOM"] == []

    # and it is not rewarded a second time
    units_operator.calculate_unit_reward(start + rd(hours=1))
    assert len(rewarded) == 1


async def test_multi_step_product_is_rewarded_once_fully_executed(
    units_operator: UnitsOperator,
):
    """
    A multi-step product is rewarded over its whole span, so it must not be rewarded
    before every one of its time steps has been executed.
    """
    marketconfig = units_operator.available_markets[0]
    units_operator.registered_markets[marketconfig.market_id] = marketconfig
    rewarded = track_rewards(units_operator)

    units_operator.handle_market_feedback(
        {
            "context": "clearing",
            "market_id": "EOM",
            "accepted_orders": [make_order(hours=4)],
            "rejected_orders": [],
        },
        {},
    )

    # [start, start + 4h) covers the time steps start .. start + 3h
    for execute_until in (start, start + rd(hours=1), start + rd(hours=2)):
        units_operator.calculate_unit_reward(execute_until)
        assert rewarded == [], f"rewarded too early at {execute_until}"

    units_operator.calculate_unit_reward(start + rd(hours=3))
    assert len(rewarded) == 1
    assert units_operator.pending_orders["EOM"] == []


async def test_rewarded_order_stays_in_market_dispatch(units_operator: UnitsOperator):
    """
    The reward drains its own order store, so the market dispatch export keeps
    aggregating the orders until their closing delta has been written.
    """
    marketconfig = units_operator.available_markets[0]
    units_operator.registered_markets[marketconfig.market_id] = marketconfig
    track_rewards(units_operator)

    units_operator.handle_market_feedback(
        {
            "context": "clearing",
            "market_id": "EOM",
            "accepted_orders": [make_order()],
            "rejected_orders": [],
        },
        {},
    )
    units_operator.calculate_unit_reward(start)

    # the closing delta at end_time is only aggregated by the export after it
    market_dispatch = units_operator.get_market_dispatch(
        "EOM", start + rd(hours=1), start + rd(hours=2)
    )
    assert len(market_dispatch) == 1


async def test_execute_dispatch_trails_by_one_time_step(units_operator: UnitsOperator):
    """
    The delivery period of a product can start in the very time step in which its
    market clears, and the clearing is not ordered against this task, so the
    execution has to stay one time step behind the current time.
    """
    clock = units_operator.context.context.clock
    calls = []

    def record(start, end):
        calls.append((start, end))
        return []

    units_operator.get_actual_dispatch = record

    last = units_operator.last_executed_dispatch
    clock.set_time(clock.time + 3600)
    await units_operator.execute_dispatch()

    now = timestamp2datetime(clock.time)
    execute_until = now - units_operator.simulation_index.freq

    # the executed range starts right after the previously executed one and ends one
    # time step before now, so no time step is executed twice or executed too early
    assert calls[0] == (timestamp2datetime(last + 1), execute_until)
    assert units_operator.last_executed_dispatch == datetime2timestamp(execute_until)


async def test_dispatch_of_the_clearing_time_step_is_not_executed_early(
    units_operator: UnitsOperator,
):
    """
    Regression test: with opening_duration == first_delivery the product delivering
    [T, T+1h) is cleared at T. Executing time step T at T would run before that
    clearing was handled.
    """
    marketconfig = units_operator.available_markets[0]
    units_operator.registered_markets[marketconfig.market_id] = marketconfig
    rewarded = track_rewards(units_operator)

    clock = units_operator.context.context.clock
    clock.set_time(datetime2timestamp(start))

    calls = []

    def record(range_start, range_end):
        calls.append((range_start, range_end))
        return []

    units_operator.get_actual_dispatch = record
    units_operator.last_executed_dispatch = datetime2timestamp(start - rd(hours=1))

    # the market clears at `start` for the product delivering [start, start + 1h)
    units_operator.handle_market_feedback(
        {
            "context": "clearing",
            "market_id": "EOM",
            "accepted_orders": [make_order()],
            "rejected_orders": [],
        },
        {},
    )

    # executing at `start` must not touch the time step `start` yet
    await units_operator.execute_dispatch()
    assert calls[-1][1] == start - rd(hours=1)
    assert rewarded == []

    # one time step later it is executed and the product is rewarded
    clock.set_time(datetime2timestamp(start + rd(hours=1)))
    await units_operator.execute_dispatch()
    assert calls[-1][1] == start
    assert len(rewarded) == 1


async def test_get_market_dispatch(units_operator: UnitsOperator):
    clock = units_operator.context.context.clock

    last = clock.time
    clock.set_time(clock.time + 3600)

    market_dispatch = units_operator.get_market_dispatch(
        "EOM", timestamp2datetime(last), timestamp2datetime(clock.time)
    )
    # no orders were cleared, so nothing is dispatched
    assert len(market_dispatch) == 0


def test_participate():
    """
    Tests that an operator without units does not participate.
    And an operator with units for the wrong market does not participate.
    A correct units operator participates correctly.
    """
    market_id = "EOM"
    marketconfig = MarketConfig(
        market_id=market_id,
        opening_hours=rr.rrule(rr.HOURLY, dtstart=start, until=end),
        opening_duration=rd(hours=1),
        market_mechanism="pay_as_clear",
        market_products=[MarketProduct(rd(hours=1), 1, rd(hours=1))],
    )
    clock = ExternalClock(0)
    units_role = UnitsOperator(available_markets=[marketconfig])

    index = FastIndex(start=start, end=end + pd.Timedelta(hours=4), freq="1h")

    assert not units_role.participate(marketconfig)

    params_dict = {
        "bidding_strategies": {"wrong_market": EnergyNaiveStrategy()},
        "technology": "energy",
        "unit_operator": "x",
        "max_power": -1000,
        "min_power": 0,
        "forecaster": DemandForecaster(index, demand=-1000),
    }
    unit = Demand("testdemand", **params_dict)
    units_role.add_unit(unit)

    assert not units_role.participate(marketconfig)

    params_dict = {
        "bidding_strategies": {"EOM": EnergyNaiveStrategy()},
        "technology": "energy",
        "unit_operator": "x",
        "max_power": -1000,
        "min_power": 0,
        "forecaster": DemandForecaster(index, demand=-1000),
    }
    unit = Demand("testdemand", **params_dict)
    units_role.add_unit(unit)

    assert units_role.participate(marketconfig)


def test_participate_lambda():
    """
    Tests that one of the selected lambda functions works correctly in the participation
    """
    market_id = "EOM"
    marketconfig = MarketConfig(
        market_id=market_id,
        opening_hours=rr.rrule(rr.HOURLY, dtstart=start, until=end),
        opening_duration=rd(hours=1),
        market_mechanism="pay_as_clear",
        market_products=[MarketProduct(rd(hours=1), 1, rd(hours=1))],
        eligible_obligations_lambda="only_renewables",
    )
    units_role = UnitsOperator(available_markets=[marketconfig])
    index = FastIndex(start=start, end=end + pd.Timedelta(hours=4), freq="1h")

    assert not units_role.participate(marketconfig)

    params_dict = {
        "bidding_strategies": {"EOM": EnergyNaiveStrategy()},
        "technology": "energy",
        "unit_operator": "x",
        "max_power": 10,
        "min_power": 0,
        "forecaster": PowerplantForecaster(index),
    }
    unit = PowerPlant("testdemand", **params_dict)
    units_role.add_unit(unit)
    assert not units_role.participate(marketconfig)

    params_dict = {
        "bidding_strategies": {"EOM": EnergyNaiveStrategy()},
        "technology": "wind offshore",
        "unit_operator": "x",
        "max_power": 1000,
        "min_power": 0,
        "forecaster": PowerplantForecaster(index),
    }
    unit = PowerPlant("testdemand", **params_dict)
    units_role.add_unit(unit)

    assert units_role.participate(marketconfig)


def test_participate_custom_lambda():
    """
    Tests that the custom lambda function is respected in the participation
    """
    market_id = "EOM"
    marketconfig = MarketConfig(
        market_id=market_id,
        opening_hours=rr.rrule(rr.HOURLY, dtstart=start, until=end),
        opening_duration=rd(hours=1),
        market_mechanism="pay_as_clear",
        market_products=[MarketProduct(rd(hours=1), 1, rd(hours=1))],
        eligible_obligations_lambda=lambda u: abs(u.get("max_power", 0)) > 100,
    )
    units_role = UnitsOperator(available_markets=[marketconfig])
    index = FastIndex(start=start, end=end + pd.Timedelta(hours=4), freq="1h")

    assert not units_role.participate(marketconfig)

    params_dict = {
        "bidding_strategies": {"EOM": EnergyNaiveStrategy()},
        "technology": "energy",
        "unit_operator": "x",
        "max_power": 10,
        "min_power": 0,
        "forecaster": PowerplantForecaster(index),
    }
    unit = PowerPlant("testdemand", **params_dict)
    units_role.add_unit(unit)
    assert not units_role.participate(marketconfig)

    params_dict = {
        "bidding_strategies": {"EOM": EnergyNaiveStrategy()},
        "technology": "energy",
        "unit_operator": "x",
        "max_power": 1000,
        "min_power": 0,
        "forecaster": PowerplantForecaster(index),
    }
    unit = PowerPlant("testdemand", **params_dict)
    units_role.add_unit(unit)

    assert units_role.participate(marketconfig)
