import calendar
from datetime import datetime

import pandas as pd
import pytest
from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd
from mango import RoleAgent, create_container
from mango.util.clock import ExternalClock
from mango.util.termination_detection import tasks_complete_or_sleeping

from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.units_operator import UnitsOperator
from assume.strategies.naive_strategies import NaiveStrategy
from assume.units.demand import Demand

start = datetime(2020, 1, 1)
end = datetime(2020, 12, 2)


@pytest.fixture
async def units_operator() -> UnitsOperator:
    market_name = "Test"
    marketconfig = MarketConfig(
        market_name,
        rr.rrule(rr.HOURLY, dtstart=start, until=end),
        rd(hours=1),
        "pay_as_clear",
        [MarketProduct(rd(hours=1), 1, rd(hours=1))],
    )
    clock = ExternalClock(0)
    container = await create_container(addr=("0.0.0.0", 9098), clock=clock)
    units_agent = RoleAgent(container, "test_operator")
    units_role = UnitsOperator(available_markets=[marketconfig])
    units_agent.add_role(units_role)

    index = pd.date_range(start=start, end=end + pd.Timedelta(hours=4), freq="1h")

    units_role.context.data_dict = {}

    params_dict = {
        "bidding_strategies": {"energy": NaiveStrategy()},
        "technology": "energy",
        "unit_operator": "test_operator",
        "max_power": 1000,
        "min_power": 0,
        "forecaster": NaiveForecast(index, demand=1000),
    }
    unit = Demand("testdemand", index=index, **params_dict)
    await units_role.add_unit(unit)

    yield units_role

    end_ts = calendar.timegm(end.utctimetuple())
    clock.set_time(end_ts)
    await tasks_complete_or_sleeping(container)
    await container.shutdown()


async def test_set_unit_dispatch(units_operator: UnitsOperator):
    orderbook = [
        {
            "start_time": start,
            "end_time": start + rd(hours=1),
            "volume": 500,
            "accepted_volume": 500,
            "price": 1000,
            "accepted_price": 1000,
            "agent_id": "gen1",
            "unit_id": "testdemand",
            "only_hours": None,
        }
    ]
    marketconfig = units_operator.available_markets[0]

    assert units_operator.units["testdemand"].outputs["energy"].max() == 0

    units_operator.set_unit_dispatch(orderbook, marketconfig)
    assert units_operator.units["testdemand"].outputs["energy"].max() == 500


async def test_formulate_bids(units_operator: UnitsOperator):
    marketconfig = units_operator.available_markets[0]
    from assume.common.utils import get_available_products

    products = get_available_products(marketconfig.market_products, start)
    orderbook = await units_operator.formulate_bids(marketconfig, products)
    assert len(orderbook) == 1

    assert orderbook[0]["volume"] == -1000
    assert orderbook[0]["price"] == 3000
