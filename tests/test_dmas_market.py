from datetime import datetime, timedelta

from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd

from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.utils import get_available_products
from assume.markets.base_market import MarketConfig, MarketProduct, MarketRole
from assume.markets.clearing_algorithms.complex_clearing_dmas import (
    complex_clearing_dmas,
)

start = datetime(2020, 1, 1)
end = datetime(2020, 12, 2)

simple_dayahead_auction_config = MarketConfig(
    "simple_dayahead_auction",
    market_products=[MarketProduct(rd(hours=+1), 24, rd(hours=1))],
    additional_fields=["exclusive_id", "link", "block_id"],
    opening_hours=rr.rrule(
        rr.HOURLY,
        dtstart=datetime(2005, 6, 1),
        cache=True,
    ),
    opening_duration=timedelta(hours=1),
    volume_unit="MW",
    price_unit="€/MW",
    market_mechanism="pay_as_clear",
)


def test_dmas_market_init():
    mr = MarketRole(simple_dayahead_auction_config)
    next_opening = simple_dayahead_auction_config.opening_hours.after(datetime.now())
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )
    assert len(products) == 24


def test_market():
    """
    For debugging, the following might help:
    from pyomo.environ import value
    sinks = [value(model.sink[key]) for key in model.sink]
    sources = [value(model.source[key]) for key in model.source]
    [model.use_hourly_ask[(block, hour, agent)].value for block, hour, agent in orders["single_ask"].keys()]
    """
    mr = MarketRole(simple_dayahead_auction_config)
    next_opening = simple_dayahead_auction_config.opening_hours.after(datetime.now())
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )
    assert len(products) == 24

    print(products)
    start = products[0][0]
    end = products[-1][1]
    only_hours = products[0][2]

    orderbook: Orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 120,
            "price": 120,
            "agent_id": "gen1",
            "bid_id": "bid1",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 80,
            "price": 58,
            "agent_id": "gen1",
            "bid_id": "bid2",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 100,
            "price": 53,
            "agent_id": "gen1",
            "bid_id": "bid3",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": -180,
            "price": 70,
            "agent_id": "dem1",
            "bid_id": "bid4",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
    ]
    simple_dayahead_auction_config.market_mechanism = complex_clearing_dmas
    mr.all_orders = orderbook
    accepted, rejected, meta = simple_dayahead_auction_config.market_mechanism(
        mr, products
    )
    assert meta[0]["demand_volume"] > 0
    assert meta[0]["price"] > 0
    import pandas as pd

    print(pd.DataFrame(mr.all_orders))
    print(pd.DataFrame(accepted))
    print(meta)
