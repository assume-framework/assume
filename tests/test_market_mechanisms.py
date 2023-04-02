from datetime import datetime, timedelta

from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd

from assume.common.market_mechanisms import (
    MarketConfig,
    MarketProduct,
    Orderbook,
    available_clearing_strategies,
)
from assume.common.utils import get_available_products
from assume.markets.base_market import MarketRole

simple_dayahead_auction_config = MarketConfig(
    "simple_dayahead_auction",
    market_products=[MarketProduct(rd(hours=+1), 1, rd(hours=1))],
    opening_hours=rr.rrule(
        rr.HOURLY,
        dtstart=datetime(2005, 6, 1),
        cache=True,
    ),
    opening_duration=timedelta(hours=1),
    amount_unit="MW",
    amount_tick=0.1,
    price_unit="€/MW",
    market_mechanism="pay_as_clear",
)


def test_market():
    mr = MarketRole(simple_dayahead_auction_config)
    next_opening = simple_dayahead_auction_config.opening_hours.after(datetime.now())
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )
    assert len(products) == 1

    print(products)
    start = products[0][0]
    end = products[0][1]
    only_hours = products[0][2]

    orderbook: Orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 120,
            "price": 120,
            "agent_id": "gen1",
            "only_hours": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 80,
            "price": 58,
            "agent_id": "gen1",
            "only_hours": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 100,
            "price": 53,
            "agent_id": "gen1",
            "only_hours": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": -180,
            "price": 70,
            "agent_id": "dem1",
            "only_hours": None,
        },
    ]
    simple_dayahead_auction_config.market_mechanism = available_clearing_strategies[
        simple_dayahead_auction_config.market_mechanism
    ]
    mr.all_orders = orderbook
    clearing_result, meta = simple_dayahead_auction_config.market_mechanism(
        mr, products
    )
    assert meta["volume"] > 0
    assert meta["price"] > 0
    import pandas as pd
    print(pd.DataFrame.from_dict(mr.all_orders))
    print(pd.DataFrame.from_dict(clearing_result))
    print(meta)
