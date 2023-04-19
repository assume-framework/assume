from datetime import datetime, timedelta

from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd

from assume.common.market_mechanisms import available_clearing_strategies
from assume.common.marketclasses import MarketConfig, MarketProduct, Order, Orderbook
from assume.common.utils import get_available_products
from assume.markets.base_market import MarketRole

simple_dayahead_auction_config = MarketConfig(
    "simple_dayahead_auction",
    market_products=[MarketProduct(rd(hours=+1), 1, rd(hours=1))],
    additional_fields=["node_id"],
    opening_hours=rr.rrule(
        rr.HOURLY,
        dtstart=datetime(2005, 6, 1),
        cache=True,
    ),
    opening_duration=timedelta(hours=1),
    amount_unit="MW",
    volume_tick=0.1,
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


import itertools

import numpy as np


def create_orderbook(order: Order = None, node_ids=[0], count=100, seed=30):
    if not order:
        start = datetime.today()
        end = datetime.today() + timedelta(hours=1)
        order: Order = {
            "start_time": start,
            "end_time": end,
            "agent_id": "dem1",
            "volume": 0,
            "price": 0,
            "only_hours": None,
            "node_id": 0,
        }
    orders = []
    np.random.seed(seed)

    for node_id, i in itertools.product(node_ids, range(count)):
        new_order = order.copy()
        new_order["price"] = np.random.randint(100)
        new_order["volume"] = np.random.randint(-10, 10)
        if new_order["volume"] > 0:
            agent_id = f"gen_{i}"
        else:
            agent_id = f"dem_{i}"
        new_order["agent_id"] = agent_id

        new_order["node_id"] = node_id
        orders.append(new_order)
    return orders


def test_market_mechanism():
    import copy

    for name, mechanism in available_clearing_strategies.items():
        print(name)
        market_config = copy.copy(simple_dayahead_auction_config)
        market_config.market_mechanism = mechanism
        mr = MarketRole(market_config)
        next_opening = market_config.opening_hours.after(datetime.now())
        products = get_available_products(market_config.market_products, next_opening)
        assert len(products) == 1
        order = {
            "start_time": products[0][0],
            "end_time": products[0][1],
            "only_hours": products[0][2],
        }

        orderbook = create_orderbook(order, node_ids=[0, 1, 2])
        mr.all_orders = orderbook
        clearing_result, meta = market_config.market_mechanism(mr, products)
        assert meta[0]["supply_volume"] > 0
        assert meta[0]["price"] > 0
        # import pandas as pd
        # print(pd.DataFrame.from_dict(mr.all_orders))
        # print(pd.DataFrame.from_dict(clearing_result))
        print(meta)

    return mr.all_orders, meta


if __name__ == "__main__":
    clearing_result, meta = test_market_mechanism()
    from assume.common.utils import plot_orderbook

    fig, ax = plot_orderbook(clearing_result, meta)
    fig.show()

    print("finished")
