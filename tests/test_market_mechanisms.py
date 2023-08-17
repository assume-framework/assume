from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd

from assume.common.market_objects import MarketConfig, MarketProduct, Order, Orderbook
from assume.common.utils import get_available_products
from assume.markets import MarketRole, clearing_mechanisms

from .utils import create_definite_orderbook, create_orderbook

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
    volume_unit="MW",
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

    orderbook = create_definite_orderbook(products[0][0], products[-1][0])

    simple_dayahead_auction_config.market_mechanism = clearing_mechanisms[
        simple_dayahead_auction_config.market_mechanism
    ]
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


def test_simple_market_mechanism():
    import copy

    for name, mechanism in clearing_mechanisms.items():
        if "complex" in name:
            continue

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
        accepted, rejected, meta = market_config.market_mechanism(mr, products)
        assert meta[0]["supply_volume"] > 0
        assert meta[0]["price"] > 0
        # import pandas as pd
        # print(pd.DataFrame(mr.all_orders))
        # print(pd.DataFrame(clearing_result))
        # print(meta)

    # return mr.all_orders, meta


def test_complex_clearing():
    import copy

    market_config = copy.copy(simple_dayahead_auction_config)

    market_config.market_mechanism = clearing_mechanisms["pay_as_clear_complex"]
    market_config.market_products = [MarketProduct(rd(hours=+1), 24, rd(hours=1))]
    mr = MarketRole(market_config)
    next_opening = market_config.opening_hours.after(datetime.now())
    products = get_available_products(market_config.market_products, next_opening)
    assert len(products) == 24
    orderbook = create_definite_orderbook(products[0][0], products[-1][0])

    mr.all_orders = orderbook
    accepted_orders, rejected_orders, meta = market_config.market_mechanism(
        mr, products
    )

    assert meta[0]["supply_volume"] == 1000
    assert meta[0]["demand_volume"] == -1000
    assert meta[0]["price"] == 100
    assert rejected_orders == []
    assert accepted_orders[0]["agent_id"] == "dem1"
    assert accepted_orders[0]["accepted_volume"] == -1000
    assert accepted_orders[1]["agent_id"] == "gen1"
    assert accepted_orders[1]["accepted_volume"] == 100
    assert accepted_orders[2]["agent_id"] == "gen2"
    assert accepted_orders[2]["accepted_volume"] == 900

    # including block order in the money


@pytest.mark.require_gurobi
def test_complex_clearing_BB():
    import copy

    market_config = copy.copy(simple_dayahead_auction_config)

    market_config.market_mechanism = clearing_mechanisms["pay_as_clear_complex_opt"]
    market_config.market_products = [MarketProduct(rd(hours=+1), 2, rd(hours=1))]
    mr = MarketRole(market_config)
    next_opening = market_config.opening_hours.after(datetime.now())
    products = get_available_products(market_config.market_products, next_opening)
    assert len(products) == 2
    orderbook = create_definite_orderbook(products[0][0], products[-1][0])

    # insert block order in-the-money
    block_order: Order = {
        "start_time": products[0][0],
        "end_time": products[0][1],
        "agent_id": "gen3_block",
        "bid_id": f"bid_{len(orderbook)+1}",
        "profile": {product[0]: 50 for product in products},
        "accepted_profile": {},
        "price": 25,
        "only_hours": None,
        "node_id": 0,
        "bid_type": "BB",
    }

    orderbook.append(block_order)

    # insert block order out-of-the-money
    block_order: Order = {
        "start_time": products[0][0],
        "end_time": products[0][1],
        "agent_id": "gen4_block",
        "bid_id": f"bid_{len(orderbook)+1}",
        "profile": {product[0]: 50 for product in products},
        "accepted_profile": {},
        "price": 150,
        "only_hours": None,
        "node_id": 0,
        "bid_type": "BB",
    }

    orderbook.append(block_order)

    mr.all_orders = orderbook
    accepted_orders, rejected_orders, meta = market_config.market_mechanism(
        mr, products
    )

    assert round(meta[0]["supply_volume"], 3) == 1000
    assert round(meta[0]["demand_volume"], 3) == -1000
    assert round(meta[0]["price"], 3) == 100

    assert accepted_orders[0]["agent_id"] == "dem1"
    assert round(accepted_orders[0]["accepted_volume"], 2) == -1000

    assert accepted_orders[1]["agent_id"] == "gen1"
    assert round(accepted_orders[1]["accepted_volume"], 2) == 50

    assert accepted_orders[2]["agent_id"] == "gen2"
    assert round(accepted_orders[2]["accepted_volume"], 2) == 900

    assert accepted_orders[3]["agent_id"] == "gen3_block"
    assert round(accepted_orders[3]["accepted_profile"][products[0][0]], 2) == 50

    assert rejected_orders[0]["agent_id"] == "gen4_block"
    assert round(rejected_orders[0]["accepted_profile"][products[0][0]], 2) == 0


if __name__ == "__main__":
    pass
    # from assume.common.utils import plot_orderbook
    # clearing_result, meta = test_market_mechanism()
    # only works with per node clearing
    # fig, ax = plot_orderbook(clearing_result, meta)
    # fig.show()
