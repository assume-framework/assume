# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
from datetime import datetime, timedelta

from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd

from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.utils import get_available_products
from assume.markets.clearing_algorithms import ComplexClearingRole

from .utils import extend_orderbook

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
    maximum_bid_volume=None,
    price_unit="€/MW",
    market_mechanism="pay_as_clear_complex",
)
eps = 1e-4


def test_complex_clearing():
    import copy

    market_config = copy.copy(simple_dayahead_auction_config)
    h = 24
    market_config.market_products = [MarketProduct(rd(hours=+1), h, rd(hours=1))]
    market_config.additional_fields = [
        "bid_type",
    ]
    next_opening = market_config.opening_hours.after(datetime.now())
    products = get_available_products(market_config.market_products, next_opening)
    assert len(products) == h

    """
    Create Orderbook with constant order volumes and prices:
        - dem1: volume = -1000, price = 3000
        - gen1: volume = 1000, price = 100
        - gen2: volume = 900, price = 50
    """
    orderbook = []
    orderbook = extend_orderbook(products, -1000, 3000, orderbook)
    orderbook = extend_orderbook(products, 1000, 100, orderbook)
    orderbook = extend_orderbook(products, 900, 50, orderbook)

    mr = ComplexClearingRole(market_config)
    accepted_orders, rejected_orders, meta = mr.clear(orderbook, products)

    assert math.isclose(meta[0]["supply_volume"], 1000, abs_tol=eps)
    assert math.isclose(meta[0]["demand_volume"], 1000, abs_tol=eps)
    assert math.isclose(meta[0]["price"], 100, abs_tol=eps)
    assert rejected_orders == []
    assert accepted_orders[0]["agent_id"] == "dem1"
    assert math.isclose(accepted_orders[0]["accepted_volume"], -1000, abs_tol=eps)
    assert accepted_orders[1]["agent_id"] == f"gen{h+1}"
    assert math.isclose(accepted_orders[1]["accepted_volume"], 100, abs_tol=eps)
    assert accepted_orders[2]["agent_id"] == f"gen{2*h+1}"
    assert math.isclose(accepted_orders[2]["volume"], 900, abs_tol=eps)


def test_complex_clearing_BB():
    import copy

    market_config = copy.copy(simple_dayahead_auction_config)
    market_config.market_products = [MarketProduct(rd(hours=+1), 2, rd(hours=1))]
    market_config.additional_fields = [
        "bid_type",
        "min_acceptance_ratio",
    ]
    next_opening = market_config.opening_hours.after(datetime.now())
    products = get_available_products(market_config.market_products, next_opening)
    assert len(products) == 2

    """
    Create Orderbook with constant order volumes and prices:
        - dem1: volume = -1000, price = 3000
        - gen1: volume = 1000, price = 100
        - gen2: volume = 1000, price = 50
        - block_gen3: volume = 100, price = 75
    """
    orderbook = []
    orderbook = extend_orderbook(
        products,
        volume=-1000,
        price=3000,
        orderbook=orderbook,
    )
    orderbook = extend_orderbook(products, 1000, 100, orderbook)
    orderbook = extend_orderbook(products, 1000, 50, orderbook)
    orderbook = extend_orderbook(
        products, 100, 75, orderbook, bid_type="BB", min_acceptance_ratio=1
    )

    mr = ComplexClearingRole(market_config)
    accepted_orders, rejected_orders, meta = mr.clear(orderbook, products)
    # accept only cheapes simple bids
    assert math.isclose(meta[0]["price"], 50, abs_tol=eps)
    assert rejected_orders[1]["agent_id"] == "block_gen7"
    assert math.isclose(
        rejected_orders[1]["accepted_volume"][products[0][0]], 0, abs_tol=eps
    )
    assert math.isclose(
        rejected_orders[1]["accepted_volume"][products[1][0]], 0, abs_tol=eps
    )
    assert mr.all_orders == []

    # change the price of the block order to be in-the-money
    assert orderbook[3]["agent_id"] == "block_gen7"
    orderbook[3]["price"] = 45

    mr = ComplexClearingRole(market_config)
    accepted_orders, rejected_orders, meta = mr.clear(orderbook, products)
    # accept block order and part of cheaper simple order
    assert math.isclose(meta[0]["price"], 50, abs_tol=eps)
    assert accepted_orders[2]["agent_id"] == "block_gen7"
    assert math.isclose(
        accepted_orders[2]["accepted_volume"][products[0][0]], 100, abs_tol=eps
    )
    assert mr.all_orders == []

    # change price of simple bid to lower the mcp for one hour
    assert orderbook[2]["bid_id"] == "bid_5"
    orderbook[2]["price"] = 41

    mr = ComplexClearingRole(market_config)
    accepted_orders, rejected_orders, meta = mr.clear(orderbook, products)

    assert math.isclose(meta[0]["price"], 41, abs_tol=eps)
    assert math.isclose(meta[1]["price"], 50, abs_tol=eps)
    # block bid should be accepted, because surplus is 91-90=1
    assert accepted_orders[2]["agent_id"] == "block_gen7"
    assert math.isclose(
        accepted_orders[2]["accepted_volume"][products[0][0]], 100, abs_tol=eps
    )
    assert math.isclose(
        accepted_orders[2]["accepted_price"][products[0][0]], 41, abs_tol=eps
    )
    assert mr.all_orders == []

    # change price of simple bid to lower the mcp for one hour even more
    assert orderbook[2]["bid_id"] == "bid_5"
    orderbook[2]["price"] = 39

    mr = ComplexClearingRole(market_config)
    accepted_orders, rejected_orders, meta = mr.clear(orderbook, products)

    assert math.isclose(meta[0]["price"], 39, abs_tol=eps)
    assert math.isclose(meta[1]["price"], 50, abs_tol=eps)
    # block bid should be rejected, because surplus is 89-90=-1
    assert rejected_orders[1]["agent_id"], "block_gen7"
    assert math.isclose(
        rejected_orders[1]["accepted_volume"][products[0][0]], 0, abs_tol=eps
    )
    assert math.isclose(
        rejected_orders[1]["accepted_volume"][products[1][0]], 0, abs_tol=eps
    )
    assert mr.all_orders == []

    # change price of simple bid to see equilibrium case
    assert orderbook[2]["bid_id"] == "bid_5"
    orderbook[2]["price"] = 40

    mr = ComplexClearingRole(market_config)
    accepted_orders, rejected_orders, meta = mr.clear(orderbook, products)
    assert math.isclose(meta[0]["price"], 40, abs_tol=eps)
    assert math.isclose(meta[1]["price"], 50, abs_tol=eps)
    # the acceptance depends on the solver:
    # if solver is glpk, the bid is rejected, if it's gurobi, its accepted

    # assert rejected_orders[1]["agent_id"] == "block_gen7"
    # assert math.isclose(
    #     rejected_orders[1]["accepted_volume"][products[0][0]], 0, abs_tol=eps
    # )
    # assert math.isclose(
    #     rejected_orders[1]["accepted_price"][products[0][0]], 40, abs_tol=eps
    # )
    assert mr.all_orders == []

    # introducing profile block order by increasing the volume for the hour with a higher mcp
    assert orderbook[3]["agent_id"] == "block_gen7"
    orderbook[3]["volume"][products[1][0]] = 900

    mr = ComplexClearingRole(market_config)
    accepted_orders, rejected_orders, meta = mr.clear(orderbook, products)
    assert math.isclose(meta[0]["price"], 40, abs_tol=eps)
    assert math.isclose(meta[1]["price"], 50, abs_tol=eps)
    # block bid should be accepted, because surplus is (40-45)*100+(50-45)*900=4000
    assert accepted_orders[2]["agent_id"] == "block_gen7"
    assert math.isclose(
        accepted_orders[2]["accepted_volume"][products[0][0]], 100, abs_tol=eps
    )
    assert math.isclose(
        accepted_orders[2]["accepted_price"][products[0][0]], 40, abs_tol=eps
    )
    assert math.isclose(
        accepted_orders[2]["accepted_volume"][products[1][0]], 900, abs_tol=eps
    )
    assert math.isclose(
        accepted_orders[2]["accepted_price"][products[1][0]], 50, abs_tol=eps
    )
    assert mr.all_orders == []


if __name__ == "__main__":
    pass
    # from assume.common.utils import plot_orderbook
    # clearing_result, meta = test_market_mechanism()
    # only works with per node clearing
    # fig, ax = plot_orderbook(clearing_result, meta)
    # fig.show()
