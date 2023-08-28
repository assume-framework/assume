import math
from datetime import datetime, timedelta

from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd

from assume.common.market_objects import MarketConfig, MarketProduct, Order
from assume.common.utils import get_available_products
from assume.markets import MarketRole, clearing_mechanisms

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
    price_unit="€/MW",
    market_mechanism="pay_as_clear",
)


def test_complex_clearing_whitepaper_a():
    """
    Example taken from Whitepaper electricity spot market design 2030-2050
    Bichler et al.
    2021
    See Figure 5 a)
    """

    import copy

    market_config = copy.copy(simple_dayahead_auction_config)

    market_config.market_mechanism = clearing_mechanisms["pay_as_clear_complex"]
    market_config.market_products = [MarketProduct(rd(hours=+1), 1, rd(hours=1))]
    market_config.additional_fields = [
        "bid_type",
    ]
    mr = MarketRole(market_config)
    next_opening = market_config.opening_hours.after(datetime.now())
    products = get_available_products(market_config.market_products, next_opening)
    assert len(products) == 1

    """
    Create Orderbook with constant (for only one hour) order volumes and prices:
        - dem1: volume = -10, price = 300
        - dem2: volume = -14, price = 10
        - gen3: volume = 1, price = 40
        - block_gen4: volume = 11, price = 40
        - gen5: volume = 13, price = 100

    """
    orderbook = []
    orderbook = extend_orderbook(products, volume=-10, price=300, orderbook=orderbook)
    orderbook = extend_orderbook(products, -14, 10, orderbook)
    orderbook = extend_orderbook(products, 1, 40, orderbook)
    orderbook = extend_orderbook(products, 11, 40, orderbook, "BB")
    orderbook = extend_orderbook(products, 13, 100, orderbook)

    # bid gen1 and block_gen3 are from one unit
    orderbook[3]["agent_id"] = orderbook[2]["agent_id"]
    assert len(orderbook) == 5

    mr.all_orders = orderbook
    accepted_orders, rejected_orders, meta = market_config.market_mechanism(
        mr, products
    )
    assert meta[0]["supply_volume"] == 10
    assert meta[0]["price"] == 40
    assert accepted_orders[0]["agent_id"] == "dem1"
    assert accepted_orders[0]["accepted_volume"] == -10
    assert accepted_orders[1]["agent_id"] == "gen3"
    assert accepted_orders[1]["accepted_volume"] == {products[0][0]: 10}


def test_complex_clearing_whitepaper_d():
    """
    Example taken from Whitepaper electricity spot market design 2030-2050
    Bichler et al.
    2021
    See figure 5 d)
    """

    import copy

    market_config = copy.copy(simple_dayahead_auction_config)

    market_config.market_mechanism = clearing_mechanisms["pay_as_clear_complex"]
    market_config.market_products = [MarketProduct(rd(hours=+1), 1, rd(hours=1))]
    market_config.additional_fields = [
        "bid_type",
        "min_acceptance_ratio",
    ]
    mr = MarketRole(market_config)
    next_opening = market_config.opening_hours.after(datetime.now())
    products = get_available_products(market_config.market_products, next_opening)
    assert len(products) == 1

    """
    Create Orderbook with constant (for only one hour) order volumes and prices:
        - dem1: volume = -10, price = 300
        - dem2: volume = -14, price = 10
        - block_gen4: volume = 12, price = 40, mar=11/12
        - gen5: volume = 13, price = 100

    """
    orderbook = []
    orderbook = extend_orderbook(
        products,
        volume=-10,
        price=300,
        orderbook=orderbook,
        bid_type="SB",
        min_acceptance_ratio=0,
    )
    orderbook = extend_orderbook(products, -14, 10, orderbook, "SB")
    orderbook = extend_orderbook(
        products, 12, 40, orderbook, "BB", min_acceptance_ratio=11 / 12
    )
    orderbook = extend_orderbook(products, 13, 100, orderbook, "SB")

    assert len(orderbook) == 4

    mr.all_orders = orderbook
    accepted_orders, rejected_orders, meta = market_config.market_mechanism(
        mr, products
    )
    assert meta[0]["supply_volume"] == 10
    assert meta[0]["price"] == 100
    assert accepted_orders[0]["agent_id"] == "dem1"
    assert accepted_orders[0]["accepted_volume"] == -10
    assert accepted_orders[1]["agent_id"] == "gen4"
    assert accepted_orders[1]["accepted_volume"] == 10


def test_clearing_non_convex_1():
    """
    Example taken from 'Pricing in non-convex markets: how to price electricity in the presence of demand response'
    Bichler et al.
    2021
    page 25
    5.1.1
    """

    import copy

    market_config = copy.copy(simple_dayahead_auction_config)

    market_config.market_mechanism = clearing_mechanisms["pay_as_clear_complex"]
    market_config.market_products = [MarketProduct(rd(hours=+1), 3, rd(hours=1))]
    market_config.additional_fields = [
        "bid_type",
        "minimum_acceptance_ratio",
    ]
    mr = MarketRole(market_config)
    next_opening = market_config.opening_hours.after(datetime.now())
    products = get_available_products(market_config.market_products, next_opening)
    assert len(products) == 3

    """
    Create Orderbook

    """
    orderbook_demand = []
    order: Order = {
        "start_time": products[0][0],
        "end_time": products[0][1],
        "agent_id": "B1",
        "bid_id": "B1_1",
        "volume": -4,
        "accepted_volume": {},
        "price": 100,
        "accepted_price": {},
        "only_hours": None,
        "bid_type": "SB",
    }

    orderbook_demand.append(order)

    order: Order = {
        "start_time": products[1][0],
        "end_time": products[1][1],
        "agent_id": "B1",
        "bid_id": "B1_2",
        "volume": -6,
        "accepted_volume": {},
        "price": 100,
        "accepted_price": {},
        "only_hours": None,
        "bid_type": "SB",
    }

    orderbook_demand.append(order)

    order: Order = {
        "start_time": products[2][0],
        "end_time": products[2][1],
        "agent_id": "B1",
        "bid_id": "B1_3",
        "volume": -10,
        "accepted_volume": {},
        "price": 100,
        "accepted_price": {},
        "only_hours": None,
        "bid_type": "SB",
    }
    orderbook_demand.append(order)

    order: Order = {
        "start_time": products[0][0],
        "end_time": products[0][1],
        "agent_id": "B2",
        "bid_id": "B2_1",
        "volume": -3,
        "accepted_volume": {},
        "price": 100,
        "accepted_price": {},
        "only_hours": None,
        "bid_type": "SB",
    }
    orderbook_demand.append(order)

    order: Order = {
        "start_time": products[1][0],
        "end_time": products[1][1],
        "agent_id": "B2",
        "bid_id": "B2_2",
        "volume": -6,
        "accepted_volume": {},
        "price": 100,
        "accepted_price": {},
        "only_hours": None,
        "bid_type": "SB",
    }
    orderbook_demand.append(order)

    order: Order = {
        "start_time": products[2][0],
        "end_time": products[2][1],
        "agent_id": "B2",
        "bid_id": "B2_3",
        "volume": -12,
        "accepted_volume": {},
        "price": 100,
        "accepted_price": {},
        "only_hours": None,
        "bid_type": "SB",
    }
    orderbook_demand.append(order)

    orderbook = orderbook_demand.copy()
    orderbook = extend_orderbook(
        products=products, volume=15, price=5, orderbook=orderbook
    )
    orderbook = extend_orderbook(products, 20, 3, orderbook)

    assert len(orderbook) == 12
    assert len(orderbook_demand) == 6

    mr.all_orders = orderbook
    accepted_orders, rejected_orders, meta = market_config.market_mechanism(
        mr, products
    )
    assert meta[0]["supply_volume"] == 7
    assert meta[0]["price"] == 3
    assert meta[1]["supply_volume"] == 12
    assert meta[1]["price"] == 3
    assert meta[2]["supply_volume"] == 22
    assert meta[2]["price"] == 5
    assert accepted_orders[0]["agent_id"] == "B1"
    assert accepted_orders[0]["accepted_volume"] == -4
    assert accepted_orders[1]["agent_id"] == "B2"
    assert accepted_orders[1]["accepted_volume"] == -3
    assert accepted_orders[2]["agent_id"] == "gen10"
    assert accepted_orders[2]["accepted_volume"] == 7
    assert accepted_orders[3]["agent_id"] == "B1"
    assert accepted_orders[3]["accepted_volume"] == -6
    assert accepted_orders[4]["agent_id"] == "B2"
    assert accepted_orders[4]["accepted_volume"] == -6
    assert accepted_orders[5]["agent_id"] == "gen10"
    assert accepted_orders[5]["accepted_volume"] == 12
    assert accepted_orders[6]["agent_id"] == "B1"
    assert accepted_orders[6]["accepted_volume"] == -10
    assert accepted_orders[7]["agent_id"] == "B2"
    assert accepted_orders[7]["accepted_volume"] == -12
    assert accepted_orders[8]["agent_id"] == "gen7"
    assert accepted_orders[8]["accepted_volume"] == 2
    assert accepted_orders[9]["agent_id"] == "gen10"
    assert accepted_orders[9]["accepted_volume"] == 20

    """
    Introduce non-convexities
    5.1.2
    including mar, BB for gen7
    no load costs cannot be integrated here, so the results differ
    """
    orderbook = orderbook_demand.copy()
    orderbook = extend_orderbook(
        products=products,
        volume=15,
        price=5,
        orderbook=orderbook,
        bid_type="BB",
        min_acceptance_ratio=2 / 15,
    )
    orderbook = extend_orderbook(
        products=products,
        volume=20,
        price=3,
        orderbook=orderbook,
        bid_type="SB",
        min_acceptance_ratio=10 / 20,
    )

    assert len(orderbook) == 10

    mr.all_orders = orderbook
    accepted_orders, rejected_orders, meta = market_config.market_mechanism(
        mr, products
    )
    assert meta[0]["supply_volume"] == 7
    assert meta[0]["price"] == 3
    assert meta[1]["supply_volume"] == 12
    assert meta[1]["price"] == 3
    assert meta[2]["supply_volume"] == 22
    # the price is a trade-off for the block bid: since 2MWh were accepted for 3€ in 2 hours
    # 3+3+9 = 5*3
    assert meta[2]["price"] == 9
    assert accepted_orders[0]["agent_id"] == "B1"
    assert accepted_orders[0]["accepted_volume"] == -4
    assert accepted_orders[1]["agent_id"] == "B2"
    assert accepted_orders[1]["accepted_volume"] == -3
    assert accepted_orders[2]["agent_id"] == "block_gen7"
    assert accepted_orders[2]["accepted_volume"] == {
        product[0]: 2 for product in products
    }
    assert accepted_orders[3]["agent_id"] == "gen8"
    assert accepted_orders[3]["accepted_volume"] == 5
    assert accepted_orders[4]["agent_id"] == "B1"
    assert accepted_orders[4]["accepted_volume"] == -6
    assert accepted_orders[5]["agent_id"] == "B2"
    assert accepted_orders[5]["accepted_volume"] == -6
    assert accepted_orders[6]["agent_id"] == "gen8"
    assert accepted_orders[6]["accepted_volume"] == 10
    assert accepted_orders[7]["agent_id"] == "B1"
    assert accepted_orders[7]["accepted_volume"] == -10
    assert accepted_orders[8]["agent_id"] == "B2"
    assert accepted_orders[8]["accepted_volume"] == -12
    assert accepted_orders[9]["agent_id"] == "gen8"
    assert accepted_orders[9]["accepted_volume"] == 20

    """
    5.1.3 price sensitive demand

    half of the demand bids are elastic
    """
    orderbook = [
        order
        for order in orderbook
        if (order["agent_id"] != "B1" and order["agent_id"] != "B2")
    ]
    for order in orderbook_demand.copy():
        order["volume"] = order["volume"] / 2
        orderbook.append(order)
        order_2 = order.copy()
        if order_2["agent_id"] == "B1":
            order_2["price"] = 10
        elif order_2["agent_id"] == "B2":
            order_2["price"] = 2
        order_2["bid_id"] = f"{order['bid_id']}_2"
        orderbook.append(order_2)

    mr.all_orders = orderbook
    accepted_orders, rejected_orders, meta = market_config.market_mechanism(
        mr, products
    )

    assert meta[0]["supply_volume"] == 5.5
    assert meta[0]["price"] == 3
    assert meta[1]["supply_volume"] == 9
    assert meta[1]["price"] == 3
    assert meta[2]["supply_volume"] == 16
    assert meta[2]["price"] == 3

    assert accepted_orders[0]["agent_id"] == "gen8"
    assert accepted_orders[0]["accepted_volume"] == 5.5
    assert accepted_orders[1]["bid_id"] == "B1_1"
    assert accepted_orders[1]["accepted_volume"] == -2
    assert accepted_orders[2]["bid_id"] == "B1_1_2"
    assert accepted_orders[2]["accepted_volume"] == -2
    assert accepted_orders[3]["bid_id"] == "B2_1"
    assert accepted_orders[3]["accepted_volume"] == -1.5
    assert accepted_orders[4]["agent_id"] == "gen8"
    assert accepted_orders[4]["accepted_volume"] == 9
    assert accepted_orders[5]["bid_id"] == "B1_2"
    assert accepted_orders[5]["accepted_volume"] == -3
    assert accepted_orders[6]["bid_id"] == "B1_2_2"
    assert accepted_orders[6]["accepted_volume"] == -3
    assert accepted_orders[7]["bid_id"] == "B2_2"
    assert accepted_orders[7]["accepted_volume"] == -3
    assert accepted_orders[8]["agent_id"] == "gen8"
    assert accepted_orders[8]["accepted_volume"] == 16
    assert accepted_orders[9]["bid_id"] == "B1_3"
    assert accepted_orders[9]["accepted_volume"] == -5
    assert accepted_orders[10]["bid_id"] == "B1_3_2"
    assert accepted_orders[10]["accepted_volume"] == -5
    assert accepted_orders[11]["bid_id"] == "B2_3"
    assert accepted_orders[11]["accepted_volume"] == -6

    """
    5.1.4 Flexible demand
    can only be implemented with exclusive bids
    """


if __name__ == "__main__":
    pass
    # from assume.common.utils import plot_orderbook
    # clearing_result, meta = test_market_mechanism()
    # only works with per node clearing
    # fig, ax = plot_orderbook(clearing_result, meta)
    # fig.show()
