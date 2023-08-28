import math
from datetime import datetime, timedelta

from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd

from assume.common.market_objects import MarketConfig, MarketProduct
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


def test_complex_clearing():
    import copy

    market_config = copy.copy(simple_dayahead_auction_config)

    market_config.market_mechanism = clearing_mechanisms["pay_as_clear_complex"]
    h = 24
    market_config.market_products = [MarketProduct(rd(hours=+1), h, rd(hours=1))]
    mr = MarketRole(market_config)
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
    assert accepted_orders[1]["agent_id"] == f"gen{h+1}"
    assert accepted_orders[1]["accepted_volume"] == 100
    assert accepted_orders[2]["agent_id"] == f"gen{2*h+1}"
    assert accepted_orders[2]["volume"] == 900


def test_complex_clearing_BB():
    import copy

    market_config = copy.copy(simple_dayahead_auction_config)

    market_config.market_mechanism = clearing_mechanisms["pay_as_clear_complex_opt"]
    market_config.market_products = [MarketProduct(rd(hours=+1), 2, rd(hours=1))]
    market_config.additional_fields = [
        "bid_type",
        "accepted_price",
        "profile",
    ]
    mr = MarketRole(market_config)
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
        products, volume=-1000, price=3000, orderbook=orderbook
    )
    orderbook = extend_orderbook(products, 1000, 100, orderbook)
    orderbook = extend_orderbook(products, 1000, 50, orderbook)
    orderbook = extend_orderbook(products, 100, 75, orderbook, "BB")

    mr.all_orders = orderbook
    accepted_orders, rejected_orders, meta = market_config.market_mechanism(
        mr, products
    )

    assert math.isclose(meta[0]["price"], 50)
    assert rejected_orders[1]["agent_id"] == "block_gen7"
    assert math.isclose(rejected_orders[1]["accepted_volume"][products[0][0]], 0)
    assert mr.all_orders == []

    # change the price of the block order to be in-the-money
    orderbook[3]["price"] = 45

    mr.all_orders = orderbook
    accepted_orders, rejected_orders, meta = market_config.market_mechanism(
        mr, products
    )

    assert math.isclose(meta[0]["price"], 50)
    assert accepted_orders[2]["agent_id"] == "block_gen7"
    assert math.isclose(accepted_orders[2]["accepted_volume"][products[0][0]], 100)
    assert mr.all_orders == []

    # change price of simple bid to lower the mcp for one hour
    orderbook[2]["price"] = 41

    mr.all_orders = orderbook
    accepted_orders, rejected_orders, meta = market_config.market_mechanism(
        mr, products
    )

    assert math.isclose(meta[0]["price"], 41)
    assert math.isclose(meta[1]["price"], 50)
    # block bid should be accepted, because surplus is 91-90=1
    assert accepted_orders[2]["agent_id"] == "block_gen7"
    assert math.isclose(accepted_orders[2]["accepted_volume"][products[0][0]], 100)
    assert math.isclose(accepted_orders[2]["accepted_price"][products[0][0]], 41)
    assert mr.all_orders == []

    # change price of simple bid to lower the mcp for one hour even more
    orderbook[2]["price"] = 39

    mr.all_orders = orderbook
    accepted_orders, rejected_orders, meta = market_config.market_mechanism(
        mr, products
    )

    assert math.isclose(meta[0]["price"], 39)
    assert math.isclose(meta[1]["price"], 50)
    # block bid should be rejected, because surplus is 89-90=-1
    assert rejected_orders[1]["agent_id"] == "block_gen7"
    assert math.isclose(rejected_orders[1]["accepted_volume"][products[0][0]], 0)
    assert mr.all_orders == []

    # change price of simple bid to see equilibrium case
    orderbook[2]["price"] = 40

    mr.all_orders = orderbook
    accepted_orders, rejected_orders, meta = market_config.market_mechanism(
        mr, products
    )

    assert math.isclose(meta[0]["price"], 40)
    assert math.isclose(meta[1]["price"], 50)
    # block bid should be rejected, because surplus for block is 90-90=0
    # and general surplus through introduction of block is also 0
    assert rejected_orders[1]["agent_id"] == "block_gen7"
    assert math.isclose(rejected_orders[1]["accepted_volume"][products[0][0]], 0)
    assert mr.all_orders == []

    # introducing profile block order by increasing the volume for the hour with a higher mcp
    orderbook[3]["volume"][products[1][0]] = 900

    mr.all_orders = orderbook
    accepted_orders, rejected_orders, meta = market_config.market_mechanism(
        mr, products
    )
    assert math.isclose(meta[0]["price"], 40)
    assert math.isclose(meta[1]["price"], 50)
    # block bid should be accepted, because surplus is (40-45)*100+(50-45)*900=4000
    assert accepted_orders[2]["agent_id"] == "block_gen7"
    assert math.isclose(accepted_orders[2]["accepted_volume"][products[0][0]], 100)
    assert math.isclose(accepted_orders[2]["accepted_price"][products[0][0]], 40)
    assert math.isclose(accepted_orders[2]["accepted_volume"][products[1][0]], 900)
    assert math.isclose(accepted_orders[2]["accepted_price"][products[1][0]], 50)
    assert mr.all_orders == []

    # what happens, if the block bid volume is higher than total demand?
    # -> opimization fails!
    # orderbook[3]["volume"][products[1][0]] = 1100

    # mr.all_orders = orderbook
    # accepted_orders, rejected_orders, meta = market_config.market_mechanism(
    #     mr, products
    # )


def test_complex_clearing_MAR():
    """
    Example taken from Whitepaper electricity spot market design 2030-2050
    Bichler et al.
    2021
    """

    import copy

    market_config = copy.copy(simple_dayahead_auction_config)

    market_config.market_mechanism = clearing_mechanisms["pay_as_clear_complex_opt_mar"]
    market_config.market_products = [MarketProduct(rd(hours=+1), 1, rd(hours=1))]
    market_config.additional_fields = [
        "bid_type",
        "accepted_price",
        "profile",
        "minimum_acceptance_ratio",
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
        products, volume=-10, price=300, orderbook=orderbook, bid_type="SB", mar=0
    )
    orderbook = extend_orderbook(products, -14, 10, orderbook, "SB", 0)
    orderbook = extend_orderbook(products, 12, 40, orderbook, "BB", 11 / 12)
    orderbook = extend_orderbook(products, 13, 100, orderbook, "SB", 0)

    assert len(orderbook) == 4

    mr.all_orders = orderbook
    accepted_orders, rejected_orders, meta = market_config.market_mechanism(
        mr, products
    )
    assert meta[0]["supply_volume"] == 10


def test_complex_clearing_Bichler_et_al():
    """
    Example taken from Whitepaper electricity spot market design 2030-2050
    Bichler et al.
    2021
    """

    import copy

    market_config = copy.copy(simple_dayahead_auction_config)

    market_config.market_mechanism = clearing_mechanisms["pay_as_clear_complex_opt"]
    market_config.market_products = [MarketProduct(rd(hours=+1), 1, rd(hours=1))]
    market_config.additional_fields = [
        "bid_type",
        "accepted_price",
        "profile",
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
    assert math.isclose(meta[0]["price"], 100)
    assert accepted_orders[0]["agent_id"] == "dem1"
    assert accepted_orders[0]["accepted_volume"] == -10
    assert accepted_orders[1]["agent_id"] == "gen3"
    assert accepted_orders[1]["accepted_volume"] == 1
    assert accepted_orders[2]["agent_id"] == "gen5"
    assert accepted_orders[2]["accepted_volume"] == 9


def test_complex_clearing_Bichler_et_al():
    """
    Example taken from Whitepaper electricity spot market design 2030-2050
    Bichler et al.
    2021
    """

    import copy

    market_config = copy.copy(simple_dayahead_auction_config)

    market_config.market_mechanism = clearing_mechanisms["pay_as_clear_complex_opt"]
    market_config.market_products = [MarketProduct(rd(hours=+1), 1, rd(hours=1))]
    market_config.additional_fields = [
        "bid_type",
        "accepted_price",
        "profile",
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
    assert math.isclose(meta[0]["price"], 100)
    assert accepted_orders[0]["agent_id"] == "dem1"
    assert accepted_orders[0]["accepted_volume"] == -10
    assert accepted_orders[1]["agent_id"] == "gen3"
    assert accepted_orders[1]["accepted_volume"] == 1
    assert accepted_orders[2]["agent_id"] == "gen5"
    assert accepted_orders[2]["accepted_volume"] == 9


if __name__ == "__main__":
    pass
    # from assume.common.utils import plot_orderbook
    # clearing_result, meta = test_market_mechanism()
    # only works with per node clearing
    # fig, ax = plot_orderbook(clearing_result, meta)
    # fig.show()
