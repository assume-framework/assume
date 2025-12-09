# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
from datetime import datetime, timedelta

import pandas as pd
from dateutil import rrule as rr

from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.utils import get_available_products
from assume.markets.clearing_algorithms import ComplexClearingRole

from .utils import extend_orderbook

simple_dayahead_auction_config = MarketConfig(
    market_id="simple_dayahead_auction",
    market_products=[MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
    additional_fields=["node"],
    opening_hours=rr.rrule(
        rr.HOURLY,
        dtstart=datetime(2005, 6, 1),
        until=datetime(2005, 6, 2),
        cache=True,
    ),
    opening_duration=timedelta(hours=1),
    volume_unit="MW",
    volume_tick=0.1,
    maximum_bid_volume=None,
    price_unit="€/MW",
    market_mechanism="complex_clearing",
)
eps = 1e-4


def test_complex_clearing():
    market_config = simple_dayahead_auction_config
    h = 24
    market_config.market_products = [
        MarketProduct(timedelta(hours=1), h, timedelta(hours=1))
    ]
    market_config.additional_fields = [
        "bid_type",
    ]
    next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
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
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)

    assert math.isclose(meta[0]["supply_volume"], 1000, abs_tol=eps)
    assert math.isclose(meta[0]["demand_volume"], 1000, abs_tol=eps)
    assert math.isclose(meta[0]["price"], 100, abs_tol=eps)
    assert rejected_orders == []
    assert accepted_orders[0]["agent_addr"] == "dem1"
    assert math.isclose(accepted_orders[0]["accepted_volume"], -1000, abs_tol=eps)
    assert accepted_orders[1]["agent_addr"] == f"gen{h + 1}"
    assert math.isclose(accepted_orders[1]["accepted_volume"], 100, abs_tol=eps)
    assert accepted_orders[2]["agent_addr"] == f"gen{2 * h + 1}"
    assert math.isclose(accepted_orders[2]["volume"], 900, abs_tol=eps)


def test_market_coupling():
    market_config = simple_dayahead_auction_config
    h = 2
    market_config.market_products = [
        MarketProduct(timedelta(hours=1), h, timedelta(hours=1))
    ]
    market_config.additional_fields = [
        "bid_type",
        "node_id",
    ]
    # Create a dictionary with the data
    nodes = {
        "name": ["node1", "node2"],
        "v_nom": [380.0, 380.0],
    }

    # Convert the dictionary to a Pandas DataFrame with 'name' as the index
    nodes = pd.DataFrame(nodes).set_index("name")

    # Create a dictionary with the data
    lines = {
        "name": ["line_1"],
        "bus0": ["node1"],
        "bus1": ["node2"],
        "s_nom": [500.0],
    }

    # Convert the dictionary to a Pandas DataFrame
    lines = pd.DataFrame(lines)

    grid_data = {"buses": nodes, "lines": lines}
    market_config.param_dict["grid_data"] = grid_data

    next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
    products = get_available_products(market_config.market_products, next_opening)
    assert len(products) == h

    """
    Create Orderbook with constant order volumes and prices:
        - dem1: volume = -1000, price = 3000, node1
        - dem2: volume = -200, price = 3000, node2
        - gen1: volume = 1000, price = 100, node1
        - gen2: volume = 1000, price = 50, node2
    """
    orderbook = []
    orderbook = extend_orderbook(
        products, volume=-1000, price=3000, orderbook=orderbook, node="node1"
    )
    orderbook = extend_orderbook(
        products, volume=-200, price=3000, orderbook=orderbook, node="node2"
    )
    orderbook = extend_orderbook(products, 1000, 100, orderbook, node="node1")
    orderbook = extend_orderbook(products, 1000, 50, orderbook, node="node2")

    mr = ComplexClearingRole(market_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)

    assert meta[0]["node"] == "node1"
    assert math.isclose(meta[0]["supply_volume"], 500, abs_tol=eps)
    assert math.isclose(meta[0]["demand_volume"], 1000, abs_tol=eps)
    assert math.isclose(meta[0]["price"], 100, abs_tol=eps)

    assert meta[2]["node"] == "node2"
    assert math.isclose(meta[2]["supply_volume"], 700, abs_tol=eps)
    assert math.isclose(meta[2]["demand_volume"], 200, abs_tol=eps)
    assert math.isclose(meta[2]["price"], 50, abs_tol=eps)

    assert rejected_orders == []
    assert accepted_orders[0]["agent_addr"] == "dem1"
    assert math.isclose(accepted_orders[0]["accepted_volume"], -1000, abs_tol=eps)
    assert accepted_orders[1]["agent_addr"] == "dem3"
    assert math.isclose(accepted_orders[1]["accepted_volume"], -200, abs_tol=eps)

    assert accepted_orders[2]["agent_addr"] == "gen5"
    assert math.isclose(accepted_orders[2]["accepted_volume"], 500, abs_tol=eps)
    assert math.isclose(accepted_orders[2]["accepted_price"], 100, abs_tol=eps)

    assert accepted_orders[3]["agent_addr"] == "gen7"
    assert math.isclose(accepted_orders[3]["accepted_volume"], 700, abs_tol=eps)
    assert math.isclose(accepted_orders[3]["accepted_price"], 50, abs_tol=eps)

    market_config.param_dict = {}


def test_market_coupling_with_island():
    market_config = simple_dayahead_auction_config
    h = 2
    market_config.market_products = [
        MarketProduct(timedelta(hours=1), h, timedelta(hours=1))
    ]
    market_config.additional_fields = [
        "bid_type",
        "node_id",
    ]
    # Create a dictionary with the data
    nodes = {
        "name": ["node1", "node2", "node3"],
        "v_nom": [380.0, 380.0, 380.0],
    }

    # Convert the dictionary to a Pandas DataFrame with 'name' as the index
    nodes = pd.DataFrame(nodes).set_index("name")

    # Create a dictionary with the data
    lines = {
        "name": ["line_1"],
        "bus0": ["node1"],
        "bus1": ["node2"],
        "s_nom": [500.0],
    }

    # Convert the dictionary to a Pandas DataFrame
    lines = pd.DataFrame(lines)
    lines.set_index("name", inplace=True)

    grid_data = {"buses": nodes, "lines": lines}
    market_config.param_dict["grid_data"] = grid_data

    next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
    products = get_available_products(market_config.market_products, next_opening)
    assert len(products) == h

    """
    Create Orderbook with constant order volumes and prices:
        - dem1: volume = -1000, price = 3000, node1
        - dem2: volume = -200, price = 3000, node2
        - dem3: volume = -500, price = 3000, node3
        - gen1: volume = 1000, price = 100, node1
        - gen2: volume = 1000, price = 50, node2
        - gen3: volume = 400, price = 75, node3
    """
    orderbook = []
    orderbook = extend_orderbook(
        products, volume=-1000, price=3000, orderbook=orderbook, node="node1"
    )
    orderbook = extend_orderbook(
        products, volume=-200, price=3000, orderbook=orderbook, node="node2"
    )
    orderbook = extend_orderbook(
        products, volume=-500, price=3000, orderbook=orderbook, node="node3"
    )
    orderbook = extend_orderbook(
        products, volume=1000, price=100, orderbook=orderbook, node="node1"
    )
    orderbook = extend_orderbook(
        products, volume=1000, price=50, orderbook=orderbook, node="node2"
    )
    orderbook = extend_orderbook(
        products, volume=400, price=75, orderbook=orderbook, node="node3"
    )

    mr = ComplexClearingRole(market_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)

    assert rejected_orders == []
    assert accepted_orders[0]["node"] == "node1"
    assert math.isclose(accepted_orders[0]["accepted_volume"], -1000, abs_tol=eps)
    assert accepted_orders[1]["node"] == "node2"
    assert math.isclose(accepted_orders[1]["accepted_volume"], -200, abs_tol=eps)
    assert accepted_orders[2]["node"] == "node3"
    assert math.isclose(accepted_orders[2]["accepted_volume"], -400, abs_tol=eps)

    assert accepted_orders[3]["node"] == "node1"
    assert math.isclose(accepted_orders[3]["accepted_volume"], 500, abs_tol=eps)
    assert math.isclose(accepted_orders[3]["accepted_price"], 100, abs_tol=eps)

    assert accepted_orders[4]["node"] == "node2"
    assert math.isclose(accepted_orders[4]["accepted_volume"], 700, abs_tol=eps)
    assert math.isclose(accepted_orders[4]["accepted_price"], 50, abs_tol=eps)

    assert accepted_orders[5]["node"] == "node3"
    assert math.isclose(accepted_orders[5]["accepted_volume"], 400, abs_tol=eps)
    assert math.isclose(accepted_orders[5]["accepted_price"], 3000, abs_tol=eps)

    # add a new line between node2 and node3
    lines = pd.concat(
        [
            lines,
            pd.DataFrame(
                {
                    "name": ["line_2"],
                    "bus0": ["node2"],
                    "bus1": ["node3"],
                    "s_nom": [200.0],
                },
            ).set_index("name"),
        ]
    )
    grid_data = {"buses": nodes, "lines": lines}
    market_config.param_dict["grid_data"] = grid_data

    mr = ComplexClearingRole(market_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)

    assert rejected_orders == []
    assert accepted_orders[0]["node"] == "node1"
    assert math.isclose(accepted_orders[0]["accepted_volume"], -1000, abs_tol=eps)
    assert accepted_orders[1]["node"] == "node2"
    assert math.isclose(accepted_orders[1]["accepted_volume"], -200, abs_tol=eps)
    assert accepted_orders[2]["node"] == "node3"
    assert math.isclose(accepted_orders[2]["accepted_volume"], -500, abs_tol=eps)

    assert accepted_orders[3]["node"] == "node1"
    assert math.isclose(accepted_orders[3]["accepted_volume"], 500, abs_tol=eps)
    assert math.isclose(accepted_orders[3]["accepted_price"], 100, abs_tol=eps)

    assert accepted_orders[4]["node"] == "node2"
    assert math.isclose(accepted_orders[4]["accepted_volume"], 900, abs_tol=eps)
    assert math.isclose(accepted_orders[4]["accepted_price"], 50, abs_tol=eps)

    assert accepted_orders[5]["node"] == "node3"
    assert math.isclose(accepted_orders[5]["accepted_volume"], 300, abs_tol=eps)
    assert math.isclose(accepted_orders[5]["accepted_price"], 75, abs_tol=eps)

    market_config.param_dict = {}


def test_complex_clearing_BB():
    market_config = simple_dayahead_auction_config
    market_config.market_products = [
        MarketProduct(timedelta(hours=1), 2, timedelta(hours=1))
    ]
    market_config.additional_fields = [
        "bid_type",
        "min_acceptance_ratio",
    ]
    next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
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
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)
    # accept only cheapes simple bids
    assert math.isclose(meta[0]["price"], 50, abs_tol=eps)
    assert rejected_orders[1]["agent_addr"] == "block_gen7"
    assert math.isclose(
        rejected_orders[1]["accepted_volume"][products[0][0]], 0, abs_tol=eps
    )
    assert math.isclose(
        rejected_orders[1]["accepted_volume"][products[1][0]], 0, abs_tol=eps
    )
    assert mr.all_orders == []

    # change the price of the block order to be in-the-money
    assert orderbook[3]["agent_addr"] == "block_gen7"
    orderbook[3]["price"] = 45

    mr = ComplexClearingRole(market_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)
    # accept block order and part of cheaper simple order
    assert math.isclose(meta[0]["price"], 50, abs_tol=eps)
    assert accepted_orders[2]["agent_addr"] == "block_gen7"
    assert math.isclose(
        accepted_orders[2]["accepted_volume"][products[0][0]], 100, abs_tol=eps
    )
    assert mr.all_orders == []

    # change price of simple bid to lower the mcp for one hour
    assert orderbook[2]["bid_id"] == "bid_5"
    orderbook[2]["price"] = 41

    mr = ComplexClearingRole(market_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)

    assert math.isclose(meta[0]["price"], 41, abs_tol=eps)
    assert math.isclose(meta[1]["price"], 50, abs_tol=eps)
    # block bid should be accepted, because surplus is 91-90=1
    assert accepted_orders[2]["agent_addr"] == "block_gen7"
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
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)

    assert math.isclose(meta[0]["price"], 39, abs_tol=eps)
    assert math.isclose(meta[1]["price"], 50, abs_tol=eps)
    # block bid should be rejected, because surplus is 89-90=-1
    assert rejected_orders[1]["agent_addr"], "block_gen7"
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
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)
    assert math.isclose(meta[0]["price"], 40, abs_tol=eps)
    assert math.isclose(meta[1]["price"], 50, abs_tol=eps)
    # the acceptance depends on the solver:
    # if solver is glpk, the bid is rejected, if it's gurobi, its accepted

    # assert rejected_orders[1]["agent_addr"] == "block_gen7"
    # assert math.isclose(
    #     rejected_orders[1]["accepted_volume"][products[0][0]], 0, abs_tol=eps
    # )
    # assert math.isclose(
    #     rejected_orders[1]["accepted_price"][products[0][0]], 40, abs_tol=eps
    # )
    assert mr.all_orders == []

    # introducing profile block order by increasing the volume for the hour with a higher mcp
    assert orderbook[3]["agent_addr"] == "block_gen7"
    orderbook[3]["volume"][products[1][0]] = 900

    mr = ComplexClearingRole(market_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)
    assert math.isclose(meta[0]["price"], 40, abs_tol=eps)
    assert math.isclose(meta[1]["price"], 50, abs_tol=eps)
    # block bid should be accepted, because surplus is (40-45)*100+(50-45)*900=4000
    assert accepted_orders[2]["agent_addr"] == "block_gen7"
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


def test_complex_clearing_LB():
    market_config = simple_dayahead_auction_config
    market_config.market_products = [
        MarketProduct(timedelta(hours=1), 2, timedelta(hours=1))
    ]
    market_config.additional_fields = [
        "bid_type",
        "min_acceptance_ratio",
        "parent_bid_id",
    ]
    next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
    products = get_available_products(market_config.market_products, next_opening)
    assert len(products) == 2

    """
    Create Orderbook with constant order volumes and prices:
        - dem1: volume = -1000, price = 3000
        - gen1: volume = 1000, price = 100
        - gen2: volume = 1000, price = 50, linked to block_gen3
        - block_gen3: volume = 100, price = 75
    """
    orderbook = []
    orderbook = extend_orderbook(
        products,
        volume=-999,
        price=3000,
        orderbook=orderbook,
    )
    orderbook = extend_orderbook(products, 1000, 100, orderbook)
    orderbook = extend_orderbook(
        products, 100, 75, orderbook, bid_type="BB", min_acceptance_ratio=1
    )
    orderbook = extend_orderbook(
        products,
        100,
        75,
        orderbook,
        bid_type="LB",
        min_acceptance_ratio=1,
        parent_bid_id=orderbook[-1]["bid_id"],
    )

    mr = ComplexClearingRole(market_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)
    # accept only cheapes simple bids
    assert math.isclose(meta[0]["price"], 100, abs_tol=eps)
    assert rejected_orders == []
    assert mr.all_orders == []

    # change the price of the block order to be out-of-the-money, but saved by child bid
    assert orderbook[2]["agent_addr"] == "block_gen5"
    orderbook[2]["price"] = 120

    mr = ComplexClearingRole(market_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)
    # accept block order and part of cheaper simple order
    assert math.isclose(meta[0]["price"], 100, abs_tol=eps)
    assert rejected_orders == []
    assert mr.all_orders == []

    # change the price of the block order to be out-of-the-money, not saved by child bid
    assert orderbook[2]["agent_addr"] == "block_gen5"
    orderbook[2]["price"] = 130

    mr = ComplexClearingRole(market_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)
    # accept block order and part of cheaper simple order
    assert math.isclose(meta[0]["price"], 100, abs_tol=eps)
    assert rejected_orders[0]["agent_addr"] == "block_gen5"
    assert rejected_orders[1]["agent_addr"] == "block_gen6"

    # change the price of the block order to be in-the-money and child bid out-of-the-money
    assert orderbook[2]["agent_addr"] == "block_gen5"
    orderbook[2]["price"] = 90
    assert orderbook[3]["agent_addr"] == "block_gen6"
    orderbook[3]["price"] = 120

    mr = ComplexClearingRole(market_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)

    assert math.isclose(meta[0]["price"], 100, abs_tol=eps)
    assert rejected_orders[0]["agent_addr"] == "block_gen6"
    assert accepted_orders[2]["agent_addr"] == "block_gen5"

    # add second level of child bids
    orderbook = extend_orderbook(
        products,
        100,
        50,
        orderbook,
        bid_type="LB",
        min_acceptance_ratio=1,
        parent_bid_id=orderbook[3]["bid_id"],
    )

    mr = ComplexClearingRole(market_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)

    assert math.isclose(meta[0]["price"], 100, abs_tol=eps)
    assert rejected_orders == []


if __name__ == "__main__":
    pass
    # from assume.common.utils import plot_orderbook
    # clearing_result, meta, flows = test_market_mechanism()
    # only works with per node clearing
    # fig, ax = plot_orderbook(clearing_result, meta)
    # fig.show()
