# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

import math
from datetime import datetime, timedelta

import pandas as pd
import pytest
from dateutil import rrule as rr

pytest.importorskip("pypsa")

from assume.common.market_objects import MarketConfig, MarketProduct, Order
from assume.common.utils import get_available_products
from assume.markets.clearing_algorithms import NodalClearingRole

simple_nodal_auction_config = MarketConfig(
    market_id="simple_nodal_auction",
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
    market_mechanism="nodal_clearing",
)
eps = 1e-4


@pytest.mark.require_network
def test_nodal_clearing_two_hours():
    market_config = simple_nodal_auction_config
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

    # Create a dictionary with lines data
    lines = {
        "name": ["line_1_2", "line_1_3", "line_2_3"],
        "bus0": ["node1", "node1", "node2"],
        "bus1": ["node2", "node3", "node3"],
        "s_nom": [5000.0, 5000.0, 5000.0],
        "x": [0.01, 0.01, 0.01],
        "r": [0.001, 0.001, 0.001],
    }
    # Convert the dictionary to a Pandas DataFrame
    lines = pd.DataFrame(lines).set_index("name")

    # Create dictionary with generators data
    generators = {
        "name": [f"gen{p}" for p in range(5, 35)],
        "node": ["node1"] * 10 + ["node2"] * 10 + ["node3"] * 10,
        "max_power": [1000.0] * 30,
    }
    generators = pd.DataFrame(generators).set_index("name")
    # Create dictionary with loads data
    loads = {
        "name": ["dem1", "dem2", "dem3"],
        "node": ["node1", "node2", "node3"],
        "max_power": [4400.0, 4400.0, 17400.0],
    }
    loads = pd.DataFrame(loads).set_index("name")

    grid_data = {
        "buses": nodes,
        "lines": lines,
        "generators": generators,
        "loads": loads,
    }
    market_config.param_dict["grid_data"] = grid_data
    market_config.param_dict["log_flows"] = True
    next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
    products = get_available_products(market_config.market_products, next_opening)
    assert len(products) == h

    orderbook = []
    order: Order = {
        "start_time": products[0][0],
        "end_time": products[0][1],
        "unit_id": "dem1",
        "bid_id": "bid1",
        "volume": 0,
        "price": 0,
        "only_hours": None,
        "node": 0,
    }
    i = 0
    for v, p in zip([-2400, -4400], [3000, 3000]):
        new_order = order.copy()
        new_order["start_time"] = products[0][i]
        new_order["end_time"] = products[0][i + 1]
        new_order["volume"] = v
        new_order["price"] = p
        new_order["node"] = "node1"
        new_order["bid_id"] = f"dem1_{i}"
        new_order["unit_id"] = "dem1"
        orderbook.append(new_order)
        i += 1

    i = 0
    for v, p in zip([-2400, -4400], [3000, 3000]):
        new_order = order.copy()
        new_order["start_time"] = products[0][i]
        new_order["end_time"] = products[0][i + 1]
        new_order["volume"] = v
        new_order["price"] = p
        new_order["node"] = "node2"
        new_order["bid_id"] = f"dem2_{i}"
        new_order["unit_id"] = "dem2"
        orderbook.append(new_order)
        i += 1

    i = 0
    for v, p in zip([-17400, -14400], [3000, 3000]):
        new_order = order.copy()
        new_order["start_time"] = products[0][i]
        new_order["end_time"] = products[0][i + 1]
        new_order["volume"] = v
        new_order["price"] = p
        new_order["node"] = "node3"
        new_order["bid_id"] = f"dem3_{i}"
        new_order["unit_id"] = "dem3"
        orderbook.append(new_order)
        i += 1

    for i in range(h):
        for p in range(5, 15):
            new_order = order.copy()
            new_order["start_time"] = products[0][i]
            new_order["end_time"] = products[0][i + 1]
            new_order["volume"] = 1000
            new_order["price"] = p
            new_order["node"] = "node1"
            new_order["bid_id"] = f"gen{p}_{i}"
            new_order["unit_id"] = f"gen{p}"
            orderbook.append(new_order)
        for p in range(15, 25):
            new_order = order.copy()
            new_order["start_time"] = products[0][i]
            new_order["end_time"] = products[0][i + 1]
            new_order["volume"] = 1000
            new_order["price"] = p
            new_order["node"] = "node2"
            new_order["bid_id"] = f"gen{p}_{i}"
            new_order["unit_id"] = f"gen{p}"
            orderbook.append(new_order)
        for p in range(25, 35):
            new_order = order.copy()
            new_order["start_time"] = products[0][i]
            new_order["end_time"] = products[0][i + 1]
            new_order["volume"] = 1000
            new_order["price"] = p
            new_order["node"] = "node3"
            new_order["bid_id"] = f"gen{p}_{i}"
            new_order["unit_id"] = f"gen{p}"
            orderbook.append(new_order)

    mr = NodalClearingRole(market_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)

    assert meta[0]["node"] == "node1"
    assert meta[2]["node"] == "node2"
    assert meta[4]["node"] == "node3"
    assert math.isclose(meta[0]["supply_volume"], 7600, abs_tol=eps)  # node1 hour 0
    assert math.isclose(meta[1]["supply_volume"], 10000, abs_tol=eps)  # node1 hour 1
    assert math.isclose(meta[2]["supply_volume"], 7000, abs_tol=eps)  # node2 hour 0
    assert math.isclose(meta[3]["supply_volume"], 8200, abs_tol=eps)  # node2 hour 1
    assert math.isclose(meta[4]["supply_volume"], 7600, abs_tol=eps)  # node3 hour 0
    assert math.isclose(meta[5]["supply_volume"], 5000, abs_tol=eps)  # node3 hour 1
    assert math.isclose(meta[0]["demand_volume"], 2400, abs_tol=eps)  # node1 hour 0
    assert math.isclose(meta[1]["demand_volume"], 4400, abs_tol=eps)  # node1 hour 1
    assert math.isclose(meta[2]["demand_volume"], 2400, abs_tol=eps)  # node2 hour 0
    assert math.isclose(meta[3]["demand_volume"], 4400, abs_tol=eps)  # node2 hour 1
    assert math.isclose(meta[4]["demand_volume"], 17400, abs_tol=eps)  # node3 hour 0
    assert math.isclose(meta[5]["demand_volume"], 14400, abs_tol=eps)  # node3 hour 1

    assert math.isclose(meta[0]["price"], 12, abs_tol=eps)  # node1 hour 0
    assert math.isclose(meta[1]["price"], 17, abs_tol=eps)  # node1 hour 1
    assert math.isclose(meta[2]["price"], 22, abs_tol=eps)  # node2 hour 0
    assert math.isclose(meta[3]["price"], 23, abs_tol=eps)  # node2 hour 1
    assert math.isclose(meta[4]["price"], 32, abs_tol=eps)  # node3 hour 0
    assert math.isclose(meta[5]["price"], 29, abs_tol=eps)  # node3 hour 1

    flows_df = pd.Series(flows).unstack()
    assert math.isclose(flows_df.loc[products[0][0], "line_1_2"], 200, abs_tol=eps)
    assert math.isclose(flows_df.loc[products[0][1], "line_1_2"], 600, abs_tol=eps)
    assert math.isclose(flows_df.loc[products[0][0], "line_1_3"], 5000, abs_tol=eps)
    assert math.isclose(flows_df.loc[products[0][1], "line_1_3"], 5000, abs_tol=eps)
    assert math.isclose(flows_df.loc[products[0][0], "line_2_3"], 4800, abs_tol=eps)
    assert math.isclose(flows_df.loc[products[0][1], "line_2_3"], 4400, abs_tol=eps)


@pytest.mark.require_network
def test_nodal_clearing_with_storage_single_hour():
    market_config = simple_nodal_auction_config
    h = 1
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

    # Create a dictionary with lines data
    lines = {
        "name": ["line_1_2", "line_1_3", "line_2_3"],
        "bus0": ["node1", "node1", "node2"],
        "bus1": ["node2", "node3", "node3"],
        "s_nom": [5000.0, 5000.0, 5000.0],
        "x": [0.01, 0.01, 0.01],
        "r": [0.001, 0.001, 0.001],
    }
    # Convert the dictionary to a Pandas DataFrame
    lines = pd.DataFrame(lines).set_index("name")

    # Create dictionary with generators data
    generators = {
        "name": [f"gen{p}" for p in range(5, 35)],
        "node": ["node1"] * 10 + ["node2"] * 10 + ["node3"] * 10,
        "max_power": [1000.0] * 30,
    }
    generators = pd.DataFrame(generators).set_index("name")
    # Create dictionary with loads data
    loads = {
        "name": ["dem1", "dem2", "dem3"],
        "node": ["node1", "node2", "node3"],
        "max_power": [4400.0, 4400.0, 17400.0],
    }
    loads = pd.DataFrame(loads).set_index("name")
    # Create dictionary with storage data
    storage_units = {
        "name": ["storage5", "storage50"],
        "node": ["node1", "node3"],
        "max_power_charge": [1000.0, 1000.0],
        "max_power_discharge": [1000.0, 1000.0],
    }
    storage_units = pd.DataFrame(storage_units).set_index("name")

    grid_data = {
        "buses": nodes,
        "lines": lines,
        "generators": generators,
        "loads": loads,
        "storage_units": storage_units,
    }
    market_config.param_dict["grid_data"] = grid_data
    market_config.param_dict["log_flows"] = True
    next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
    products = get_available_products(market_config.market_products, next_opening)
    assert len(products) == h

    orderbook = []
    order: Order = {
        "start_time": products[0][0],
        "end_time": products[0][1],
        "unit_id": "dem1",
        "bid_id": "bid1",
        "volume": 0,
        "price": 0,
        "only_hours": None,
        "node": 0,
    }

    new_order = order.copy()
    new_order["start_time"] = products[0][0]
    new_order["end_time"] = products[0][1]
    new_order["volume"] = -2400
    new_order["price"] = 3000
    new_order["node"] = "node1"
    new_order["bid_id"] = f"dem1_{0}"
    new_order["unit_id"] = "dem1"
    orderbook.append(new_order)

    new_order = order.copy()
    new_order["start_time"] = products[0][0]
    new_order["end_time"] = products[0][1]
    new_order["volume"] = -2400
    new_order["price"] = 3000
    new_order["node"] = "node2"
    new_order["bid_id"] = f"dem2_{0}"
    new_order["unit_id"] = "dem2"
    orderbook.append(new_order)

    new_order = order.copy()
    new_order["start_time"] = products[0][0]
    new_order["end_time"] = products[0][1]
    new_order["volume"] = -16400
    new_order["price"] = 3000
    new_order["node"] = "node3"
    new_order["bid_id"] = f"dem3_{0}"
    new_order["unit_id"] = "dem3"
    orderbook.append(new_order)

    for p in range(5, 15):
        new_order = order.copy()
        new_order["start_time"] = products[0][0]
        new_order["end_time"] = products[0][1]
        new_order["volume"] = 1000
        new_order["price"] = p
        new_order["node"] = "node1"
        new_order["bid_id"] = f"gen{p}_{0}"
        new_order["unit_id"] = f"gen{p}"
        orderbook.append(new_order)
    for p in range(15, 25):
        new_order = order.copy()
        new_order["start_time"] = products[0][0]
        new_order["end_time"] = products[0][1]
        new_order["volume"] = 1000
        new_order["price"] = p
        new_order["node"] = "node2"
        new_order["bid_id"] = f"gen{p}_{0}"
        new_order["unit_id"] = f"gen{p}"
        orderbook.append(new_order)
    for p in range(25, 35):
        new_order = order.copy()
        new_order["start_time"] = products[0][0]
        new_order["end_time"] = products[0][1]
        new_order["volume"] = 1000
        new_order["price"] = p
        new_order["node"] = "node3"
        new_order["bid_id"] = f"gen{p}_{0}"
        new_order["unit_id"] = f"gen{p}"
        orderbook.append(new_order)

    # add storage bids (1000 discharging @ 5 €/MW at node1)
    new_order = order.copy()
    new_order["start_time"] = products[0][0]
    new_order["end_time"] = products[0][1]
    new_order["volume"] = 1000
    new_order["price"] = 5
    new_order["node"] = "node1"
    new_order["bid_id"] = f"discharge{5}_{0}"
    new_order["unit_id"] = f"storage{5}"
    orderbook.append(new_order)
    # add storage bids (1000 charging @ 50 €/MW at node3)
    new_order = order.copy()
    new_order["start_time"] = products[0][0]
    new_order["end_time"] = products[0][1]
    new_order["volume"] = -1000
    new_order["price"] = 50
    new_order["node"] = "node3"
    new_order["bid_id"] = f"charge{50}_{0}"
    new_order["unit_id"] = f"storage{50}"
    orderbook.append(new_order)

    mr = NodalClearingRole(market_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)

    assert meta[0]["node"] == "node1"
    assert meta[1]["node"] == "node2"
    assert meta[2]["node"] == "node3"
    assert math.isclose(meta[0]["supply_volume"], 7600, abs_tol=eps)  # node1 hour 0
    assert math.isclose(meta[1]["supply_volume"], 7000, abs_tol=eps)  # node2 hour 0
    assert math.isclose(meta[2]["supply_volume"], 7600, abs_tol=eps)  # node3 hour 0
    assert math.isclose(meta[0]["demand_volume"], 2400, abs_tol=eps)  # node1 hour 0
    assert math.isclose(meta[1]["demand_volume"], 2400, abs_tol=eps)  # node2 hour 0
    assert math.isclose(meta[2]["demand_volume"], 17400, abs_tol=eps)  # node3 hour 0

    assert math.isclose(meta[0]["price"], 11, abs_tol=eps)  # node1 hour 0
    assert math.isclose(meta[1]["price"], 21.5, abs_tol=eps)  # node2 hour 0
    assert math.isclose(meta[2]["price"], 32, abs_tol=eps)  # node3 hour 0

    flows_df = pd.Series(flows).unstack()
    assert math.isclose(flows_df.loc[products[0][0], "line_1_2"], 200, abs_tol=eps)
    assert math.isclose(flows_df.loc[products[0][0], "line_1_3"], 5000, abs_tol=eps)
    assert math.isclose(flows_df.loc[products[0][0], "line_2_3"], 4800, abs_tol=eps)


def _two_node_grid(generators, loads, storage_units=None):
    """generators and loads map unit name to (node, max_power)."""
    nodes = pd.DataFrame(
        {"name": ["node1", "node2"], "v_nom": [380.0, 380.0]}
    ).set_index("name")
    lines = pd.DataFrame(
        {
            "name": ["line_1_2"],
            "bus0": ["node1"],
            "bus1": ["node2"],
            "s_nom": [1000.0],
            "x": [0.01],
            "r": [0.001],
        }
    ).set_index("name")
    columns = ["node", "max_power"]
    grid_data = {
        "buses": nodes,
        "lines": lines,
        "generators": pd.DataFrame.from_dict(
            generators, orient="index", columns=columns
        ),
        "loads": pd.DataFrame.from_dict(loads, orient="index", columns=columns),
    }
    if storage_units is not None:
        grid_data["storage_units"] = storage_units
    return grid_data


def _order(unit_id, node, product, volume, price, bid_id=None):
    return {
        "start_time": product[0],
        "end_time": product[1],
        "only_hours": None,
        "unit_id": unit_id,
        "bid_id": bid_id or f"{unit_id}_{product[0]}",
        "node": node,
        "volume": volume,
        "price": price,
    }


@pytest.mark.require_network
def test_nodal_clearing_units_without_bids_are_not_dispatched():
    """
    Units in grid_data which do not bid must not be available to the clearing.
    Only gen_bid and dem_bid bid here, so gen_bid has to cover the demand.
    """
    market_config = simple_nodal_auction_config
    market_config.market_products = [
        MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))
    ]
    storage_units = pd.DataFrame(
        {
            "name": ["storage_silent"],
            "node": ["node2"],
            "max_power_charge": [100.0],
            "max_power_discharge": [100.0],
        }
    ).set_index("name")
    market_config.param_dict["grid_data"] = _two_node_grid(
        generators={"gen_bid": ("node1", 200.0), "gen_silent": ("node2", 200.0)},
        loads={"dem_bid": ("node1", 100.0), "dem_silent": ("node2", 50.0)},
        storage_units=storage_units,
    )
    next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
    products = get_available_products(market_config.market_products, next_opening)

    orderbook = [
        _order("gen_bid", "node1", products[0], 200, 50),
        _order("dem_bid", "node1", products[0], -100, 3000),
    ]

    mr = NodalClearingRole(market_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)

    accepted = {o["unit_id"]: o["accepted_volume"] for o in accepted_orders}
    assert math.isclose(accepted.get("gen_bid", 0), 100, abs_tol=eps)
    assert math.isclose(accepted["dem_bid"], -100, abs_tol=eps)
    supply = sum(m["supply_volume"] for m in meta)
    demand = sum(m["demand_volume"] for m in meta)
    assert math.isclose(supply, demand, abs_tol=eps)


@pytest.mark.require_network
def test_nodal_clearing_unit_bidding_in_some_hours_only():
    """
    gen_partial bids only in the first hour. In the second hour it has no bid
    and must not be available, so gen_bid has to cover the demand.
    """
    market_config = simple_nodal_auction_config
    market_config.market_products = [
        MarketProduct(timedelta(hours=1), 2, timedelta(hours=1))
    ]
    market_config.param_dict["grid_data"] = _two_node_grid(
        generators={"gen_bid": ("node1", 200.0), "gen_partial": ("node1", 200.0)},
        loads={"dem_bid": ("node1", 100.0)},
    )
    next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
    products = get_available_products(market_config.market_products, next_opening)

    orderbook = [
        _order("gen_bid", "node1", products[0], 200, 50),
        _order("gen_bid", "node1", products[1], 200, 50),
        _order("gen_partial", "node1", products[0], 200, 10),
        _order("dem_bid", "node1", products[0], -100, 3000),
        _order("dem_bid", "node1", products[1], -100, 3000),
    ]

    mr = NodalClearingRole(market_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)

    accepted = {
        (o["unit_id"], o["start_time"]): o["accepted_volume"] for o in accepted_orders
    }
    assert math.isclose(
        accepted.get(("gen_partial", products[0][0]), 0), 100, abs_tol=eps
    )
    assert math.isclose(accepted.get(("gen_bid", products[1][0]), 0), 100, abs_tol=eps)
    for product in products:
        supply = sum(
            m["supply_volume"] for m in meta if m["product_start"] == product[0]
        )
        demand = sum(
            m["demand_volume"] for m in meta if m["product_start"] == product[0]
        )
        assert math.isclose(supply, demand, abs_tol=eps)


@pytest.mark.require_network
def test_nodal_clearing_multiple_bids_per_unit():
    """
    A unit may place more than one bid for the same product, as the flexable
    strategy for units with a minimum output, the two-price learning strategy
    and the elastic demand strategy do. Every bid has to be cleared on its own
    price and volume.
    """
    market_config = simple_nodal_auction_config
    market_config.market_products = [
        MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))
    ]
    storage_units = pd.DataFrame(
        {
            "name": ["storage_multi"],
            "node": ["node2"],
            "max_power_charge": [100.0],
            "max_power_discharge": [100.0],
        }
    ).set_index("name")
    market_config.param_dict["grid_data"] = _two_node_grid(
        generators={"gen_multi": ("node1", 200.0), "gen_ref": ("node2", 200.0)},
        loads={"dem": ("node1", 100.0)},
        storage_units=storage_units,
    )
    next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
    products = get_available_products(market_config.market_products, next_opening)

    orderbook = [
        _order("gen_multi", "node1", products[0], 50, 10, bid_id="gen_multi_cheap"),
        _order("gen_multi", "node1", products[0], 150, 80, bid_id="gen_multi_dear"),
        _order("gen_ref", "node2", products[0], 200, 50),
        _order(
            "storage_multi", "node2", products[0], 40, 15, bid_id="storage_multi_cheap"
        ),
        _order(
            "storage_multi", "node2", products[0], 60, 70, bid_id="storage_multi_dear"
        ),
        _order("dem", "node1", products[0], -100, 3000),
    ]

    mr = NodalClearingRole(market_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)

    volumes = {
        o["bid_id"]: o["accepted_volume"] for o in accepted_orders + rejected_orders
    }
    # merit order for 100 MW of demand: 50 at 10, 40 at 15, then 10 of gen_ref at 50
    assert math.isclose(volumes["gen_multi_cheap"], 50, abs_tol=eps)
    assert math.isclose(volumes["gen_multi_dear"], 0, abs_tol=eps)
    assert math.isclose(volumes["storage_multi_cheap"], 40, abs_tol=eps)
    assert math.isclose(volumes["storage_multi_dear"], 0, abs_tol=eps)
    assert math.isclose(volumes[f"gen_ref_{products[0][0]}"], 10, abs_tol=eps)
    assert math.isclose(volumes[f"dem_{products[0][0]}"], -100, abs_tol=eps)

    # gen_ref is the marginal unit, so both nodes clear at its bid price
    for m in meta:
        assert math.isclose(m["price"], 50, abs_tol=eps)
    supply = sum(m["supply_volume"] for m in meta)
    demand = sum(m["demand_volume"] for m in meta)
    assert math.isclose(supply, 100, abs_tol=eps)
    assert math.isclose(supply, demand, abs_tol=eps)


@pytest.mark.require_network
def test_nodal_clearing_multiple_bids_over_two_hours():
    """
    A unit may split into several bids in one hour and send a single bid in the
    next, on both the supply and the demand side. The number of bids per unit
    must not change the result for the other hours.
    """
    market_config = simple_nodal_auction_config
    market_config.market_products = [
        MarketProduct(timedelta(hours=1), 2, timedelta(hours=1))
    ]
    market_config.param_dict["grid_data"] = _two_node_grid(
        generators={"gen_cheap": ("node1", 100.0), "gen_multi": ("node1", 200.0)},
        loads={"dem_elastic": ("node1", 300.0)},
    )
    next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
    products = get_available_products(market_config.market_products, next_opening)

    orderbook = [
        _order("gen_cheap", "node1", products[0], 100, 20),
        _order("gen_cheap", "node1", products[1], 100, 20),
        _order("gen_multi", "node1", products[0], 60, 30, bid_id="gen_multi_h0_cheap"),
        _order("gen_multi", "node1", products[0], 140, 90, bid_id="gen_multi_h0_dear"),
        _order("gen_multi", "node1", products[1], 200, 30),
        _order("dem_elastic", "node1", products[0], -100, 100, bid_id="dem_h0_high"),
        _order("dem_elastic", "node1", products[0], -100, 50, bid_id="dem_h0_low"),
        _order("dem_elastic", "node1", products[1], -150, 100),
    ]

    mr = NodalClearingRole(market_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)

    volumes = {
        o["bid_id"]: o["accepted_volume"] for o in accepted_orders + rejected_orders
    }
    # hour 0: 100 at 20 and 60 at 30 are below the second demand bid of 50,
    # the 140 MW at 90 are not, so 160 MW of the 200 MW asked for are served
    assert math.isclose(volumes[f"gen_cheap_{products[0][0]}"], 100, abs_tol=eps)
    assert math.isclose(volumes["gen_multi_h0_cheap"], 60, abs_tol=eps)
    assert math.isclose(volumes["gen_multi_h0_dear"], 0, abs_tol=eps)
    assert math.isclose(volumes["dem_h0_high"], -100, abs_tol=eps)
    assert math.isclose(volumes["dem_h0_low"], -60, abs_tol=eps)
    # hour 1: 150 MW of demand, gen_multi is marginal with its single bid
    assert math.isclose(volumes[f"gen_cheap_{products[1][0]}"], 100, abs_tol=eps)
    assert math.isclose(volumes[f"gen_multi_{products[1][0]}"], 50, abs_tol=eps)
    assert math.isclose(volumes[f"dem_elastic_{products[1][0]}"], -150, abs_tol=eps)

    prices = {(m["node"], m["product_start"]): m["price"] for m in meta}
    # hour 0 is set by the partly served demand bid, hour 1 by gen_multi
    assert math.isclose(prices[("node1", products[0][0])], 50, abs_tol=eps)
    assert math.isclose(prices[("node1", products[1][0])], 30, abs_tol=eps)
    for product in products:
        supply = sum(
            m["supply_volume"] for m in meta if m["product_start"] == product[0]
        )
        demand = sum(
            m["demand_volume"] for m in meta if m["product_start"] == product[0]
        )
        assert math.isclose(supply, demand, abs_tol=eps)
