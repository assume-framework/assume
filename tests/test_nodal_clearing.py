# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
from datetime import datetime, timedelta

import pandas as pd
import pytest
from dateutil import rrule as rr

from assume.common.market_objects import MarketConfig, MarketProduct, Order
from assume.common.utils import get_available_products

try:
    from assume.markets.clearing_algorithms import NodalClearingRole
except ImportError:
    pass

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
