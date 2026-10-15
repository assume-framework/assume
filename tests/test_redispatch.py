# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT
from datetime import datetime, timedelta

import pandas as pd
import pytest
from dateutil import rrule as rr

from assume.common.market_objects import MarketConfig, MarketProduct, Order
from assume.common.utils import get_available_products

pytest.importorskip("pypsa")

from assume.markets.clearing_algorithms import RedispatchMarketRole


@pytest.fixture
def simple_redispatch_market_config():
    return MarketConfig(
        market_id="simple_redispatch",
        market_products=[MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
        additional_fields=["node", "min_power", "max_power"],
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
        market_mechanism="redispatch",
    )


eps = 1e-4

nodes = pd.DataFrame(
    {
        "name": ["N", "S"],
        "v_nom": [380.0, 380.0],
    }
).set_index("name")

lines = pd.DataFrame(
    {
        "name": ["line_N_S"],
        "bus0": ["N"],
        "bus1": ["S"],
        "s_nom": [500.0],
        "x": [0.01],
        "r": [0.001],
    }
).set_index("name")

generators = pd.DataFrame(
    {
        "name": ["coal_N", "coal_S", "gas_S"],
        "node": ["N", "S", "S"],
        "marginal_cost": [10.0, 50.0, 100.0],
        "max_power": [1000.0, 1000.0, 1000.0],
    }
).set_index("name")

demand = pd.DataFrame(
    {
        "name": ["dem_S"],
        "node": ["S"],
        "max_power": [-1000.0],
    }
).set_index("name")


flexible_loads = pd.DataFrame(
    {
        "name": ["flex_S"],
        "node": ["S"],
        "p_nom": [500.0],
        "unit_type": ["sector_coupled_system"],
    }
).set_index("name")


@pytest.fixture
def grid_data_dict_2_nodes():
    return {
        "buses": nodes,
        "lines": lines,
        "generators": generators,
        "loads": demand,
    }


@pytest.fixture
def grid_data_dict_2_nodes_with_flexible_load():
    return {
        "buses": nodes,
        "lines": lines,
        "generators": generators,
        "loads": demand,
        "flexible_loads": flexible_loads,
    }


def test_initialization(simple_redispatch_market_config, grid_data_dict_2_nodes):
    """
    Test the successful initialization of RedispatchMarketRole.
    It should correctly set up a PyPSA network when given valid grid data.
    """
    mc = simple_redispatch_market_config
    mc.param_dict["grid_data"] = grid_data_dict_2_nodes
    rmr = RedispatchMarketRole(mc)

    assert rmr.network is not None
    assert len(rmr.network.buses) == len(nodes)
    # network should have: base fixed DA generator + up and down for each generator + backup up and down at each node
    expected_num_generators = len(generators) * 3 + len(nodes) * 2
    assert len(rmr.network.generators) == expected_num_generators
    assert len(rmr.network.loads) == len(demand)
    assert rmr.flexible_loads.empty
    # network snapshots are not indexed by time, but length should match
    assert len(rmr.network.snapshots) == mc.market_products[0].count
    assert rmr.payment_mechanism == "pay_as_bid"  # Default value
    assert rmr.solver_name == "highs"  # Default solver



def test_initialization_adds_flexible_load_assets(
    simple_redispatch_market_config,
    grid_data_dict_2_nodes_with_flexible_load,
):
    """
    Flexible DSM units are static grid assets.

    Each flexible consumer is represented by:
      - one PyPSA Load for its baseline demand,
      - one *_up Generator for load reduction,
      - one *_down Generator for load increase.
    """
    mc = simple_redispatch_market_config
    mc.param_dict["grid_data"] = grid_data_dict_2_nodes_with_flexible_load

    rmr = RedispatchMarketRole(mc)

    assert list(rmr.flexible_loads.index) == ["flex_S"]
    assert "flex_S" in rmr.network.loads.index
    assert "flex_S_up" in rmr.network.generators.index
    assert "flex_S_down" in rmr.network.generators.index

    assert rmr.network.loads.at["flex_S", "bus"] == "S"
    assert rmr.network.generators.at["flex_S_up", "bus"] == "S"
    assert rmr.network.generators.at["flex_S_down", "bus"] == "S"

    assert rmr.network.generators.at["flex_S_up", "p_nom"] == pytest.approx(
        500.0
    )
    assert rmr.network.generators.at["flex_S_down", "p_nom"] == pytest.approx(
        500.0
    )

    # Flexible proxies should be unavailable until clear() applies the bid envelope.
    assert rmr.network.generators_t.p_max_pu["flex_S_up"].iloc[0] == pytest.approx(
        0.0
    )
    assert rmr.network.generators_t.p_max_pu["flex_S_down"].iloc[0] == pytest.approx(
        0.0
    )


def test_initialization_missing_grid_data(simple_redispatch_market_config):
    """
    Test that ValueError is raised when grid_data is missing.
    """
    mc = simple_redispatch_market_config
    mc.param_dict.pop("grid_data", None)

    with pytest.raises(ValueError, match="grid_data is missing"):
        RedispatchMarketRole(mc)


def test_initialization_invalid_payment_mechanism(
    simple_redispatch_market_config, grid_data_dict_2_nodes
):
    """
    Test that ValueError is raised when an invalid payment mechanism is configured.
    """
    mc = simple_redispatch_market_config
    mc.param_dict["grid_data"] = grid_data_dict_2_nodes
    mc.param_dict["payment_mechanism"] = "invalid_mechanism"

    with pytest.raises(ValueError, match="Invalid payment mechanism."):
        RedispatchMarketRole(mc)


# Tests with 2 nodes and 3 generators (coal_N, coal_S, gas_S) and 1 demand (dem_S)
# generators index: ["coal_N", "coal_S", "gas_S"]
# demand index: ["dem_S"]
units_index = ["coal_N", "coal_S", "gas_S", "dem_S"]


@pytest.mark.parametrize(
    "dummy_generation, dummy_demand, expected_up_volume, expected_down_volume, expected_accepted_orders_volume, expected_accepted_orders_price, expected_volume_change, expected_flows",
    [
        (
            pd.Series([0, 1000, 0], index=generators.index),
            -1000,
            [0, 0],
            [0, 0],
            [],
            [],
            0,
            [0],
        ),  # all demand covered by coal_S
        (
            pd.Series([1000, 0, 0], index=generators.index),
            -1000,
            [0, 500],
            [500, 0],
            [-500, 500],
            [10, 50],
            0,
            [500],
        ),  # downwards redispatch of wind_N, upward redispatch of coal_S
        (
            pd.Series([0, 0, 0], index=generators.index),
            -1000,
            [500, 500],
            [0, 0],
            [500, 500],
            [10, 50],
            1000,
            [500],
        ),  # bids do not cover demand -> find cheapest solution
    ],
)
@pytest.mark.require_network
def test_two_nodes_redispatch(
    simple_redispatch_market_config,
    grid_data_dict_2_nodes,
    dummy_generation,
    dummy_demand,
    expected_up_volume,
    expected_down_volume,
    expected_accepted_orders_volume,
    expected_accepted_orders_price,
    expected_volume_change,
    expected_flows,
):
    market_config = simple_redispatch_market_config
    market_config.param_dict["grid_data"] = grid_data_dict_2_nodes
    market_config.param_dict["log_flows"] = True

    next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
    products = get_available_products(market_config.market_products, next_opening)
    order: Order = {
        "start_time": products[0][0],
        "end_time": products[0][1],
        "unit_id": "x",
        "bid_id": "y",
        "volume": 0,
        "min_power": 0,
        "max_power": 0,
        "price": 0,
        "only_hours": None,
        "node": 0,
    }

    assert len(products) == 1
    orderbook = []
    for _ in generators.index:
        supply_bid = order.copy()
        supply_bid["unit_id"] = _
        supply_bid["volume"] = dummy_generation[_]
        supply_bid["min_power"] = 0
        supply_bid["max_power"] = generators.max_power[_]
        supply_bid["p_nom"] = generators.max_power[_]
        supply_bid["price"] = generators.marginal_cost[_]
        supply_bid["node"] = generators.node[_]
        supply_bid["bid_id"] = _ + "_bid"
        orderbook.append(supply_bid)

    for _ in demand.index:
        demand_bid = order.copy()
        demand_bid["unit_id"] = _
        demand_bid["volume"] = dummy_demand
        demand_bid["min_power"] = 0
        demand_bid["max_power"] = demand.max_power[_]
        demand_bid["price"] = 1000.0
        demand_bid["p_nom"] = demand.max_power[_]
        demand_bid["node"] = demand.node[_]
        demand_bid["bid_id"] = _ + "_bid"
        orderbook.append(demand_bid)

    assert len(orderbook) == len(generators) + len(demand)

    rmr = RedispatchMarketRole(market_config)
    accepted_orders, _, meta, flows = rmr.clear(orderbook, products)

    assert "supply_volume" in meta[0]
    assert "supply_volume" in meta[1]
    assert "demand_volume" in meta[0]
    assert "demand_volume" in meta[1]

    assert [meta[i]["supply_volume"] for i in range(len(nodes))] == pytest.approx(
        expected_up_volume
    )
    assert [meta[i]["demand_volume"] for i in range(len(nodes))] == pytest.approx(
        expected_down_volume
    )

    assert [o["accepted_volume"] for o in accepted_orders] == pytest.approx(
        expected_accepted_orders_volume
    )
    assert [o["accepted_price"] for o in accepted_orders] == pytest.approx(
        expected_accepted_orders_price
    )
    assert sum([o["accepted_volume"] for o in accepted_orders]) == pytest.approx(
        expected_volume_change
    )
    # check flows
    flows_df = pd.Series(flows).unstack()
    assert flows_df.loc[0, "line_N_S"] == pytest.approx(expected_flows[0], abs=eps)



@pytest.mark.require_network
def test_flexible_load_participates_in_redispatch(
    simple_redispatch_market_config,
    grid_data_dict_2_nodes_with_flexible_load,
):
    """
    The flexible load at node S reduces consumption when that is the cheapest
    way to relieve the N->S congestion.

    Baseline:
      coal_N = 1000 MW
      fixed load at S = 500 MW
      flexible load at S = 500 MW
      line limit = 500 MW

    Expected redispatch:
      coal_N down by 500 MW
      flex_S load reduction by 500 MW
    """
    mc = simple_redispatch_market_config
    mc.param_dict["grid_data"] = grid_data_dict_2_nodes_with_flexible_load
    mc.param_dict["log_flows"] = True

    next_opening = mc.opening_hours.after(datetime(2005, 6, 1))
    products = get_available_products(mc.market_products, next_opening)
    product = products[0]

    def order(unit_id, volume, min_power, max_power, p_nom, price, node):
        return {
            "start_time": product[0],
            "end_time": product[1],
            "only_hours": product[2],
            "unit_id": unit_id,
            "bid_id": f"{unit_id}_bid",
            "volume": volume,
            "min_power": min_power,
            "max_power": max_power,
            "p_nom": p_nom,
            "price": price,
            "node": node,
        }

    orderbook = [
        order("coal_N", 1000.0, 0.0, 1000.0, 1000.0, 10.0, "N"),
        order("coal_S", 0.0, 0.0, 1000.0, 1000.0, 50.0, "S"),
        order("gas_S", 0.0, 0.0, 1000.0, 1000.0, 100.0, "S"),
        # Ordinary load: fixed in redispatch.
        order("dem_S", -500.0, 0.0, -1000.0, -1000.0, 1000.0, "S"),
        # Flexible load: physical consumption may move from 500 MW to 0 MW.
        order("flex_S", -500.0, -500.0, 0.0, 500.0, 1.0, "S"),
    ]

    rmr = RedispatchMarketRole(mc)
    accepted_orders, _, _, flows = rmr.clear(orderbook, products)

    accepted = {
        item["unit_id"]: item["accepted_volume"]
        for item in accepted_orders
    }

    assert accepted["coal_N"] == pytest.approx(-500.0, abs=eps)
    assert accepted["flex_S"] == pytest.approx(500.0, abs=eps)
    assert sum(accepted.values()) == pytest.approx(0.0, abs=eps)

    flows_df = pd.Series(flows).unstack()
    assert flows_df.loc[0, "line_N_S"] == pytest.approx(500.0, abs=eps)


def test_component_level_sector_coupled_bid_is_rejected(
    simple_redispatch_market_config,
    grid_data_dict_2_nodes_with_flexible_load,
):
    """
    The current redispatch market supports the aggregate parent representation.
    Independently clearing child/component bids is deliberately blocked until
    coupling constraints are implemented.
    """
    mc = simple_redispatch_market_config

    sector_flexible_load = pd.DataFrame(
        {
            "node": ["S"],
            "p_nom": [500.0],
            "unit_type": ["sector_coupled_system"],
        },
        index=["hs_south"],
    )

    grid_data = {
        **grid_data_dict_2_nodes_with_flexible_load,
        "flexible_loads": sector_flexible_load,
    }
    mc.param_dict["grid_data"] = grid_data

    next_opening = mc.opening_hours.after(datetime(2005, 6, 1))
    products = get_available_products(mc.market_products, next_opening)
    product = products[0]

    component_order = {
        "start_time": product[0],
        "end_time": product[1],
        "only_hours": product[2],
        "unit_id": "hs_south_heat_pump",
        "parent_unit_id": "hs_south",
        "bid_id": "hs_south_heat_pump_bid",
        "volume": -250.0,
        "min_power": -500.0,
        "max_power": 0.0,
        "p_nom": 500.0,
        "price": 10.0,
        "node": "S",
    }

    rmr = RedispatchMarketRole(mc)

    with pytest.raises(
        NotImplementedError,
        match="Component-level redispatch clearing is not supported yet",
    ):
        rmr.clear([component_order], products)


# Tests with 3 nodes and 30 generators (gen5 to gen34) and 3 demand (dem1 to dem3)
nodes_3 = pd.DataFrame(
    {
        "name": ["node1", "node2", "node3"],
        "v_nom": [380.0, 380.0, 380.0],
    }
).set_index("name")

lines_3 = pd.DataFrame(
    {
        "name": ["line_1_2", "line_1_3", "line_2_3"],
        "bus0": ["node1", "node1", "node2"],
        "bus1": ["node2", "node3", "node3"],
        "s_nom": [5000.0, 5000.0, 5000.0],
        "x": [0.01, 0.01, 0.01],
        "r": [0.001, 0.001, 0.001],
    }
).set_index("name")

generators_3 = pd.DataFrame(
    {
        "name": [f"gen{p}" for p in range(5, 35)],
        "node": ["node1"] * 10 + ["node2"] * 10 + ["node3"] * 10,
        "marginal_cost": list(range(5, 15)) + list(range(15, 25)) + list(range(25, 35)),
        "max_power": [1000.0] * 30,
    }
).set_index("name")

demand_3 = pd.DataFrame(
    {
        "name": ["dem1", "dem2", "dem3"],
        "node": ["node1", "node2", "node3"],
        "marginal_cost": [3000.0, 3000.0, 3000.0],
        "max_power": [4400.0, 4400.0, 17400.0],
    }
).set_index("name")


@pytest.fixture
def grid_data_dict_3_nodes():
    return {
        "buses": nodes_3,
        "lines": lines_3,
        "generators": generators_3,
        "loads": demand_3,
    }


# node1: marginal cost from 5 to 14
# node2: marginal cost from 15 to 24
# node3: marginal cost from 25 to 34
@pytest.mark.parametrize(
    "dummy_generation_3, dummy_demand_3, expected_up_volume_3, expected_down_volume_3, expected_accepted_orders_volume_3, expected_accepted_nodal_price_3, expected_volume_change_3, expected_flows_3",
    [
        (
            # nodal clearing market outcome hour 0, no redispatch needed
            pd.Series(
                [1000] * 7
                + [600]
                + [0] * 2
                + [1000] * 7
                + [0] * 3
                + [1000] * 7
                + [600]
                + [0] * 2,
                index=generators_3.index,
            ),
            pd.Series([-2400, -2400, -17400], index=demand_3.index),
            [0, 0, 0],  # no up volume
            [0, 0, 0],  # no down volume
            [],  # no accepted orders
            [0, 0, 0],  # no prices
            0,  # no volume change
            [200, 5000, 4800],  # flows on lines
        ),
        (
            # copper plate market outcome hour 1
            # there are 2 equally expensive solutions to the problem
            pd.Series(
                [1000] * 10 + [1000] * 10 + [1000] * 2 + [200] + [0] * 7,
                index=generators_3.index,
            ),
            pd.Series([-2400, -2400, -17400], index=demand_3.index),
            [0, 0, 5200],  # up at node 3
            # [0, 0, 5400], # alternative up
            [2600, 2600, 0],  # down at nodes 1 and 2
            # [2400, 3000, 0], # alternative down
            [-600, -1000, -1000, -600, -1000, -1000, 800, 1000, 1000, 1000, 1000, 400],
            [12, 22, 32],  # nodal prices emerge similar to nodal clearing solution
            0,  # no volume change, as up and down volumes are equal
            [0, 5000, 5000],  # flows same as in nodal clearing solution
            # [200, 5000, 4800], # alternative flows with alternative up and down volumes
        ),
        (
            # copper plate market outcome hour 2
            pd.Series([1000] * 23 + [200] + [0] * 6, index=generators_3.index),
            pd.Series([-4400, -4400, -14400], index=demand_3.index),
            [0, 0, 1800],  # up at node 3
            [0, 1800, 0],  # down at node 2
            [-800, -1000, 800, 1000],
            [17, 23, 29],  # nodal prices similar to nodal clearing solution
            0,  # no volume change
            [600, 5000, 4400],
        ),
        (
            # EOM did not provide enough energy to cover demand, redispatch to find cheapest solution (equal to nodal clearing)
            pd.Series(
                [1000] + [0] * 9 + [200] + [0] * 9 + [0] * 10, index=generators_3.index
            ),  # generation of 1200 MW
            pd.Series(
                [-4400, -4400, -14400], index=demand_3.index
            ),  # demand of 23200 MW
            [9000, 8000, 5000],  # up at all nodes
            [0, 0, 0],  # down at no nodes
            [1000] * 9 + [800] + [1000] * 7 + [200] + [1000] * 5,
            [17, 23, 29],  # nodal prices similar to nodal clearing solution
            22000,  # + 22000 MW to match demand of 23200 MW
            [600, 5000, 4400],
        ),  # flows
    ],
)
def test_three_nodes_redispatch(
    grid_data_dict_3_nodes,
    simple_redispatch_market_config,
    dummy_generation_3,
    dummy_demand_3,
    expected_up_volume_3,
    expected_down_volume_3,
    expected_accepted_orders_volume_3,
    expected_accepted_nodal_price_3,
    expected_volume_change_3,
    expected_flows_3,
):
    market_config = simple_redispatch_market_config
    market_config.param_dict["grid_data"] = grid_data_dict_3_nodes
    market_config.param_dict["log_flows"] = True
    h = 1
    market_config.market_products = [
        MarketProduct(timedelta(hours=1), h, timedelta(hours=1))
    ]
    next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
    products = get_available_products(market_config.market_products, next_opening)
    order: Order = {
        "start_time": products[0][0],
        "end_time": products[0][1],
        "unit_id": "x",
        "bid_id": "y",
        "volume": 0,
        "min_power": 0,
        "max_power": 0,
        "price": 0,
        "only_hours": None,
        "node": 0,
    }

    assert len(products) == h
    orderbook = []

    for _ in generators_3.index:
        supply_bid = order.copy()
        supply_bid["unit_id"] = _
        supply_bid["volume"] = dummy_generation_3[_]
        supply_bid["min_power"] = 0
        supply_bid["max_power"] = generators_3.max_power[_]
        supply_bid["p_nom"] = generators_3.max_power[_]
        supply_bid["price"] = generators_3.marginal_cost[_]
        supply_bid["node"] = generators_3.node[_]
        supply_bid["bid_id"] = _ + "_bid"
        orderbook.append(supply_bid)

    for _ in demand_3.index:
        demand_bid = order.copy()
        demand_bid["unit_id"] = _
        demand_bid["volume"] = dummy_demand_3[_]
        demand_bid["min_power"] = 0
        demand_bid["max_power"] = demand_3.max_power[_]
        demand_bid["price"] = 1000.0
        demand_bid["p_nom"] = demand_3.max_power[_]
        demand_bid["node"] = demand_3.node[_]
        demand_bid["bid_id"] = _ + "_bid"
        orderbook.append(demand_bid)

    assert len(orderbook) == (len(generators_3) + len(demand_3)) * h

    rmr = RedispatchMarketRole(market_config)
    accepted_orders, rejected_orders, meta, flows = rmr.clear(orderbook, products)

    assert meta[0]["node"] == "node1"
    assert meta[1]["node"] == "node2"
    assert meta[2]["node"] == "node3"

    assert [meta[i]["supply_volume"] for i in range(len(nodes_3))] == pytest.approx(
        expected_up_volume_3
    )
    assert [meta[i]["demand_volume"] for i in range(len(nodes_3))] == pytest.approx(
        expected_down_volume_3
    )
    assert [meta[i]["price"] for i in range(len(nodes_3))] == pytest.approx(
        expected_accepted_nodal_price_3
    )

    assert [o["accepted_volume"] for o in accepted_orders] == pytest.approx(
        expected_accepted_orders_volume_3
    )
    assert sum([o["accepted_volume"] for o in accepted_orders]) == pytest.approx(
        expected_volume_change_3
    )
    # check flows
    flows_df = pd.Series(flows).unstack()
    assert flows_df.loc[0, "line_1_2"] == pytest.approx(expected_flows_3[0], abs=eps)
    assert flows_df.loc[0, "line_1_3"] == pytest.approx(expected_flows_3[1], abs=eps)
    assert flows_df.loc[0, "line_2_3"] == pytest.approx(expected_flows_3[2], abs=eps)
