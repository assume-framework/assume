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


@pytest.fixture
def grid_data_dict_2_nodes():
    return {
        "buses": nodes,
        "lines": lines,
        "generators": generators,
        "loads": demand,
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
    # network snapshots are not indexed by time, but length should match
    assert len(rmr.network.snapshots) == mc.market_products[0].count
    assert rmr.payment_mechanism == "pay_as_bid"  # Default value
    assert rmr.solver_name == "highs"  # Default solver


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


# Storage in redispatch: a cheap northern generator feeds a southern demand across
# an undersized line, so the line is congested. A storage at the southern bus can
# discharge (upward redispatch) to relieve the congestion.
storage_lines = pd.DataFrame(
    {
        "name": ["line_N_S"],
        "bus0": ["N"],
        "bus1": ["S"],
        "s_nom": [100.0],
        "x": [0.01],
        "r": [0.001],
    }
).set_index("name")

storage_generators = pd.DataFrame(
    {
        "name": ["gen_N"],
        "node": ["N"],
        "marginal_cost": [10.0],
        "max_power": [1000.0],
    }
).set_index("name")

storage_demand = pd.DataFrame(
    {
        "name": ["dem_S"],
        "node": ["S"],
        "max_power": [-1000.0],
    }
).set_index("name")

storage_units_df = pd.DataFrame(
    {
        "name": ["storage_S"],
        "node": ["S"],
        "max_power_discharge": [500.0],
        "max_power_charge": [-500.0],
        "capacity": [2000.0],
        "efficiency_charge": [1.0],
        "efficiency_discharge": [1.0],
    }
).set_index("name")

STORAGE_P_NOM = 500.0 - (-500.0)  # full power span sent in the bids


@pytest.fixture
def grid_data_with_storage():
    return {
        "buses": nodes,
        "lines": storage_lines,
        "generators": storage_generators,
        "loads": storage_demand,
        "storage_units": storage_units_df,
    }


def storage_bid(start, end, volume, **fields):
    """
    A complete redispatch bid of the southern storage (see the storage strategy).
    Capacity, SOC limits and efficiencies come from the grid data.
    """
    bid = {
        "start_time": start,
        "end_time": end,
        "only_hours": None,
        "unit_id": "storage_S",
        "bid_id": f"storage_S_{start}",
        "volume": volume,
        "min_power": -500.0,
        "max_power": 500.0,
        "p_nom": STORAGE_P_NOM,
        "price": 20.0,
        "price_discharge": 20.0,
        "price_charge": 15.0,
        "stored_energy_value": 20.0,
        "soc": 0.5,
        "node": "S",
    }
    bid.update(fields)
    return bid


def unit_bid(start, end, unit_id, node, volume, price, min_power=0.0, max_power=1000.0):
    return {
        "start_time": start,
        "end_time": end,
        "only_hours": None,
        "unit_id": unit_id,
        "bid_id": f"{unit_id}_{start}",
        "volume": volume,
        "min_power": min_power,
        "max_power": max_power,
        "p_nom": 1000.0,
        "price": price,
        "node": node,
    }


def test_initialization_with_storage(
    simple_redispatch_market_config, grid_data_with_storage
):
    """Storage is a PyPSA StorageUnit, not a set of generators."""
    mc = simple_redispatch_market_config
    mc.param_dict["grid_data"] = grid_data_with_storage
    rmr = RedispatchMarketRole(mc)

    # generators*3 + nodes*2 backups, no storage generators
    assert len(rmr.network.generators) == len(storage_generators) * 3 + len(nodes) * 2
    assert list(rmr.network.storage_units.index) == ["storage_S"]
    assert rmr.network.storage_units.at["storage_S", "bus"] == "S"


@pytest.mark.parametrize(
    "storage_baseline, gen_cleared, expected_volume, expected_price",
    [
        # idle storage discharges 200 MW more; settled at the discharge price
        (0.0, 300.0, 200.0, 20.0),
        # charging storage (-100) stops charging (charge price 15) and discharges
        # 200 MW (discharge price 20): (100 * 15 + 200 * 20) / 300
        (-100.0, 400.0, 300.0, (100 * 15 + 200 * 20) / 300),
    ],
)
def test_redispatch_with_storage(
    simple_redispatch_market_config,
    grid_data_with_storage,
    storage_baseline,
    gen_cleared,
    expected_volume,
    expected_price,
):
    market_config = simple_redispatch_market_config
    market_config.param_dict["grid_data"] = grid_data_with_storage
    market_config.param_dict["log_flows"] = True

    next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
    products = get_available_products(market_config.market_products, next_opening)
    start, end = products[0][0], products[0][1]

    orderbook = [
        unit_bid(start, end, "gen_N", "N", gen_cleared, 10.0),
        unit_bid(start, end, "dem_S", "S", -300.0, 3000.0, max_power=-1000.0),
        storage_bid(start, end, storage_baseline),
    ]

    rmr = RedispatchMarketRole(market_config)
    accepted_orders, _, meta, flows = rmr.clear(orderbook, products)
    accepted = {o["unit_id"]: o for o in accepted_orders}

    storage = accepted["storage_S"]
    assert storage["accepted_volume"] == pytest.approx(expected_volume, abs=1e-1)
    assert storage["accepted_price"] == pytest.approx(expected_price, abs=1e-3)
    # ... while the cheap northern generator is redispatched downward
    assert accepted["gen_N"]["accepted_volume"] == pytest.approx(
        -expected_volume, abs=1e-1
    )
    # upward and downward redispatch balance out
    assert sum(o["accepted_volume"] for o in accepted_orders) == pytest.approx(
        0.0, abs=1e-1
    )
    # the storage redispatch is part of the meta data of its bus
    meta_s = next(m for m in meta if m["node"] == "S")
    assert meta_s["supply_volume"] == pytest.approx(expected_volume, abs=1e-1)
    # congestion is resolved: the final flow respects the line rating
    flows_df = pd.Series(flows).unstack()
    assert (
        abs(flows_df.loc[0, "line_N_S"]) <= storage_lines.loc["line_N_S", "s_nom"] + eps
    )


# The four moves of a storage around its day-ahead schedule. Storage at bus
# "node", 200 MWh with 100 MWh stored, 100 MW, efficiencies 0.9, value of the
# stored energy 54 EUR/MWh (reference price 60, "sell"): price_discharge 60,
# price_charge 48.6. The line N -> S carries 100 MW; every case needs 20 MW of
# redispatch. gen_N is must-run where the storage has to go down.
FOUR_CASES = [
    # name, node, schedule, gen_N, gen_N min, load S, move, volume, price
    ("discharge_less", "N", 50.0, 70.0, 70.0, 120.0, -20.0, 60.0),
    ("discharge_more", "S", 50.0, 120.0, 0.0, 170.0, 20.0, 60.0),
    ("charge_less", "S", -50.0, 120.0, 0.0, 70.0, 20.0, 48.6),
    ("charge_more", "N", -50.0, 170.0, 170.0, 120.0, -20.0, 48.6),
]


def clear_four_case(market_config, node, schedule, gen_n, gen_n_min, load_s):
    market_config.param_dict["grid_data"] = {
        "buses": nodes,
        "lines": storage_lines,
        "generators": pd.DataFrame(
            {
                "name": ["gen_N", "gen_S"],
                "node": ["N", "S"],
                "marginal_cost": [10.0, 100.0],
                "max_power": [1000.0, 1000.0],
            }
        ).set_index("name"),
        "loads": storage_demand,
        "storage_units": storage_units_df.assign(
            node=node,
            max_power_discharge=100.0,
            max_power_charge=-100.0,
            capacity=200.0,
            efficiency_charge=0.9,
            efficiency_discharge=0.9,
        ),
    }
    start = datetime(2005, 6, 1, 1)
    end = start + timedelta(hours=1)
    orderbook = [
        unit_bid(start, end, "gen_N", "N", gen_n, 10.0, min_power=gen_n_min),
        unit_bid(start, end, "gen_S", "S", 0.0, 100.0),
        unit_bid(start, end, "dem_S", "S", -load_s, 3000.0, max_power=-1000.0),
        storage_bid(
            start,
            end,
            schedule,
            node=node,
            min_power=-100.0,
            max_power=100.0,
            price=60.0,
            price_discharge=60.0,
            price_charge=48.6,
            stored_energy_value=54.0,
        ),
    ]
    rmr = RedispatchMarketRole(market_config)
    accepted_orders, _, _, _ = rmr.clear(orderbook, [(start, end, None)])
    return {o["unit_id"]: o for o in accepted_orders}


@pytest.mark.parametrize(
    "move, node, schedule, gen_n, gen_n_min, load_s, volume, price", FOUR_CASES
)
def test_storage_moves_are_settled_cost_neutral(
    simple_redispatch_market_config,
    move,
    node,
    schedule,
    gen_n,
    gen_n_min,
    load_s,
    volume,
    price,
):
    """
    Every move is settled at its own price, and the payment equals the value of
    the stored energy the storage gains or loses: it ends up neither better nor
    worse off.
    """
    accepted = clear_four_case(
        simple_redispatch_market_config, node, schedule, gen_n, gen_n_min, load_s
    )
    storage = accepted["storage_S"]
    assert storage["accepted_volume"] == pytest.approx(volume, abs=1e-3)
    assert storage["accepted_price"] == pytest.approx(price, abs=1e-3)

    # change of the stored energy (MWh) caused by the move, valued at 54 EUR/MWh
    if move.startswith("discharge"):
        stored = -volume / 0.9
    else:
        stored = -volume * 0.9
    cashflow = storage["accepted_price"] * storage["accepted_volume"]
    assert cashflow + stored * 54.0 == pytest.approx(0.0, abs=1e-2)


def test_storage_bid_without_charge_price_uses_its_price(
    simple_redispatch_market_config, grid_data_with_storage
):
    """
    A storage strategy may send one price: the clearing then uses it for both
    directions. The charging storage (-100) stops charging and discharges 200 MW,
    all settled at 20.
    """
    market_config = simple_redispatch_market_config
    market_config.param_dict["grid_data"] = grid_data_with_storage
    next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
    products = get_available_products(market_config.market_products, next_opening)
    start, end = products[0][0], products[0][1]
    bid = storage_bid(start, end, -100.0)
    del bid["price_charge"], bid["price_discharge"]
    orderbook = [
        unit_bid(start, end, "gen_N", "N", 400.0, 10.0),
        unit_bid(start, end, "dem_S", "S", -300.0, 3000.0, max_power=-1000.0),
        bid,
    ]
    accepted_orders, _, _, _ = RedispatchMarketRole(market_config).clear(
        orderbook, products
    )
    storage = next(o for o in accepted_orders if o["unit_id"] == "storage_S")
    assert storage["accepted_volume"] == pytest.approx(300.0, abs=1e-1)
    assert storage["accepted_price"] == pytest.approx(20.0, abs=1e-3)


def test_storage_downward_moves_do_not_redispatch_beyond_the_congestion(
    simple_redispatch_market_config,
):
    """
    Downward storage moves cost nothing in the optimisation, so the optimiser has
    no reason to charge a storage more than the congestion needs: here the
    flexible cheap plant is turned down instead (its refund makes it the cheaper
    option), and the storage keeps its schedule.
    """
    accepted = clear_four_case(
        simple_redispatch_market_config, "N", -50.0, 170.0, 0.0, 120.0
    )
    assert "storage_S" not in accepted
    assert accepted["gen_N"]["accepted_volume"] == pytest.approx(-20.0, abs=1e-3)


def four_hour_market(market_config, n_hours, storage_units):
    market_config.market_products = [
        MarketProduct(timedelta(hours=1), n_hours, timedelta(hours=1))
    ]
    market_config.param_dict["grid_data"] = {
        "buses": nodes,
        "lines": storage_lines,
        "generators": pd.DataFrame(
            {
                "name": ["gen_N", "gen_S"],
                "node": ["N", "S"],
                "marginal_cost": [10.0, 100.0],
                "max_power": [1000.0, 1000.0],
            }
        ).set_index("name"),
        "loads": storage_demand,
        "storage_units": storage_units,
    }
    next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
    return get_available_products(market_config.market_products, next_opening)


def total_by_unit(accepted_orders):
    volume = {}
    for o in accepted_orders:
        hours = (o["end_time"] - o["start_time"]).total_seconds() / 3600
        volume[o["unit_id"]] = (
            volume.get(o["unit_id"], 0.0) + o["accepted_volume"] * hours
        )
    return volume


@pytest.mark.parametrize(
    "efficiency_discharge, min_soc, expected",
    [
        # 300 MWh stored above min_soc, no losses
        (1.0, 0.0, 300.0),
        # 300 MWh inside the storage give 270 MWh at the grid
        (0.9, 0.0, 270.0),
        # min_soc 0.05 of 2000 MWh keeps 100 MWh: 200 MWh can be used
        (1.0, 0.05, 200.0),
    ],
)
def test_redispatch_storage_energy_limit(
    simple_redispatch_market_config, efficiency_discharge, min_soc, expected
):
    """
    The storage may not deliver more redispatch energy over the horizon than it
    holds. Every hour needs 200 MW of upward redispatch in the south, the storage
    offers 500 MW per hour but holds only 300 MWh (inside the storage), so the
    expensive southern generator has to cover the rest. The state of charge uses
    the exact discharge efficiency and the lower SOC limit.
    """
    n_hours = 4
    market_config = simple_redispatch_market_config
    storages = storage_units_df.assign(
        efficiency_discharge=efficiency_discharge, min_soc=min_soc
    )
    products = four_hour_market(market_config, n_hours, storages)
    soc = 300.0 / 2000.0  # 300 MWh inside the storage

    orderbook = []
    for start, end, _ in products:
        orderbook += [
            unit_bid(start, end, "gen_N", "N", 300.0, 10.0),
            unit_bid(start, end, "gen_S", "S", 0.0, 100.0),
            unit_bid(start, end, "dem_S", "S", -300.0, 3000.0, max_power=-1000.0),
            storage_bid(start, end, 0.0, soc=soc),
        ]

    rmr = RedispatchMarketRole(market_config)
    accepted_orders, _, _, _ = rmr.clear(orderbook, products)
    volume = total_by_unit(accepted_orders)

    assert volume["storage_S"] == pytest.approx(expected, abs=1e-1)
    assert volume["gen_S"] == pytest.approx(n_hours * 200.0 - expected, abs=1e-1)
    assert volume["gen_N"] == pytest.approx(-n_hours * 200.0, abs=1e-1)


def test_redispatch_storage_charges_back_what_it_delivered(
    simple_redispatch_market_config,
):
    """
    The state of charge links the snapshots: an empty storage can only deliver in
    hour 2 what redispatch lets it charge in hour 1. Hour 1 has the opposite
    congestion (too much in the south), hour 2 needs 200 MW more in the south.
    Charging in hour 1 (covered by gen_N at 10) and discharging in hour 2 (20) is
    cheaper than gen_S (100), so the storage charges the full 200 MWh in hour 1 -
    more than the 50 MW the congestion of hour 1 needs.
    """
    market_config = simple_redispatch_market_config
    products = four_hour_market(market_config, 2, storage_units_df)
    (t1, e1, _), (t2, e2, _) = products
    orderbook = [
        # hour 1: southern surplus of 150 MW flows north over the 100 MW line
        unit_bid(t1, e1, "gen_N", "N", 0.0, 10.0),
        unit_bid(t1, e1, "gen_S", "S", 450.0, 100.0, min_power=450.0),
        unit_bid(t1, e1, "dem_S", "S", -300.0, 3000.0, max_power=-1000.0),
        storage_bid(t1, e1, 0.0, soc=0.0),
        # hour 2: northern supply of 300 MW for the south
        unit_bid(t2, e2, "gen_N", "N", 300.0, 10.0),
        unit_bid(t2, e2, "gen_S", "S", 0.0, 100.0),
        unit_bid(t2, e2, "dem_S", "S", -300.0, 3000.0, max_power=-1000.0),
        storage_bid(t2, e2, 0.0, soc=0.0),
    ]
    rmr = RedispatchMarketRole(market_config)
    accepted_orders, _, _, _ = rmr.clear(orderbook, products)
    storage = {
        o["start_time"]: o for o in accepted_orders if o["unit_id"] == "storage_S"
    }
    # hour 1: charge 200 MW more (downward), hour 2: discharge 200 MW more
    assert storage[t1]["accepted_volume"] == pytest.approx(-200.0, abs=1e-1)
    # the storage delivers exactly what it charged in hour 1
    assert storage[t2]["accepted_volume"] == pytest.approx(200.0, abs=1e-1)


def test_redispatch_storage_without_soc_raises(
    simple_redispatch_market_config, grid_data_with_storage
):
    """A storage bid without state of charge cannot be modelled: clear() stops."""
    market_config = simple_redispatch_market_config
    market_config.param_dict["grid_data"] = grid_data_with_storage

    next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
    products = get_available_products(market_config.market_products, next_opening)
    start, end = products[0][0], products[0][1]
    bid = storage_bid(start, end, -50.0)
    del bid["soc"]
    orderbook = [
        unit_bid(start, end, "gen_N", "N", 350.0, 10.0),
        unit_bid(start, end, "dem_S", "S", -300.0, 3000.0, max_power=-1000.0),
        bid,
    ]

    rmr = RedispatchMarketRole(market_config)
    with pytest.raises(ValueError, match="needs 'soc'"):
        rmr.clear(orderbook, products)


def test_redispatch_storage_energy_limit_with_different_durations(
    simple_redispatch_market_config,
):
    """
    Products of different length: 1 h and then 2 h. The storage holds 30 MWh, so
    redispatch_0 * 1 h + redispatch_1 * 2 h must not exceed 30 MWh. Using one
    duration for all snapshots would let it deliver 20 MWh + 10 MW * 2 h = 40 MWh.
    """
    energy_available = 30.0
    market_config = simple_redispatch_market_config
    market_config.market_products = [
        MarketProduct(timedelta(hours=1), 2, timedelta(hours=1))
    ]
    market_config.param_dict["grid_data"] = {
        "buses": nodes,
        "lines": storage_lines,
        "generators": pd.DataFrame(
            {
                "name": ["gen_N", "gen_S"],
                "node": ["N", "S"],
                "marginal_cost": [10.0, 100.0],
                "max_power": [1000.0, 1000.0],
            }
        ).set_index("name"),
        "loads": storage_demand,
        "storage_units": storage_units_df,
    }

    t0 = datetime(2005, 6, 1, 1)
    products = [
        (t0, t0 + timedelta(hours=1), None),
        (t0 + timedelta(hours=1), t0 + timedelta(hours=3), None),
    ]
    orderbook = []
    for start, end, _ in products:
        orderbook += [
            unit_bid(start, end, "gen_N", "N", 120.0, 10.0),
            unit_bid(start, end, "gen_S", "S", 0.0, 100.0),
            unit_bid(start, end, "dem_S", "S", -120.0, 3000.0, max_power=-1000.0),
            storage_bid(start, end, 0.0, soc=energy_available / 2000.0),
        ]

    rmr = RedispatchMarketRole(market_config)
    accepted_orders, _, _, _ = rmr.clear(orderbook, products)

    # 20 MW of congestion in both products: the storage gives exactly its 30 MWh
    assert total_by_unit(accepted_orders)["storage_S"] == pytest.approx(
        energy_available, abs=1e-1
    )
