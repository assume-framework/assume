# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import pandas as pd
import pytest
from dateutil import rrule as rr

from assume.common.fast_pandas import FastIndex, FastSeries
from assume.common.market_objects import MarketConfig, MarketProduct, Order
# from assume.common.grid_utils import get_supported_solver_linopy
from assume.common.utils import get_available_products
from assume.scenario.loader_csv import make_market_config
from tests.test_exchanges import market_config
try:
    from assume.markets.clearing_algorithms import RedispatchMarketRole
except ImportError:
    pass


def test_initialization():
    pass

def test_clear():
    pass

def test_process_dispatch_data():
    pass


simple_zonal_market_config = MarketConfig(
    market_id="simple_zonal_auction",
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
    price_unit="€/MW",
    market_mechanism="complex_clearing",
)
simple_redispatch_market_config = MarketConfig(
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

nodes = pd.DataFrame({
        "name": ["N", "S"],
        "v_nom": [380.0, 380.0],
    }).set_index("name")
lines = pd.DataFrame({
        "name": ["line_N_S"],
        "bus0": ["N"],
        "bus1": ["S"],
        "s_nom": [2000.0],
        "x": [0.01],
        "r": [0.001],
    }).set_index("name")
generators = pd.DataFrame({
        "name": ["coal_N", "coal_S", "gas_S"],
        "node": ["N", "S", "S"],
        "marginal_cost": [10.0, 50.0, 100.0],
        "max_power": [1000.0, 1000.0, 1000.0],
    }).set_index("name")
    
demand = pd.DataFrame({
    "name": ["dem_S"],
    "node": ["S"],
    "max_power": [3000.0],
}).set_index("name")

# add this as a fixture and use in test_grid_utils as well
@pytest.fixture
def grid_data_dict():
    return {
        "buses": nodes,
        "lines": lines,
        "generators": generators,
        "loads": demand,
}

next_opening = simple_redispatch_market_config.opening_hours.after(datetime(2005, 6, 1))
products = get_available_products(simple_redispatch_market_config.market_products, next_opening)
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

@pytest.mark.parametrize("dummy_generation, dummy_demand, expected_redispatch",
                         [
                             (pd.Series([0, 3000, 0], index=generators.index), -3000, pd.Series([0, 0, 0], index=generators.index)), # all demand covered by coal_S
                             (pd.Series([3000, 0, 0], index=generators.index), -3000, pd.Series([-1000, 1000, 0], index=generators.index)), # downwards redispatch of wind_N, upward redispatch of coal_S
                            (pd.Series([0, 0, 0], index=generators.index), -3000, pd.Series([2000, 1000, 0], index=generators.index)), # zonal market does not cover demand -> find cheapest solution
                         ])

@pytest.mark.require_network
def test_two_nodes_redispatch(dummy_generation, dummy_demand, expected_redispatch):
    market_config = simple_redispatch_market_config
    market_config.param_dict["grid_data"] = grid_data_dict()
    market_config.param_dict["log_flows"] = True
    assert len(products) == 1
    orderbook = []
    for _ in generators.index:
        supply_bid = order.copy()
        supply_bid["unit_id"] = _
        supply_bid["volume"] = dummy_generation[_]
        supply_bid["min_power"] = 0
        supply_bid["max_power"] = generators.max_power[_]
        supply_bid["price"] = generators.marginal_cost[_]
        supply_bid["node"] = generators.node[_]
        supply_bid["bid_id"] = _ + "_bid"
        orderbook.append(supply_bid)
    
    demand_bid = order.copy()
    unit_id = demand.index[0]
    demand_bid["unit_id"] = unit_id
    demand_bid["volume"] = dummy_demand
    demand_bid["min_power"] = 0
    demand_bid["max_power"] = demand.max_power[unit_id]
    demand_bid["price"] = 1000.0
    demand_bid["node"] = demand.node[unit_id]
    demand_bid["bid_id"] = unit_id + "_bid"
    orderbook.append(demand_bid)
    for o in orderbook:
        print(o)

    rmr = RedispatchMarketRole(market_config)
    accepted_orders, rejected_orders, meta, flows = rmr.clear(orderbook, products)
    # meta
    print("meta:")
    print(meta)
    # assert (meta[0]["supply_volume"] == expected_redispatch.sum()).all()

    # test orders
    print("accepted orders:")
    for o in accepted_orders:
        print(o)
    print("rejected orders:")
    for o in rejected_orders:
        print(o)

    # test flows
    print("flows:")
    for f in flows: 
        print(f)

    # test cost

    # test up == -1 * down

@pytest.mark.require_network
def test_redispatch_with_availability():

    pass

@pytest.mark.require_network
def test_line_loading_after_redispatch():

    pass

@pytest.mark.require_network
def test_redispatch_with_storage():

    pass

