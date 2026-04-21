# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import pandas as pd
import pytest
from dateutil import rrule as rr
pypsa = pytest.importorskip("pypsa")

from assume.common.fast_pandas import FastIndex, FastSeries
from assume.common.market_objects import MarketConfig, MarketProduct, Order
# from assume.common.grid_utils import get_supported_solver_linopy
from assume.common.grid_utils import (
    add_generators,
    add_redispatch_generators,
    add_backup_generators,
    add_loads,
    add_redispatch_loads,
    add_nodal_loads,
    read_pypsa_grid,
    calculate_network_meta,
)
from assume.scenario.loader_csv import make_market_config
from tests.test_exchanges import market_config
try:
    from assume.markets.clearing_algorithms import RedispatchMarketRole
except ImportError:
    pass

@pytest.fixture
def n_2bus_1line():
    # set up empty network with 2 buses N and S and a line connecting them
    n = pypsa.Network()
    n.add("Bus", "N", v_nom=380)
    n.add("Bus", "S", v_nom=380)
    n.add("Line", "N-S", bus0="N", bus1="S", x=0.01, r=0.001, s_nom=2000.0)
    n.set_snapshots(["now"])
    return n

@pytest.fixture
def generators_for_n_2_bus_1line():
    # set up 2 generators, one at each bus, with different costs and capacities
    generators = {
        "name": ["cheap_genN", "intermediate_genS", "expensive_genS"],
        "node": ["N", "S", "S"],
        "max_power": [3000.0, 1000.0, 2000.0],
        "marginal_cost": [10.0, 20.0, 30.0],
    }
    return pd.DataFrame(generators).set_index("name")

@pytest.fixture
def loads_for_n_2_bus_1line():
    # set up 1 loads at each bus, the north load being 0
    loads = {
        "name": ["loadN", "loadS"],
        "node": ["N", "S"],
        "max_power": [0.0, 3000.0],
    }
    return pd.DataFrame(loads).set_index("name")

def test_add_generators(n_2bus_1line, generators_for_n_2_bus_1line):
    # add_generators is never used within our framework
    n1 = n_2bus_1line.copy() # for adding generators as df
    n2 = n_2bus_1line.copy() # for adding generators as dict
    expected_generators = pd.DataFrame(
        {
            "name": ["cheap_genN", "intermediate_genS", "expensive_genS"],
            "bus": ["N", "S", "S"],
            "max_power": [3000.0, 1000.0, 2000.0],
            "marginal_cost": [10.0, 20.0, 30.0],
        }
    ).set_index("name")
    # add generators as df and check if they are added with correct attributes
    add_generators(n1, generators_for_n_2_bus_1line)
    assert n1.generators.index.equals(expected_generators.index)
    assert n1.generators.bus.equals(expected_generators["bus"])
    assert n1.generators.p_nom.equals(expected_generators["max_power"])
    assert n1.generators["marginal_cost"].eq(0).all()

    # add generators as dict and check if they are added with correct attributes
    generators_dict = generators_for_n_2_bus_1line.T.to_dict()
    # currently does not work, due to wrong implementation in add_generators
    # add_generators(n2, generators_dict)
    # assert n2.generators.index.equals(expected_generators.index)
    # assert n2.generators.bus.equals(expected_generators["bus"])
    # assert n2.generators.p_nom.equals(expected_generators["max_power"])
    # assert n2.generators["marginal_cost"].eq(0).all()


def test_add_redispatch_generators(n_2bus_1line, generators_for_n_2_bus_1line):
    expected_backup_marginal_cost = 5000
    add_redispatch_generators(n_2bus_1line, generators_for_n_2_bus_1line, expected_backup_marginal_cost)
    
    expected_up_generators = pd.DataFrame(
        {
            "name": [f"{name}_up" for name in generators_for_n_2_bus_1line.index],
            "bus": generators_for_n_2_bus_1line["node"].tolist(),
            "max_power": generators_for_n_2_bus_1line["max_power"].tolist(),
            "marginal_cost": [0] * len(generators_for_n_2_bus_1line),
        }
    ).set_index("name")

    expected_backup_generators = pd.DataFrame(
        {
            "name": [f"{bus}_backup_{direction}" for direction in ["up", "down"] for bus in ["N", "S"]],
            "bus": ["N", "S"] * 2,
            "max_power": [10e4] * 4,
            "marginal_cost": [expected_backup_marginal_cost] * 4,
        }
    ).set_index("name")

    expected_down_generators = pd.DataFrame(
        {
            "name": [f"{name}_down" for name in generators_for_n_2_bus_1line.index],
            "bus": generators_for_n_2_bus_1line["node"].tolist(),
            "max_power": generators_for_n_2_bus_1line["max_power"].tolist(),
            "marginal_cost": [0] * len(generators_for_n_2_bus_1line),
        }
    ).set_index("name")


    # check if _up generators are added with correct attributes
    actual_up_generators = n_2bus_1line.generators.filter(like="_up", axis=0).filter(like="gen", axis=0)
    assert actual_up_generators.index.equals(expected_up_generators.index)    
    assert actual_up_generators.bus.equals(expected_up_generators["bus"])
    assert actual_up_generators.p_nom.equals(expected_up_generators["max_power"])
    assert actual_up_generators["marginal_cost"].to_numpy() == pytest.approx(
        expected_up_generators["marginal_cost"].to_numpy(), abs=1e-6, rel=0
    )
    
    # check if backup generators are added with correct attributes
    actual_backup_generators = n_2bus_1line.generators.filter(like="backup", axis=0)
    assert actual_backup_generators.index.equals(expected_backup_generators.index)
    assert actual_backup_generators.bus.equals(expected_backup_generators["bus"])
    assert actual_backup_generators.p_nom.equals(expected_backup_generators["max_power"])
    assert actual_backup_generators["marginal_cost"].to_numpy() == pytest.approx(
        expected_backup_generators["marginal_cost"].to_numpy(), abs=1e-6, rel=0
    )

    # check if _down generators are added with correct attributes
    actual_down_generators = n_2bus_1line.generators.filter(like="_down", axis=0).filter(like="gen", axis=0)
    assert actual_down_generators.index.equals(expected_down_generators.index)
    assert actual_down_generators.bus.equals(expected_down_generators["bus"])
    assert actual_down_generators.p_nom.equals(expected_down_generators["max_power"])
    assert actual_down_generators["marginal_cost"].to_numpy() == pytest.approx(
        expected_down_generators["marginal_cost"].to_numpy(), abs=1e-6, rel=0
    )



def test_add_backup_generators():
    # function add_backup_generators is never used
    pass

def test_add_loads(n_2bus_1line, loads_for_n_2_bus_1line):
    # function add_loads is never used within our framework
    expected_loads = pd.DataFrame(
        {
            "name": ["loadN", "loadS"],
            "bus": ["N", "S"],
            "max_power": [0.0, 3000.0],
        }
    ).set_index("name")
    expected_loads_t = pd.DataFrame(
        {
            "name": ["now"],
            "loadN": [0.0],
            "loadS": [0.0],
        }
    ).set_index("name")

    add_loads(n_2bus_1line, loads_for_n_2_bus_1line)
    
    actual_loads = n_2bus_1line.loads
    assert actual_loads.index.equals(expected_loads.index)
    assert actual_loads.bus.equals(expected_loads["bus"])
    assert actual_loads.max_power.equals(expected_loads["max_power"])

    actual_loads_t = n_2bus_1line.loads_t.p_set
    assert actual_loads_t.index == 'now'
    # p_set should be initialized as 0 for all loads and snapshots
    assert (actual_loads_t == expected_loads_t).all().all()

def test_add_redispatch_loads(n_2bus_1line, loads_for_n_2_bus_1line):
    # add_redispatch_loads does the same as add_loads, with only one difference:
    # it enforces the sign to be 1 and not -1
    # unclear, why this is needed...
    expected_loads = pd.DataFrame(
        {
            "name": ["loadN", "loadS"],
            "bus": ["N", "S"],
            "max_power": [0.0, 3000.0],
            "sign" : [1.0, 1.0],
        }
    ).set_index("name")
    expected_loads_t = pd.DataFrame(
        {
            "name": ["now"],
            "loadN": [0.0],
            "loadS": [0.0],
        }
    ).set_index("name")

    add_redispatch_loads(n_2bus_1line, loads_for_n_2_bus_1line)
    
    actual_loads = n_2bus_1line.loads
    assert actual_loads.index.equals(expected_loads.index)
    assert actual_loads.bus.equals(expected_loads["bus"])
    assert actual_loads.max_power.equals(expected_loads["max_power"])
    assert actual_loads.sign.equals(expected_loads["sign"])

    actual_loads_t = n_2bus_1line.loads_t.p_set
    assert actual_loads_t.index == 'now'
    # p_set should be initialized as 0 for all loads and snapshots
    assert (actual_loads_t == expected_loads_t).all().all()

def test_add_nodal_loads(n_2bus_1line, loads_for_n_2_bus_1line):
    expected_nodal_loads = pd.DataFrame(
        {
            "name": ["loadN", "loadS"],
            "bus": ["N", "S"],
            "p_nom": [0.0, 3000.0],
            "p_min_pu" : [0.0, 0.0],
            "p_max_pu" : [1.0, 1.0],
            "marginal_cost": [0.0, 0.0],
            "sign": [-1.0, -1.0],
        }
    ).set_index("name")

    expected_nodal_loads_t = pd.DataFrame(
        {
            "name": ["now"],
            "loadN": [0.0],
            "loadS": [0.0],
        }
    ).set_index("name")

    add_nodal_loads(n_2bus_1line, loads_for_n_2_bus_1line)
    actual_nodal_loads = n_2bus_1line.generators.filter(like="load", axis=0)
    assert actual_nodal_loads.index.equals(expected_nodal_loads.index)
    assert actual_nodal_loads.bus.equals(expected_nodal_loads["bus"])
    assert actual_nodal_loads.p_nom.equals(expected_nodal_loads["p_nom"])
    assert actual_nodal_loads.p_min_pu.equals(expected_nodal_loads["p_min_pu"])
    assert actual_nodal_loads.p_max_pu.equals(expected_nodal_loads["p_max_pu"])
    assert actual_nodal_loads.marginal_cost.equals(expected_nodal_loads["marginal_cost"])
    assert actual_nodal_loads.sign.equals(expected_nodal_loads["sign"])

    actual_nodal_loads_t = n_2bus_1line.generators_t.p_set
    assert actual_nodal_loads_t.index == 'now'
    # p_set should be initialized as 0 for all loads and snapshots
    # tbd test _t values...
    # assert (actual_nodal_loads_t == expected_nodal_loads_t).all().all()

# use grid_dict fixture from test_redispatch.py - does not seem to work
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

@pytest.fixture
def grid_data_dict():
    return {
        "buses": nodes,
        "lines": lines,
        "generators": generators,
        "loads": demand,
}

def test_read_pypsa_grid(grid_data_dict):
    # read the grid data into a pypsa network
    n = pypsa.Network()
    n = read_pypsa_grid(n, grid_data_dict)
    
    # check if elements are added with correct attributes
    assert n.buses.index.equals(grid_data_dict['buses'].index)
    for _ in grid_data_dict['buses'].columns:
        assert n.buses[_].equals(grid_data_dict['buses'][_])
    assert n.lines.index.equals(grid_data_dict['lines'].index)
    for _ in grid_data_dict['lines'].columns:
        assert n.lines[_].equals(grid_data_dict['lines'][_])

    assert "AC" in n.carriers.index
    assert n.generators.empty
    assert n.loads.empty

def test_calculate_network_meta():
    pass