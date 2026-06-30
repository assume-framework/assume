# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import pandas as pd
import pytest
from pytest import importorskip

pytest.importorskip("pypsa")
import pypsa

from assume.common.grid_utils import (
    add_nodal_loads,
    add_redispatch_generators,
    add_redispatch_loads,
    read_pypsa_grid,
)

get_supported_solver_linopy = importorskip(
    "assume.common.grid_utils",
    reason="linopy/pypsa dependencies not installed",
).get_supported_solver_linopy


@pytest.mark.require_network
def test_solver_available():
    assert get_supported_solver_linopy() == "highs"
    assert get_supported_solver_linopy("unknown_solver") == "highs"


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


def test_add_redispatch_generators(n_2bus_1line, generators_for_n_2_bus_1line):
    expected_backup_marginal_cost = 5000
    add_redispatch_generators(
        n_2bus_1line, generators_for_n_2_bus_1line, expected_backup_marginal_cost
    )

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
            "name": [
                f"{bus}_backup_{direction}"
                for direction in ["up", "down"]
                for bus in ["N", "S"]
            ],
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
    actual_up_generators = n_2bus_1line.generators.filter(like="_up", axis=0).filter(
        like="gen", axis=0
    )
    assert (actual_up_generators.index == expected_up_generators.index).all()
    assert (actual_up_generators.bus == expected_up_generators["bus"]).all()
    assert (actual_up_generators.p_nom == expected_up_generators["max_power"]).all()
    assert actual_up_generators["marginal_cost"].to_numpy() == pytest.approx(
        expected_up_generators["marginal_cost"].to_numpy(), abs=1e-6, rel=0
    )

    # check if backup generators are added with correct attributes
    actual_backup_generators = n_2bus_1line.generators.filter(like="backup", axis=0)
    assert (actual_backup_generators.index == expected_backup_generators.index).all()
    assert (actual_backup_generators.bus == expected_backup_generators["bus"]).all()
    assert (
        actual_backup_generators.p_nom == expected_backup_generators["max_power"]
    ).all()
    assert actual_backup_generators["marginal_cost"].to_numpy() == pytest.approx(
        expected_backup_generators["marginal_cost"].to_numpy(), abs=1e-6, rel=0
    )

    # check if _down generators are added with correct attributes
    actual_down_generators = n_2bus_1line.generators.filter(
        like="_down", axis=0
    ).filter(like="gen", axis=0)
    assert (actual_down_generators.index == expected_down_generators.index).all()
    assert (actual_down_generators.bus == expected_down_generators["bus"]).all()
    assert (actual_down_generators.p_nom == expected_down_generators["max_power"]).all()
    assert actual_down_generators["marginal_cost"].to_numpy() == pytest.approx(
        expected_down_generators["marginal_cost"].to_numpy(), abs=1e-6, rel=0
    )


def test_add_redispatch_loads(n_2bus_1line, loads_for_n_2_bus_1line):
    expected_loads = pd.DataFrame(
        {
            "name": ["loadN", "loadS"],
            "bus": ["N", "S"],
            "max_power": [0.0, 3000.0],
            "sign": [-1.0, -1.0],
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
    assert (actual_loads.index == expected_loads.index).all()
    assert (actual_loads.bus == expected_loads["bus"]).all()
    assert (actual_loads.max_power == expected_loads["max_power"]).all()
    assert (actual_loads.sign == expected_loads["sign"]).all()

    actual_loads_t = n_2bus_1line.loads_t.p_set
    assert actual_loads_t.index == "now"
    # p_set should be initialized as 0 for all loads and snapshots
    assert (actual_loads_t == expected_loads_t).all().all()


def test_add_nodal_loads(n_2bus_1line, loads_for_n_2_bus_1line):
    expected_nodal_loads = pd.DataFrame(
        {
            "name": ["loadN", "loadS"],
            "bus": ["N", "S"],
            "p_nom": [0.0, 3000.0],
            "p_min_pu": [0.0, 0.0],
            "p_max_pu": [1.0, 1.0],
            "marginal_cost": [0.0, 0.0],
            "sign": [-1.0, -1.0],
        }
    ).set_index("name")

    expected_nodal_loads_t = pd.DataFrame(
        {
            "name": ["now"],  # empty dataframe
        }
    ).set_index("name")

    add_nodal_loads(n_2bus_1line, loads_for_n_2_bus_1line)
    actual_nodal_loads = n_2bus_1line.generators.filter(like="load", axis=0)
    assert (actual_nodal_loads.index == expected_nodal_loads.index).all()
    assert (actual_nodal_loads.bus == expected_nodal_loads["bus"]).all()
    assert (actual_nodal_loads.p_nom == expected_nodal_loads["p_nom"]).all()
    assert (actual_nodal_loads.p_min_pu == expected_nodal_loads["p_min_pu"]).all()
    assert (actual_nodal_loads.p_max_pu == expected_nodal_loads["p_max_pu"]).all()
    assert (
        actual_nodal_loads.marginal_cost == expected_nodal_loads["marginal_cost"]
    ).all()
    assert (actual_nodal_loads.sign == expected_nodal_loads["sign"]).all()

    actual_nodal_loads_t = n_2bus_1line.generators_t.p_set
    assert actual_nodal_loads_t.index == "now"
    # p_set should be initialized as 0 for all loads and snapshots
    assert (actual_nodal_loads_t == expected_nodal_loads_t).all().all()


# use grid_dict fixture from test_redispatch.py - does not seem to work
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
        "s_nom": [2000.0],
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
        "max_power": [3000.0],
    }
).set_index("name")


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
    assert (n.buses.index == grid_data_dict["buses"].index).all()
    for _ in grid_data_dict["buses"].columns:
        assert (n.buses[_] == grid_data_dict["buses"][_]).all()
    assert (n.lines.index == grid_data_dict["lines"].index).all()
    for _ in grid_data_dict["lines"].columns:
        assert (n.lines[_] == grid_data_dict["lines"][_]).all()

    assert "AC" in n.carriers.index
    assert n.generators.empty
    assert n.loads.empty
