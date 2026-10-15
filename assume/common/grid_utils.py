# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import pypsa
from linopy import available_solvers

from assume.common.market_objects import MarketProduct
from assume.common.utils import SUPPORTED_SOLVERS

logger = logging.getLogger(__name__)


def add_redispatch_generators(
    network: pypsa.Network,
    generators: pd.DataFrame,
    backup_marginal_cost: float = 1e5,
) -> None:
    """
    Adds the given generators for redispatch.
    This includes functions to optimize up as well as down and adds backup capacities of powerplants to be able to adjust accordingly when a congestion happens.

    Args:
        network (pypsa.Network): the pypsa network to which the generators are
        generators (pandas.DataFrame): the generators dataframe
        backup_marginal_cost (float, optional): The cost of dispatching the backup units in [€/MW]. Defaults to 1e5.
    """
    zeros = pd.DataFrame(
        np.zeros((len(network.snapshots), len(generators.index))),
        index=network.snapshots,
        columns=generators.index,
    )

    # add generators and their sold capacities as Generator to have fixed feed in
    # The actual fixed dispatch is set later in redispatch.py by
    # p_min_pu=p_max_pu=cleared_dispatch/p_nom
    network.add(
        "Generator",
        name=generators.index,
        bus=generators["node"],  # bus to which the generator is connected to
        p_nom=generators["max_power"],
        p_set=zeros,
        p_min_pu=zeros,
        p_max_pu=zeros,
        sign=1,
    )

    # add upward redispatch generators
    network.add(
        "Generator",
        name=generators.index,
        suffix="_up",
        bus=generators["node"],  # bus to which the generator is connected to
        p_nom=generators["max_power"],  # Nominal capacity of the powerplant/generator
        p_min_pu=zeros,
        p_max_pu=zeros + 1,
        marginal_cost=zeros,
    )

    # add downward redispatch generators
    network.add(
        "Generator",
        name=generators.index,
        suffix="_down",
        bus=generators["node"],  # bus to which the generator is connected to
        p_nom=generators["max_power"],  # Nominal capacity of the powerplant/generator
        p_min_pu=zeros,
        p_max_pu=zeros + 1,
        marginal_cost=zeros,
        sign=-1,
    )

    # add upward and downward backup generators at each node
    network.add(
        "Generator",
        name=network.buses.index,
        suffix="_backup_up",
        bus=network.buses.index,  # bus to which the generator is connected to
        p_nom=10e4,
        marginal_cost=backup_marginal_cost,
    )

    network.add(
        "Generator",
        name=network.buses.index,
        suffix="_backup_down",
        bus=network.buses.index,  # bus to which the generator is connected to
        p_nom=10e4,
        marginal_cost=backup_marginal_cost,
        sign=-1,
    )


def add_redispatch_loads(
    network: pypsa.Network,
    loads: pd.DataFrame,
) -> None:
    """
    This adds loads to the redispatch PyPSA network with respective bus data to which they are connected
    """
    loads_c = loads.copy()
    if "sign" in loads_c.columns:
        del loads_c["sign"]

    # add loads
    network.add(
        "Load",
        name=loads.index,
        bus=loads["node"],  # bus to which the generator is connected to
        **loads_c,
    )

    if "p_set" not in loads.columns:
        network.loads_t["p_set"] = pd.DataFrame(
            np.zeros((len(network.snapshots), len(loads.index))),
            index=network.snapshots,
            columns=loads.index,
        )


def add_redispatch_flexible_loads(
    network: pypsa.Network,
    flexible_loads: pd.DataFrame,
) -> None:
    """
    Add flexible electrical consumers to the redispatch network.
    Each flexible consumer is represented by:

    """
    if flexible_loads.empty:
        return

    required_columns = {"node", "p_nom"}
    missing_columns = (required_columns- set(flexible_loads.columns))

    if missing_columns:
        raise ValueError(
            "Flexible redispatch loads are missing required "
            f"columns: {sorted(missing_columns)}"
        )

    flexible_loads = flexible_loads.copy()

    flexible_loads["p_nom"] = (
        flexible_loads["p_nom"]
        .astype(float)
        .abs()
    )

    if (flexible_loads["p_nom"] <= 0).any():
        invalid = flexible_loads.index[
            flexible_loads["p_nom"] <= 0
        ].tolist()

        raise ValueError(
            "Flexible redispatch loads require p_nom > 0. "
            f"Invalid units: {invalid}"
        )
    zeros = pd.DataFrame(
        0.0,
        index=network.snapshots,
        columns=flexible_loads.index,
    )

    # 1. Physical electricity consumption
    missing_loads = flexible_loads.index.difference(network.loads.index)

    if len(missing_loads) > 0:
        network.add(
            "Load",
            name=missing_loads,
            bus=flexible_loads.loc[
                missing_loads, "node"
            ],
            p_set=zeros[missing_loads],
        )

    # 2. Upward redispatch
    # Reducing consumption is electrically equivalent to positive generation.
    up_names = pd.Index([f"{unit_id}_up" for unit_id in flexible_loads.index])
    existing_up = pd.Index(network.generators.index).intersection(up_names)

    if len(existing_up) > 0:
        raise ValueError(
            "Redispatch assets already exist: "
            f"{existing_up.tolist()}"
        )

    network.add(
        "Generator",
        name=flexible_loads.index,
        suffix="_up",
        bus=flexible_loads["node"],
        p_nom=flexible_loads["p_nom"],
        p_min_pu=zeros,
        p_max_pu=zeros,
        marginal_cost=zeros,
        sign=1,
    )

    # 3. Downward redispatch
    # Increasing consumption is electrically equivalent to negative generation.
    down_names = pd.Index([f"{unit_id}_down" for unit_id in flexible_loads.index])
    existing_down = pd.Index(network.generators.index).intersection(down_names)

    if len(existing_down) > 0:
        raise ValueError(
            "Redispatch assets already exist: "
            f"{existing_down.tolist()}"
        )

    network.add(
        "Generator",
        name=flexible_loads.index,
        suffix="_down",
        bus=flexible_loads["node"],
        p_nom=flexible_loads["p_nom"],
        p_min_pu=zeros,
        p_max_pu=zeros,
        marginal_cost=zeros,
        sign=-1,
    )


def read_pypsa_grid(
    network: pypsa.Network,
    grid_dict: dict[str, pd.DataFrame],
):
    """
    Generates the pypsa grid from a grid dictionary.
    Does not add the generators, as they are added in different ways, depending on whether redispatch is used.

    Args:
        network (pypsa.Network): the pypsa network to which the components will be added
        grid_dict (dict[str, pd.DataFrame]): the dictionary containing dataframes for generators, loads, buses and links
    """

    def add_buses(network: pypsa.Network, buses: pd.DataFrame) -> None:
        network.add("Bus", buses.index, **buses)

    def add_lines(network: pypsa.Network, lines: pd.DataFrame) -> None:
        network.add("Line", lines.index, **lines)

    # setup the network
    add_buses(network, grid_dict["buses"])
    add_lines(network, grid_dict["lines"])
    network.add("Carrier", "AC")
    return network


def calculate_network_meta(network, product: MarketProduct, i: int):
    """
    This function calculates the meta data such as supply and demand volumes, and nodal prices.

    Args:
        product (MarketProduct): The product for which clearing happens.
        i (int): The index of the product in the market products list.

    Returns:
        dict: The meta data.
    """

    meta = []
    duration_hours = (product[1] - product[0]) / timedelta(hours=1)
    # Only redispatch-related generator components should enter redispatch meta.
    redispatch_generator_names = network.generators.index[
        network.generators.index.str.endswith(("_up", "_down"))
    ]

    for bus in network.buses.index:
        generators_connected_to_bus = network.generators[
            (network.generators.bus == bus)
            & (network.generators.index.isin(redispatch_generator_names))
        ].index

        if len(generators_connected_to_bus) > 0:
            dispatch_for_bus = network.generators_t.p.reindex(
                columns=generators_connected_to_bus,
                fill_value=0.0,
            ).iloc[i]

            dispatch_for_bus = (
                dispatch_for_bus * network.generators.sign[generators_connected_to_bus]
            )
        else:
            dispatch_for_bus = pd.Series(dtype=float)

        supply_volume = dispatch_for_bus[dispatch_for_bus > 0].sum()
        demand_volume = -dispatch_for_bus[dispatch_for_bus < 0].sum()

        if not network.buses_t.marginal_price.empty:
            price = network.buses_t.marginal_price[str(bus)].iat[i]
        else:
            price = 0

        meta.append(
            {
                "supply_volume": supply_volume,
                "demand_volume": demand_volume,
                "demand_volume_energy": demand_volume * duration_hours,
                "supply_volume_energy": supply_volume * duration_hours,
                "price": price,
                "node": bus,
                "product_start": product[0],
                "product_end": product[1],
                "only_hours": product[2],
            }
        )

    return meta


def get_supported_solver_linopy(default_solver: str | None = None):
    """
    Get an available solver for linopy optimization.

    Filters the list of supported solvers to find which ones are installed,
    then returns the default solver if available, otherwise falls back to the first available solver.

    Args:
        default_solver (str | None, optional): Preferred solver name. If not available,
            falls back to the first available solver. Defaults to None.

    Returns:
        str: Name of the selected solver.

    Raises:
        RuntimeError: If none of the supported solvers (highs, gurobi, glpk, cbc, cplex) are available.

    Warning:
        Logs a warning if the default_solver is not available and a fallback is used.
    """
    solvers_priority = SUPPORTED_SOLVERS

    # Filter available solvers while preserving the shared fallback priority.
    solvers = [solver for solver in solvers_priority if solver in available_solvers]
    if not solvers:
        raise RuntimeError(f"None of {solvers_priority} are available")

    solver = default_solver or solvers[0]

    if solver not in solvers:
        logger.warning("Solver %s not available, using %s", solver, solvers[0])
        solver = solvers[0]

    return solver
