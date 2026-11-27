# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

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


def add_redispatch_storages(
    network: pypsa.Network,
    storages: pd.DataFrame,
) -> None:
    """
    Adds the given storage units for redispatch.

    Storage is a special case: unlike a power plant it can be pushed both ways
    around its cleared dispatch (discharge more / charge less as well as charge
    more / discharge less), and every move changes its state of charge. It is
    therefore modelled as a PyPSA ``StorageUnit``: ``p_dispatch`` and ``p_store``
    are free around the cleared schedule and the state of charge follows them with
    the exact charge and discharge efficiencies.

    The static attributes come from the grid data: capacity (``max_hours``),
    ``max_soc`` and the efficiencies. What changes per clearing - the schedule,
    the power range and the state of charge at the start - is set by the
    redispatch clearing from the storages' bids.

    ``p_nom`` is the larger of the discharge and charge power, so that both
    ``p_max_pu`` (discharge) and ``p_min_pu`` (charge, negative) stay within
    ``[-1, 1]``.

    The per-bus backup generators added by :func:`add_redispatch_generators`
    already cover the storage buses, so no additional backups are added here.

    Args:
        network (pypsa.Network): the pypsa network to which the storages are added
        storages (pandas.DataFrame): the storage units dataframe
    """
    max_discharge = storages["max_power_discharge"].astype(float)
    max_charge = storages["max_power_charge"].astype(float)
    p_nom = np.maximum(max_discharge, -max_charge)
    p_nom = p_nom.where(p_nom > 0, 1.0)

    def column(name, default):
        if name in storages:
            return storages[name].fillna(default).astype(float)
        return pd.Series(default, index=storages.index, dtype=float)

    capacity = column("capacity", 0.0)
    max_soc = column("max_soc", 1.0)

    network.add(
        "StorageUnit",
        name=storages.index,
        bus=storages["node"],  # bus to which the storage is connected to
        # technology as carrier, e.g. to decide later which storages may take part
        carrier=storages.get("technology", "storage"),
        p_nom=p_nom,
        p_max_pu=max_discharge / p_nom,
        p_min_pu=max_charge / p_nom,
        max_hours=capacity * max_soc / p_nom,
        efficiency_store=column("efficiency_charge", 1.0),
        efficiency_dispatch=column("efficiency_discharge", 1.0),
        state_of_charge_initial=0.0,
        cyclic_state_of_charge=False,
        marginal_cost=0.0,
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

        # storage units: redispatch = dispatch after clearing minus the schedule
        # (p_set), signed like the generators (+ upward, - downward)
        storages_connected_to_bus = network.storage_units.index[
            network.storage_units.bus == bus
        ]
        if len(storages_connected_to_bus) > 0 and not network.storage_units_t.p.empty:
            storage_redispatch = (
                network.storage_units_t.p.reindex(
                    columns=storages_connected_to_bus, fill_value=0.0
                ).iloc[i]
                - network.storage_units_t.p_set.reindex(
                    columns=storages_connected_to_bus, fill_value=0.0
                ).iloc[i]
            )
            dispatch_for_bus = pd.concat(
                [dispatch_for_bus, storage_redispatch.fillna(0.0)]
            )

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
