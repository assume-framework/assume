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


def add_redispatch_storage_units(
    network: pypsa.Network,
    storage_units: pd.DataFrame | None,
) -> None:
    """
    Adds storage units to the redispatch network so they can offer bidirectional
    redispatch around their market-committed schedule: upward (more discharge / less
    charge) and downward (less discharge / more charge).

    Each storage unit is modelled like a redispatchable generator: a fixed base
    generator that holds the committed net dispatch (which may be negative while
    charging), plus an ``_up`` and a ``_down`` generator carrying the redispatch
    flexibility. All three share a single nominal power ``p_nom = |max_power_discharge|
    + |max_power_charge|`` (the full charge+discharge span) so the per-unit dispatch
    stays within ``[-1, 1]`` even though the charging side is negative. The actual
    dispatch and per-snapshot head-room are filled later in ``RedispatchMarketRole.clear``.

    Args:
        network (pypsa.Network): the pypsa network to which the storage units are added
        storage_units (pandas.DataFrame | None): the storage units dataframe, or ``None``
            when the scenario has no storage units.
    """
    if storage_units is None or storage_units.empty:
        return

    zeros = pd.DataFrame(
        np.zeros((len(network.snapshots), len(storage_units.index))),
        index=network.snapshots,
        columns=storage_units.index,
    )

    # full charge+discharge power span; abs() guards against the sign convention used
    # for charge power in the input data
    span = (
        storage_units["max_power_discharge"].abs()
        + storage_units["max_power_charge"].abs()
    )

    # fixed base generator holding the committed net dispatch (may be negative while
    # charging). The actual dispatch is set later via p_min_pu=p_max_pu=base/p_nom.
    network.add(
        "Generator",
        name=storage_units.index,
        bus=storage_units["node"],
        p_nom=span,
        p_set=zeros,
        p_min_pu=zeros,
        p_max_pu=zeros,
        sign=1,
    )

    # upward redispatch generator (more discharge / less charge)
    network.add(
        "Generator",
        name=storage_units.index,
        suffix="_up",
        bus=storage_units["node"],
        p_nom=span,
        p_min_pu=zeros,
        p_max_pu=zeros + 1,
        marginal_cost=zeros,
    )

    # downward redispatch generator (less discharge / more charge)
    network.add(
        "Generator",
        name=storage_units.index,
        suffix="_down",
        bus=storage_units["node"],
        p_nom=span,
        p_min_pu=zeros,
        p_max_pu=zeros + 1,
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


def calculate_line_loading(network, market_products) -> list[dict]:
    """
    Calculates the loading of every line for every snapshot as a fraction of the line's
    thermal rating: ``|p0| / (s_nom * s_max_pu)``. A value greater than 1 means the line
    is overloaded.

    Args:
        network (pypsa.Network): the solved pypsa network (the flows are read from
            ``lines_t.p0``, so call this after ``lpf``/``optimize`` for the state of
            interest).
        market_products (list[MarketProduct]): the cleared products, used to map each
            integer snapshot to its real product start time.

    Returns:
        list[dict]: one record per line and snapshot with ``datetime``, ``line``,
        ``line_loading`` (per-unit), ``flow_mw`` (signed ``p0``) and ``s_nom``.
    """
    p0 = network.lines_t.p0
    rating = network.lines.s_nom * network.lines.s_max_pu
    loading = p0.abs() / rating

    records = []
    for i, product in enumerate(market_products):
        timestamp = product[0]
        p0_row = p0.iloc[i]
        loading_row = loading.iloc[i]
        for line in network.lines.index:
            records.append(
                {
                    "datetime": timestamp,
                    "line": line,
                    "line_loading": float(loading_row[line]),
                    "flow_mw": float(p0_row[line]),
                    "s_nom": float(network.lines.at[line, "s_nom"]),
                }
            )

    return records


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
