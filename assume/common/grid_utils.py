# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
import pypsa


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
    p_set = pd.DataFrame(
        np.zeros((len(network.snapshots), len(generators.index))),
        index=network.snapshots,
        columns=generators.index,
    )

    # add upward redispatch generators
    network.madd(
        "Generator",
        names=generators.index,
        suffix="_up",
        bus=generators["node"],  # bus to which the generator is connected to
        p_nom=generators["max_power"],  # Nominal capacity of the powerplant/generator
        p_min_pu=p_set,
        p_max_pu=p_set + 1,
        marginal_cost=p_set,
    )

    # add downward redispatch generators
    network.madd(
        "Generator",
        names=generators.index,
        suffix="_down",
        bus=generators["node"],  # bus to which the generator is connected to
        p_nom=generators["max_power"],  # Nominal capacity of the powerplant/generator
        p_min_pu=p_set,
        p_max_pu=p_set + 1,
        marginal_cost=p_set,
        sign=-1,
    )

    # add upward and downward backup generators at each node
    network.madd(
        "Generator",
        names=network.buses.index,
        suffix="_backup_up",
        bus=network.buses.index,  # bus to which the generator is connected to
        p_nom=10e4,
        marginal_cost=backup_marginal_cost,
    )

    network.madd(
        "Generator",
        names=network.buses.index,
        suffix="_backup_down",
        bus=network.buses.index,  # bus to which the generator is connected to
        p_nom=10e4,
        marginal_cost=backup_marginal_cost,
        sign=-1,
    )

    # add generators and their sold capacities as load with reversed sign to have fixed feed in
    network.madd(
        "Load",
        names=generators.index,
        bus=generators["node"],  # bus to which the generator is connected to
        p_set=p_set,
        sign=1,
    )


def add_generators(
    network: pypsa.Network,
    generators: pd.DataFrame,
) -> None:
    """
    Add generators normally to the grid

    Args:
        network (pypsa.Network): the pypsa network to which the generators are
        generators (pandas.DataFrame): the generators dataframe
    """
    p_set = pd.DataFrame(
        np.zeros((len(network.snapshots), len(generators.index))),
        index=network.snapshots,
        columns=generators.index,
    )

    # normal generator
    network.madd(
        "Generator",
        names=generators.index,
        suffix="_up",
        bus=generators["node"],  # bus to which the generator is connected to
        p_nom=generators["max_power"],  # Nominal capacity of the powerplant/generator
        p_min_pu=p_set,
        p_max_pu=p_set + 1,
        marginal_cost=p_set,
    )


def read_pypsa_grid(
    network: pypsa.Network,
    grid_dict: dict[str, pd.DataFrame],
):
    """
    Generates the pypsa grid from a grid dictionary.
    Does not add the generators, as they are added in different ways, depending on wether redispatch is used.

    Args:
        network (pypsa.Network): the pypsa network to which the components will be added
        grid_dict (dict[str, pd.DataFrame]): the dictionary containing dataframes for generators, loads, buses and links
    """
    def add_buses(network: pypsa.Network, buses: pd.DataFrame) -> None:
        network.madd(
            "Bus",
            names=buses.index,
            **buses,
        )

    def add_lines(network: pypsa.Network, lines: pd.DataFrame) -> None:
        network.madd(
            "Line",
            names=lines.index,
            **lines,
        )

    def add_loads(network: pypsa.Network, loads: pd.DataFrame) -> None:
        p_set = pd.DataFrame(
            np.zeros((len(network.snapshots), len(loads.index))),
            index=network.snapshots,
            columns=loads.index,
        )

        # Iterate through time steps and add generators
        network.madd(
            "Load",
            names=loads.index,
            bus=loads["node"],  # bus to which the generator is connected to
            p_set=p_set,
            sign=1,
        )

    # setup the network
    add_buses(network, grid_dict["buses"])
    add_lines(network, grid_dict["lines"])
    add_loads(network, grid_dict["loads"])

    return network
