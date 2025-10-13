# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import timedelta

import numpy as np
import pandas as pd
import pypsa

from assume.common.market_objects import MarketProduct


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

    if isinstance(generators, dict):
        gen_c = generators.copy()

        if "p_min_pu" not in gen_c.columns:
            gen_c["p_min_pu"] = p_set
        if "p_max_pu" not in gen_c.columns:
            gen_c["p_max_pu"] = p_set + 1
        if "marginal_cost" not in gen_c.columns:
            gen_c["marginal_cost"] = p_set

        network.add(
            "Generator",
            name=generators.index,
            bus=generators["node"],  # bus to which the generator is connected to
            p_nom=generators[
                "max_power"
            ],  # Nominal capacity of the powerplant/generator
            **gen_c,
        )
    else:
        # add generators
        generators.drop(
            ["p_min_pu", "p_max_pu", "marginal_cost"],
            axis=1,
            inplace=True,
            errors="ignore",
        )
        network.add(
            "Generator",
            name=generators.index,
            bus=generators["node"],  # bus to which the generator is connected to
            p_nom=generators[
                "max_power"
            ],  # Nominal capacity of the powerplant/generator
            p_min_pu=p_set,
            p_max_pu=p_set + 1,
            marginal_cost=p_set,
            **generators,
        )


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

    # add generators and their sold capacities as load with reversed sign to have fixed feed in
    network.add(
        "Load",
        name=generators.index,
        bus=generators["node"],  # bus to which the generator is connected to
        p_set=p_set,
        sign=1,
    )

    # add upward redispatch generators
    network.add(
        "Generator",
        name=generators.index,
        suffix="_up",
        bus=generators["node"],  # bus to which the generator is connected to
        p_nom=generators["max_power"],  # Nominal capacity of the powerplant/generator
        p_min_pu=p_set,
        p_max_pu=p_set + 1,
        marginal_cost=p_set,
    )

    # add downward redispatch generators
    network.add(
        "Generator",
        name=generators.index,
        suffix="_down",
        bus=generators["node"],  # bus to which the generator is connected to
        p_nom=generators["max_power"],  # Nominal capacity of the powerplant/generator
        p_min_pu=p_set,
        p_max_pu=p_set + 1,
        marginal_cost=p_set,
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


def add_backup_generators(
    network: pypsa.Network,
    backup_marginal_cost: float = 1e5,
) -> None:
    """
    Add generators normally to the grid

    Args:
        network (pypsa.Network): the pypsa network to which the generators are
        generators (pandas.DataFrame): the generators dataframe
    """

    # add backup generators at each node
    network.add(
        "Generator",
        name=network.buses.index,
        suffix="_backup",
        bus=network.buses.index,  # bus to which the generator is connected to
        p_nom=10e4,
        marginal_cost=backup_marginal_cost,
    )


def add_loads(
    network: pypsa.Network,
    loads: pd.DataFrame,
) -> None:
    """
    Add loads normally to the grid

    Args:
        network (pypsa.Network): the pypsa network to which the loads are
        loads (pandas.DataFrame): the loads dataframe
    """

    # add loads
    network.add(
        "Load",
        name=loads.index,
        bus=loads["node"],  # bus to which the generator is connected to
        **loads,
    )

    if "p_set" not in loads.columns:
        network.loads_t["p_set"] = pd.DataFrame(
            np.zeros((len(network.snapshots), len(loads.index))),
            index=network.snapshots,
            columns=loads.index,
        )


def add_fix_units(
    network: pypsa.Network,
    units: pd.DataFrame,
) -> None:
    """
    This adds the loads as fix units to the nodal PyPSA network. It also adds storage units & exchange units
    as fix_units because in the current redispatch market/mechanism these units are not allowed to do redispatch.

    Args:
        network (pypsa.Network): the pypsa network to which the units are added
        loads (pandas.DataFrame): the loads dataframe
    """
    if units is None or units.empty:
        return

    units_c = units.copy()
    if "sign" in units_c.columns:
        del units_c["sign"]

    # add loads with opposite sign (default for loads is -1). This is needed to properly model the redispatch
    network.add(
        "Load",
        name=units.index,
        bus=units["node"],  # bus to which the generator is connected to
        sign=1,
        **units_c,
    )

    if "p_set" not in units.columns:
        network.loads_t["p_set"] = pd.DataFrame(
            np.zeros((len(network.snapshots), len(units.index))),
            index=network.snapshots,
            columns=units.index,
        )


def add_nodal_loads(
    network: pypsa.Network,
    loads: pd.DataFrame,
) -> None:
    """
    This adds loads to the nodal PyPSA network with respective bus data to which they are connected.
    The loads are added as generators with negative sign so their dispatch can be also curtailed,
    since regular load in PyPSA represents only an inelastic demand.
    """
    p_set = pd.DataFrame(
        np.zeros((len(network.snapshots), len(loads.index))),
        index=network.snapshots,
        columns=loads.index,
    )
    loads_c = loads.copy()

    if "sign" in loads_c.columns:
        del loads_c["sign"]

    # add loads as negative generators
    network.add(
        "Generator",
        name=loads.index,
        bus=loads["node"],  # bus to which the generator is connected to
        p_nom=loads["max_power"],  # Nominal capacity of the powerplant/generator
        p_min_pu=p_set,
        p_max_pu=p_set + 1,
        marginal_cost=p_set,
        sign=-1,
        **loads_c,
    )

def add_redispatch_dsm(
    network: pypsa.Network, industrial_dsm_units: pd.DataFrame
) -> None:
    """
    This adds the industrial_dsm_units with the sold capacity in the DAM as loads
    and the flexible capacity to ramp up and ramp down as upward and downward generators

    Args:
        network (pypsa.Network): the pypsa network to which the units are added
        industrial_dsm_units (pandas.DataFrame): the industrial_dsm_units dataframe
    """

    # simply copy the DataFrame straight over
    dsm_units = industrial_dsm_units.copy()

    # now build p_set exactly as before...
    p_set = pd.DataFrame(
        0,
        index=network.snapshots,
        columns=dsm_units.index,
    )

    network.madd(
        "Load",
        names=dsm_units.index,
        bus=dsm_units["node"],
        p_set=p_set,
        sign=1,
    )

    # upward redispatch
    network.madd(
        "Generator",
        names=dsm_units.index,
        suffix="_up",
        bus=dsm_units["node"],
        p_nom=1,
        p_min_pu=p_set,
        p_max_pu=p_set,
        marginal_cost=p_set,
        sign=-1,
    )

    # downward redispatch
    network.madd(
        "Generator",
        names=dsm_units.index,
        suffix="_down",
        bus=dsm_units["node"],
        p_nom=1,
        p_min_pu=p_set,
        p_max_pu=p_set,
        marginal_cost=p_set,
        sign=1,
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

    def add_carriers(network: pypsa.Network, carriers: pd.DataFrame) -> None:
        if carriers is not None and not carriers.empty:
            if "name" in carriers.columns:
                carriers = carriers.set_index("name", drop=False)
            network.add("Carrier", carriers.index, **carriers)

    def add_buses(network: pypsa.Network, buses: pd.DataFrame) -> None:
        network.add("Bus", buses.index, **buses)

    def add_lines(network: pypsa.Network, lines: pd.DataFrame) -> None:
        if lines is not None and not lines.empty:
            network.add("Line", lines.index, **lines)

    def add_links(network: pypsa.Network, links: pd.DataFrame) -> None:
        if links is None or links.empty:
            return

        # Ensure 'name' is index so those become component names
        if "name" in links.columns:
            links = links.set_index("name", drop=False)

        # Required columns
        need = ["bus0", "bus1", "p_nom"]
        miss = [c for c in need if c not in links.columns]
        if miss:
            raise ValueError(f"links.csv missing columns: {miss}")

        # Strong typing
        links = links.copy()
        links["bus0"] = links["bus0"].astype(str)
        links["bus1"] = links["bus1"].astype(str)
        links["p_nom"] = pd.to_numeric(links["p_nom"], errors="coerce").astype(float)
        if "efficiency" in links.columns:
            links["efficiency"] = pd.to_numeric(links["efficiency"], errors="coerce").astype(float)
        else:
            links["efficiency"] = 1.0
        if "carrier" not in links.columns:
            links["carrier"] = "DC"
        links["carrier"] = links["carrier"].astype(str)

        # Add all links at once (names + arrays). This is a safe drop-in replacing madd.
        network.add(
            "Link",
            name=links.index.astype(str).to_list(),
            bus0=links["bus0"].to_list(),
            bus1=links["bus1"].to_list(),
            p_nom=links["p_nom"].to_list(),
            efficiency=links["efficiency"].to_list(),
            carrier=links["carrier"].to_list(),
            p_max_pu=1.0,
            p_min_pu=-1.0,
        )

        # Optional: assert sanity right here
        bad = links.index[network.links.loc[links.index, "p_nom"].isna() | (network.links.loc[links.index, "p_nom"] == 0)]
        if len(bad):
            raise ValueError(f"Some links have zero/NaN p_nom after add: {list(bad)}")


    # Add carriers first (so buses/lines reference existing carriers)
    if "carriers" in grid_dict:
        add_carriers(network, grid_dict["carriers"])

    # setup the network
    add_buses(network, grid_dict["buses"])

    if "lines" in grid_dict:
        add_lines(network, grid_dict["lines"])

    if "links" in grid_dict:
        add_links(network, grid_dict["links"])

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
    # iterate over buses
    for bus in network.buses.index:
        # add backup dispatch to dispatch
        # Step 1: Identify generators connected to the specified bus
        generators_connected_to_bus = network.generators[
            network.generators.bus == bus
        ].index

        # Step 2: Select dispatch levels for these generators from network.generators_t.p
        dispatch_for_bus = network.generators_t.p[generators_connected_to_bus].iloc[i]
        # multiple by network.generators.sign to get the correct sign for dispatch
        dispatch_for_bus = (
            dispatch_for_bus * network.generators.sign[generators_connected_to_bus]
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
