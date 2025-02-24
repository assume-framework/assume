# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import timedelta

import pandas as pd
import pypsa
from dateutil import rrule as rr

from assume import World
from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct

logger = logging.getLogger(__name__)


def load_pypsa(
    world: World,
    scenario: str,
    study_case: str,
    network: pypsa.Network,
    marketdesign: list[MarketConfig],
    bidding_strategies: dict[str, dict[str, str]],
    save_frequency_hours: int = 4,
):
    """
    This initializes a scenario from the given pypsa grid.
    One can load a grid from pypsa `import_from_csv_folder`, adjust its properties and add it to this function to create an energy market scenario from it.
    This is also compatible with netCDF, PyPower, PandaPower and HDF5 pypsa-compatible datasets.

    Args:
        world (World): the world to add this scenario to
        scenario (str): scenario name
        study_case (str): study case name
        network (pypsa.Network): pypsa Network from which the simulation properties and timeseries data is extracted
        marketdesign (list[MarketConfig]): description of the market design which will be used with the scenario
    """
    index = network.snapshots
    index.freq = index.inferred_freq
    start = index[0]
    end = index[-1]
    simulation_id = f"{scenario}_{study_case}"
    logger.info(f"loading scenario {simulation_id}")

    world.setup(
        start=start,
        end=end,
        save_frequency_hours=save_frequency_hours,
        simulation_id=simulation_id,
    )
    # setup eom market

    mo_id = "market_operator"
    world.add_market_operator(id=mo_id)

    network.generators.rename(
        columns={"bus": "node", "p_nom": "max_power"}, inplace=True
    )
    network.loads.rename(columns={"bus": "node", "p_set": "min_power"}, inplace=True)
    if "max_power" not in network.loads.columns:
        network.loads["max_power"] = 0
    grid_data = {
        "buses": network.buses,
        "lines": network.lines,
        "generators": network.generators,
        "loads": network.loads,
    }

    for market_config in marketdesign:
        market_config.param_dict["grid_data"] = grid_data
        world.add_market(mo_id, market_config)

    world.add_unit_operator("powerplant_operator")
    for _, generator in network.generators.iterrows():
        if generator.name in network.generators_t["p_max_pu"].columns:
            av = network.generators_t["p_max_pu"][generator.name]
        else:
            av = 1

        unit_type = "power_plant"

        max_power = generator.max_power or 1000
        # if p_nom is not set, generator.p_nom_extendable must be
        ramp_up = generator.ramp_limit_start_up * max_power
        ramp_down = generator.ramp_limit_shut_down * max_power
        world.add_unit(
            generator.name,
            unit_type,
            "powerplant_operator",
            {
                "min_power": generator.p_nom_min,
                "max_power": max_power,
                "bidding_strategies": bidding_strategies[unit_type][generator.name],
                "technology": "conventional",
                "node": generator.node,
                "efficiency": 1,  # do not use generator.efficiency as it is respected in marginal_cost,
                "fuel_type": generator.carrier,
                "ramp_up": ramp_up,
                "ramp_down": ramp_down,
                "min_operating_time": generator.min_up_time,
                "min_down_time": generator.min_down_time,
            },
            NaiveForecast(
                index,
                fuel_price=generator.marginal_cost,
                availability=av,
            ),
        )

    world.add_unit_operator("demand_operator")
    for _, load in network.loads.iterrows():
        if load.name not in network.loads_t["p_set"].columns:
            # we have no load
            continue

        load_t = network.loads_t["p_set"][load.name]
        unit_type = "demand"

        kwargs = {load.name: load_t}

        world.add_unit(
            load.name,
            unit_type,
            "demand_operator",
            {
                "min_power": 0,
                "max_power": load_t.max(),
                "bidding_strategies": bidding_strategies[unit_type][load.name],
                "technology": "demand",
                "node": load.node,
                "price": 1e3,
            },
            NaiveForecast(index, demand=load_t, **kwargs),
        )

    world.add_unit_operator("storage_operator")
    for _, storage in network.storage_units.iterrows():
        if storage.name in network.storage_units_t["p_set"].columns:
            storage = network.storage_units_t["p_set"][storage.name]
        else:
            # we have no storage
            continue

        unit_type = "storage"
        max_power_charge = storage.p_nom * storage.p_min_pu
        max_power_discharge = storage.p_nom * storage.p_max_pu

        world.add_unit(
            f"StorageTrader_{storage.name}",
            unit_type,
            "storage_operator",
            {
                "max_power_charge": max_power_charge,
                "max_power_discharge": max_power_discharge,
                "efficiency_charge": storage.efficiency_store,
                "efficiency_discharge": storage.efficiency_dispatch,
                "initial_soc": storage.state_of_charge_initial,
                "max_soc": storage.p_nom,
                "bidding_strategies": bidding_strategies[unit_type][storage.name],
                "technology": "hydro",
                "emission_factor": 0,
                "node": storage.bus,
            },
            NaiveForecast(index, fuel_price=storage.marginal_cost),
        )


if __name__ == "__main__":
    db_uri = "postgresql://assume:assume@localhost:5432/assume"
    world = World(database_uri=db_uri)
    scenario = "world_pypsa"
    study_case = "scigrid_de"
    # "pay_as_clear", "redispatch" or "nodal"
    market_mechanism = "complex_clearing"

    match study_case:
        case "ac_dc_meshed":
            network = pypsa.examples.ac_dc_meshed(from_master=True)
        case "scigrid_de":
            network = pypsa.examples.scigrid_de(True, from_master=True)
        case "storage_hvdc":
            network = pypsa.examples.storage_hvdc(True)
        case _:
            logger.info(f"invalid studycase: {study_case}")
            network = pd.DataFrame()

    study_case = f"{study_case}_{market_mechanism}"

    start = network.snapshots[0]
    end = network.snapshots[-1]
    marketdesign = [
        MarketConfig(
            "EOM",
            rr.rrule(rr.HOURLY, interval=1, dtstart=start, until=end),
            timedelta(hours=1),
            market_mechanism,
            [MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
            additional_fields=["node", "max_power", "min_power", "bid_type"],
            maximum_bid_volume=1e9,
            maximum_bid_price=1e9,
            param_dict={"log_flows": True},
        )
    ]
    if market_mechanism == "redispatch":
        marketdesign.append(
            MarketConfig(
                "EOM",
                rr.rrule(
                    rr.HOURLY,
                    interval=1,
                    dtstart=start - timedelta(hours=0.5),
                    until=end,
                ),
                timedelta(hours=0.25),
                "pay_as_clear",
                [MarketProduct(timedelta(hours=1), 1, timedelta(hours=1.5))],
                additional_fields=["node", "max_power", "min_power"],
                maximum_bid_volume=1e9,
                maximum_bid_price=1e9,
            )
        )
    default_strategies = {
        mc.market_id: (
            "naive_redispatch" if mc.market_mechanism == "redispatch" else "naive_eom"
        )
        for mc in marketdesign
    }
    from collections import defaultdict

    bidding_strategies = {
        "power_plant": defaultdict(lambda: default_strategies),
        "demand": defaultdict(
            lambda: {mc.market_id: "naive_eom" for mc in marketdesign}
        ),
        "storage": defaultdict(lambda: default_strategies),
    }

    load_pypsa(world, scenario, study_case, network, marketdesign, bidding_strategies)
    world.run()
