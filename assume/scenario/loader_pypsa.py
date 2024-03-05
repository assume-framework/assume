# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pypsa
from dateutil import rrule as rr

from assume import World
from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct


async def load_pypsa_async(
    world: World,
    scenario: str,
    study_case: str,
    network: pypsa.Network,
    marketdesign: list[MarketConfig],
):
    """
    This initializes a scenario from the given pypsa grid.

    Args:
        world (World): the world to add this scenario to
        scenario (str): scenario name
        study_case (str): study case name
        network (str): database uri to connect to the OEDS
        marketdesign (list[MarketConfig]): description of the market design which will be used with the scenario
    """
    index = network.snapshots
    index.freq = "h"
    start = index[0]
    end = index[-1]
    sim_id = f"{scenario}_{study_case}"
    print(f"loading scenario {sim_id}")

    await world.setup(
        start=start,
        end=end,
        save_frequency_hours=4,
        simulation_id=sim_id,
        index=index,
    )
    # setup eom market

    mo_id = "market_operator"
    world.add_market_operator(id=mo_id)
    grid_data = {
        "buses": network.buses,
        "lines": network.lines,
        "generators": network.generators,
        "loads": network.loads,
    }
    for market_config in marketdesign:
        market_config.param_dict["grid_data"] = grid_data
        world.add_market(mo_id, market_config)

    # naive_eom
    default_strategy = {mc.market_id: "naive_redispatch" for mc in marketdesign}

    bidding_strategies = {
        "hard coal": default_strategy,
        "lignite": default_strategy,
        "oil": default_strategy,
        "gas": default_strategy,
        "biomass": default_strategy,
        "nuclear": default_strategy,
        "wind": default_strategy,
        "solar": default_strategy,
        "demand": default_strategy,
    }

    world.add_unit_operator("powerplant_operator")
    for _, generator in network.generators.iterrows():
        if generator.name in network.generators_t["p_max_pu"].columns:
            av = network.generators_t["p_max_pu"][generator.name]
        else:
            av = 1

        unit_type = "power_plant"

        max_power = generator.p_nom or 1000
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
                "bidding_strategies": bidding_strategies["gas"],
                "technology": "demand",
                "node": generator.bus,
                "efficiency": generator.efficiency,
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

        world.add_unit(
            f"demand_{load.name}",
            unit_type,
            "demand_operator",
            {
                "min_power": 0,
                "max_power": load_t.max(),
                "bidding_strategies": bidding_strategies["demand"],
                "technology": "demand",
                "node": load.bus,
                "price": 1e3,
            },
            NaiveForecast(index, demand=load_t),
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
            f"StorageTrader_{agent['Id']}",
            unit_type,
            "storage_operator",
            {
                "max_power_charge": max_power_charge,
                "max_power_discharge": max_power_discharge,
                "efficiency_charge": storage.efficiency_store,
                "efficiency_discharge": storage.efficiency_dispatch,
                "initial_soc": storage.state_of_charge_initial,
                "max_volume": storage.p_nom,
                "bidding_strategies": bidding_strategies["storage"],
                "technology": "hydro",
                "emission_factor": 0,
                "node": storage.bus,
            },
            NaiveForecast(index, fuel_price=storage.marginal_cost),
        )


def load_pypsa(
    world: World,
    scenario: str,
    study_case: str,
    network: pypsa.Network,
    marketdesign: list[MarketConfig],
):
    """
    Load a scenario from a given path.

    Args:
        world (World): the world to add the oeds scenario to
        scenario (str): the scenario name of the simulation
        study_case (str): the study case name of the simulation
        network (str): the pypsa network to create the config from
        marketdesign (list[MarketConfig]): the list of marketconfigs, which form the market design
    """
    world.loop.run_until_complete(
        load_pypsa_async(
            world=world,
            scenario=scenario,
            study_case=study_case,
            network=network,
            marketdesign=marketdesign,
        )
    )


if __name__ == "__main__":
    db_uri = "postgresql://assume:assume@localhost:5432/assume"
    world = World(database_uri=db_uri)
    scenario = "world_pypsa"
    study_case = "scigrid_de"
    market_mechanism = "redispatch"  # "pay_as_clear"

    match study_case:
        case "ac_dc_meshed":
            network = pypsa.examples.ac_dc_meshed(from_master=True)
        case "scigrid_de":
            network = pypsa.examples.scigrid_de(True, from_master=True)
        case "storage_hvdc":
            network = pypsa.examples.storage_hvdc(True)
        case _:
            print("invalid studycase")
            network = pd.DataFrame()

    start = network.snapshots[0]
    end = network.snapshots[-1]
    marketdesign = [
        MarketConfig(
            "EOM",
            rr.rrule(rr.HOURLY, interval=1, dtstart=start, until=end),
            timedelta(hours=1),
            market_mechanism,
            [MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
            additional_fields=["node"],
            maximum_bid_volume=1e9,
            maximum_bid_price=1e9,
        )
    ]
    load_pypsa(world, scenario, study_case, network, marketdesign)
    world.run()
