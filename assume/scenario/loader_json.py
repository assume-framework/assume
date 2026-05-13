# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import pandas as pd
from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta

from assume import MarketConfig, MarketProduct, World
from assume.common.forecaster import (
    DemandForecaster,
    ExchangeForecaster,
    PowerplantForecaster,
    SteelplantForecaster,
    UnitForecaster,
)
from assume.common.market_objects import OnlyHours


def source_target(connection: str):
    return connection.split("#")[0], connection.split("#")[2]


def getType(id: str):
    return id.split("_")[0]


def load_world_from_gui_json(data: dict, world: World) -> World:
    nodes = {i["id"]: i for i in data["nodes"]}
    edges = {}
    for i in data["edges"]:
        source, target = source_target(i["id"])
        i["target"] = target
        edges.setdefault(source, {}).setdefault(getType(target), []).append(i)
    worldData = nodes["world"]["data"]
    start = datetime.fromisoformat(worldData["start"])
    end = datetime.fromisoformat(worldData["end"])
    index = pd.date_range(
        start=start,
        end=end + datetime.timedelta(hours=24),
        freq=worldData["frequency"],
    )
    world.setup(
        start=start,
        end=end,
        save_frequency_hours=int(worldData["save_frequency_hours"]),
        simulation_id=worldData["simulation_id"],
    )

    add_markets(world, edges, nodes)
    add_units(world, edges, nodes, index)
    return world


def add_markets(world: World, edges: dict, nodes: dict):
    # add markets
    for market_operator in edges["world"]["marketProvider"]:
        target_market_operator = market_operator["target"]
        world.add_market_operator(target_market_operator)
        for market in edges[target_market_operator]["market"]:
            target_market = market["target"]
            market_products = []
            for market_product in edges[target_market]["marketProduct"]:
                target_market_product = market_product["target"]
                productData = nodes[target_market_product]["data"]
                print(productData)
                market_products.append(
                    MarketProduct(
                        duration=relativedelta(minutes=int(productData["duration"])),
                        count=int(productData["count"]),
                        first_delivery=relativedelta(
                            minutes=int(productData["first_delivery"])
                        ),
                        only_hours=_only_hours(productData.get("only_hours", "")),
                        eligible_lambda_function=_optional_string(
                            productData.get("eligible_lambda_function")
                        ),
                    )
                )
            data = nodes[target_market]["data"]
            world.add_market(
                market_operator_id=target_market_operator,
                market_config=MarketConfig(
                    market_id=target_market,
                    market_mechanism=data["market_mechanism"],
                    opening_hours=rr.rrule(
                        rr.HOURLY, interval=24, dtstart=world.start, until=world.end
                    ),
                    opening_duration=timedelta(minutes=int(data["opening_duration"])),
                    market_products=market_products,
                ),
            )


def add_units(world: World, edges: dict, nodes: dict, index):
    for unit_operator in edges["world"]["unitOperator"]:
        target_unit_operator = unit_operator["target"]
        world.add_unit_operator(target_unit_operator)
        for unit in edges[target_unit_operator]["unit"]:
            target_unit = unit["target"]
            bidding_strategies = {}
            for connection in edges[target_unit]["market"]:
                bidding_strategies[connection["target"]] = connection["data"][
                    "strategy"
                ]
            unitData = nodes[target_unit]["data"]
            world.add_unit(
                id=target_unit,
                unit_operator_id=target_unit_operator,
                unit_type=unitData["unitType"],
                unit_params={
                    "bidding_strategies": bidding_strategies,
                    "technology": unitData["technology"],
                    "min_power": int(unitData.get("min_power", 0)),
                    "max_power": int(unitData.get("max_power", 0)),
                    "price": float(unitData.get("price", 0)),
                    "efficiency": float(unitData.get("efficiency", 1.0)),
                    "ramp_up": int(unitData.get("ramp_up", 0)),
                    "ramp_down": int(unitData.get("ramp_down", 0)),
                    "emission_factor": float(unitData.get("emission_factor", 0)),
                    "min_operating_time": int(unitData.get("min_operating_time", 0)),
                    "min_downtime": int(unitData.get("min_downtime", 0)),
                    "max_power_charge": int(unitData.get("max_power_charge", 0)),
                    "max_power_discharge": int(unitData.get("max_power_discharge", 0)),
                    "max_soc": int(unitData.get("max_soc", 0)),
                    "volume_import": int(unitData.get("volume_import", 0)),
                    "volume_export": int(unitData.get("volume_export", 0)),
                },
                forecaster=forecaster_for_type(unitData, index),
            )
    return world


def forecaster_for_type(data: dict, index: pd.DatetimeIndex) -> UnitForecaster:
    default_args = {
        "availability": data.get("forecast_availability", 1.0),
        "market_prices": data.get("forecast_price", 50.0),
    }
    fuel_prices = {
        "co2": data.get("forecast_co2_price", 10.0),
        data.get("fuel_type", "others"): data.get("forecast_fuel_price", 10.0),
    }
    match data["unitType"]:
        case "power_plant":
            return PowerplantForecaster(index, fuel_prices=fuel_prices, **default_args)
        case "storage":
            return UnitForecaster(index, **default_args)
        case "demand":
            return DemandForecaster(
                index, **default_args, demand=data["forecast_demand"]
            )
        case "exchange":
            return ExchangeForecaster(index, **default_args)
        case "steel_plant":
            return SteelplantForecaster(index, fuel_prices=fuel_prices, **default_args)
        case "building":
            return UnitForecaster(index, **default_args)  # TODO
        case "hydrogen_plant":
            return UnitForecaster(index, **default_args)  # TODO
        case "steam_generation":
            return UnitForecaster(index, **default_args)  # TODO
    raise ValueError(f"Unknown unit type {data['unitType']}")


def _only_hours(s: str) -> OnlyHours | None:
    if s is None or s == "" or len(s.split(",")) != 2:
        return None
    return OnlyHours(int(s.split(",")[0]), int(s.split(",")[1]))


def _optional_string(s: str) -> str | None:
    if s is None or s == "" or s.lower() == "none":
        return None
    return s
