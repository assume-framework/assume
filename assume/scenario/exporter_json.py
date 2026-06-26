# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import base64
import json
from random import randbytes

from assume import World
from assume.common.base import BaseStrategy, BaseUnit
from assume.common.market_objects import lambda_functions
from assume.strategies import bidding_strategies
from assume.units import (
    Building,
    Demand,
    Exchange,
    HydrogenPlant,
    PowerPlant,
    SteelPlant,
)


def identify_strategy(strategy: BaseStrategy) -> str:
    for name, s in bidding_strategies.items():
        if isinstance(strategy, s):
            return name
    return ""


def unit_type(u: BaseUnit) -> str:
    if isinstance(u, Demand):
        return "demand"
    if isinstance(u, PowerPlant):
        return "powerplant"
    if isinstance(u, Building):
        return "building"
    if isinstance(u, HydrogenPlant):
        return "hydrogen_plant"
    if isinstance(u, SteelPlant):
        return "steel_plant"
    if isinstance(u, Exchange):
        return "exchange"
    return "storage"


def lambda_fn(fn) -> str:
    for k, v in lambda_functions.items():
        if fn == v:
            return k
    return ""


def position_left(depth=0) -> dict:
    return {"x": 0, "y": depth * 100}


def position_right(depth=0) -> dict:
    return {"x": 500, "y": depth * 100}


def get_id(node_type: str) -> str:
    bytes = randbytes(3)
    return f"{node_type}_{base64.b64encode(bytes).decode('ascii')}"


def _val(field):
    if field is None:
        return None
    if isinstance(field, str) or isinstance(field, int) or isinstance(field, float):
        return field
    return "TODO"


def to_gui_json(w: World) -> str:
    nodes, edges = [], []
    tmp_edges = {}
    nodes.append(
        {
            "id": "world",
            "type": "world",
            "deletable": False,
            "data": {
                "name": "world",
                "save_frequency_hours": w.output_role.save_frequency_hours,
                "start": str(w.start),
                "end": str(w.end),
                "simulation_id": w.simulation_id,
            },
            "position": {"x": 250, "y": 0},
        }
    )
    for operator_name, operator in w.unit_operators.items():
        operator_id = get_id("unitOperator")
        nodes.append(
            {
                "id": operator_id,
                "type": "unitOperator",
                "data": {
                    "name": operator_name,
                },
                "position": position_right(1),
            }
        )
        edges.append(
            {
                "id": f"world#unitOperator_handle#{operator_id}#world_handle",
                "source": "world",
                "sourceHandle": "unitOperator_handle",
                "target": operator_id,
                "targetHandle": "world_handle",
                "type": "default",
                "data": {"name": f"world-{operator_id}"},
            }
        )
        for unit in operator.units.values():
            unit_id = get_id("unit")
            nodes.append(
                {
                    "id": unit_id,
                    "type": "unit",
                    "data": {
                        "name": unit.id,
                        "unitType": unit_type(unit),
                        "technology": unit.technology,
                        "min_power": _val(getattr(unit, "min_power", 0)),
                        "max_power": _val(getattr(unit, "max_power", 0)),
                        "price": _val(getattr(unit, "price", 0)),
                        "efficiency": _val(getattr(unit, "efficiency", 1.0)),
                        "ramp_up": _val(getattr(unit, "ramp_up", 0)),
                        "ramp_down": _val(getattr(unit, "ramp_down", 0)),
                        "emission_factor": _val(getattr(unit, "emission_factor", 0)),
                        "min_operating_time": _val(
                            getattr(unit, "min_operating_time", 0)
                        ),
                        "min_downtime": _val(getattr(unit, "min_downtime", 0)),
                        "max_power_charge": _val(getattr(unit, "max_power_charge", 0)),
                        "max_power_discharge": _val(
                            getattr(unit, "max_power_discharge", 0)
                        ),
                        "max_soc": _val(getattr(unit, "max_soc", 0)),
                        "volume_import": _val(getattr(unit, "volume_import", 0)),
                        "volume_export": _val(getattr(unit, "volume_export", 0)),
                    },
                    "position": position_right(2),
                }
            )
            edges.append(
                {
                    "id": f"{operator_id}#unit_handle#{unit_id}#unitOperator_handle",
                    "source": operator_id,
                    "sourceHandle": "unit_handle",
                    "target": unit_id,
                    "targetHandle": "unitOperator_handle",
                    "type": "default",
                    "data": {"name": f"{operator_id}-{unit_id}"},
                }
            )
            for market, strategy in unit.bidding_strategies.items():
                e = tmp_edges.get(market, [])
                e.append(
                    {
                        "unit_id": unit_id,
                        "strategy": identify_strategy(strategy),
                    }
                )
                tmp_edges[market] = e
    for provider_name, provider in w.market_operators.items():
        provider_id = get_id("marketProvider")
        nodes.append(
            {
                "id": provider_id,
                "type": "marketProvider",
                "data": {
                    "name": provider_name,
                },
                "position": position_left(1),
            }
        )
        edges.append(
            {
                "id": f"world#marketProvider_handle#{provider_id}#world_handle",
                "source": "world",
                "sourceHandle": "marketProvider_handle",
                "target": provider_id,
                "targetHandle": "world_handle",
                "type": "default",
                "data": {"name": f"world-{provider_id}"},
            }
        )
        for market in provider.markets:
            market_id = get_id(market.market_id)
            nodes.append(
                {
                    "id": market_id,
                    "type": "market",
                    "data": {
                        "name": market.market_id,
                        "opening_duration": str(market.opening_duration.seconds // 60),
                        "market_mechanism": market.market_mechanism,
                    },
                    "position": position_left(2),
                }
            )
            edges.append(
                {
                    "id": f"{provider_id}#market_handle#{market_id}#marketProvider_handle",
                    "source": provider_id,
                    "sourceHandle": "market_handle",
                    "target": market_id,
                    "targetHandle": "marketProvider_handle",
                    "type": "default",
                    "data": {"name": f"{provider_id}-{market_id}"},
                }
            )
            for t in tmp_edges.get(market.market_id, []):
                edges.append(
                    {
                        "id": f"{t['unit_id']}#market_handle#{market_id}#unit_handle",
                        "source": t["unit_id"],
                        "sourceHandle": "market_handle",
                        "target": market_id,
                        "targetHandle": "unit_handle",
                        "type": "unit-market",
                        "data": {
                            "name": f"{t['unit_id']}-{market_id}",
                            "strategy": t["strategy"],
                        },
                    }
                )
            for product in market.market_products:
                id = get_id("marketProduct")
                nodes.append(
                    {
                        "id": id,
                        "type": "marketProduct",
                        "data": {
                            "name": id,
                            "duration": str(product.duration.seconds // 60),
                            "count": product.count,
                            "first_delivery": str(product.first_delivery.seconds // 60),
                            "eligible_lambda_function": lambda_fn(
                                product.eligible_lambda_function
                            ),
                        },
                        "position": position_left(3),
                    }
                )
                edges.append(
                    {
                        "id": f"{market_id}#marketProduct_handle#{id}#market_handle",
                        "source": market_id,
                        "sourceHandle": "marketProduct_handle",
                        "target": id,
                        "targetHandle": "market_handle",
                        "type": "default",
                        "data": {"name": f"{market_id}-{id}"},
                    }
                )

    return json.dumps({"nodes": nodes, "edges": edges})
