# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import json
import uuid

from assume import MarketConfig, World
from assume.common.base import BaseUnit


def unit_type(u: BaseUnit) -> str:
    return ""  # TODO


def lambda_fn(fn) -> str:
    return ""  # TODO


def position_left(depth=0) -> dict:
    return {"x": 0, "y": depth * 100}


def position_right(depth=0) -> dict:
    return {"x": 500, "y": depth * 100}


def to_json(w: World) -> str:
    nodes, edges = [], []
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
    for operator_id, operator in w.unit_operators.items():
        nodes.append(
            {
                "id": operator_id,
                "type": "unitOperator",
                "data": {
                    "name": operator_id,
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
            nodes.append(
                {
                    "id": unit.id,
                    "type": "unit",
                    "data": {
                        "name": unit.id,
                        "unit_type": unit_type(unit),
                        "unit_operator": unit.unit_operator,
                        "technology": unit.unit_operator,
                    },
                    "position": position_right(2),
                }
            )
            edges.append(
                {
                    "id": f"{unit.unit_operator}#unit_handle#{unit.id}#unitOperator_handle",
                    "source": unit.unit_operator,
                    "sourceHandle": "unit_handle",
                    "target": unit.id,
                    "targetHandle": "unitOperator_handle",
                    "type": "default",
                    "data": {"name": f"{unit.unit_operator}-{unit.id}"},
                }
            )
            for market, strategy in unit.bidding_strategies.items():
                edges.append(
                    {
                        "id": f"{unit.id}#market_handle#{market}#unit_handle",
                        "source": unit.id,
                        "sourceHandle": "market_handle",
                        "target": market,
                        "targetHandle": "unit_handle",
                        "type": "unit-market",
                        "data": {
                            "name": f"{unit.id}-{market}",
                            "strategy": str(strategy),
                        },
                    }
                )
    for provider_name, provider in w.market_operators.items():
        nodes.append(
            {
                "id": provider_name,
                "type": "marketProvider",
                "data": {
                    "name": provider_name,
                },
                "position": position_left(1),
            }
        )
        edges.append(
            {
                "id": f"world#marketProvider_handle#{provider_name}#world_handle",
                "source": "world",
                "sourceHandle": "marketProvider_handle",
                "target": provider_name,
                "targetHandle": "world_handle",
                "type": "default",
                "data": {"name": f"world-{provider_name}"},
            }
        )
        for market in provider.markets:
            market: MarketConfig
            nodes.append(
                {
                    "id": market.market_id,
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
                    "id": f"{provider_name}#market_handle#{market.market_id}#marketProvider_handle",
                    "source": provider_name,
                    "sourceHandle": "market_handle",
                    "target": market.market_id,
                    "targetHandle": "marketProvider_handle",
                    "type": "default",
                    "data": {"name": f"{provider_name}-{market.market_id}"},
                }
            )
            for product in market.market_products:
                id = str(uuid.uuid4())
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
                        "id": f"{market.market_id}#marketProduct_handle#{id}#market_handle",
                        "source": market.market_id,
                        "sourceHandle": "marketProduct_handle",
                        "target": id,
                        "targetHandle": "market_handle",
                        "type": "default",
                        "data": {"name": f"{market.market_id}-{id}"},
                    }
                )

    return json.dumps({"nodes": nodes, "edges": edges})


def __main():
    w = World()
    to_json(w)
