# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field
from typing import Any

import narwhals as nw
import pandas as pd

from assume.scenario.loader_datarecord import load_datarecord
from assume.world import World


@dataclass
class DummyRecord:
    """Mock RecordLike object providing frames for load_datarecord."""

    dims: dict[str, Any] = field(default_factory=dict)
    entity_types: dict[str, Any] = field(default_factory=dict)
    groups: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)


def test_loader_datarecord_setup():
    time_index = pd.date_range("2026-01-01", periods=4, freq="h")
    time_df = pd.DataFrame({"snapshot": time_index})

    gen_df = pd.DataFrame(
        {
            "entity": ["gen_gas"],
            "p_nom": [250.0],
            "carrier": ["gas"],
            "marginal_cost": [55.0],
            "bus": ["bus_north"],
        }
    )

    demand_df = pd.DataFrame(
        {
            "entity": ["load_city"],
            "price": [1000.0],
            "bus": ["bus_north"],
        }
    )

    conn_df = pd.DataFrame(
        {
            "entity": ["gen_gas", "load_city"],
            "bus": ["bus_north", "bus_north"],
        }
    )

    p_max_df = pd.DataFrame(
        {
            "entity": ["gen_gas"] * 4,
            "snapshot": time_index,
            "value": [1.0, 0.9, 0.8, 1.0],
        }
    )

    p_set_df = pd.DataFrame(
        {
            "entity": ["load_city"] * 4,
            "snapshot": time_index,
            "value": [-100.0, -120.0, -110.0, -90.0],
        }
    )

    record = DummyRecord(
        dims={"snapshot": nw.from_native(time_df).lazy()},
        entity_types={
            "power_plant": nw.from_native(gen_df).lazy(),
            "demand": nw.from_native(demand_df).lazy(),
        },
        groups={"connection": nw.from_native(conn_df).lazy()},
        attributes={
            "p_max_pu": nw.from_native(p_max_df).lazy(),
            "p_set": nw.from_native(p_set_df).lazy(),
        },
    )

    world = World()
    load_datarecord(
        world=world,
        scenario="datarecord_test",
        study_case="unit_case",
        record=record,
    )

    assert "gen_gas" in world.units
    assert "load_city" in world.units
    assert "powerplant_operator" in world.unit_operators
    assert "demand_operator" in world.unit_operators

    gen_unit = world.units["gen_gas"]
    assert gen_unit.max_power == 250.0
    assert gen_unit.fuel_type == "gas"

    load_unit = world.units["load_city"]
    assert load_unit.node == "bus_north"
