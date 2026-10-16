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

    storage_df = pd.DataFrame(
        {
            "entity": ["batt_1"],
            "p_nom": [50.0],
            "max_hours": [4.0],
            "carrier": ["battery"],
            "bus": ["bus_north"],
            "state_of_charge_initial": [0.5],
            "efficiency_store": [0.95],
            "efficiency_dispatch": [0.95],
        }
    )

    conn_df = pd.DataFrame(
        {
            "entity": ["gen_gas", "load_city", "batt_1"],
            "bus": ["bus_north", "bus_north", "bus_north"],
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
            "storage": nw.from_native(storage_df).lazy(),
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
        record=record,
        scenario="datarecord_test",
        study_case="unit_case",
    )

    assert "gen_gas" in world.units, "generator gen_gas must be registered in world"
    assert "load_city" in world.units, "load_city must be registered in world"
    assert "batt_1" in world.units, "storage batt_1 must be registered in world"
    assert "powerplant_operator" in world.unit_operators, (
        "powerplant_operator must be registered"
    )
    assert "demand_operator" in world.unit_operators, (
        "demand_operator must be registered"
    )
    assert "storage_operator" in world.unit_operators, (
        "storage_operator must be registered"
    )

    gen_unit = world.units["gen_gas"]
    assert gen_unit.max_power == 250.0, "generator max_power must match p_nom"
    assert gen_unit.fuel_type == "gas", "generator fuel_type must match carrier"

    load_unit = world.units["load_city"]
    assert load_unit.node == "bus_north", "load node must match bus"

    storage_unit = world.units["batt_1"]
    assert storage_unit.max_power_charge == -50.0, (
        "storage max_power_charge must be negative"
    )
    assert storage_unit.max_power_discharge == 50.0, (
        "storage max_power_discharge must match p_nom"
    )
    assert storage_unit.capacity == 200.0, (
        "storage capacity must match p_nom * max_hours"
    )


def test_loader_datarecord_simulation_run():
    """`load_datarecord` constructs a valid world capable of executing simulation steps."""
    time_index = pd.date_range("2026-01-01", periods=3, freq="h")
    time_df = pd.DataFrame({"snapshot": time_index})

    gen_df = pd.DataFrame(
        {
            "entity": ["gen_solar"],
            "p_nom": [100.0],
            "carrier": ["solar"],
            "marginal_cost": [10.0],
            "bus": ["node_0"],
        }
    )
    demand_df = pd.DataFrame(
        {
            "entity": ["load_main"],
            "price": [500.0],
            "bus": ["node_0"],
        }
    )
    storage_df = pd.DataFrame(
        {
            "entity": ["batt_1"],
            "p_nom": [50.0],
            "max_hours": [4.0],
            "bus": ["node_0"],
        }
    )
    conn_df = pd.DataFrame(
        {
            "entity": ["gen_solar", "load_main", "batt_1"],
            "bus": ["node_0", "node_0", "node_0"],
        }
    )
    p_set_df = pd.DataFrame(
        {
            "entity": ["load_main"] * 3,
            "snapshot": time_index,
            "value": [-40.0, -50.0, -45.0],
        }
    )

    record = DummyRecord(
        dims={"snapshot": nw.from_native(time_df).lazy()},
        entity_types={
            "power_plant": nw.from_native(gen_df).lazy(),
            "demand": nw.from_native(demand_df).lazy(),
            "storage": nw.from_native(storage_df).lazy(),
        },
        groups={"connection": nw.from_native(conn_df).lazy()},
        attributes={"p_set": nw.from_native(p_set_df).lazy()},
    )

    world = World(database_uri=None, export_csv_path=None)
    load_datarecord(world, record)
    world.run()

    gen = world.units["gen_solar"]
    assert "energy" in gen.outputs, "unit outputs must record dispatched energy"
    assert gen.outputs["energy"].at[pd.Timestamp("2026-01-01 01:00:00")] == 50.0, (
        "dispatched energy at step 1 must be 50.0 MW"
    )
    batt = world.units["batt_1"]
    assert "energy" in batt.outputs, "storage unit outputs must record energy"
    assert "soc" in batt.outputs, "storage unit outputs must record soc"
