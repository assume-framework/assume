# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

import asyncio
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from assume.common.outputs import WriteOutput
from assume.common.units_operator import UnitsOperator
from assume.common.utils import datetime2timestamp

from .test_steel_plant import create_steel_plant, dsm_components  # noqa: F401


@pytest.fixture
def plant(dsm_components):  # noqa: F811
    return create_steel_plant(dsm_components, "cost_based_load_shift")


def _by_key(dispatch):
    return {(d["technology"], d["variable"]): d for d in dispatch}


def test_baseline_dispatch_is_captured_per_technology(plant):
    dispatch = _by_key(plant.get_component_schedule(plant.index[0], plant.index[-1]))

    assert {"electrolyser", "dri_plant", "eaf", "unit"} == {
        tech for tech, _ in dispatch
    }
    for entry in dispatch.values():
        assert entry["unit"] == plant.id
        assert len(entry["time"]) == len(entry["baseline"]) == len(plant.index)
        # no flexible operation determined yet
        assert np.isnan(entry["flex"]).all()

    # the technology schedules add up to the total power requirement of the unit
    power = sum(
        entry["baseline"]
        for (tech, var), entry in dispatch.items()
        if var == "power_in" and tech != "unit"
    )
    np.testing.assert_allclose(power, plant.opt_power_requirement.data, atol=1e-6)

    # unit-level series are captured under the pseudo technology "unit"
    np.testing.assert_allclose(
        dispatch[("unit", "total_power_input")]["baseline"],
        plant.opt_power_requirement.data,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        dispatch[("unit", "variable_cost")]["baseline"],
        plant.variable_cost_series.data,
        atol=1e-6,
    )
    # the electricity price is captured at solve time
    np.testing.assert_allclose(
        dispatch[("unit", "electricity_price")]["baseline"],
        plant.forecaster.electricity_price,
        atol=1e-6,
    )


def test_dispatch_carries_variable_metadata(plant):
    dispatch = _by_key(plant.get_component_schedule(plant.index[0], plant.index[-1]))

    assert dispatch[("electrolyser", "power_in")]["carrier"] == "electricity"
    assert dispatch[("electrolyser", "power_in")]["direction"] == "in"
    assert dispatch[("electrolyser", "power_in")]["unit_of_measure"] == "MW"
    assert dispatch[("electrolyser", "hydrogen_out")]["carrier"] == "hydrogen"
    assert dispatch[("electrolyser", "hydrogen_out")]["direction"] == "out"
    assert dispatch[("eaf", "steel_output")]["carrier"] == "steel"
    assert dispatch[("eaf", "operating_cost")]["direction"] == "cost"
    assert dispatch[("unit", "electricity_price")]["direction"] == "price"


def test_component_specs_are_captured(plant):
    specs = plant.get_component_specs()

    assert all(spec["unit"] == plant.id for spec in specs)
    by_key = {(s["technology"], s["parameter"]): s["value"] for s in specs}
    assert by_key[("electrolyser", "component_class")] == "Electrolyser"
    assert by_key[("electrolyser", "max_power")] == 100
    assert by_key[("eaf", "component_class")] == "ElectricArcFurnace"


def test_flex_dispatch_is_added_after_flex_optimisation(plant):
    plant.determine_optimal_operation_with_flex()
    dispatch = _by_key(plant.get_component_schedule(plant.index[0], plant.index[-1]))

    # the load shift variables are only part of the flexible optimisation
    flex_only = {"load_shift_pos", "load_shift_neg"}
    for (_, variable), entry in dispatch.items():
        assert not np.isnan(entry["flex"]).any()
        if variable in flex_only:
            assert np.isnan(entry["baseline"]).all()
        else:
            assert not np.isnan(entry["baseline"]).any()


def test_dispatch_is_sliced_to_requested_window(plant):
    start, end = plant.index[4], plant.index[9]
    entry = _by_key(plant.get_component_schedule(start, end))[("eaf", "power_in")]

    assert len(entry["time"]) == len(entry["baseline"]) == 6
    assert entry["time"][0] == start
    assert entry["time"][-1] == end


def test_dsm_schedule_is_written_to_csv(plant, tmp_path):
    plant.determine_optimal_operation_with_flex()
    output = WriteOutput(
        "test_sim",
        datetime(2023, 1, 1),
        datetime(2023, 1, 2),
        save_frequency_hours=None,
        export_csv_path=str(tmp_path),
    )
    output.handle_output_message(
        {
            "context": "write_results",
            "type": "dsm_schedule",
            "data": plant.get_component_schedule(plant.index[0], plant.index[-1]),
        },
        {"sender_id": None},
    )
    asyncio.run(output.store_dfs())

    df = pd.read_csv(tmp_path / "test_sim" / "dsm_schedule.csv", index_col="time")
    assert set(df.columns) == {
        "baseline",
        "carrier",
        "direction",
        "flex",
        "simulation",
        "technology",
        "unit",
        "unit_of_measure",
        "variable",
    }
    assert (df["simulation"] == "test_sim").all()
    assert (df["unit"] == plant.id).all()
    eaf = df[(df.technology == "eaf") & (df.variable == "power_in")]
    assert len(eaf) == len(plant.index)


def test_dsm_specs_are_written_to_csv(plant, tmp_path):
    output = WriteOutput(
        "test_sim",
        datetime(2023, 1, 1),
        datetime(2023, 1, 2),
        save_frequency_hours=None,
        export_csv_path=str(tmp_path),
    )
    output.handle_output_message(
        {
            "context": "write_results",
            "type": "dsm_specs",
            "data": plant.get_component_specs(),
        },
        {"sender_id": None},
    )
    asyncio.run(output.store_dfs())

    df = pd.read_csv(tmp_path / "test_sim" / "dsm_specs.csv", index_col="unit")
    assert set(df.columns) == {"parameter", "simulation", "technology", "value"}
    assert (df.index == plant.id).all()
    assert {"electrolyser", "dri_plant", "eaf"} <= set(df["technology"])


class _OutputContextStub:
    """Collects the messages the units operator sends to the output agent."""

    def __init__(self, now: datetime):
        self.current_timestamp = datetime2timestamp(now)
        self.data = {"output_agent_addr": "output_agent"}
        self.sent: list[dict] = []

    async def send_message(self, content, receiver_addr):
        self.sent.append(content)

    def schedule_instant_message(self, receiver_addr, content):
        self.sent.append(content)


class _PlainUnit:
    """A unit without component specs, e.g. a demand or power plant."""

    def as_dict(self) -> dict:
        return {"id": "plain", "unit_type": "demand"}


def _operator_with(units, now: datetime) -> UnitsOperator:
    operator = UnitsOperator(available_markets=[])
    operator._context = _OutputContextStub(now)
    for i, unit in enumerate(units):
        operator.units[f"unit_{i}"] = unit
    return operator


def _sent_types(operator):
    return [message["type"] for message in operator.context.sent]


def test_dsm_specs_are_sent_with_the_unit_meta_data(plant):
    operator = _operator_with([plant], now=plant.index[1])
    asyncio.run(operator.store_units())

    assert _sent_types(operator).count("dsm_specs") == 1
    specs = next(m for m in operator.context.sent if m["type"] == "dsm_specs")
    assert specs["data"] == plant.get_component_specs()


def test_dsm_specs_are_not_sent_without_dsm_units():
    operator = _operator_with([_PlainUnit()], now=datetime(2023, 1, 1))
    asyncio.run(operator.store_units())

    assert "dsm_specs" not in _sent_types(operator)


def test_dsm_specs_are_not_sent_again_on_every_dispatch(plant):
    """The specs are static, so they must not be exported with each dispatch."""
    operator = _operator_with([plant], now=plant.index[1])
    operator.write_actual_dispatch("energy")

    assert "dsm_specs" not in _sent_types(operator)
    assert "dsm_schedule" in _sent_types(operator)
