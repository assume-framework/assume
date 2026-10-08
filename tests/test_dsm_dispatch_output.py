# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

import asyncio
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from assume.common.outputs import WriteOutput

from .test_steel_plant import create_steel_plant, dsm_components  # noqa: F401


@pytest.fixture
def plant(dsm_components):  # noqa: F811
    return create_steel_plant(dsm_components, "cost_based_load_shift")


def _by_key(dispatch):
    return {(d["technology"], d["variable"]): d for d in dispatch}


def test_baseline_dispatch_is_captured_per_technology(plant):
    dispatch = _by_key(plant.get_component_dispatch(plant.index[0], plant.index[-1]))

    assert {"electrolyser", "dri_plant", "eaf"} == {tech for tech, _ in dispatch}
    for entry in dispatch.values():
        assert entry["unit"] == plant.id
        assert len(entry["time"]) == len(entry["baseline"]) == len(plant.index)
        # no flexible operation determined yet
        assert np.isnan(entry["flex"]).all()

    # the technology schedules add up to the total power requirement of the unit
    power = sum(
        entry["baseline"] for (_, var), entry in dispatch.items() if var == "power_in"
    )
    np.testing.assert_allclose(power, plant.opt_power_requirement.data, atol=1e-6)


def test_flex_dispatch_is_added_after_flex_optimisation(plant):
    plant.determine_optimal_operation_with_flex()
    dispatch = _by_key(plant.get_component_dispatch(plant.index[0], plant.index[-1]))

    for entry in dispatch.values():
        assert not np.isnan(entry["flex"]).any()
        assert not np.isnan(entry["baseline"]).any()


def test_dispatch_is_sliced_to_requested_window(plant):
    start, end = plant.index[4], plant.index[9]
    entry = _by_key(plant.get_component_dispatch(start, end))[("eaf", "power_in")]

    assert len(entry["time"]) == len(entry["baseline"]) == 6
    assert entry["time"][0] == start
    assert entry["time"][-1] == end


def test_dsm_dispatch_is_written_to_csv(plant, tmp_path):
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
            "type": "dsm_dispatch",
            "data": plant.get_component_dispatch(plant.index[0], plant.index[-1]),
        },
        {"sender_id": None},
    )
    asyncio.run(output.store_dfs())

    df = pd.read_csv(tmp_path / "test_sim" / "dsm_dispatch.csv", index_col="time")
    assert set(df.columns) == {
        "baseline",
        "flex",
        "simulation",
        "technology",
        "unit",
        "variable",
    }
    assert (df["simulation"] == "test_sim").all()
    assert (df["unit"] == plant.id).all()
    eaf = df[(df.technology == "eaf") & (df.variable == "power_in")]
    assert len(eaf) == len(plant.index)
