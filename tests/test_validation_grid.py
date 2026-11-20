# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from assume.scenario.validation import verify_scenario_folder

EXAMPLE = Path("examples/inputs/example_01d")


@pytest.fixture
def grid(tmp_path) -> Path:
    """A copy of example_01d (nodal, zonal and redispatch markets on a grid)."""
    target = tmp_path / "scenario"
    shutil.copytree(EXAMPLE, target)
    return target


def edit_df(folder: Path, name: str, change) -> None:
    path = folder / name
    df = pd.read_csv(path, index_col=0)
    result = change(df)  # returns a new frame or edits in place and returns None
    (df if result is None else result).to_csv(path)


def verify(folder: Path, severity: str = "error") -> str:
    report = verify_scenario_folder(folder, dry_run=False)
    return "\n".join(str(i) for i in report.issues if i.severity == severity)


def test_valid_grid(grid):
    assert verify(grid) == ""


def test_grid_file_missing(grid):
    (grid / "lines.csv").unlink()
    assert "needs 'lines.csv'" in verify(grid)


def test_grid_market_without_network_path(grid):
    config = yaml.safe_load((grid / "config.yaml").read_text())
    for case in config.values():
        for market in case["markets_config"].values():
            if market["market_mechanism"] == "nodal_clearing":
                del market["param_dict"]["network_path"]
    (grid / "config.yaml").write_text(yaml.safe_dump(config))
    assert "needs a grid" in verify(grid)


def test_unknown_zones_identifier(grid):
    edit_df(grid, "buses.csv", lambda df: df.rename(columns={"zone_id": "zone"}))
    assert "zones_identifier 'zone_id' is no column of buses.csv" in verify(grid)


def test_line_columns_and_buses(grid):
    edit_df(grid, "lines.csv", lambda df: df.drop(columns="s_nom"))
    assert "missing column 's_nom'" in verify(grid)


def test_line_refers_to_unknown_bus(grid):
    edit_df(
        grid,
        "lines.csv",
        lambda df: df.__setitem__("bus1", ["suth", "south", "north_2"]),
    )
    errors = verify(grid)
    assert "refers to buses missing in buses.csv: suth" in errors
    assert "did you mean 'south'" in errors


def test_line_values(grid):
    edit_df(grid, "lines.csv", lambda df: df.__setitem__("s_nom", ["a", 5000, -1]))
    assert "'s_nom' must be a number" in verify(grid)


def test_duplicate_names(grid):
    edit_df(grid, "buses.csv", lambda df: df.set_axis(["south", "south", "south"]))
    assert "duplicate bus names: south" in verify(grid)


def test_unit_on_unknown_bus_is_warned(grid):
    edit_df(grid, "demand_units.csv", lambda df: df.__setitem__("node", "nowhere"))
    assert "is no bus of buses.csv" in verify(grid, "warning")
