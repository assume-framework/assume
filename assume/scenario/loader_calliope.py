# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import calendar
import logging
from datetime import timedelta

import dateutil.rrule as rr
import pandas as pd
import yaml
from dateutil.relativedelta import relativedelta as rd
from yamlinclude import YamlIncludeConstructor

import assume
from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.world import World
from pathlib import Path
import importlib

# must be installed for import path to work
import calliope

#model = calliope.Model("path/to/model.yaml", override_dict=override_dict)
model = calliope.examples.urban_scale()


def read_csv(base_path, filename):
    return pd.read_csv(
        base_path + "/" + filename,
        date_format="%Y-%m-%d_%H:%M:%S",
        sep=";",
        header=None,
        names=["time", "load"],
        index_col="time",
    )["load"]


input_path = Path(importlib.resources.files("calliope") / "example_models")
name = "urban_scale"
model_path = input_path / name
override_dict = {"config": {"init": {"time_subset": ["2005-07-07", "2005-07-08"]}}}


def update_dict_keys(initial_dict: dict, override_dict: dict):
    for key, value in override_dict.items():
        if isinstance(value, dict):
            if key in initial_dict:
                update_dict_keys(initial_dict[key], value)
            else:
                initial_dict[key] = value
        else:
            initial_dict[key] = value


async def load_calliope_async(
    world: World, name: str, input_path: Path, override_dict: dict = {}
):
    model_path = input_path / name
    with open(model_path / "model.yaml", "r") as f:
        model_data = yaml.safe_load(f)

    for path in model_data["import"]:
        with open(model_path / path, "r") as f:
            import_dict = yaml.safe_load(f)
            update_dict_keys(model_data, import_dict)

    update_dict_keys(model_data, override_dict)

    if model_data["config"]["mode"] == "plan":
        raise Exception("Only operate is currently supported in ASSUME")

    time_subset = model_data["config"]["init"]["time_subset"]

    start, end = time_subset

    save_interval = 48

    index = pd.date_range(start=start, end=end, freq="1H")

    # TODO replace data_sources with actual data

    await world.setup(
        start=start,
        end=end,
        save_frequency_hours=save_interval,
        simulation_id=name,
        index=index,
    )

    # TODO add market operator

    # todo merge techs with data_sources
    model_data["techs"]
    for source_name, data_source in model_data["data_sources"].items():
        # add assume agent
        await world.add_agent(
            data_source["agent"],
            data_source["agent_id"],
            data_source["agent_name"],
            data_source["agent_type"],
            data_source["agent_config"],
        )

        world.add_unit_operator(source_name)
        world.add_unit(
            f"{source_name}1",
            "demand",  # TODO - power_plant?
            source_name,  # units_operator name
            # the unit_params have no hints
            {
                "min_power": 0,
                "max_power": 1000,
                "bidding_strategies": {"EOM": "naive_eom"},
                "technology": "demand",
                "node": "location1",
            },
            NaiveForecast(
                index, demand=100, availability=1
            ),  # hier zeitreihen hinterlegen
        )


if __name__ == "__main__":
    db_uri = "postgresql://assume:assume@localhost:5432/assume"
    world = World(database_uri=db_uri)
    scenario = "urban_scale"

    base_path = Path(importlib.resources.files("calliope") / "example_models")
    world.loop.run_until_complete(
        load_calliope_async(
            world,
            scenario,
            base_path,
        )
    )
    print(f"did load {scenario} - now simulating")
    world.run()
