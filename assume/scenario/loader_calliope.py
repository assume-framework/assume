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

# model = calliope.Model("path/to/model.yaml", override_dict=override_dict)
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

    if model_data["config"]["build"]["mode"] == "plan":
        raise Exception("Only operate is currently supported in ASSUME")

    time_subset = model_data["config"]["init"]["time_subset"]

    start, end = time_subset

    save_interval = 48

    index = pd.date_range(start=start, end=end, freq="1h")

    # TODO replace data_sources with actual data

    await world.setup(
        start=start,
        end=end,
        save_frequency_hours=save_interval,
        simulation_id=name,
        index=index,
    )

    # TODO add market operator

    agents = []

    for source_name, data_source in model_data["data_sources"].items():
        if data_source["source"]:
            source_path = data_source["source"]
            if data_source.get("drop"):
                skip = 1
                header = 0
            else:
                skip = 0
                if isinstance(data_source["columns"], list):
                    header = len(data_source["columns"])
                else:
                    header = 0
            df = pd.read_csv(
                model_path / source_path,
                header=header,
                skiprows=skip,
                parse_dates=True,
                index_col=0,
            )
        else:
            df = pd.DataFrame(index=index)

        # TODO don't know what this is for
        # data_source now has the actual tech
        tech = data_source["add_dimensions"]["techs"]

        translate_dict = {}  # TODO

        # amiris annotation
        #         {
        #     "NUCLEAR": "nuclear",
        #     "LIGNITE": "lignite",
        #     "HARD_COAL": "hard coal",
        #     "NATURAL_GAS": "natural gas",
        #     "OIL": "oil",
        #     "HYDROGEN": "hydrogen",
        #     "Biogas": "biomass",
        #     "PV": "solar",
        #     "WindOn": "wind_onshore",
        #     "WindOff": "wind_offshore",
        #     "RunOfRiver": "hydro",
        #     "Other": "other",
        # }

        world.unit_types.keys()

        for column in df.columns:
            tech, node = column

        tech_dict = model_data["techs"][tech]

        match tech_dict["base_tech"]:
            case "supply":
                # here we know exactly which parameters exist
                pass
            case "conversion":
                max_power = tech_dict["flow_cap_max"]["data"]
                efficiency = tech_dict["flow_out_eff"]["data"]
                # TODO durch mögliche colums iterieren
                agents.append(
                    {
                        "agent": df.columns[0] + "random",
                        "node": node,
                        "technology": actual_tech,
                    }
                )
            case "demand":
                max_power = df["demand_heat", "X1"].max()
                availability = df["demand_heat", "X1"] / max_power
                min_power = 0
                for column in df.columns:
                    tech, node = column
                actual_tech, product_type = translate_dict.get(tech)
                # TODO check if product type is demand or actual tech
                agents.append(
                    {
                        "agent": df.columns,
                        "node": node,
                        "technology": actual_tech,
                    }
                )

                world.add_unit_operator(source_name)
                world.add_unit(
                    f"{source_name}1",
                    "demand",  # TODO - power_plant?
                    source_name,  # units_operator name
                    # the unit_params have no hints
                    {
                        "min_power": min_power,
                        "max_power": max_power,
                        "bidding_strategies": {"EOM": "naive_eom"},
                        "technology": "demand",
                        "node": "location1",
                    },
                    NaiveForecast(
                        index, demand=df["demand_heat", "X1"], availability=1
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
