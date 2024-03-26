#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import argcomplete
import yaml
from sqlalchemy import make_url


def db_uri_completer(prefix, parsed_args, **kwargs):
    return {
        "sqlite:///example.db": "example",
        f"sqlite:///examples/local_db/{parsed_args.scenario}.db": "current scenario",
        "sqlite://": "in-memory",
        "postgresql://assume:assume@localhost:5432/assume": "localhost",
        "postgresql://assume:assume@assume_db:5432/assume": "docker",
        "mysql://username:password@localhost:3306/database": "mysql",
    }


def config_directory_completer(prefix, parsed_args, **kwargs):
    directory = Path(parsed_args.input_path)
    if directory.is_dir():
        config_folders = [
            folder
            for folder in directory.iterdir()
            if folder.is_dir() and (folder / "config.yaml").exists()
        ]
        return [
            folder.name for folder in config_folders if folder.name.startswith(prefix)
        ]
    return [""]


def config_case_completer(prefix, parsed_args, **kwargs):
    config_file = (
        Path(parsed_args.input_path) / Path(parsed_args.scenario) / "config.yaml"
    )
    if config_file.is_file():
        with open(str(config_file), "r") as f:
            config = yaml.safe_load(f)
        return list(config.keys())
    return [""]


def cli(args=None):
    if not args:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Command Line Interface for ASSUME simulations",
    )
    parser.add_argument(
        "-s",
        "--scenario",
        help="name of the scenario file which should be used",
        default="example_01a",
        type=str,
    ).completer = config_directory_completer
    parser.add_argument(
        "-c",
        "--case-study",
        help="name of the case in that scenario which should be simulated",
        default="",
        type=str,
    ).completer = config_case_completer
    parser.add_argument(
        "-csv",
        "--csv-export-path",
        help="optional path to the csv export",
        default="",
        type=str,
    ).completer = argcomplete.DirectoriesCompleter()
    parser.add_argument(
        "-db",
        "--db-uri",
        help="uri string for a database",
        default="",
        type=str,
    ).completer = db_uri_completer
    parser.add_argument(
        "-i",
        "--input-path",
        help="path to the input folder",
        default="examples/inputs",
        type=str,
    ).completer = argcomplete.DirectoriesCompleter()
    parser.add_argument(
        "-l",
        "--loglevel",
        help="logging level used for file log",
        default="INFO",
        type=str,
        metavar="LOGLEVEL",
        choices=set(logging._nameToLevel.keys()),
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args(args)
    name = args.scenario
    if args.db_uri:
        db_uri = make_url(args.db_uri)
    else:
        db_uri = f"sqlite:///./examples/local_db/{name}.db"

    # add these two weird hacks for now
    warnings.filterwarnings("ignore", "coroutine.*?was never awaited.*")
    logging.getLogger("asyncio").setLevel("FATAL")

    try:
        # import package after argcomplete.autocomplete
        # to improve autocompletion speed
        from assume import World
        from assume.scenario.loader_csv import load_scenario_folder, run_learning

        os.makedirs("./examples/local_db", exist_ok=True)

        world = World(
            database_uri=db_uri,
            export_csv_path=args.csv_export_path,
            log_level=args.loglevel,
            #distributed_role=True,
            #addr=("0.0.0.0", 9099)
        )
        load_scenario_folder(
            world,
            inputs_path=args.input_path,
            scenario=args.scenario,
            study_case=args.case_study,
        )

        if world.learning_config.get("learning_mode", False):
            run_learning(
                world,
                inputs_path=args.input_path,
                scenario=args.scenario,
                study_case=args.case_study,
            )

        world.run()

    except KeyboardInterrupt:
        pass
    except Exception:
        logging.exception("Simulation aborted")


if __name__ == "__main__":
    cli()

    # args = "-s example_02 -db postgresql://assume:assume@localhost:5432/assume"

    # cli(args.split(" "))
