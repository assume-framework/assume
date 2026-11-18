#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

import argparse
import logging
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
        with open(str(config_file)) as f:
            config = yaml.safe_load(f)
        return list(config.keys())
    return [""]


def create_parser() -> argparse.ArgumentParser:
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

    parser.add_argument(
        "-p",
        "--parallel",
        help="run simulation with multiple processes",
        action="store_true",
    )
    return parser


def cli(args=None):
    if not args:
        args = sys.argv[1:]
    parser = create_parser()

    argcomplete.autocomplete(parser)
    args = parser.parse_args(args)
    if args.db_uri:
        db_uri = make_url(args.db_uri)
    else:
        db_uri = ""

    # add these two weird hacks for now
    warnings.filterwarnings("ignore", "coroutine.*?was never awaited.*")
    logging.getLogger("asyncio").setLevel("FATAL")

    # import package after argcomplete.autocomplete
    # to improve autocompletion speed
    from assume import World
    from assume.common.exceptions import AssumeException
    from assume.scenario.loader_csv import load_scenario_folder, run_learning

    try:
        Path("./examples/local_db").mkdir(parents=True, exist_ok=True)

        if args.parallel:
            distributed_role = True
            addr = ("localhost", 9100)
        else:
            distributed_role = None
            addr = "world"

        world = World(
            database_uri=db_uri,
            export_csv_path=args.csv_export_path,
            log_level=args.loglevel,
            distributed_role=distributed_role,
            addr=addr,
        )
        load_scenario_folder(
            world,
            inputs_path=args.input_path,
            scenario=args.scenario,
            study_case=args.case_study,
        )

        logging.info(f"loaded {args.scenario} - {args.case_study}")

        if world.learning_mode:
            run_learning(world)

        world.run()

    except KeyboardInterrupt:
        sys.exit(1)
    except AssumeException as e:
        logging.error(f"Stopping: {e}")
        sys.exit(1)
    except Exception:
        logging.exception("Simulation aborted")
        sys.exit(1)


def market_config_case_completer(prefix, parsed_args, **kwargs):
    config_file = Path(parsed_args.config)
    if config_file.is_dir():
        config_file = config_file / "config.yaml"
    if config_file.is_file():
        with open(str(config_file)) as f:
            config = yaml.safe_load(f)
        return list(config.keys())
    return [""]


def create_market_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show and check the timing of the markets of an ASSUME config before running a simulation",
    )
    parser.add_argument(
        "config",
        help="path to the config.yaml or to the scenario folder containing it",
        type=str,
    ).completer = argcomplete.FilesCompleter()
    parser.add_argument(
        "-c",
        "--case-study",
        help="name of the case in the config which should be shown, defaults to the first one",
        default="",
        type=str,
    ).completer = market_config_case_completer
    parser.add_argument(
        "--start",
        help="start of the shown interval, defaults to the first market opening",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--end",
        help="end of the shown interval, defaults to one full cycle of all markets",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--ascii",
        help="use ASCII instead of unicode box drawing characters",
        action="store_true",
    )
    parser.add_argument(
        "--mermaid",
        help="print the mermaid gantt diagram instead of rendering it in the terminal",
        action="store_true",
    )
    return parser


def market_cli(args=None):
    """
    Render the market timing of a config and print the timing issues.
    Exits with code 1 if the timing contains errors.
    """
    if not args:
        args = sys.argv[1:]
    parser = create_market_parser()

    argcomplete.autocomplete(parser)
    args = parser.parse_args(args)

    import pandas as pd

    from assume.common.market_timing import (
        check_market_timing,
        market_timing_to_mermaid,
        render_market_timing,
    )
    from assume.scenario.loader_csv import load_market_configs

    try:
        market_configs = load_market_configs(args.config, args.case_study)
    except (OSError, ValueError) as e:
        parser.error(str(e))

    start = pd.Timestamp(args.start).to_pydatetime() if args.start else None
    end = pd.Timestamp(args.end).to_pydatetime() if args.end else None
    title = args.case_study or "Market timing"
    try:
        "█▓◆┊".encode(sys.stdout.encoding or "ascii")
        use_ascii = args.ascii
    except UnicodeEncodeError:
        # e.g. piped output on windows can not show box drawing characters
        use_ascii = True

    if args.mermaid:
        print(market_timing_to_mermaid(market_configs, start, end, title))
    else:
        print(
            render_market_timing(market_configs, start, end, title, use_ascii=use_ascii)
        )

    issues = check_market_timing(market_configs, start, end)
    if any(issue.level == "error" for issue in issues):
        sys.exit(1)


if __name__ == "__main__":
    cli()

    # args = "-s example_03 -db postgresql://assume:assume@localhost:5432/assume"

    # cli(args.split(" "))
