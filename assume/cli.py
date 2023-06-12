import argparse
import os
import sys

from assume import World

os.makedirs("./examples/outputs", exist_ok=True)
os.makedirs("./examples/local_db", exist_ok=True)


def cli(args=None):
    if not args:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Command Line Interface for ASSUME simulations"
    )
    parser.add_argument(
        "-s",
        "--scenario",
        help="name of the scenario file which should be used",
        default="example_01a",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--case-study",
        help="name of the case in that scenario which should be simulated",
        default="",
        type=str,
    )
    parser.add_argument(
        "-csv",
        "--csv-export-path",
        help="optional path to the csv export",
        default="",
        type=str,
    )
    parser.add_argument(
        "-db",
        "--db-uri",
        help="uri string for a database",
        default="",
        type=str,
    )
    parser.add_argument(
        "-input",
        "--input-path",
        help="path to the input folder",
        default="examples/inputs",
        type=str,
    )

    args = parser.parse_args(args)
    print(args)
    name = args.scenario
    if args.db_uri:
        db_uri = args.db_uri
    else:
        db_uri = f"sqlite:///./examples/local_db/{name}.db"

    world = World(database_uri=db_uri, export_csv_path=args.csv_export_path)
    world.load_scenario(
        inputs_path=args.input_path,
        scenario=args.scenario,
        study_case=args.case_study,
    )
    world.run()


if __name__ == "__main__":
    # cli()
    args = "-s example_01a -c example_01a -db postgresql://assume:assume@localhost:5432/assume"
    args = "-s example_02 -c ltm_case05 -db postgresql://assume:assume@localhost:5432/assume"
    cli(args.split(" "))
