# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Command line entry: ``python -m assume_dash --csv <folder>`` or ``--db <url>``."""

import argparse

from assume_dash.app import create_app
from assume_dash.data import CsvSource, SqlSource


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ASSUME results dashboard")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--db", help="SQLAlchemy URL of the ASSUME database")
    group.add_argument("--csv", help="Folder with exported ASSUME CSV results")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--theme", choices=["light", "dark"], default="light")
    parser.add_argument("--css", help="CSS file overriding the theme variables")
    args = parser.parse_args(argv)

    source = SqlSource(args.db) if args.db else CsvSource(args.csv)
    app = create_app(source, theme=args.theme, custom_css=args.css)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
