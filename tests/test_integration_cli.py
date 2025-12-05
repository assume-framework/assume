# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import os

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sqlalchemy import create_engine

from assume_cli.cli import cli


@pytest.mark.slow
def test_cli():
    dburi = "sqlite:///./examples/local_db/test_mini.db"
    args = f"-s example_01a -c tiny -db {dburi}"
    cli(args.split(" "))

    db = create_engine(dburi)
    got = {}
    with db.begin() as conn:
        got = {
            "market_meta": pd.read_sql("SELECT * FROM market_meta", conn),
            "demand_meta": pd.read_sql("SELECT * FROM demand_meta", conn),
            "exchange_meta": pd.read_sql("SELECT * FROM exchange_meta", conn),
        }
    expected = {
        "market_meta": pd.read_csv("./tests/fixtures/01a_results/market_meta.csv"),
        "demand_meta": pd.read_csv("./tests/fixtures/01a_results/demand_meta.csv"),
        "exchange_meta": pd.read_csv("./tests/fixtures/01a_results/exchange_meta.csv"),
    }
    for key in expected.keys():
        assert_frame_equal(got[key], expected[key], check_dtype=False)


@pytest.mark.slow
@pytest.mark.require_network
def test_cli_network():
    dburi = "sqlite:///./examples/local_db/test_mini_net.db"
    args = f"-s example_01d -c base -db {dburi}"
    cli(args.split(" "))


@pytest.mark.slow
@pytest.mark.require_learning
def test_cli_learning():
    os.environ["NON_INTERACTIVE"] = "1"
    dburi = "sqlite:///./examples/local_db/test_mini_rl.db"
    args = f"-s example_02a -c tiny -db {dburi}"
    cli(args.split(" "))

    db = create_engine(dburi)
    with db.begin() as conn:
        got = pd.read_sql(
            """
                          SELECT episode,
                                 unit,
                                 SUM(reward) AS total_reward
                          FROM rl_params
                          WHERE simulation = 'example_02a_tiny'
                            AND evaluation_mode = FALSE
                          GROUP BY episode, unit
                          """,
            conn,
        )
    assert len(got) == 10
    # check improvement during training (excluding initial exploration)
    assert got["total_reward"][3] < got["total_reward"][9]
