# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine

from assume.common.outputs import WriteOutput

os.makedirs("./examples/local_db", exist_ok=True)
DB_URI = "sqlite:///./examples/local_db/test_outputs.db"


def test_output_market_orders():
    engine = create_engine(DB_URI)
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 2)
    output_writer = WriteOutput("test_sim", start, end, engine)
    assert len(output_writer.write_dfs.keys()) == 0
    meta = {"sender_id": None}
    content = {
        "context": "write_results",
        "type": "store_order_book",
        "sender": "CRM_pos",
        "data": [],
    }
    output_writer.handle_message(content, meta)
    assert len(output_writer.write_dfs["market_orders"]) == 0

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 120,
            "price": 120,
            "agent_id": "gen1",
            "only_hours": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 80,
            "price": 58,
            "agent_id": "gen1",
            "only_hours": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 100,
            "price": 53,
            "agent_id": "gen1",
            "only_hours": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": -180,
            "price": 70,
            "agent_id": "dem1",
            "only_hours": None,
        },
    ]

    content = {
        "context": "write_results",
        "type": "store_order_book",
        "sender": "CRM_pos",
        "data": orderbook,
    }
    output_writer.handle_message(content, meta)
    assert len(output_writer.write_dfs["market_orders"]) == 1


def test_output_market_results():
    engine = create_engine(DB_URI)
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 2)
    output_writer = WriteOutput("test_sim", start, end, engine)
    assert len(output_writer.write_dfs.keys()) == 0
    meta = {"sender_id": None}
    content = {
        "context": "write_results",
        "type": "store_market_results",
        "sender": "CRM_pos",
        "data": [
            {
                "supply_volume": 0,
                "demand_volume": 0,
                "demand_volume_energy": 0.0,
                "supply_volume_energy": 0.0,
                "price": 0.0,
                "max_price": 0,
                "min_price": 0,
                "node": None,
                "product_start": datetime(2019, 1, 1, 2),
                "product_end": datetime(2019, 1, 1, 6),
                "only_hours": None,
                "market_id": "CRM_pos",
                "time": 1546302600,
            }
        ],
    }
    output_writer.handle_message(content, meta)
    assert len(output_writer.write_dfs["market_meta"]) == 1, "market_meta"


def test_output_market_dispatch():
    engine = create_engine(DB_URI)
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 2)
    output_writer = WriteOutput("test_sim", start, end, engine)
    assert len(output_writer.write_dfs.keys()) == 0
    meta = {"sender_id": None}
    content = {"context": "write_results", "type": "market_dispatch", "data": []}
    output_writer.handle_message(content, meta)
    # empty dfs are discarded
    assert len(output_writer.write_dfs["market_dispatch"]) == 0, "market_dispatch"

    content = {
        "context": "write_results",
        "type": "market_dispatch",
        "data": [[start, 90, "EOM", "TestUnit"]],
    }
    output_writer.handle_message(content, meta)
    assert len(output_writer.write_dfs["market_dispatch"]) == 1, "market_dispatch"


def test_output_unit_dispatch():
    engine = create_engine(DB_URI)
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 2)
    output_writer = WriteOutput("test_sim", start, end, engine)
    assert len(output_writer.write_dfs.keys()) == 0
    meta = {"sender_id": None}
    content = {
        "context": "write_results",
        "type": "unit_dispatch",
        "data": pd.DataFrame(
            {
                "power": {
                    pd.Timestamp("2019-01-01 00:00:00"): 0.0,
                    pd.Timestamp("2019-01-01 01:00:00"): 0.0,
                },
                "unit": {
                    pd.Timestamp("2019-01-01 00:00:00"): "demand_EOM",
                    pd.Timestamp("2019-01-01 01:00:00"): "demand_EOM",
                },
            }
        ),
    }
    output_writer.handle_message(content, meta)
    assert len(output_writer.write_dfs["unit_dispatch"]) == 1, "unit_dispatch"
