# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine

from assume.common.outputs import WriteOutput

os.makedirs("./examples/local_db", exist_ok=True)
DB_URI = "sqlite:///./examples/local_db/test_outputs.db"


def test_output_market_orders():
    engine = create_engine(DB_URI)
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 2)
    output_writer = WriteOutput("test_sim", start, end, engine)
    assert len(output_writer.write_buffers.keys()) == 0
    meta = {"sender_id": None}
    content = {
        "context": "write_results",
        "type": "market_orders",
        "sender": "CRM_pos",
        "data": [],
    }
    output_writer.handle_output_message(content, meta)
    assert len(output_writer.write_buffers["market_orders"]) == 0

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 120,
            "price": 120,
            "agent_addr": "gen1",
            "only_hours": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 80,
            "price": 58,
            "agent_addr": "gen1",
            "only_hours": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 100,
            "price": 53,
            "agent_addr": "gen1",
            "only_hours": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": -180,
            "price": 70,
            "agent_addr": "dem1",
            "only_hours": None,
        },
    ]

    content = {
        "context": "write_results",
        "type": "market_orders",
        "sender": "CRM_pos",
        "data": orderbook,
    }
    output_writer.handle_output_message(content, meta)
    assert len(output_writer.write_buffers["market_orders"]) == 1


def test_output_market_results():
    engine = create_engine(DB_URI)
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 2)
    output_writer = WriteOutput("test_sim", start, end, engine)
    assert len(output_writer.write_buffers.keys()) == 0
    meta = {"sender_id": None}
    content = {
        "context": "write_results",
        "type": "market_meta",
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
    output_writer.handle_output_message(content, meta)
    assert len(output_writer.write_buffers["market_meta"]) == 1, "market_meta"


def test_output_market_dispatch():
    engine = create_engine(DB_URI)
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 2)
    output_writer = WriteOutput("test_sim", start, end, engine)
    assert len(output_writer.write_buffers.keys()) == 0
    meta = {"sender_id": None}
    content = {"context": "write_results", "type": "market_dispatch", "data": []}
    output_writer.handle_output_message(content, meta)
    # empty dfs are discarded
    assert len(output_writer.write_buffers["market_dispatch"]) == 0, "market_dispatch"

    content = {
        "context": "write_results",
        "type": "market_dispatch",
        "data": [[start, 90, "EOM", "TestUnit"]],
    }
    output_writer.handle_output_message(content, meta)
    assert len(output_writer.write_buffers["market_dispatch"]) == 1, "market_dispatch"


def test_output_unit_dispatch():
    engine = create_engine(DB_URI)
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 2)
    output_writer = WriteOutput("test_sim", start, end, engine)
    assert len(output_writer.write_buffers.keys()) == 0
    meta = {"sender_id": None}
    content = {
        "context": "write_results",
        "type": "unit_dispatch",
        "data": [
            {
                "power": np.array([0.0, 1000.0]),
                "energy_cashflow": np.array([0.0, 45050.0]),
                "time": [datetime(2022, 1, 1, 0), datetime(2022, 1, 1, 1)],
                "unit": "Unit 2",
            }
        ],
    }

    output_writer.handle_output_message(content, meta)
    assert len(output_writer.write_buffers["unit_dispatch"]) == 1, "unit_dispatch"


def test_output_write_flows():
    engine = create_engine(DB_URI)
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 2)
    output_writer = WriteOutput("test_sim", start, end, engine)
    assert len(output_writer.write_buffers.keys()) == 0
    meta = {"sender_id": None}
    content = {
        "context": "write_results",
        "type": "grid_flows",
        "data": {(datetime(2019, 1, 1, 0, 0), "north_south_example"): 0.0},
    }

    output_writer.handle_output_message(content, meta)
    assert len(output_writer.write_buffers["grid_flows"]) == 1, "grid_flows"


@pytest.mark.asyncio
async def test_adaptive_merit_order_forecast_output_schema(tmp_path):
    simulation_start = datetime(2025, 1, 1)
    db_uri = f"sqlite:///{tmp_path / 'adaptive-merit-order.db'}"
    writer = WriteOutput(
        "adaptive-merit-order-test",
        simulation_start,
        simulation_start + timedelta(days=1),
        None,
        db_uri=db_uri,
        export_csv_path=str(tmp_path / "csv"),
    )
    writer.db = create_engine(db_uri)
    record = {
        "forecast_id": "operator|EOM|issue|product",
        "unit_operator_id": "operator",
        "market_id": "EOM",
        "issue_time": simulation_start,
        "product_start": simulation_start + timedelta(hours=1),
        "merit_order_price_forecast": 55.0,
        "residual_mean_forecast": -4.0,
        "corrected_price_mean_forecast": 51.0,
        "residual_std_forecast": 3.0,
        "price_q10": 47.0,
        "price_q50": 51.0,
        "price_q90": 55.0,
        "training_status": "trained",
        "training_sample_count": 168,
        "realised_price": 49.0,
        "realised_residual": -6.0,
        "post_forecast_residual": -2.0,
    }
    writer.handle_output_message(
        {
            "context": "write_results",
            "type": "adaptive_merit_order_forecast",
            "data": [record],
        },
        {},
    )
    frame = writer.convert_adaptive_merit_order_forecasts(
        writer.write_buffers["adaptive_merit_order_forecast"]
    )

    assert frame.index.name == "product_start"
    assert frame.iloc[0]["realised_residual"] == -6
    await writer.store_dfs()
    csv_frame = pd.read_csv(
        tmp_path
        / "csv"
        / "adaptive-merit-order-test"
        / "adaptive_merit_order_forecast.csv"
    )
    with writer.db.connect() as connection:
        db_frame = pd.read_sql("adaptive_merit_order_forecast", connection)
    assert csv_frame.loc[0, "forecast_id"] == record["forecast_id"]
    assert db_frame.loc[0, "forecast_id"] == record["forecast_id"]
