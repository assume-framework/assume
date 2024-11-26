# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime

import pandas as pd
import pytest

from assume.common.forecasts import NaiveForecast
from assume.strategies import (
    flexableEOMBlock,
    flexableEOMLinked,
)
from assume.units import PowerPlant

start = datetime(2023, 7, 1)
end = datetime(2023, 7, 2)


@pytest.fixture
def power_plant() -> PowerPlant:
    # Create a PowerPlant instance with some example parameters
    index = pd.date_range("2023-07-01", periods=48, freq="h")
    ff = NaiveForecast(index, availability=1, fuel_price=10, co2_price=10)
    return PowerPlant(
        id="test_pp",
        unit_operator="test_operator",
        technology="coal",
        max_power=1000,
        min_power=200,
        efficiency=0.5,
        additional_cost=10,
        bidding_strategies={},
        fuel_type="lignite",
        emission_factor=0.5,
        forecaster=ff,
    )


def test_eom_with_blocks(mock_market_config, power_plant):
    power_plant.ramp_up = 400
    product_index = pd.date_range("2023-07-01", periods=24, freq="h")
    strategy = flexableEOMBlock()
    mc = mock_market_config
    mc.product_type = "energy_eom"
    product_tuples = [
        (start, start + pd.Timedelta(hours=1), None) for start in product_index
    ]
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert len(bids) == 25
    assert bids[0]["price"] == 40
    assert bids[0]["volume"] == 200
    assert bids[1]["price"] == 40
    assert bids[1]["volume"] == 600
    assert bids[2]["price"] == 40
    assert bids[2]["volume"] == 800
    assert bids[-1]["price"] == 40
    assert bids[-1]["volume"] == {time: 200 for time in product_index}

    # add min_down_time, min_operating_time
    power_plant.index = pd.date_range("2023-07-01", periods=48, freq="h")
    power_plant.min_down_time = 2
    power_plant.min_operating_time = 3
    product_index = pd.date_range("2023-07-02", periods=24, freq="h")
    product_tuples = [
        (start, start + pd.Timedelta(hours=1), None) for start in product_index
    ]

    # set the outputs before the first bid to 0
    power_plant.outputs["energy"][power_plant.index[0:24]] = 0
    assert power_plant.get_operation_time(product_index[0]) == -3

    # test min_down_time, set op_time to -1
    power_plant.outputs["energy"][power_plant.index[10:23]] = 200
    assert power_plant.get_operation_time(product_index[0]) == -1

    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert len(bids) == 24
    assert bids[0]["price"] == 40
    assert bids[0]["volume"] == 200
    assert bids[1]["price"] == 40
    assert bids[1]["volume"] == 600
    assert bids[2]["price"] == 40
    assert bids[2]["volume"] == 800
    assert bids[-1]["price"] == 40
    assert bids[-1]["volume"] == {
        product_index[i]: 0 if i == 0 else 200 for i in range(len(product_index))
    }


def test_eom_with_links(mock_market_config, power_plant):
    power_plant.ramp_up = 400
    product_index = pd.date_range("2023-07-01", periods=24, freq="h")
    strategy = flexableEOMLinked()
    mc = mock_market_config
    mc.product_type = "energy_eom"
    product_tuples = [
        (start, start + pd.Timedelta(hours=1), None) for start in product_index
    ]
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert len(bids) == 25
    assert bids[0]["price"] == 40
    assert bids[0]["volume"] == {product_index[0]: 200}
    assert bids[0]["parent_bid_id"] == power_plant.id + "_block"
    assert bids[1]["price"] == 40
    assert bids[1]["volume"] == {product_index[1]: 600}
    assert bids[1]["parent_bid_id"] == power_plant.id + "_block"
    assert bids[2]["price"] == 40
    assert bids[2]["volume"] == {product_index[2]: 800}
    assert bids[2]["parent_bid_id"] == power_plant.id + "_block"
    assert bids[-1]["price"] == 40
    assert bids[-1]["volume"] == {time: 200 for time in product_index}
    assert bids[-1]["bid_id"] == power_plant.id + "_block"

    # add min_down_time, min_operating_time
    power_plant.index = pd.date_range("2023-07-01", periods=48, freq="h")
    power_plant.min_down_time = 2
    power_plant.min_operating_time = 3
    product_index = pd.date_range("2023-07-02", periods=24, freq="h")
    product_tuples = [
        (start, start + pd.Timedelta(hours=1), None) for start in product_index
    ]

    # set the outputs before the first bid to 0
    power_plant.outputs["energy"][power_plant.index[0:24]] = 0
    assert power_plant.get_operation_time(product_index[0]) == -3

    # test min_down_time, set op_time to -1
    power_plant.outputs["energy"][power_plant.index[10:23]] = 200
    assert power_plant.get_operation_time(product_index[0]) == -1

    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert len(bids) == 24
    assert bids[0]["price"] == 40
    assert bids[0]["volume"] == {product_index[1]: 200}
    assert bids[0]["parent_bid_id"] == power_plant.id + "_block"
    assert bids[1]["price"] == 40
    assert bids[1]["volume"] == {product_index[2]: 600}
    assert bids[1]["parent_bid_id"] == power_plant.id + "_block"
    assert bids[-1]["price"] == 40
    assert bids[-1]["volume"] == {
        product_index[i]: 0 if i == 0 else 200 for i in range(len(product_index))
    }
    assert bids[-1]["bid_id"] == power_plant.id + "_block"
