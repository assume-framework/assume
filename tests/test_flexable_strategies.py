# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
from datetime import datetime

import pandas as pd
import pytest

from assume.common.forecasts import NaiveForecast
from assume.strategies import (
    flexableEOM,
    flexableNegCRM,
    flexablePosCRM,
)
from assume.units import PowerPlant

start = datetime(2023, 7, 1)
end = datetime(2023, 7, 2)


@pytest.fixture
def power_plant() -> PowerPlant:
    # Create a PowerPlant instance with some example parameters
    index = pd.date_range("2023-07-01", periods=48, freq="h")
    forecaster = NaiveForecast(index, availability=1, fuel_price=10, co2_price=10)
    return PowerPlant(
        id="test_pp",
        unit_operator="test_operator",
        technology="hard coal",
        index=forecaster.index,
        max_power=1000,
        min_power=200,
        efficiency=0.5,
        additional_cost=10,
        bidding_strategies={},
        fuel_type="lignite",
        emission_factor=0.5,
        forecaster=forecaster,
    )


def test_flexable_eom(mock_market_config, power_plant):
    end = datetime(2023, 7, 1, 1)
    strategy = flexableEOM()
    mc = mock_market_config
    product_tuples = [(start, end, None)]
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert len(bids) == 2
    assert bids[0]["price"] == 40
    assert bids[0]["volume"] == 200
    assert bids[1]["price"] == 40
    assert bids[1]["volume"] == 800

    # start-up situation with ramping restriction
    power_plant.ramp_up = 400
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert len(bids) == 2
    assert bids[0]["price"] == 40
    assert bids[0]["volume"] == 200
    assert bids[1]["price"] == 40
    assert bids[1]["volume"] == 200

    # CHP with ramping restrictions, but without start-up
    power_plant.outputs["heat"][start] = 300
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert len(bids) == 2
    assert math.isclose(bids[0]["price"], (40 - 10 / 0.9))
    assert bids[0]["volume"] == 300
    assert bids[1]["price"] == 40
    assert bids[1]["volume"] == 100


def test_flexable_pos_reserve(mock_market_config, power_plant):
    end = datetime(2023, 7, 1, 4, 0, 0)
    strategy = flexablePosCRM()
    mc = mock_market_config
    mc.product_type = "energy_pos"
    product_tuples = [(start, end, None)]
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    assert bids[0]["price"] == 40
    assert bids[0]["volume"] == 1000

    mc.product_type = "capacity_pos"
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    assert bids[0]["price"] == 50
    assert bids[0]["volume"] == 1000

    # increase the marginal cost to ensure specific_revenue < 0
    power_plant.marginal_cost[start] = 60
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    assert bids[0]["price"] == 10
    assert bids[0]["volume"] == 1000


def test_flexable_neg_reserve(mock_market_config, power_plant):
    end = datetime(2023, 7, 1, 4, 0, 0)
    strategy = flexableNegCRM()
    mc = mock_market_config
    # Calculations for negative energy
    mc.product_type = "energy_neg"
    product_tuples = [(start, end, None)]
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert bids == []

    # change previous power to max_power/2 to enable ramp down
    power_plant.outputs["energy"][start] = power_plant.max_power / 2
    product_tuples = [
        (start + pd.Timedelta(hours=1), end + pd.Timedelta(hours=1), None)
    ]

    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    assert bids[0]["price"] == -40
    assert bids[0]["volume"] == 300

    # Calculations for negative capacity
    mc.product_type = "capacity_neg"
    product_tuples = [(start, end, None)]
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert bids == []

    # change previous power to max_power/2 to enable ramp down
    power_plant.outputs["energy"][start] = power_plant.max_power / 2
    product_tuples = [
        (start + pd.Timedelta(hours=1), end + pd.Timedelta(hours=1), None)
    ]

    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    assert bids[0]["price"] == 0
    assert bids[0]["volume"] == 300

    # increase the marginal cost to ensure specific_revenue < 0
    power_plant.marginal_cost[start + pd.Timedelta(hours=1)] = 60

    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    assert math.isclose(bids[0]["price"], 50 / 3)
    assert bids[0]["volume"] == 300


if __name__ == "__main__":
    # run pytest and enable prints
    import pytest

    pytest.main(["-s", __file__])
