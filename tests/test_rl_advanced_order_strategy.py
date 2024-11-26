# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime

import pandas as pd
import pytest

try:
    from assume.reinforcement_learning.learning_role import LearningConfig
    from assume.strategies.learning_advanced_orders import RLAdvancedOrderStrategy
except ImportError:
    pass

from assume.common.forecasts import NaiveForecast
from assume.units import PowerPlant

start = datetime(2023, 7, 1)
end = datetime(2023, 7, 2)


@pytest.fixture
def power_plant() -> PowerPlant:
    # Create a PowerPlant instance with some example parameters
    index = pd.date_range("2023-07-01", periods=48, freq="h")
    ff = NaiveForecast(index, availability=1, fuel_price=10, co2_price=10)
    learning_config: LearningConfig = {
        "algorithm": "matd3",
        "learning_mode": True,
        "training_episodes": 3,
        "order_types": ["SB", "BB", "LB"],
        "unit_id": "test_pp",
    }

    return PowerPlant(
        id="test_pp",
        unit_operator="test_operator",
        technology="coal",
        index=ff.index,
        max_power=1000,
        min_power=200,
        efficiency=0.5,
        additional_cost=10,
        bidding_strategies={"EOM": RLAdvancedOrderStrategy(**learning_config)},
        fuel_type="lignite",
        emission_factor=0.5,
        forecaster=ff,
    )


@pytest.mark.require_learning
def test_learning_advanced_orders(mock_market_config, power_plant):
    product_index = pd.date_range("2023-07-01", periods=24, freq="h")
    mc = mock_market_config
    mc.product_type = "energy_eom"
    product_tuples = [
        (start, start + pd.Timedelta(hours=1), None) for start in product_index
    ]

    strategy = power_plant.bidding_strategies["EOM"]

    strategy.order_types = ["SB"]
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)

    assert len(bids) == 48
    assert bids[0]["bid_type"] == "SB"
    assert bids[0]["volume"] == 200
    assert bids[0]["bid_id"] == "test_pp_SB_1"
    assert bids[-1]["bid_id"] == "test_pp_SB_48"

    strategy.order_types = ["SB", "BB"]
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)

    assert len(bids) == 25
    assert bids[0]["bid_type"] == "SB"
    assert bids[0]["volume"] == 800
    assert bids[0]["bid_id"] == "test_pp_SB_1"
    assert bids[-1]["bid_id"] == "test_pp_block"

    assert bids[-1]["bid_type"] == "BB"
    assert bids[-1]["volume"][product_tuples[0][0]] == 200
    assert bids[0]["price"] >= bids[-1]["price"]

    strategy.order_types = ["SB", "LB"]
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)

    assert len(bids) == 48
    assert bids[0]["bid_type"] == "SB"
    assert bids[0]["volume"] == 200

    assert bids[1]["bid_type"] == "LB"
    assert bids[1]["volume"][product_tuples[0][0]] == 800
    assert bids[0]["price"] <= bids[1]["price"]

    assert bids[0]["bid_id"] == "test_pp_SB_1"
    assert bids[-1]["bid_id"] == "test_pp_LB_48"

    strategy.order_types = ["SB", "BB", "LB"]
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)

    assert len(bids) == 25
    assert bids[0]["bid_type"] == "LB"
    assert bids[0]["volume"][product_tuples[0][0]] == 800

    assert bids[-1]["bid_type"] == "BB"
    assert bids[-1]["volume"][product_tuples[0][0]] == 200
    assert bids[0]["price"] >= bids[-1]["price"]

    assert bids[0]["bid_id"] == "test_pp_LB_1"
    assert bids[-1]["bid_id"] == "test_pp_block"
