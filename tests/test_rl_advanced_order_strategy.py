# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import random as rd
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import torch as th

try:
    from assume.reinforcement_learning.learning_role import LearningConfig
    from assume.reinforcement_learning.learning_utils import Actor, CriticTD3
except ImportError:
    pass

from assume.common.forecasts import NaiveForecast
from assume.strategies.learning_advanced_orders import RLAdvancedOrderStrategy
from assume.units import PowerPlant

np.random.seed(0)
rd.seed(0)
th.manual_seed(0)
th.use_deterministic_algorithms(True)

start = datetime(2023, 7, 1)
end = datetime(2023, 7, 2)


@pytest.fixture
def power_plant() -> PowerPlant:
    # Create a PowerPlant instance with some example parameters
    index = pd.date_range("2023-07-01", periods=48, freq="H")
    ff = NaiveForecast(index, availability=1, fuel_price=10, co2_price=10)
    return PowerPlant(
        id="test_pp",
        unit_operator="test_operator",
        technology="coal",
        index=index,
        max_power=1000,
        min_power=200,
        efficiency=0.5,
        fixed_cost=10,
        bidding_strategies={},
        fuel_type="lignite",
        emission_factor=0.5,
        forecaster=ff,
    )


@pytest.mark.require_learning
def test_learning_advanced_orders(mock_market_config, power_plant):
    learning_config: LearningConfig = {
        "observation_dimension": 97,
        "action_dimension": 2,
        "algorithm": "matd3",
        "learning_mode": True,
        "training_episodes": 3,
        "order_types": ["SB", "BB", "LB"],
        "unit_id": "test_pp",
    }

    product_index = pd.date_range("2023-07-01", periods=24, freq="H")
    mc = mock_market_config
    mc.product_type = "energy_eom"
    product_tuples = [
        (start, start + pd.Timedelta(hours=1), None) for start in product_index
    ]

    learning_config["order_types"] = ["SB"]
    strategy = RLAdvancedOrderStrategy(**learning_config)
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)

    assert len(bids) == 48
    assert bids[0]["bid_type"] == "SB"
    assert bids[0]["volume"] == 200

    learning_config["order_types"] = ["SB", "BB"]
    strategy = RLAdvancedOrderStrategy(**learning_config)
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)

    assert len(bids) == 25
    assert bids[0]["bid_type"] == "SB"
    assert bids[0]["volume"] == 800

    assert bids[-1]["bid_type"] == "BB"
    assert bids[-1]["volume"][product_tuples[0][0]] == 200
    assert bids[0]["price"] >= bids[-1]["price"]

    learning_config["order_types"] = ["SB", "LB"]
    strategy = RLAdvancedOrderStrategy(**learning_config)
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)

    assert len(bids) == 48
    assert bids[0]["bid_type"] == "SB"
    assert bids[0]["volume"] == 200

    assert bids[1]["bid_type"] == "LB"
    assert bids[1]["volume"][product_tuples[0][0]] == 800
    assert bids[0]["price"] <= bids[1]["price"]

    learning_config["order_types"] = ["SB", "BB", "LB"]
    strategy = RLAdvancedOrderStrategy(**learning_config)
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)

    assert len(bids) == 25
    assert bids[0]["bid_type"] == "LB"
    assert bids[0]["volume"][product_tuples[0][0]] == 800

    assert bids[-1]["bid_type"] == "BB"
    assert bids[-1]["volume"][product_tuples[0][0]] == 200
    assert bids[0]["price"] >= bids[-1]["price"]
