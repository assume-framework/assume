# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime

import pandas as pd
import pytest

try:
    from assume.reinforcement_learning.learning_role import LearningConfig
    from assume.strategies.learning_strategies import RLStrategy
except ImportError:
    pass

from assume.common.forecasts import NaiveForecast
from assume.units import PowerPlant

start = datetime(2023, 7, 1)
end = datetime(2023, 7, 2)


@pytest.fixture
def power_plant_mcp() -> PowerPlant:
    # Create a PowerPlant instance with some example parameters
    index = pd.date_range("2023-06-30 22:00:00", periods=48, freq="h")
    ff = NaiveForecast(index, availability=1, fuel_price=10, co2_price=10)
    learning_config: LearningConfig = {
        "observation_dimension": 50,
        "action_dimension": 2,
        "algorithm": "matd3",
        "learning_mode": True,
        "training_episodes": 3,
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
        bidding_strategies={"EOM": RLStrategy(**learning_config)},
        fuel_type="lignite",
        emission_factor=0.5,
        forecaster=ff,
    )


@pytest.fixture
def power_plant_lstm() -> PowerPlant:
    # Create a PowerPlant instance with some example parameters
    index = pd.date_range("2023-06-30 22:00:00", periods=48, freq="h")
    ff = NaiveForecast(index, availability=1, fuel_price=10, co2_price=10)
    learning_config: LearningConfig = {
        "observation_dimension": 50,
        "action_dimension": 2,
        "algorithm": "matd3",
        "actor_architecture": "lstm",
        "learning_mode": True,
        "training_episodes": 3,
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
        bidding_strategies={"EOM": RLStrategy(**learning_config)},
        fuel_type="lignite",
        emission_factor=0.5,
        forecaster=ff,
    )


@pytest.mark.require_learning
def test_learning_strategies(mock_market_config, power_plant_mcp):
    product_index = pd.date_range("2023-07-01", periods=1, freq="h")
    mc = mock_market_config
    mc.product_type = "energy_eom"
    product_tuples = [
        (start, start + pd.Timedelta(hours=1), None) for start in product_index
    ]

    strategy = power_plant_mcp.bidding_strategies["EOM"]
    bids = strategy.calculate_bids(power_plant_mcp, mc, product_tuples=product_tuples)

    assert len(bids) == 2
    assert bids[0]["volume"] == 200
    assert bids[1]["volume"] == 800

    for order in bids:
        order["accepted_price"] = 50
        order["accepted_volume"] = order["volume"]

    strategy.calculate_reward(power_plant_mcp, mc, orderbook=bids)
    reward = power_plant_mcp.outputs["reward"].loc[product_index]
    profit = power_plant_mcp.outputs["profit"].loc[product_index]
    regret = power_plant_mcp.outputs["regret"].loc[product_index]
    costs = power_plant_mcp.outputs["total_costs"].loc[product_index]

    assert reward[0] == 0.8
    assert profit[0] == 10000.0
    assert regret[0] == 2000.0
    assert costs[0] == 40000.0


@pytest.mark.require_learning
def test_lstm_learning_strategies(mock_market_config, power_plant_lstm):
    product_index = pd.date_range("2023-07-01", periods=1, freq="h")
    mc = mock_market_config
    mc.product_type = "energy_eom"
    product_tuples = [
        (start, start + pd.Timedelta(hours=1), None) for start in product_index
    ]

    strategy = power_plant_lstm.bidding_strategies["EOM"]
    bids = strategy.calculate_bids(power_plant_lstm, mc, product_tuples=product_tuples)

    assert len(bids) == 2
    assert bids[0]["volume"] == 200
    assert bids[1]["volume"] == 800

    for order in bids:
        order["accepted_price"] = 50
        order["accepted_volume"] = order["volume"]

    strategy.calculate_reward(power_plant_lstm, mc, orderbook=bids)
    reward = power_plant_lstm.outputs["reward"].loc[product_index]
    profit = power_plant_lstm.outputs["profit"].loc[product_index]
    regret = power_plant_lstm.outputs["regret"].loc[product_index]
    costs = power_plant_lstm.outputs["total_costs"].loc[product_index]

    assert reward[0] == 0.8
    assert profit[0] == 10000.0
    assert regret[0] == 2000.0
    assert costs[0] == 40000.0
