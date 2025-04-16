# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime

import pandas as pd
import pytest

try:
    from assume.reinforcement_learning.learning_role import LearningConfig
    from assume.strategies.learning_strategies import RLStrategy, RLStrategySingleBid
except ImportError:
    pass

from assume.common.forecasts import NaiveForecast
from assume.units import PowerPlant

start = datetime(2023, 7, 1)
end = datetime(2023, 7, 2)


@pytest.fixture
def power_plant() -> PowerPlant:
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


@pytest.mark.require_learning
@pytest.mark.parametrize(
    "strategy_class, obs_dim, act_dim, actor_architecture, expected_bid_count, expected_volumes",
    [
        (RLStrategy, 50, 2, "mlp", 2, [200, 800]),
        (RLStrategy, 50, 2, "lstm", 2, [200, 800]),
        (RLStrategySingleBid, 74, 1, "mlp", 1, [1000]),
    ],
)
def test_learning_strategies_parametrized(
    mock_market_config,
    power_plant,
    strategy_class,
    obs_dim,
    act_dim,
    actor_architecture,
    expected_bid_count,
    expected_volumes,
):
    product_index = pd.date_range("2023-07-01", periods=1, freq="h")
    mc = mock_market_config
    mc.product_type = "energy_eom"
    product_tuples = [
        (start, start + pd.Timedelta(hours=1), None) for start in product_index
    ]

    # Build learning config dynamically
    learning_config: LearningConfig = {
        "observation_dimension": obs_dim,
        "action_dimension": act_dim,
        "algorithm": "matd3",
        "learning_mode": True,
        "training_episodes": 3,
        "unit_id": power_plant.id,
    }
    if actor_architecture != "mlp":
        learning_config["actor_architecture"] = actor_architecture

    # Override the strategy
    power_plant.bidding_strategies["EOM"] = strategy_class(**learning_config)

    strategy = power_plant.bidding_strategies["EOM"]
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)

    assert len(bids) == expected_bid_count
    for bid, expected_volume in zip(bids, expected_volumes):
        assert bid["volume"] == expected_volume

    for order in bids:
        order["accepted_price"] = 50
        order["accepted_volume"] = order["volume"]

    strategy.calculate_reward(power_plant, mc, orderbook=bids)
    reward = power_plant.outputs["reward"].loc[product_index]
    profit = power_plant.outputs["profit"].loc[product_index]
    regret = power_plant.outputs["regret"].loc[product_index]
    costs = power_plant.outputs["total_costs"].loc[product_index]

    assert reward[0] == 0.01
    assert profit[0] == 1000.0
    assert regret[0] == 0.0
    assert costs[0] == 40000.0  # Assumes hot_start_cost = 20000 by default
