# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime

import pandas as pd
import pytest

from assume.common.base import LearningConfig
from assume.common.forecaster import PowerplantForecaster

try:
    from assume.reinforcement_learning import Learning
    from assume.strategies.learning_strategies import (
        EnergyLearningSingleBidStrategy,
        EnergyLearningStrategy,
    )

except ImportError:
    EnergyLearningStrategy = None
    EnergyLearningSingleBidStrategy = None

from assume.units import PowerPlant

start = datetime(2023, 7, 1)
end = datetime(2023, 7, 2)


@pytest.fixture
def power_plant() -> PowerPlant:
    # Create a PowerPlant instance with some example parameters
    index = pd.date_range("2023-06-30 22:00:00", periods=48, freq="h")
    ff = PowerplantForecaster(
        index,
        fuel_prices={"lignite": 10, "co2": 10},
        residual_load={"EOM": 0},
    )
    config = {
        "unit_id": "test_pp",
        "learning_config": LearningConfig(
            algorithm="matd3",
            learning_mode=True,
            training_episodes=3,
        ),
    }
    learning_role = Learning(config["learning_config"], start, end)

    return PowerPlant(
        id="test_pp",
        unit_operator="test_operator",
        technology="coal",
        index=ff.index,
        max_power=1000,
        min_power=200,
        efficiency=0.5,
        additional_cost=10,
        bidding_strategies={
            "EOM": EnergyLearningStrategy(learning_role=learning_role, **config)
        },
        fuel_type="lignite",
        emission_factor=0.5,
        forecaster=ff,
    )


@pytest.mark.require_learning
@pytest.mark.parametrize(
    "strategy_class, obs_dim, act_dim, unique_obs_dim, actor_architecture, expected_bid_count, expected_volumes",
    [
        (EnergyLearningStrategy, 38, 2, 2, "mlp", 2, [200, 800]),
        (EnergyLearningStrategy, 38, 2, 2, "lstm", 2, [200, 800]),
        (EnergyLearningSingleBidStrategy, 74, 1, 2, "mlp", 1, [1000]),
    ],
)
def test_learning_strategies_parametrized(
    mock_market_config,
    power_plant,
    strategy_class,
    obs_dim,
    act_dim,
    unique_obs_dim,
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
    # Build LearningConfig dynamically
    config = {
        "unit_id": power_plant.id,
        "learning_config": LearningConfig(
            algorithm="matd3",
            actor_architecture=actor_architecture,
            learning_mode=True,
            training_episodes=3,
        ),
    }

    learning_role = Learning(config["learning_config"], start, end)
    # Override the strategy
    power_plant.bidding_strategies[mc.market_id] = strategy_class(
        learning_role=learning_role, **config
    )
    strategy = power_plant.bidding_strategies[mc.market_id]

    # Check if observation dimension is set accordingly and follows current default structure
    first_observation = strategy.create_observation(
        power_plant,
        mc.market_id,
        product_index[0],
        product_index[0] + pd.Timedelta(hours=1),
    )
    assert len(first_observation) == obs_dim
    assert (
        strategy.unique_obs_dim + strategy.foresight * strategy.num_timeseries_obs_dim
        == obs_dim
    )

    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)

    assert len(bids) == expected_bid_count
    for bid, expected_volume in zip(bids, expected_volumes):
        assert bid["volume"] == expected_volume

    for order in bids:
        order["accepted_price"] = 50
        order["accepted_volume"] = order["volume"]

    strategy.calculate_reward(power_plant, mc, orderbook=bids)

    # Fetch reward, profit, regret from learning_role cache instead of outputs
    # Get the latest timestamp used for reward cache
    learning_role = strategy.learning_role
    reward_cache = learning_role.all_rewards
    profit_cache = learning_role.all_profits
    regret_cache = learning_role.all_regrets

    # Use the last timestamp (should be the one just written)
    last_ts = sorted(reward_cache.keys())[-1]
    unit_id = (
        power_plant.id
        if power_plant.id in reward_cache[last_ts]
        else list(reward_cache[last_ts].keys())[0]
    )

    reward = reward_cache[last_ts][unit_id][0]
    profit = profit_cache[last_ts][unit_id][0]
    regret = regret_cache[last_ts][unit_id][0]
    costs = power_plant.outputs["total_costs"].loc[product_index]

    assert reward == 0.1
    assert profit == 10000.0
    assert regret == 0.0
    assert costs[0] == 40000.0  # Assumes hot_start_cost = 20000 by default
