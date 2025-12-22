# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
from unittest.mock import patch

import pandas as pd
import pytest

from assume.common.base import LearningConfig
from assume.common.forecaster import UnitForecaster

try:
    import torch as th

    from assume.reinforcement_learning import Learning
    from assume.strategies.learning_strategies import StorageEnergyLearningStrategy
except ImportError:
    th = None

from assume.common.market_objects import MarketConfig
from assume.units import Storage


@pytest.fixture
def storage_unit() -> Storage:
    """
    Fixture to create a Storage unit instance with example parameters.
    """
    # Define the learning configuration for the StorageRLStrategy
    config = {
        "obs_dim": 50,
        "act_dim": 2,
        "unit_id": "test_storage",
        "max_demand": 1000,
        "learning_config": LearningConfig(
            algorithm="matd3",
            learning_mode=True,
            training_episodes=3,
            max_bid_price=100,
        ),
    }

    index = pd.date_range("2023-06-30 22:00:00", periods=48, freq="h")
    ff = UnitForecaster(index, market_prices={"test_market": 50})
    learning_role = Learning(config["learning_config"], index[0], index[-1])
    return Storage(
        id="test_storage",
        unit_operator="test_operator",
        technology="storage",
        bidding_strategies={
            "test_market": StorageEnergyLearningStrategy(
                learning_role=learning_role, **config
            )
        },
        max_power_charge=-500,  # Negative for charging
        max_power_discharge=500,
        capacity=1000,
        min_soc=0,
        max_soc=1,
        initial_soc=0.5,
        efficiency_charge=0.9,
        efficiency_discharge=0.9,
        additional_cost_charge=5,
        additional_cost_discharge=5,
        forecaster=ff,
    )


@pytest.fixture
def mock_market_config():
    """
    Fixture to create a mock MarketConfig instance.
    """
    mc = MarketConfig()
    mc.market_id = "test_market"
    mc.product_type = "energy"
    return mc


@pytest.mark.require_learning
def test_storage_rl_strategy_sell_bid(mock_market_config, storage_unit):
    """
    Test the StorageEnergyLearningStrategy for a 'sell' bid action.
    """

    # Define the product index and tuples
    product_index = pd.date_range("2023-07-01", periods=1, freq="h")
    mc = mock_market_config
    product_tuples = [
        (start, start + pd.Timedelta(hours=1), None) for start in product_index
    ]

    # get the strategy
    strategy = storage_unit.bidding_strategies["test_market"]

    # Define the 'sell' action: [0.2, 0.5] -> price=20, direction='sell'
    sell_action = [0.2, 0.5]

    # Mock the get_actions method to return the sell action
    with patch.object(
        StorageEnergyLearningStrategy,
        "get_actions",
        return_value=(th.tensor(sell_action), th.tensor(0.0)),
    ):
        # Mock the calculate_marginal_cost method to return a fixed marginal cost
        with patch.object(Storage, "calculate_marginal_cost", return_value=10.0):
            # Calculate bids using the strategy
            bids = strategy.calculate_bids(  # TODO
                storage_unit, mc, product_tuples=product_tuples
            )

            # Assert that exactly one bid is generated
            assert len(bids) == 1, f"Expected 1 bid, got {len(bids)}"

            # Extract the bid
            bid = bids[0]

            # Assert the bid price is correctly scaled
            expected_bid_price = sell_action[0] * strategy.max_bid_price  # 20.0
            assert bid["price"] == expected_bid_price, (
                f"Expected bid price {expected_bid_price}, got {bid['price']}"
            )

            # Assert the bid direction is 'sell' and volume is max_power_discharge
            expected_volume = (
                storage_unit.max_power_discharge * storage_unit.efficiency_discharge
            )  # 500
            assert bid["volume"] == expected_volume, (
                f"Expected bid volume {expected_volume}, got {bid['volume']}"
            )

            # Simulate bid acceptance by setting accepted_price and accepted_volume
            bid["accepted_price"] = expected_bid_price  # 20.0
            bid["accepted_volume"] = expected_volume  # 500

            # Calculate rewards based on the accepted bids
            strategy.calculate_reward(storage_unit, mc, orderbook=bids)

            # Fetch reward, profit, costs from learning_role cache
            learning_role = strategy.learning_role
            reward_cache = learning_role.all_rewards
            profit_cache = learning_role.all_profits

            # Use the last timestamp
            last_ts = sorted(reward_cache.keys())[-1]
            unit_id = (
                storage_unit.id
                if storage_unit.id in reward_cache[last_ts]
                else list(reward_cache[last_ts].keys())[0]
            )

            reward = reward_cache[last_ts][unit_id][0]
            profit = profit_cache[last_ts][unit_id][0]
            costs = storage_unit.outputs["total_costs"].loc[product_index]

            # Calculate expected values
            duration_hours = 1  # Since the product tuple is 1 hour
            expected_profit = (
                expected_bid_price * bid["volume"] * duration_hours
            )  # 20 * 500 * 1 = 10000
            expected_costs = (
                10.0 * bid["volume"] * duration_hours
            )  # 10 * 500 * 1 = 5000
            scaling_factor = 1 / (
                storage_unit.max_power_discharge * strategy.max_bid_price
            )  # 1 / (500*100) = 0.00002
            expected_reward = (
                expected_profit - expected_costs
            ) * scaling_factor  # (10000 - 5000) * 0.0002 = 1.0

            # Assert the calculated reward
            assert reward == expected_reward, (
                f"Expected reward {expected_reward}, got {reward}"
            )

            # Assert the calculated profit
            assert profit == expected_profit - expected_costs, (
                f"Expected profit {expected_profit}, got {profit}"
            )

            # Assert the calculated costs
            assert costs[0] == expected_costs, (
                f"Expected costs {expected_costs}, got {costs[0]}"
            )


@pytest.mark.require_learning
def test_storage_rl_strategy_buy_bid(mock_market_config, storage_unit):
    """
    Test the StorageEnergyLearningStrategy for a 'buy' bid action.
    """
    # Define the product index and tuples
    product_index = pd.date_range("2023-07-01", periods=1, freq="h")
    mc = mock_market_config
    product_tuples = [
        (start, start + pd.Timedelta(hours=1), None) for start in product_index
    ]

    # Instantiate the StorageEnergyLearningStrategy
    strategy = storage_unit.bidding_strategies["test_market"]

    # Define the 'buy' action: [-0.3] -> price=30, direction='buy'
    buy_action = [-0.3]

    # Mock the get_actions method to return the buy action
    with patch.object(
        StorageEnergyLearningStrategy,
        "get_actions",
        return_value=(th.tensor(buy_action), th.tensor(0.0)),
    ):
        # Mock the calculate_marginal_cost method to return a fixed marginal cost
        with patch.object(Storage, "calculate_marginal_cost", return_value=15.0):
            # Calculate bids using the strategy
            bids = strategy.calculate_bids(
                storage_unit, mc, product_tuples=product_tuples
            )

            # Assert that exactly one bid is generated
            assert len(bids) == 1, f"Expected 1 bid, got {len(bids)}"

            # Extract the bid
            bid = bids[0]

            # Assert the bid price is correctly scaled
            expected_bid_price = abs(buy_action[0]) * strategy.max_bid_price  # 30.0
            assert math.isclose(bid["price"], expected_bid_price, abs_tol=1e3), (
                f"Expected bid price {expected_bid_price}, got {bid['price']}"
            )

            # Assert the bid direction is 'buy' and volume is abs(max_power_charge)
            expected_volume = storage_unit.max_power_charge  # 500
            assert bid["volume"] == expected_volume, (
                f"Expected bid volume {expected_volume}, got {bid['volume']}"
            )

            # Simulate bid acceptance by setting accepted_price and accepted_volume
            bid["accepted_price"] = expected_bid_price  # 30.0
            bid["accepted_volume"] = expected_volume  # 500

            # Calculate rewards based on the accepted bids
            strategy.calculate_reward(storage_unit, mc, orderbook=bids)

            # Fetch reward, profit, costs from learning_role cache
            learning_role = strategy.learning_role
            reward_cache = learning_role.all_rewards
            profit_cache = learning_role.all_profits

            # Use the last timestamp
            last_ts = sorted(reward_cache.keys())[-1]
            unit_id = (
                storage_unit.id
                if storage_unit.id in reward_cache[last_ts]
                else list(reward_cache[last_ts].keys())[0]
            )

            reward = reward_cache[last_ts][unit_id][0]
            profit = profit_cache[last_ts][unit_id][0]
            costs = storage_unit.outputs["total_costs"].loc[product_index]

            # Calculate expected values
            duration_hours = 1  # Since the product tuple is 1 hour
            expected_profit = (
                expected_bid_price * bid["volume"] * duration_hours
            )  # 30 * 500 * 1 = 15000
            expected_costs = (
                15.0 * abs(bid["volume"]) * duration_hours
            )  # 15 * 500 * 1 = 7500
            scaling_factor = 1 / (
                storage_unit.max_power_discharge * strategy.max_bid_price
            )  # 1 / (500*100) = 0.00002
            expected_reward = (
                expected_profit - expected_costs
            ) * scaling_factor  # (15000 - 7500) * 0.0002 = 1.5

            # Assert the calculated reward
            assert reward == expected_reward, (
                f"Expected reward {expected_reward}, got {reward}"
            )

            # Assert the calculated profit
            assert profit == expected_profit - expected_costs, (
                f"Expected profit {expected_profit}, got {profit}"
            )

            # Assert the calculated costs
            assert costs[0] == expected_costs, (
                f"Expected costs {expected_costs}, got {costs[0]}"
            )


@pytest.mark.require_learning
def test_storage_rl_strategy_soc_and_cost_stored_energy(
    mock_market_config, storage_unit
):
    """
    Test the StorageEnergyLearningStrategy if unique observations are created as expected.
    """
    # Define the product index and tuples
    product_index = pd.date_range("2023-07-01", periods=3, freq="h")
    mc = mock_market_config
    product_tuples = [
        (start, start + pd.Timedelta(hours=1), None) for start in product_index
    ]

    # Instantiate the StorageEnergyLearningStrategy
    strategy = storage_unit.bidding_strategies["test_market"]

    # Define sequence of actions over 3 hours: [charge, sell, sell]
    # Format: [normalized_price (-1, 1), direction indicated by sign (negative: buy bid, positive: sell bid)]
    actions = [
        [-0.3],  # Charge at price 30
        [0.6],  # Sell at price 60
        [0.8],  # Sell at price 80
    ]

    # Patch get_actions to return one action at a time
    get_actions_patch = patch.object(StorageEnergyLearningStrategy, "get_actions")
    calc_cost_patch = patch.object(Storage, "calculate_marginal_cost")

    with get_actions_patch as mock_get_actions, calc_cost_patch as mock_cost:
        # Set up side effects for each call
        mock_get_actions.side_effect = [(th.tensor(a), th.tensor(0.0)) for a in actions]
        mock_cost.side_effect = [5.0, 15.0, 25.0]

        all_bids = []

        for i, product_tuple in enumerate(product_tuples):
            # Call calculate_bids for a single product_tuple
            bids = strategy.calculate_bids(
                storage_unit, mock_market_config, product_tuples=[product_tuple]
            )

            # Only one bid per call
            assert len(bids) == 1
            bid = bids[0]

            # Simulate market clearing
            bid["accepted_price"] = bid["price"].item()
            bid["accepted_volume"] = bid["volume"]

            all_bids.append(bid)

            # Set dispatch plan
            storage_unit.set_dispatch_plan(mc, orderbook=[bid])

            # Apply profit and reward update for this step
            strategy.calculate_reward(storage_unit, mock_market_config, orderbook=[bid])

        # Get resulting energy costs
        cost_stored_energy = storage_unit.outputs["cost_stored_energy"].loc[
            product_index.union([product_index[-1] + product_index.freq])
        ]

        soc = storage_unit.outputs["soc"].loc[
            product_index.union([product_index[-1] + product_index.freq])
        ]
        expected_soc_t0 = 0.5  # initial soc
        assert math.isclose(soc[0], expected_soc_t0, rel_tol=1e-3), (
            f"Expected SoC at t=0 to be {expected_soc_t0}, got {soc[0]}"
        )
        # Initial state: 500 MWh at default energy costs of 0 €/MWh
        # 1. Charge 500 MWh at 30 €/MWh: cost_stored_energy_t1 = (0 €/MWh * 500 MWh - ((30 €/MWh + 5 €/MWh) * - 500 MW * 1h)) / 950 MWh = 18.41 €/MWh
        expected_soc_t1 = 0.5 + (500 * 0.9 / 1000)  # 0.95
        assert math.isclose(soc[1], expected_soc_t1, rel_tol=1e-3), (
            f"Expected SoC at t=1 to be {expected_soc_t1}, got {soc[1]}"
        )
        expected_cost_t1 = (500 * 35) / 950
        assert math.isclose(cost_stored_energy[1], expected_cost_t1, rel_tol=1e-3), (
            f"Expected energy cost at t=1 to be {expected_cost_t1}, got {cost_stored_energy[1]}"
        )
        # 2. Discharge 500 MWh at 60 €/MWh: cost_stored_energy_t2 = 18.41 €/MWh unchanged
        expected_soc_t2 = 0.95 - (500 / 0.9 / 1000)  # 0.3944
        assert math.isclose(soc[2], expected_soc_t2, rel_tol=1e-3), (
            f"Expected SoC at t=2 to be {expected_soc_t2}, got {soc[2]}"
        )
        expected_cost_t2 = expected_cost_t1
        assert math.isclose(cost_stored_energy[2], expected_cost_t2, rel_tol=1e-3), (
            f"Expected energy cost at t=2 to be {expected_cost_t2}, got {cost_stored_energy[2]}"
        )
        # 3. Discharge remaining 355 MWh at 80 €/Mwh: SoC < 1 --> cost_stored_energy_t3 = 0 €/MWh
        expected_soc_t3 = 0.3944 - (355 / 0.9 / 1000)  # 0
        # use abs_tol here as values are close to zero
        assert math.isclose(soc[3], expected_soc_t3, abs_tol=0.1), (
            f"Expected SoC at t=3 to be {expected_soc_t3}, got {soc[3]}"
        )
        expected_cost_t3 = 0
        assert math.isclose(cost_stored_energy[3], expected_cost_t3, rel_tol=1e-3), (
            f"Expected energy cost at t=3 to be {expected_cost_t3}, got {cost_stored_energy[3]}"
        )

        print("Energy cost series:\n", cost_stored_energy)
