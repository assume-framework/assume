# tests/test_drl_storage_strategy.py

# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
from unittest.mock import patch

import pandas as pd
import pytest

try:
    import torch as th

    from assume.reinforcement_learning.learning_role import LearningConfig
    from assume.strategies.learning_strategies import StorageRLStrategy
except ImportError:
    th = None

from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig
from assume.units import Storage


@pytest.fixture
def storage_unit() -> Storage:
    """
    Fixture to create a Storage unit instance with example parameters.
    """
    # Define the learning configuration for the StorageRLStrategy
    learning_config: LearningConfig = {
        "observation_dimension": 50,
        "action_dimension": 2,
        "algorithm": "matd3",
        "learning_mode": True,
        "training_episodes": 3,
        "unit_id": "test_storage",
        "max_bid_price": 100,
        "max_demand": 1000,
    }

    index = pd.date_range("2023-06-30 22:00:00", periods=48, freq="h")
    ff = NaiveForecast(index, availability=1, fuel_price=10, co2_price=10)
    return Storage(
        id="test_storage",
        unit_operator="test_operator",
        technology="storage",
        bidding_strategies={"EOM": StorageRLStrategy(**learning_config)},
        max_power_charge=500,  # Negative for charging
        max_power_discharge=500,
        max_soc=1000,
        min_soc=0,
        initial_soc=500,
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
    mc.product_type = "energy_eom"
    return mc


@pytest.mark.require_learning
def test_storage_rl_strategy_sell_bid(mock_market_config, storage_unit):
    """
    Test the StorageRLStrategy for a 'sell' bid action.
    """

    # Define the product index and tuples
    product_index = pd.date_range("2023-07-01", periods=1, freq="h")
    mc = mock_market_config
    product_tuples = [
        (start, start + pd.Timedelta(hours=1), None) for start in product_index
    ]

    # get the strategy
    strategy = storage_unit.bidding_strategies["EOM"]

    # Define the 'sell' action: [0.2, 0.5] -> price=20, direction='sell'
    sell_action = [0.2, 0.5]

    # Mock the get_actions method to return the sell action
    with patch.object(
        StorageRLStrategy,
        "get_actions",
        return_value=(th.tensor(sell_action), th.tensor(0.0)),
    ):
        # Mock the calculate_marginal_cost method to return a fixed marginal cost
        with patch.object(Storage, "calculate_marginal_cost", return_value=10.0):
            # Calculate bids using the strategy
            bids = strategy.calculate_bids(
                storage_unit, mc, product_tuples=product_tuples
            )

            # Assert that exactly one bid is generated
            assert len(bids) == 1, f"Expected 1 bid, got {len(bids)}"

            # Extract the bid
            bid = bids[0]

            # Assert the bid price is correctly scaled
            expected_bid_price = sell_action[0] * strategy.max_bid_price  # 20.0
            assert (
                bid["price"] == expected_bid_price
            ), f"Expected bid price {expected_bid_price}, got {bid['price']}"

            # Assert the bid direction is 'sell' and volume is max_power_discharge + 1e-6
            expected_volume = (
                storage_unit.max_power_discharge * storage_unit.efficiency_discharge
                + 1e-6
            )  # 500 + 1e-6
            assert (
                bid["volume"] == expected_volume
            ), f"Expected bid volume {expected_volume}, got {bid['volume']}"

            # Simulate bid acceptance by setting accepted_price and accepted_volume
            bid["accepted_price"] = expected_bid_price  # 20.0
            bid["accepted_volume"] = expected_volume  # 500 + 1e-6

            # Calculate rewards based on the accepted bids
            strategy.calculate_reward(storage_unit, mc, orderbook=bids)

            # Extract outputs
            reward = storage_unit.outputs["reward"].loc[product_index]
            profit = storage_unit.outputs["profit"].loc[product_index]
            costs = storage_unit.outputs["total_costs"].loc[product_index]

            # Calculate expected values
            duration_hours = 1  # Since the product tuple is 1 hour
            expected_profit = (
                expected_bid_price * bid["volume"] * duration_hours
            )  # 20 * 500 * 1 = 10000
            expected_costs = (
                10.0 * bid["volume"] * duration_hours
            )  # 10 * 500 * 1 = 5000
            scaling_factor = (
                0.1 / storage_unit.max_power_discharge
            )  # 0.1 / 500 = 0.0002
            expected_reward = (
                expected_profit - expected_costs
            ) * scaling_factor  # (10000 - 5000) * 0.0002 = 1.0

            # Assert the calculated reward
            assert (
                reward[0] == expected_reward
            ), f"Expected reward {expected_reward}, got {reward[0]}"

            # Assert the calculated profit
            assert (
                profit[0] == expected_profit - expected_costs
            ), f"Expected profit {expected_profit}, got {profit[0]}"

            # Assert the calculated costs
            assert (
                costs[0] == expected_costs
            ), f"Expected costs {expected_costs}, got {costs[0]}"


@pytest.mark.require_learning
def test_storage_rl_strategy_buy_bid(mock_market_config, storage_unit):
    """
    Test the StorageRLStrategy for a 'buy' bid action.
    """
    # Define the product index and tuples
    product_index = pd.date_range("2023-07-01", periods=1, freq="h")
    mc = mock_market_config
    product_tuples = [
        (start, start + pd.Timedelta(hours=1), None) for start in product_index
    ]

    # Instantiate the StorageRLStrategy
    strategy = storage_unit.bidding_strategies["EOM"]

    # Define the 'buy' action: [0.3, -0.5] -> price=30, direction='buy'
    buy_action = [0.3, -0.5]

    # Mock the get_actions method to return the buy action
    with patch.object(
        StorageRLStrategy,
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
            expected_bid_price = buy_action[0] * strategy.max_bid_price  # 30.0
            assert math.isclose(
                bid["price"], expected_bid_price, abs_tol=1e3
            ), f"Expected bid price {expected_bid_price}, got {bid['price']}"

            # Assert the bid direction is 'buy' and volume is abs(max_power_charge) + 1e-6
            expected_volume = storage_unit.max_power_charge + 1e-6  # 500 + 1e-6
            assert (
                bid["volume"] == expected_volume
            ), f"Expected bid volume {expected_volume}, got {bid['volume']}"

            # Simulate bid acceptance by setting accepted_price and accepted_volume
            bid["accepted_price"] = expected_bid_price  # 30.0
            bid["accepted_volume"] = expected_volume  # 500 + 1e-6

            # Calculate rewards based on the accepted bids
            strategy.calculate_reward(storage_unit, mc, orderbook=bids)

            # Extract outputs
            reward = storage_unit.outputs["reward"].loc[product_index]
            profit = storage_unit.outputs["profit"].loc[product_index]
            costs = storage_unit.outputs["total_costs"].loc[product_index]

            # Calculate expected values
            duration_hours = 1  # Since the product tuple is 1 hour
            expected_profit = (
                expected_bid_price * bid["volume"] * duration_hours
            )  # 30 * 500 * 1 = 15000
            expected_costs = (
                15.0 * abs(bid["volume"]) * duration_hours
            )  # 15 * 500 * 1 = 7500
            scaling_factor = (
                0.1 / storage_unit.max_power_discharge
            )  # 0.1 / 500 = 0.0002
            expected_reward = (
                expected_profit - expected_costs
            ) * scaling_factor  # (15000 - 7500) * 0.0002 = 1.5

            # Assert the calculated reward
            assert (
                reward[0] == expected_reward
            ), f"Expected reward {expected_reward}, got {reward[0]}"

            # Assert the calculated profit
            assert (
                profit[0] == expected_profit - expected_costs
            ), f"Expected profit {expected_profit}, got {profit[0]}"

            # Assert the calculated costs
            assert (
                costs[0] == expected_costs
            ), f"Expected costs {expected_costs}, got {costs[0]}"


@pytest.mark.require_learning
def test_storage_rl_strategy_ignore_bid(mock_market_config, storage_unit):
    """
    Test the StorageRLStrategy for an 'ignore' bid action.
    """

    # Define the product index and tuples
    product_index = pd.date_range("2023-07-01", periods=1, freq="h")
    mc = mock_market_config
    product_tuples = [
        (start, start + pd.Timedelta(hours=1), None) for start in product_index
    ]

    # Instantiate the StorageRLStrategy
    strategy = storage_unit.bidding_strategies["EOM"]

    # Define the 'ignore' action: [0.0, 0.0] -> price=0, direction='ignore'
    ignore_action = [0.0, 0.0]

    # Mock the get_actions method to return the ignore action
    with patch.object(
        StorageRLStrategy,
        "get_actions",
        return_value=(th.tensor(ignore_action), th.tensor(0.0)),
    ):
        # Mock the calculate_marginal_cost method to return a fixed marginal cost
        with patch.object(Storage, "calculate_marginal_cost", return_value=0.0):
            # Calculate bids using the strategy
            bids = strategy.calculate_bids(
                storage_unit, mc, product_tuples=product_tuples
            )

            # Assert that exactly one bid is generated
            assert len(bids) == 1, f"Expected 1 bid, got {len(bids)}"

            # Extract the bid
            bid = bids[0]

            # Assert the bid price is correctly scaled
            expected_bid_price = ignore_action[0] * strategy.max_bid_price  # 0.0
            assert (
                bid["price"] == expected_bid_price
            ), f"Expected bid price {expected_bid_price}, got {bid['price']}"

            # Assert the bid direction is 'ignore' and volume is 1e-6
            expected_volume = 1e-6  # Volume for ignore
            assert (
                bid["volume"] == expected_volume
            ), f"Expected bid volume {expected_volume}, got {bid['volume']}"

            # Simulate bid acceptance by setting accepted_price and accepted_volume
            bid["accepted_price"] = expected_bid_price  # 0.0
            bid["accepted_volume"] = expected_volume  # 1e-6

            # Calculate rewards based on the accepted bids
            strategy.calculate_reward(storage_unit, mc, orderbook=bids)

            # Extract outputs
            reward = storage_unit.outputs["reward"].loc[product_index]
            profit = storage_unit.outputs["profit"].loc[product_index]
            costs = storage_unit.outputs["total_costs"].loc[product_index]

            # Calculate expected values
            duration_hours = 1  # Since the product tuple is 1 hour
            expected_profit = 0.0  # No profit from ignore
            expected_costs = 0.0  # No costs from ignore
            scaling_factor = (
                0.1 / storage_unit.max_power_discharge
            )  # 0.1 / 500 = 0.0002
            expected_reward = 0.0  # No profit or costs

            # Assert the calculated reward
            assert (
                reward[0] == expected_reward
            ), f"Expected reward {expected_reward}, got {reward[0]}"

            # Assert the calculated profit
            assert (
                profit[0] == expected_profit
            ), f"Expected profit {expected_profit}, got {profit[0]}"

            # Assert the calculated costs
            assert (
                costs[0] == expected_costs
            ), f"Expected costs {expected_costs}, got {costs[0]}"
