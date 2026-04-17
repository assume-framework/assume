# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd

from assume.common.base import LearningConfig
from assume.common.forecaster import PowerplantForecaster
from assume.common.market_objects import MarketConfig, MarketProduct

try:
    from assume.reinforcement_learning import Learning
    from assume.strategies.portfolio_learning_strategies import (
        PortfolioLearningStrategy,
    )
except ImportError:
    Learning = None
    PortfolioLearningStrategy = None

from assume.strategies.naive_strategies import EnergyNaiveStrategy
from assume.units.powerplant import PowerPlant

start = datetime(2023, 7, 1)
end = datetime(2023, 7, 2)
MARKET_ID = "EOM"

# Expected observation dimension with defaults foresight=12, nbins=4:
# unique_obs_dim = nbins*2 + 2 = 10;  obs_dim = 3*foresight + unique_obs_dim = 46
EXPECTED_OBS_DIM = 46


@dataclass
class MockUnitsOperator:
    id: str
    units: dict


@pytest.fixture
def portfolio_market_config():
    return MarketConfig(
        market_id=MARKET_ID,
        opening_hours=rr.rrule(rr.HOURLY, dtstart=start, until=end),
        opening_duration=rd(hours=1),
        market_mechanism="pay_as_clear",
        market_products=[MarketProduct(rd(hours=1), 1, rd(hours=1))],
        product_type="energy_eom",
    )


@pytest.fixture
def portfolio_units_operator():
    """
    MockUnitsOperator standing in for a UnitsOperator with 4 PowerPlant units,
    each having a distinct marginal cost (via fuel_price) to populate 4 cost bins.
    """
    index = pd.date_range(start, periods=48, freq="h")
    market_price_forecast = np.linspace(50, 150, 48)
    res_load_forecast = np.linspace(500, 1000, 48)
    units = {}
    for i, fuel_price in enumerate([10, 20, 30, 40]):
        ff = PowerplantForecaster(
            index,
            fuel_prices={"lignite": fuel_price, "co2": 10},
            residual_load={MARKET_ID: res_load_forecast},
            market_prices={MARKET_ID: market_price_forecast},
        )
        unit = PowerPlant(
            id=f"pp_{i}",
            unit_operator="test_portfolio_operator",
            technology="lignite",
            index=index,
            max_power=500 + i * 100,  # 500, 600, 700, 800 MW
            min_power=0,
            efficiency=0.5,
            additional_cost=5,
            bidding_strategies={MARKET_ID: EnergyNaiveStrategy()},
            fuel_type="lignite",
            emission_factor=0.5,
            forecaster=ff,
        )
        units[f"pp_{i}"] = unit
    return MockUnitsOperator(id="test_portfolio_operator", units=units)


@pytest.fixture
def portfolio_strategy():
    """
    Create a PortfolioLearningStrategy together with an initialised Learning role.

    initialize_policy() must be called explicitly here because
    PortfolioLearningStrategy.get_actions() accesses self.actor.min_output /
    self.actor.max_output even in collect_initial_experience_mode, unlike
    EnergyLearningStrategy which avoids the actor in that phase.
    """
    learning_config = LearningConfig(
        algorithm="matd3",
        learning_mode=True,
        training_episodes=3,
    )
    lr = Learning(learning_config, start, end)
    strategy = PortfolioLearningStrategy(
        learning_role=lr,
        unit_id="test_portfolio_operator",
        nbins=4,
    )
    lr.initialize_policy()
    return strategy, lr


@pytest.mark.require_learning
def test_portfolio_observation_dimensions(portfolio_units_operator, portfolio_strategy):
    """Observation tensor length equals the expected concrete value and decomposes correctly."""
    strategy, _ = portfolio_strategy
    product_index = pd.date_range(start, periods=1, freq="h")

    obs = strategy.create_observation(
        portfolio_units_operator,
        MARKET_ID,
        product_index[0],
        product_index[0] + pd.Timedelta(hours=1),
    )

    assert len(obs) == EXPECTED_OBS_DIM
    assert strategy.obs_dim == EXPECTED_OBS_DIM
    assert (
        strategy.unique_obs_dim + strategy.foresight * strategy.num_timeseries_obs_dim
        == EXPECTED_OBS_DIM
    )


@pytest.mark.require_learning
def test_portfolio_calculate_bids(
    portfolio_units_operator, portfolio_market_config, portfolio_strategy
):
    """
    calculate_bids returns one flex bid per unit (min_power=0 → no inflexible bids)
    with valid prices, and caches observation + action in the learning role.
    """
    strategy, lr = portfolio_strategy
    n_units = len(portfolio_units_operator.units)

    product_index = pd.date_range(start, periods=1, freq="h")
    product_tuples = [(t, t + pd.Timedelta(hours=1), None) for t in product_index]

    bids = strategy.calculate_bids(
        portfolio_units_operator, portfolio_market_config, product_tuples
    )

    # With min_power=0 for all units, inflexible generation is 0 → only flex bids
    assert len(bids) == n_units

    # Observation and action must be cached for subsequent learning updates
    assert start in lr.all_obs
    assert "test_portfolio_operator" in lr.all_obs[start]
    assert start in lr.all_actions
    assert "test_portfolio_operator" in lr.all_actions[start]


@pytest.mark.require_learning
def test_portfolio_calculate_reward(
    portfolio_units_operator, portfolio_market_config, portfolio_strategy
):
    """Reward and profit are stored in the learning-role cache after clearing feedback."""
    strategy, lr = portfolio_strategy

    product_index = pd.date_range(start, periods=1, freq="h")
    product_tuples = [(t, t + pd.Timedelta(hours=1), None) for t in product_index]

    bids = strategy.calculate_bids(
        portfolio_units_operator, portfolio_market_config, product_tuples
    )
    # Simulate full acceptance at highest bid price
    accepted_price = max(order["price"] for order in bids)
    for order in bids:
        order["accepted_price"] = accepted_price
        order["accepted_volume"] = order["volume"]

    strategy.calculate_reward(
        portfolio_units_operator, portfolio_market_config, orderbook=bids
    )

    assert len(lr.all_rewards) == 1
    assert "test_portfolio_operator" in lr.all_rewards[start]
    assert "test_portfolio_operator" in lr.all_profits[start]
