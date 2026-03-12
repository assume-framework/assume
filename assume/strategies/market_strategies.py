# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime, timedelta

import numpy as np
import torch as th

from assume.common.fast_pandas import FastSeries
from assume.common.utils import min_max_scale
from assume.markets.base_market import MarketRole
from assume.strategies.learning_strategies import TorchLearningStrategy

logger = logging.getLogger(__name__)


class TorchMarketLearningStrategy(TorchLearningStrategy):
    """
    A strategy to enable machine learning with pytorch for Markets.
    """

    def __init__(self, *args, market_role: MarketRole | None = None, unit_id, **kwargs):
        self.market_role = market_role
        self.unit_id = unit_id

        super().__init__(*args, **kwargs)

    def adjust_market_config(self, action):
        """This is the actual actions function, mirroring calculate_bids of the learning units and unit operator."""

        return self.market.market_config

    def calculate_reward(self):
        # calculate reward based on the market state and the action taken
        reward = 0
        # example: reward could be the profit or loss from the action taken
        # reward = self.market_role.calculate_profit_loss(action)

        # write to learning role
        # TODO: integrate with learning_role reward tracking
        return reward

    def prepare_observations(self, unit, market_id):
        pass  # TODO: implement

    def create_observation(self, unit, market_id: str, start: datetime, end: datetime):
        # create an observation for the given unit and market
        # example: observation could be the current order book state, recent trades, etc.
        observation = {}
        # observation['order_book'] = self.market_role.get_order_book(market_id)
        # observation['recent_trades'] = self.market_role.get_recent_trades(market_id)
        return observation

    def get_actions(self, next_observation):
        """
        Determines actions based on the current observation, applying noise for exploration if in learning mode.

        Args
        ----
        next_observation : torch.Tensor
            Observation data influencing bid price and direction.

        Returns
        -------
        torch.Tensor
            Actions that include bid price and direction.
        torch.Tensor
            Noise component which is already added to actions for exploration, if applicable.

        Notes
        -----
        In learning mode, actions incorporate noise for exploration. Initial exploration relies
        solely on noise to cover the action space broadly.
        """

        # distinction whether we are in learning mode or not to handle exploration realised with noise
        if self.learning_mode and not self.evaluation_mode:
            # if we are in learning mode the first x episodes we want to explore the entire action space
            # to get a good initial experience, in the area around the costs of the agent
            if self.collect_initial_experience_mode:
                # define current action as solely noise
                noise = th.normal(
                    mean=0.0,
                    std=self.exploration_noise_std,
                    size=(self.act_dim,),
                    dtype=self.float_type,
                    device=self.device,
                )

                # =============================================================================
                # 2.1 Get Actions and handle exploration
                # =============================================================================
                # only use noise as the action to enforce exploration
                curr_action = noise

            else:
                # if we are not in the initial exploration phase we chose the action with the actor neural net
                # and add noise to the action
                curr_action = self.actor(next_observation).detach()
                noise = self.action_noise.noise(
                    device=self.device, dtype=self.float_type
                )
                curr_action += noise

                # make sure that noise adding does not exceed the actual output of the NN as it pushes results in a direction that actor can't even reach
                curr_action = th.clamp(
                    curr_action, self.actor.min_output, self.actor.max_output
                )
        else:
            # if we are not in learning mode we just use the actor neural net to get the action without adding noise
            curr_action = self.actor(next_observation).detach()
            # noise is an tensor with zeros, because we are not in learning mode
            noise = th.zeros_like(curr_action, dtype=self.float_type)

        return curr_action, noise


class MarketLearningMaxPriceStrategy(TorchMarketLearningStrategy):
    """
    A specific implementation of the TorchMarketLearningStrategy that focuses on learning to set the maximum bid price in a market.
    """

    def __init__(
        self, *args, market_role: MarketRole | None = None, unit_id: str, **kwargs
    ):
        foresight = kwargs.pop("foresight", 1)
        act_dim = kwargs.pop("act_dim", 1)
        unique_obs_dim = kwargs.pop("unique_obs_dim", 0)
        num_timeseries_obs_dim = kwargs.pop("num_timeseries_obs_dim", 2)
        super().__init__(
            market_role=market_role,
            unit_id=unit_id,
            foresight=foresight,
            act_dim=act_dim,
            unique_obs_dim=unique_obs_dim,
            num_timeseries_obs_dim=num_timeseries_obs_dim,
            *args,
            **kwargs,
        )

        self.initial_max_bid_price = self.market_role.marketconfig.maximum_bid_price

        # Per-episode accumulators (reset each episode in adjust_market_config)
        self._episode_total_cost = 0.0
        self._episode_total_energy = 0.0
        self._episode_price_penalty = 0.0
        self._episode_n_products = 0

    def prepare_observations(self, market):
        """
        Prepares the observation for the market learning strategy, which could include features such as current market price, order book state, recent trades, etc.
        """
        market_id = market
        upper_scaling_factor_price = max(self.market_role.forecaster.price[market_id])
        lower_scaling_factor_price = min(self.market_role.forecaster.price[market_id])
        residual_load = self.market_role.forecaster.residual_load.get(
            market_id, FastSeries(index=self.market_role.forecaster.index, value=0)
        )
        upper_scaling_factor_res_load = max(residual_load)
        lower_scaling_factor_res_load = min(residual_load)

        self.scaled_res_load_obs = min_max_scale(
            residual_load,
            lower_scaling_factor_res_load,
            upper_scaling_factor_res_load,
        )

        self.scaled_prices_obs = min_max_scale(
            self.market_role.forecaster.price[market_id],
            lower_scaling_factor_price,
            upper_scaling_factor_price,
        )

    def create_observation(self, market_id: str, start: datetime, end: datetime):
        """
        Constructs a scaled observation tensor based on the unit's forecast data and internal state.

        Args
        ----
        unit : BaseUnit
            The unit providing forecast and internal state data.
        market_id : str
            Identifier for the specific market.
        start : datetime
            Start time for the observation period.
        end : datetime
            End time for the observation period.

        Returns
        -------
        torch.Tensor
            Observation tensor with data on forecasted residual load, price, and unit-specific values.

        Notes
        -----
        Observations are constructed from forecasted residual load and price over the foresight period,
        scaled by maximum demand and bid price. The last values in the observation vector represent
        unit-specific values, depending on the strategy and unit-type.
        """

        # ensure scaled observations are prepared
        if not hasattr(self, "scaled_res_load_obs") or not hasattr(
            self, "scaled_prices_obs"
        ):
            self.prepare_observations(market_id)

        # =============================================================================
        # 1.1 Get the Observations, which are the basis of the action decision
        # =============================================================================
        # as market learning only changes market wokrings one per episode the forecast is always the entrie episode from start to end
        foresight = int((end - start).total_seconds() / 3600)  # convert to hours
        # --- 1. Forecasted residual load and price (forward-looking) ---
        scaled_res_load_forecast_max = self.scaled_res_load_obs.window(
            start, foresight, direction="forward"
        ).max()

        scaled_price_forecast_max = self.scaled_prices_obs.window(
            start, foresight, direction="forward"
        ).max()

        # concat all observations into one array
        observation = np.array(
            [scaled_res_load_forecast_max, scaled_price_forecast_max]
        )

        # transfer array to GPU for NN processing
        observation = th.as_tensor(
            observation, dtype=self.float_type, device=self.device
        ).flatten()

        if self.learning_mode:
            self.learning_role.add_observation_to_cache(
                self.unit_id, start, observation
            )

        return observation

    def adjust_market_config(self):
        """Adjusts the market configuration by setting the maximum bid price based on the action taken."""

        # Reset episode accumulators at the start of each episode
        self._episode_total_cost = 0.0
        self._episode_total_energy = 0.0
        self._episode_price_penalty = 0.0
        self._episode_n_products = 0

        next_observation = self.create_observation(
            market_id=self.market_role.marketconfig.market_id,
            start=self.learning_role.start_datetime,
            end=self.learning_role.end_datetime,
        )
        actions, noise = self.get_actions(next_observation)

        self.market_role.marketconfig.maximum_bid_price = (
            actions[0]
            * 200  # TODO: what scaling for reduced action space instead of [-3000, 3000]
        )

        logger.info(
            f"MarketLearningMaxPriceStrategy: Adjusted max bid price to {self.market_role.marketconfig.maximum_bid_price:.2f} based on action {actions[0]:.4f} and noise {noise[0]:.4f}"
        )

        if self.learning_mode:
            self.learning_role.add_actions_to_cache(
                self.unit_id, self.learning_role.start_datetime, actions, noise
            )

    def calculate_reward(self, accepted_orderbook, market_meta):
        """
        Calculate reward that minimizes overall system costs and punishes market power abuse.

        Called after every clearing (e.g., once per hour). Accumulates cost and market power
        metrics internally, then writes the final reward to the learning role at episode end.

        The reward has two components:
        1. **Cost component**: Negative of total procurement cost normalized per MWh.
           Lower system cost -> higher reward.
        2. **Market power penalty**: For each cleared product, compares the realized clearing
           price to the merit-order price forecast. Penalizes when clearing price exceeds
           the forecast, indicating potential market power abuse.

        Args:
            accepted_orderbook (list[dict]): Orders accepted in this clearing.
            market_meta (list[dict]): Clearing metadata per product (prices, volumes, etc.).
        """
        if not market_meta:
            return 0.0

        market_id = self.market_role.marketconfig.market_id
        price_forecast = self.market_role.forecaster.price[market_id]

        for meta in market_meta:
            duration_hours = (meta["product_end"] - meta["product_start"]) / timedelta(
                hours=1
            )
            clearing_price = meta.get("price", 0.0)
            supply_volume = meta.get("supply_volume", 0.0)

            # Accumulate total system cost
            self._episode_total_cost += clearing_price * supply_volume * duration_hours
            self._episode_total_energy += meta.get("supply_volume_energy", 0.0)

            # Compare clearing price to merit-order forecast price
            merit_price = price_forecast.at[meta["product_start"]]
            price_excess = max(clearing_price - merit_price, 0.0)
            # Normalize by initial max bid price so penalty is in [0, 1] range
            self._episode_price_penalty += price_excess / self.initial_max_bid_price
            self._episode_n_products += 1

        # Only compute and submit the final reward at the end of the episode
        last_product_end = market_meta[-1]["product_end"]

        from assume.common.utils import datetime2timestamp

        if datetime2timestamp(last_product_end) < self.learning_role.end:
            return 0.0

        # --- Final reward computation ---
        # 1. Cost component: average cost per MWh, scaled to [-1, 0] range
        avg_cost = (
            self._episode_total_cost / self._episode_total_energy
            if self._episode_total_energy > 0
            else 0.0
        )
        cost_reward = -avg_cost / self.initial_max_bid_price

        # 2. Market power penalty: average price excess across all products
        avg_price_penalty = (
            self._episode_price_penalty / self._episode_n_products
            if self._episode_n_products > 0
            else 0.0
        )

        market_power_weight = 1.0
        reward = -market_power_weight * avg_price_penalty

        # Write to learning role
        start = self.learning_role.start_datetime
        if self.learning_mode:
            self.learning_role.add_reward_to_cache(
                self.unit_id, start, reward, avg_price_penalty, -avg_cost
            )

        return reward
