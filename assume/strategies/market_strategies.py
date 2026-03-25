# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime

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
                    mean=1.0,  # exploration "without price cap"
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
        self._episode_sw = 0.0  # realised weighted social welfare
        self._episode_max_sw = (
            0.0  # maximum attainable SW (all demand served at zero cost)
        )
        self._episode_price_volume = 0.0  # sum of P* * Q* across clearings
        self._episode_cleared_volume = 0.0  # sum of Q* across clearings

    def prepare_observations(self, market):
        """
        Prepares the observation for the market learning strategy, which could include features such as current market price, order book state, recent trades, etc.
        """
        market_id = market
        self.upper_scaling_factor_price = max(
            self.market_role.forecaster.price[market_id]
        )
        self.lower_scaling_factor_price = min(
            self.market_role.forecaster.price[market_id]
        )
        upper_scaling_factor_price = self.upper_scaling_factor_price
        lower_scaling_factor_price = self.lower_scaling_factor_price
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
        # as market learning only changes market wokrings one per episode the forecast is always the entire episode from start to end
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
        self._episode_sw = 0.0
        self._episode_max_sw = 0.0
        self._episode_error = 0.0
        self._episode_price_volume = 0.0
        self._episode_cleared_volume = 0.0
        self._episode_served_volume = 0.0
        self._episode_demanded_volume = 0.0

        next_observation = self.create_observation(
            market_id=self.market_role.marketconfig.market_id,
            start=self.learning_role.start_datetime,
            end=self.learning_role.end_datetime,
        )
        actions, noise = self.get_actions(next_observation)

        self.market_role.marketconfig.maximum_bid_price = (
            actions[0]
            * 100  # TODO: what scaling for reduced action space instead of [-3000, 3000]
        )

        logger.info(
            f"MarketLearningMaxPriceStrategy: Adjusted max bid price to {self.market_role.marketconfig.maximum_bid_price:.2f} based on action {actions[0]:.4f} and noise {noise[0]:.4f}"
        )

        if self.learning_mode:
            self.learning_role.add_actions_to_cache(
                self.unit_id, self.learning_role.start_datetime, actions, noise
            )

    def calculate_reward(self, orderbook, market_meta):
        """
        Calculate reward using a weighted social welfare (SW) function.

        Called after every clearing (e.g., once per hour). Accumulates SW internally,
        then writes the normalised reward to the learning role at episode end.

        The SW formula is the discrete analogue of equation (5) from the price-cap
        literature:

            SW = alpha  * sum_{accepted demand}  price_d * |v_d*|
               - (1-alpha) * sum_{accepted supply}  price_s * v_s*
               + (1 - 2*alpha) * P* * Q*

        where alpha in [0,1] encodes the regulator's preference:
          - alpha = 1  -> pure consumer welfare (penalises high P*)
          - alpha = 0  -> pure producer welfare (rewards high P*)
          - alpha = 0.5 -> standard allocative efficiency (transfer term cancels)

        Lost load is handled implicitly: rejected demand orders have accepted_volume=0,
        so their bid value is never added to SW. With alpha > 0.5 these forgone values
        cost more than the supply savings, naturally discouraging a cap that causes
        lost load without any hand-tuned penalty weight.

        The reward submitted to the learning role is SW / max_SW, normalised to (-inf, 1].

        Args:
            orderbook (list[dict]): All orders in the market (accepted and rejected).
            market_meta (list[dict]): Clearing metadata per product (prices, volumes, etc.).
        """
        if not orderbook and not market_meta:
            return 0.0

        # alpha in [0, 1]: 1 = pure consumer welfare, 0 = pure producer welfare,
        # 0.5 = standard allocative efficiency (transfer term cancels).
        # Values > 0.5 penalise high clearing prices (market power).
        alpha = 1
        # price_penalty_weight controls how strongly the agent is penalised for
        # high clearing prices, independently of VOLL.
        price_penalty_weight = 1
        sw = 0.0
        max_sw = 0.0
        error = 0.0
        price_volume = 0.0
        cleared_volume = 0.0

        # --- Demand side ---
        # Accepted demand contributes alpha * bid_price * accepted_volume.
        # Rejected demand contributes 0 — this is the implicit lost-load penalty.
        # max_sw is the theoretical SW ceiling: all demand served AND clearing price=0
        # (zero-cost supply, e.g. fully renewable dispatch). The demand bid price is VOLL
        # (e.g. 3000 €/MWh), so max_sw = alpha * VOLL * Q_demand. This makes the
        # normalised reward = (Q*/Q_D) * (1 - P*/VOLL), the product of service ratio
        # and price efficiency, both in [0,1].
        served_volume = 0.0
        demanded_volume = 0.0
        for order in orderbook:
            if order["volume"] < 0:
                accepted_vol = abs(order.get("accepted_volume") or 0.0)
                bid_vol = abs(order["volume"])
                sw += alpha * order["price"] * accepted_vol
                max_sw += alpha * order["price"] * bid_vol
                served_volume += accepted_vol
                demanded_volume += bid_vol

        # --- Supply side ---
        # Accepted supply subtracts (1-alpha) * bid_price * accepted_volume (supply cost).
        for order in orderbook:
            if order["volume"] > 0:
                accepted_vol = order.get("accepted_volume") or 0.0
                sw -= (1 - alpha) * order["price"] * accepted_vol

        # --- Transfer term (per product from market_meta) ---
        # (1 - 2*alpha) * P* * Q*: positive when alpha < 0.5 (favour producers),
        # negative when alpha > 0.5 (penalise high clearing prices / market power).
        # With price_penalty_weight the transfer term is amplified so price
        # changes are visible relative to the VOLL-dominated demand value.
        for meta in market_meta:
            clearing_price = meta.get("price", 0.0)
            meta_cleared_volume = meta.get("supply_volume", 0.0)
            sw += (
                price_penalty_weight
                * (1 - 2 * alpha)
                * clearing_price
                * meta_cleared_volume
            )
            price_volume += clearing_price * meta_cleared_volume
            cleared_volume += meta_cleared_volume

            # Unscale forecast price from [0,1] back to €/MWh
            scaled_forecast = self.scaled_prices_obs[meta["product_start"]]
            forecast_price = self.lower_scaling_factor_price + scaled_forecast * (
                self.upper_scaling_factor_price - self.lower_scaling_factor_price
            )
            error += -1 * max(
                (clearing_price - forecast_price) / forecast_price
                if forecast_price > 0
                else 0.0,
                0,
            )

        self._episode_sw += sw
        self._episode_max_sw += max_sw
        self._episode_price_volume += price_volume
        self._episode_cleared_volume += cleared_volume
        self._episode_served_volume += served_volume
        self._episode_demanded_volume += demanded_volume

        self._episode_error += error

        # Only compute and submit the final reward at the end of the episode.
        if market_meta:
            last_product_end = market_meta[-1]["product_end"]
        else:
            last_product_end = max(order["end_time"] for order in orderbook)

        from assume.common.utils import datetime2timestamp

        if datetime2timestamp(last_product_end) < self.learning_role.end:
            return 0.0

        # # similar to Renshaw-Whitman et al. (2024) ?
        # reward = self._episode_sw / 24 / 1000

        # Normalise by max attainable SW so the reward is in (-inf, 1].
        # A reward of 1 means perfect welfare (all demand served, zero supply cost).
        reward = (
            self._episode_sw / self._episode_max_sw if self._episode_max_sw > 0 else 0.0
        )

        # reward hacking to check whether learning is successful
        reward = (
            self._episode_error / 24 + reward
        )  # TODO: consider how to best combine error and SW into a single reward signal, e.g. with a weighting factor

        # # Simple reward: serve all load, minimize cap.
        # # service_ratio ∈ [0,1] rewards dispatch; max(cap, 0) ∈ [0,cap] penalises a loose cap.
        # # Optimal point: lowest cap that still clears all demand.
        # service_ratio = (
        #     self._episode_served_volume / self._episode_demanded_volume
        #     if self._episode_demanded_volume > 0 else 0.0
        # )
        # cap = self.market_role.marketconfig.maximum_bid_price
        # reward = service_ratio - max(cap, 0) / 300.0

        start = self.learning_role.start_datetime
        if self.learning_mode:
            self.learning_role.add_reward_to_cache(
                self.unit_id,
                start,
                reward,
                1 - reward,
                self._episode_sw / self._episode_max_sw,
            )
            # The recurrent store_to_buffer_and_update fires at the same simulation
            # timestamp as this final clearing, so it may run before the reward is
            # in the cache. Scheduling an instant task here guarantees it runs after
            # this synchronous call returns, by which point obs+action+reward are all
            # present in the cache.
            self.learning_role.trigger_buffer_flush()

        return reward
