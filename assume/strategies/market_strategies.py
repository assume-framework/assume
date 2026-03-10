# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime

import torch as th

from assume.strategies.learning_strategies import TorchLearningStrategy
from assume.markets.base_market import MarketRole
from assume.common.utils import min_max_scale
from assume.common.fast_pandas import FastSeries

logger = logging.getLogger(__name__)


class TorchMarketLearningStrategy(TorchLearningStrategy):
    """
    A strategy to enable machine learning with pytorch for Markets.
    """

    def __init__(self, *args, market_role: MarketRole | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.market_role = market_role

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

    def create_observation(
        self, unit, market_id: str, start: datetime, end: datetime
    ):
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

    def prepare_observations(self, market):
        """
        Prepares the observation for the market learning strategy, which could include features such as current market price, order book state, recent trades, etc.
        """
        
        upper_scaling_factor_price = max(market.forecaster.price[market.market_id])
        lower_scaling_factor_price = min(market.forecaster.price[market.market_id])
        residual_load = market.forecaster.residual_load.get(
            market.market_id, FastSeries(index=market.index, value=0)
        )
        upper_scaling_factor_res_load = max(residual_load)
        lower_scaling_factor_res_load = min(residual_load)

        self.scaled_res_load_obs = min_max_scale(
            residual_load,
            lower_scaling_factor_res_load,
            upper_scaling_factor_res_load,
        )

        self.scaled_prices_obs = min_max_scale(
            market.forecaster.price[market.market_id],
            lower_scaling_factor_price,
            upper_scaling_factor_price,
        )
        
        
    def create_observation(
        self, unit: BaseUnit, market_id: str, start: datetime, end: datetime
    ):
    
    
    def adjust_market_config(self, action):
        """Adjusts the market configuration by setting the maximum bid price based on the action taken."""
        self.market_role.marketconfig.maximum_bid_price = (
            action * self.market_role.marketconfig.price_tick
        )
        
    def calculate_reward(self):
        # calculate reward based on the market state and the action taken
        reward = 0
        # example: reward could be the profit or loss from the action taken
        # reward = self.market_role.calculate_profit_loss(action)

        # write to learning role
        # TODO: integrate with learning_role reward tracking
        return reward