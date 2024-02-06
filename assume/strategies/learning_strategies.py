# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th

from assume.common.base import LearningStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.reinforcement_learning.learning_utils import Actor, NormalActionNoise

logger = logging.getLogger(__name__)


class RLStrategy(LearningStrategy):
    """
    Reinforcement Learning Strategy, that lets agent learn to bid on an Energy Only Makret.

    The agent submittes two price bids
    - one for the infelxible (P_min) and one for the flexible part (P_max-P_min) of ist capacity.

    Parameters:
        foresight (int): Number of time steps to look ahead. Defaults to 24.
        max_bid_price (float): Maximum bid price. Defaults to 100.
        max_demand (float): Maximum demand. Defaults to 10e3.
        device (str): Device to run on. Defaults to "cpu".
        float_type (str): Float type to use. Defaults to "float32".
        learning_mode (bool): Whether to use learning mode. Defaults to False.
        actor (torch.nn.Module): Actor network. Defaults to None.
        order_types (list[str]): Order types to use. Defaults to ["SB"].
        action_noise (NormalActionNoise): Action noise. Defaults to None.
        collect_initial_experience_mode (bool): Whether to collect initial experience. Defaults to True.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.unit_id = kwargs["unit_id"]

        # defines bounds of actions space
        self.max_bid_price = kwargs.get("max_bid_price", 100)
        self.max_demand = kwargs.get("max_demand", 10e3)

        # tells us whether we are training the agents or just executing per-learnind stategies
        self.learning_mode = kwargs.get("learning_mode", False)

        # sets the devide of the actor network
        device = kwargs.get("device", "cpu")
        self.device = th.device(device if th.cuda.is_available() else "cpu")
        if not self.learning_mode:
            self.device = th.device("cpu")

        # future: add option to choose between float16 and float32
        # float_type = kwargs.get("float_type", "float32")
        self.float_type = th.float

        # for definition of observation space
        self.foresight = kwargs.get("foresight", 24)

        # define used order types
        self.order_types = kwargs.get("order_types", ["SB"])

        if self.learning_mode:
            self.learning_role = None
            self.collect_initial_experience_mode = kwargs.get(
                "episodes_collecting_initial_experience", True
            )

            self.action_noise = NormalActionNoise(
                mu=0.0,
                sigma=kwargs.get("noise_sigma", 0.1),
                action_dimension=self.act_dim,
                scale=kwargs.get("noise_scale", 1.0),
                dt=kwargs.get("noise_dt", 1.0),
            )

        elif Path(kwargs["trained_policies_save_path"]).is_dir():
            self.load_actor_params(load_path=kwargs["trained_policies_save_path"])
        else:
            logger.error("did not have learning mode and folder did not exist")

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Calculates bids for a unit, based on the actions from the actors.

        Args:
            unit (SupportsMinMax): Unit to calculate bids for.
            market_config (MarketConfig): Market configuration.
            product_tuples (list[Product]): Product tuples.
            **kwargs: Keyword arguments.

        Returns:
            Orderbook: Bids containing start time, end time, price, volume and bid type.

        """

        bid_quantity_inflex, bid_price_inflex = 0, 0
        bid_quantity_flex, bid_price_flex = 0, 0

        start = product_tuples[0][0]
        end = product_tuples[0][1]
        # get technical bounds for the unit output from the unit
        min_power, max_power = unit.calculate_min_max_power(start, end)
        min_power = min_power[start]
        max_power = max_power[start]

        # =============================================================================
        # 1. Get the Observations, which are the basis of the action decision
        # =============================================================================
        next_observation = self.create_observation(
            unit=unit,
            start=start,
            end=end,
        )

        # =============================================================================
        # 2. Get the Actions, based on the observations
        # =============================================================================
        actions, noise = self.get_actions(next_observation)

        # =============================================================================
        # 3. Transform Actions into bids
        # =============================================================================
        # actions are in the range [0,1], we need to transform them into actual bids
        # we can use our domain knowledge to guide the bid formulation
        bid_prices = actions * self.max_bid_price

        # 3.1 formulate the bids for Pmin
        # Pmin, the minium run capacity is the inflexible part of the bid, which should always be accepted
        bid_quantity_inflex = min_power
        bid_price_inflex = min(bid_prices)

        # 3.1 formulate the bids for Pmax - Pmin
        # Pmin, the minium run capacity is the inflexible part of the bid, which should always be accepted
        bid_quantity_flex = max_power - bid_quantity_inflex
        bid_price_flex = max(bid_prices)

        # actually formulate bids in orderbook format
        bids = [
            {
                "start_time": start,
                "end_time": end,
                "only_hours": None,
                "price": bid_price_inflex,
                "volume": bid_quantity_inflex,
            },
            {
                "start_time": start,
                "end_time": end,
                "only_hours": None,
                "price": bid_price_flex,
                "volume": bid_quantity_flex,
            },
        ]

        # store results in unit outputs which are written to database by unit operator
        unit.outputs["rl_observations"][start] = next_observation
        unit.outputs["rl_actions"][start] = actions
        unit.outputs["rl_exploration_noise"][start] = noise

        return bids

    def get_actions(self, next_observation):
        """
        Gets actions for a unit containing two bid prices depending on the observation.

        Args:
            next_observation (torch.Tensor): Next observation.

        Returns:
            Actions (torch.Tensor): Actions containing two bid prices.

        Note:
            If the agent is in learning mode, the actions are chosen by the actor neuronal net and noise is added to the action
            In the first x episodes the agent is in initial exploration mode, where the action is chosen by noise only to explore the entire action space.
            X is defined by episodes_collecting_initial_experience.
            If the agent is not in learning mode, the actions are chosen by the actor neuronal net without noise.
        """

        # distinction whether we are in learning mode or not to handle exploration realised with noise
        if self.learning_mode:
            # if we are in learning mode the first x episodes we want to explore the entire action space
            # to get a good initial experience, in the area around the costs of the agent
            if self.collect_initial_experience_mode:
                # define current action as soley noise
                noise = (
                    th.normal(
                        mean=0.0, std=0.2, size=(1, self.act_dim), dtype=self.float_type
                    )
                    .to(self.device)
                    .squeeze()
                )

                # =============================================================================
                # 2.1 Get Actions and handle exploration
                # =============================================================================
                base_bid = next_observation[-1]

                # add noise to the last dimension of the observation
                # needs to be adjusted if observation space is changed, because only makes sense
                # if the last dimension of the observation space are the marginal cost
                curr_action = noise + base_bid.clone().detach()

            else:
                # if we are not in the initial exploration phase we chose the action with the actor neural net
                # and add noise to the action
                curr_action = self.actor(next_observation).detach()
                noise = th.tensor(
                    self.action_noise.noise(), device=self.device, dtype=self.float_type
                )
                curr_action += noise
        else:
            # if we are not in learning mode we just use the actor neural net to get the action without adding noise

            curr_action = self.actor(next_observation).detach()
            noise = tuple(0 for _ in range(self.act_dim))

        curr_action = curr_action.clamp(-1, 1)

        return curr_action, noise

    def create_observation(
        self,
        unit: SupportsMinMax,
        start: datetime,
        end: datetime,
    ):
        """
        Creates an observation.

        Args:
            unit (SupportsMinMax): Unit to create observation for.
            start (datetime.datetime): Start time.
            end (datetime.datetime): End time.

        Returns:
            Observation (torch.Tensor): Observation.
        """
        end_excl = end - unit.index.freq

        # get the forecast length depending on the tme unit considered in the modelled unit
        forecast_len = pd.Timedelta((self.foresight - 1) * unit.index.freq)

        # =============================================================================
        # 1.1 Get the Observations, which are the basis of the action decision
        # =============================================================================
        scaling_factor_res_load = self.max_demand

        # price forecast
        scaling_factor_price = self.max_bid_price

        # total capacity and marginal cost
        scaling_factor_total_capacity = unit.max_power

        # marginal cost
        # Obs[2*foresight+1:2*foresight+2]
        scaling_factor_marginal_cost = self.max_bid_price

        # checks if we are at end of simulation horizon, since we need to change the forecast then
        # for residual load and price forecast and scale them
        if end_excl + forecast_len > unit.forecaster["residual_load_EOM"].index[-1]:
            scaled_res_load_forecast = (
                unit.forecaster["residual_load_EOM"].loc[start:].values
                / scaling_factor_res_load
            )
            scaled_res_load_forecast = np.concatenate(
                [
                    scaled_res_load_forecast,
                    unit.forecaster["residual_load_EOM"].iloc[
                        : self.foresight - len(scaled_res_load_forecast)
                    ],
                ]
            )

        else:
            scaled_res_load_forecast = (
                unit.forecaster["residual_load_EOM"]
                .loc[start : end_excl + forecast_len]
                .values
                / scaling_factor_res_load
            )

        if end_excl + forecast_len > unit.forecaster["price_EOM"].index[-1]:
            scaled_price_forecast = (
                unit.forecaster["price_EOM"].loc[start:].values / scaling_factor_price
            )
            scaled_price_forecast = np.concatenate(
                [
                    scaled_price_forecast,
                    unit.forecaster["price_EOM"].iloc[
                        : self.foresight - len(scaled_price_forecast)
                    ],
                ]
            )

        else:
            scaled_price_forecast = (
                unit.forecaster["price_EOM"].loc[start : end_excl + forecast_len].values
                / scaling_factor_price
            )

        # get last accapted bid volume and the current marginal costs of the unit
        current_volume = unit.get_output_before(start)
        current_costs = unit.calc_marginal_cost_with_partial_eff(current_volume, start)

        # scale unit outpus
        scaled_total_capacity = current_volume / scaling_factor_total_capacity
        scaled_marginal_cost = current_costs / scaling_factor_marginal_cost

        # concat all obsverations into one array
        observation = np.concatenate(
            [
                scaled_res_load_forecast,
                scaled_price_forecast,
                np.array([scaled_total_capacity, scaled_marginal_cost]),
            ]
        )

        # transfer arry to GPU for NN processing
        observation = (
            th.tensor(observation, dtype=self.float_type)
            .to(self.device, non_blocking=True)
            .view(-1)
        )

        return observation.detach().clone()

    def calculate_reward(
        self,
        unit,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ):
        """
        Calculates the reward for the unit.

        Args:
            unit (SupportsMinMax): Unit to calculate reward for.
            marketconfig (MarketConfig): Market configuration.
            orderbook (Orderbook): Orderbook.
        """

        # =============================================================================
        # 4. Calculate Reward
        # =============================================================================
        # function is called after the market is cleared and we get the market feedback,
        # so we can calculate the profit

        product_type = marketconfig.product_type

        profit = 0
        reward = 0
        opportunity_cost = 0

        # iterate over all orders in the orderbook, to calculate order specific profit
        for order in orderbook:
            start = order["start_time"]
            end = order["end_time"]
            end_excl = end - unit.index.freq

            # depending on way the unit calaculates marginal costs we take costs
            if unit.marginal_cost is not None:
                marginal_cost = (
                    unit.marginal_cost[start]
                    if len(unit.marginal_cost) > 1
                    else unit.marginal_cost
                )
            else:
                marginal_cost = unit.calc_marginal_cost_with_partial_eff(
                    power_output=unit.outputs[product_type].loc[start:end_excl],
                    timestep=start,
                )

            duration = (end - start) / timedelta(hours=1)

            # calculate profit as income - running_cost from this event
            price_difference = order["accepted_price"] - marginal_cost
            order_profit = price_difference * order["accepted_volume"] * duration

            # calculate opportunity cost
            # as the loss of income we have because we are not running at full power
            order_opportunity_cost = (
                price_difference
                * (
                    unit.max_power - unit.outputs[product_type].loc[start:end_excl]
                ).sum()
                * duration
            )

            # if our opportunity costs are negative, we did not miss an opportunity to earn money and we set them to 0
            order_opportunity_cost = max(order_opportunity_cost, 0)

            # collect profit and opportunity cost for all orders
            opportunity_cost += order_opportunity_cost
            profit += order_profit

        # consideration of start-up costs, which are evenly divided between the
        # upward and downward regulation events
        if (
            unit.outputs[product_type].loc[start] != 0
            and unit.outputs[product_type].loc[start - unit.index.freq] == 0
        ):
            profit = profit - unit.hot_start_cost / 2
        elif (
            unit.outputs[product_type].loc[start] == 0
            and unit.outputs[product_type].loc[start - unit.index.freq] != 0
        ):
            profit = profit - unit.hot_start_cost / 2

        # ---------------------------
        # 4.1 Calculate Reward
        # The straight forward implemntation would be reward = profit, yet we would like to give the agent more guidance
        # in the learning process, so we add a regret term to the reward, which is the opportunity cost
        # define the reward and scale it

        scaling = 0.1 / unit.max_power
        regret_scale = 0.2
        reward = float(profit - regret_scale * opportunity_cost) * scaling

        # store results in unit outputs which are written to database by unit operator
        unit.outputs["profit"].loc[start:end_excl] += profit
        unit.outputs["reward"].loc[start:end_excl] = reward
        unit.outputs["regret"].loc[start:end_excl] = opportunity_cost

    def load_actor_params(self, load_path):
        """
        Loads actor parameters.

        Args:
            load_path (str): Path to load from.
        """
        directory = f"{load_path}/actors/actor_{self.unit_id}.pt"

        params = th.load(directory, map_location=self.device)

        self.actor = Actor(self.obs_dim, self.act_dim, self.float_type)
        self.actor.load_state_dict(params["actor"])

        if self.learning_mode:
            self.actor_target = Actor(self.obs_dim, self.act_dim, self.float_type)
            self.actor_target.load_state_dict(params["actor_target"])
            self.actor_target.eval()
            self.actor.optimizer.load_state_dict(params["actor_optimizer"])
