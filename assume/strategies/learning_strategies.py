from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th

from assume.common.base import LearningStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Order, Orderbook, Product
from assume.reinforcement_learning.learning_utils import Actor, NormalActionNoise


class RLStrategy(LearningStrategy):
    """
    Reinforcement Learning Strategy

    :param foresight: Number of time steps to look ahead. Default 24.
    :type foresight: int
    :param max_bid_price: Maximum bid price
    :type max_bid_price: float
    :param max_demand: Maximum demand
    :type max_demand: float
    :param device: Device to run on
    :type device: str
    :param float_type: Float type to use
    :type float_type: str
    :param learning_mode: Whether to use learning mode
    :type learning_mode: bool
    :param actor: Actor network
    :type actor: torch.nn.Module
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.unit_id = kwargs["unit_id"]

        # defines bounds of actions space
        self.max_bid_price = kwargs.get("max_bid_price", 100)
        self.max_demand = kwargs.get("max_demand", 10e3)

        # sets the devide of the actor network
        device = kwargs.get("device", "cpu")
        self.device = th.device(device)

        float_type = kwargs.get("float_type", "float32")
        self.float_type = th.float if float_type == "float32" else th.float16

        # tells us whether we are training the agents or just executing per-learnind stategies
        self.learning_mode = kwargs.get("learning_mode", False)

        # for definition of observation space
        self.foresight = kwargs.get("foresight", 24)

        if self.learning_mode:
            self.learning_role = None
            self.collect_initial_experience_mode = kwargs.get(
                "collecting_initial_experience", True
            )

            self.action_noise = NormalActionNoise(
                mu=0.0,
                sigma=kwargs.get("noise_sigma", 0.1),
                action_dimension=self.act_dim,
                scale=kwargs.get("noise_scale", 1.0),
                dt=kwargs.get("noise_dt", 1.0),
            )

        elif Path(load_path=kwargs["load_learned_path"]).is_dir():
            self.load_actor_params(load_path=kwargs["load_learned_path"])

        self.curr_reward = None

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Calculate bids for a unit

        :param unit: Unit to calculate bids for
        :type unit: SupportsMinMax
        :param market_config: Market configuration
        :type market_config: MarketConfig
        :param product_tuples: Product tuples
        :type product_tuples: list[Product]
        :return: Bids containing start time, end time, price and volume
        :rtype: Orderbook
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

        # => enter your code in self.create_observation(...)
        next_observation = self.create_observation(
            unit=unit,
            start=start,
            end=end,
        )

        # =============================================================================
        # 2. Get the Actions, based on the observations
        # =============================================================================

        # => enter your code in self.get_actions(...)
        actions, noise = self.get_actions(next_observation)

        # =============================================================================
        # 3. Transform Actions into bids
        # =============================================================================
        # actions are in the range [0,1], we need to transform them into actual bids
        # we can use our domain knowledge to guide the bid formulation

        # => enter your code here

        # ---------------------------
        # 3.1 rescale actiosn to actual prices
        # bid_prices = ?

        bid_prices = actions * self.max_bid_price
        # ---------------------------

        # ---------------------------
        # 3.1 formulate the bids for Pmin
        # Pmin, the minium run capacity is the inflexible part of the bid, which should always be accepted
        # bid_quantity_inflex = ?
        # bid_price_inflex = ?

        bid_quantity_inflex = min_power
        bid_price_inflex = min(bid_prices)
        # ---------------------------

        # ---------------------------
        # 3.1 formulate the bids for Pmax - Pmin
        # Pmin, the minium run capacity is the inflexible part of the bid, which should always be accepted
        # bid_quantity_flex = ?
        # bid_price_flex = ?

        bid_quantity_flex = max_power - bid_quantity_inflex
        bid_price_flex = max(bid_prices)
        # ---------------------------

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
        Get actions

        :param next_observation: Next observation
        :type next_observation: torch.Tensor
        :return: Actions
        :rtype: torch.Tensor
        """

        # distinction whetere we are in learning mode or not to handle exploration realised with noise
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

                # => Your code here:
                base_bid = next_observation[-1]

                # add niose to the last dimension of the observation
                # needs to be adjusted if observation space is changed, because only makes sense
                # if the last dimension of the observation space are the marginal cost

                curr_action = noise + base_bid.clone().detach()

            else:
                # if we are not in the initial exploration phase we chose the action with the actor neuronal net
                # and add noise to the action

                curr_action = self.actor(next_observation).detach()
                noise = th.tensor(
                    self.action_noise.noise(), device=self.device, dtype=self.float_type
                )
                curr_action += noise
        else:
            # if we are not in learning mode we just use the actor neuronal net to get the action without adding noise

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
        Create observation

        :param unit: Unit to create observation for
        :type unit: SupportsMinMax
        :param start: Start time
        :type start: datetime
        :param end: End time
        :type end: datetime
        :return: Observation
        :rtype: torch.Tensor"""
        end_excl = end - unit.index.freq

        # get the forecast length depending on the tme unit considered in the modelled unit
        forecast_len = pd.Timedelta((self.foresight - 1) * unit.index.freq)

        # =============================================================================
        # 1.1 Get the Observations, which are the basis of the action decision
        # =============================================================================

        # => Your code here:
        # Pick suitable scaling factors for the individual observations
        # Why do we need scaling? What should you pay attention to while scaling?

        # residual load forecast
        # Obs[0:foresight-1]
        scaling_factor_res_load = self.max_demand

        # price forecast
        # Obs[foresight:2*foresight-1]
        scaling_factor_price = self.max_bid_price

        # total capacity and marginal cost
        # Obs[2*foresight:2*foresight+1]
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

        if end_excl + forecast_len > unit.forecaster["price_forecast"].index[-1]:
            scaled_price_forecast = (
                unit.forecaster["price_forecast"].loc[start:].values
                / scaling_factor_price
            )
            scaled_price_forecast = np.concatenate(
                [
                    scaled_price_forecast,
                    unit.forecaster["price_forecast"].iloc[
                        : self.foresight - len(scaled_price_forecast)
                    ],
                ]
            )

        else:
            scaled_price_forecast = (
                unit.forecaster["price_forecast"]
                .loc[start : end_excl + forecast_len]
                .values
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
        Calculate reward

        :param unit: Unit to calculate reward for
        :type unit: SupportsMinMax
        :param marketconfig: Market configuration
        :type marketconfig: MarketConfig
        :param orderbook: Orderbook
        :type orderbook: Orderbook
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

            # calculate profit as income - running_cost from this event
            duration = (end - start) / timedelta(hours=1)
            order_profit = order["accepted_price"] * order["volume"] * duration

            # calculate opportunity cost
            # as the loss of income we have because we are not running at full power
            price_difference = order["price"] - marginal_cost

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

        scaling = 0.1 / unit.max_power
        regret_scale = 0.2
        reward = float(profit - regret_scale * opportunity_cost) * scaling

        # store results in unit outputs which are written to database by unit operator
        unit.outputs["profit"].loc[start:end_excl] += float(profit)
        unit.outputs["reward"].loc[start:end_excl] = reward
        unit.outputs["regret"].loc[start:end_excl] = float(opportunity_cost)
        unit.outputs["learning_mode"].loc[start:end_excl] = self.learning_mode

    def load_actor_params(self, load_path):
        """
        Load actor parameters

        :param simulation_id: Simulation ID
        :type simulation_id: str
        """
        directory = f"{load_path}/actors/actor_{self.unit_id}.pt"

        params = th.load(directory)

        self.actor = Actor(self.obs_dim, self.act_dim, self.float_type)
        self.actor.load_state_dict(params["actor"])

        if self.learning_mode:
            self.actor_target = Actor(self.obs_dim, self.act_dim, self.float_type)
            self.actor_target.load_state_dict(params["actor_target"])
            self.actor_target.eval()
            self.actor.optimizer.load_state_dict(params["actor_optimizer"])


class RlUCStrategy(RLStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_bid_multiplier = 2

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Calculate bids for a unit

        :param unit: Unit to calculate bids for
        :type unit: SupportsMinMax
        :param market_config: Market configuration
        :type market_config: MarketConfig
        :param product_tuples: Product tuples
        :type product_tuples: list[Product]
        :return: Bids containing start time, end time, price and volume
        :rtype: Orderbook
        """
        start = product_tuples[0][0]
        end_all = product_tuples[-1][1]
        end = product_tuples[0][1]

        min_power, max_power = unit.calculate_min_max_power(start, end)
        min_power = min_power[start]
        max_power = max_power[start]

        next_observation = self.create_observation(
            unit=unit,
            start=start,
            end=end,
        )
        initial_output = unit.get_output_before(start)
        marginal_cost = unit.calculate_marginal_cost(start, initial_output)
        current_status = 1 if unit.get_operation_time(start) > 0 else 0

        actions, noise = self.get_actions(next_observation)
        # convert actions from -1 to 1 into 1 to 2
        bid_prices = (actions + 3) / 2 * marginal_cost
        price_profile = {
            product[0]: bid_prices[i] for i, product in enumerate(product_tuples)
        }

        order: Order = {
            "start_time": start,
            "end_time": end_all,
            "volume": -1,
            "price": price_profile,
            "min_power": unit.min_power,
            "max_power": unit.max_power,
            "ramp_up": unit.ramp_up,
            "ramp_down": unit.ramp_down,
            "no_load_cost": unit.no_load_cost,
            "start_up_cost": unit.hot_start_cost,
            "shut_down_cost": unit.shut_down_cost,
            "initial_output": initial_output,
            "initial_status": current_status,
            "bid_type": "MPB",
        }

        bids = [order]

        # store results in unit outputs which are written to database by unit operator
        unit.outputs["rl_observations"][start] = next_observation
        unit.outputs["rl_actions"][start] = actions
        unit.outputs["rl_exploration_noise"][start] = noise

        return bids

    def calculate_reward(
        self,
        unit,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ):
        """
        Calculate reward

        :param unit: Unit to calculate reward for
        :type unit: SupportsMinMax
        :param marketconfig: Market configuration
        :type marketconfig: MarketConfig
        :param orderbook: Orderbook
        :type orderbook: Orderbook
        """

        product_type = marketconfig.product_type

        profit = []
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

            # calculate profit as income - running_cost from this event
            duration = (end - start) / timedelta(hours=1)
            order_profit = (
                order["accepted_price"][start]
                * order["accepted_volume"][start]
                * duration
            )

            # calculate opportunity cost
            # as the loss of income we have because we are not running at full power
            price_difference = order["price"][start] - marginal_cost

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

            # consideration of start-up costs, which are evenly divided between the
            # upward and downward regulation events
            if (
                unit.outputs[product_type].loc[start] != 0
                and unit.outputs[product_type].loc[start - unit.index.freq] == 0
            ):
                order_profit -= unit.hot_start_cost
            elif (
                unit.outputs[product_type].loc[start] == 0
                and unit.outputs[product_type].loc[start - unit.index.freq] != 0
            ):
                order_profit -= unit.shut_down_cost

            profit.append(order_profit)

            # store results in unit outputs which are written to database by unit operator
            unit.outputs["profit"].loc[start:end_excl] += float(order_profit)
            unit.outputs["regret"].loc[start:end_excl] = float(opportunity_cost)
            unit.outputs["learning_mode"].loc[start:end_excl] = self.learning_mode

        scaling = 0.01 / unit.max_power / len(profit)
        regret_scale = 0.2
        reward = float(sum(profit) - regret_scale * opportunity_cost) * scaling
        unit.outputs["reward"].loc[start:end_excl] = reward
