from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch as th

from assume.common.base import LearningStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook, Product
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
        self.foresight = kwargs.get("foresight", 24)

        # RL agent parameters

        self.max_bid_price = kwargs.get("max_bid_price", 100)
        self.max_demand = kwargs.get("max_demand", 10e3)

        device = kwargs.get("device", "cpu")
        self.device = th.device(device)

        float_type = kwargs.get("float_type", "float32")
        self.float_type = th.float if float_type == "float32" else th.float16

        self.learning_mode = kwargs.get("learning_mode", False)

        if self.learning_mode:
            self.learning_role = None
            self.collect_initial_experience = kwargs.get(
                "collect_initial_experience", True
            )

            self.action_noise = NormalActionNoise(
                mu=0.0,
                sigma=kwargs.get("noise_sigma", 0.1),
                action_dimension=self.act_dim,
                scale=kwargs.get("noise_scale", 1.0),
                dt=kwargs.get("noise_dt", 1.0),
            )

        else:
            self.load_actor_params(load_path=kwargs["load_learned_path"])

        self.reset()

    def reset(
        self,
    ):
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
        min_power, max_power = unit.calculate_min_max_power(start, end)
        min_power = min_power[start]
        max_power = max_power[start]

        # =============================================================================
        # Calculate bid-prices from action output of RL strategies
        # Calculating possible bid amount
        # artificially force inflex_bid to be the smaller price
        # =============================================================================

        next_observation = self.create_observation(
            unit=unit,
            start=start,
            end=end,
        )

        actions, noise = self.get_actions(next_observation)
        unit.outputs["rl_actions"][start] = actions
        unit.outputs["rl_exploration_noise"][start] = noise

        bid_prices = actions * self.max_bid_price

        # =============================================================================
        # Powerplant is either on, or is able to turn on
        # Calculating possible bid amount
        # =============================================================================
        bid_quantity_inflex = min_power
        bid_price_inflex = min(bid_prices)

        # Flex-bid price formulation
        bid_quantity_flex = max_power - bid_quantity_inflex
        bid_price_flex = max(bid_prices)

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

        unit.outputs["rl_observations"][start] = next_observation

        return bids

    def get_actions(self, next_observation):
        """
        Get actions

        :param next_observation: Next observation
        :type next_observation: torch.Tensor
        :return: Actions
        :rtype: torch.Tensor
        """
        if self.learning_mode:
            if self.collect_initial_experience:
                noise = (
                    th.normal(
                        mean=0.0, std=0.2, size=(1, self.act_dim), dtype=self.float_type
                    )
                    .to(self.device)
                    .squeeze()
                )

                curr_action = noise + next_observation[-1].clone().detach()
            else:
                curr_action = self.actor(next_observation).detach()
                noise = th.tensor(
                    self.action_noise.noise(), device=self.device, dtype=self.float_type
                )
                curr_action += noise
        else:
            curr_action = self.actor(next_observation).detach()
            noise = tuple([0 for i in range(self.act_dim)])

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

        # in rl_units operator in ASSUME
        # TODO consider that the last forecast_length time steps cant be used
        # TODO enable difference between actual residual load realisation and residual load forecast
        # currently no difference so historical res_demand values can also be taken from res_demand_forecast
        forecast_len = pd.Timedelta(
            (self.foresight - 1) * unit.index.freq
        )  # in metric of market

        if end_excl + forecast_len > unit.forecaster["residual_load_EOM"].index[-1]:
            scaled_res_load_forecast = (
                unit.forecaster["residual_load_EOM"].loc[start:].values
                / self.max_demand
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
                / self.max_demand
            )

        if end_excl + forecast_len > unit.forecaster["price_EOM"].index[-1]:
            scaled_price_forecast = (
                unit.forecaster["price_EOM"].loc[start:].values / self.max_bid_price
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
                / self.max_bid_price
            )

        current_volume = unit.get_output_before(start)
        current_price = unit.calc_marginal_cost_with_partial_eff(current_volume, start)
        # scale unit outpus
        scaled_total_capacity = current_volume / unit.max_power
        scaled_marginal_cost = current_price / self.max_bid_price

        observation = np.concatenate(
            [
                scaled_res_load_forecast,
                scaled_price_forecast,
                np.array([scaled_total_capacity, scaled_marginal_cost]),
            ]
        )

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
        product_type = marketconfig.product_type
        scaling = 0.1 / unit.max_power
        regret_scale = 0.2
        profit = 0
        reward = 0
        opportunity_cost = 0

        for order in orderbook:
            start = order["start_time"]
            end = order["end_time"]
            end_excl = end - unit.index.freq
            # gets market feedback from set_dispatch
            # based on calculated market success in dispatch we calculate the profit
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
            # calculate profit, now based on actual mc considering the power output
            price_difference = order["price"] - marginal_cost
            duration = (end - start) / timedelta(hours=1)
            # calculate profit as income - running_cost from this event
            order_profit = order["price"] * order["accepted_volume"] * duration

            # calculate opportunity cost as the loss of income we have because we are not running at full power
            order_opportunity_cost = (
                price_difference
                * (
                    unit.max_power - unit.outputs[product_type].loc[start:end_excl]
                ).sum()
                * duration
            )
            # opportunity cost must be positive
            order_opportunity_cost = max(order_opportunity_cost, 0)
            opportunity_cost += order_opportunity_cost

            # sum up order profit
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

        reward = float(profit - regret_scale * opportunity_cost) * scaling
        unit.outputs["profit"].loc[start:end_excl] += float(profit)
        # if multi market rl is used, the reward should be added up for all market results
        unit.outputs["reward"].loc[start:end_excl] = reward
        unit.outputs["regret"].loc[start:end_excl] = float(opportunity_cost)

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
