from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch as th

from assume.common.base import LearningStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.reinforcement_learning.learning_utils import Actor, NormalActionNoise


class RLStrategy(LearningStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.foresight = kwargs.get("foresight", 24)

        # RL agent parameters

        self.max_bid_price = kwargs.get("max_bid_price", 100)
        self.max_demand = kwargs.get("max_demand", 10e3)

        device = kwargs.get("device", "cpu")
        self.device = th.device(device)

        float_type = kwargs.get("float_type", "float32")
        self.float_type = th.float if float_type == "float32" else th.float16

        self.learning_mode = kwargs.get("learning_mode", False)

        self.actor = Actor(self.obs_dim, self.act_dim, self.float_type).to(self.device)

        if self.learning_mode:
            self.learning_role = None
            self.collect_initial_experience = kwargs.get(
                "collect_initial_experience", False
            )

            self.actor_target = Actor(self.obs_dim, self.act_dim, self.float_type).to(
                self.device
            )
            self.actor_target.load_state_dict(self.actor.state_dict())
            # Target networks should always be in eval mode
            self.actor_target.train(mode=False)

            self.action_noise = NormalActionNoise(
                mu=0.0,
                sigma=kwargs.get("noise_sigma", 0.2),
                action_dimension=self.act_dim,
                scale=kwargs.get("noise_scale", 1.0),
                dt=kwargs.get("noise_dt", 1.0),
            )

        else:
            self.actor_target = None

        if kwargs.get("load_learned_strategies", False):
            self.load_params(kwargs["load_learned_path"])

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
        data_dict: dict,
        **kwargs,
    ) -> Orderbook:
        bid_quantity_inflex, bid_price_inflex = 0, 0
        bid_quantity_flex, bid_price_flex = 0, 0

        start = product_tuples[0][0]
        end = product_tuples[0][1]
        min_power, max_power = unit.calculate_min_max_power(start, end)
        min_power = min_power[start]
        max_power = max_power[start]

        # needed so that type of series is object
        unit.outputs["rl_actions"][start] = {}
        unit.outputs["rl_observations"][start] = {}
        unit.outputs["rl_rewards"][start] = {}

        # =============================================================================
        # Calculate bid-prices from action output of RL strategies
        # Calculating possible bid amount
        # artificially force inflex_bid to be the smaller price
        # =============================================================================

        next_observation = self.create_observation(
            unit=unit,
            start=start,
            end=end,
            data_dict=data_dict,
        )

        actions = self.get_actions(next_observation)
        unit.outputs["rl_actions"][start] = actions

        bid_prices = actions * self.max_bid_price

        # =============================================================================
        # Powerplant is either on, or is able to turn on
        # Calculating possible bid amount
        # =============================================================================
        bid_quantity_inflex = min_power
        bid_price_inflex = min(bid_prices)

        # Flex-bid price formulation
        if unit.current_status:
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
        if self.learning_mode:
            if self.collect_initial_experience:
                curr_action = (
                    th.normal(
                        mean=0.0, std=0.2, size=(1, self.act_dim), dtype=self.float_type
                    )
                    .to(self.device)
                    .squeeze()
                )

                curr_action += th.tensor(
                    next_observation[-1],
                    device=self.device,
                    dtype=self.float_type,
                )
            else:
                curr_action = self.actor(next_observation).detach()
                curr_action += th.tensor(
                    self.action_noise.noise(), device=self.device, dtype=self.float_type
                )
        else:
            curr_action = self.actor(next_observation).detach()

        curr_action = curr_action.clamp(-1, 1)

        return curr_action

    def create_observation(
        self,
        unit: SupportsMinMax,
        start: datetime,
        end: datetime,
        data_dict: dict,
    ):
        end_excl = end - unit.index.freq

        # in rl_units operator in ASSUME
        # TODO consider that the last forecast_length time steps cant be used
        # TODO enable difference between actual residual load realisation and residual load forecast
        # currently no difference so historical res_demand values can also be taken from res_demand_forecast
        forecast_len = pd.Timedelta(
            (self.foresight - 1) * unit.index.freq
        )  # in metric of market

        if end_excl + forecast_len > data_dict["residual_load_forecast"].index[-1]:
            scaled_res_load_forecast = (
                data_dict["residual_load_forecast"].loc[start:].values / self.max_demand
            )
            scaled_res_load_forecast = np.concatenate(
                [
                    scaled_res_load_forecast,
                    data_dict["residual_load_forecast"].iloc[
                        : self.foresight - len(scaled_res_load_forecast)
                    ],
                ]
            )

        else:
            scaled_res_load_forecast = (
                data_dict["residual_load_forecast"]
                .loc[start : end_excl + forecast_len]
                .values
                / self.max_demand
            )

        if end_excl + forecast_len > data_dict["price_forecast"].index[-1]:
            scaled_price_forecast = (
                data_dict["price_forecast"].loc[start:].values / self.max_bid_price
            )
            scaled_price_forecast = np.concatenate(
                [
                    scaled_price_forecast,
                    data_dict["price_forecast"].iloc[
                        : self.foresight - len(scaled_price_forecast)
                    ],
                ]
            )

        else:
            scaled_price_forecast = (
                data_dict["price_forecast"].loc[start : end_excl + forecast_len].values
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
        product_type = marketconfig.product_type
        scaling = 0.1 / unit.max_power
        regret_scale = 0.2
        profit = 0
        reward = 0

        for order in orderbook:
            start = order["start_time"]
            end = order["end_time"]
            end_excl = end - unit.index.freq
            # gets market feedback from set_dispatch
            # based on calculated market success in dispatch we calculate the profit
            if unit.marginal_cost is not None:
                marginal_cost = (
                    unit.marginal_cost[start]
                    if isinstance(unit.marginal_cost, dict)
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
            order_profit = order["price"] * order["volume"] * duration

            # calculate opportunity cost as the loss of income we have because we are not running at full power
            opportunity_cost = (
                price_difference
                * (
                    unit.max_power - unit.outputs[product_type].loc[start:end_excl]
                ).sum()
                * duration
            )
            # opportunity cost must be positive
            opportunity_cost = max(opportunity_cost, 0)

            profit += order_profit
            # reward and opportunity cost does not work well for multiple biddings on the same timeframe
            reward = (profit - regret_scale * opportunity_cost) * scaling
            unit.outputs["profit"].loc[start:end_excl] += float(profit)
            unit.outputs["reward"].loc[start:end_excl] = float(reward)
            unit.outputs["regret"].loc[start:end_excl] = float(opportunity_cost)

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

        reward = (profit - regret_scale * opportunity_cost) * scaling
        unit.outputs["rl_rewards"][start] = reward
