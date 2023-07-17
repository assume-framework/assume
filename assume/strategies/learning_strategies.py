import numpy as np
import pandas as pd
import torch as th

from assume.common.learning_utils import Actor, NormalActionNoise
from assume.common.market_objects import MarketConfig
from assume.strategies.base_strategy import BaseStrategy, OperationalWindow
from assume.units.base_unit import BaseUnit


class RLStrategy(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_learning_strategy = True
        self.foresight = pd.Timedelta("12h")
        self.current_time = None

        # RL agent parameters
        self.obs_dim = kwargs.get("observation_dimension", 52)
        self.act_dim = kwargs.get("action_dimension", 2)
        self.max_bid_price = kwargs.get("max_bid_price", 100)
        self.max_demand = kwargs.get("max_demand", 10e3)

        device = kwargs.get("device", "cuda:0")
        self.device = th.device(device)

        float_type = kwargs.get("float_type", "float32")
        self.float_type = th.float if float_type == "float32" else th.float16

        self.actor = Actor(self.obs_dim, self.act_dim, self.float_type).to(self.device)

        self.learning_mode = kwargs.get("learning_mode", False)

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
        self.next_obs = None

        self.curr_action = None
        self.curr_reward = None
        self.curr_experience = []

    def calculate_bids(
        self,
        unit: BaseUnit,
        operational_window: OperationalWindow,
        data_dict: dict,
        **kwargs,
    ):
        bid_quantity_inflex, bid_price_inflex = 0, 0
        bid_quantity_flex, bid_price_flex = 0, 0

        if operational_window is not None:
            # =============================================================================
            # Calculate bid-prices from action output of RL strategies
            # Calculating possible bid amount
            # artificially force inflex_bid to be the smaller price
            # =============================================================================

            bid_prices = self.get_actions(
                unit=unit,
                operational_window=operational_window,
                data_dict=data_dict,
            )
            bid_prices *= self.max_bid_price

            bid_price_inflex = min(bid_prices)
            bid_price_flex = max(bid_prices)

            # =============================================================================
            # Powerplant is either on, or is able to turn on
            # Calculating possible bid amount
            # =============================================================================
            bid_quantity_inflex = operational_window["states"]["min_power"]["volume"]
            bid_price_inflex = min(bid_prices)

            # Flex-bid price formulation
            if unit.current_status:
                bid_quantity_flex = (
                    operational_window["states"]["max_power"]["volume"]
                    - bid_quantity_inflex
                )
                bid_price_flex = max(bid_prices)

        bids = [
            {"price": bid_price_inflex, "volume": bid_quantity_inflex},
            {"price": bid_price_flex, "volume": bid_quantity_flex},
        ]

        return bids

    def get_actions(self, unit, operational_window, data_dict):
        self.curr_obs = self.create_obs(
            unit=unit,
            operational_window=operational_window,
            data_dict=data_dict,
        )

        if self.learning_mode:
            if self.collect_initial_experience:
                self.curr_action = (
                    th.normal(
                        mean=0.0, std=0.2, size=(1, self.act_dim), dtype=self.float_type
                    )
                    .to(self.device)
                    .squeeze()
                )

                # trick that makes the bidding close to marginal cost for exploration purposes
                self.curr_action += th.tensor(
                    self.curr_obs[-1],  # this doesnt work yet
                    device=self.device,
                    dtype=self.float_type,
                )
            else:
                self.curr_action = self.actor(self.curr_obs).detach()
                self.curr_action += th.tensor(
                    self.action_noise.noise(), device=self.device, dtype=self.float_type
                )
        else:
            self.curr_action = self.actor(self.curr_obs).detach()

        self.curr_action = self.curr_action.clamp(-1, 1)

        return self.curr_action

    def create_obs(
        self,
        unit,
        operational_window,
        data_dict,
    ):
        start = operational_window["window"][0]
        end_excl = operational_window["window"][1] - unit.index.freq


        # in rl_units operator in ASSUME
        # TODO consider that the last forecast_length time steps cant be used
        # TODO enable difference between actual residual load realisation and residual load forecast
        # currently no difference so historical res_demand values can also be taken from res_demand_forecast
        forecast_len = pd.Timedelta('24h')  # in metric of market

        scaled_res_load_forecast = data_dict["residual_load_forecast"].loc[start : end_excl + forecast_len].values / self.max_demand

        scaled_price_forecast = data_dict["price_forecast"].loc[start : end_excl + forecast_len].values / self.max_bid_price

        # scale unit outpus
        scaled_total_capacity = operational_window["states"]["current_power"]["volume"] / unit.max_power
        scaled_marginal_cost = (
            operational_window["states"]["current_power"]["cost"] / self.max_bid_price
        )

        obs = np.concatenate(
            [
                scaled_res_load_forecast,
                scaled_price_forecast,
                np.array([scaled_total_capacity, scaled_marginal_cost]),
            ]
        )

        obs = (
            th.tensor(obs, dtype=self.float_type)
            .to(self.device, non_blocking=True)
            .view(-1)
        )

        return obs.detach().clone()

    def calculate_reward(
        self,
        start,
        end_excl,
        product_type,
        clearing_price,
        unit: BaseUnit = None,
    ):
        # gets market feedback from set_dispacth

        # Calculates market success
        # first for sold capacity
        if unit.outputs[product_type].loc[start:end_excl] < unit.min_power:
            unit.outputs[product_type].loc[start:end_excl] = 0

        unit.total_scaled_capacity[start] = (
            unit.outputs[product_type].loc[start:end_excl] / unit.max_power
        )

        # calculate profit, now based on actual mc considering the power output
        price_difference = clearing_price - unit.calc_marginal_cost_with_partial_eff(
            power_output=unit.outputs[product_type].loc[start:end_excl], timestep=start
        )
        profit = (
            price_difference
            * unit.outputs[product_type].loc[start:end_excl]
            * (start - end_excl).total_seconds()
            / 3600
        )
        opportunity_cost = (
            price_difference
            * (unit.max_power - unit.outputs[product_type].loc[start:end_excl])
            * (start - end_excl).total_seconds()
            / 3600
        )
        opportunity_cost = max(opportunity_cost, 0)

        scaling = 0.1 / unit.max_power
        regret_scale = 0.2

        if (
            unit.outputs[product_type].loc[start:end_excl] != 0
            and unit.outputs[product_type].loc[start - unit.index.freq : start] == 0
        ):
            profit = profit - unit.hot_start_cost / 2
        elif (
            unit.outputs[product_type].loc[start:end_excl] == 0
            and unit.outputs[product_type].loc[start - unit.index.freq : start] != 0
        ):
            profit = profit - unit.hot_start_cost / 2

        self.rewards[start] = (profit - regret_scale * opportunity_cost) * scaling
        self.profits[start] = profit
        self.regrets[start] = opportunity_cost

        self.curr_reward = self.rewards[start]

        # TODO I do not have next obs yet, since I get it in calculate rewards
        self.curr_experience = [
            self.curr_obs,
            self.next_obs,
            self.curr_action,
            self.curr_reward,
        ]
