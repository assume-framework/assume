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
        self.foresight = kwargs.get("foresight", 24)
        self.current_time = None

        # RL agent parameters
        self.obs_dim = kwargs.get("observation_dimension", 50)
        self.act_dim = kwargs.get("action_dimension", 2)
        self.max_bid_price = kwargs.get("max_bid_price", 100)
        self.max_demand = kwargs.get("max_demand", 10e3)

        device = kwargs.get("device", "cpu")
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
        self.observation = None
        self.next_observation = None

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

        # =============================================================================
        # Calculate bid-prices from action output of RL strategies
        # Calculating possible bid amount
        # artificially force inflex_bid to be the smaller price
        # =============================================================================

        self.next_observation = self.create_observation(
            unit=unit,
            operational_window=operational_window,
            data_dict=data_dict,
        )

        self.curr_action = self.get_actions()

        bid_prices = self.curr_action * self.max_bid_price

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

        self.curr_observation = self.next_observation

        return bids

    def get_actions(self):
        if self.learning_mode:
            if self.collect_initial_experience:
                curr_action = (
                    th.normal(
                        mean=0.0, std=0.2, size=(1, self.act_dim), dtype=self.float_type
                    )
                    .to(self.device)
                    .squeeze()
                )

                # trick that makes the bidding close to marginal cost for exploration purposes
                curr_action += th.tensor(
                    self.next_observation[-1],  # this doesnt work yet
                    device=self.device,
                    dtype=self.float_type,
                )
            else:
                curr_action = self.actor(self.next_observation).detach()
                curr_action += th.tensor(
                    self.action_noise.noise(), device=self.device, dtype=self.float_type
                )
        else:
            #curr_action = self.actor(self.next_observation).detach()
            curr_action = (5, 5)

        curr_action = curr_action.clamp(-1, 1)

        return curr_action

    def create_observation(
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

        # scale unit outpus
        scaled_total_capacity = (
            operational_window["states"]["current_power"]["volume"] / unit.max_power
        )
        scaled_marginal_cost = (
            operational_window["states"]["current_power"]["cost"] / self.max_bid_price
        )

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
        start,
        end_excl,
        product_type,
        clearing_price,
        unit: BaseUnit,
    ):
        
        if not self.is_learning_strategy:
            return
        
        # gets market feedback from set_dispacth
        # based on calculated market success in dispatch we calculate the profit

        unit.total_scaled_capacity[start] = (
            unit.outputs[product_type].loc[start:end_excl] / unit.max_power
        )

            # calculate profit, now based on actual mc considering the power output
        marginal_cost = (
            unit.marginal_cost.loc[start]
            if unit.marginal_cost is not None
            else unit.calc_marginal_cost_with_partial_eff(
                power_output=unit.outputs[product_type].loc[start:end_excl],
                timestep=start,
            )
        )
        price_difference = clearing_price - marginal_cost


        profit = (unit.outputs["cashflow"].loc[start:end_excl] - marginal_cost*unit.outputs[product_type].loc[start:end_excl] * (start - end_excl).total_seconds()
            / 3600).sum()


        opportunity_cost = (
            price_difference
            * (unit.max_power - unit.outputs[product_type].loc[start:end_excl]).sum()
            * (start - end_excl).total_seconds()
            / 3600
        )
        opportunity_cost = max(opportunity_cost, 0)



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

        scaling = 0.1 / unit.max_power
        regret_scale = 0.2

        unit.outputs["rewards"].loc[start:end_excl]=(profit - regret_scale * opportunity_cost) * scaling
        unit.outputs["profit"].loc[start:end_excl]=profit
        unit.outputs["regret"].loc[start:end_excl]=opportunity_cost

        self.curr_reward = self.rewards[start]


