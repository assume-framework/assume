# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th

from assume.common.base import LearningStrategy, SupportsMinMaxCharge
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.reinforcement_learning.learning_utils import NormalActionNoise
from strategies.utils import load_actor_params

logger = logging.getLogger(__name__)


class EOMBatteryRLStrategy(LearningStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(obs_dim=51, act_dim=2, unique_obs_dim=2, *args,
                         **kwargs)  # need to be 50 and 2, all RL units the same
        self.world = None
        print("Init Battery RL Strategy for", kwargs["unit_id"])
        self.unit_id = kwargs["unit_id"]

        # tells us whether we are training the agents or just executing per-learnind stategies
        self.learning_mode = kwargs.get("learning_mode", False)
        self.perform_evaluation = kwargs.get("perform_evaluation", False)

        self.max_power_charge = kwargs.get("max_power_charge", 100)
        self.max_power_discharge = kwargs.get("max_power_discharge", 90)
        self.efficiency_charge = kwargs.get("efficiency_charge", 0.95)
        self.efficiency_discharge = kwargs.get("efficiency_discharge", 0.95)
        self.min_volume = kwargs.get("min_volume", 1)
        self.max_volume = kwargs.get("max_volume", 190)
        self.variable_cost_charge = kwargs.get("variable_cost_charge", 30)
        self.variable_cost_discharge = kwargs.get("variable_cost_discharge", 30)
        self.natural_inflow = kwargs.get("natural_inflow", 0)

        self.max_bid_price = kwargs.get("max_bid_price", 100)

        self.float_type = th.float

        # sets the devide of the actor network
        device = kwargs.get("device", "cpu")
        self.device = th.device(device if th.cuda.is_available() else "cpu")
        if not self.learning_mode:
            self.device = th.device("cpu")

        # for definition of observation space
        self.foresight = pd.Timedelta(kwargs.get("eom_foresight", "24h"))
        self.foresight_int = kwargs.get("foresight", 24)

        # define used order types
        self.order_types = kwargs.get("order_types", ["SB"])
        if self.learning_mode or self.perform_evaluation:
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
            load_actor_params(self, load_path=(kwargs["trained_policies_save_path"]))
        else:
            raise FileNotFoundError(
                f"No policies were provided for DRL unit {self.unit_id}!. Please provide a valid path to the trained policies."
            )

    def calculate_bids(
            self,
            unit: SupportsMinMaxCharge,
            market_config: MarketConfig,
            product_tuples: list[Product],
            **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market.

        Args:
            unit (SupportsMinMaxCharge): The unit that is dispatched.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): List of product tuples.
            **kwargs: Additional keyword arguments.

        Returns:
            Orderbook: Bids containing start_time, end_time, only_hours, price, volume.

        Note:
            The strategy is analogue to flexABLE
        """
        start = product_tuples[0][0]
        end_all = product_tuples[-1][1]
        next_observation = self.create_observation(
            unit=unit,
            market_id=market_config.market_id,
            start=start,
            end=end_all,
        )
        # =============================================================================
        # Storage Unit is either charging, discharging, or off
        # =============================================================================
        previous_power = unit.get_output_before(start)
        # save a theoretic SOC to calculate the ramping

        actions, noise = self.get_actions(next_observation)
        # =============================================================================
        # 3. Transform Actions into bids
        # =============================================================================
        # actions are in the range [0,1], we need to transform them into actual bids
        # we can use our domain knowledge to guide the bid formulation
        bid_prices = actions * self.max_bid_price

        min_charge, max_charge = unit.calculate_min_max_charge(start, end_all)
        min_discharge, max_discharge = unit.calculate_min_max_discharge(start, end_all)

        bids = []

        for bid in bid_prices:
            if bid > 0:  ## Demand == we have to pay, positive bid
                bids.append({
                    "start_time": start,
                    "end_time": end_all,
                    "only_hours": None,
                    "price": bid,
                    "volume": max_charge[start],  # Charge == Demand, negative Values
                    "node": unit.node,
                })
            if bid < 0:  ## Supply == we get payed, bid need converted to positive price
                bids.append({
                    "start_time": start,
                    "end_time": end_all,
                    "only_hours": None,
                    "price": -bid,
                    "volume": max_discharge[start],  ## Discharge == Supply, positive Values
                    "node": unit.node,
                })
        bids = self.remove_empty_bids(bids)

        # Try to fix inhomogeneous shapes if don't submit min 2 bids
        if len(bids) == 0:
            if max_charge[start] < 0:  # Charging possible, bid with zero to get never accepted
                for i in range(2):
                    bids.append({
                        "start_time": start,
                        "end_time": end_all,
                        "only_hours": None,
                        "price": 0,
                        "volume": max_charge[start] / 2,
                        "node": unit.node,
                    })
            if max_charge[start] >= 0 and max_discharge[start] > 0:  # Charging impossible, discharge the highest price
                for i in range(2):
                    bids.append({
                        "start_time": start,
                        "end_time": end_all,
                        "only_hours": None,
                        "price": self.max_bid_price,
                        "volume": max_discharge[start] / 2,
                        "node": unit.node,
                    })
        bids = self.remove_empty_bids(bids)
        unit.outputs["rl_observations"].append(next_observation)
        unit.outputs["rl_actions"].append(actions)

        # store results in unit outputs as series to be written to the database by the unit operator
        unit.outputs["actions"][start] = actions
        unit.outputs["exploration_noise"][start] = noise

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
        if self.learning_mode and not self.perform_evaluation:
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

    def calculate_reward(
            self,
            unit: SupportsMinMaxCharge,
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

        order = None
        profit = 0
        costs = 0
        marginal_cost = 0
        last_order = None
        duration = 0
        start = None
        end_excl = None
        # iterate over all orders in the orderbook, to calculate order specific profit
        for order in orderbook:
            start = order["start_time"]
            end = order["end_time"]
            end_excl = end - unit.index.freq

            # depending on way the unit calculates marginal costs we take costs
            marginal_cost = unit.calculate_marginal_cost(
                pd.Timestamp(start), unit.outputs[product_type].loc[start]
            )

            duration = (end - start) / timedelta(hours=1)

            marginal_cost += unit.get_starting_costs(int(duration))

            # calculate profit as income - running_cost from this event
            order_profit = order["accepted_price"] * order["accepted_volume"] * duration
            order_cost = marginal_cost * order["accepted_volume"] * duration

            # collect profit and opportunity cost for all orders
            profit += order_profit
            costs += order_cost

        # calculate opportunity cost
        # as the loss of income we have because we are not running at full power
        opportunity_cost = (
                (order["accepted_price"] - marginal_cost)
                * (unit.max_power_charge - unit.outputs[product_type].loc[start:end_excl]).sum()
                * duration
        )

        # if our opportunity costs are negative, we did not miss an opportunity to earn money and we set them to 0
        opportunity_cost = max(opportunity_cost, 0)

        profit = profit - costs

        scaling = 0.1 / unit.max_volume
        regret_scale = 0.2
        reward = float(profit - regret_scale * opportunity_cost) * scaling

        # store results in unit outputs which are written to database by unit operator
        unit.outputs["profit"].loc[start:end_excl] += profit
        unit.outputs["reward"].loc[start:end_excl] = reward
        unit.outputs["regret"].loc[start:end_excl] = opportunity_cost
        unit.outputs["total_costs"].loc[start:end_excl] = costs
        # Cause error if orderbook does not contain 2 bids the same as powerplant
        unit.outputs["rl_rewards"].append(reward)

        if start == datetime(2019, 4, 1, 0, 0):
            print("SOC Printing at the end" + str(np.average(unit.outputs["soc"])))
            pd.Series(unit.outputs["soc"]).to_csv(path_or_buf="./outputs/storage/soc_" + unit.id + ".csv")

    def create_observation(
            self,
            unit: SupportsMinMaxCharge,
            market_id: str,
            start: datetime,
            end: datetime,
    ):

        end_excl = end - unit.index.freq

        # get the forecast length depending on the tme unit considered in the modelled unit
        forecast_len = pd.Timedelta((self.foresight_int - 1) * unit.index.freq)

        # =============================================================================
        # 1.1 Get the Observations, which are the basis of the action decision
        # =============================================================================
        scaling_factor_res_load = self.max_volume

        # price forecast
        scaling_factor_price = 100

        # total capacity and marginal cost
        scaling_factor_total_capacity = unit.max_power_charge

        # marginal cost
        # Obs[2*foresight+1:2*foresight+2]
        scaling_factor_marginal_cost = 100

        # checks if we are at end of simulation horizon, since we need to change the forecast then
        # for residual load and price forecast and scale them
        if (
                end_excl + forecast_len
                > unit.forecaster[f"residual_load_{market_id}"].index[-1]
        ):
            scaled_res_load_forecast = (
                    unit.forecaster[f"residual_load_{market_id}"].loc[start:].values
                    / scaling_factor_res_load
            )
            scaled_res_load_forecast = np.concatenate(
                [
                    scaled_res_load_forecast,
                    unit.forecaster[f"residual_load_{market_id}"].iloc[
                    : self.foresight - len(scaled_res_load_forecast)
                    ],
                ]
            )

        else:
            scaled_res_load_forecast = (
                    unit.forecaster[f"residual_load_{market_id}"]
                    .loc[start: end_excl + forecast_len]
                    .values
                    / scaling_factor_res_load
            )

        if end_excl + forecast_len > unit.forecaster[f"price_{market_id}"].index[-1]:
            scaled_price_forecast = (
                    unit.forecaster[f"price_{market_id}"].loc[start:].values
                    / scaling_factor_price
            )
            scaled_price_forecast = np.concatenate(
                [
                    scaled_price_forecast,
                    unit.forecaster[f"price_{market_id}"].iloc[
                    : self.foresight - len(scaled_price_forecast)
                    ],
                ]
            )

        else:
            scaled_price_forecast = (
                    unit.forecaster[f"price_{market_id}"]
                    .loc[start: end_excl + forecast_len]
                    .values
                    / scaling_factor_price
            )

        # get last accapted bid volume and the current marginal costs of the unit
        current_volume = unit.get_output_before(start)
        current_costs = unit.calculate_marginal_cost(start, current_volume)

        soc = unit.outputs["soc"].loc[start:end_excl]
        soc_scaled = np.average(soc / scaling_factor_res_load)

        # scale unit outpus
        scaled_total_capacity = current_volume / scaling_factor_total_capacity
        scaled_marginal_cost = current_costs / scaling_factor_marginal_cost

        # concat all obsverations into one array
        observation = np.concatenate(
            [
                scaled_res_load_forecast,
                scaled_price_forecast,
                np.array([scaled_total_capacity, scaled_marginal_cost, soc_scaled]),
            ]
        )

        # transfer arry to GPU for NN processing
        observation = (
            th.tensor(observation, dtype=self.float_type)
            .to(self.device, non_blocking=True)
            .view(-1)
        )

        return observation.detach().clone()