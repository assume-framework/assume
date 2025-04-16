# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime

import numpy as np
import torch as th

from assume.common.base import SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.common.utils import get_products_index
from assume.strategies.learning_strategies import RLStrategy


class RLAdvancedOrderStrategy(RLStrategy):
    """
    Reinforcement Learning Strategy for an Energy-Only-Market with simple hourly, block and linked orders.

    Attributes:
        foresight (int): Number of time steps to look ahead. Default 24.
        max_bid_price (float): The maximum bid price.
        max_demand (float): The maximum demand.
        device (str): The device to run on.
        float_type (str): The float type to use.
        learning_mode (bool): Whether to use learning mode.
        algorithm (str): RL algorithm. Defaults to "matd3".
        actor_architecture_class (type[torch.nn.Module]): Actor network class. Defaults to "MLPActor".
        actor (torch.nn.Module): The actor network.
        order_types (list[str]): The list of order types to use (SB, LB, BB).
        episodes_collecting_initial_experience (int): Number of episodes to collect initial experience.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Note:
        This strategy is based on the strategy in flexable.py, but uses the actor network to determine the prices
        instead of using the marginal costs as in flexable.py.
        The two prices for flexible and inflexible power are determined by the actor network,
        which is trained with the MATD3 algorithm.
        The maximum of those two prices is used for the flexible bid and the minimum for the inflexible bid.
        The order type is set implicitly (not by the RL agent itself) and the structure depends on
        the allowed order types:
        If only simple hourly orders (SB) are allowed, the strategy will only use SB for both inflexible and flexible power.
        If SB and linked orders (LB) are allowed, the strategy will use SB for the inflexible power and LB for the flexible power.
        If SB and block orders (BB) are allowed, the strategy will use BB for the inflexible power and SB for the flexible power.
        If all three order types (SB, BB, LB) are allowed, the strategy will use BB for the inflexible power
        and LB for the flexible power, except the inflexible power is 0,
        then it will use SB for the flexible power (as for VREs).
    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Calculates bids for a unit.

        Args:
            unit (SupportsMinMax): Unit to calculate bids for
            market_config (MarketConfig): Market configuration
            product_tuples (list[Product]): Product tuples

        Returns:
            Orderbook: Bids containing start time, end time, price, volume and bid type

        """

        start = product_tuples[0][0]
        end = product_tuples[-1][1]

        # =============================================================================
        # 1. Get the Observations, which are the basis of the action decision
        # =============================================================================
        next_observation = self.create_observation(
            unit=unit,
            market_id=market_config.market_id,
            start=start,
        )

        # =============================================================================
        # 2. Get the Actions, based on the observations
        # =============================================================================
        actions, noise = self.get_actions(next_observation)

        # =============================================================================
        # 3. Transform Actions into bids
        # =============================================================================
        # actions are in the range [-1,1], we need to transform them into actual bids
        # we can use our domain knowledge to guide the bid formulation

        bid_price_1 = actions[0].item() * self.max_bid_price
        bid_price_2 = actions[1].item() * self.max_bid_price

        # use the higher price for the flexible bid and the lower price for the inflexible bid
        bid_price_inflex = min(bid_price_1, bid_price_2)
        bid_price_flex = max(bid_price_1, bid_price_2)

        op_time = unit.get_operation_time(start)

        previous_power = unit.get_output_before(start)
        min_power_values, max_power_values = unit.calculate_min_max_power(start, end)

        # calculate the quantities and transform the bids into orderbook format
        bids = []
        bid_quantity_block = {}
        for product, min_power, max_power in zip(
            product_tuples, min_power_values, max_power_values
        ):
            start = product[0]
            end = product[1]

            bid_quantity_inflex = 0
            bid_quantity_flex = 0

            current_power = unit.outputs["energy"].at[start]

            # get technical bounds for the unit output from the unit
            # adjust for ramp speed
            max_power = unit.calculate_ramp(
                op_time, previous_power, max_power, current_power
            )
            # adjust for ramp speed
            min_power = unit.calculate_ramp(
                op_time, previous_power, min_power, current_power
            )

            # 3.1 formulate the bids for Pmin
            bid_quantity_inflex = min_power

            # 3.1 formulate the bids for Pmax - Pmin
            # Pmin, the minimum run capacity is the inflexible part of the bid, which should always be accepted

            if op_time <= -unit.min_down_time or op_time > 0:
                bid_quantity_flex = max_power - bid_quantity_inflex

            if "BB" in self.order_types:
                bid_quantity_block[start] = bid_quantity_inflex

            # if no BB in order_types, then add the inflex bid as SB
            if "BB" not in self.order_types:
                bids.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": None,
                        "price": bid_price_inflex,
                        "volume": bid_quantity_inflex,
                        "bid_type": "SB",
                        "bid_id": f"{unit.id}_SB_{len(bids) + 1}",
                        "node": unit.node,
                    },
                )

            # actually formulate bids in orderbook format
            # if LB is in order_types, then formulate the linked bid depending on the block bid or simple bid
            if "LB" in self.order_types and bid_quantity_inflex != 0:
                if "BB" in self.order_types:
                    parent_bid_id = unit.id + "_block"
                else:
                    parent_bid_id = f"{unit.id}_{len(bids)}"
                bids.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": None,
                        "price": bid_price_flex,
                        "volume": {start: bid_quantity_flex},
                        "bid_type": "LB",
                        "parent_bid_id": parent_bid_id,
                        "bid_id": f"{unit.id}_LB_{len(bids) + 1}",
                        "node": unit.node,
                    },
                )
            # otherwise just add the flex bid as SB
            else:
                bids.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": None,
                        "price": bid_price_flex,
                        "volume": bid_quantity_flex,
                        "bid_type": "SB",
                        "bid_id": f"{unit.id}_SB_{len(bids) + 1}",
                        "node": unit.node,
                    },
                )

            # calculate previous power with planned dispatch (bid_quantity)
            previous_power = bid_quantity_inflex + bid_quantity_flex + current_power
            op_time = max(op_time, 0) + 1 if previous_power > 0 else min(op_time, 0) - 1

        if "BB" in self.order_types:
            bids.append(
                {
                    "start_time": product_tuples[0][0],
                    "end_time": product_tuples[-1][1],
                    "only_hours": product_tuples[0][2],
                    "price": bid_price_inflex,
                    "volume": bid_quantity_block,
                    "bid_type": "BB",
                    "min_acceptance_ratio": 1,
                    "bid_id": unit.id + "_block",
                    "node": unit.node,
                }
            )

        # store results in unit outputs as lists to be written to the buffer for learning
        unit.outputs["rl_observations"].append(next_observation)
        unit.outputs["rl_actions"].append(actions)

        # store results in unit outputs as series to be written to the database by the unit operator
        unit.outputs["actions"].at[start] = actions
        unit.outputs["exploration_noise"].at[start] = noise

        bids = self.remove_empty_bids(bids)

        return bids

    def create_observation(
        self,
        unit: SupportsMinMax,
        market_id: str,
        start: datetime,
    ):
        """
        Create observation.

        It is important to keep in mind, that the DRL method and the centralized critic relies on
        unique observation of individual units. The algorithm is designed in such a way, that
        the unique observations are always placed at the end of the observation space. Please follow this
        convention when adding new observations.

        Args:
            unit (SupportsMinMax): Unit to create observation for
            start (datetime.datetime): Start time
            end (datetime.datetime): End time

        Returns:
            Observation (torch.Tensor): Observation containing residual load forecast, price forecast, must run time, max power and marginal cost

        Note:
            The dimension of the observation space is defined by
            2 * product_len (int): number of hours in the clearing horizon
            + 2 * (foresight-1) (int): number of hours we look ahead
            + 3 (int): must run time, max power and marginal cost
            The observation space is scaled to the range [-1,1] to make it easier for the actor neuronal net to learn.
            The scaling factors are defined by the maximum residual load, the maximum bid price
            and the maximum capacity of the unit.
        """

        # check if scaled observations are already available and if not prepare them
        if not hasattr(self, "scaled_res_load_obs") or not hasattr(
            self, "scaled_prices_obs"
        ):
            self.prepare_observations(unit, market_id)

        # get the forecast length depending on the time unit considered in the modelled unit
        forecast_len = (self.foresight - 1) * unit.index.freq

        # =============================================================================
        # 1.1 Get the Observations, which are the basis of the action decision
        # =============================================================================

        # checks if we are at end of simulation horizon, since we need to change the forecast then
        # for residual load and price forecast and scale them
        if start + forecast_len > self.scaled_res_load_obs.index[-1]:
            scaled_res_load_forecast = self.scaled_res_load_obs.loc[start:]

            scaled_res_load_forecast = np.concatenate(
                [
                    scaled_res_load_forecast,
                    self.scaled_res_load_obs.iloc[
                        : self.foresight - len(scaled_res_load_forecast)
                    ],
                ]
            )

        else:
            scaled_res_load_forecast = self.scaled_res_load_obs.loc[
                start : start + forecast_len
            ]

        if start + forecast_len > self.scaled_prices_obs.index[-1]:
            scaled_price_forecast = self.scaled_prices_obs.loc[start:]
            scaled_price_forecast = np.concatenate(
                [
                    scaled_price_forecast,
                    self.scaled_prices_obs.iloc[
                        : self.foresight - len(scaled_price_forecast)
                    ],
                ]
            )

        else:
            scaled_price_forecast = self.scaled_prices_obs.loc[
                start : start + forecast_len
            ]

        # get last accepted bid volume and the current marginal costs of the unit
        current_volume = unit.get_output_before(start)
        current_costs = unit.calculate_marginal_cost(start, current_volume)

        # scale unit outputs
        # if unit is not running, total dispatch is 0
        scaled_total_dispatch = current_volume / unit.max_power

        # marginal cost
        scaled_marginal_cost = current_costs / self.max_bid_price

        # calculate the time the unit has to continue to run or be down
        op_time = unit.get_operation_time(start)
        if op_time > 0:
            must_run_time = max(op_time - unit.min_operating_time, 0)
        elif op_time < 0:
            must_run_time = min(op_time + unit.min_down_time, 0)
        scaling_factor_must_run_time = max(unit.min_operating_time, unit.min_down_time)

        scaled_must_run_time = must_run_time / scaling_factor_must_run_time

        # concat all obsverations into one array
        observation = np.concatenate(
            [
                scaled_res_load_forecast,
                scaled_price_forecast,
                np.array(
                    [scaled_must_run_time, scaled_total_dispatch, scaled_marginal_cost]
                ),
            ]
        )

        # transfer array to GPU for NN processing
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
        Calculate and write reward, profit and regret to unit outputs.

        Args:
            unit (SupportsMinMax): The unit to calculate reward for.
            marketconfig (MarketConfig): The market configuration.
            orderbook (Orderbook): The Orderbook.

        Note:
            The reward is calculated as the profit minus the opportunity cost,
            which is the loss of income we have because we are not running at full power.
            The regret is the opportunity cost.
            Because the regret_scale is set to 0 the reward equals the profit.
            The profit is the income we have from the accepted bids.
            The total costs are the running costs and the start-up costs.

        """

        product_type = marketconfig.product_type
        products_index = get_products_index(orderbook)

        max_power_values = (
            unit.forecaster.get_availability(unit.id)[products_index] * unit.max_power
        )

        # Initialize intermediate results as numpy arrays for better performance
        profit = np.zeros(len(products_index))
        reward = np.zeros(len(products_index))
        opportunity_cost = np.zeros(len(products_index))
        costs = np.zeros(len(products_index))

        # Map products_index to their positions for faster updates
        index_map = {time: i for i, time in enumerate(products_index)}

        for order in orderbook:
            start = order["start_time"]
            end_excl = order["end_time"] - unit.index.freq

            order_times = unit.index[start:end_excl]
            accepted_volume = order.get("accepted_volume", 0)
            accepted_price = order.get("accepted_price", 0)

            for start, max_power in zip(order_times, max_power_values):
                idx = index_map.get(start)

                marginal_cost = unit.calculate_marginal_cost(
                    start, unit.outputs[product_type].at[start]
                )

                if isinstance(accepted_volume, dict):
                    accepted_volume = accepted_volume.get(start, 0)
                else:
                    accepted_volume = accepted_volume

                if isinstance(accepted_price, dict):
                    accepted_price = accepted_price.get(start, 0)
                else:
                    accepted_price = accepted_price

                price_difference = accepted_price - marginal_cost

                # calculate opportunity cost
                # as the loss of income we have because we are not running at full power
                order_opportunity_cost = price_difference * (
                    max_power - unit.outputs[product_type].at[start]
                )
                # if our opportunity costs are negative, we did not miss an opportunity to earn money and we set them to 0
                # don't consider opportunity_cost more than once! Always the same for one timestep and one market
                opportunity_cost[idx] = max(order_opportunity_cost, 0)
                profit[idx] += accepted_price * accepted_volume

        # consideration of start-up costs
        for i, start in enumerate(products_index):
            op_time = unit.get_operation_time(start)

            output = unit.outputs[product_type].at[start]
            marginal_cost = unit.calculate_marginal_cost(start, output)
            costs[i] += marginal_cost * output

            if output != 0 and op_time < 0:
                start_up_cost = unit.get_starting_costs(op_time)
                costs[i] += start_up_cost

        profit -= costs
        scaling = 0.1 / unit.max_power
        regret_scale = 0.2
        reward = (profit - regret_scale * opportunity_cost) * scaling

        # store results in unit outputs which are written to database by unit operator
        unit.outputs["profit"].loc[products_index] = profit
        unit.outputs["reward"].loc[products_index] = reward
        unit.outputs["regret"].loc[products_index] = opportunity_cost
        unit.outputs["total_costs"].loc[products_index] = costs

        unit.outputs["rl_reward"].append(reward)
