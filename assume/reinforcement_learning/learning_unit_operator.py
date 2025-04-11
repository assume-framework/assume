# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import numpy as np
import torch as th

from assume.common import UnitsOperator
from assume.common.market_objects import (
    MarketConfig,
    Orderbook,
)
from assume.common.utils import convert_tensors, create_rrule, get_products_index
from assume.strategies import BaseStrategy, LearningStrategy
from assume.units import BaseUnit

logger = logging.getLogger(__name__)


class RLUnitsOperator(UnitsOperator):
    def __init__(
        self,
        available_markets: list[MarketConfig],
        opt_portfolio: tuple[bool, BaseStrategy] | None = None,
    ):
        super().__init__(available_markets, opt_portfolio)

        self.rl_units = []
        self.learning_strategies = {
            "obs_dim": 0,
            "act_dim": 0,
            "device": "cpu",
        }

    def on_ready(self):
        super().on_ready()

        # todo
        recurrency_task = create_rrule(
            start=self.context.data["train_start"],
            end=self.context.data["train_end"],
            freq=self.context.data.get("train_freq", "24h"),
        )

        self.context.schedule_recurrent_task(
            self.write_to_learning_role, recurrency_task
        )

    def add_unit(
        self,
        unit: BaseUnit,
    ) -> None:
        """
        Create a unit.

        Args:
            unit (BaseUnit): The unit to be added.
        """
        self.units[unit.id] = unit

        # check if unit has learning strategy for any of the available markets
        for market in self.available_markets:
            strategy = unit.bidding_strategies.get(market.market_id)

            if isinstance(strategy, LearningStrategy):
                self.learning_strategies.update(
                    {
                        "obs_dim": strategy.obs_dim,
                        "act_dim": strategy.act_dim,
                        "device": strategy.device,
                    }
                )

                self.rl_units.append(unit)
                break

    def write_learning_to_output(self, orderbook: Orderbook, market_id: str) -> None:
        """
        Sends the current rl_strategy update to the output agent.

        Args:
            products_index (pandas.DatetimeIndex): The index of all products.
            marketconfig (MarketConfig): The market configuration.
        """

        products_index = get_products_index(orderbook)

        # should write learning results if at least one bidding_strategy is a learning strategy
        if not (len(self.rl_units) and orderbook):
            return

        output_agent_list = []
        start = products_index[0]

        for unit in self.rl_units:
            strategy = unit.bidding_strategies.get(market_id)

            # rl only for energy market for now!
            if isinstance(strategy, LearningStrategy):
                output_dict = {
                    "datetime": start,
                    "unit": unit.id,
                }

                output_dict.update(
                    {
                        "profit": unit.outputs["profit"].at[start],
                        "reward": unit.outputs["reward"].at[start],
                        "regret": unit.outputs["regret"].at[start],
                    }
                )

                action_tuple = unit.outputs["actions"].at[start]
                noise_tuple = unit.outputs["exploration_noise"].at[start]
                action_dim = action_tuple.numel()

                for i in range(action_dim):
                    output_dict[f"exploration_noise_{i}"] = noise_tuple[i]
                    output_dict[f"actions_{i}"] = action_tuple[i]

                output_agent_list.append(output_dict)

        db_addr = self.context.data.get("learning_output_agent_addr")

        if db_addr and output_agent_list:
            self.context.schedule_instant_message(
                receiver_addr=db_addr,
                content={
                    "context": "write_results",
                    "type": "rl_params",
                    "data": output_agent_list,
                },
            )

    async def write_to_learning_role(
        self,
    ) -> None:
        """
        Writes learning results to the learning agent.

        """
        if len(self.rl_units) == 0:
            return

        obs_dim = self.learning_strategies["obs_dim"]
        act_dim = self.learning_strategies["act_dim"]
        device = self.learning_strategies["device"]

        learning_unit_count = len(self.rl_units)

        # Collect the number of reward values for each unit.
        # This represents how many complete transitions we have for each unit.
        # Using a set ensures we capture only unique lengths across all units.
        values_len_set = {len(unit.outputs["rl_rewards"]) for unit in self.rl_units}

        # Check if all units have the same number of reward values.
        # If the set contains more than one unique length, it means at least one unit
        # has a different number of rewards, indicating an inconsistency.
        # This is considered an error condition, so we raise an exception.
        if len(values_len_set) > 1:
            raise ValueError(
                "Mismatch in reward value lengths: All units must have the same number of rewards."
            )

        # Since all units have the same length, extract the common length
        values_len = values_len_set.pop()

        # return if no data is available
        if values_len == 0:
            return

        all_observations = th.zeros(
            (values_len, learning_unit_count, obs_dim), device=device
        )
        all_actions = th.zeros(
            (values_len, learning_unit_count, act_dim), device=device
        )
        all_rewards = []

        # Iterate through each RL unit and collect all of their observations, actions, and rewards
        # making it dependent on values_len ensures that data is not stored away for which the reward was not calculated yet
        for i, unit in enumerate(self.rl_units):
            # Convert pandas Series to torch Tensor
            obs_tensor = th.stack(unit.outputs["rl_observations"][:values_len], dim=0)
            actions_tensor = th.stack(unit.outputs["rl_actions"][:values_len], dim=0)

            all_observations[:, i, :] = obs_tensor
            all_actions[:, i, :] = actions_tensor
            all_rewards.append(unit.outputs["rl_rewards"])

            # reset the outputs
            unit.reset_saved_rl_data()

        all_observations = all_observations.numpy(force=True)
        all_actions = all_actions.numpy(force=True)

        all_rewards = np.array(all_rewards).T

        rl_agent_data = (all_observations, all_actions, all_rewards)

        learning_role_addr = self.context.data.get("learning_agent_addr")

        if learning_role_addr:
            self.context.schedule_instant_message(
                content={
                    "context": "rl_training",
                    "type": "save_buffer_and_update",
                    "data": rl_agent_data,
                },
                receiver_addr=learning_role_addr,
            )

    async def formulate_bids(
        self, market: MarketConfig, products: list[tuple]
    ) -> Orderbook:
        """
        Formulates the bid to the market according to the bidding strategy of the each unit individually.

        Args:
            market (MarketConfig): The market to formulate bids for.
            products (list[tuple]): The products to formulate bids for.

        Returns:
            OrderBook: The orderbook that is submitted as a bid to the market.
        """

        orderbook: Orderbook = []

        for unit_id, unit in self.units.items():
            product_bids = unit.calculate_bids(
                market,
                product_tuples=products,
            )
            for i, order in enumerate(product_bids):
                order["agent_addr"] = self.context.addr

                if market.volume_tick:
                    order["volume"] = round(order["volume"] / market.volume_tick)
                if market.price_tick:
                    order["price"] = round(order["price"] / market.price_tick)
                if "bid_id" not in order.keys() or order["bid_id"] is None:
                    order["bid_id"] = f"{unit_id}_{i+1}"
                order["unit_id"] = unit_id
                orderbook.append(order)

        # Convert all CUDA tensors to CPU in one pass
        return convert_tensors(orderbook)
