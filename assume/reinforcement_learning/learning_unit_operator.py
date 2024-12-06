# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import numpy as np
import torch as th

from assume.common import UnitsOperator
from assume.common.market_objects import (
    ClearingMessage,
    MarketConfig,
    MetaDict,
    Orderbook,
)
from assume.common.utils import create_rrule, get_products_index
from assume.strategies import BaseStrategy, LearningStrategy, RLAdvancedOrderStrategy
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

    def handle_market_feedback(self, content: ClearingMessage, meta: MetaDict) -> None:
        """
        Handles the feedback which is received from a market we did bid at.

        Args:
            content (ClearingMessage): The content of the clearing message.
            meta (MetaDict): The meta data of the market.
        """
        logger.debug("%s got market result: %s", self.id, content)
        accepted_orders: Orderbook = content["accepted_orders"]
        rejected_orders: Orderbook = content["rejected_orders"]
        orderbook = accepted_orders + rejected_orders

        for order in orderbook:
            order["market_id"] = content["market_id"]

        marketconfig = self.registered_markets[content["market_id"]]
        self.valid_orders[marketconfig.product_type].extend(orderbook)
        self.set_unit_dispatch(orderbook, marketconfig)
        self.write_learning_to_output(orderbook, marketconfig.market_id)
        self.write_actual_dispatch(marketconfig.product_type)

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

                if isinstance(strategy, RLAdvancedOrderStrategy):
                    output_dict.update(
                        {
                            "profit": unit.outputs["profit"].loc[products_index].sum(),
                            "reward": unit.outputs["reward"].loc[products_index].sum()
                            / 24,
                            "regret": unit.outputs["regret"].loc[products_index].sum(),
                        }
                    )
                else:
                    output_dict.update(
                        {
                            "profit": unit.outputs["profit"].at[start],
                            "reward": unit.outputs["reward"].at[start],
                            "regret": unit.outputs["regret"].at[start],
                        }
                    )

                # Only for MATD3, not for PPO
                # Check if exploration_noise is not empty (MATD3)
                action_tuple = unit.outputs["actions"].at[start]
                if "exploration_noise" in unit.outputs and hasattr(unit.outputs["exploration_noise"].at[start], "numel"):
                    noise_tuple = unit.outputs["exploration_noise"].at[start]
                
                action_dim = action_tuple.numel()

                for i in range(action_dim):
                    # Only for MATD3, not for PPO
                    if "exploration_noise" in unit.outputs and hasattr(unit.outputs["exploration_noise"].loc[start], "numel"):
                        output_dict[f"exploration_noise_{i}"] = (
                            noise_tuple[i] if action_dim > 1 else noise_tuple
                        )
                    # For MATD3 and PPO
                    output_dict[f"actions_{i}"] = (
                        action_tuple[i] if action_dim > 1 else action_tuple
                    )

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

    # Executed in the interval set by train_frequency
    async def write_to_learning_role(
        self,
    ) -> None:
        """
        Writes learning results to the learning agent.

        """

        # print("write_to_learning_role in learning_unit_operator.py")

        if len(self.rl_units) == 0:
            return

        obs_dim = self.learning_strategies["obs_dim"]
        act_dim = self.learning_strategies["act_dim"]
        device = self.learning_strategies["device"]
        learning_unit_count = len(self.rl_units)

        # How many reward values are available in the first learning unit -> equals the number of steps
        values_len = len(self.rl_units[0].outputs["rl_rewards"])
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

        # For PPO
        # dimensions: steps, learning units, one log-prob for multiple observations/actions dimensions
        all_log_probs = th.zeros(
            (values_len, learning_unit_count, 1), device=device
        )

        # i is the index of the learning unit, unit is the learning unit object
        for i, unit in enumerate(self.rl_units):

            # Convert pandas Series to torch Tensor
            obs_tensor = th.stack(unit.outputs["rl_observations"][:values_len], dim=0)

            actions_tensor = th.stack(
                unit.outputs["rl_actions"][:values_len], dim=0
            ).reshape(-1, act_dim)

            # In the second dimension, the tensors include the number of the learning units
            # Three dimensions: Steps, learning units, observation/action dimensions
            all_observations[:, i, :] = obs_tensor
            all_actions[:, i, :] = actions_tensor
            all_rewards.append(unit.outputs["rl_rewards"])

            # For PPO
            # Check whether the list of tensors is not empty and whether the tensors contain elements
            if unit.outputs["rl_log_probs"]: # and all(t.numel() > 0 for t in unit.outputs["rl_log_probs"][:values_len]):
                
                log_prob_tensor = th.stack(
                    unit.outputs["rl_log_probs"][:values_len], dim=0
                ).unsqueeze(-1)
                
                all_log_probs[:, i, :] = log_prob_tensor

            # reset the outputs
            unit.reset_saved_rl_data()

        # convert all_actions list of tensor to numpy 2D array
        all_observations = (
            all_observations.squeeze()
            .cpu()
            .numpy()
            .reshape(-1, learning_unit_count, obs_dim)
        )
        all_actions = (
            all_actions.squeeze()
            .cpu()
            .numpy()
            .reshape(-1, learning_unit_count, act_dim)
        )


        all_rewards = np.array(all_rewards).reshape(-1, learning_unit_count)

        # For PPO
        if unit.outputs["rl_log_probs"]: # and all(t.numel() > 0 for t in unit.outputs["rl_log_probs"][:values_len]):
            all_log_probs = all_log_probs.detach().cpu().numpy().reshape(-1, learning_unit_count, 1)

            rl_agent_data = (all_observations, all_actions, all_rewards, all_log_probs)
        # For MATD3
        else:    
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

        return orderbook
