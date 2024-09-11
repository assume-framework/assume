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
from assume.common.utils import (
    get_products_index,
)
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

    async def add_unit(
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

        db_aid = self.context.data.get("output_agent_id")
        db_addr = self.context.data.get("output_agent_addr")
        if db_aid and db_addr:
            # send unit data to db agent to store it
            message = {
                "context": "write_results",
                "type": "store_units",
                "data": self.units[unit.id].as_dict(),
            }
            await self.context.send_acl_message(
                receiver_id=db_aid,
                receiver_addr=db_addr,
                content=message,
                acl_metadata={
                    "sender_addr": self.context.addr,
                    "sender_id": self.context.aid,
                },
            )

    def handle_market_feedback(self, content: ClearingMessage, meta: MetaDict) -> None:
        """
        Handles the feedback which is received from a market we did bid at.

        Args:
            content (ClearingMessage): The content of the clearing message.
            meta (MetaDict): The meta data of the market.
        """
        logger.debug(f"{self.id} got market result: {content}")
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
                            "profit": unit.outputs["profit"].loc[start],
                            "reward": unit.outputs["reward"].loc[start],
                            "regret": unit.outputs["regret"].loc[start],
                        }
                    )

                action_tuple = unit.outputs["actions"].loc[start]
                noise_tuple = unit.outputs["exploration_noise"].loc[start]
                action_dim = action_tuple.numel()

                for i in range(action_dim):
                    output_dict[f"exploration_noise_{i}"] = (
                        noise_tuple[i] if action_dim > 1 else noise_tuple
                    )
                    output_dict[f"actions_{i}"] = (
                        action_tuple[i] if action_dim > 1 else action_tuple
                    )

                output_agent_list.append(output_dict)

        db_aid = self.context.data.get("learning_output_agent_id")
        db_addr = self.context.data.get("learning_output_agent_addr")

        if db_aid and db_addr and output_agent_list:
            self.context.schedule_instant_acl_message(
                receiver_id=db_aid,
                receiver_addr=db_addr,
                content={
                    "context": "write_results",
                    "type": "rl_learning_params",
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

        for i, unit in enumerate(self.rl_units):
            # Convert pandas Series to torch Tensor
            obs_tensor = th.stack(unit.outputs["rl_observations"][:values_len], dim=0)
            actions_tensor = th.stack(
                unit.outputs["rl_actions"][:values_len], dim=0
            ).reshape(-1, act_dim)

            all_observations[:, i, :] = obs_tensor
            all_actions[:, i, :] = actions_tensor
            all_rewards.append(unit.outputs["rl_rewards"])

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
        rl_agent_data = (all_observations, all_actions, all_rewards)

        learning_role_id = self.context.data.get("learning_agent_id")
        learning_role_addr = self.context.data.get("learning_agent_addr")

        if learning_role_id and learning_role_addr:
            self.context.schedule_instant_acl_message(
                receiver_id=learning_role_id,
                receiver_addr=learning_role_addr,
                content={
                    "context": "rl_training",
                    "type": "save_buffer_and_update",
                    "data": rl_agent_data,
                },
            )
