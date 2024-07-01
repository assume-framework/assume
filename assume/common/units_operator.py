# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from collections import defaultdict
from datetime import datetime
from itertools import groupby
from operator import itemgetter

import numpy as np
import pandas as pd
from mango import Role
from mango.messages.message import Performatives

from assume.common.market_objects import (
    ClearingMessage,
    DataRequestMessage,
    MarketConfig,
    MetaDict,
    OpeningMessage,
    Orderbook,
    RegistrationMessage,
)
from assume.common.utils import (
    aggregate_step_amount,
    get_products_index,
    timestamp2datetime,
)
from assume.strategies import BaseStrategy, LearningStrategy
from assume.units import BaseUnit

logger = logging.getLogger(__name__)

try:
    import torch as th
except ImportError:
    pass


class UnitsOperator(Role):
    """
    The UnitsOperator is the agent that manages the units.
    It receives the opening hours of the market and sends back the bids for the market.

    Attributes:
        available_markets (list[MarketConfig]): The available markets.
        registered_markets (dict[str, MarketConfig]): The registered markets.
        last_sent_dispatch (int): The last sent dispatch.
        use_portfolio_opt (bool): Whether to use portfolio optimization.
        portfolio_strategy (BaseStrategy): The portfolio strategy.
        valid_orders (defaultdict): The valid orders.
        units (dict[str, BaseUnit]): The units.
        id (str): The id of the agent.
        context (Context): The context of the agent.

    Args:
        available_markets (list[MarketConfig]): The available markets.
        opt_portfolio (tuple[bool, BaseStrategy] | None, optional): Optimized portfolio strategy. Defaults to None.
    """

    def __init__(
        self,
        available_markets: list[MarketConfig],
        opt_portfolio: tuple[bool, BaseStrategy] | None = None,
    ):
        super().__init__()

        self.available_markets = available_markets
        self.registered_markets: dict[str, MarketConfig] = {}
        self.last_sent_dispatch = defaultdict(lambda: 0)

        if opt_portfolio is None:
            self.use_portfolio_opt = False
            self.portfolio_strategy = None
        else:
            self.use_portfolio_opt = opt_portfolio[0]
            self.portfolio_strategy = opt_portfolio[1]

        # valid_orders per product_type
        self.valid_orders = defaultdict(list)
        self.units: dict[str, BaseUnit] = {}

        self.rl_units = []
        self.learning_strategies = {
            "obs_dim": 0,
            "act_dim": 0,
            "device": "cpu",
        }

    def setup(self):
        super().setup()
        self.id = self.context.aid
        self.context.subscribe_message(
            self,
            self.handle_opening,
            lambda content, meta: content.get("context") == "opening",
        )

        self.context.subscribe_message(
            self,
            self.handle_market_feedback,
            lambda content, meta: content.get("context") == "clearing",
        )

        self.context.subscribe_message(
            self,
            self.handle_registration_feedback,
            lambda content, meta: content.get("context") == "registration",
        )

        self.context.subscribe_message(
            self,
            self.handle_data_request,
            lambda content, meta: content.get("context") == "data_request",
        )

        for market in self.available_markets:
            if self.participate(market):
                self.context.schedule_timestamp_task(
                    self.register_market(market),
                    1,  # register after time was updated for the first time
                )

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

    def participate(self, market: MarketConfig) -> bool:
        """
        Method which decides if we want to participate on a given Market.
        This always returns true for now.

        Args:
            market (MarketConfig): The market to participate in.

        Returns:
            bool: True if participate, False otherwise.
        """
        return True

    async def register_market(self, market: MarketConfig) -> None:
        """
        Register a market.

        Args:
            market (MarketConfig): The market to register.
        """

        await self.context.send_acl_message(
            {
                "context": "registration",
                "market_id": market.market_id,
                "information": [u.as_dict() for u in self.units.values()],
            },
            receiver_addr=market.addr,
            receiver_id=market.aid,
            acl_metadata={
                "sender_addr": self.context.addr,
                "sender_id": self.context.aid,
                "reply_with": market.market_id,
            },
        )
        logger.debug(f"{self.id} sent market registration to {market.market_id}")

    def handle_opening(self, opening: OpeningMessage, meta: MetaDict) -> None:
        """
        When we receive an opening from the market, we schedule sending back our list of orders as a response.

        Args:
            opening (OpeningMessage): The opening message.
            meta (MetaDict): The meta data of the market.
        """
        logger.debug(
            f'{self.id} received opening from: {opening["market_id"]} {opening["start_time"]} until: {opening["end_time"]}.'
        )
        self.context.schedule_instant_task(coroutine=self.submit_bids(opening, meta))

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

    def handle_registration_feedback(
        self, content: RegistrationMessage, meta: MetaDict
    ) -> None:
        """
        Handles the feedback received from a market regarding registration.

        Args:
            content (RegistrationMessage): The content of the registration message.
            meta (MetaDict): The meta data of the market.
        """
        logger.debug("Market %s accepted our registration", content["market_id"])
        if content["accepted"]:
            found = False
            for market in self.available_markets:
                if content["market_id"] == market.market_id:
                    self.registered_markets[market.market_id] = market
                    found = True
                    break
            if not found:
                logger.error(
                    "Market %s sent registation but is unknown", content["market_id"]
                )
        else:
            logger.error("Market %s did not accept registration", meta["sender_id"])

    def handle_data_request(self, content: DataRequestMessage, meta: MetaDict) -> None:
        """
        Handles the data request received from other agents.

        Args:
            content (DataRequestMessage): The content of the data request message.
            meta (MetaDict): The meta data of the market.
        """
        unit = content["unit"]
        metric_type = content["metric"]
        start = content["start_time"]
        end = content["end_time"]

        data = []
        try:
            data = self.units[unit].outputs[metric_type][start:end]
        except Exception:
            logger.exception("error handling data request")
        self.context.schedule_instant_acl_message(
            content={
                "context": "data_response",
                "data": data,
            },
            receiver_addr=meta["sender_addr"],
            receiver_id=meta["sender_id"],
            acl_metadata={
                "sender_addr": self.context.addr,
                "sender_id": self.context.aid,
                "in_reply_to": meta.get("reply_with"),
            },
        )

    def set_unit_dispatch(
        self, orderbook: Orderbook, marketconfig: MarketConfig
    ) -> None:
        """
        Feeds the current market result back to the units.

        Args:
            orderbook (Orderbook): The orderbook of the market.
            marketconfig (MarketConfig): The market configuration.
        """
        orderbook.sort(key=itemgetter("unit_id"))
        for unit_id, orders in groupby(orderbook, itemgetter("unit_id")):
            orderbook = list(orders)
            self.units[unit_id].set_dispatch_plan(
                marketconfig=marketconfig,
                orderbook=orderbook,
            )

    def get_actual_dispatch(
        self, product_type: str, last: datetime
    ) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
        """
        Retrieves the actual dispatch and commits it in the unit.
        We calculate the series of the actual market results dataframe with accepted bids.
        And the unit_dispatch for all units taken care of in the UnitsOperator.

        Args:
            product_type (str): The product type for which this is done
            last (datetime): the last date until which the dispatch was already sent

        Returns:
            tuple[pd.DataFrame, list[pd.DataFrame]]: market_dispatch and unit_dispatch dataframes
        """
        now = timestamp2datetime(self.context.current_timestamp)
        start = timestamp2datetime(last + 1)

        market_dispatch = aggregate_step_amount(
            orderbook=self.valid_orders[product_type],
            begin=timestamp2datetime(last),
            end=now,
            groupby=["market_id", "unit_id"],
        )

        unit_dispatch_dfs = []
        for unit_id, unit in self.units.items():
            current_dispatch = unit.execute_current_dispatch(start, now)
            end = now
            current_dispatch.name = "power"
            data = pd.DataFrame(current_dispatch)

            # TODO: this needs to be fixed. For now it is consuming too much time and is deactivated
            # unit.calculate_generation_cost(start, now, "energy")
            valid_outputs = ["soc", "cashflow", "marginal_costs", "total_costs"]

            for key in unit.outputs.keys():
                for output in valid_outputs:
                    if output in key:
                        data[key] = unit.outputs[key][start:end]

            data["unit"] = unit_id
            unit_dispatch_dfs.append(data)

        return market_dispatch, unit_dispatch_dfs

    def write_actual_dispatch(self, product_type: str) -> None:
        """
        Sends the actual aggregated dispatch curve to the output agent.

        Args:
            product_type (str): The type of the product.
        """

        last = self.last_sent_dispatch[product_type]
        if self.context.current_timestamp == last:
            # stop if we exported at this time already
            return
        self.last_sent_dispatch[product_type] = self.context.current_timestamp

        market_dispatch, unit_dispatch_dfs = self.get_actual_dispatch(
            product_type, last
        )

        now = timestamp2datetime(self.context.current_timestamp)
        self.valid_orders[product_type] = list(
            filter(
                lambda x: x["end_time"] > now,
                self.valid_orders[product_type],
            )
        )

        db_aid = self.context.data.get("output_agent_id")
        db_addr = self.context.data.get("output_agent_addr")
        if db_aid and db_addr:
            self.context.schedule_instant_acl_message(
                receiver_id=db_aid,
                receiver_addr=db_addr,
                content={
                    "context": "write_results",
                    "type": "market_dispatch",
                    "data": market_dispatch,
                },
            )
            if unit_dispatch_dfs:
                unit_dispatch = pd.concat(unit_dispatch_dfs)
                self.context.schedule_instant_acl_message(
                    receiver_id=db_aid,
                    receiver_addr=db_addr,
                    content={
                        "context": "write_results",
                        "type": "unit_dispatch",
                        "data": unit_dispatch,
                    },
                )

    async def submit_bids(self, opening: OpeningMessage, meta: MetaDict) -> None:
        """
        Formulates an orderbook and sends it to the market.

        Args:
            opening (OpeningMessage): The opening message.
            meta (MetaDict): The meta data of the market.

        Note:
            This function will accomodate the portfolio optimization in the future.
        """

        products = opening["products"]
        market = self.registered_markets[opening["market_id"]]
        logger.debug(f"{self.id} setting bids for {market.market_id} - {products}")

        # the given products just became available on our market
        # and we need to provide bids
        # [whole_next_hour, quarter1, quarter2, quarter3, quarter4]
        # algorithm should buy as much baseload as possible, then add up with quarters
        products.sort(key=lambda p: (p[0] - p[1], p[0]))
        if self.use_portfolio_opt:
            orderbook = await self.formulate_bids_portfolio(
                market=market,
                products=products,
            )
        else:
            orderbook = await self.formulate_bids(
                market=market,
                products=products,
            )
        acl_metadata = {
            "performative": Performatives.inform,
            "sender_id": self.context.aid,
            "sender_addr": self.context.addr,
            "conversation_id": "conversation01",
            "in_reply_to": meta.get("reply_with"),
        }
        await self.context.send_acl_message(
            content={
                "context": "submit_bids",
                "market_id": market.market_id,
                "orderbook": orderbook,
            },
            receiver_addr=market.addr,
            receiver_id=market.aid,
            acl_metadata=acl_metadata,
        )

    async def formulate_bids_portfolio(
        self, market: MarketConfig, products: list[tuple]
    ) -> Orderbook:
        """
        Formulates the bid to the market according to the bidding strategy of the unit operator.

        Args:
            market (MarketConfig): The market to formulate bids for.
            products (list[tuple]): The products to formulate bids for.

        Returns:
            OrderBook: The orderbook that is submitted as a bid to the market.

        Note:
            Placeholder for future portfolio optimization.
        """
        orderbook: Orderbook = []
        # TODO sort units by priority
        # execute operator bidding strategy..?
        for unit_id, unit in self.units.items():
            unit.technology
            # TODO calculate bids from sum of available power

        return orderbook

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
                order["agent_id"] = (self.context.addr, self.context.aid)
                if market.volume_tick:
                    order["volume"] = round(order["volume"] / market.volume_tick)
                if market.price_tick:
                    order["price"] = round(order["price"] / market.price_tick)
                if "bid_id" not in order.keys() or order["bid_id"] is None:
                    order["bid_id"] = f"{unit_id}_{i+1}"
                order["unit_id"] = unit_id
                orderbook.append(order)

        return orderbook

    def write_learning_to_output(self, orderbook: Orderbook, market_id: str) -> None:
        """
        Sends the current rl_strategy update to the output agent.

        Args:
            products_index (pandas.DatetimeIndex): The index of all products.
            marketconfig (MarketConfig): The market configuration.
        """
        try:
            from assume.strategies import (
                RLAdvancedOrderStrategy,
            )

        except ImportError:
            pass

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
