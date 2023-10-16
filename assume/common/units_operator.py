import logging
from datetime import datetime
from itertools import groupby
from operator import itemgetter

import numpy as np
import pandas as pd
from mango import Role
from mango.messages.message import Performatives

from assume.common.market_objects import (
    ClearingMessage,
    MarketConfig,
    OpeningMessage,
    Orderbook,
)
from assume.common.utils import aggregate_step_amount
from assume.strategies import BaseStrategy, LearningStrategy
from assume.units import BaseUnit

logger = logging.getLogger(__name__)


class UnitsOperator(Role):
    """
    The UnitsOperator is the agent that manages the units.
    It receives the opening hours of the market and sends back the bids for the market.

    :param available_markets: the available markets
    :type available_markets: list[MarketConfig]
    :param opt_portfolio: optimized portfolio strategy
    :type opt_portfolio: tuple[bool, BaseStrategy] | None
    """

    def __init__(
        self,
        available_markets: list[MarketConfig],
        opt_portfolio: tuple[bool, BaseStrategy] | None = None,
    ):
        super().__init__()

        self.bids_map = {}
        self.available_markets = available_markets
        self.registered_markets: dict[str, MarketConfig] = {}
        self.last_sent_dispatch = 0

        if opt_portfolio is None:
            self.use_portfolio_opt = False
            self.portfolio_strategy = None
        else:
            self.use_portfolio_opt = opt_portfolio[0]
            self.portfolio_strategy = opt_portfolio[1]

        # should be a list per product_type
        self.valid_orders = []
        self.units: dict[str, BaseUnit] = {}

    def setup(self):
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

        for market in self.available_markets:
            if self.participate(market):
                self.register_market(market)
                self.registered_markets[market.name] = market

    async def add_unit(
        self,
        unit: BaseUnit,
    ):
        """
        Create a unit.

        :param unit: the unit to be added
        :type unit: BaseUnit
        """
        self.units[unit.id] = unit

        db_aid = self.context.data_dict.get("output_agent_id")
        db_addr = self.context.data_dict.get("output_agent_addr")
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

    def participate(self, market: MarketConfig):
        """
        Method which decides if we want to participate on a given Market.
        This always returns true for now

        :param market: the market to participate in
        :type market: MarketConfig
        """
        return True

    def register_market(self, market: MarketConfig):
        """
        Register a market.

        :param market: the market to register
        :type market: MarketConfig
        """
        self.context.schedule_timestamp_task(
            self.context.send_acl_message(
                {"context": "registration", "market": market.name},
                market.addr,
                receiver_id=market.aid,
                acl_metadata={
                    "sender_addr": self.context.addr,
                    "sender_id": self.context.aid,
                },
            ),
            1,  # register after time was updated for the first time
        )
        logger.debug(f"{self.id} tried to register at market {market.name}")

    def handle_opening(self, opening: OpeningMessage, meta: dict[str, str]):
        """
        When we receive an opening from the market, we schedule sending back
        our list of orders as a response

        :param opening: the opening message
        :type opening: OpeningMessage
        :param meta: the meta data of the market
        :type meta: dict[str, str]
        """
        logger.debug(
            f'{self.id} received opening from: {opening["market_id"]} {opening["start"]} until: {opening["stop"]}.'
        )
        self.context.schedule_instant_task(coroutine=self.submit_bids(opening))

    def handle_market_feedback(self, content: ClearingMessage, meta: dict[str, str]):
        """
        handles the feedback which is received from a market we did bid at
        stores accepted orders, sets the received power
        writes result back for the learning
        and executes the dispatch, including ramping for times in the past

        :param content: the content of the clearing message
        :type content: ClearingMessage
        :param meta: the meta data of the market
        :type meta: dict[str, str]
        """
        logger.debug(f"{self.id} got market result: {content}")
        accepted_orders: Orderbook = content["accepted_orders"]
        rejected_orders: Orderbook = content["rejected_orders"]
        orderbook = accepted_orders + rejected_orders

        for order in orderbook:
            order["market_id"] = content["market_id"]
            # map bid id to unit id
            order["unit_id"] = self.bids_map[order["bid_id"]]

        self.valid_orders.extend(orderbook)
        marketconfig = self.registered_markets[content["market_id"]]
        self.set_unit_dispatch(orderbook, marketconfig)
        self.write_learning_params(orderbook, marketconfig)
        self.write_actual_dispatch()

    def set_unit_dispatch(self, orderbook: Orderbook, marketconfig: MarketConfig):
        """
        feeds the current market result back to the units
        this does not respect bids from multiple markets
        for the same time period, as we only have access to the current orderbook here

        :param orderbook: the orderbook of the market
        :type orderbook: Orderbook
        :param marketconfig: the market configuration
        :type marketconfig: MarketConfig
        """
        orderbook.sort(key=itemgetter("unit_id"))
        for unit_id, orders in groupby(orderbook, itemgetter("unit_id")):
            orderbook = list(orders)
            self.units[unit_id].set_dispatch_plan(
                marketconfig=marketconfig,
                orderbook=orderbook,
            )

    def write_actual_dispatch(self):
        """
        sends the actual aggregated dispatch curve
        works across multiple markets
        sends dispatch at or after it actually happens
        """

        last = self.last_sent_dispatch
        if self.context.current_timestamp == last:
            # stop if we exported at this time already
            return
        self.last_sent_dispatch = self.context.current_timestamp

        now = datetime.utcfromtimestamp(self.context.current_timestamp)
        start = datetime.utcfromtimestamp(last)

        market_dispatch = aggregate_step_amount(
            self.valid_orders, start, now, groupby=["market_id", "unit_id"]
        )
        unit_dispatch_dfs = []
        for unit_id, unit in self.units.items():
            current_dispatch = unit.execute_current_dispatch(start, now)
            end = now
            current_dispatch.name = "power"
            data = pd.DataFrame(current_dispatch)
            data["soc"] = unit.outputs["soc"][start:end]

            for key in unit.outputs.keys():
                if "cashflow" in key:
                    data[key] = unit.outputs[key][start:end]
                if "marginal_costs" in key:
                    data[key] = unit.outputs[key][start:end]
                if "total_costs" in key:
                    data[key] = unit.outputs[key][start:end]

            data["unit"] = unit_id
            unit_dispatch_dfs.append(data)

        self.valid_orders = list(
            filter(lambda x: x["end_time"] > now, self.valid_orders)
        )

        db_aid = self.context.data_dict.get("output_agent_id")
        db_addr = self.context.data_dict.get("output_agent_addr")
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

    async def submit_bids(self, opening: OpeningMessage):
        """
        formulates an orderbook and sends it to the market.
        This will handle optional portfolio processing

        :param opening: the opening message
        :type opening: OpeningMessage
        """

        products = opening["products"]
        market = self.registered_markets[opening["market_id"]]
        logger.debug(f"{self.id} setting bids for {market.name} - {products}")

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
        }
        await self.context.send_acl_message(
            content={
                "context": "submit_bids",
                "market": market.name,
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
        Takes information from all units that the unit operator manages and
        formulates the bid to the market from that according to the bidding strategy of the unit operator.

        This is the portfolio optimization version

        :param market: the market to formulate bids for
        :type market: MarketConfig
        :param products: the products to formulate bids for
        :type products: list[tuple]

        :return: OrderBook that is submitted as a bid to the market
        :rtype: OrderBook
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
        Takes information from all units that the unit operator manages and
        formulates the bid to the market from that according to the bidding strategy of the unit itself.

        :param market: the market to formulate bids for
        :type market: MarketConfig
        :param products: the products to formulate bids for
        :type products: list[tuple]

        :return OrderBook that is submitted as a bid to the market
        :rtype OrderBook
        """

        orderbook: Orderbook = []

        for unit_id, unit in self.units.items():
            product_bids = unit.calculate_bids(
                market,
                product_tuples=products,
            )
            for i, order in enumerate(product_bids):
                if isinstance(order["volume"], dict):
                    if all(volume == 0 for volume in order["volume"].values()):
                        continue
                elif order["volume"] == 0:
                    continue
                order["agent_id"] = (self.context.addr, self.context.aid)
                if market.volume_tick:
                    order["volume"] = round(order["volume"] / market.volume_tick)
                if market.price_tick:
                    order["price"] = round(order["price"] / market.price_tick)

                order["bid_id"] = f"{unit_id}_{i+1}"
                orderbook.append(order)
                self.bids_map[order["bid_id"]] = unit_id

        return orderbook

    def write_learning_to_output(self, start: datetime, marketconfig: MarketConfig):
        output_agent_list = []
        for unit_id, unit in self.units.items():
            # rl only for energy market for now!
            if isinstance(
                unit.bidding_strategies.get(marketconfig.product_type),
                LearningStrategy,
            ):
                output_dict = {
                    "datetime": start,
                    "profit": unit.outputs["profit"].loc[start],
                    "reward": unit.outputs["reward"].loc[start],
                    "regret": unit.outputs["regret"].loc[start],
                    "unit": unit_id,
                }
                noise_tuple = unit.outputs["rl_exploration_noise"].loc[start]
                action_tuple = unit.outputs["rl_actions"].loc[start]
                action_dim = len(action_tuple)
                for i in range(action_dim):
                    output_dict[f"exploration_noise_{i}"] = noise_tuple[i]
                    output_dict[f"actions_{i}"] = action_tuple[i]

                output_agent_list.append(output_dict)

        db_aid = self.context.data_dict.get("learning_output_agent_id")
        db_addr = self.context.data_dict.get("learning_output_agent_addr")

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

    def write_to_learning(
        self,
        start: datetime,
        marketconfig: MarketConfig,
        obs_dim: int,
        act_dim: int,
        device: str,
        learning_unit_count: int,
    ):
        all_observations = []
        all_rewards = []
        try:
            import torch as th

        except ImportError:
            logger.error("tried writing learning_params, but torch is not installed")
            return

        all_observations = th.zeros((learning_unit_count, obs_dim), device=device)
        all_actions = th.zeros((learning_unit_count, act_dim), device=device)

        i = 0
        for unit in self.units.values():
            # rl only for energy market for now!
            if isinstance(
                unit.bidding_strategies.get(marketconfig.product_type),
                LearningStrategy,
            ):
                all_observations[i, :] = unit.outputs["rl_observations"][start]
                all_actions[i, :] = unit.outputs["rl_actions"][start]
                all_rewards.append(unit.outputs["reward"][start])
                i += 1

        # convert all_actions list of tensor to numpy 2D array
        all_observations = all_observations.squeeze().cpu().numpy()
        all_actions = all_actions.squeeze().cpu().numpy()
        all_rewards = np.array(all_rewards)
        rl_agent_data = (np.array(all_observations), all_actions, all_rewards)

        learning_role_id = self.context.data_dict.get("learning_agent_id")
        learning_role_addr = self.context.data_dict.get("learning_agent_addr")

        if learning_role_id and learning_role_addr:
            self.context.schedule_instant_acl_message(
                receiver_id=learning_role_id,
                receiver_addr=learning_role_addr,
                content={
                    "context": "rl_training",
                    "type": "replay_buffer",
                    "data": rl_agent_data,
                },
            )

    def write_learning_params(self, orderbook: Orderbook, marketconfig: MarketConfig):
        """
        sends the current rl_strategy update to the output agent

        :param orderbook: the orderbook of the market
        :type orderbook: Orderbook
        :param marketconfig: the market configuration
        :type marketconfig: MarketConfig
        """
        learning_strategies = []

        for unit in self.units.values():
            bidding_strategy = unit.bidding_strategies.get(marketconfig.product_type)
            if isinstance(bidding_strategy, LearningStrategy):
                learning_strategies.append(bidding_strategy)
                # should be the same across all strategies
                obs_dim = bidding_strategy.obs_dim
                act_dim = bidding_strategy.act_dim
                device = bidding_strategy.device

        # should write learning results if at least one bidding_strategy is a learning strategy
        if learning_strategies and orderbook:
            start = orderbook[0]["start_time"]
            # write learning output
            self.write_learning_to_output(start, marketconfig)

            # we are using the first learning_strategy to check learning_mode
            # as this should be the same value for all strategies
            if learning_strategies[0].learning_mode:
                # in learning mode we are sending data to learning
                self.write_to_learning(
                    start=start,
                    marketconfig=marketconfig,
                    obs_dim=obs_dim,
                    act_dim=act_dim,
                    device=device,
                    learning_unit_count=len(learning_strategies),
                )
