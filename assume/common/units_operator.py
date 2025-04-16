# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from collections import defaultdict
from datetime import datetime
from itertools import groupby
from operator import itemgetter

from mango import Role, create_acl, sender_addr
from mango.messages.message import Performatives

from assume.common.market_objects import (
    ClearingMessage,
    DataRequestMessage,
    MarketConfig,
    MetaDict,
    OpeningMessage,
    Orderbook,
    RegistrationMessage,
    lambda_functions,
)
from assume.common.utils import (
    aggregate_step_amount,
    timestamp2datetime,
)
from assume.strategies import BaseStrategy
from assume.units import BaseUnit

logger = logging.getLogger(__name__)


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

    def setup(self):
        super().setup()
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

    def on_ready(self):
        super().on_ready()
        self.id = self.context.aid

        for market in self.available_markets:
            if self.participate(market):
                self.context.schedule_timestamp_task(
                    self.register_market(market),
                    1,  # register after time was updated for the first time
                )

        self.context.schedule_timestamp_task(
            self.store_units(),
            1,  # register after time was updated for the first time
        )

    async def store_units(self) -> None:
        db_addr = self.context.data.get("output_agent_addr")
        logger.debug("store units to %s", db_addr)
        if db_addr:
            # send unit data to db agent to store it
            for unit in self.units.values():
                message = {
                    "context": "write_results",
                    "type": "store_units",
                    "data": unit.as_dict(),
                }
                await self.context.send_message(
                    content=message,
                    receiver_addr=db_addr,
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

    def participate(self, market: MarketConfig) -> bool:
        """
        Method which decides if we want to participate on a given Market.
        This always returns true for now.

        Args:
            market (MarketConfig): The market to participate in.

        Returns:
            bool: True if participate, False otherwise.
        """
        if callable(market.eligible_obligations_lambda):
            requirement = market.eligible_obligations_lambda
        else:
            requirement = lambda_functions.get(
                market.eligible_obligations_lambda, lambda u: True
            )

        for u in self.units.values():
            if market.market_id in u.bidding_strategies.keys() and requirement(
                u.as_dict()
            ):
                return True

        return False

    async def register_market(self, market: MarketConfig) -> None:
        """
        Register a market.

        Args:
            market (MarketConfig): The market to register.
        """
        if not market.addr:
            logger.error("Market %s has no address", market.market_id)
            return
        await self.context.send_message(
            create_acl(
                {
                    "context": "registration",
                    "market_id": market.market_id,
                    "information": [u.as_dict() for u in self.units.values()],
                },
                market.addr,
                self.context.addr,
                acl_metadata={
                    "reply_with": market.market_id,
                    "performative": Performatives.propose,
                },
            ),
            receiver_addr=market.addr,
        )
        logger.debug("%s sent market registration to %s", self.id, market.market_id)

    def handle_opening(self, opening: OpeningMessage, meta: MetaDict) -> None:
        """
        When we receive an opening from the market, we schedule sending back our list of orders as a response.

        Args:
            opening (OpeningMessage): The opening message.
            meta (MetaDict): The meta data of the market.
        """
        logger.debug(
            "%s received opening from: %s %s until: %s.",
            self.id,
            opening["market_id"],
            opening["start_time"],
            opening["end_time"],
        )
        self.context.schedule_instant_task(coroutine=self.submit_bids(opening, meta))

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
        self.write_actual_dispatch(marketconfig.product_type)

        # now once we have the market results and the dispatch has been set
        # we can calculate the cashflow and reward for the units
        self.calculate_unit_cashflow_and_reward(orderbook, marketconfig)

        # if unit operator is a subclass of learning unit operator
        # we need to write the learning data to the output agent
        if hasattr(self, "write_learning_to_output"):
            self.write_learning_to_output(orderbook, marketconfig.market_id)

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
                    "Market %s sent registration but is unknown", content["market_id"]
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
            data = (
                self.units[unit].outputs[metric_type].as_pd_series(start=start, end=end)
            )
        except Exception:
            logger.exception("error handling data request")
        self.context.schedule_instant_message(
            create_acl(
                content={
                    "context": "data_response",
                    "data": data,
                },
                receiver_addr=sender_addr(meta),
                sender_addr=self.context.addr,
                acl_metadata={
                    "in_reply_to": meta.get("reply_with"),
                    "performative": Performatives.inform,
                },
            ),
            receiver_addr=sender_addr(meta),
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

    def calculate_unit_cashflow_and_reward(
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
            self.units[unit_id].calculate_cashflow_and_reward(
                marketconfig=marketconfig,
                orderbook=orderbook,
            )

    def get_actual_dispatch(
        self, product_type: str, last: datetime
    ) -> tuple[list[tuple[datetime, float, str, str]], list[dict]]:
        """
        Retrieves the actual dispatch since the last dispatch and commits it in the unit.
        We calculate the series of the actual market results dataframe with accepted bids.
        And the unit_dispatch for all units taken care of in the UnitsOperator.

        Args:
            product_type (str): The product type for which this is done
            last (datetime.datetime): the last date until which the dispatch was already sent

        Returns:
            tuple[list[tuple[datetime, float, str, str]], list[dict]]: market_dispatch and unit_dispatch dataframes
        """
        now = timestamp2datetime(self.context.current_timestamp)
        # add one second to exclude the first time stamp, because it is already executed in the last step
        start = timestamp2datetime(last + 1)

        market_dispatch = aggregate_step_amount(
            orderbook=self.valid_orders[product_type],
            begin=timestamp2datetime(last),
            end=now,
            groupby=["market_id", "unit_id"],
        )

        unit_dispatch = []
        for unit_id, unit in self.units.items():
            current_dispatch = unit.execute_current_dispatch(start, now)
            end = now
            dispatch = {"power": current_dispatch}
            unit.calculate_generation_cost(start, now, "energy")
            valid_outputs = [
                "soc",
                "cashflow",
                "generation_costs",
                "total_costs",
                "heat",
            ]

            for key in unit.outputs.keys():
                for output in valid_outputs:
                    if output in key:
                        dispatch[key] = unit.outputs[key].loc[start:end]
            dispatch["time"] = unit.index.get_date_list(start, end)
            dispatch["unit"] = unit_id
            unit_dispatch.append(dispatch)

        return market_dispatch, unit_dispatch

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

        market_dispatch, unit_dispatch = self.get_actual_dispatch(product_type, last)

        now = timestamp2datetime(self.context.current_timestamp)
        self.valid_orders[product_type] = list(
            filter(
                lambda x: x["end_time"] > now,
                self.valid_orders[product_type],
            )
        )

        db_addr = self.context.data.get("output_agent_addr")
        if db_addr:
            self.context.schedule_instant_message(
                receiver_addr=db_addr,
                content={
                    "context": "write_results",
                    "type": "market_dispatch",
                    "data": market_dispatch,
                },
            )
            if unit_dispatch:
                self.context.schedule_instant_message(
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
            This function will accommodate the portfolio optimization in the future.
        """

        products = opening["products"]
        market = self.registered_markets[opening["market_id"]]
        logger.debug("%s setting bids for %s - %s", self.id, market.market_id, products)

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
        if not market.addr:
            logger.error("Market %s has no address", market.market_id)
            return
        await self.context.send_message(
            create_acl(
                content={
                    "context": "submit_bids",
                    "market_id": market.market_id,
                    "orderbook": orderbook,
                },
                receiver_addr=market.addr,
                sender_addr=self.context.addr,
                acl_metadata={
                    "performative": Performatives.inform,
                    "conversation_id": "conversation01",
                    "in_reply_to": meta.get("reply_with"),
                },
            ),
            receiver_addr=market.addr,
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
