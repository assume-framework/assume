# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import math
from itertools import groupby
from operator import itemgetter

from mango import AgentAddress, Performatives, Role, create_acl, sender_addr

from assume.common.market_objects import (
    ClearingMessage,
    DataRequestMessage,
    MarketConfig,
    MarketProduct,
    MetaDict,
    OpeningMessage,
    Orderbook,
    OrderBookMessage,
    RegistrationMessage,
    RegistrationReplyMessage,
    lambda_functions,
)
from assume.common.utils import (
    datetime2timestamp,
    get_available_products,
    separate_orders,
    timestamp2datetime,
)

logger = logging.getLogger(__name__)

process_time = 0


class MarketMechanism:
    """
    This class represents a market mechanism.

    It is different than the MarketRole, in the way that the functionality is unrelated to mango.
    The MarketMechanism is embedded into the general MarketRole, which takes care of simulation concerns.
    In the Marketmechanism, all data needed for the clearing is present.

    Attributes:
        marketconfig (MarketConfig): The configuration of the market.

    Args:
        marketconfig (MarketConfig): The configuration of the market.
    """

    def __init__(self, marketconfig: MarketConfig):
        super().__init__()
        self.marketconfig = marketconfig
        # calculate last possible market opening as the difference between the market end
        # and the length of the longest product plus the delivery time of the products
        self.last_market_opening = marketconfig.opening_hours._until - max(
            market_product.duration * market_product.count
            + market_product.first_delivery
            for market_product in marketconfig.market_products
        )

    def clear(
        self, orderbook: Orderbook, market_products: list[MarketProduct]
    ) -> tuple[Orderbook, Orderbook, list[dict], dict[tuple, float]]:
        """
        Clears the market.

        Args:
            orderbook (Orderbook): The orderbook to be cleared.
            market_products (list[MarketProduct]): The products to be traded.

        Returns:
            (Orderbook, Orderbook, list[dict], dict[tuple, float]): The empty accepted orderbook, the empty rejected orderbook and the empty market metadata and the flows between market zones.
        """
        return [], [], [], {}


class MarketRole(MarketMechanism, Role):
    """
    This is the base class for all market roles. It implements the basic functionality of a market role, such as
    registering agents, clearing the market and sending the results to the database agent.

    Args:
        marketconfig (MarketConfig): The configuration of the market.

    Methods
    -------
    """

    longitude: float
    latitude: float
    marketconfig: MarketConfig
    registered_agents: dict[AgentAddress, dict]
    required_fields: list[str] = []

    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)
        self.registered_agents = {}
        self.open_auctions = set()
        self.all_orders = []
        self.results = []
        if marketconfig.price_tick:
            if marketconfig.maximum_bid_price % marketconfig.price_tick != 0:
                logger.warning(
                    f"{marketconfig.market_id} - max price not a multiple of tick size"
                )
            if marketconfig.minimum_bid_price % marketconfig.price_tick != 0:
                logger.warning(
                    f"{marketconfig.market_id} - min price not a multiple of tick size"
                )

        if marketconfig.volume_tick and marketconfig.maximum_bid_volume:
            if marketconfig.maximum_bid_volume % marketconfig.volume_tick != 0:
                logger.warning(
                    f"{marketconfig.market_id} - max volume not a multiple of tick size"
                )

        self.grid_data = marketconfig.param_dict.get("grid_data")

    def setup(self):
        """
        Sets up the initial configuration and subscriptions for the market role.

        This method performs the following actions:

        - Sets the address and agent ID of the market configuration to match the current context.
        - Validates that all required fields are present in the market configuration.
        - Defines filter methods (`accept_orderbook`, `accept_registration`, `accept_get_unmatched`, `accept_data_request`) that serve as validation steps for different types of incoming messages.
        - Subscribes the role to handle incoming messages using the appropriate handler methods.
        - Schedules the `opening()` method to run at the next opening time of the market.
        - Sends grid topology data once, if available.

        Raises:
            ValueError: If a required field is missing from the market configuration.
        """

        super().setup()
        self.marketconfig.addr = self.context.addr

        market_id = getattr(self.marketconfig, "market_id", "Unknown Market ID")

        # Validate required fields in market configuration
        missing_fields = [
            field
            for field in self.required_fields
            if field not in self.marketconfig.additional_fields
        ]
        if missing_fields:
            error_message = (
                f"Missing required field(s) {missing_fields} from additional_fields "
                f"for market '{market_id}'."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        def accept_orderbook(content: OrderBookMessage, meta: MetaDict):
            if (
                not isinstance(content, dict)
                and content.get("context") == "submit_bids"
            ):
                return False

            if isinstance(meta["sender_addr"], list):
                meta["sender_addr"] = tuple(meta["sender_addr"])

            return (
                content.get("market_id") == self.marketconfig.market_id
                and content.get("orderbook") is not None
                and sender_addr(meta) in self.registered_agents.keys()
            )

        def accept_registration(content: RegistrationMessage, meta: MetaDict):
            if (
                not isinstance(content, dict)
                or content.get("context") != "registration"
            ):
                return False
            if isinstance(meta["sender_addr"], list):
                meta["sender_addr"] = tuple(meta["sender_addr"])

            return content.get("market_id") == self.marketconfig.market_id

        def accept_get_unmatched(content: dict, meta: MetaDict):
            if not isinstance(content, dict):
                return False
            if isinstance(meta["sender_addr"], list):
                meta["sender_addr"] = tuple(meta["sender_addr"])
            return (
                content.get("context") == "get_unmatched"
                and content.get("market_id") == self.marketconfig.market_id
            )

        def accept_data_request(content: dict, meta: MetaDict):
            return (
                content.get("context") == "data_request"
                and content.get("market_id") == self.marketconfig.market_id
            )

        self.context.subscribe_message(
            self, self.handle_data_request, accept_data_request
        )
        self.context.subscribe_message(self, self.handle_orderbook, accept_orderbook)
        self.context.subscribe_message(
            self, self.handle_registration, accept_registration
        )

        if self.marketconfig.supports_get_unmatched:
            self.context.subscribe_message(
                self, self.handle_get_unmatched, accept_get_unmatched
            )

    def on_ready(self):
        current = timestamp2datetime(self.context.current_timestamp)
        next_opening = self.marketconfig.opening_hours.after(current, inc=True)
        opening_ts = datetime2timestamp(next_opening)
        self.context.schedule_timestamp_task(self.opening(), opening_ts)

        # send grid topology data once
        if self.grid_data is not None and self.context.data.get("output_agent_addr"):
            self.context.schedule_instant_message(
                {
                    "context": "write_results",
                    "type": "grid_topology",
                    "data": self.grid_data,
                    "market_id": self.marketconfig.market_id,
                },
                receiver_addr=self.context.data.get("output_agent_addr"),
            )

    async def opening(self):
        """
        Sends an opening message to all registered agents, handles scheduling the clearing of the market and the next opening.

        """
        # scheduled to be opened now
        market_open = timestamp2datetime(self.context.current_timestamp)
        market_closing = market_open + self.marketconfig.opening_duration
        products = get_available_products(
            self.marketconfig.market_products, market_open
        )
        until = self.marketconfig.opening_hours._until
        if until and market_closing > until:
            # this market should not open, as the clearing is after the markets end time
            return

        opening_message: OpeningMessage = {
            "context": "opening",
            "market_id": self.marketconfig.market_id,
            "start_time": market_open,
            "end_time": market_closing,
            "products": products,
        }

        self.open_auctions |= set(opening_message["products"])

        for agent in self.registered_agents.keys():
            await self.context.send_message(
                create_acl(
                    opening_message,
                    receiver_addr=agent,
                    sender_addr=self.context.addr,
                    acl_metadata={
                        "reply_with": f"{self.marketconfig.market_id}_{market_open}",
                        "performative": Performatives.call_for_proposal,
                    },
                ),
                receiver_addr=agent,
            )

        # schedule closing this market
        closing_ts = datetime2timestamp(market_closing)
        self.context.schedule_timestamp_task(self.clear_market(products), closing_ts)

        # schedule the next opening too
        next_opening = self.marketconfig.opening_hours.after(market_open)
        if next_opening <= self.last_market_opening:
            next_opening_ts = datetime2timestamp(next_opening)
            self.context.schedule_timestamp_task(self.opening(), next_opening_ts)
            logger.debug(
                "market opening: %s - %s - %s",
                self.marketconfig.market_id,
                market_open,
                market_closing,
            )
        else:
            logger.debug("market %s - does not reopen", self.marketconfig.market_id)

    def validate_registration(
        self, content: RegistrationMessage, meta: MetaDict
    ) -> bool:
        """
        Validates a given registration.
        Used to check if a participant is eligible to bid on this market.

        Args:
            content (RegistrationMessage): The content of the message.
            meta (MetaDict): The metadata of the message.

        Returns:
            bool: True if the registration is valid, False otherwise.
        """
        if callable(self.marketconfig.eligible_obligations_lambda):
            requirement = self.marketconfig.eligible_obligations_lambda
        else:
            requirement = lambda_functions.get(
                self.marketconfig.eligible_obligations_lambda, lambda u: True
            )

        return all([requirement(unit_info) for unit_info in content["information"]])

    def validate_orderbook(
        self, orderbook: Orderbook, agent_addr: AgentAddress
    ) -> None:
        """
        Validates a given orderbook.

        This is needed to check if all required fields for this mechanism are present.

        Args:
            orderbook (Orderbook): The orderbook to be validated.
            agent_addr (AgentAddress): The address of the agent.

        Raises:
            ValueError: If max_price, min_price, or max_volume are unset when required.
            KeyError: If a required field is missing in an order.
            TypeError: If order['price'] or order['volume'] is not an integer when required.
        """
        max_price = self.marketconfig.maximum_bid_price
        min_price = self.marketconfig.minimum_bid_price
        max_volume = self.marketconfig.maximum_bid_volume

        market_id = self.marketconfig.market_id  # Get the market ID for error messages

        # Validate and adjust max_price, min_price, and max_volume if price_tick is set
        if self.marketconfig.price_tick:
            if max_price is None:
                raise ValueError(f"max_price is unset for market '{market_id}'.")
            if min_price is None:
                raise ValueError(f"min_price is unset for market '{market_id}'.")
            if max_volume is None:
                raise ValueError(f"max_volume is unset for market '{market_id}'.")

            # Convert prices to units based on price_tick
            max_price = math.floor(max_price / self.marketconfig.price_tick)
            min_price = math.ceil(min_price / self.marketconfig.price_tick)

        # Validate and adjust max_volume if volume_tick is set
        if self.marketconfig.volume_tick:
            if max_volume is None:
                raise ValueError(f"max_volume is unset for market '{market_id}'.")
            max_volume = math.floor(max_volume / self.marketconfig.volume_tick)

        # Validate each order in the orderbook
        for order in orderbook:
            order["agent_addr"] = agent_addr

            # Ensure 'only_hours' field is present
            if not order.get("only_hours"):
                order["only_hours"] = None

            # Check for additional required fields
            for field in self.marketconfig.additional_fields:
                if field not in order:
                    raise KeyError(
                        f"Missing required field '{field}' for order {order} in market '{market_id}'."
                    )

        # Process separated orders
        sep_orders = separate_orders(orderbook.copy())
        for order in sep_orders:
            # Adjust order price if it exceeds max_price or is below min_price
            if order["price"] > max_price:
                logger.warning(
                    f"Order price {order['price']} exceeds maximum price {max_price} in market '{market_id}'. Setting to max_price."
                )
                order["price"] = max_price
            elif order["price"] < min_price:
                logger.warning(
                    f"Order price {order['price']} is below minimum price {min_price} in market '{market_id}'. Setting to min_price."
                )
                order["price"] = min_price

            # Check that the product is part of an open auction
            product = (order["start_time"], order["end_time"], order["only_hours"])
            if product not in self.open_auctions:
                logger.warning(
                    f"Product {product} is not part of an open auction in market '{market_id}'. Skipping this order."
                )
                continue  # Skip to the next order

            # Adjust order volume if it exceeds max_volume
            if abs(order["volume"]) > max_volume:
                logger.warning(
                    f"Order volume {order['volume']} exceeds max_volume {max_volume} in market '{market_id}'. Adjusting volume."
                )
                order["volume"] = max_volume if order["volume"] > 0 else -max_volume

            # Ensure 'price' is an integer if price_tick is set
            if self.marketconfig.price_tick:
                if not isinstance(order["price"], int):
                    raise TypeError(
                        f"Order price {order['price']} must be an integer when price_tick is set in market '{market_id}'."
                    )

            # Ensure 'volume' is an integer if volume_tick is set
            if self.marketconfig.volume_tick:
                if not isinstance(order["volume"], int):
                    raise TypeError(
                        f"Order volume {order['volume']} must be an integer when volume_tick is set in market '{market_id}'."
                    )

    def handle_registration(self, content: RegistrationMessage, meta: MetaDict):
        """
        Handles incoming registration messages and adds the sender to the list of registered agents.

        This method performs the following actions:
            - Validates that the incoming message's market ID matches the current market configuration.
            - Validates the registration details using the `validate_registration` method.
            - Registers the agent if validation is successful.
            - Sends a registration reply message indicating acceptance or rejection.

        Args:
            content (RegistrationMessage): The content of the registration message.
            meta (MetaDict): The metadata of the message, including sender information.

        Raises:
            KeyError: If required keys are missing in `content` or `meta`.
        """
        agent_addr = sender_addr(meta)

        incoming_market_id = content.get("market_id")
        current_market_id = self.marketconfig.market_id

        # Validate market ID
        if incoming_market_id != current_market_id:
            logger.warning(
                f"Received registration for market '{incoming_market_id}' which does not match current market '{current_market_id}'. Registration rejected."
            )
            accepted = False
        else:
            # Validate registration details
            if self.validate_registration(content, meta):
                self.registered_agents[agent_addr] = content["information"]
                accepted = True
                logger.debug(
                    f"Agent '{agent_addr}' successfully registered for market '{current_market_id}'."
                )
            else:
                accepted = False
                logger.warning(
                    f"Agent '{agent_addr}' failed registration validation for market '{current_market_id}'."
                )

        # Prepare the registration reply message
        msg: RegistrationReplyMessage = {
            "context": "registration",
            "market_id": current_market_id,
            "accepted": accepted,
        }

        performative = (
            Performatives.accept_proposal if accepted else Performatives.reject_proposal
        )

        self.context.schedule_instant_message(
            create_acl(
                content=msg,
                receiver_addr=agent_addr,
                sender_addr=self.context.addr,
                acl_metadata={
                    "in_reply_to": meta.get("reply_with"),
                    "performative": performative,
                },
            ),
            receiver_addr=agent_addr,
        )
        logger.debug("Sent registration reply to agent '%s': %s", agent_addr, msg)

    def handle_orderbook(self, content: OrderBookMessage, meta: MetaDict):
        """
        Handles incoming order book messages, validates the order book, and adds valid orders to the list of all orders.

        If the order book is invalid or an error occurs during processing, logs the error and sends a single rejection
        message to the sender.

        Args:
            content (OrderBookMessage): The content of the message, expected to contain an 'orderbook'.
            meta (MetaDict): The metadata of the message, expected to contain 'sender_addr' and 'sender_id'.

        Raises:
            KeyError: If required keys ('orderbook', 'sender_addr', 'sender_id') are missing in the message.
            ValueError: If the order book fails validation.
            Exception: If an unexpected error occurs during processing.
        """
        try:
            # Safely retrieve required keys from 'content' and 'meta'
            orderbook: Orderbook = content.get("orderbook")
            if orderbook is None:
                raise KeyError("Missing 'orderbook' in content.")

            agent_addr = sender_addr(meta)
            # Validate the order book
            self.validate_orderbook(orderbook, agent_addr)

            # Add each validated order to 'all_orders'
            for order in orderbook:
                self.all_orders.append(order)

        except Exception as e:
            # Log the error with agent details for better traceability
            logger.error(
                f"Error handling orderbook message from agent '{agent_addr}': {e}"
            )

            # Prepare a rejection message with a generic error description
            rejection_message = {
                "context": "submit_bids",
                "message": "Rejected: Unable to process your orderbook submission due to an error.",
            }

            # Send the single rejection message back to the agent
            self.context.schedule_instant_message(
                create_acl(
                    content=rejection_message,
                    receiver_addr=agent_addr,
                    sender_addr=self.context.addr,
                    acl_metadata={
                        "in_reply_to": meta.get("reply_with", 1),
                        "performative": Performatives.refuse,
                    },
                ),
                receiver_addr=agent_addr,
            )
            logger.debug(
                f"Sent rejection message to agent '{agent_addr}': {rejection_message}"
            )

    def handle_data_request(self, content: DataRequestMessage, meta: MetaDict):
        """
        Handles incoming data request messages.

        Args:
            content (DataRequestMessage): The content of the message.
            meta (MetaDict): The metadata of the message.

        """
        metric_type = content["metric"]
        start = content["start_time"]
        end = content["end_time"]

        data = []
        try:
            import pandas as pd

            data = pd.DataFrame(self.results)
            data.index = data["time"]
            data = data[metric_type].loc[start:end]
        except Exception:
            logger.exception("Error handling data request")

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

    def handle_get_unmatched(self, content: dict, meta: MetaDict):
        """
        Sends the orderbook with unmatched orders to an agent and allows querying a subset of the orderbook.

        Args:
            content (dict): The content of the message.
            meta (MetaDict): The metadata of the message.

        Raises:
            KeyError: If required keys ('sender_addr', 'sender_id') are missing in `meta`.
        """
        try:
            order = content.get("order")
            agent_addr = sender_addr(meta)

            if order:

                def order_matches_req(o):
                    return (
                        o.get("start_time") == order.get("start_time")
                        and o.get("end_time") == order.get("end_time")
                        and o.get("only_hours") == order.get("only_hours")
                    )

                available_orders = list(filter(order_matches_req, self.all_orders))
            else:
                available_orders = self.all_orders

            self.context.schedule_instant_message(
                content={
                    "context": "get_unmatched",
                    "available_orders": available_orders,
                },
                receiver_addr=agent_addr,
            )
            logger.debug("Sent unmatched orders to agent '%s'.", agent_addr)

        except KeyError as ke:
            logger.error(f"Missing key in meta data: {ke}")
            # Optionally, handle the missing key scenario here

    async def clear_market(self, market_products: list[MarketProduct]):
        """
        This method clears the market and sends the results to the database agent.

        Args:
            market_products (list[MarketProduct]): The products to be traded.
        """
        if not self.all_orders:
            logger.warning(
                f"[{self.context.current_timestamp}] The order book for market {self.marketconfig.market_id} with products {market_products} is empty. No orders were found."
            )
            return

        try:
            (accepted_orderbook, rejected_orderbook, market_meta, flows) = self.clear(
                self.all_orders, market_products
            )
        except Exception as e:
            logger.error("clearing failed: %s", e)
            raise e

        self.all_orders = []

        for order in rejected_orderbook:
            if "accepted_volume" not in order and "accepted_price" not in order:
                if isinstance(order["volume"], dict):
                    order["accepted_volume"] = {
                        start: 0.0 for start in order["volume"].keys()
                    }
                    order["accepted_price"] = {
                        start: market_meta[i]["price"]
                        for i, start in enumerate(order["volume"].keys())
                    }
                else:
                    order["accepted_volume"] = 0.0
                    order["accepted_price"] = market_meta[0]["price"]

        self.open_auctions - set(market_products)

        accepted_orderbook = sorted(
            accepted_orderbook, key=lambda x: str(x["agent_addr"])
        )
        rejected_orderbook = sorted(
            rejected_orderbook, key=lambda x: str(x["agent_addr"])
        )

        accepted_orders = {
            agent: list(bids)
            for agent, bids in groupby(accepted_orderbook, itemgetter("agent_addr"))
        }
        rejected_orders = {
            agent: list(bids)
            for agent, bids in groupby(rejected_orderbook, itemgetter("agent_addr"))
        }

        for agent in self.registered_agents.keys():
            meta = {
                "sender_addr": self.context.addr,
                "sender_id": self.context.aid,
                "performative": Performatives.accept_proposal,
            }
            closing: ClearingMessage = {
                "context": "clearing",
                "market_id": self.marketconfig.market_id,
                "accepted_orders": accepted_orders.get(agent, []),
                "rejected_orders": rejected_orders.get(agent, []),
            }
            await self.context.send_message(
                create_acl(
                    closing,
                    acl_metadata=meta,
                    receiver_addr=agent,
                    sender_addr=self.context.addr,
                ),
                receiver_addr=agent,
            )
        # store order book in db agent
        await self.store_order_book(accepted_orderbook + rejected_orderbook)

        for meta in market_meta:
            logger.debug(
                "clearing price for %s is %.2f, volume: %f",
                self.marketconfig.market_id,
                meta.get("price", 0.0),
                meta.get("demand_volume", 0.0),
            )
            meta["market_id"] = self.marketconfig.market_id
            meta["time"] = meta["product_start"]
            self.results.append(meta)

        await self.store_market_results(market_meta)

        if flows is not None and len(flows) > 0:
            await self.store_flows(flows)

        return accepted_orderbook, market_meta

    async def store_order_book(self, orderbook: Orderbook):
        # Send a message to the OutputRole to update data in the database
        """
        Sends a message to the OutputRole to update data in the database.

        Args:
            orderbook (Orderbook): The order book to be stored.
        """

        db_addr = self.context.data.get("output_agent_addr")

        if db_addr:
            message = {
                "context": "write_results",
                "type": "market_orders",
                "market_id": self.marketconfig.market_id,
                "data": orderbook,
            }
            await self.context.send_message(
                receiver_addr=db_addr,
                content=message,
            )

    async def store_market_results(self, market_meta):
        """
        Sends a message to the OutputRole to update data in the database.

        Args:
            market_meta: The metadata of the market.
        """

        db_addr = self.context.data.get("output_agent_addr")

        if db_addr:
            message = {
                "context": "write_results",
                "type": "market_meta",
                "market_id": self.marketconfig.market_id,
                "data": market_meta,
            }
            await self.context.send_message(
                receiver_addr=db_addr,
                content=message,
            )

    async def store_flows(self, flows: dict[tuple, float]):
        """
        Sends a message to the OutputRole to update data in the database.

        Args:
            flows (flows): The electric flows between nodes to be stored.
        """

        db_addr = self.context.data.get("output_agent_addr")

        if db_addr:
            message = {
                "context": "write_results",
                "type": "grid_flows",
                "market_id": self.marketconfig.market_id,
                "data": flows,
            }
            await self.context.send_message(
                content=message,
                receiver_addr=db_addr,
            )
