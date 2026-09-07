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

from assume.common.fast_pandas import FastIndex
from assume.common.forecaster import UnitsOperatorForecaster
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
    create_rrule,
    datetime2timestamp,
    timestamp2datetime,
)
from assume.strategies import (
    UnitOperatorStrategy,
    UnitsOperatorDirectStrategy,
)
from assume.units import BaseUnit

logger = logging.getLogger(__name__)


class UnitsOperator(Role):
    """
    The UnitsOperator is the agent that manages the units.
    It receives the opening hours of the market and sends back the bids for the market.

    Attributes:
        available_markets (list[MarketConfig]): The available markets.
        registered_markets (dict[str, MarketConfig]): The registered markets.
        last_sent_market_dispatch (dict[str, int]): The time until which the market dispatch was sent, per market.
        last_executed_dispatch (int): The timestamp of the last executed dispatch.
        portfolio_strategies (UnitOperatorStrategy): The portfolio strategy.
        valid_orders (defaultdict): The valid orders, per market.
        pending_orders (defaultdict): The orders awaiting the end of their delivery period, per market.
        units (dict[str, BaseUnit]): The units.
        id (str): The id of the agent.
        context (Context): The context of the agent.

    Args:
        available_markets (list[MarketConfig]): The available markets.
        portfolio_strategies (dict[str, UnitOperatorStrategy], optional): Optimized portfolio strategy. Defaults to an empty dict.
        forecaster (UnitsOperatorForecaster, optional): Operator-level forecaster providing market
            price and residual load forecasts shared across the operator's units. Defaults to None.
    """

    def __init__(
        self,
        available_markets: list[MarketConfig],
        portfolio_strategies: dict[str, UnitOperatorStrategy] = {},
        forecaster: UnitsOperatorForecaster = None,
    ):
        super().__init__()

        self.available_markets = available_markets
        self.registered_markets: dict[str, MarketConfig] = {}

        self.last_sent_market_dispatch = defaultdict(lambda: 0)
        self.last_executed_dispatch = 0

        self.forecaster = forecaster

        self.portfolio_strategies = portfolio_strategies
        for market in self.available_markets:
            if market.market_id not in self.portfolio_strategies.keys():
                self.portfolio_strategies[market.market_id] = (
                    UnitsOperatorDirectStrategy()
                )

        # valid_orders per market_id, used for the market dispatch export at clearing (before delivery)
        self.valid_orders = defaultdict(list)
        # pending_orders per market_id awaiting the end of their delivery period to calculate reward based on actual dispatch
        self.pending_orders = defaultdict(list)
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

    @property
    def simulation_index(self) -> FastIndex | None:
        """
        The shared simulation index, taken from the operator forecaster if one is
        given and from any of the managed units otherwise.

        Returns:
            FastIndex | None: The simulation index, or None if the operator has no units.
        """
        if self.forecaster is not None:
            return self.forecaster.index
        if self.units:
            return next(iter(self.units.values())).index
        return None

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

        # execute the dispatch once per time step, independent of the markets.
        index = self.simulation_index
        if index is not None:
            self.context.schedule_recurrent_task(
                self.execute_dispatch,
                create_rrule(
                    start=index.start + index.freq,
                    end=index.end,
                    freq=index.freq,
                ),
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
        self.valid_orders[marketconfig.market_id].extend(orderbook)
        self.set_unit_dispatch(orderbook, marketconfig)

        # the cashflow is fully determined by the clearing result and can be booked now
        self.calculate_unit_cashflow(orderbook, marketconfig)

        # the reward, in contrast, depends on the dispatch that is actually realized
        self.pending_orders[marketconfig.market_id].extend(orderbook)

        self.write_market_dispatch(marketconfig)

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

    def calculate_unit_cashflow(
        self, orderbook: Orderbook, marketconfig: MarketConfig
    ) -> None:
        """
        Books the cashflow of the given market result in the units.

        The cashflow follows directly from the accepted prices and volumes and is
        therefore known as soon as the market is cleared.

        Args:
            orderbook (Orderbook): The orderbook of the market.
            marketconfig (MarketConfig): The market configuration.
        """
        orderbook.sort(key=itemgetter("unit_id"))
        for unit_id, orders in groupby(orderbook, itemgetter("unit_id")):
            self.units[unit_id].calculate_cashflow(
                product_type=marketconfig.product_type,
                orderbook=list(orders),
            )

    def calculate_unit_reward(self, execute_until: datetime) -> None:
        """
        Calculates the reward of all orders whose delivery period has ended.

        This is called from execute_dispatch, so the reward is based on the dispatch
        that was actually realized, including the contribution of all markets that
        cleared for the respective delivery period.

        Args:
            execute_until (datetime.datetime): The last time step which will be executed.
        """
        freq = self.simulation_index.freq

        # find all orders due for execution because their delivery period has ended
        due: dict[str, Orderbook] = {}
        for market_id, orders in list(self.pending_orders.items()):
            delivered = [
                order for order in orders if order["end_time"] - freq <= execute_until
            ]
            if delivered:
                due[market_id] = delivered
                self.pending_orders[market_id] = [
                    order
                    for order in orders
                    if order["end_time"] - freq > execute_until
                ]

        for market_id, orders in due.items():
            marketconfig = self.registered_markets[market_id]

            orders.sort(key=itemgetter("unit_id"))
            for unit_id, unit_orders in groupby(orders, itemgetter("unit_id")):
                self.units[unit_id].calculate_reward(
                    marketconfig=marketconfig,
                    orderbook=list(unit_orders),
                )

            # Calculate reward for the portfolio strategy
            self.portfolio_strategies.get(market_id).calculate_reward(
                units_operator=self,
                marketconfig=marketconfig,
                orderbook=orders,
            )

    async def execute_dispatch(self) -> None:
        """
        Executes the dispatch of all units for the time steps which have passed since
        the last execution, exports it and calculates the reward of every product
        whose delivery period ended in the meantime.
        """
        now = timestamp2datetime(self.context.current_timestamp)
        execute_until = now - self.simulation_index.freq

        last_ts = self.last_executed_dispatch
        self.last_executed_dispatch = datetime2timestamp(execute_until)

        try:
            # add one second to exclude the first time stamp,
            # because it is already executed in the last step
            actual_dispatch = self.get_actual_dispatch(
                timestamp2datetime(last_ts + 1), execute_until
            )
            self.write_actual_dispatch(actual_dispatch)

            # now that the dispatch is realized, the reward of every product whose
            # delivery period has been executed can be calculated
            self.calculate_unit_reward(execute_until)
        except (
            AttributeError,
            NameError,
            TypeError,
        ):  # TODO: check if this is intended error behavior
            # these indicate a coding error rather than a problem with the data,
            # so they must not be hidden in the log
            raise
        except Exception:
            # any other exception escaping here would stop the recurrent task,
            # which would silently disable the dispatch for the rest of the run
            logger.exception("error while executing the dispatch at %s", now)

    def get_actual_dispatch(self, start: datetime, end: datetime) -> list[dict]:
        """
        Retrieves the actual dispatch of all units in the given time range and commits
        it in the unit. This checks the feasibility of the planned dispatch and adjusts
        it to the closest feasible one if needed, so it has to happen once the time
        range has passed and all markets for it have cleared.

        The actual dispatch is the volume a unit really dispatched across all of the
        markets it participated in, which is why this is independent of the product
        type and can only be determined after the delivery period.

        Args:
            start (datetime.datetime): The start of the range to execute.
            end (datetime.datetime): The end of the range to execute, inclusive.

        Returns:
            list[dict]: the unit_dispatch dataframes
        """
        unit_dispatch = []
        for unit_id, unit in self.units.items():
            current_dispatch = unit.execute_current_dispatch(start, end)
            dispatch = {"power": current_dispatch}
            unit.calculate_generation_cost(start, end, "energy")
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

        return unit_dispatch

    def get_market_dispatch(
        self, market_id: str, last: datetime, until: datetime
    ) -> list[tuple[datetime, float, str, str]]:
        """
        Aggregates the accepted orders of the given market into the dispatch per unit.

        Args:
            market_id (str): The market for which this is done.
            last (datetime.datetime): The last date until which the dispatch was already sent.
            until (datetime.datetime): The date up to which the dispatch is
                aggregated, exclusive.

        Returns:
            list[tuple[datetime, float, str, str]]: the market_dispatch dataframe
        """
        return aggregate_step_amount(
            orderbook=self.valid_orders[market_id],
            begin=last,
            end=until,
            groupby=["market_id", "unit_id"],
        )

    def write_actual_dispatch(self, actual_dispatch: list[dict]) -> None:
        """
        Sends the actual dispatch of the units to the output agent.

        Args:
            actual_dispatch (list[dict]): The unit dispatch dataframes.

        """
        db_addr = self.context.data.get("output_agent_addr")
        if db_addr and actual_dispatch:
            self.context.schedule_instant_message(
                receiver_addr=db_addr,
                content={
                    "context": "write_results",
                    "type": "unit_dispatch",
                    "data": actual_dispatch,
                },
            )

    def write_market_dispatch(self, marketconfig: MarketConfig) -> None:
        """
        Sends the aggregated market dispatch curve of the given market to the output agent.
        This has to be called at a clearing of the given market, as the dispatch which is
        final by now is derived from the opening this clearing belongs to.

        Args:
            marketconfig (MarketConfig): The market configuration.
        """

        now = timestamp2datetime(self.context.current_timestamp)
        # the clearing belongs to the opening one opening duration ago, and an opening
        # after the last possible one is never scheduled by the market
        next_opening = marketconfig.opening_hours.after(
            now - marketconfig.opening_duration
        )
        if next_opening is not None and next_opening <= marketconfig.last_opening:
            until = now
        else:
            # no export follows which could aggregate the rest, so the dispatch is final
            # until the market end. The closing delta there lies beyond the simulation
            # and is excluded by the aggregation.
            until = marketconfig.opening_hours._until

        market_id = marketconfig.market_id
        last = timestamp2datetime(self.last_sent_market_dispatch[market_id])
        if until <= last:
            # stop if nothing became final since the last export
            return
        self.last_sent_market_dispatch[market_id] = datetime2timestamp(until)

        market_dispatch = self.get_market_dispatch(market_id, last, until)

        # orders have to be kept until their closing delta has been aggregated,
        # which only happens in the export following their end_time
        self.valid_orders[market_id] = list(
            filter(
                lambda x: x["end_time"] > until,
                self.valid_orders[market_id],
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

    async def submit_bids(self, opening: OpeningMessage, meta: MetaDict) -> None:
        """
        Formulates an orderbook and sends it to the market.

        Args:
            opening (OpeningMessage): The opening message.
            meta (MetaDict): The meta data of the market.

        """

        products = opening["products"]
        market = self.registered_markets[opening["market_id"]]
        logger.debug("%s setting bids for %s - %s", self.id, market.market_id, products)

        # the given products just became available on our market
        # and we need to provide bids
        # [whole_next_hour, quarter1, quarter2, quarter3, quarter4]
        # algorithm should buy as much baseload as possible, then add up with quarters
        products.sort(key=lambda p: (p[0] - p[1], p[0]))
        strategy = self.portfolio_strategies.get(
            opening["market_id"],
        )
        orderbook = strategy.calculate_bids(
            units_operator=self,
            market_config=market,
            product_tuples=products,
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
