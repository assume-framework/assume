# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from dateutil import rrule as rr

from assume.common.market_objects import (
    MarketConfig,
    Product,
    lambda_functions,
)

logger = logging.getLogger(__name__)


@dataclass
class MarketInfo:
    """
    Everything a multi-market strategy needs to know about one market.

    Contains the market configuration itself plus the chronological
    schedule of its opening windows (gate closures) and the delivery
    windows of its products.

    Args:
        market_config (MarketConfig): The configuration of the market.
        product_type (str): The product type traded at this market
            (e.g. "energy", "capacity_pos", "capacity_neg").
        openings (list[datetime]): The chronological opening (start) times
            of the market. Gate closure is opening time + opening_duration.
        deliveries (list[tuple[datetime, datetime]]): The chronological
            delivery windows of the products traded at this market.
        gate_closures (dict[tuple[datetime, datetime], datetime]): Maps each
            delivery window (start, end) to the gate closure (i.e. the end of
            the opening window) after which its bid is fixed.
    """

    market_config: MarketConfig
    product_type: str
    openings: list[datetime] = field(default_factory=list)
    deliveries: list[tuple[datetime, datetime]] = field(default_factory=list)
    gate_closures: dict[tuple[datetime, datetime], datetime] = field(
        default_factory=dict
    )

    @property
    def market_id(self) -> str:
        return self.market_config.market_id

    def next_opening(self, now: datetime) -> tuple[datetime, datetime] | None:
        """
        Returns the next opening window (start, gate closure) at or after `now`,
        or None if the market does not open anymore.
        """
        for opening in self.openings:
            if opening >= now:
                return opening, opening + self.market_config.opening_duration
        return None

    def open_at(self, now: datetime) -> bool:
        """
        Returns True if the market is open for bidding at time `now`.
        """
        duration = self.market_config.opening_duration
        return any(opening <= now < opening + duration for opening in self.openings)


def _is_eligible(unit, market_config: MarketConfig) -> bool:
    """
    Checks whether the unit is eligible to participate in the given market.

    Mirrors the checks done in UnitsOperator.participate: the unit must have
    a bidding strategy registered for the market and pass the market's
    eligible_obligations_lambda (either a callable or a named function from
    assume.common.market_objects.lambda_functions).
    """
    if market_config.market_id not in unit.bidding_strategies:
        return False

    if callable(market_config.eligible_obligations_lambda):
        requirement = market_config.eligible_obligations_lambda
    else:
        requirement = lambda_functions.get(
            market_config.eligible_obligations_lambda, lambda u: True
        )

    return requirement(unit.as_dict())


def _rrule_dates(rrule: rr.rrule, start: datetime, end: datetime) -> list[datetime]:
    """
    Returns all occurrences of the rrule between start and end (inclusive).
    Uses the rrule cache if available.
    """
    return list(rrule.between(start, end, inc=True))


def discover_markets(
    unit,
    available_markets: list[MarketConfig],
    start: datetime,
    end: datetime,
) -> dict[str, MarketInfo]:
    """
    Discovers all markets the given unit can participate in and builds their
    chronological schedules.

    Should be called once at the beginning of the simulation. For each market
    where the unit is eligible, the opening windows (bidding windows ending in
    a gate closure) and the delivery windows of the traded products are
    extracted from the market configuration.

    Args:
        unit (BaseUnit): The unit that wants to bid on the markets.
        available_markets (list[MarketConfig]): All markets available in the world.
        start (datetime.datetime): The start of the simulation horizon.
        end (datetime.datetime): The end of the simulation horizon.

    Returns:
        dict[str, MarketInfo]: Mapping of market_id to the market information,
            sorted by the first gate closure of the market.
    """
    market_infos: dict[str, MarketInfo] = {}

    for market_config in available_markets:
        if not _is_eligible(unit, market_config):
            logger.debug(
                "unit %s not eligible for market %s",
                unit.id,
                market_config.market_id,
            )
            continue

        openings = _rrule_dates(
            market_config.opening_hours, start, end
        )
        if not openings:
            logger.debug(
                "market %s has no openings in simulation horizon",
                market_config.market_id,
            )
            continue

        # delivery windows of all products traded at this market
        deliveries: list[tuple[datetime, datetime]] = []
        for opening in openings:
            for product in market_config.market_products:
                first_delivery = opening + product.first_delivery
                for i in range(product.count):
                    period_start = first_delivery + product.duration * i
                    period_end = first_delivery + product.duration * (i + 1)
                    deliveries.append((period_start, period_end))

        market_infos[market_config.market_id] = MarketInfo(
            market_config=market_config,
            product_type=market_config.product_type,
            openings=openings,
            deliveries=deliveries,
            gate_closures={
                (period_start, period_end): opening + market_config.opening_duration
                for opening in openings
                for period_start, period_end in (
                    (
                        opening + product.first_delivery + product.duration * i,
                        opening + product.first_delivery + product.duration * (i + 1),
                    )
                    for product in market_config.market_products
                    for i in range(product.count)
                )
            },
        )

    # sort markets by their first gate closure so that the strategy
    # commits bids in the chronological order of the markets
    sorted_infos = sorted(
        market_infos.values(),
        key=lambda info: info.openings[0],
    )
    return {info.market_id: info for info in sorted_infos}


def get_market_events(
    market_infos: dict[str, MarketInfo],
) -> list[tuple[datetime, str, str]]:
    """
    Builds a chronological event list over all discovered markets.

    Each event is a tuple of (time, event_type, market_id) where event_type
    is one of "opening", "gate_closure" or "delivery_start". This defines
    the order in which markets open, close and start delivering within the
    foresight window.

    Args:
        market_infos (dict[str, MarketInfo]): The discovered markets.

    Returns:
        list[tuple[datetime, str, str]]: Chronologically sorted market events.
    """
    events: list[tuple[datetime, str, str]] = []
    for info in market_infos.values():
        duration = info.market_config.opening_duration
        for opening in info.openings:
            events.append((opening, "opening", info.market_id))
            events.append((opening + duration, "gate_closure", info.market_id))
        for delivery_start, _ in info.deliveries:
            events.append((delivery_start, "delivery_start", info.market_id))

    events.sort(key=lambda event: (event[0], event[1], event[2]))
    return events


def get_products_at_time(
    market_info: MarketInfo,
    now: datetime,
) -> list[Product]:
    """
    Returns the products of a market that can still be bid on at time `now`.

    A product can be bid on if its market is open at `now` and its delivery
    period has not started yet (i.e. the delivery start is after the gate
    closure following `now`).

    Args:
        market_info (MarketInfo): The market to get the products for.
        now (datetime.datetime): The current decision time.

    Returns:
        list[Product]: The products available for bidding.
    """
    market_config = market_info.market_config
    if not market_info.open_at(now):
        return []

    # the gate closure of the currently open window
    gate_closure = None
    for opening in market_info.openings:
        closure = opening + market_config.opening_duration
        if opening <= now < closure:
            gate_closure = closure
            break
    if gate_closure is None:
        return []

    products: list[Product] = []
    for opening in market_info.openings:
        if opening != now:
            continue
        for product in market_config.market_products:
            first_delivery = opening + product.first_delivery
            for i in range(product.count):
                period_start = first_delivery + product.duration * i
                period_end = first_delivery + product.duration * (i + 1)
                if period_start >= gate_closure:
                    products.append(
                        Product(start=period_start, end=period_end, only_hours=None)
                    )
    return products
