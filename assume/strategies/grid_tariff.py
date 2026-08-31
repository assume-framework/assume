# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bidding strategies for a grid-fee (tariff) market.

The tariff market is an ordinary ``pay_as_clear`` market with a non-energy
``product_type`` (``grid_fee``) that opens ahead of the EOM.  A DSO unit is the
sole supplier and submits a stepwise supply curve; every flexible unit behind
the transformer submits a price-taking demand bid for its planned withdrawal.

Because ``pay_as_clear`` writes ``accepted_price`` on accepted *and* rejected
orders alike, the clearing price reaches every participant through the standard
clearing path and lands in ``unit.outputs["grid_fee_accepted_price"]``.  No new
market mechanism is required.
"""

import logging

from assume.common.base import BaseUnit, MinMaxStrategy
from assume.common.market_objects import MarketConfig, Orderbook, Product

logger = logging.getLogger(__name__)

# Marker output written by GridFeeAnnouncementStrategy for every hour it has
# received a clearing for.  The forecast update algorithm uses it to tell
# "announced fee of 0" apart from "no announcement yet".
ANNOUNCED_FLAG = "grid_fee_announced"


class GridFeeDsoStrategy(MinMaxStrategy):
    """Supply-side bidding strategy of the DSO on the grid-fee market.

    The submitted supply curve *is* the tariff: ``pay_as_clear`` sets the
    uniform price to the highest accepted supply step, so the fee published to
    all participants is a function of the announced aggregate withdrawal.

    With ``grid_fee_headroom`` unset the strategy emits a single unlimited block
    at ``grid_fee_base`` - a constant volumetric tariff.  Setting
    ``grid_fee_headroom`` adds a second, effectively unlimited block at
    ``grid_fee_base + grid_fee_scarcity`` which only clears once announced
    demand exceeds the headroom.

    Args:
        grid_fee_base: Baseline network fee in EUR/MWh charged up to the headroom.
        grid_fee_scarcity: Adder in EUR/MWh on top of the baseline for demand
            beyond the headroom. Ignored when ``grid_fee_headroom`` is None.
        grid_fee_headroom: Volume in MW available at the baseline fee, normally
            ``s_nom - forecast_inflexible_load``. None means "no scarcity step".
        grid_fee_peak_adder: Time-of-use adder in EUR/MWh applied to every block
            during ``grid_fee_peak_hours``.
        grid_fee_peak_hours: Hours of the day the adder applies to, e.g.
            ``"17,18,19,20"`` or ``[17, 18, 19, 20]``.

    Note:
        A fee that is *constant over time* cannot change the schedule of a fleet
        whose total energy is fixed by its trips: it adds the same amount to the
        charging cost and to the discharging revenue in every hour. Use the
        time-of-use adder or the scarcity step to get a signal that actually
        shifts load.
    """

    def __init__(
        self,
        *args,
        grid_fee_base: float = 0.0,
        grid_fee_scarcity: float = 0.0,
        grid_fee_headroom: float | None = None,
        grid_fee_peak_adder: float = 0.0,
        grid_fee_peak_hours: str | list[int] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.grid_fee_base = float(grid_fee_base)
        self.grid_fee_scarcity = float(grid_fee_scarcity)
        self.grid_fee_headroom = (
            None if grid_fee_headroom is None else float(grid_fee_headroom)
        )
        self.grid_fee_peak_adder = float(grid_fee_peak_adder)
        if grid_fee_peak_hours is None:
            self.grid_fee_peak_hours = set()
        elif isinstance(grid_fee_peak_hours, str):
            self.grid_fee_peak_hours = {
                int(h) for h in grid_fee_peak_hours.split(",") if h.strip()
            }
        else:
            self.grid_fee_peak_hours = {int(h) for h in grid_fee_peak_hours}

    def base_fee_at(self, start) -> float:
        """Baseline fee in EUR/MWh for the product starting at *start*."""
        if start.hour in self.grid_fee_peak_hours:
            return self.grid_fee_base + self.grid_fee_peak_adder
        return self.grid_fee_base

    def headroom_at(self, unit: BaseUnit, start) -> float | None:
        """Headroom in MW offered at the baseline fee for the product at *start*.

        Extension point: a subclass reading ``unit.forecaster.residual_load`` can
        make the headroom time-varying without touching the bid construction.
        """
        return self.grid_fee_headroom

    def calculate_bids(
        self,
        unit: BaseUnit,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        max_volume = market_config.maximum_bid_volume
        bids: Orderbook = []

        for product in product_tuples:
            start, end, only_hours = product[0], product[1], product[2]
            headroom = self.headroom_at(unit, start)
            base = self.base_fee_at(start)

            if headroom is None:
                blocks = [(max_volume, base)]
            else:
                headroom = max(0.0, min(float(headroom), max_volume))
                blocks = [
                    (headroom, base),
                    (max_volume - headroom, base + self.grid_fee_scarcity),
                ]

            for volume, price in blocks:
                bids.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": only_hours,
                        "price": price,
                        "volume": volume,
                        "node": unit.node,
                    }
                )

        return self.remove_empty_bids(bids)


class GridFeeAnnouncementStrategy(MinMaxStrategy):
    """Demand-side bidding strategy on the grid-fee market.

    Announces the unit's already-planned withdrawal for the delivery hour and
    bids it at the market cap, so the unit is a price taker on the fee by
    construction.  It deliberately does **not** trigger a re-optimisation: the
    plan announced here is the one the last optimisation produced, and the fee
    it receives is what the next optimisation reacts to.

    Every hour is bid with at least ``token_volume``.  Zero-volume orders are
    dropped by the clearing, and because ``accepted_price`` is written by
    assignment rather than accumulation an unbid hour would silently keep a
    stale fee.

    Args:
        token_volume: Minimum announced volume in MW, used for hours in which
            the unit plans no withdrawal.
    """

    def __init__(self, *args, token_volume: float = 1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_volume = float(token_volume)

    def planned_withdrawal(self, unit: BaseUnit, start) -> float:
        """Planned net withdrawal in MW for the product starting at *start*.

        Prefers ``planned_power_requirement`` - the provisional schedule over the
        whole look-ahead window - because the tariff market delivers further
        ahead than a rolling-horizon unit commits, so ``opt_power_requirement``
        is generally still stale for the announced hour.
        """
        for attr in ("planned_power_requirement", "opt_power_requirement"):
            plan = getattr(unit, attr, None)
            if plan is None:
                continue
            try:
                return float(plan.at[start])
            except (KeyError, IndexError, ValueError, TypeError):
                continue
        # No optimisation has run yet (first market round).
        return 0.0

    def calculate_bids(
        self,
        unit: BaseUnit,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        bids: Orderbook = []

        for product in product_tuples:
            start, end, only_hours = product[0], product[1], product[2]
            withdrawal = max(self.planned_withdrawal(unit, start), self.token_volume)
            volume = min(withdrawal, market_config.maximum_bid_volume)

            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": only_hours,
                    # price taker: accept any fee the DSO's curve produces
                    "price": market_config.maximum_bid_price,
                    "volume": -volume,
                    "node": unit.node,
                }
            )

        return bids

    def calculate_reward(
        self,
        unit: BaseUnit,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ) -> None:
        """Mark the hours for which a fee has actually been published.

        Runs after ``set_dispatch_plan`` has written
        ``grid_fee_accepted_price``, so the flag and the price are always
        consistent.
        """
        for order in orderbook:
            end_excl = order["end_time"] - unit.index.freq
            unit.outputs[ANNOUNCED_FLAG].loc[order["start_time"] : end_excl] = 1.0
