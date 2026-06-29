# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Intraday-continuous (IDC) bidding strategies, modelled as **data-anchored
position adjustments** after the day-ahead (EOM) clearing.

The IDC market shares ``product_type="energy"`` and opens *after* EOM, so its
accepted volumes accumulate onto the day-ahead position via ``set_dispatch_plan``
(``outputs["energy"] += accepted_volume``). Each unit therefore bids only the
**delta** between the position it now wants -- given a near-real-time updated
forecast -- and its committed day-ahead position:

    * positive delta -> SUPPLY bid (sell more),
    * negative delta -> DEMAND bid (buy back / reduce),
    * unmatched delta -> residual imbalance.

Bids are priced at the **data-anchored IDC price signal**
``forecaster.price[<idc market id>]`` (price-taker), so the pay-as-bid clearing
settles matched volume at approximately the observed intraday price.

Drivers of the delta:
    * Renewables: the updated infeed ``forecaster.availability_intraday`` (e.g.
      realised generation used as a near-delivery proxy) vs the day-ahead
      availability that set the EOM commitment.
    * Dispatchable thermal: price arbitrage -- ramp toward full available output
      when the IDC price covers marginal cost, otherwise buy back.
    * Storage: shift discharging/charging toward the IDC price relative to the
      day-ahead reference, within SOC and power limits.
"""

from assume.common.base import (
    MinMaxChargeStrategy,
    MinMaxStrategy,
    SupportsMinMax,
    SupportsMinMaxCharge,
)
from assume.common.market_objects import MarketConfig, Orderbook, Product


def _clip(price: float, market_config: MarketConfig) -> float:
    if price >= 0:
        return min(price, market_config.maximum_bid_price)
    return max(price, market_config.minimum_bid_price)


class EnergyIntradayAdjustmentStrategy(MinMaxStrategy):
    """
    Intraday position-adjustment for power plants (thermal and renewables).

    For each (15-min) product the unit re-trades the difference between its
    profit-maximising feasible output at the IDC price and its committed
    day-ahead position. Renewables move toward their updated infeed; thermal
    units arbitrage the IDC price against marginal cost. The bid is priced at the
    IDC price signal of the market being bid into.
    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        price_signal = unit.forecaster.price.get(market_config.market_id)
        if price_signal is None:
            # no intraday price reference -> no continuous trading
            return []

        bids: list[dict] = []
        for product in product_tuples:
            start, end = product[0], product[1]

            committed = float(unit.outputs["energy"].at[start])
            price_idc = float(price_signal.at[start])

            # available (possibly updated) infeed at this step
            availability = float(unit.forecaster.availability_intraday.at[start])
            max_avail = availability * unit.max_power

            marginal_cost = float(
                unit.calculate_marginal_cost(start, max(committed, 0.0))
            )

            # price-taker target: produce up to what is available when the IDC
            # price covers marginal cost, otherwise cut back (buy back)
            target = max_avail if price_idc >= marginal_cost else 0.0
            target = min(max(target, 0.0), max_avail)

            # bound the move from the committed position by ramp limits
            up = unit.ramp_up if unit.ramp_up is not None else float("inf")
            down = unit.ramp_down if unit.ramp_down is not None else float("inf")
            desired = min(committed + up, max(committed - down, target))

            delta = desired - committed
            if abs(delta) < 1e-6:
                continue

            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": None,
                    "price": _clip(price_idc, market_config),
                    "volume": delta,
                    "node": unit.node,
                }
            )

        return self.remove_empty_bids(bids)


class StorageEnergyIntradayAdjustmentStrategy(MinMaxChargeStrategy):
    """
    Intraday position-adjustment for storage units.

    The storage shifts its position toward the IDC price relative to a day-ahead
    price reference: discharge more (sell) when the IDC price is at or above the
    reference, charge more (buy) when it is below -- always within the SOC- and
    power-feasible bounds. The traded delta is the difference to the committed
    day-ahead position, priced at the IDC signal.

    Args:
        eom_market_id (str): forecaster key for the day-ahead price reference.
            Default "EOM".
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eom_market_id = kwargs.get("eom_market_id", "EOM")

    def calculate_bids(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        price_signal = unit.forecaster.price.get(market_config.market_id)
        if price_signal is None:
            return []
        ref_signal = unit.forecaster.price.get(self.eom_market_id)

        bids: list[dict] = []
        for product in product_tuples:
            start, end = product[0], product[1]

            price_idc = float(price_signal.at[start])
            reference = (
                float(ref_signal.at[start]) if ref_signal is not None else price_idc
            )

            # SOC- and power-feasible adjustment *relative to the committed
            # day-ahead position* (calculate_min_max_* already net out base_load),
            # i.e. these are directly the delta to bid.
            _, max_discharge = unit.calculate_min_max_discharge(start, end)
            _, max_charge = unit.calculate_min_max_charge(start, end)

            # move toward the favourable side of the IDC-vs-day-ahead spread
            if price_idc >= reference:
                delta = float(max_discharge[0])  # additional discharge (>= 0)
            else:
                delta = float(max_charge[0])  # additional charge (<= 0)

            if abs(delta) < 1e-6:
                continue

            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": None,
                    "price": _clip(price_idc, market_config),
                    "volume": delta,
                    "node": unit.node,
                }
            )

        return self.remove_empty_bids(bids)
