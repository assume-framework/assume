# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import timedelta

import numpy as np
import pandas as pd

from assume.common.base import BaseStrategy, SupportsMinMaxCharge
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.common.utils import parse_duration


class flexableEOMStorage(BaseStrategy):
    """
    The strategy is analogue to the storage strategy in flexABLE.

    If the current price forecast is higher than the average price forecast for a given foresight,
    the unit will discharge.
    The price is then set as the average price divided by the discharge efficiency of the unit.
    Otherwise, the unit will charge with the price defined as the average price multiplied by the charge efficiency of the unit.

    Attributes:
        foresight (datetime.timedelta): Foresight for the average price calculation.

    Args:
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.foresight = parse_duration(kwargs.get("eom_foresight", "24h"))

    def calculate_bids(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market.

        Args:
            unit (SupportsMinMaxCharge): The unit that is dispatched.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): List of product tuples.
            **kwargs: Additional keyword arguments.

        Returns:
            Orderbook: Bids containing start_time, end_time, only_hours, price, volume.

        Note:
            The strategy is analogue to flexABLE
        """

        # =============================================================================
        # Storage Unit is either charging, discharging, or off
        # =============================================================================
        start = product_tuples[0][0]
        end_all = product_tuples[-1][1]

        previous_power = unit.get_output_before(start)

        # save a theoretic SOC to calculate the ramping
        theoretic_SOC = unit.get_soc_before(start)

        # calculate min and max power for charging and discharging
        min_power_charge_values, max_power_charge_values = (
            unit.calculate_min_max_charge(start, end_all)
        )
        min_power_discharge_values, max_power_discharge_values = (
            unit.calculate_min_max_discharge(start, end_all, soc=theoretic_SOC)
        )

        # =============================================================================
        # Calculate bids
        # =============================================================================
        bids = []

        for (
            product,
            max_power_discharge,
            min_power_discharge,
            max_power_charge,
            min_power_charge,
        ) in zip(
            product_tuples,
            max_power_discharge_values,
            min_power_discharge_values,
            max_power_charge_values,
            min_power_charge_values,
        ):
            start, end = product[0], product[1]

            current_power = unit.outputs["energy"].at[start]
            current_power_discharge = max(current_power, 0)
            current_power_charge = min(current_power, 0)

            # Calculate ramping constraints using helper function
            max_power_discharge = unit.calculate_ramp_discharge(
                theoretic_SOC,
                previous_power,
                max_power_discharge,
                current_power_discharge,
                min_power_discharge,
            )
            min_power_discharge = unit.calculate_ramp_discharge(
                theoretic_SOC,
                previous_power,
                min_power_discharge,
                current_power_discharge,
                min_power_discharge,
            )
            max_power_charge = unit.calculate_ramp_charge(
                theoretic_SOC,
                previous_power,
                max_power_charge,
                current_power_charge,
                min_power_charge,
            )
            min_power_charge = unit.calculate_ramp_charge(
                theoretic_SOC,
                previous_power,
                min_power_charge,
                current_power_charge,
                min_power_charge,
            )
            price_forecast = unit.forecaster[f"price_{market_config.market_id}"]

            # calculate average price
            average_price = calculate_price_average(
                current_time=start,
                foresight=self.foresight,
                price_forecast=price_forecast,
            )

            # if price is higher than average price, discharge
            # if price is lower than average price, charge
            # if price forecast favors discharge, but max discharge is zero, set a bid for charging
            if price_forecast[start] >= average_price and max_power_discharge:
                price = average_price / unit.efficiency_discharge
                bid_quantity = max_power_discharge
            else:
                price = average_price * unit.efficiency_charge
                bid_quantity = max_power_charge

            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": None,
                    "price": price,
                    "volume": bid_quantity,
                    "node": unit.node,
                }
            )

            # calculate theoretic SOC
            time_delta = (end - start) / timedelta(hours=1)
            if bid_quantity + current_power > 0:
                delta_soc = -(
                    (bid_quantity + current_power)
                    * time_delta
                    / unit.efficiency_discharge
                )
            elif bid_quantity + current_power < 0:
                delta_soc = -(
                    (bid_quantity + current_power) * time_delta * unit.efficiency_charge
                )
            else:
                delta_soc = 0

            theoretic_SOC += delta_soc
            previous_power = bid_quantity + current_power

        bids = self.remove_empty_bids(bids)

        return bids

    def calculate_reward(
        self,
        unit: SupportsMinMaxCharge,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ):
        """
        Calculates the reward (costs and profit).

        The profit is defined by the cashflow minus the costs.

        Args:
            unit (SupportsMinMaxCharge): The unit to calculate reward for.
            marketconfig (MarketConfig): The market configuration.
            orderbook (Orderbook): The orderbook.
        """
        product_type = marketconfig.product_type

        for order in orderbook:
            start = order["start_time"]
            # end includes the end of the last product, to get the last products' start time we deduct the frequency once
            end_excl = order["end_time"] - unit.index.freq

            # Extract outputs and costs in one step
            outputs = unit.outputs[product_type].loc[start:end_excl]
            costs = np.where(
                outputs != 0,
                np.abs(outputs)
                * np.array([unit.calculate_marginal_cost(start, x) for x in outputs]),
                0,
            )

            unit.outputs["profit"].loc[start:end_excl] = (
                unit.outputs[f"{product_type}_cashflow"].loc[start:end_excl] - costs
            )
            unit.outputs["total_costs"].loc[start:end_excl] = costs


class flexablePosCRMStorage(BaseStrategy):
    """
    The strategy is analogue to the storage strategy in flexABLE.

    The strategy bids the energy_price for the energy_pos product if the specific revenue is positive.
    Otherwise, the strategy bids the capacity_price for the capacity_pos product.

    Attributes:
        foresight (datetime.timedelta): Foresight for the average price calculation.

    Args:
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check if kwargs contains crm_foresight argument
        self.foresight = parse_duration(kwargs.get("crm_foresight", "4h"))

    def calculate_bids(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Calculates bids for the positive CRM market.

        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market and returns bids
        containing start_time, end_time, only_hours, price, volume.

        Args:
            unit (SupportsMinMaxCharge): The unit that is dispatched.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): List of product tuples.
            **kwargs: Additional keyword arguments.

        Returns:
            Orderbook: A list of bids.
        """
        start = product_tuples[0][0]
        end = product_tuples[-1][1]

        previous_power = unit.get_output_before(start)
        theoretic_SOC = unit.get_soc_before(start)

        _, max_power_discharge_values = unit.calculate_min_max_discharge(
            start, end, soc=theoretic_SOC
        )
        bids = []

        for product, max_power_discharge in zip(
            product_tuples, max_power_discharge_values
        ):
            start = product[0]
            current_power = unit.outputs["energy"].at[start]

            # calculate ramping constraints for discharge
            bid_quantity = unit.calculate_ramp_discharge(
                theoretic_SOC,
                previous_power,
                max_power_discharge,
                current_power,
            )

            if bid_quantity == 0:
                continue

            marginal_cost = unit.calculate_marginal_cost(
                start=start, power=bid_quantity
            )

            specific_revenue = get_specific_revenue(
                unit=unit,
                marginal_cost=marginal_cost,
                t=start,
                foresight=self.foresight,
                price_forecast=unit.forecaster[f"price_{market_config.market_id}"],
            )

            # if specific revenue is positive, bid specific_revenue
            if specific_revenue >= 0:
                capacity_price = specific_revenue
            else:
                capacity_price = (
                    abs(specific_revenue) * unit.min_power_discharge / bid_quantity
                )

            energy_price = capacity_price

            if market_config.product_type == "capacity_pos":
                bids.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": None,
                        "price": capacity_price,
                        "volume": bid_quantity,
                        "node": unit.node,
                    }
                )
                previous_power = current_power

            elif market_config.product_type == "energy_pos":
                bids.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": None,
                        "price": energy_price,
                        "volume": bid_quantity,
                        "node": unit.node,
                    }
                )
                # calculate theoretic SOC
                time_delta = (end - start) / timedelta(hours=1)
                delta_soc = -(
                    (bid_quantity + current_power)
                    * time_delta
                    / unit.efficiency_discharge
                )
                theoretic_SOC += delta_soc
                previous_power = bid_quantity + current_power
            else:
                previous_power = current_power
                raise ValueError(
                    f"Product {market_config.product_type} is not supported by this strategy."
                )

        bids = self.remove_empty_bids(bids)

        return bids


class flexableNegCRMStorage(BaseStrategy):
    """
    A strategy that bids the energy_price or the capacity_price of the unit on the negative CRM(reserve market).

    Attributes:
        foresight (datetime.timedelta): Foresight for the average price calculation.

    Args:
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.foresight = parse_duration(kwargs.get("crm_foresight", "4h"))

    def calculate_bids(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market.
        Returns a list of bids consisting of the start time, end time, only hours, price and volume.

        Args:
            unit (SupportsMinMax): A unit that the unit operator manages.
            market_config (MarketConfig): A market configuration.
            product_tuples (list[Product]): A list of tuples containing the start and end time of each product.
            kwargs (dict): Additional arguments.

        Returns:
            Orderbook: A list of bids.
        """
        start = product_tuples[0][0]
        end = product_tuples[-1][1]

        previous_power = unit.get_output_before(start)

        theoretic_SOC = unit.get_soc_before(start)

        _, max_power_charge_values = unit.calculate_min_max_charge(start, end)

        bids = []
        for product, max_power_charge in zip(product_tuples, max_power_charge_values):
            start = product[0]
            current_power = unit.outputs["energy"].at[start]
            bid_quantity = abs(
                unit.calculate_ramp_charge(
                    theoretic_SOC,
                    previous_power,
                    max_power_charge,
                    current_power,
                )
            )

            # if bid_quantity >= min_bid_volume  --> not checked here
            if bid_quantity == 0:
                previous_power = current_power
                continue

            if market_config.product_type == "capacity_neg":
                bids.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": None,
                        "price": 0,
                        "volume": bid_quantity,
                        "node": unit.node,
                    }
                )
                previous_power = current_power

            elif market_config.product_type == "energy_neg":
                bids.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": None,
                        "price": 0,
                        "volume": bid_quantity,
                        "node": unit.node,
                    }
                )
                # calculate theoretic SOC
                time_delta = (end - start) / timedelta(hours=1)
                delta_soc = (
                    (bid_quantity + current_power) * time_delta * unit.efficiency_charge
                )
                theoretic_SOC += delta_soc
                previous_power = bid_quantity + current_power
            else:
                previous_power = current_power
                raise ValueError(
                    f"Product {market_config.product_type} is not supported by this strategy."
                )

        bids = self.remove_empty_bids(bids)

        return bids

class flexableRedispatchStorage(BaseStrategy):
    """
    Redispatch for storage, flexABLE-style but robust to forecast/index shapes.

    - Aligns all time series to unit.outputs.index (a pandas DatetimeIndex).
    - Two bids per hour if feasible:
        * UP   (discharge, +MW) at price ≈ avg_price / efficiency_discharge
        * DOWN (charge,    −MW) at price ≈ avg_price * efficiency_charge
    - Respects SoC, limits, and ramps; advances a theoretical SoC only by the base profile.
    - Sign convention: +MW = discharge (supply), −MW = charge (demand).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.foresight = parse_duration(kwargs.get("eom_foresight", "24h"))
        self.lookahead = parse_duration(kwargs.get("lookahead", "0h"))

    # ---------- helpers ----------

    def _unit_dt_index(self, unit):
        """
        Always use a pandas.DatetimeIndex.
        Prefer unit.outputs.index (flexABLE style); fall back sensibly.
        """
        try:
            idx = unit.outputs.index
            if isinstance(idx, pd.DatetimeIndex):
                return idx
        except Exception:
            pass

        # Try to coerce other index-like attributes
        for candidate in (getattr(unit, "index", None),
                          getattr(unit, "snapshots", None)):
            if candidate is None:
                continue
            try:
                idx = pd.DatetimeIndex(candidate)
                return idx
            except Exception:
                continue

        # Last resort: try the energy series index
        try:
            return pd.DatetimeIndex(unit.outputs["energy"].index)
        except Exception:
            # Absolute fallback to avoid crashing (empty index)
            return pd.DatetimeIndex([])

    def _as_series(self, unit, price_like):
        """
        Coerce any forecast (Series/list/ndarray/scalar) to a float Series
        on the unit's DatetimeIndex.
        """
        idx = self._unit_dt_index(unit)
        if price_like is None:
            return pd.Series(0.0, index=idx)

        # If it's already a pandas object, reindex onto our idx
        if hasattr(price_like, "reindex"):
            try:
                ser = price_like.reindex(idx)
            except Exception:
                ser = pd.Series(price_like, index=idx)
        else:
            # list/ndarray/scalar → Series; if length mismatch, fall back to zeros
            try:
                ser = pd.Series(price_like, index=idx)
            except Exception:
                ser = pd.Series(0.0, index=idx)

        # Ensure floats, no NaNs
        ser = pd.to_numeric(ser, errors="coerce").fillna(0.0)
        return ser

    def _select_forecast(self, unit, market_config):
        """
        Prefer price_{market_id}; fall back to price_EOM. Always return a Series on unit index.
        """
        raw = None
        try:
            raw = unit.forecaster[f"price_{market_config.market_id}"]
        except Exception:
            try:
                raw = unit.forecaster["price_EOM"]
            except Exception:
                raw = None
        return self._as_series(unit, raw)

    def _avg_price(self, series, t, window):
        """
        Mean of `series` over [t - window, t + window], clipped to index.
        """
        if series is None or series.empty:
            return 0.0
        left = max(t - window, series.index[0])
        right = min(t + window, series.index[-1])
        if left > right:
            return float(series.get(t, 0.0))
        return float(series.loc[left:right].mean())

    # ---------- main ----------

    def calculate_bids(self, unit, market_config, product_tuples, **kwargs):
        bids = []

        # Proper aligned index
        idx = self._unit_dt_index(unit)

        # Seeds (flexABLE-style)
        start0 = product_tuples[0][0]
        end_all = product_tuples[-1][1]
        previous_power = unit.get_output_before(start0)
        soc_theory = unit.get_soc_before(start0)

        # Envelopes over horizon
        min_ch, max_ch = unit.calculate_min_max_charge(start0, end_all)
        min_dis, max_dis = unit.calculate_min_max_discharge(start0, end_all, soc=soc_theory)

        # Forecast series on our DatetimeIndex
        price_series = self._select_forecast(unit, market_config)

        for (product, max_dis_i, min_dis_i, max_ch_i, min_ch_i) in zip(
            product_tuples, max_dis, min_dis, max_ch, min_ch
        ):
            t0 = product[0]
            t1 = product[1]
            only_hours = product[2] if len(product) > 2 else None

            # base profile (+discharge, −charge)
            base_p = float(unit.outputs["energy"].at[t0])

            # ramp-feasible headroom (discharge, ≥0)
            feas_up = unit.calculate_ramp_discharge(
                soc_theory,
                previous_power,
                power_discharge=max(0.0, float(max_dis_i)),
                current_power=max(base_p, 0.0),
                min_power_discharge=float(min_dis_i),
            )
            feas_up = max(0.0, float(feas_up))

            # ramp-feasible headroom (charge, ≤0)
            feas_down = unit.calculate_ramp_charge(
                soc_theory,
                previous_power,
                power_charge=float(max_ch_i),
                current_power=min(base_p, 0.0),
                min_power_charge=float(min_ch_i),
            )
            feas_down = min(0.0, float(feas_down))

            # Average price around t0
            avg_p = self._avg_price(price_series, t0, self.foresight)

            # Prices consistent with flexABLE storage logic
            ask_up = max(0.0, avg_p / max(unit.efficiency_discharge, 1e-9))  # discharge
            bid_down = max(0.0, avg_p * unit.efficiency_charge)               # charge

            # Create bids only if feasible non-zero
            if feas_up > 0.0:
                bids.append({
                    "start_time": t0,
                    "end_time": t1,
                    "only_hours": only_hours,
                    "price": float(ask_up),
                    "volume": float(feas_up),   # +MW (discharge)
                    "node": unit.node,
                })

            if feas_down < 0.0:
                bids.append({
                    "start_time": t0,
                    "end_time": t1,
                    "only_hours": only_hours,
                    "price": float(bid_down),
                    "volume": float(feas_down), # −MW (charge)
                    "node": unit.node,
                })

            # advance theoretical SoC by base profile only
            dt_h = (t1 - t0) / timedelta(hours=1)
            if base_p > 0.0:   # discharging
                d_soc = -(base_p * dt_h) / max(unit.efficiency_discharge, 1e-9)
            elif base_p < 0.0: # charging
                d_soc = -(base_p * dt_h) * unit.efficiency_charge
            else:
                d_soc = 0.0

            soc_theory += float(d_soc)
            previous_power = base_p

        return self.remove_empty_bids(bids)

def calculate_price_average(current_time, foresight, price_forecast):
    """
    Calculates the average price for a given foresight and returns the average price.

    Args:
        current_time (datetime.datetime): The current time.
        foresight (datetime.timedelta): The foresight.
        price_forecast (FastSeries): The price forecast.

    Returns:
        float: The average price.
    """
    start = max(current_time - foresight, price_forecast.index[0])
    end = min(current_time + foresight, price_forecast.index[-1])

    average_price = np.mean(price_forecast.loc[start:end])

    return average_price


def get_specific_revenue(unit, marginal_cost, t, foresight, price_forecast):
    """
    Calculates the specific revenue as difference between price forecast
    and marginal costs for the time defined by the foresight.

    Args:
        unit (SupportsMinMaxCharge): The unit that is dispatched.
        marginal_cost (float): The marginal cost.
        t (datetime.datetime): The start time of the product.
        foresight (datetime.timedelta): The foresight.
        price_forecast (FastSeries): The price forecast.

    Returns:
        float: The specific revenue.
    """

    possible_revenue = 0
    soc = unit.outputs["soc"][t]
    theoretic_SOC = soc

    if t + foresight > price_forecast.index[-1]:
        _, max_power_discharge_values = unit.calculate_min_max_discharge(
            start=t, end=price_forecast.index[-1] + unit.index.freq, soc=theoretic_SOC
        )
        price_forecast = price_forecast.loc[t:]
    else:
        _, max_power_discharge_values = unit.calculate_min_max_discharge(
            start=t, end=t + foresight + unit.index.freq, soc=theoretic_SOC
        )
        price_forecast = price_forecast.loc[t : t + foresight]

    previous_power = unit.get_output_before(t)

    for market_price, max_power_discharge in zip(
        price_forecast, max_power_discharge_values
    ):
        theoretic_power_discharge = unit.calculate_ramp_discharge(
            theoretic_SOC,
            previous_power=previous_power,
            power_discharge=max_power_discharge,
        )
        possible_revenue += (market_price - marginal_cost) * theoretic_power_discharge
        theoretic_SOC -= theoretic_power_discharge
        previous_power = theoretic_power_discharge

    if soc != theoretic_SOC:
        possible_revenue = possible_revenue / (soc - theoretic_SOC)
    else:
        possible_revenue = possible_revenue / unit.max_soc

    return possible_revenue
