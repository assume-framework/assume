# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import timedelta

import numpy as np
import pandas as pd

from assume.common.base import BaseStrategy, SupportsMinMaxCharge
from assume.common.market_objects import MarketConfig, Orderbook, Product


class flexableEOMStorage(BaseStrategy):
    """
    The strategy is analogue to the storage strategy in flexABLE.

    If the current price forecast is higher than the average price forecast for a given foresight,
    the unit will discharge.
    The price is then set as the average price divided by the discharge efficiency of the unit.
    Otherwise, the unit will charge with the price defined as the average price multiplied by the charge efficiency of the unit.

    Parameters:
        foresight (pd.Timedelta): Foresight for the average price calculation.

    Args:
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.foresight = pd.Timedelta(kwargs.get("eom_foresight", "12h"))

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
        theoretic_SOC = unit.outputs["soc"][start]

        # calculate min and max power for charging and discharging
        min_power_charge, max_power_charge = unit.calculate_min_max_charge(
            start, end_all
        )
        min_power_discharge, max_power_discharge = unit.calculate_min_max_discharge(
            start, end_all
        )

        # =============================================================================
        # Calculate bids
        # =============================================================================
        bids = []
        for product in product_tuples:
            start = product[0]
            end = product[1]

            current_power = unit.outputs["energy"].at[start]
            current_power_discharge = max(current_power, 0)
            current_power_charge = min(current_power, 0)

            # calculate ramping constraints
            max_power_discharge[start] = unit.calculate_ramp_discharge(
                theoretic_SOC,
                previous_power,
                max_power_discharge[start],
                current_power_discharge,
                min_power_discharge[start],
            )
            min_power_discharge[start] = unit.calculate_ramp_discharge(
                theoretic_SOC,
                previous_power,
                min_power_discharge[start],
                current_power_discharge,
                min_power_discharge[start],
            )
            max_power_charge[start] = unit.calculate_ramp_charge(
                theoretic_SOC,
                previous_power,
                max_power_charge[start],
                current_power_charge,
                min_power_charge[start],
            )
            min_power_charge[start] = unit.calculate_ramp_charge(
                theoretic_SOC,
                previous_power,
                min_power_charge[start],
                current_power_charge,
                min_power_charge[start],
            )

            price_forecast = unit.forecaster["price_EOM"]

            # calculate average price
            average_price = calculate_price_average(
                unit=unit,
                current_time=start,
                foresight=self.foresight,
                price_forecast=price_forecast,
            )

            # if price is higher than average price, discharge
            # if price is lower than average price, charge
            if price_forecast[start] >= average_price:
                price = average_price / unit.efficiency_discharge
                bid_quantity = max_power_discharge[start]
            else:
                price = average_price * unit.efficiency_charge
                bid_quantity = max_power_charge[start]

            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": None,
                    "price": price,
                    "volume": bid_quantity,
                }
            )

            # calculate theoretic SOC
            time_delta = (end - start) / timedelta(hours=1)
            if bid_quantity + current_power > 0:
                delta_soc = -(
                    (bid_quantity + current_power)
                    * time_delta
                    / unit.efficiency_discharge
                    / unit.max_volume
                )
            elif bid_quantity + current_power < 0:
                delta_soc = -(
                    (bid_quantity + current_power)
                    * time_delta
                    * unit.efficiency_charge
                    / unit.max_volume
                )
            else:
                delta_soc = 0

            theoretic_SOC += delta_soc
            previous_power = bid_quantity + current_power

        return bids

    def calculate_reward(
        self,
        unit,
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
            end = order["end_time"]
            end_excl = end - unit.index.freq
            index = pd.date_range(start, end_excl, freq=unit.index.freq)
            costs = pd.Series(float(unit.fixed_cost), index=index)
            for start in index:
                if unit.outputs[product_type][start] != 0:
                    costs[start] += unit.outputs[product_type][
                        start
                    ] * unit.calculate_marginal_cost(
                        start, unit.outputs[product_type][start]
                    )

            unit.outputs["profit"][index] = (
                unit.outputs[f"{product_type}_cashflow"][index] - costs
            )
            unit.outputs["total_costs"][index] = costs


class flexablePosCRMStorage(BaseStrategy):
    """
    The strategy is analogue to the storage strategy in flexABLE.

    The strategy bids the energy_price for the energy_pos product if the specific revenue is positive.
    Otherwise, the strategy bids the capacity_price for the capacity_pos product.

    Parameters:
        foresight (pd.Timedelta): Foresight for the average price calculation.

    Args:
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check if kwargs contains crm_foresight argument
        self.foresight = pd.Timedelta(kwargs.get("crm_foresight", "4h"))

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

        min_power_discharge, max_power_discharge = unit.calculate_min_max_discharge(
            start, end
        )
        bids = []
        theoretic_SOC = unit.outputs["soc"][start]
        for product in product_tuples:
            start = product[0]
            current_power = unit.outputs["energy"].at[start]

            # calculate ramping constraints for discharge
            bid_quantity = unit.calculate_ramp_discharge(
                theoretic_SOC,
                previous_power,
                max_power_discharge[start],
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
                current_time=start,
                foresight=self.foresight,
                price_forecast=unit.forecaster["price_EOM"],
            )

            # if specific revenue is positive, bid specific_revenue
            if specific_revenue >= 0:
                capacity_price = specific_revenue
            else:
                capacity_price = (
                    abs(specific_revenue) * unit.min_power_discharge / bid_quantity
                )

            energy_price = capacity_price / (theoretic_SOC * unit.max_volume)

            if market_config.product_type == "capacity_pos":
                bids.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": None,
                        "price": capacity_price,
                        "volume": bid_quantity,
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
                    }
                )
                # calculate theoretic SOC
                time_delta = (end - start) / timedelta(hours=1)
                delta_soc = -(
                    (bid_quantity + current_power)
                    * time_delta
                    / unit.efficiency_discharge
                    / unit.max_volume
                )
                theoretic_SOC += delta_soc
                previous_power = bid_quantity + current_power
                time_delta = (end - start) / timedelta(hours=1)
                delta_soc = -(
                    (bid_quantity + current_power)
                    * time_delta
                    / unit.efficiency_discharge
                    / unit.max_volume
                )
                theoretic_SOC += delta_soc
                previous_power = bid_quantity + current_power
            else:
                previous_power = current_power
                raise ValueError(
                    f"Product {market_config.product_type} is not supported by this strategy."
                )

        return bids


class flexableNegCRMStorage(BaseStrategy):
    """
    A strategy that bids the energy_price or the capacity_price of the unit on the negative CRM(reserve market).

    Parameters:
        foresight (pd.Timedelta): Foresight for the average price calculation.

    Args:
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.foresight = pd.Timedelta(kwargs.get("crm_foresight", "4h"))

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

        theoretic_SOC = unit.outputs["soc"][start]

        min_power_charge, max_power_charge = unit.calculate_min_max_charge(start, end)

        bids = []
        for product in product_tuples:
            start = product[0]
            current_power = unit.outputs["energy"].at[start]
            bid_quantity = unit.calculate_ramp_charge(
                theoretic_SOC,
                previous_power,
                max_power_charge[start],
                current_power,
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
                    }
                )
                # calculate theoretic SOC
                time_delta = (end - start) / timedelta(hours=1)
                delta_soc = (
                    (bid_quantity + current_power)
                    * time_delta
                    * unit.efficiency_charge
                    / unit.max_volume
                )
                theoretic_SOC += delta_soc
                previous_power = bid_quantity + current_power
                time_delta = (end - start) / timedelta(hours=1)
                delta_soc = (
                    (bid_quantity + current_power)
                    * time_delta
                    * unit.efficiency_charge
                    / unit.max_volume
                )
                theoretic_SOC += delta_soc
                previous_power = bid_quantity + current_power
            else:
                previous_power = current_power
                raise ValueError(
                    f"Product {market_config.product_type} is not supported by this strategy."
                )

        return bids


def calculate_price_average(unit, current_time, foresight, price_forecast):
    """
    Calculates the average price for a given foresight and returns the average price.

    Args:
        unit (SupportsMinMaxCharge): The unit that is dispatched.
        current_time (pd.Timestamp): The current time.
        foresight (pd.Timedelta): The foresight.
        price_forecast (pd.Series): The price forecast.

    Returns:
        float: The average price.
    """
    average_price = np.mean(
        price_forecast[current_time - foresight : current_time + foresight]
    )

    return average_price


def get_specific_revenue(unit, marginal_cost, current_time, foresight, price_forecast):
    """
    Calculates the specific revenue as difference between price forecast
    and marginal costs for the time defined by the foresight.

    Args:
        unit (SupportsMinMaxCharge): The unit that is dispatched.
        marginal_cost (float): The marginal cost.
        current_time (pd.Timestamp): The current time.
        foresight (pd.Timedelta): The foresight.
        price_forecast (pd.Series): The price forecast.

    Returns:
        float: The specific revenue.
    """
    t = current_time

    if t + foresight > price_forecast.index[-1]:
        price_forecast = price_forecast.loc[t:]
        _, max_power_discharge = unit.calculate_min_max_discharge(
            start=t, end=price_forecast.index[-1] + unit.index.freq
        )
    else:
        price_forecast = price_forecast.loc[t : t + foresight]
        _, max_power_discharge = unit.calculate_min_max_discharge(
            start=t, end=t + foresight + unit.index.freq
        )

    possible_revenue = 0
    soc = unit.outputs["soc"][t]
    theoretic_SOC = soc

    previous_power = unit.get_output_before(t)
    for i, market_price in enumerate(price_forecast):
        theoretic_power_discharge = unit.calculate_ramp_discharge(
            theoretic_SOC,
            previous_power=previous_power,
            power_discharge=max_power_discharge.iloc[i],
        )
        possible_revenue += (market_price - marginal_cost) * theoretic_power_discharge
        theoretic_SOC -= theoretic_power_discharge / unit.max_volume
        previous_power = theoretic_power_discharge

    if soc != theoretic_SOC:
        possible_revenue = possible_revenue / (soc - theoretic_SOC) / unit.max_volume
    else:
        possible_revenue = possible_revenue / unit.max_volume

    return possible_revenue
