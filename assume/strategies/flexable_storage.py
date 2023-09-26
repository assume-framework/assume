from datetime import timedelta

import numpy as np
import pandas as pd

from assume.common.base import BaseStrategy, SupportsMinMaxCharge
from assume.common.market_objects import MarketConfig, Orderbook, Product


class flexableEOMStorage(BaseStrategy):
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
        defines how it is dispatched to the market

        :param unit: Unit that is dispatched
        :type unit: SupportsMinMaxCharge
        :param market_config: Market configuration
        :type market_config: MarketConfig
        :param product_tuples: List of product tuples
        :type product_tuples: list[Product]
        :param kwargs: Additional keyword arguments
        :type kwargs: dict
        :return: bids containing start_time, end_time, only_hours, price, volume
        :rtype: Orderbook

        Strategy analogue to flexABLE

        """
        # =============================================================================
        # Storage Unit is either charging, discharging, or off
        # =============================================================================
        start = product_tuples[0][0]
        end_all = product_tuples[-1][1]

        previous_power = unit.get_output_before(start)

        theoretic_SOC = unit.outputs["soc"][start]

        min_power_charge, max_power_charge = unit.calculate_min_max_charge(
            start, end_all
        )
        min_power_discharge, max_power_discharge = unit.calculate_min_max_discharge(
            start, end_all
        )

        bids = []
        for product in product_tuples:
            start = product[0]
            end = product[1]
            end_excl = end - unit.index.freq

            current_power = unit.outputs["energy"].at[start]
            current_power_discharge = max(current_power, 0)
            current_power_charge = min(current_power, 0)

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

            average_price = calculate_price_average(
                unit=unit,
                current_time=start,
                foresight=self.foresight,
                price_forecast=price_forecast,
            )

            if price_forecast[start] >= average_price / unit.efficiency_discharge:
                bid_quantity = max_power_discharge[start]
            elif price_forecast[start] <= average_price * unit.efficiency_charge:
                bid_quantity = max_power_charge[start]
            else:
                previous_power = current_power
                continue

            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": None,
                    "price": average_price,
                    "volume": bid_quantity,
                }
            )

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
        Calculate reward (costs)
        :param unit: Unit to calculate reward for
        :type unit: SupportsMinMax
        :param marketconfig: Market configuration
        :type marketconfig: MarketConfig
        :param orderbook: Orderbook
        :type orderbook: Orderbook
        """
        # TODO: Calculate profits over all markets
        product_type = marketconfig.product_type

        for order in orderbook:
            start = order["start_time"]
            end = order["end_time"]
            end_excl = end - unit.index.freq
            index = pd.date_range(start, end_excl, freq=unit.index.freq)
            costs = pd.Series(unit.fixed_cost, index=index)
            for start in index:
                if unit.outputs[product_type][start] != 0:
                    costs[start] += unit.outputs[product_type][
                        start
                    ] * unit.calculate_marginal_cost(
                        start, unit.outputs[product_type][start]
                    )

            unit.outputs["total_costs"][index] = costs


class flexablePosCRMStorage(BaseStrategy):
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
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market
        Returns bids containing start_time, end_time, only_hours, price, volume

        :param unit: Unit that is dispatched
        :type unit: SupportsMinMaxCharge
        :param market_config: Market configuration
        :type market_config: MarketConfig
        :param product_tuples: List of product tuples
        :type product_tuples: list[Product]
        :param kwargs: Additional keyword arguments
        :type kwargs: dict
        :return: bids containing start_time, end_time, only_hours, price, volume
        :rtype: Orderbook
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
        defines how it is dispatched to the market

        :param unit: Unit that is dispatched
        :type unit: SupportsMinMaxCharge
        :param market_config: Market configuration
        :type market_config: MarketConfig
        :param product_tuples: List of product tuples
        :type product_tuples: list[Product]
        :param kwargs: Additional keyword arguments
        :type kwargs: dict
        :return: bids containing start_time, end_time, only_hours, price, volume
        :rtype: Orderbook
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
    Calculates the average price for a given time period
    Returns the average price

    :param unit: Unit that is dispatched
    :type unit: SupportsMinMaxCharge
    :param current_time: Current time
    :type current_time: pd.Timestamp
    :param foresight: Foresight
    :type foresight: pd.Timedelta
    :param price_forecast: Price forecast
    :type price_forecast: pd.Series
    :return: Average price
    :rtype: float
    """
    average_price = np.mean(
        price_forecast[current_time - foresight : current_time + foresight]
    )

    return average_price


def get_specific_revenue(unit, marginal_cost, current_time, foresight, price_forecast):
    """
    Calculates the specific revenue for a given time period
    Returns the specific revenue

    :param unit: Unit that is dispatched
    :type unit: SupportsMinMaxCharge
    :param marginal_cost: Marginal cost
    :type marginal_cost: float
    :param current_time: Current time
    :type current_time: pd.Timestamp
    :param foresight: Foresight
    :type foresight: pd.Timedelta
    :param price_forecast: Price forecast
    :type price_forecast: pd.Series
    :return: Specific revenue
    :rtype: float
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
