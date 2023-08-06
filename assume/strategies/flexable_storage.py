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
        data_dict: dict,
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market

        Return: volume, price
        Strategy analogue to flexABLE

        """
        # =============================================================================
        # Storage Unit is either charging, discharging, or off
        # =============================================================================
        start = product_tuples[0][0]
        end_all = product_tuples[-1][1]

        previous_power = unit.get_output_before(start)
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
                previous_power, max_power_discharge[start], current_power_discharge
            )
            min_power_discharge[start] = unit.calculate_ramp_discharge(
                previous_power, min_power_discharge[start], current_power_discharge
            )
            max_power_charge[start] = unit.calculate_ramp_charge(
                previous_power, max_power_charge[start], current_power_charge
            )
            min_power_charge[start] = unit.calculate_ramp_charge(
                previous_power, min_power_charge[start], current_power_charge
            )

            price_forecast = data_dict["price_forecast"]

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
                bid_quantity = 0

            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": None,
                    "price": average_price,
                    "volume": bid_quantity,
                }
            )
            previous_power = bid_quantity + current_power

        return bids


class flexablePosCRMStorage(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check if kwargs contains crm_foresight argument
        self.foresight = pd.Timedelta(kwargs.get("crm_foresight", "4h"))

    def calculate_bids(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        data_dict: dict,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        start = product_tuples[0][0]
        end = product_tuples[-1][1]

        previous_power = unit.get_output_before(start)

        min_power_discharge, max_power_discharge = unit.calculate_min_max_discharge(
            start, end
        )
        bids = []
        for product in product_tuples:
            start = product[0]
            current_power = unit.outputs["energy"].at[start]

            bid_quantity = unit.calculate_ramp_discharge(
                previous_power,
                max_power_discharge[start],
                current_power,
            )

            if bid_quantity == 0:
                return []

            marginal_cost = unit.calc_marginal_cost(timestep=start, discharge=True)

            specific_revenue = get_specific_revenue(
                unit=unit,
                marginal_cost=marginal_cost,
                current_time=start,
                foresight=self.foresight,
                price_forecast=data_dict["price_forecast"],
            )

            if specific_revenue >= 0:
                capacity_price = specific_revenue
            else:
                capacity_price = (
                    abs(specific_revenue) * unit.min_power_discharge / bid_quantity
                )

            energy_price = capacity_price / unit.current_SOC

            if market_config.product_type == "capacity_pos":
                bids.appen(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": None,
                        "price": capacity_price,
                        "volume": bid_quantity,
                    }
                )
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
            else:
                raise ValueError(
                    f"Product {market_config.product_type} is not supported by this strategy."
                )
            previous_power = bid_quantity + current_power

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
        data_dict: dict,
        **kwargs,
    ) -> Orderbook:
        start = product_tuples[0][0]
        end = product_tuples[-1][1]

        previous_power = unit.get_output_before(start)

        min_power_charge, max_power_charge = unit.calculate_min_max_charge(start, end)

        bids = []
        for product in product_tuples:
            start = product[0]
            current_power = unit.outputs["energy"].at[start]
            bid_quantity = unit.calculate_ramp_charge(
                previous_power,
                max_power_charge[start],
                current_power,
            )

            # if bid_quantity >= min_bid_volume  --> not checked here

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
            elif market_config.product_type == "energy_neg":
                bids.appen(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": None,
                        "price": 0,
                        "volume": bid_quantity,
                    }
                )
            else:
                raise ValueError(
                    f"Product {market_config.product_type} is not supported by this strategy."
                )
            previous_power = current_power + bid_quantity

        return bids


def calculate_price_average(unit, current_time, foresight, price_forecast):
    average_price = np.mean(
        price_forecast[current_time - foresight : current_time + foresight]
    )

    return average_price


def get_specific_revenue(unit, marginal_cost, current_time, foresight, price_forecast):
    t = current_time

    if t + foresight > price_forecast.index[-1]:
        price_forecast = price_forecast.loc[t:]
    else:
        price_forecast = price_forecast.loc[t : t + foresight]

    possible_revenue = 0
    theoretic_SOC = unit.current_SOC
    for market_price in price_forecast:
        theoretic_power_discharge = min(
            max(theoretic_SOC - unit.min_SOC, 0), unit.max_power_discharge
        )
        possible_revenue += (market_price - marginal_cost) * theoretic_power_discharge
        theoretic_SOC -= theoretic_power_discharge

    if unit.current_SOC - theoretic_SOC != 0:
        possible_revenue = possible_revenue / (unit.current_SOC - theoretic_SOC)

    return possible_revenue
