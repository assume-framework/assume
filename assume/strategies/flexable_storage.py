import numpy as np
import pandas as pd

from assume.common.base import BaseStrategy, SupportsMinMaxCharge
from assume.common.market_objects import MarketConfig, Product


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
    ):
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
        end = product_tuples[0][1]
        end_excl = end - unit.index.freq

        previous_power = unit.get_output_before(start)
        current_power = unit.outputs["energy"].at[start]
        current_power_discharge = max(current_power, 0)
        current_power_charge = min(current_power, 0)

        min_power_charge, max_power_charge = unit.calculate_min_max_charge(start, end)
        min_power_discharge, max_power_discharge = unit.calculate_min_max_discharge(
            start, end
        )

        for t in unit.index[start:end_excl]:
            max_power_discharge[t] = unit.calculate_ramp_up_discharge(
                previous_power, max_power_discharge[t], current_power_discharge
            )
            min_power_discharge[t] = unit.calculate_ramp_down_discharge(
                previous_power, min_power_discharge[t], current_power_discharge
            )
            max_power_charge[t] = unit.calculate_ramp_up_charge(
                previous_power, max_power_charge[t], current_power_charge
            )
            min_power_charge[t] = unit.calculate_ramp_down_charge(
                previous_power, min_power_charge[t], current_power_charge
            )
            previous_power = current_power
            current_power = max_power_charge[t]

        price_forecast = data_dict["price_forecast"]

        average_price = calculate_price_average(
            unit=unit,
            current_time=start,
            foresight=self.foresight,
            price_forecast=price_forecast,
        )
        bid_quantity = 0
        bid_quantity = max_power_discharge.where(
            price_forecast[start:end_excl] >= average_price / unit.efficiency_discharge,
            bid_quantity,
        )

        bid_quantity = max_power_charge.where(
            price_forecast[start:end_excl] <= average_price * unit.efficiency_charge,
            bid_quantity,
        )

        return [{"price": average_price, "volume": bid_quantity.max()}]


class flexablePosCRMStorage(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check if kwargs contains crm_foresight argument
        self.foresight = pd.Timedelta(kwargs.get("crm_foresight", "4h"))

        self.current_time = None

    def calculate_bids(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        data_dict: dict,
        product_tuples: list[Product],
        **kwargs,
    ):
        start = product_tuples[0][0]
        end = product_tuples[0][1]
        end_excl = end - unit.index.freq

        previous_power = unit.get_output_before(start)
        current_power = unit.outputs["energy"].at[start]

        min_power_discharge, max_power_discharge = unit.calculate_min_max_discharge(
            start, end
        )

        for t in unit.index[start:end_excl]:
            max_power_discharge[t] = unit.calculate_ramp_up_discharge(
                previous_power,
                max_power_discharge[t],
                current_power,
            )
            previous_power = current_power
            current_power = max_power_discharge[t]

        bid_quantity = max(max_power_discharge)
        if bid_quantity == 0:
            return []

        marginal_cost = unit.calc_marginal_cost(timestep=start, discharge=True)

        specific_revenue = get_specific_revenue(
            unit=unit,
            marginal_cost=marginal_cost,
            current_time=self.current_time,
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
            bids = [
                {"price": capacity_price, "volume": bid_quantity},
            ]
        elif market_config.product_type == "energy_pos":
            bids = [
                {"price": energy_price, "volume": bid_quantity},
            ]
        else:
            raise ValueError(
                f"Product {market_config.product_type} is not supported by this strategy."
            )

        return bids


class flexableNegCRMStorage(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.foresight = pd.Timedelta(kwargs.get("crm_foresight", "4h"))
        self.current_time = None

    def calculate_bids(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        product_tuples: list[Product],
        data_dict: dict,
        **kwargs,
    ):
        start = product_tuples[0][0]
        end = product_tuples[0][1]
        end_excl = end - unit.index.freq

        previous_power = unit.get_output_before(start)
        current_power = unit.outputs["energy"].at[start]

        min_power_charge, max_power_charge = unit.calculate_min_max_charge(start, end)

        for t in unit.index[start:end_excl]:
            max_power_charge[t] = unit.calculate_ramp_up_charge(
                previous_power,
                max_power_charge[t],
                current_power,
            )
            previous_power = current_power
            current_power = max_power_charge[t]

        # in flexable no prices calculated for CRM_neg
        bid_quantity = max(max_power_charge)
        if bid_quantity == 0:
            return []
        # if bid_quantity >= min_bid_volume

        if market_config.product_type == "capacity_neg":
            bids = [
                {"price": 0, "volume": bid_quantity},
            ]
        elif market_config.product_type == "energy_neg":
            bids = [
                {"price": 0, "volume": bid_quantity},
            ]
        else:
            raise ValueError(
                f"Product {market_config.product_type} is not supported by this strategy."
            )

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
