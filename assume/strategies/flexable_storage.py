import numpy as np
import pandas as pd

from assume.common.market_objects import MarketConfig
from assume.strategies.base_strategy import BaseStrategy, OperationalWindow
from assume.units.base_unit import BaseUnit


class flexableEOMStorage(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.foresight = pd.Timedelta(kwargs.get("eom_foresight", "12h"))

    def calculate_bids(
        self,
        unit: BaseUnit,
        operational_window: OperationalWindow,
        market_config: MarketConfig,
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

        start = operational_window["window"][0]
        end = operational_window["window"][1]
        time_delta = pd.date_range(
            start=start,
            end=end - unit.index.freq,
            freq=unit.index.freq,
        )

        average_price = self.calculate_price_average(unit, time_delta)

        bid_quantity = 0

        if (
            unit.price_forecast[start] >= average_price / unit.efficiency_discharge
        ) and (operational_window["states"]["max_power_discharge"]["volume"] > 0):
            # place bid to discharge
            bid_quantity = operational_window["states"]["max_power_discharge"]["volume"]

        elif (
            unit.price_forecast[start] <= average_price * unit.efficiency_charge
        ) and (operational_window["states"]["max_power_charge"]["volume"] < 0):
            # place bid to charge
            bid_quantity = operational_window["states"]["max_power_charge"]["volume"]

        if bid_quantity != 0:
            return [{"price": average_price, "volume": bid_quantity}]
        else:
            return []

    def calculate_price_average(self, unit, time_delta):
        average_price = np.mean(
            unit.price_forecast[
                time_delta[0] - self.foresight : time_delta[-1] + self.foresight
            ]
        )

        return average_price


class flexableCRMStorage(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.foresight = pd.Timedelta("12h")

    def calculate_bids(
        self,
        unit: BaseUnit,
        operational_window: OperationalWindow,
        market_config: MarketConfig,
        **kwargs,
    ):
        pass
