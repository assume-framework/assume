import numpy as np
import pandas as pd

from assume.strategies.base_strategy import BaseStrategy
from assume.units.storage import Storage


class flexableEOMStorage(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.foresight = pd.Timedelta(kwargs.get("eom_foresight", "12h"))

    def calculate_bids(
        self,
        unit: Storage = None,
        market_config=None,
        operational_window: dict = None,
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

        start = operational_window["window"]["start"]
        end = operational_window["window"]["end"]
        time_delta = pd.date_range(
            start=start,
            end=end - unit.index.freq,
            freq=unit.index.freq,
        )

        average_price = self.calculate_price_average(unit, time_delta)

        bid_quantity = 0

        if (
            unit.price_forecast[start] >= average_price / unit.efficiency_discharge
        ) and (operational_window["max_power_discharge"]["power_discharge"] > 0):
            # place bid to discharge
            bid_quantity = operational_window["max_power_discharge"]["power_discharge"]

        elif (
            unit.price_forecast[start] <= average_price * unit.efficiency_charge
        ) and (operational_window["max_power_charge"]["power_charge"] < 0):
            # place bid to charge
            bid_quantity = operational_window["max_power_charge"]["power_charge"]

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
        unit: Storage = None,
        market_config=None,
        operational_window: dict = None,
    ):
        pass
