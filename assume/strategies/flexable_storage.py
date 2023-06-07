import pandas as pd
import numpy as np

from assume.strategies.base_strategy import BaseStrategy
from assume.units.storage_unit import StorageUnit


class flexableEOMStorage(BaseStrategy):
    def __init__(self):
        super().__init__()

        self.foresight = pd.Timedelta("12h")
        self.current_time = None

    def calculate_bids(
        self,
        unit: StorageUnit = None,
        operational_window: dict = None,
    ):
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market

        Return: volume, price
        Strategy analogue to flexABLE

        """
        if operational_window is not None:
            # =============================================================================
            # Storage Unit is either charging, discharging, or off
            # =============================================================================
            self.current_time = operational_window["window"]["start"]

            average_price = self.calculate_price_average(unit)

            if (
                unit.price_forecast[self.current_time]
                >= average_price / unit.efficiency_discharge
            ):
                # place bid to discharge
                bid_quantity = min(
                    max(
                        (
                            (unit.current_SOC - unit.min_SOC)
                            - unit.pos_capacity_reserve[self.current_time]
                        )
                        * unit.efficiency_discharge,
                        0,
                    ),
                    unit.max_power_discharge,
                )

                bids = [{"price": average_price, "volume": bid_quantity}]

            elif (
                unit.price_forecast[self.current_time]
                <= average_price * unit.efficiency_charge
            ):
                # place bid to charge
                bid_quantity = min(
                    max(
                        (
                            (unit.max_SOC - unit.current_SOC)
                            - unit.neg_capacity_reserve[self.current_time]
                        )
                        / unit.efficiency_charge,
                        0,
                    ),
                    unit.max_power_charge,
                )

                bids = [{"price": average_price, "volume": -bid_quantity}]

            else:
                bids = []

        return bids

    def calculate_price_average(self, unit):
        t = self.current_time
        """if t - self.foresight < pd.Timedelta("0h"):
            average_price = np.mean(unit.price_forecast[t-self.foresight:] 
                                    + unit.price_forecast[:t+self.foresight])
        else:"""
        average_price = np.mean(
            unit.price_forecast[t - self.foresight : t + self.foresight]
        )

        return average_price


class flexableCRMStorage(BaseStrategy):
    def __init__(self):
        super().__init__()

        self.foresight = pd.Timedelta("12h")
        self.current_time = None

    def calculate_bids(
        self,
        unit: StorageUnit = None,
        operational_window: dict = None,
    ):
        pass
