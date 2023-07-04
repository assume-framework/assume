import numpy as np
import pandas as pd

from assume.strategies.base_strategy import BaseStrategy
from assume.units.storage import StorageUnit


class flexableEOMStorage(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.foresight = pd.Timedelta("12h")

    def calculate_bids(
        self,
        unit: StorageUnit = None,
        market_config=None,
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
            start = operational_window["window"]["start"]
            end = operational_window["window"]["end"]
            time_delta = pd.date_range(
                start=start,
                end=end - unit.index.freq,
                freq=unit.index.freq,
            )

            average_price = self.calculate_price_average(unit, time_delta)

            if (
                np.mean(unit.price_forecast[time_delta])
                >= average_price / unit.efficiency_discharge
            ):
                # place bid to discharge
                bid_quantity = min(
                    max(
                        (
                            (unit.current_SOC - unit.min_SOC)
                            - unit.outputs["pos_capacity"][self.current_time]
                        )
                        * unit.efficiency_discharge,
                        0,
                    ),
                    operational_window["max_power_discharge"]["power_discharge"],
                )

                if bid_quantity < 1e-5 and bid_quantity > -1e-5:
                    bids = []
                else:
                    bids = [{"price": average_price, "volume": bid_quantity}]

            elif (
                np.mean(unit.price_forecast[time_delta])
                <= average_price * unit.efficiency_charge
            ):
                # place bid to charge
                bid_quantity = max(
                    -max(
                        (
                            (
                                (unit.max_SOC - unit.current_SOC)
                                - np.sum(unit.neg_capacity_reserve[time_delta])
                            )
                            / unit.efficiency_charge
                        ),
                        0,
                    ),
                    operational_window["max_power_charge"]["power_charge"],
                )

                if bid_quantity < 1e-5 and bid_quantity > -1e-5:
                    bids = []
                else:
                    bids = [{"price": average_price, "volume": bid_quantity}]
            else:
                bids = []

        return bids

    def calculate_price_average(self, unit, time_delta):
        """if t - self.foresight < pd.Timedelta("0h"):
            average_price = np.mean(unit.price_forecast[t-self.foresight:]
                                    + unit.price_forecast[:t+self.foresight])
        else:"""

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
        unit: StorageUnit = None,
        market_config=None,
        operational_window: dict = None,
    ):
        pass
