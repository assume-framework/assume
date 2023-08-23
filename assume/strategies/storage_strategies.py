from datetime import datetime

import numpy as np
import pandas as pd

from assume.common.base import BaseStrategy, SupportsMinMaxCharge
from assume.common.market_objects import MarketConfig, Orderbook, Product


class complexEOMStorage(BaseStrategy):
    def __init__(self):
        super().__init__()

        self.foresight = pd.Timedelta("12h")

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

        Return: volume, price
        Strategy analogue to flexABLE
        """
        product = product_tuples[0]
        start = product[0]
        end = product[1]

        min_power_charge, max_power_charge = unit.calculate_min_max_charge(start, end)
        min_power_discharge, max_power_discharge = unit.calculate_min_max_discharge(
            start, end
        )

        # =============================================================================
        # Storage Unit is either charging, discharging, or off
        # =============================================================================
        bid_quantity_mr_discharge = min_power_discharge
        bid_quantity_flex_discharge = max_power_discharge - min_power_discharge
        bid_quantity_mr_charge = min_power_charge
        bid_quantity_flex_charge = max_power_charge - min_power_charge

        cost_mr_discharge = unit.calculate_marginal_cost(start, min_power_discharge)
        cost_flex_discharge = unit.calculate_marginal_cost(start, max_power_discharge)
        cost_mr_charge = unit.calculate_marginal_cost(start, min_power_charge)
        cost_flex_charge = unit.calculate_marginal_cost(start, max_power_charge)

        average_price = self.calculate_price_average(unit)

        previous_power = unit.get_output_before(start)

        price_forecast = unit.forecaster["price_EOM"][t : t + self.foresight]

        if price_forecast[start] >= average_price / unit.efficiency_discharge:
            # place bid to discharge
            if previous_power > 0:
                # was discharging before
                bid_price_mr = self.calculate_EOM_price_continue_discharging(
                    start, unit, cost_mr_discharge, bid_quantity_mr_discharge
                )
                bid_quantity_mr = bid_quantity_mr_discharge
                bid_price_flex = cost_flex_discharge
                bid_quantity_flex = bid_quantity_flex_discharge

            elif previous_power < 0:
                # was charging before
                if unit.min_down_time > 0:
                    bid_quantity_mr = 0
                    bid_price_mr = 0

                else:
                    bid_price_mr = self.calculate_EOM_price_if_off(
                        unit,
                        cost_flex_discharge,
                        bid_quantity_mr_discharge,
                    )
                    bid_quantity_mr = bid_quantity_mr_discharge
                    bid_price_flex = cost_flex_discharge
                    bid_quantity_flex = bid_quantity_flex_discharge
            else:
                bid_price_mr = 0
                bid_quantity_mr = 0
                bid_price_flex = 0
                bid_quantity_flex = 0

        elif price_forecast[start] <= average_price * unit.efficiency_charge:
            # place bid to charge
            if previous_power > 0:
                # was discharging before
                if unit.min_down_time > 0:
                    bid_quantity_mr = 0
                    bid_price_mr = 0
                else:
                    bid_price_mr = self.calculate_EOM_price_if_off(
                        unit, cost_mr_charge, bid_quantity_mr_charge
                    )
                    bid_quantity_mr = bid_quantity_mr_charge
                    bid_price_flex = cost_flex_charge
                    bid_quantity_flex = bid_quantity_flex_charge

            elif previous_power < 0:
                # was charging before
                bid_price_mr = bid_quantity_mr_charge
                bid_quantity_mr = cost_mr_charge
                bid_price_flex = cost_flex_charge
                bid_quantity_flex = bid_quantity_flex_charge
            else:
                bid_price_mr = 0
                bid_quantity_mr = 0
                bid_price_flex = 0
                bid_quantity_flex = 0

        else:
            bid_price_mr = 0
            bid_quantity_mr = 0
            bid_price_flex = 0
            bid_quantity_flex = 0

        bids = [
            {
                "start_time": product[0],
                "end_time": product[1],
                "only_hours": product[2],
                "price": bid_price_mr,
                "volume": bid_quantity_mr,
            },
            {
                "start_time": product[0],
                "end_time": product[1],
                "only_hours": product[2],
                "price": bid_price_flex,
                "volume": bid_quantity_flex,
            },
        ]
        return bids

    def calculate_price_average(self, unit: SupportsMinMaxCharge, t: datetime):
        average_price = np.mean(
            unit.forecaster["price_EOM"][t - self.foresight : t + self.foresight]
        )

        return average_price

    def calculate_EOM_price_if_off(self, unit, marginal_cost_mr, bid_quantity_mr):
        av_operating_time = max(
            unit.mean_market_success, unit.min_operating_time, 1
        )  # 1 prevents division by 0

        op_time = unit.get_operation_time(start)
        starting_cost = unit.get_starting_costs(op_time)
        markup = starting_cost / av_operating_time / bid_quantity_mr

        bid_price_mr = min(marginal_cost_mr + markup, 3000.0)

        return bid_price_mr

    def calculate_EOM_price_continue_discharging(
        self, start, unit, marginal_cost_flex, bid_quantity_mr
    ):
        if bid_quantity_mr == 0:
            return 0

        t = start
        op_time = unit.get_operation_time(start)
        starting_cost = unit.get_starting_costs(op_time)

        price_reduction_restart = starting_cost / min_down_time / bid_quantity_mr

        possible_revenue = self.get_possible_revenues(
            marginal_cost=marginal_cost_flex,
            unit=unit,
            t=start,
        )
        if (
            possible_revenue >= 0
            and unit.forecaster["price_EOM"][t] < marginal_cost_flex
        ):
            marginal_cost_flex = 0

        bid_price_mr = max(
            -price_reduction_restart + marginal_cost_flex,
            -2999.00,
        )

        return bid_price_mr

    def get_starting_costs(self, time, unit):
        if time < unit.downtime_hot_start:
            return unit.hot_start_cost

        elif time < unit.downtime_warm_start:
            return unit.warm_start_cost

        else:
            return unit.cold_start_cost

    def get_possible_revenues(self, marginal_cost, unit, t):
        price_forecast = unit.forecaster["price_EOM"][t : t + self.foresight]

        possible_revenue = sum(
            marketPrice - marginal_cost for marketPrice in price_forecast
        )

        return possible_revenue
