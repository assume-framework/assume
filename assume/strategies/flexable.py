import pandas as pd

from assume.strategies.base_strategy import BaseStrategy
from assume.units.base_unit import BaseUnit


class flexableEOM(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.foresight = pd.Timedelta("12h")

    def calculate_bids(
        self,
        unit: BaseUnit = None,
        market_config=None,
        operational_window: dict = None,
    ):
        bid_quantity_inflex, bid_price_inflex = 0, 0
        bid_quantity_flex, bid_price_flex = 0, 0

        if operational_window is not None:
            #TODO: check where start has to be replaced by an aggregation over time_delta
            start = operational_window["window"]["start"]
            end = operational_window["window"]["end"]
            time_delta = pd.date_range(start=start, 
                                      end=end-unit.index.freq, 
                                      freq=unit.index.freq,
                                      )
            # =============================================================================
            # Powerplant is either on, or is able to turn on
            # Calculating possible bid amount
            # =============================================================================
            bid_quantity_inflex = operational_window["min_power"]["power"]

            marginal_cost_mr = operational_window["min_power"]["marginal_cost"]
            marginal_cost_flex = operational_window["max_power"]["marginal_cost"]
            # =============================================================================
            # Calculating possible price
            # =============================================================================
            if unit.current_status:
                bid_price_inflex = self.calculate_EOM_price_if_on(
                    unit, marginal_cost_mr, bid_quantity_inflex, time_delta
                )
            else:
                bid_price_inflex = self.calculate_EOM_price_if_off(
                    unit, marginal_cost_flex, bid_quantity_inflex, time_delta
                )

            if unit.total_heat_output[start] > 0:
                power_loss_ratio = (
                    unit.power_loss_chp[start]
                    / unit.total_heat_output[start]
                )
            else:
                power_loss_ratio = 0.0

            # Flex-bid price formulation
            if unit.current_status:
                bid_quantity_flex = (
                    operational_window["max_power"]["power"] - bid_quantity_inflex
                )
                bid_price_flex = (1 - power_loss_ratio) * marginal_cost_flex

        bids = [
            {"price": bid_price_inflex, "volume": bid_quantity_inflex},
            {"price": bid_price_flex, "volume": bid_quantity_flex},
        ]

        return bids

    def calculate_EOM_price_if_off(self, unit, marginal_cost_mr, bid_quantity_inflex, time_delta):
        # The powerplant is currently off and calculates a startup markup as an extra
        # to the marginal cost
        # Calculating the average uninterrupted operating period
        av_operating_time = max(
            unit.mean_market_success, unit.min_operating_time, 1
        )  # 1 prevents division by 0

        starting_cost = self.get_starting_costs(time=unit.current_down_time, unit=unit)
        if bid_quantity_inflex == 0:
            markup = starting_cost / av_operating_time
        else:
            markup = starting_cost / av_operating_time / bid_quantity_inflex

        bid_price_inflex = min(marginal_cost_mr + markup, 3000.0)

        return bid_price_inflex

    def calculate_EOM_price_if_on(self, unit, marginal_cost_flex, bid_quantity_inflex, time_delta):
        """
        Check the description provided by Thomas in last version, the average downtime is not available
        """
        if bid_quantity_inflex == 0:
            return 0

        t = time_delta[0]
        min_down_time = max(unit.min_down_time, 1)

        starting_cost = self.get_starting_costs(time=min_down_time, unit=unit)

        price_reduction_restart = starting_cost / min_down_time / bid_quantity_inflex

        if unit.total_heat_output[t] > 0:
            heat_gen_cost = (
                unit.total_heat_output[t] * (unit.fuel_price["natural gas"][t] / 0.9)
            ) / bid_quantity_inflex
        else:
            heat_gen_cost = 0.0

        possible_revenue = self.get_possible_revenues(
            marginal_cost=marginal_cost_flex,
            unit=unit,
            time_delta=time_delta,
        )
        if possible_revenue >= 0 and unit.price_forecast[t] < marginal_cost_flex:
            marginal_cost_flex = 0

        bid_price_inflex = max(
            -price_reduction_restart - heat_gen_cost + marginal_cost_flex,
            -499.00,
        )

        return bid_price_inflex

    def get_starting_costs(self, time, unit):
        if time < unit.downtime_hot_start:
            return unit.hot_start_cost

        elif time < unit.downtime_warm_start:
            return unit.warm_start_cost

        else:
            return unit.cold_start_cost

    def get_possible_revenues(self, marginal_cost, unit, time_delta):
        t = time_delta[0]
        price_forecast = []

        if t + self.foresight > unit.price_forecast.index[-1]:
            price_forecast = unit.price_forecast.loc[t:]
        else:
            price_forecast = unit.price_forecast.loc[t : t + self.foresight]

        possible_revenue = sum(
            marketPrice - marginal_cost for marketPrice in price_forecast
        )

        return possible_revenue


class flexableCRM(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.foresight = pd.Timedelta("12h")

    def calculate_bids(
        self,
        unit: BaseUnit = None,
        market_config=None,
        operational_window: dict = None,
    ):
        pass
