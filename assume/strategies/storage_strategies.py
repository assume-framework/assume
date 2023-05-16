import pandas as pd
import numpy as np

from assume.strategies.base_strategy import BaseStrategy
from assume.units.storage_unit import StorageUnit

class NaiveStorageStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()

        self.foresight = pd.Timedelta("12h")

    def calculate_bids(
        self,
        unit: StorageUnit = None,
        operational_window: dict = None,
        current_time: pd.Timestamp = None
    ):
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market

        Return: volume, price
        Strategy analogue to flexABLE

        """
        bid_quantity_mr_charge, bid_price_mr_charge = 0, 0
        bid_quantity_mr_discharge, bid_price_mr_discharge = 0, 0
        bid_quantity_flex_charge, bid_price_flex_charge = 0, 0
        bid_quantity_flex_discharge, bid_price_flex_discharge = 0, 0

        if operational_window is not None:
            # =============================================================================
            # Storage Unit is either charging, discharging, or off
            # =============================================================================
            bid_quantity_mr_discharge = operational_window["min_power_discharge"]["power_discharge"]
            bid_quantity_flex_discharge = (
                operational_window["max_power_discharge"]["power_discharge"] - bid_quantity_mr_discharge
            )
            bid_quantity_mr_charge = operational_window["min_power_charge"]["power_charge"]
            bid_quantity_flex_charge = (
                operational_window["max_power_charge"]["power_charge"] - bid_quantity_mr_charge
                )
            
            marginal_cost_mr_discharge = operational_window["min_power_discharge"]["marginal_cost"]
            marginal_cost_flex_discharge = operational_window["max_power_discharge"]["marginal_cost"]
            marginal_cost_mr_charge = operational_window["min_power_charge"]["marginal_cost"]
            marginal_cost_flex_charge = operational_window["max_power_charge"]["marginal_cost"]

            average_price = self.calculate_price_average(unit, t=current_time)

            if unit.price_forecast[current_time] >= average_price/unit.efficiency_discharge:
            #place bid to discharge

                if operational_window["current_power_discharge"]["power_discharge"] > 0:
                #was discharging before
                    bid_price_mr = self.calculate_EOM_price_continue_discharging(
                        unit, marginal_cost_flex_discharge, bid_quantity_mr_discharge
                    )
                    bid_quantity_mr = bid_quantity_mr_discharge
                    bid_price_flex = marginal_cost_flex_discharge
                    bid_quantity_flex = bid_quantity_flex_discharge

                elif operational_window["current_power_charge"]["power_charge"] < 0:
                #was charging before
                    if unit.min_down_time > 0:
                        bid_quantity_mr = 0
                        bid_price_mr = 0

                    else:
                        bid_price_mr = self.calculate_EOM_price_if_off(
                            unit, marginal_cost_flex_discharge, bid_quantity_mr_discharge
                        )
                        bid_quantity_mr = bid_quantity_mr_discharge
                        bid_price_flex = marginal_cost_flex_discharge
                        bid_quantity_flex = bid_quantity_flex_discharge
                
                
            elif unit.price_forecast[current_time] < average_price * unit.efficiency_charge: 
            #place bid to charge
                if operational_window["current_power_discharge"]["power_discharge"] > 0:
                #was discharging before
                    if unit.min_down_time > 0:
                        bid_quantity_mr = 0
                        bid_price_mr = 0
                    else:    
                        bid_price_mr = self.calculate_EOM_price_if_off(
                            unit, marginal_cost_mr_charge, bid_quantity_mr_charge
                        )
                        bid_quantity_mr = bid_quantity_mr_charge
                        bid_price_flex = marginal_cost_flex_charge
                        bid_quantity_flex = bid_quantity_flex_charge

                elif operational_window["current_power_charge"]["power_charge"] < 0:
                #was charging before
                    bid_price_mr = bid_quantity_mr_charge
                    bid_quantity_mr = marginal_cost_mr_charge
                    bid_price_flex = marginal_cost_flex_charge
                    bid_quantity_flex = bid_quantity_flex_charge
                
            bids = [
            {"price": bid_price_mr, "volume": bid_quantity_mr},
            {"price": bid_price_flex, "volume": bid_quantity_flex},
        ]
        
        return bids
    
    def calculate_price_agerage(self, unit):
        t = unit.current_time
        if t - self.foresight < 0:
            average_price = np.mean(unit.price_forecast[t-self.foresight:] 
                                    + unit.price_forecast[:t+self.foresight])
        else:
            average_price = np.mean(unit.price_forecast[t-self.foresight:t+self.foresight])
        
        return average_price


    def calculate_EOM_price_if_off(
            self, unit, marginal_cost_mr, bid_quantity_mr
        ):
        
        av_operating_time = max(
            unit.mean_market_success, unit.min_operating_time, 1
        ) # 1 prevents division by 0

        starting_cost = self.get_starting_costs(time=unit.current_down_time, unit=unit)
        markup = starting_cost / av_operating_time / bid_quantity_mr

        bid_price_mr = min(marginal_cost_mr + markup, 3000.0)

        return bid_price_mr
          
    def calculate_EOM_price_continue_discharging(
            self, unit, marginal_cost_flex, bid_quantity_mr
            ):
        if bid_quantity_mr == 0:
            return 0
        
        t = unit.current_time
        min_down_time = max(unit.min_down_time, 1)

        starting_cost = self.get_starting_costs(time=min_down_time, unit=unit)
        
        price_reduction_restart = starting_cost / min_down_time / bid_quantity_mr

        possible_revenue = self.get_possible_revenues(
            marginal_cost=marginal_cost_flex,
            unit=unit,
        )
        if possible_revenue >= 0 and unit.price_forecast[t] < marginal_cost_flex:
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
        
    def get_possible_revenues(self, marginal_cost, unit):
        t = unit.current_time
        price_forecast = []

        if t + self.foresight > unit.price_forecast.index[-1]:
            price_forecast = unit.price_forecast.loc[t:]
        else:
            price_forecast = unit.price_forecast.loc[t : t + self.foresight]

        possible_revenue = sum(
            marketPrice - marginal_cost for marketPrice in price_forecast
        )

        return possible_revenue