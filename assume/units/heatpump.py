from ..strategies import BaseStrategy
from .base_unit import BaseUnit
import numpy as np
import pandas as pd


class HeatPump(BaseUnit):
    def __init__(
        self,
        id: str,
        technology: str,
        bidding_strategies: dict,
        max_power: float or pd.Series,
        min_power: float or pd.Series,
        cop_curve: dict,
        electricity_price: float,
        ramp_up: float,
        ramp_down: float,
        fixed_cost: float = 0,
        min_operating_time: float = 0,
        min_down_time: float = 0,
        downtime_hot_start: int = 0.001,  # hours
        downtime_warm_start: int = 0.005,  # hours
        source_temp: float or pd.Series = 0.0,
        sink_temp: float or pd.Series = 0.0,
        heat_extraction: float = 0,
        max_heat_extraction: float = 0,
        index: pd.DatetimeIndex = None,
        location: tuple[float, float] = None,
        node: str = None,
        dr_factor= None,
        **kwargs,
    ):
        super().__init__(
            id, 
            technology=technology, 
            node=node,
            bidding_strategies=bidding_strategies,
            index=index,
            )
        self.max_thermal_output: float = None
        self.min_thermal_output: float = None

        self.cop_curve = cop_curve
        self.source_temp = source_temp
        self.sink_temp = sink_temp
        self.cop_curve = cop_curve

        self.electricity_price =(
            electricity_price if electricity_price is not None else pd.Series(0, index=index)
        )
        self.fixed_cost = fixed_cost

        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.min_operating_time = min_operating_time
        self.min_down_time = min_down_time
        self.downtime_hot_start = downtime_hot_start
        
        self.heat_extraction = heat_extraction
        self.max_heat_extraction = max_heat_extraction

        self.location = location

        self.dr_factor = dr_factor

    def reset(self):
        """Reset the unit to its initial state."""

        self.current_status = 1
        self.current_down_time = self.min_down_time

        self.total_heat_output = pd.Series(0.0, index=self.index)

        self.pos_capacity_reserve = pd.Series(0.0, index=self.index)
        self.neg_capacity_reserve = pd.Series(0.0, index=self.index)

        self.mean_market_success = 0
        self.market_success_list = [0]

    def calculate_delta_t(source_temperature, sink_temperature):
        """
        Calculates the temperature difference between the source and sink temperatures for a heat pump.
        
        Parameters:
        source_temperature (float): the current temperature of the heat source, in degrees Celsius
        sink_temperature (float): the current temperature of the heat sink, in degrees Celsius
        
        Returns:
        float: the temperature difference between the source and sink temperatures, in degrees Celsius
        """
        delta_t = sink_temperature - source_temperature
        return delta_t
    
    def calculate_cop(delta_t, source="air"):
        """
        Calculates the COP of a heat pump given the temperature difference between the source and sink temperatures.
        
        Parameters:
        delta_t (float): temperature difference between the source and sink temperatures, in degrees Celsius
        heat_pump_type (str): type of heat pump, either 'ASHP' for air-sourced heat pumps or 'GSHP' for ground-sourced heat pumps
        
        Returns:
        float: the calculated COP
        """
        
        if source =="air":
            cop = 6.81 + 0.121 * delta_t + 0.000630 * delta_t ** 2
        elif source == 'soil':
            cop = 8.77 + 0.150 * delta_t + 0.000734 * delta_t ** 2
        else:
            raise ValueError("Invalid heat pump type. Must be either 'ASHP' or 'GSHP'")
            
        return cop

    def calculate_operational_window(
        self,
        source: str,
        product_type: str,
        current_time: pd.Timestamp, 
        demand_response_signal=None
    ) -> dict:
        """Calculate the operation window for the next time step.

        Returns
        -------
        operational_window : dict
            Dictionary containing the operational window for the next time step.
        """

        # Calculate the maximum and minimum heat that can be produced during the time window
        max_heat = min(self.max_heat, self.calculate_cop(source=source) * self.max_power)
        min_heat = min(self.min_heat, self.calculate_cop(source=source) * self.min_power)
        
        if self.current_status == 0 and self.current_down_time < self.min_down_time:
            return None

        current_heat = self.total_heat_output.at[current_time]

        min_heat = (
            self.min_heat[current_time]
            if type(self.min_heat) is pd.Series
            else self.min_heat
        )
        if self.ramp_down != -1:
            min_heat = max(current_heat - self.ramp_down, min_heat)
        else:
            min_heat = min_heat

        max_heat = (
            self.max_heat[current_time]
            if type(self.max_heat) is pd.Series
            else self.max_heat
        )
        if self.ramp_up != -1:
            max_heat = min(current_heat + self.ramp_up, max_heat)
        else:
            max_heat = max_heat

        operational_window = {
            "current_heat": {
                "heat": current_heat,
                "marginal_cost": self.calc_marginal_cost(
                    heat_output=current_heat,
                    current_time=current_time,
                    partial_load_eff=True,
                ),
            },
            "min_power": {
                "power": min_heat,
                "marginal_cost": self.calc_marginal_cost(
                    heat_output=min_heat,
                    current_time=current_time,
                    partial_load_eff=True,
                ),
            },
            "max_power": {
                "power": max_heat,
                "marginal_cost": self.calc_marginal_cost(
                    heat_output=max_heat,
                    current_time=current_time,
                    partial_load_eff=True,
                ),
            },
        }
        
        if demand_response_signal is not None:
            # If demand response signal is high, reduce heat output by the specified factor
            if demand_response_signal > 0:
                operational_window["current_heat_output"]["heat"] *= 1 - self.dr_factor
                operational_window["min_heat_output"]["heat"] *= 1 - self.dr_factor
                operational_window["max_heat_output"]["heat"] *= 1 - self.dr_factor

        return operational_window
    
    def get_dispatch_plan(self, dispatch_plan, current_time):
        if dispatch_plan["total_capacity"] > self.min_power:
            self.market_success_list[-1] += 1
            self.current_status = 1
            self.current_down_time = 0
            self.total_heat_output.at[current_time] = dispatch_plan["total_capacity"]

        elif dispatch_plan["total_capacity"] < self.min_power:
            self.current_status = 0
            self.current_down_time += 1
            self.total_heat_output.at[current_time] = 0

            if self.market_success_list[-1] != 0:
                self.mean_market_success = sum(self.market_success_list) / len(
                    self.market_success_list
                )
                self.market_success_list.append(0)
        

    def calc_price(self, outside_temp: float):
        """
        Calculate the marginal cost of the heat pump in €/MWh.
        """
        cop = self.calculate_cop(outside_temp)
        return self.electricity_price / cop + self.fixed_cost