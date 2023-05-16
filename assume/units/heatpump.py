from assume.strategies import BaseStrategy
from assume.units.base_unit import BaseUnit
import numpy as np
import pandas as pd


class HeatPump(BaseUnit):
    def __init__(
        self,
        id: str,
        technology: str,
        bidding_strategies: dict,
        max_thermal_output: float or pd.Series,
        min_thermal_output: float or pd.Series,
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

        self.location = location

        self.dr_factor = dr_factor

    def reset(self):
        """Reset the unit to its initial state."""

        self.current_status = 1
        self.current_down_time = self.min_down_time

        self.total_thermal_output = pd.Series(0.0, index=self.index)
        self.total_thermal_output.iat[0] = self.min_power

        self.pos_capacity_reserve = pd.Series(0.0, index=self.index)
        self.neg_capacity_reserve = pd.Series(0.0, index=self.index)

        self.mean_market_success = 0
        self.market_success_list = [0]

    def calculate_delta_t(source_temp, sink_temp):
        """
        Calculates the temperature difference between the source and sink temperatures for a heat pump.
        
        Parameters:
        source_temperature (float): the current temperature of the heat source, in degrees Celsius
        sink_temperature (float): the current temperature of the heat sink, in degrees Celsius
        
        Returns:
        float: the temperature difference between the source and sink temperatures, in degrees Celsius
        """
        delta_t = sink_temp - source_temp
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
        product_tuple: tuple,
        current_time: pd.Timestamp, 
        demand_response_signal=None
    ) -> dict:
        """Calculate the operation window for the next time step.

        Returns
        -------
        operational_window : dict
            Dictionary containing the operational window for the next time step.
        """
        start, end, only_hours = product_tuple
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        cop = self.calculate_cop(source)

        # Calculate the maximum and minimum heat that can be produced during the time window
        #max_heat = min(self.max_heat, self.calculate_cop(source=source) * self.max_power)
        #min_heat = min(self.min_heat, self.calculate_cop(source=source) * self.min_power)
        
        if self.current_status == 0 and self.current_down_time < self.min_down_time:
            return None
        
        # Calculate the maximum and minimum heat that can be produced during the time window
        max_power = self.max_thermal_output.loc[current_time]/ cop.loc[current_time]
        min_power = self.min_thermal_output.loc[current_time]/ cop.loc[current_time]
        current_power_input = self.total_thermal_output.loc[current_time]/ cop.loc[current_time]

        # check if min_power is a series or a float
        min_thermal_output = (
            self.min_thermal_output.at[start]
            if type(self.min_thermal_output) is pd.Series
            else self.min_thermal_output
        )

        # adjust for ramp down speed
        if self.ramp_down != -1:
            min_power = max(current_power_input - self.ramp_down, min_power)
        else:
            min_power = min_power

        # adjust min_power if sold negative reserve capacity on control reserve market
        min_power = min_power + self.neg_capacity_reserve.at[start]

        # check if max_power is a series or a float
        max_power = (
            self.max_thermal_output.at[start]
            if type(self.max_thermal_output) is pd.Series
            else self.max_thermal_output
        )

        # adjust for ramp up speed
        if self.ramp_up != -1:
            max_power = min(current_power_input + self.ramp_up, max_power)
        else:
            max_power = max_power

        # adjust max_power if sold positive reserve capacity on control reserve market
        max_power = max_power - self.pos_capacity_reserve.at[start]

        operational_window = {
            "window": {"start": start, "end": end},
            "current_power": {
                "power": current_power_input,
                "marginal_cost": self.calc_marginal_cost(
                    source,
                    timestep=start,
                ),
            },
            "min_power": {
                "power": min_power,
                "marginal_cost": self.calc_marginal_cost(
                    source,
                    timestep=start,
                ),
            },
            "max_power": {
                "power": max_power,
                "marginal_cost": self.calc_marginal_cost(
                    source,
                    timestep=start,
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
    
    def calculate_bids(
        self,
        product_type,
        product_tuple,
    ):
        return super().calculate_bids(
            product_type=product_type,
            product_tuple=product_tuple,
        )
    
    def get_dispatch_plan(self, dispatch_plan, current_time):
        if dispatch_plan["total_capacity"] > self.min_power:
            self.market_success_list[-1] += 1
            self.current_status = 1
            self.current_down_time = 0
            self.total_thermal_output.at[current_time] = dispatch_plan["total_capacity"]

        elif dispatch_plan["total_capacity"] < self.min_power:
            self.current_status = 0
            self.current_down_time += 1
            self.total_thermal_output.at[current_time] = 0

            if self.market_success_list[-1] != 0:
                self.mean_market_success = sum(self.market_success_list) / len(
                    self.market_success_list
                )
                self.market_success_list.append(0)
    
    def calc_marginal_cost(self, current_time: pd.Timestamp, source) -> float:
        """
        Calculate the marginal cost for the heat pump at the given time step.

        Parameters
        ----------
        current_time : pd.Timestamp
            The current time step.

        Returns
        -------
        marginal_cost : float
            The calculated marginal cost.
        """
        electricity_price = self.electricity_price.loc[current_time]
        cop = self.calculate_cop(source)
        cop_t = self.cop.loc[current_time]
        marginal_cost = electricity_price / cop_t

        return marginal_cost