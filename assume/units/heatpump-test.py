from ..strategies import BaseStrategy
from .base_unit import BaseUnit
import numpy as np


class HeatPump(BaseUnit):
    def __init__(
        self,
        id: str,
        technology: str,
        bidding_strategies: dict,
        max_heat: float,
        min_heat: float,
        cop_curve: dict,
        cold_source_temp: float,
        hot_source_temp: float,
        electricity_price: float,
        ramp_up: float,
        ramp_down: float,
        fixed_cost: float = 0,
        min_operating_time: float = 0,
        min_down_time: float = 0,
        downtime_hot_start: int = 0.001,  # hours
        downtime_warm_start: int = 0.005,  # hours
        availability: dict = None,
        node: str = None,
        location: tuple[float, float] = None,
        dr_factor= None,
        **kwargs,
    ):
        super().__init__(id, technology, node, bidding_strategy)
        self.max_heat = max_heat
        self.min_heat = min_heat
        self.cold_source_temp = cold_source_temp
        self.hot_source_temp = hot_source_temp
        self.cop_curve = cop_curve
        self.ramp_up = ramp_up if ramp_up > 0 else max_heat
        self.ramp_down = ramp_down if ramp_down > 0 else max_heat
        # self.min_operating_time = max(min_operating_time, 0)
        # self.min_down_time = max(min_down_time, 0)
        self.electricity_price = electricity_price
        self.fixed_cost = fixed_cost
        self.location = location
        self.availability = availability
        self.dr_factor = dr_factor

    def reset(self):
        """Reset the unit to its initial state."""

        self.current_time_step = 0
        self.current_status = 1

        self.total_thermal_output = [0.0]
        self.pos_capacity_reserve = [0.0]
        self.neg_capacity_reserve = [0.0]

    def calculate_cop(self, outside_temp):
        """
        Calculate the coefficient of performance (COP) of the heat pump based on the outside air temperature.
        """
        if self.cop_curve is None:
            raise ValueError("COP curve is not defined.")

        # Create arrays of x and y values from the COP curve dictionary
        x = np.array([temp for temp, cop in self.cop_curve.items()])
        y = np.array([cop for temp, cop in self.cop_curve.items()])

        # Use numpy's piecewise function to interpolate between data points
        return np.piecewise(outside_temp,
                             [outside_temp < x[0], outside_temp >= x[-1]] +
                             [(outside_temp >= x[i]) & (outside_temp < x[i+1]) for i in range(len(x)-1)],
                             [y[0], y[-1]] + [(y[i+1] - y[i]) / (x[i+1] - x[i]) * (outside_temp - x[i]) + y[i] for i in range(len(x)-1)])

    def calculate_operational_window(self, outside_temp: float, demand_response_signal=None):
        """
        Calculate the operational window for the next time step based on the outside air temperature and availability.
        Returns a dictionary with keys "current_heat", "min_heat", "max_heat", "marginal_cost".
        """
        if self.availability is None:
            raise ValueError("Availability data is not defined.")

        # Calculate the COP based on the outside air temperature
        cop = self.calculate_cop(outside_temp)

        # Calculate the maximum heat output based on the availability data and COP
        max_thermal_output = min(self.max_heat, self.availability[self.current_time_step] * cop)

        # Calculate the minimum heat output based on the minimum heat and COP
        min_thermal_output = max(self.min_heat, max_thermal_output * cop - self.max_heat + self.min_heat)

        # Calculate the current heat output based on the previous time step's heat output and ramp up/down limits
        current_thermal_output = self.total_thermal_output[-1]
        if current_thermal_output < max_thermal_output - self.ramp_up:
            current_thermal_output += self.ramp_up
        elif current_thermal_output > min_thermal_output + self.ramp_down:
            current_thermal_output -= self.ramp_down
        else:
            current_thermal_output = max_thermal_output

        operational_window = {
            "current_heat_output": {
                "heat": current_thermal_output,
                "marginal_cost": self.calc_marginal_cost,
            },
            "min_heat_output": {
                "heat": min_thermal_output,
                "marginal_cost": self.calc_marginal_cost,
            },
            "max_heat_output": {
                "heat": max_thermal_output,
                "marginal_cost": self.calc_marginal_cost,
            },
        }

        if demand_response_signal is not None:
            # If demand response signal is high, reduce heat output by the specified factor
            if demand_response_signal > 0:
                operational_window["current_heat_output"]["heat"] *= 1 - self.dr_factor
                operational_window["min_heat_output"]["heat"] *= 1 - self.dr_factor
                operational_window["max_heat_output"]["heat"] *= 1 - self.dr_factor

        return operational_window

    def calc_marginal_cost(self, outside_temp: float):
        """
        Calculate the marginal cost of the heat pump in €/MWh.
        """
        cop = self.calculate_cop(outside_temp)
        return self.electricity_price / cop + self.fixed_cost