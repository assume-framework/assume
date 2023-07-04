import numpy as np
import pandas as pd

from assume.strategies import BaseStrategy
from assume.units.base_unit import BaseUnit


class HeatPump(BaseUnit):
    def __init__(
        self,
        id: str,
        technology: str,
        bidding_strategies: dict,
        # max_thermal_output: float or pd.Series,
        # min_thermal_output: float or pd.Series,
        max_power: float or pd.Series,
        min_power: float or pd.Series,
        volume: float or pd.Series = 1000,
        electricity_price: pd.Series = pd.Series(),
        ramp_up: float = -1,
        ramp_down: float = 1,
        fixed_cost: float = 0,
        min_operating_time: float = 0,
        min_down_time: float = 0,
        downtime_hot_start: int = 0.001,  # hours
        downtime_warm_start: int = 0.005,  # hours
        source: str = None,
        source_temp: float or pd.Series = 15,
        sink_temp: float or pd.Series = 35,
        index: pd.DatetimeIndex = None,
        location: tuple[float, float] = None,
        node: str = None,
        dr_factor=None,
        **kwargs,
    ):
        super().__init__(
            id=id,
            technology=technology,
            node=node,
            bidding_strategies=bidding_strategies,
            index=index,
        )

        self.source_temp = source_temp
        self.sink_temp = sink_temp
        self.max_power = max_power
        self.min_power = min_power
        self.source = source
        # self.max_thermal_output = min_thermal_output
        # self.min_thermal_output = min_thermal_output

        self.volume = volume
        self.electricity_price = (
            electricity_price
            if electricity_price is not None
            else pd.Series(0, index=index)
        )
        self.sink_temp = (
            sink_temp if sink_temp is not None else pd.Series(0, index=index)
        )
        self.source_temp = (
            source_temp if source_temp is not None else pd.Series(0, index=index)
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

        self.outputs["heat"] = pd.Series(0.0, index=self.index)
        min_thermal_output = self.outputs["heat"].loc[
            self.index[0] : self.index[0] + pd.Timedelta("24h")
        ]

        self.outputs["pos_capacity"] = pd.Series(0.0, index=self.index)
        self.outputs["neg_capacity"] = pd.Series(0.0, index=self.index)

    def calculate_delta_t(self, source_temp, sink_temp):
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

    def calculate_cop(self):
        """
        Calculates the COP of a heat pump given the temperature difference between the source and sink temperatures.

        Parameters:
        delta_t (float): temperature difference between the source and sink temperatures, in degrees Celsius
        heat_pump_type (str): type of heat pump, either 'ASHP' for air-sourced heat pumps or 'GSHP' for ground-sourced heat pumps

        Returns:
        float: the calculated COP
        """
        delta_t = self.calculate_delta_t(
            source_temp=self.source_temp, sink_temp=self.sink_temp
        )
        if self.source == "air":
            cop = 6.81 + 0.121 * delta_t + 0.000630 * delta_t**2
        elif self.source == "soil":
            cop = 8.77 + 0.150 * delta_t + 0.000734 * delta_t**2
        else:
            raise ValueError("Invalid heat pump type. Must be either 'ASHP' or 'GSHP'")

        return cop

    def calculate_operational_window(
        self,
        # source: str,
        product_type: str,
        product_tuple: tuple,
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
        timestep: pd.Timestamp
        cop = self.calculate_cop()

        if self.current_status == 0 and self.current_down_time < self.min_down_time:
            return None

        # check if min_power is a series or a float
        min_power = (
            self.min_power.at[start]
            if type(self.min_power) is pd.Series
            else self.min_power
        )

        # check if max_power is a series or a float
        max_power = (
            self.max_power.at[start]
            if type(self.max_power) is pd.Series
            else self.max_power
        )

        # Calculate the maximum and minimum heat that can be produced during the time window
        max_thermal_output = max_power * cop.at[start]
        min_thermal_output = min_power * cop.at[start]

        if type(min_thermal_output) == pd.Series:
            bid_volume = min_power[start]
        else:
            bid_volume = self.volume

        current_power_input = self.outputs["heat"].at[start] / cop.at[start]

        # adjust for ramp down speed
        if self.ramp_down != -1:
            min_power = max(current_power_input - self.ramp_down, min_power)
        else:
            min_power = min_power

        # adjust min_power if sold negative reserve capacity on control reserve market
        min_power = min_power + self.neg_capacity_reserve.at[start]

        # adjust for ramp up speed
        if self.ramp_up != -1:
            max_power = min(current_power_input + self.ramp_up, max_power)
        else:
            max_power = max_power

        # adjust max_power if sold positive reserve capacity on control reserve market
        max_power = max_power - self.outputs["pos_capacity"].at[start]

        operational_window = {
            "window": {"start": start, "end": end},
            "current_power": {
                "power": -current_power_input,
                "marginal_cost": self.calc_marginal_cost(
                    timestep=start,
                ),
            },
            "min_power": {
                "power": -min_power,
                "marginal_cost": self.calc_marginal_cost(
                    timestep=start,
                ),
            },
            "max_power": {
                "power": -max_power,
                "marginal_cost": self.calc_marginal_cost(
                    timestep=start,
                ),
            },
        }

        return operational_window

    def calculate_bids(self, market_config, product_tuple):
        return super().calculate_bids(
            market_config=market_config,
            product_tuple=product_tuple,
        )

    def set_dispatch_plan(
        self,
        dispatch_plan: dict,
        start: pd.Timestamp,
        end: pd.Timestamp,
        product_type: str,
    ):
        # TODO checks should be at execute_current_dispatch - see powerplant
        if dispatch_plan["total_power"] > self.min_power:
            self.market_success_list[-1] += 1
            self.current_status = 1
            self.current_down_time = 0
            self.outputs["heat"].loc[time_period] = dispatch_plan["total_power"]

        elif dispatch_plan["total_power"] < self.min_power:
            self.current_status = 0
            self.current_down_time += 1
            self.outputs["heat"].loc[time_period] = 0

    def calc_marginal_cost(self, timestep: pd.Timestamp) -> float or pd.Series:
        """
        Calculate the marginal cost for the heat pump at the given time step.

        Parameters
        ----------
        current_time : pd.Timestamp
            The current time step.

        Returns
        -------
        bid_price : float
            The calculated bid_price.
        """
        cop = self.calculate_cop()
        cop_t = cop.loc[timestep]

        if type(self.electricity_price) == pd.Series:
            bid_price = self.electricity_price.at[timestep] / cop_t
        else:
            bid_price = self.electricity_price / cop_t

        return bid_price
