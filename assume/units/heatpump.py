import logging
import numpy as np
import pandas as pd

from assume.strategies import BaseStrategy
from assume.units.base_unit import BaseUnit
logger = logging.getLogger(__name__)

class HeatPump(BaseUnit):
    def __init__(
        self,
        id: str,
        unit_operator: str,
        technology: str,
        bidding_strategies: dict,
        max_power: float or pd.Series,
        min_power: float or pd.Series,
        volume: float or pd.Series,
        electricity_price: float or pd.Series = 0.0,
        ramp_up: float = -1,
        ramp_down: float = -1,
        fixed_cost: float = 0,
        min_operating_time: float = 0,
        min_down_time: float = 0,
        downtime_hot_start: int = 0,  # hours
        downtime_warm_start: int = 0,  # hours
        source: str = None,
        source_temp: float or pd.Series = 15,
        sink_temp: float or pd.Series = 35,
        index: pd.DatetimeIndex = None,
        location: tuple[float, float] = (0.0, 0.0),
        node: str = "bus0",
        dr_factor=None,
        **kwargs,
    ):
        super().__init__(
            id=id,
            technology=technology,
            unit_operator=unit_operator,
            node=node,
            bidding_strategies=bidding_strategies,
            index=index,
        )

        self.source_temp = source_temp
        self.sink_temp = sink_temp
        self.max_power = max_power
        self.min_power = min_power
        self.source = source

        self.volume = -volume  # demand is negative
        self.electricity_price = electricity_price
        self.sink_temp = (
            sink_temp if sink_temp is not None else pd.Series(0, index=index)
        )
        self.source_temp = (
            source_temp if source_temp is not None else pd.Series(0, index=index)
        )
        self.fixed_cost = fixed_cost

        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        # check ramping enabled
        if self.ramp_down == -1:
            self.ramp_down = max_power
        if self.ramp_up == -1:
            self.ramp_up = max_power
        self.min_operating_time = min_operating_time if min_operating_time > 0 else 1
        self.min_down_time = min_down_time if min_down_time > 0 else 1
        self.downtime_hot_start = downtime_hot_start
        self.downtime_warm_start = downtime_warm_start
        self.location = location
        self.total_power_output = []
        self.dr_factor = dr_factor

    def reset(self):
        """Reset the unit to its initial state."""

        self.current_status = 1
        self.current_down_time = self.min_down_time

        self.total_power_output = pd.Series(0.0, index=self.index)

        self.pos_capacity_reserve = pd.Series(0.0, index=self.index)
        self.neg_capacity_reserve = pd.Series(0.0, index=self.index)

    def calculate_delta_t(self, timestep: pd.Timestamp):
        """
        Calculates the temperature difference between the source and sink temperatures for a heat pump.

        Parameters:
        source_temperature (float): the current temperature of the heat source, in degrees Celsius
        sink_temperature (float): the current temperature of the heat sink, in degrees Celsius

        Returns:
        float: the temperature difference between the source and sink temperatures, in degrees Celsius
        """

        sink_temp = (
            self.sink_temp.at[timestep]
            if type(self.sink_temp) is pd.Series
            else self.sink_temp
        )
        source_temp = (
            self.source_temp.at[timestep]
            if type(self.source_temp) is pd.Series
            else self.source_temp
        )

        delta_t = sink_temp - source_temp

        return delta_t

    def calculate_cop(self, start: pd.Timestamp):
        """
        Calculates the COP of a heat pump given the temperature difference between the source and sink temperatures.

        Parameters:
        delta_t (float): temperature difference between the source and sink temperatures, in degrees Celsius
        heat_pump_type (str): type of heat pump, either 'ASHP' for air-sourced heat pumps or 'GSHP' for ground-sourced heat pumps

        Returns:
        float: the calculated COP
        """
        delta_t = self.calculate_delta_t(start)
        if self.source == "air":
            cop = 6.81 + 0.121 * delta_t + 0.000630 * delta_t**2
        elif self.source == "soil":
            cop = 8.77 + 0.150 * delta_t + 0.000734 * delta_t**2
        else:
            raise ValueError("Invalid heat pump type. Must be either 'ASHP' or 'GSHP'")

        return cop

    def calculate_operational_window(
        self,
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

        if product_type == "energy":
            return self.calculate_energy_operational_window(start, end)
        elif product_type in {"capacity_pos", "capacity_neg"}:
            return self.calculate_reserve_operational_window(start, end)

    def calculate_energy_operational_window(
            self, start: pd.Timestamp, end: pd.Timestamp
    ) -> dict:

        if self.current_status == 0 and self.current_down_time < self.min_down_time:
            return None

        # cop = self.calculate_cop(start)
        # current_power_input = self.total_thermal_output.at[start - self.index.freq] / cop
        current_power = (
            self.volume.at[start - self.index.freq] / cop
            if type(self.volume) is pd.Series
            else self.volume / cop
        )

        #self.total_thermal_output.at[start - self.index.freq] = bid_volume

        # check if min_power is a series or a float

        min_power = (
            self.min_power[start]
            if type(self.min_power) is pd.Series
            else self.min_power
        )

        # check if max_power is a series or a float
        max_power = (
            self.max_power[start]
            if type(self.max_power) is pd.Series
            else self.max_power
        )

        # adjust for ramp down speed
        min_power = max(current_power - self.ramp_down, min_power)

        # adjust for ramp up speed
        max_power = min(current_power + self.ramp_up, max_power)


        # Calculate the maximum and minimum heat that can be produced during the time window
        # max_thermal_output = max_power * cop
        # min_thermal_output = min_power * cop

        # # adjust min_power if sold negative reserve capacity on control reserve market
        # min_power = min_power #+ self.neg_capacity_reserve.at[start]
        
        # # adjust max_power if sold positive reserve capacity on control reserve market
        # max_power = max_power #- self.pos_capacity_reserve.at[start]

        operational_window = {
            "window": {"start": start, "end": end},
            "current_power": {
                "power": current_power,
                "marginal_cost": self.calc_marginal_cost(
                    timestep=start,
                ),
            },
            "min_power": {
                "power": min_power,
                "marginal_cost": self.calc_marginal_cost(
                    timestep=start,
                ),
            },
            "max_power": {
                "power": max_power,
                "marginal_cost": self.calc_marginal_cost(
                    timestep=start,
                ),
            },
        }

        return operational_window


    #To be developed
    def calculate_reserve_operational_window(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> dict:

        # cop = self.calculate_cop()
        # current_power_input = self.total_thermal_output.at[start - self.index.freq] / cop.at[start, end]
        #
        # operational_window = {
        #     "window": {"start": start, "end": end},
        #     "pos_reserve": {
        #         "capacity": available_pos_reserve,
        #     },
        #     "neg_reserve": {
        #         "capacity": available_neg_reserve,
        #     },
        # }
        #
        # if available_neg_reserve < 0:
        #     logger.error("available_neg_reserve < 0")

        # return operational_window
        pass

    def calculate_bids(
            self,
            product_type,
            product_tuple
    ):
        return super().calculate_bids(
            product_type=product_type,
            product_tuple=product_tuple,
        )

    def get_dispatch_plan(
            self,
            dispatch_plan: dict,
            start: pd.Timestamp,
            end: pd.Timestamp,
            product_type: str,
    ):
        end_excl = end - self.index.freq
        self.total_power_output.loc[start:end_excl] += dispatch_plan["total_power"]

        if self.total_power_output[start:end_excl].min() < self.min_power:
            self.total_power_output.loc[start:end_excl] = 0
            self.current_status = 0
            self.current_down_time += 1

        else:
            self.current_status = 1
            self.current_down_time = 0

    def calc_marginal_cost(
            self,
            timestep: pd.Timestamp
    ) -> float or pd.Series:
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
        # cop = self.calculate_cop()
        # cop_t = cop.loc[timestep]

        electricity_price = (
            self.electricity_price.at[timestep]
            if type(self.electricity_price) is pd.Series
            else self.electricity_price
        )

        marginal_cost = (
                electricity_price + self.fixed_cost
        )

        bid_price = marginal_cost

        return bid_price
