import pandas as pd

from assume.common.base import SupportsMinMax


class HeatPump(SupportsMinMax):
    def __init__(
        self,
        id: str,
        technology: str,
        bidding_strategies: dict,
        # max_thermal_output: float | pd.Series,
        # min_thermal_output: float | pd.Series,
        max_power: float | pd.Series,
        min_power: float | pd.Series,
        volume: float | pd.Series = 1000,
        electricity_price: pd.Series = pd.Series(),
        ramp_up: float = -1,
        ramp_down: float = 1,
        fixed_cost: float = 0,
        min_operating_time: float = 0,
        min_down_time: float = 0,
        downtime_hot_start: int = 0.001,  # hours
        downtime_warm_start: int = 0.005,  # hours
        source: str = None,
        source_temp: float | pd.Series = 15,
        sink_temp: float | pd.Series = 35,
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

        # check ramping enabled
        self.ramp_down = max_power if ramp_down == -1 else ramp_down
        self.ramp_up = max_power if ramp_up == -1 else ramp_up
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

    def calculate_cop(self):
        """
        Calculates the COP of a heat pump given the temperature difference between the source and sink temperatures.

        Parameters:
        heat_pump_type (str): type of heat pump, either 'air' for air-sourced heat pumps or 'soil' for ground-sourced heat pumps

        Returns:
        float: the calculated COP
        """
        delta_t = self.sink_temp - self.source_temp
        if self.source == "air":
            cop = 6.81 + 0.121 * delta_t + 0.000630 * delta_t**2
        elif self.source == "soil":
            cop = 8.77 + 0.150 * delta_t + 0.000734 * delta_t**2
        else:
            raise ValueError("Invalid heat pump type. Must be either 'air' or 'soil'")

        return cop

    def calculate_min_max_power(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type="energy"
    ) -> tuple[float]:
        """Calculate the min and max power for the given time step.

        Returns
        -------
        min_power : float
        max_power : float
        """
        end_excl = end - self.index.freq
        heat_demand = self.outputs["heat"][start:end_excl]
        assert heat_demand.min() >= 0

        cop = self.calculate_cop()

        if self.current_status == 0 and self.current_down_time < self.min_down_time:
            return None

        # check if min_power is a series or a float
        min_power = (
            self.min_power.at[start]
            if isinstance(self.min_power, pd.Series)
            else self.min_power
        )

        # check if max_power is a series or a float
        max_power = (
            self.max_power.at[start]
            if isinstance(self.max_power, pd.Series)
            else self.max_power
        )

        # Calculate the maximum and minimum heat that can be produced during the time window
        # TODO needs fixing
        max_thermal_output = max_power * cop.at[start]
        min_thermal_output = min_power * cop.at[start]
        current_power_input = self.outputs["energy"].at[start] / cop.at[start]

        # adjust for ramp down speed
        max(current_power_input - self.ramp_down, min_power)

        # adjust min_power if sold negative reserve capacity on control reserve market
        min_power = min_power + self.neg_capacity_reserve.at[start]

        # adjust for ramp up speed
        max_power = min(current_power_input + self.ramp_up, max_power)

        # adjust max_power if sold positive reserve capacity on control reserve market
        max_power = max_power - self.outputs["pos_capacity"].at[start]

        return -min_power, -max_power

    def execute_current_dispatch(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ):
        """
        check if the total dispatch plan is feasible
        This checks if the market feedback is feasible for the given unit.
        And sets the closest dispatch if not.
        The end param should be inclusive.
        """
        end_excl = end - self.index.freq

        if self.outputs["energy"][start:end_excl].min() < self.min_power:
            self.outputs["energy"].loc[start:end_excl] = 0
            self.outputs["heat"].loc[start:end_excl] = 0
            self.current_status = 0
            self.current_down_time += 1
        else:
            self.current_status = 1
            self.current_down_time = 0
            self.outputs["heat"].loc[start:end_excl] = self.outputs["energy"][
                start:end_excl
            ]
        return self.outputs["energy"][start:end_excl]

    def calc_marginal_cost(self, timestep: pd.Timestamp) -> float | pd.Series:
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

        if isinstance(self.electricity_price, pd.Series):
            bid_price = self.electricity_price.at[timestep] / cop_t
        else:
            bid_price = self.electricity_price / cop_t

        return bid_price
