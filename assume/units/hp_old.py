import pandas as pd

from assume.common.base import SupportsMinMax


class HP(SupportsMinMax):
    """A class for a heatpump unit.

    :param id: unique identifier for the unit
    :type id: str
    :param technology: the technology of the unit
    :type technology: str
    :param bidding_strategies: the bidding strategies of the unit
    :type bidding_strategies: dict
    :param max_power: the maximum power output of the unit (kW)
    :type max_power: float | pd.Series
    :param min_power: the minimum power output of the unit (kW)
    :type min_power: float | pd.Series
    :param volume: the volume of the heat pump
    :type volume: float | pd.Series
    :param ramp_up: the ramp up speed of the unit (MW/15Minutes)
    :type ramp_up: float
    :param ramp_down: the ramp down speed of the unit (MW/15Minutes)
    :type ramp_down: float
    :param fixed_cost: the fixed cost of the unit (€/MW)
    :type fixed_cost: float
    :param min_operating_time: the minimum operating time of the unit
    :type min_operating_time: float
    :param min_down_time: the minimum down time of the unit
    :type min_down_time: float
    :param downtime_hot_start: the downtime hot start of the unit (hours)
    :type downtime_hot_start: int
    :param downtime_warm_start: the downtime warm start of the unit (hours)
    :type downtime_warm_start: int
    :param source: the source of the heat pump
    :type source: str
    :param index: the index of the unit
    :type index: pd.DatetimeIndex
    :param location: the location of the unit (latitude, longitude)
    :type location: tuple[float, float]
    :param node: the node of the unit
    :type node: str
    :param dr_factor: the demand response factor of the unit
    :type dr_factor: float
    :param kwargs: additional keyword arguments
    :type kwargs: dict"""

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
        ramp_up: float = -1,
        ramp_down: float = 1,
        fixed_cost: float = 0,
        min_operating_time: float = 0,
        min_down_time: float = 0,
        downtime_hot_start: int = 0.001,  # hours
        downtime_warm_start: int = 0.005,  # hours
        source: str = None,
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

        self.max_power = max_power
        self.min_power = min_power
        self.source = source
        # self.max_thermal_output = min_thermal_output
        # self.min_thermal_output = min_thermal_output

        self.volume = volume
        self.electricity_price = self.forecaster["electricity_price"]
        self.fixed_cost = fixed_cost

        # check ramping enabled
        self.ramp_down = max_power if ramp_down == -1 else ramp_down
        self.ramp_up = max_power if ramp_up == -1 else ramp_up
        self.min_operating_time = min_operating_time
        self.min_down_time = min_down_time
        self.downtime_hot_start = downtime_hot_start

        self.location = location

        self.dr_factor = dr_factor

    def calculate_cop(self):
        """
        Calculates the COP of a heat pump given the temperature difference between the source and sink temperatures.
        Returns the calculated COP as a float.

        :param heat_pump_type: the type of heat pump, either 'air' for air-sourced heat pumps or 'soil' for ground-sourced heat pumps
        :type heat_pump_type: str
        :return: the calculated COP
        :rtype: float
        """
        delta_t = (
            self.forecaster["sink_temperature"] - self.forecaster["source_temperature"]
        )
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
        Returns None if the unit is not available for dispatch.
        Returns the min and max power if the unit is available for dispatch.

        :param start: the start time of the dispatch
        :type start: pd.Timestamp
        :param end: the end time of the dispatch
        :type end: pd.Timestamp
        :param product_type: the product type of the unit
        :type product_type: str
        :return: the min and max power of the unit
        :rtype: tuple[float]
        """
        end_excl = end - self.index.freq
        heat_demand = self.outputs["heat"][start:end_excl]
        assert heat_demand.min() >= 0

        cop = self.calculate_cop()

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

        :param start: the start time of the dispatch
        :type start: pd.Timestamp
        :param end: the end time of the dispatch
        :type end: pd.Timestamp
        :return: the volume of the unit within the given time range
        :rtype: pd.Series
        """
        end_excl = end - self.index.freq

        if self.outputs["energy"][start:end_excl].min() < self.min_power:
            self.outputs["energy"].loc[start:end_excl] = 0
            self.outputs["heat"].loc[start:end_excl] = 0
        else:
            self.outputs["heat"].loc[start:end_excl] = self.outputs["energy"][
                start:end_excl
            ]
        return self.outputs["energy"][start:end_excl]

    def calc_marginal_cost(self, timestep: pd.Timestamp) -> float | pd.Series:
        """
        Calculate the marginal cost for the heat pump at the given time step.
        Returns the calculated bid price as a float or pd.Series.

        :param timestep: the current time step
        :type timestep: pd.Timestamp
        :return: the calculated bid price
        :rtype: float | pd.Series
        """

        cop = self.calculate_cop()
        cop_t = cop.loc[timestep]

        if isinstance(self.electricity_price, pd.Series):
            bid_price = self.electricity_price.at[timestep] / cop_t
        else:
            bid_price = self.electricity_price / cop_t

        return bid_price
