import pandas as pd

from assume.common.base import BaseUnit


class Electrolyser(BaseUnit):
    """
    An electrolyser unit.

    :param id: unique identifier for the unit
    :type id: str
    :param technology: the technology of the unit
    :type technology: str
    :param bidding_strategies: the bidding strategies of the unit
    :type bidding_strategies: dict
    :param max_hydrogen_output: the maximum hydrogen output of the unit (kW)
    :type max_hydrogen_output: float | pd.Series
    :param min_hydrogen_output: the minimum hydrogen output of the unit (kW)
    :type min_hydrogen_output: float | pd.Series
    :param efficiency: the efficiency of the unit
    :type efficiency: float | pd.Series
    :param volume: the volume of the electrolyser (MWh)
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
    :param index: the index of the unit
    :type index: pd.DatetimeIndex
    :param location: the location of the unit (latitude, longitude)
    :type location: tuple[float, float]
    :param node: the node of the unit
    :type node: str
    :param dr_factor: the demand response factor of the unit
    :type dr_factor: float
    :param kwargs: additional keyword arguments
    :type kwargs: dict
    """

    def __init__(
        self,
        id: str,
        technology: str,
        bidding_strategies: dict,
        max_hydrogen_output: float | pd.Series,
        min_hydrogen_output: float | pd.Series,
        efficiency: float | pd.Series,
        volume: float | pd.Series = 1000,
        ramp_up: float = -1,
        ramp_down: float = 1,
        fixed_cost: float = 0,
        min_operating_time: float = 0,
        min_down_time: float = 0,
        downtime_hot_start: int = 0.001,  # hours
        downtime_warm_start: int = 0.005,  # hours
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

        self.max_hydrogen_output = max_hydrogen_output
        self.min_hydrogen_output = min_hydrogen_output
        self.efficiency = efficiency
        self.volume = volume
        self.electricity_price = self.forecaster["electricity_price"]
        self.fixed_cost = fixed_cost
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.min_operating_time = min_operating_time
        self.min_down_time = min_down_time
        self.downtime_hot_start = downtime_hot_start
        self.location = location
        self.dr_factor = dr_factor

    def calculate_min_max_power(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type="energy"
    ) -> tuple[float]:
        """
        Calculate the operational window for the next time step.
        Returns None if the unit is not available for dispatch.
        Returns the operational window if the unit is available for dispatch.

        :param start: the start time of the dispatch
        :type start: pd.Timestamp
        :param end: the end time of the dispatch
        :type end: pd.Timestamp
        :param product_type: the product type of the unit
        :type product_type: str
        :return: the operational window of the unit
        :rtype: tuple[float]
        """
        max_power = self.max_hydrogen_output.at[start]
        min_power = self.min_hydrogen_output.at[start]

        # Adjust for ramp down speed
        if self.ramp_down != -1:
            min_power = max(0, min_power - self.ramp_down)
        else:
            min_power = min_power

        current_power_input = self.outputs["hydrogen"].at[start]

        # Adjust min_power if sold negative reserve capacity on control reserve market
        min_power = min_power + self.neg_capacity_reserve.at[start]

        # Adjust for ramp up speed
        max_power = min(current_power_input + self.ramp_up, max_power)

        # Adjust max_power if sold positive reserve capacity on control reserve market
        max_power = max_power - self.outputs["pos_capacity"].at[start]

        return min_power, max_power

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
            self.outputs["hydrogen"].loc[start:end_excl] = 0
        else:
            self.outputs["hydrogen"].loc[start:end_excl] = self.outputs["energy"][
                start:end_excl
            ]
        return self.outputs["energy"][start:end_excl]

    def calc_marginal_cost(
        self,
        timestep: pd.Timestamp,
    ) -> float | pd.Series:
        """
        Calculate the marginal cost for the electrolyser at the given time step.
        Returns the calculated bid price.

        :param timestep: the current time step
        :type timestep: pd.Timestamp
        :return: The calculated bid price.
        :rtype: float | pd.Series
        """

        efficiency_t = self.efficiency.loc[timestep]

        if isinstance(self.electricity_price, pd.Series):
            bid_price = self.electricity_price.at[timestep] / efficiency_t
        else:
            bid_price = self.electricity_price / efficiency_t

        return bid_price
