import logging
from datetime import datetime, timedelta
from functools import lru_cache

import pandas as pd

from assume.common.base import SupportsMinMax

logger = logging.getLogger(__name__)


class PowerPlant(SupportsMinMax):
    """
    A class for a powerplant unit

    :param id: The ID of the storage unit.
    :type id: str
    :param technology: The technology of the storage unit.
    :type technology: str
    :param bidding_strategies: The bidding strategies of the storage unit.
    :type bidding_strategies: dict
    :param index: The index of the storage unit.
    :type index: pd.DatetimeIndex
    :param max_power: The maximum power output capacity of the power plant in MW.
    :type max_power: float
    :param min_power: The minimum power output capacity of the power plant in MW. (Defaults to 0.0 MW)
    :type min_power: float, optional
    :param efficiency: The efficiency of the poewr plant in converting fuel to electricity (Defaults to 1.0)
    :type efficiency: float, optional
    :param fixed_cost: The fixed operating cost of the power plant, independent of the power output (Defaults to 0.0 monetary units)
    :type fixed_cost: float, optional
    :param partial_load_eff: Does the efficiency varies at part loads? (Defaults to False)
    :type partial_load_eff: bool, optional
    :param fuel_type: The type of fuel used by the power plant for power generation (Defaults to "others")
    :type fuel_type: str, optional
    :param emission_factor: The emission factor associated with the power plants fuel type -> CO2 emissions per unit of energy produced (Defaults to 0.0.)
    :type emission_factor: float, optional
    :param ramp_up: The ramp-up rate of the power plant, indicating how quickly it can increase power output (Defaults to -1)
    :type ramp_up: float, optional
    :param ramp_down: The ramp-down rate of the power plant, indicating how quickly it can decrease power output. (Defaults to -1)
    :type ramp_down: float, optional
    :param hot_start_cost: The cost of a hot start, where the power plant is restarted after a recent shutdown.(Defaults to 0 monetary units.)
    :type hot_start_cost: float, optional
    :param warm_start_cost: The cost of a warm start, where the power plant is restarted after a moderate downtime.(Defaults to 0 monetary units.)
    :type warm_start_cost: float, optional
    :param cold_start_cost: The cost of a cold start, where the power plant is restarted after a prolonged downtime.(Defaults to 0 monetary units.)
    :type cold_start_cost: float, optional
    :param min_operating_time: The minimum duration that the power plant must operate once started, in hours.(Defaults to 0 hours.)
    :type min_operating_time: float, optional
    :param min_down_time: The minimum downtime required after a shutdown before the power plant can be restarted, in hours.(Defaults to 0 hours.)
    :type min_down_time: float, optional
    :param downtime_hot_start: The downtime required after a hot start before the power plant can be restarted, in hours.(Defaults to 8 hours.)
    :type downtime_hot_start: int, optional
    :param downtime_warm_start: The downtime required after a warm start before the power plant can be restarted, in hours.( Defaults to 48 hours.)
    :type downtime_warm_start: int, optional
    :param heat_extraction: A boolean indicating whether the power plant can extract heat for external purposes.(Defaults to False.)
    :type heat_extraction: bool, optional
    :param max_heat_extraction: The maximum amount of heat that the power plant can extract for external use, in some suitable unit.(Defaults to 0.)
    :type max_heat_extraction: float, optional
    :param location: The geographical coordinates (latitude and longitude) of the power plant's location.(Defaults to (0.0, 0.0).)
    :type location: tuple[float, float], optional
    :param node: The identifier of the electrical bus or network node to which the power plant is connected.(Defaults to "bus0".)
    :type node: str, optional
    :param kwargs: Additional keyword arguments to be passed to the base class.
    :type kwargs: dict, optional

    """

    def __init__(
        self,
        id: str,
        unit_operator: str,
        technology: str,
        bidding_strategies: dict,
        index: pd.DatetimeIndex,
        max_power: float,
        min_power: float = 0.0,
        efficiency: float = 1.0,
        fixed_cost: float = 0.0,
        partial_load_eff: bool = False,
        fuel_type: str = "others",
        emission_factor: float = 0.0,
        ramp_up: float | None = None,
        ramp_down: float | None = None,
        hot_start_cost: float = 0,
        warm_start_cost: float = 0,
        cold_start_cost: float = 0,
        min_operating_time: float = 0,
        min_down_time: float = 0,
        downtime_hot_start: int = 8,  # hours
        downtime_warm_start: int = 48,  # hours
        heat_extraction: bool = False,
        max_heat_extraction: float = 0,
        location: tuple[float, float] = (0.0, 0.0),
        node: str = "bus0",
        **kwargs,
    ):
        super().__init__(
            id=id,
            unit_operator=unit_operator,
            technology=technology,
            bidding_strategies=bidding_strategies,
            index=index,
            node=node,
            **kwargs,
        )

        self.max_power = max_power
        self.min_power = min_power
        self.efficiency = efficiency
        self.partial_load_eff = partial_load_eff
        self.fuel_type = fuel_type
        self.emission_factor = emission_factor

        # check ramping enabled
        self.ramp_down = max_power if ramp_down == 0 or ramp_down is None else ramp_down
        self.ramp_up = max_power if ramp_up == 0 or ramp_up is None else ramp_up
        self.min_operating_time = min_operating_time if min_operating_time > 0 else 1
        self.min_down_time = min_down_time if min_down_time > 0 else 1
        self.downtime_hot_start = downtime_hot_start / (
            self.index.freq / timedelta(hours=1)
        )
        self.downtime_warm_start = downtime_warm_start / (
            self.index.freq / timedelta(hours=1)
        )

        self.fixed_cost = fixed_cost
        self.hot_start_cost = hot_start_cost * max_power
        self.warm_start_cost = warm_start_cost * max_power
        self.cold_start_cost = cold_start_cost * max_power

        self.heat_extraction = heat_extraction
        self.max_heat_extraction = max_heat_extraction

        self.location = location

        self.init_marginal_cost()

    def init_marginal_cost(self):
        """
        Initialize the marginal cost of the unit.
        """
        self.marginal_cost = self.calc_simple_marginal_cost()

    def execute_current_dispatch(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ):
        """
        Executes the current dispatch of the unit based on the provided timestamps.
        The dispatch is only executed, if it is in the constraints given by the unit.
        Returns the volume of the unit within the given time range.

        :param start: the start time of the dispatch
        :type start: pd.Timestamp
        :param end: the end time of the dispatch
        :type end: pd.Timestamp
        :return: the volume of the unit within the given time range
        :rtype: float
        """

        max_power = (
            self.forecaster.get_availability(self.id)[start:end] * self.max_power
        )

        for t in self.outputs["energy"][start:end].index:
            current_power = self.outputs["energy"][t]
            previous_power = self.get_output_before(t)

            max_power_t = self.calculate_ramp(previous_power, max_power[t])
            min_power_t = self.calculate_ramp(previous_power, self.min_power)

            if current_power > max_power_t:
                self.outputs["energy"][t] = max_power_t

            elif current_power < min_power_t and current_power > 0:
                self.outputs["energy"][t] = 0

        return self.outputs["energy"].loc[start:end]

    def calc_simple_marginal_cost(
        self,
    ):
        """
        Calculate the marginal cost of the unit (simple method)
        Returns the marginal cost of the unit.

        :return: the marginal cost of the unit
        :rtype: float
        """
        fuel_price = self.forecaster.get_price(self.fuel_type)
        marginal_cost = (
            fuel_price / self.efficiency
            + self.forecaster.get_price("co2") * self.emission_factor / self.efficiency
            + self.fixed_cost
        )

        return marginal_cost

    @lru_cache(maxsize=256)
    def calc_marginal_cost_with_partial_eff(
        self,
        power_output: float,
        timestep: pd.Timestamp = None,
    ) -> float | pd.Series:
        """
        Calculates the marginal cost of the unit based on power output and timestamp, considering partial efficiency.
        Returns the marginal cost of the unit.

        :param power_output: the power output of the unit
        :type power_output: float
        :param timestep: the timestamp of the unit
        :type timestep: pd.Timestamp, optional
        :return: the marginal cost of the unit
        :rtype: float | pd.Series
        """
        fuel_price = self.forecaster.get_price(self.fuel_type).at[timestep]

        capacity_ratio = power_output / self.max_power

        if self.fuel_type in ["lignite", "hard coal"]:
            eta_loss = (
                0.095859 * (capacity_ratio**4)
                - 0.356010 * (capacity_ratio**3)
                + 0.532948 * (capacity_ratio**2)
                - 0.447059 * capacity_ratio
                + 0.174262
            )

        elif self.fuel_type == "combined cycle gas turbine":
            eta_loss = (
                0.178749 * (capacity_ratio**4)
                - 0.653192 * (capacity_ratio**3)
                + 0.964704 * (capacity_ratio**2)
                - 0.805845 * capacity_ratio
                + 0.315584
            )

        elif self.fuel_type == "open cycle gas turbine":
            eta_loss = (
                0.485049 * (capacity_ratio**4)
                - 1.540723 * (capacity_ratio**3)
                + 1.899607 * (capacity_ratio**2)
                - 1.251502 * capacity_ratio
                + 0.407569
            )

        else:
            eta_loss = 0

        efficiency = self.efficiency - eta_loss
        co2_price = self.forecaster.get_price("co2").at[timestep]

        marginal_cost = (
            fuel_price / efficiency
            + co2_price * self.emission_factor / efficiency
            + self.fixed_cost
        )

        return marginal_cost

    def calculate_min_max_power(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type="energy"
    ) -> tuple[pd.Series, pd.Series]:
        """
        Calculate the minimum and maximum power output of the unit.
        Returns the minimum and maximum power output of the unit.
        does not include ramping
        can be used for arbitrary start times in the future

        :param start: the start time of the dispatch
        :type start: pd.Timestamp
        :param end: the end time of the dispatch
        :type end: pd.Timestamp
        :param product_type: the product type of the unit
        :type product_type: str
        :return: the minimum and maximum power output of the unit
        :rtype: tuple[pd.Series, pd.Series]
        """
        end_excl = end - self.index.freq

        base_load = self.outputs["energy"][start:end_excl]
        heat_demand = self.outputs["heat"][start:end_excl]
        assert heat_demand.min() >= 0

        capacity_neg = self.outputs["capacity_neg"][start:end_excl]
        # needed minimum + capacity_neg - what is already sold is actual minimum
        min_power = self.min_power + capacity_neg - base_load
        # min_power should be at least the heat demand at that time
        min_power = min_power.clip(lower=heat_demand)

        max_power = (
            self.forecaster.get_availability(self.id)[start:end_excl] * self.max_power
        )
        # provide reserve for capacity_pos
        max_power = max_power - self.outputs["capacity_pos"][start:end_excl]
        # remove what has already been bid
        max_power = max_power - base_load
        # make sure that max_power is > 0 for all timesteps
        max_power = max_power.clip(lower=0)

        return min_power, max_power

    def calculate_marginal_cost(self, start: datetime, power: float):
        """
        Calculates the marginal cost of the unit based on the provided start time and power output.
        Returns the marginal cost of the unit.

        :param start: the start time of the dispatch
        :type start: datetime
        :param power: the power output of the unit
        :type power: float
        :return: the marginal cost of the unit
        :rtype: float
        """
        # if marginal costs already exists, return it
        if self.marginal_cost is not None:
            return self.marginal_cost[start]
        # if not, calculate it
        else:
            return self.calc_marginal_cost_with_partial_eff(
                power_output=power,
                timestep=start,
            )

    def as_dict(self) -> dict:
        """
        Returns the attributes of the unit as a dictionary, including specific attributes.

        :return: the attributes of the unit as a dictionary
        :rtype: dict
        """
        unit_dict = super().as_dict()
        unit_dict.update(
            {
                "max_power": self.max_power,
                "min_power": self.min_power,
                "emission_factor": self.emission_factor,
                "efficiency": self.efficiency,
                "unit_type": "power_plant",
            }
        )

        return unit_dict
