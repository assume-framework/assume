# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime, timedelta
from functools import lru_cache

import pandas as pd

from assume.common.base import SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook
from assume.common.utils import get_products_index

logger = logging.getLogger(__name__)


class PowerPlant(SupportsMinMax):
    """
    A class for a powerplant unit.

    Parameters:
        id (str): The ID of the unit.
        unit_operator (str): The operator of the unit.
        technology (str): The technology of the unit.
        bidding_strategies (dict): The bidding strategies of the unit.
        index (pd.DatetimeIndex): The index of the unit.
        max_power (float): The maximum power output capacity of the power plant in MW.
        min_power (float, optional): The minimum power output capacity of the power plant in MW. Defaults to 0.0 MW.
        efficiency (float, optional): The efficiency of the poewr plant in converting fuel to electricity (Defaults to 1.0). Defaults to 1.0.
        fixed_cost (float, optional): The fixed operating cost of the power plant, independent of the power output (Defaults to 0.0 monetary units). Defaults to 0.0.
        variable_cost (float | pd.Series, optional): The variable operating cost of the power plant, dependent on the power output (Defaults to 0.0 monetary units). Defaults to 0.0.
        partial_load_eff (bool, optional): Does the efficiency varies at part loads? (Defaults to False). Defaults to False.
        fuel_type (str, optional): The type of fuel used by the power plant for power generation (Defaults to "others"). Defaults to "others".
        emission_factor (float, optional): The emission factor associated with the power plants fuel type -> CO2 emissions per unit of energy produced (Defaults to 0.0.). Defaults to 0.0.
        ramp_up (float | None, optional): The ramp-up rate of the power plant, indicating how quickly it can increase power output (Defaults to -1). Defaults to None.
        ramp_down (float | None, optional): The ramp-down rate of the power plant, indicating how quickly it can decrease power output. (Defaults to -1). Defaults to None.
        hot_start_cost (float, optional): The cost of a hot start, where the power plant is restarted after a recent shutdown.(Defaults to 0 monetary units.). Defaults to 0.
        warm_start_cost (float, optional): The cost of a warm start, where the power plant is restarted after a moderate downtime.(Defaults to 0 monetary units.). Defaults to 0.
        cold_start_cost (float, optional): The cost of a cold start, where the power plant is restarted after a prolonged downtime.(Defaults to 0 monetary units.). Defaults to 0.
        min_operating_time (float, optional): The minimum duration that the power plant must operate once started, in hours.(Defaults to 0 hours.). Defaults to 0.
        min_down_time (float, optional): The minimum downtime required after a shutdown before the power plant can be restarted, in hours.(Defaults to 0 hours.). Defaults to 0.
        downtime_hot_start (int, optional): The downtime required after a hot start before the power plant can be restarted, in hours.(Defaults to 8 hours.). Defaults to 8.
        downtime_warm_start (int, optional): The downtime required after a warm start before the power plant can be restarted, in hours.( Defaults to 48 hours.). Defaults to 48.
        heat_extraction (bool, optional): A boolean indicating whether the power plant can extract heat for external purposes.(Defaults to False.). Defaults to False.
        max_heat_extraction (float, optional): The maximum amount of heat that the power plant can extract for external use, in some suitable unit.(Defaults to 0.). Defaults to 0.

    Args:
        id (str): The ID of the storage unit.
        unit_operator (str): The operator of the unit.
        technology (str): The technology of the unit.
        bidding_strategies (dict): The bidding strategies of the unit.
        index (pd.DatetimeIndex): The index of the unit.
        max_power (float): The maximum power output capacity of the power plant in MW.
        min_power (float, optional): The minimum power output capacity of the power plant in MW. Defaults to 0.0 MW.
        efficiency (float, optional): The efficiency of the poewr plant in converting fuel to electricity (Defaults to 1.0). Defaults to 1.0.
        fixed_cost (float, optional): The fixed operating cost of the power plant, independent of the power output (Defaults to 0.0 monetary units). Defaults to 0.0.
        variable_cost (float | pd.Series, optional): The variable operating cost of the power plant, dependent on the power output (Defaults to 0.0 monetary units). Defaults to 0.0.
        partial_load_eff (bool, optional): Does the efficiency varies at part loads? (Defaults to False). Defaults to False.
        fuel_type (str, optional): The type of fuel used by the power plant for power generation (Defaults to "others"). Defaults to "others".
        emission_factor (float, optional): The emission factor associated with the power plants fuel type -> CO2 emissions per unit of energy produced (Defaults to 0.0.). Defaults to 0.0.
        ramp_up (float | None, optional): The ramp-up rate of the power plant, indicating how quickly it can increase power output (Defaults to -1). Defaults to None.
        ramp_down (float | None, optional): The ramp-down rate of the power plant, indicating how quickly it can decrease power output. (Defaults to -1). Defaults to None.
        hot_start_cost (float, optional): The cost of a hot start, where the power plant is restarted after a recent shutdown.(Defaults to 0 monetary units.). Defaults to 0.
        warm_start_cost (float, optional): The cost of a warm start, where the power plant is restarted after a moderate downtime.(Defaults to 0 monetary units.). Defaults to 0.
        cold_start_cost (float, optional): The cost of a cold start, where the power plant is restarted after a prolonged downtime.(Defaults to 0 monetary units.). Defaults to 0.
        min_operating_time (float, optional): The minimum duration that the power plant must operate once started, in hours.(Defaults to 0 hours.). Defaults to 0.
        min_down_time (float, optional): The minimum downtime required after a shutdown before the power plant can be restarted, in hours.(Defaults to 0 hours.). Defaults to 0.
        downtime_hot_start (int, optional): The downtime required after a hot start before the power plant can be restarted, in hours.(Defaults to 8 hours.). Defaults to 8.
        downtime_warm_start (int, optional): The downtime required after a warm start before the power plant can be restarted, in hours.( Defaults to 48 hours.). Defaults to 48.
        heat_extraction (bool, optional): A boolean indicating whether the power plant can extract heat for external purposes.(Defaults to False.). Defaults to False.
        max_heat_extraction (float, optional): The maximum amount of heat that the power plant can extract for external use, in some suitable unit.(Defaults to 0.). Defaults to 0.
        location (tuple[float, float], optional): The geographical coordinates (latitude and longitude) of the power plant's location.(Defaults to (0.0, 0.0).). Defaults to (0.0, 0.0).
        node (str, optional): The identifier of the electrical bus or network node to which the power plant is connected.(Defaults to "bus0".). Defaults to "bus0".
        **kwargs (dict, optional): Additional keyword arguments to be passed to the base class. Defaults to {}.

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
        variable_cost: float | pd.Series = 0.0,
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
            location=location,
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
        self.variable_cost = variable_cost
        self.hot_start_cost = hot_start_cost * max_power
        self.warm_start_cost = warm_start_cost * max_power
        self.cold_start_cost = cold_start_cost * max_power

        self.heat_extraction = heat_extraction
        self.max_heat_extraction = max_heat_extraction

        self.init_marginal_cost()

    def init_marginal_cost(self):
        """
        Initializes the marginal cost of the unit using calc_cimple_marginal_cost().

        Args:
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

        Args:
            start (pd.Timestamp): The start time of the dispatch.
            end (pd.Timestamp): The end time of the dispatch.

        Returns:
            pd.Series: The volume of the unit within the given time range.
        """
        start = max(start, self.index[0])

        max_power = (
            self.forecaster.get_availability(self.id)[start:end] * self.max_power
        )

        for t in self.outputs["energy"][start:end].index:
            current_power = self.outputs["energy"][t]

            previous_power = self.get_output_before(t)
            op_time = self.get_operation_time(t)

            current_power = self.calculate_ramp(op_time, previous_power, current_power)

            if current_power > 0:
                current_power = min(current_power, max_power[t])
                current_power = max(current_power, self.min_power)

            self.outputs["energy"][t] = current_power

        return self.outputs["energy"].loc[start:end]

    def set_dispatch_plan(
        self,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ) -> None:
        """
        Adds the dispatch plan from the current market result to the total dispatch plan and claculates the cshflow.

        Args:
            marketconfig (MarketConfig): The market configuration.
            orderbook (Orderbook): The orderbook.
        """
        products_index = get_products_index(orderbook)

        max_power = (
            self.forecaster.get_availability(self.id)[products_index] * self.max_power
        )

        product_type = marketconfig.product_type
        for order in orderbook:
            start = order["start_time"]
            end = order["end_time"]
            end_excl = end - self.index.freq
            if isinstance(order["accepted_volume"], dict):
                self.outputs[product_type].loc[start:end_excl] += [
                    order["accepted_volume"][key]
                    for key in order["accepted_volume"].keys()
                ]
            else:
                self.outputs[product_type].loc[start:end_excl] += order[
                    "accepted_volume"
                ]

        self.calculate_cashflow(product_type, orderbook)

        for start in products_index:
            current_power = self.outputs[product_type][start]

            previous_power = self.get_output_before(start)
            op_time = self.get_operation_time(start)

            current_power = self.calculate_ramp(op_time, previous_power, current_power)

            if current_power > 0:
                current_power = min(current_power, max_power[start])
                current_power = max(current_power, self.min_power)

            self.outputs[product_type][start] = current_power

        self.bidding_strategies[product_type].calculate_reward(
            unit=self,
            marketconfig=marketconfig,
            orderbook=orderbook,
        )

    def calc_simple_marginal_cost(
        self,
    ):
        """
        Calculates the marginal cost of the unit (simple method) and returns the marginal cost of the unit.

        Returns:
            float: The marginal cost of the unit.
        """
        fuel_price = self.forecaster.get_price(self.fuel_type)
        marginal_cost = (
            fuel_price / self.efficiency
            + self.forecaster.get_price("co2") * self.emission_factor / self.efficiency
            + self.variable_cost
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

        Args:
            power_output (float): The power output of the unit.
            timestep (pd.Timestamp, optional): The timestamp of the unit. Defaults to None.

        Returns:
            float | pd.Series: The marginal cost of the unit.
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

        variable_cost = (
            self.variable_cost
            if isinstance(self.variable_cost, float)
            else self.variable_cost[timestep]
        )

        marginal_cost = (
            fuel_price / efficiency
            + co2_price * self.emission_factor / efficiency
            + variable_cost
        )

        return marginal_cost

    def calculate_min_max_power(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type="energy"
    ) -> tuple[pd.Series, pd.Series]:
        """
        Calculates the minimum and maximum power output of the unit and returns it.

        Args:
            start (pd.Timestamp): The start time of the dispatch.
            end (pd.Timestamp): The end time of the dispatch.
            product_type (str, optional): The product type of the unit. Defaults to "energy".

        Returns:
            tuple[pd.Series, pd.Series]: The minimum and maximum power output of the unit.

        Note:
            The calculation does not include ramping constraints and can be used for arbitrary start times in the future.
        """
        end_excl = end - self.index.freq

        base_load = self.outputs["energy"][start:end_excl]
        heat_demand = self.outputs["heat"][start:end_excl]
        # assert heat_demand.min() >= 0

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
        Calculates the marginal cost of the unit based on the provided start time and power output and returns it.
        Returns the marginal cost of the unit.

        Args:
            start (datetime.datetime): The start time of the dispatch.
            power (float): The power output of the unit.

        Returns:
            float: The marginal cost of the unit.
        """
        # if marginal costs already exists, return it
        if self.marginal_cost is not None:
            return (
                self.marginal_cost[start]
                if len(self.marginal_cost) > 1
                else self.marginal_cost
            )
        # if not, calculate it
        else:
            return self.calc_marginal_cost_with_partial_eff(
                power_output=power,
                timestep=start,
            )

    def as_dict(self) -> dict:
        """
        Returns the attributes of the unit as a dictionary, including specific attributes.

        Returns:
            dict: The attributes of the unit as a dictionary.
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
