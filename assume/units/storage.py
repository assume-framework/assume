# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import timedelta
from functools import lru_cache

import pandas as pd

from assume.common.base import SupportsMinMaxCharge

logger = logging.getLogger(__name__)
EPS = 1e-4


class Storage(SupportsMinMaxCharge):
    """
    A class for a storage unit.

    Parameters:
        max_power_charge (float): The maximum power input of the storage unit in MW (negative value).
        min_power_charge (float): The minimum power input of the storage unit in MW (negative value).
        max_power_discharge (float): The maximum power output of the storage unit in MW.
        min_power_discharge (float): The minimum power output of the storage unit in MW.
        max_volume (float): The maximum state of charge of the storage unit in MWh (equivalent to capacity).
        min_volume (float): The minimum state of charge of the storage unit in MWh.
        efficiency_charge (float): The efficiency of the storage unit while charging.
        efficiency_discharge (float): The efficiency of the storage unit while discharging.
        variable_cost_charge (float): Variable costs to charge the storage unit in €/MW.
        variable_costs_discharge (float): Variable costs to discharge the storage unit in €/MW.
        emission_factor (float): The emission factor of the storage unit.
        ramp_up_charge (float): The ramp up rate of charging the storage unit in MW/15 minutes (negative value).
        ramp_down_charge (float): The ramp down rate of charging the storage unit in MW/15 minutes (negative value).
        ramp_up_discharge (float): The ramp up rate of discharging the storage unit in MW/15 minutes.
        ramp_down_discharge (float): The ramp down rate of discharging the storage unit in MW/15 minutes.
        fixed_cost (float): The fixed cost of the storage unit in €/MW. (related to capacity?)
        hot_start_cost (float): The hot start cost of the storage unit in €/MW.
        warm_start_cost (float): The warm start cost of the storage unit in €/MW.
        cold_start_cost (float): The cold start cost of the storage unit in €/MW.
        downtime_hot_start (float): Definition of downtime before hot start in h.
        downtime_warm_start (float): Definition of downtime before warm start in h.
        min_operating_time (float): The minimum operating time of the storage unit in hours.
        min_down_time (float): The minimum down time of the storage unit in hours.
        bidding_startegies (str): In case the unit is active it has to be defined which bidding strategy should be used.

    Args:
        id (str): The ID of the storage unit.
        technology (str): The technology of the storage unit.
        node (str): The node of the storage unit.
        max_power_charge (float): The maximum power input of the storage unit in MW (negative value).
        min_power_charge (float): The minimum power input of the storage unit in MW (negative value).
        max_power_discharge (float): The maximum power output of the storage unit in MW.
        min_power_discharge (float): The minimum power output of the storage unit in MW.
        max_volume (float): The maximum state of charge of the storage unit in MWh (equivalent to capacity).
        min_volume (float): The minimum state of charge of the storage unit in MWh.
        efficiency_charge (float): The efficiency of the storage unit while charging.
        efficiency_discharge (float): The efficiency of the storage unit while discharging.
        variable_cost_charge (float): Variable costs to charge the storage unit in €/MW.
        variable_costs_discharge (float): Variable costs to discharge the storage unit in €/MW.
        emission_factor (float): The emission factor of the storage unit.
        ramp_up_charge (float): The ramp up rate of charging the storage unit in MW/15 minutes (negative value).
        ramp_down_charge (float): The ramp down rate of charging the storage unit in MW/15 minutes (negative value).
        ramp_up_discharge (float): The ramp up rate of discharging the storage unit in MW/15 minutes.
        ramp_down_discharge (float): The ramp down rate of discharging the storage unit in MW/15 minutes.
        fixed_cost (float): The fixed cost of the storage unit in €/MW. (related to capacity?)
        hot_start_cost (float): The hot start cost of the storage unit in €/MW.
        warm_start_cost (float): The warm start cost of the storage unit in €/MW.
        cold_start_cost (float): The cold start cost of the storage unit in €/MW.
        downtime_hot_start (float): Definition of downtime before hot start in h.
        downtime_warm_start (float): Definition of downtime before warm start in h.
        min_operating_time (float): The minimum operating time of the storage unit in hours.
        min_down_time (float): The minimum down time of the storage unit in hours.
        is_active (bool): Defines whether or not the unit bids itself or is portfolio optimized.
        bidding_startegy (str): In case the unit is active it has to be defined which bidding strategy should be used

    """

    def __init__(
        self,
        id: str,
        unit_operator: str,
        technology: str,
        bidding_strategies: dict,
        max_power_charge: float | pd.Series,
        max_power_discharge: float | pd.Series,
        max_volume: float,
        min_power_charge: float | pd.Series = 0.0,
        min_power_discharge: float | pd.Series = 0.0,
        min_volume: float = 0.0,
        initial_soc: float = 0.5,
        soc_tick: float = 0.01,
        efficiency_charge: float = 1,
        efficiency_discharge: float = 1,
        variable_cost_charge: float | pd.Series = 0.0,
        variable_cost_discharge: float | pd.Series = 0.0,
        emission_factor: float = 0.0,
        ramp_up_charge: float = None,
        ramp_down_charge: float = None,
        ramp_up_discharge: float = None,
        ramp_down_discharge: float = None,
        fixed_cost: float = 0,
        hot_start_cost: float = 0,
        warm_start_cost: float = 0,
        cold_start_cost: float = 0,
        min_operating_time: float = 0,
        min_down_time: float = 0,
        downtime_hot_start: int = 8,  # hours
        downtime_warm_start: int = 48,  # hours
        index: pd.DatetimeIndex = None,
        location: tuple[float, float] = (0, 0),
        node: str = None,
        **kwargs,
    ):
        super().__init__(
            id=id,
            technology=technology,
            node=node,
            location=location,
            bidding_strategies=bidding_strategies,
            index=index,
            unit_operator=unit_operator,
            **kwargs,
        )

        self.max_power_charge = (
            max_power_charge if max_power_charge <= 0 else -max_power_charge
        )
        self.min_power_charge = (
            min_power_charge if min_power_charge <= 0 else -min_power_charge
        )
        self.max_power_discharge = max_power_discharge
        self.min_power_discharge = min_power_discharge
        self.initial_soc = initial_soc
        self.outputs["soc"] = pd.Series(self.initial_soc, index=self.index, dtype=float)

        self.soc_tick = soc_tick

        self.max_volume = max_volume
        self.min_volume = min_volume

        # The efficiency of the storage unit while charging.
        self.efficiency_charge = efficiency_charge if 0 < efficiency_charge < 1 else 1
        self.efficiency_discharge = (
            efficiency_discharge if 0 < efficiency_discharge < 1 else 1
        )

        # The variable costs to charge/discharge the storage unit.
        self.variable_cost_charge = variable_cost_charge
        self.variable_cost_discharge = variable_cost_discharge
        self.fixed_cost = fixed_cost

        # The emission factor of the storage unit.
        self.emission_factor = emission_factor

        # The ramp up/down rate of charging/discharging the storage unit.
        # if ramp_up_charge == 0, the ramp_up_charge is set to the max_power_charge
        # else the ramp_up_charge is set to the negative value of the ramp_up_charge
        self.ramp_up_charge = (
            self.max_power_charge if ramp_up_charge is None else -abs(ramp_up_charge)
        )
        self.ramp_down_charge = (
            self.min_power_charge
            if ramp_down_charge is None
            else -abs(ramp_down_charge)
        )
        self.ramp_up_discharge = (
            self.max_power_discharge if ramp_up_discharge is None else ramp_up_discharge
        )
        self.ramp_down_discharge = (
            self.min_power_discharge
            if ramp_down_discharge is None
            else ramp_down_discharge
        )

        # How long the storage unit has to be in operation before it can be shut down.
        self.min_operating_time = min_operating_time
        # How long the storage unit has to be shut down before it can be started.
        self.min_down_time = min_down_time
        # The downtime before hot start of the storage unit.
        self.downtime_hot_start = downtime_hot_start
        # The downtime before warm start of the storage unit.
        self.downtime_warm_start = downtime_warm_start

        self.fixed_cost = fixed_cost

        self.hot_start_cost = hot_start_cost * max_power_discharge
        self.warm_start_cost = warm_start_cost * max_power_discharge
        self.cold_start_cost = cold_start_cost * max_power_discharge

    def execute_current_dispatch(self, start: pd.Timestamp, end: pd.Timestamp):
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
        time_delta = self.index.freq / timedelta(hours=1)

        for t in self.outputs["energy"][start:end].index:
            delta_soc = 0
            soc = self.outputs["soc"][t]
            if self.outputs["energy"][t] > self.max_power_discharge:
                self.outputs["energy"][t] = self.max_power_discharge
            elif self.outputs["energy"][t] < self.max_power_charge:
                self.outputs["energy"][t] = self.max_power_charge
            elif (
                self.outputs["energy"][t] < self.min_power_discharge
                and self.outputs["energy"][t] > self.min_power_charge
                and self.outputs["energy"][t] != 0
            ):
                self.outputs["energy"][t] = 0

            # discharging
            if self.outputs["energy"][t] > 0:
                max_soc_discharge = self.calculate_soc_max_discharge(soc)

                if self.outputs["energy"][t] > max_soc_discharge:
                    self.outputs["energy"][t] = max_soc_discharge

                time_delta = self.index.freq / timedelta(hours=1)
                delta_soc = (
                    -self.outputs["energy"][t]
                    * time_delta
                    / self.efficiency_discharge
                    / self.max_volume
                )

            # charging
            elif self.outputs["energy"][t] < 0:
                max_soc_charge = self.calculate_soc_max_charge(soc)

                if self.outputs["energy"][t] < max_soc_charge:
                    self.outputs["energy"][t] = max_soc_charge

                time_delta = self.index.freq / timedelta(hours=1)
                delta_soc = (
                    -self.outputs["energy"][t]
                    * time_delta
                    * self.efficiency_charge
                    / self.max_volume
                )

            self.outputs["soc"][t + self.index.freq :] = soc + delta_soc

        return self.outputs["energy"].loc[start:end]

    @lru_cache(maxsize=256)
    def calculate_marginal_cost(
        self,
        start: pd.Timestamp,
        power: float,
    ) -> float:
        """
        Calculates the marginal cost of the unit based on the provided start time and power output and returns it.
        Returns the marginal cost of the unit.

        Args:
            start (datetime.datetime): The start time of the dispatch.
            power (float): The power output of the unit.

        Returns:
            float: The marginal cost of the unit.
        """

        if power > 0:
            variable_cost = (
                self.variable_cost_discharge.at[start]
                if isinstance(self.variable_cost_discharge, pd.Series)
                else self.variable_cost_discharge
            )
            efficiency = self.efficiency_discharge

        else:
            variable_cost = (
                self.variable_cost_charge.at[start]
                if isinstance(self.variable_cost_charge, pd.Series)
                else self.variable_cost_charge
            )
            efficiency = self.efficiency_charge

        marginal_cost = variable_cost / efficiency

        return marginal_cost

    def calculate_soc_max_discharge(self, soc) -> float:
        """
        Calculates the maximum discharge power depending on the current state of charge.

        Args:
            soc (float): The current state of charge.

        Returns:
            float: The maximum discharge power.
        """
        duration = self.index.freq / timedelta(hours=1)
        power = max(
            0,
            (
                (soc * self.max_volume - self.min_volume)
                * self.efficiency_discharge
                / duration
            ),
        )
        return round(power, 3)

    def calculate_soc_max_charge(
        self,
        soc,
    ) -> float:
        """
        Calculates the maximum charge power depending on the current state of charge.

        Args:
            soc (float): The current state of charge.

        Returns:
            float: The maximum charge power.
        """
        duration = self.index.freq / timedelta(hours=1)
        power = min(
            0,
            (
                (soc * self.max_volume - self.max_volume)
                / self.efficiency_charge
                / duration
            ),
        )
        return round(power, 3)

    def calculate_min_max_charge(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type="energy"
    ) -> tuple[pd.Series]:
        """
        Calculates the min and max charging power for the given time period.

        Args:
            start (pd.Timestamp): The start of the current dispatch.
            end (pd.Timestamp): The end of the current dispatch.
            product_type (str): The product type of the storage unit.

        Returns:
            tuple[pd.Series]: The minimum and maximum charge power levels of the storage unit in MW.
        """
        end_excl = end - self.index.freq

        base_load = self.outputs["energy"][start:end_excl]
        capacity_pos = self.outputs["capacity_pos"][start:end_excl]
        capacity_neg = self.outputs["capacity_neg"][start:end_excl]

        min_power_charge = (
            self.min_power_charge[start:end_excl]
            if isinstance(self.min_power_charge, pd.Series)
            else self.min_power_charge
        )
        min_power_charge -= base_load + capacity_pos
        min_power_charge = min_power_charge.clip(upper=0)

        max_power_charge = (
            self.max_power_charge[start:end_excl]
            if isinstance(self.max_power_charge, pd.Series)
            else self.max_power_charge
        )
        max_power_charge -= base_load + capacity_neg
        max_power_charge = max_power_charge.where(
            max_power_charge <= min_power_charge, 0
        )

        min_power_charge = min_power_charge.where(
            min_power_charge >= max_power_charge, 0
        )

        # restrict charging according to max_volume
        max_soc_charge = self.calculate_soc_max_charge(self.outputs["soc"][start])
        max_power_charge = max_power_charge.clip(lower=max_soc_charge)

        return min_power_charge, max_power_charge

    def calculate_min_max_discharge(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type="energy"
    ) -> tuple[pd.Series]:
        """
        Calculates the min and max discharging power for the given time period.

        Args:
            start (pd.Timestamp): The start of the current dispatch.
            end (pd.Timestamp): The end of the current dispatch.
            product_type (str): The product type of the storage unit.

        Returns:
            tuple[pd.Series]: The minimum and maximum discharge power levels of the storage unit in MW.
        """
        end_excl = end - self.index.freq

        base_load = self.outputs["energy"][start:end_excl]
        capacity_pos = self.outputs["capacity_pos"][start:end_excl]
        capacity_neg = self.outputs["capacity_neg"][start:end_excl]

        min_power_discharge = (
            self.min_power_discharge[start:end_excl]
            if isinstance(self.min_power_discharge, pd.Series)
            else self.min_power_discharge
        )
        min_power_discharge -= base_load + capacity_neg
        min_power_discharge = min_power_discharge.clip(lower=0)

        max_power_discharge = (
            self.max_power_discharge[start:end_excl]
            if isinstance(self.max_power_discharge, pd.Series)
            else self.max_power_discharge
        )
        max_power_discharge -= base_load + capacity_pos
        max_power_discharge = max_power_discharge.where(
            max_power_discharge >= min_power_discharge, 0
        )

        min_power_discharge = min_power_discharge.where(
            min_power_discharge < max_power_discharge, 0
        )

        # restrict according to min_volume
        max_soc_discharge = self.calculate_soc_max_discharge(self.outputs["soc"][start])
        max_power_discharge = max_power_discharge.clip(upper=max_soc_discharge)

        return min_power_discharge, max_power_discharge

    def calculate_ramp_discharge(
        self,
        soc: float,
        previous_power: float,
        power_discharge: float,
        current_power: float = 0,
        min_power_discharge: float = 0,
    ) -> float:
        """
        Adjusts the discharging power to the ramping constraints.

        Args:
            soc (float): The current state of charge.
            previous_power (float): The previous power output of the unit.
            power_discharge (float): The discharging power output of the unit.
            current_power (float, optional): The current power output of the unit. Defaults to 0.
            min_power_discharge (float, optional): The minimum discharging power output of the unit. Defaults to 0.

        Returns:
            float: The discharging power adjusted to the ramping constraints.
        """
        power_discharge = super().calculate_ramp_discharge(
            previous_power,
            power_discharge,
            current_power,
        )
        # restrict according to min_SOC

        max_soc_discharge = self.calculate_soc_max_discharge(soc)
        power_discharge = min(power_discharge, max_soc_discharge)
        if power_discharge < min_power_discharge:
            power_discharge = 0

        return power_discharge

    def calculate_ramp_charge(
        self,
        soc: float,
        previous_power: float,
        power_charge: float,
        current_power: float = 0,
        min_power_charge: float = 0,
    ) -> float:
        """
        Adjusts the charging power to the ramping constraints.

        Args:
            soc (float): The current state of charge.
            previous_power (float): The previous power output of the unit.
            power_charge (float): The charging power output of the unit.
            current_power (float, optional): The current power output of the unit. Defaults to 0.
            min_power_charge (float, optional): The minimum charging power output of the unit. Defaults to 0.

        Returns:
            float: The charging power adjusted to the ramping constraints.
        """
        power_charge = super().calculate_ramp_charge(
            previous_power,
            power_charge,
            current_power,
        )

        # restrict charging according to max_SOC
        max_soc_charge = self.calculate_soc_max_charge(soc)

        power_charge = max(power_charge, max_soc_charge)
        if power_charge > min_power_charge:
            power_charge = 0

        return power_charge

    def get_starting_costs(self, op_time):
        """
        Calculates the starting costs of the unit depending on how long it was shut down

        Args:
            op_time (float): The time the unit was shut down in hours.

        Returns:
            float: The starting costs of the unit.
        """
        if op_time > 0:
            # unit is running
            return 0
        if -op_time < self.downtime_hot_start:
            return self.hot_start_cost
        elif -op_time < self.downtime_warm_start:
            return self.warm_start_cost
        else:
            return self.cold_start_cost

    def as_dict(self) -> dict:
        """
        Return the storage unit's attributes as a dictionary, including specific attributes.

        Returns:
            dict: The storage unit's attributes as a dictionary.
        """
        unit_dict = super().as_dict()
        unit_dict.update(
            {
                "max_power_charge": self.max_power_charge,
                "max_power_discharge": self.max_power_discharge,
                "min_power_charge": self.min_power_charge,
                "min_power_discharge": self.min_power_discharge,
                "efficiency_charge": self.efficiency_discharge,
                "efficiency_discharge": self.efficiency_charge,
                "unit_type": "storage",
            }
        )

        return unit_dict
