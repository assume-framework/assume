import logging
from functools import lru_cache

import pandas as pd

from assume.common.base import BaseUnit, SupportsMinMaxCharge

logger = logging.getLogger(__name__)


class Storage(SupportsMinMaxCharge):
    """A class for a storage unit.

    Attributes
    ----------
    id : str
        The ID of the storage unit.
    technology : str
        The technology of the storage unit.
    node : str
        The node of the storage unit.
    max_power_charge : float
        The maximum power input of the storage unit in MW (negative value).
    min_power_charge : float
        The minimum power input of the storage unit in MW (negative value).
    max_power_discharge : float
        The maximum power output of the storage unit in MW.
    min_power_discharge : float
        The minimum power output of the storage unit in MW.
    max_SOC : float
        The maximum state of charge of the storage unit in MWh (equivalent to capacity).
    min_SOC : float
        The minimum state of charge of the storage unit in MWh.
    efficiency_charge : float
        The efficiency of the storage unit while charging.
    efficiency_discharge : float
        The efficiency of the storage unit while discharging.
    variable_cost_charge : float
        Variable costs to charge the storage unit in €/MW.
    variable_costs_discharge : float
        Variable costs to discharge the storage unit in €/MW.
    emission_factor : float
        The emission factor of the storage unit.
    ramp_up_charge : float, optional
        The ramp up rate of charging the storage unit in MW/15 minutes (negative value).
    ramp_down_charge : float, optional
        The ramp down rate of charging the storage unit in MW/15 minutes (negative value).
    ramp_up_discharge : float, optional
        The ramp up rate of discharging the storage unit in MW/15 minutes.
    ramp_down_discharge : float, optional
        The ramp down rate of discharging the storage unit in MW/15 minutes.
    fixed_cost : float, optional
        The fixed cost of the storage unit in €/MW. (related to capacity?)
    hot_start_cost : float, optional
        The hot start cost of the storage unit in €/MW.
    warm_start_cost : float, optional
        The warm start cost of the storage unit in €/MW.
    cold_start_cost : float, optional
        The cold start cost of the storage unit in €/MW.
    downtime_hot_start : float, optional
        Definition of downtime before hot start in h.
    downtime_warm_start : float
        Definition of downtime before warm start in h.
    min_operating_time : float, optional
        The minimum operating time of the storage unit in hours.
    min_down_time : float, optional
        The minimum down time of the storage unit in hours.
    min_down_time : float, optional
        The minimum down time of the storage unit in hours.
    availability : dict, optional
        The availability of the storage unit in MW for each time step.
    is_active: bool
        Defines whether or not the unit bids itself or is portfolio optimized.
    bidding_startegy: str
        In case the unit is active it has to be defined which bidding strategy should be used
    **kwargs
        Additional keyword arguments.

    Methods
    -------
    reset()
        Reset the storage unit.
    calc_marginal_cost(power_output, partial_load_eff)
        Calculate the marginal cost of the storage unit.
    """

    def __init__(
        self,
        id: str,
        unit_operator: str,
        technology: str,
        bidding_strategies: dict,
        max_power_charge: float | pd.Series,
        max_power_discharge: float | pd.Series,
        max_SOC: float,
        min_power_charge: float | pd.Series = 0.0,
        min_power_discharge: float | pd.Series = 0.0,
        min_SOC: float = 0.0,
        availability: pd.Series = None,
        efficiency_charge: float = 1,
        efficiency_discharge: float = 1,
        variable_cost_charge: float | pd.Series = 0.0,
        variable_cost_discharge: float | pd.Series = 0.0,
        price_forecast: pd.Series = None,
        emission_factor: float = 0.0,
        ramp_up_charge: float = 0.0,
        ramp_down_charge: float = 0.0,
        ramp_up_discharge: float = -1,
        ramp_down_discharge: float = -1,
        fixed_cost: float = 0,
        hot_start_cost: float = 0,
        warm_start_cost: float = 0,
        cold_start_cost: float = 0,
        min_operating_time: float = 0,
        min_down_time: float = 0,
        downtime_hot_start: int = 8,  # hours
        downtime_warm_start: int = 48,  # hours
        index: pd.DatetimeIndex = None,
        location: tuple[float, float] = None,
        node: str = None,
        **kwargs,
    ):
        super().__init__(
            id=id,
            technology=technology,
            node=node,
            bidding_strategies=bidding_strategies,
            index=index,
            unit_operator=unit_operator,
        )

        self.max_power_charge = (
            max_power_charge if max_power_charge <= 0 else -max_power_charge
        )
        self.min_power_charge = (
            min_power_charge if min_power_charge <= 0 else -min_power_charge
        )
        self.max_power_discharge = max_power_discharge
        self.min_power_discharge = min_power_discharge

        self.max_SOC = max_SOC
        self.min_SOC = min_SOC

        self.availability = availability or pd.Series(1, index=self.index)

        self.efficiency_charge = efficiency_charge if 0 < efficiency_charge < 1 else 1
        self.efficiency_discharge = (
            efficiency_discharge if 0 < efficiency_discharge < 1 else 1
        )

        self.variable_cost_charge = variable_cost_charge
        self.variable_cost_discharge = variable_cost_discharge

        self.price_forecast = (
            price_forecast if price_forecast is not None else pd.Series(0, index=index)
        )

        self.emission_factor = emission_factor

        self.ramp_up_charge = (
            self.max_power_charge if ramp_up_charge == 0 else -abs(ramp_up_charge)
        )
        self.ramp_down_charge = (
            self.min_power_charge if ramp_down_charge == 0 else -abs(ramp_down_charge)
        )
        self.ramp_up_discharge = (
            self.max_power_discharge if ramp_up_discharge == -1 else ramp_up_discharge
        )
        self.ramp_down_discharge = (
            self.min_power_discharge
            if ramp_down_discharge == -1
            else ramp_down_discharge
        )

        self.min_operating_time = min_operating_time
        self.min_down_time = min_down_time
        self.downtime_hot_start = downtime_hot_start
        self.warm_start_cost = downtime_warm_start

        self.fixed_cost = fixed_cost

        self.hot_start_cost = hot_start_cost * max_power_discharge
        self.warm_start_cost = warm_start_cost * max_power_discharge
        self.cold_start_cost = cold_start_cost * max_power_discharge

        self.location = location

    def reset(self):
        """Reset the unit to its initial state."""

        # current_status = 0 means the unit is not dispatched
        self.current_status = 1
        self.current_down_time = self.min_down_time

        # outputs["energy"] > 0 discharging, outputs["energy"] < 0 charging
        self.outputs["energy"] = pd.Series(0.0, index=self.index)

        # always starting with discharging?
        # self.outputs["energy"].iat[0] = self.min_power_discharge

        # starting half way charged
        self.current_SOC = self.max_SOC * 0.5

        self.outputs["pos_capacity"] = pd.Series(0.0, index=self.index)
        self.outputs["neg_capacity"] = pd.Series(0.0, index=self.index)

        self.mean_market_success = 0
        self.market_success_list = [0]

    def execute_current_dispatch(self, start: pd.Timestamp, end: pd.Timestamp):
        end_excl = end - self.index.freq

        for t in self.outputs["energy"][start:end_excl].index:
            if self.outputs["energy"][t] > self.max_power_discharge:
                self.outputs["energy"][t] = self.max_power_discharge
                logger.error(
                    f"The energy dispatched is greater the maximum power to discharge, dispatched amount is adjusted."
                )
            elif self.outputs["energy"][t] < self.max_power_charge:
                self.outputs["energy"][t] = self.max_power_charge
                logger.error(
                    f"The energy dispatched is greater than the maximum power to charge, dispatched amount is adjusted."
                )
            elif (
                self.outputs["energy"][t] < self.min_power_discharge
                and self.outputs["energy"][t] > self.min_power_charge
            ):
                self.outputs["energy"][t] = 0
                logger.error(
                    f"The energy dispatched is between min_power_charge and min_power_discharge, no energy is dispatched"
                )

            # discharging
            if self.outputs["energy"][t] > 0:
                if (
                    self.current_SOC
                    - (
                        self.outputs["energy"][t]
                        * pd.Timedelta(self.index.freq).total_seconds()
                        / 3600
                        / self.efficiency_discharge
                    )
                    < self.min_SOC
                ):
                    self.outputs["energy"][t] = (
                        (self.current_SOC - self.min_SOC)
                        * self.efficiency_discharge
                        * 3600
                        / pd.Timedelta(self.index.freq).total_seconds()
                    )
                    logger.error(
                        f"The energy dispatched exceeds the minimum SOC, the dispatched amount is adjusted."
                    )

                self.current_SOC -= (
                    self.outputs["energy"][t]
                    * pd.Timedelta(self.index.freq).total_seconds()
                    / 3600
                    / self.efficiency_discharge
                )

            # charging
            elif self.outputs["energy"][t] < 0:
                if (
                    self.current_SOC
                    - (
                        self.outputs["energy"][t]
                        * pd.Timedelta(self.index.freq).total_seconds()
                        / 3600
                        * self.efficiency_charge
                    )
                    > self.max_SOC
                ):
                    self.outputs["energy"][t] = (
                        (self.current_SOC - self.max_SOC)
                        / self.efficiency_charge
                        * 3600
                        / pd.Timedelta(self.index.freq).total_seconds()
                    )
                    logger.error(
                        f"The energy dispatched exceeds the maximum SOC, the dispatched amount is adjusted."
                    )

                self.current_SOC -= (
                    self.outputs["energy"][t]
                    * pd.Timedelta(self.index.freq).total_seconds()
                    / 3600
                    * self.efficiency_charge
                )

            if self.outputs["energy"][t] == 0:
                self.set_market_failure()
            else:
                self.set_market_sucess()

        return self.outputs["energy"].loc[start:end_excl]

    def set_market_failure(self):
        self.current_status = 0
        self.current_down_time += 1
        if self.market_success_list[-1] != 0:
            self.mean_market_success = sum(self.market_success_list) / len(
                self.market_success_list
            )
            self.market_success_list.append(0)

    def set_market_sucess(self):
        self.market_success_list[-1] += 1
        self.current_status = 1  # discharging or charging
        self.current_down_time = 0

    @lru_cache(maxsize=256)
    def calc_marginal_cost(
        self,
        timestep: pd.Timestamp,
        discharge: bool = True,
    ) -> float:
        if discharge:
            variable_cost = (
                self.variable_cost_discharge.at[timestep]
                if isinstance(self.variable_cost_discharge, pd.Series)
                else self.variable_cost_discharge
            )
            efficiency = self.efficiency_discharge

        else:
            variable_cost = (
                self.variable_cost_charge.at[timestep]
                if isinstance(self.variable_cost_charge, pd.Series)
                else self.variable_cost_charge
            )
            efficiency = self.efficiency_charge

        marginal_cost = variable_cost / efficiency + self.fixed_cost

        return marginal_cost

    def as_dict(self) -> dict:
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

    def calculate_min_max_charge(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> tuple[float]:
        end_excl = end - self.index.freq
        duration = pd.Timedelta(end - start).seconds / 3600

        base_load = self.outputs["energy"][start:end_excl]
        capacity_neg = self.outputs["capacity_neg"][start:end_excl]

        previous_output = self.get_output_before(start)
        current_power_charge = min(previous_output, 0)

        min_power_charge = (
            self.min_power_charge[start:end_excl]
            if isinstance(self.min_power_charge, pd.Series)
            else self.min_power_charge
        )
        min_power_charge -= base_load
        min_power_charge = (min_power_charge).where(min_power_charge <= 0, 0)

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
            min_power_charge > max_power_charge, 0
        )

        # restrict charging according to max_SOC
        max_SOC_charge = min(
            0,
            ((self.current_SOC - self.max_SOC) / self.efficiency_charge / duration),
        )
        max_power_charge = max_power_charge.where(
            max_power_charge > max_SOC_charge, max_SOC_charge
        )
        max_power_charge = round(max_power_charge, 3)
        min_power_charge = round(min_power_charge, 3)

        return min_power_charge, max_power_charge

    def calculate_min_max_discharge(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> tuple[float]:
        end_excl = end - self.index.freq
        duration = pd.Timedelta(end - start).seconds / 3600

        base_load = self.outputs["energy"][start:end_excl]
        capacity_pos = self.outputs["capacity_pos"][start:end_excl]

        previous_output = self.get_output_before(start)
        current_power_discharge = max(previous_output, 0)

        min_power_discharge = (
            self.min_power_discharge[start:end_excl]
            if isinstance(self.min_power_discharge, pd.Series)
            else self.min_power_discharge
        )
        min_power_discharge -= base_load
        min_power_discharge = (min_power_discharge).where(min_power_discharge >= 0, 0)

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

        # restrict according to min_SOC
        max_SOC_discharge = max(
            0,
            ((self.current_SOC - self.min_SOC) * self.efficiency_discharge / duration),
        )
        max_power_discharge = max_power_discharge.where(
            max_power_discharge < max_SOC_discharge, max_SOC_discharge
        )
        max_power_discharge = round(max_power_discharge, 3)
        min_power_discharge = round(min_power_discharge, 3)

        return min_power_discharge, max_power_discharge
