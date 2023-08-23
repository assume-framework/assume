import logging
import math
from functools import lru_cache

import pandas as pd

from assume.common.base import SupportsMinMaxCharge

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
    max_volume : float
        The maximum state of charge of the storage unit in MWh (equivalent to capacity).
    min_volume : float
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
        self.soc_tick = soc_tick

        self.max_volume = max_volume
        self.min_volume = min_volume

        self.efficiency_charge = efficiency_charge if 0 < efficiency_charge < 1 else 1
        self.efficiency_discharge = (
            efficiency_discharge if 0 < efficiency_discharge < 1 else 1
        )

        self.variable_cost_charge = variable_cost_charge
        self.variable_cost_discharge = variable_cost_discharge

        self.emission_factor = emission_factor

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
        self.current_SOC = self.initial_soc

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
                and self.outputs["energy"][t] != 0
            ):
                self.outputs["energy"][t] = 0
                logger.error(
                    f"The energy dispatched is between min_power_charge and min_power_discharge, no energy is dispatched"
                )

            # discharging
            if self.outputs["energy"][t] > 0:
                max_soc_discharge = self.calculate_soc_max_discharge(self.current_SOC)

                if self.outputs["energy"][t] > max_soc_discharge:
                    if not math.isclose(self.outputs["energy"][t], max_soc_discharge):
                        logger.error(
                            f"The energy dispatched exceeds the minimum SOC significantly, the dispatched amount is adjusted."
                        )
                    self.outputs["energy"][t] = max_soc_discharge

                delta_soc = (
                    self.outputs["energy"][t]
                    * pd.Timedelta(self.index.freq).total_seconds()
                    / 3600
                    / self.efficiency_discharge
                    / self.max_volume
                )

            # charging
            elif self.outputs["energy"][t] < 0:
                max_soc_charge = self.calculate_soc_max_charge(self.current_SOC)

                if self.outputs["energy"][t] < max_soc_charge:
                    if not math.isclose(self.outputs["energy"][t], max_soc_charge):
                        logger.error(
                            f"The energy dispatched exceeds the maximum SOC, the dispatched amount is adjusted."
                        )
                    self.outputs["energy"][t] = max_soc_charge

                delta_soc = (
                    self.outputs["energy"][t]
                    * pd.Timedelta(self.index.freq).total_seconds()
                    / 3600
                    * self.efficiency_charge
                    / self.max_volume
                )

            if self.outputs["energy"][t] == 0:
                self.set_market_failure()
            else:
                self.set_market_sucess(delta_soc)

        return self.outputs["energy"].loc[start:end_excl]

    def set_market_failure(self):
        self.current_status = 0
        self.current_down_time += 1
        if self.market_success_list[-1] != 0:
            self.mean_market_success = sum(self.market_success_list) / len(
                self.market_success_list
            )
            self.market_success_list.append(0)

    def set_market_sucess(self, delta_soc):
        self.current_SOC -= round(delta_soc / self.soc_tick) * self.soc_tick
        self.market_success_list[-1] += 1
        self.current_status = 1  # discharging or charging
        self.current_down_time = 0

    @lru_cache(maxsize=256)
    def calculate_marginal_cost(
        self,
        start: pd.Timestamp,
        power: float,
    ) -> float:
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

    def calculate_soc_max_discharge(self, soc) -> float:
        duration = pd.Timedelta(self.index.freq).total_seconds() / 3600
        power = max(
            0,
            (
                (soc * self.max_volume - self.min_volume)
                * self.efficiency_discharge
                / duration
            ),
        )
        return round(power, 1)

    def calculate_soc_max_charge(
        self,
        soc,
    ) -> float:
        duration = pd.Timedelta(self.index.freq).seconds / 3600
        power = min(
            0,
            (
                (soc * self.max_volume - self.max_volume)
                / self.efficiency_charge
                / duration
            ),
        )
        return round(power, 1)

    def calculate_min_max_charge(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type="energy"
    ) -> tuple[pd.Series]:
        end_excl = end - self.index.freq
        duration = pd.Timedelta(self.index.freq).seconds / 3600

        base_load = self.outputs["energy"][start:end_excl]
        capacity_pos = self.outputs["capacity_pos"][start:end_excl]
        capacity_neg = self.outputs["capacity_neg"][start:end_excl]

        min_power_charge = (
            self.min_power_charge[start:end_excl]
            if isinstance(self.min_power_charge, pd.Series)
            else self.min_power_charge
        )
        min_power_charge -= base_load + capacity_pos
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
            min_power_charge >= max_power_charge, 0
        )

        # restrict charging according to max_volume
        max_soc_charge = self.calculate_soc_max_charge(self.current_SOC)
        max_power_charge = max_power_charge.where(
            max_power_charge > max_soc_charge, max_soc_charge
        )

        return min_power_charge, max_power_charge

    def calculate_min_max_discharge(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type="energy"
    ) -> tuple[pd.Series]:
        end_excl = end - self.index.freq
        duration = pd.Timedelta(self.index.freq).seconds / 3600

        base_load = self.outputs["energy"][start:end_excl]
        capacity_pos = self.outputs["capacity_pos"][start:end_excl]
        capacity_neg = self.outputs["capacity_neg"][start:end_excl]

        min_power_discharge = (
            self.min_power_discharge[start:end_excl]
            if isinstance(self.min_power_discharge, pd.Series)
            else self.min_power_discharge
        )
        min_power_discharge -= base_load + capacity_neg
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

        # restrict according to min_volume
        max_soc_discharge = self.calculate_soc_max_discharge(self.current_SOC)
        max_power_discharge = max_power_discharge.where(
            max_power_discharge < max_soc_discharge, max_soc_discharge
        )

        return min_power_discharge, max_power_discharge

    def calculate_ramp_discharge(
        self,
        previous_power: float,
        power_discharge: float,
        current_power: float = 0,
        min_power_discharge: float = 0,
        SOC: float | None = None,
    ) -> float:
        if power_discharge == 0:
            return power_discharge
        if SOC is None:
            SOC = self.current_SOC

        # assuming ramping down discharge restricts ramp up of charging if storage was discharging before and ramp_down_discharge is defined
        if previous_power < 0 and self.ramp_down_charge != None:
            power_discharge = max(
                previous_power - self.ramp_down_charge - current_power, 0
            )
        else:
            if previous_power < 0:
                previous_power = 0

            power_discharge = min(
                power_discharge,
                max(0, previous_power + self.ramp_up_discharge - current_power),
            )
            # restrict only if ramping defined
            if self.ramp_down_discharge != None:
                power_discharge = max(
                    power_discharge,
                    previous_power - self.ramp_down_discharge - current_power,
                    0,
                )

        # restrict according to min_SOC

        max_soc_discharge = self.calculate_soc_max_discharge(SOC)
        if power_discharge > max_soc_discharge:
            power_discharge = max_soc_discharge
        if power_discharge < min_power_discharge:
            power_discharge = 0

        return power_discharge

    def calculate_ramp_charge(
        self,
        previous_power: float,
        power_charge: float,
        current_power: float = 0,
        min_power_charge: float = 0,
        SOC: float | None = None,
    ) -> float:
        if power_charge == 0:
            return power_charge
        if SOC is None:
            SOC = self.current_SOC

        # assuming ramping down discharge restricts ramp up of charge if storage was discharging before and ramp_down_discharge is defined
        if previous_power > 0 and self.ramp_down_discharge != None:
            power_charge = min(
                previous_power - self.ramp_down_discharge - current_power, 0
            )
        else:
            if previous_power > 0:
                previous_power = 0

            power_charge = max(
                power_charge,
                min(previous_power + self.ramp_up_charge - current_power, 0),
            )
            # restrict only if ramping defined
            if self.ramp_down_charge != None:
                power_charge = min(
                    power_charge,
                    previous_power - self.ramp_down_charge - current_power,
                    0,
                )

        # restrict charging according to max_SOC
        max_soc_charge = self.calculate_soc_max_charge(SOC)

        if power_charge < max_soc_charge:
            power_charge = max_soc_charge
        if power_charge > min_power_charge:
            power_charge = 0

        return power_charge
