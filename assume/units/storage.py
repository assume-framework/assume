import logging
from functools import lru_cache

import pandas as pd

from assume.units.base_unit import BaseUnit

logger = logging.getLogger(__name__)


class Storage(BaseUnit):
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
    calculate_operational_window()
        Calculate the operation window for the next time step.
    calc_marginal_cost(power_output, partial_load_eff)
        Calculate the marginal cost of the storage unit.
    """

    def __init__(
        self,
        id: str,
        unit_operator: str,
        technology: str,
        bidding_strategies: dict,
        max_power_charge: float or pd.Series,
        max_power_discharge: float or pd.Series,
        max_SOC: float,
        min_power_charge: float or pd.Series = 0.0,
        min_power_discharge: float or pd.Series = 0.0,
        min_SOC: float = 0.0,
        efficiency_charge: float = 1,
        efficiency_discharge: float = 1,
        variable_cost_charge: float or pd.Series = 0.0,
        variable_cost_discharge: float or pd.Series = 0.0,
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
        **kwargs
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

        self.ramp_up_charge = ramp_up_charge if ramp_up_charge <= 0 else -ramp_up_charge
        self.ramp_down_charge = (
            ramp_down_charge if ramp_down_charge <= 0 else -ramp_up_charge
        )
        self.ramp_up_discharge = ramp_up_discharge
        self.ramp_down_discharge = ramp_down_discharge

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
        duration = (end - start).total_seconds() / 3600

        if self.current_status == 0 and self.current_down_time < self.min_down_time:
            return None

        current_power_discharge = max(
            self.outputs["energy"].at[start - self.index.freq], 0
        )

        current_power_charge = min(
            self.outputs["energy"].at[start - self.index.freq], 0
        )

        min_SOC = (
            self.min_SOC[start] if type(self.min_SOC) is pd.Series else self.min_SOC
        )
        max_SOC = (
            self.max_SOC[start] if type(self.max_SOC) is pd.Series else self.max_SOC
        )

        min_max_power = self.calculate_min_max_power(
            start=start,
            current_power_charge=current_power_charge,
            current_power_discharge=current_power_discharge,
            min_SOC=min_SOC,
            max_SOC=max_SOC,
            duration=duration,
        )

        # what form does the operational window have?
        operational_window = {
            "window": {"start": start, "end": end},
            "current_power_discharge": {
                "power_discharge": current_power_discharge,
                "marginal_cost": self.calc_marginal_cost(
                    timestep=start,
                    discharge=True,
                ),
            },
            "current_power_charge": {
                "power_charge": current_power_charge,
                "marginal_cost": self.calc_marginal_cost(
                    timestep=start,
                    discharge=False,
                ),
            },
            "min_power_discharge": {
                "power_discharge": min_max_power["min_power_discharge"],
                "marginal_cost": self.calc_marginal_cost(
                    timestep=start,
                    discharge=True,
                ),
            },
            "max_power_discharge": {
                "power_discharge": min_max_power["max_power_discharge"],
                "marginal_cost": self.calc_marginal_cost(
                    timestep=start,
                    discharge=True,
                ),
            },
            "min_power_charge": {
                "power_charge": min_max_power["min_power_charge"],
                "marginal_cost": self.calc_marginal_cost(
                    timestep=start,
                    discharge=False,
                ),
            },
            "max_power_charge": {
                "power_charge": min_max_power["max_power_charge"],
                "marginal_cost": self.calc_marginal_cost(
                    timestep=start,
                    discharge=False,
                ),
            },
        }
        return operational_window

    def calculate_reserve_operational_window(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> dict:
        duration = (end - start).total_seconds() / 3600
        # capacity calculation has to be added
        current_power = self.outputs["energy"].at[start - self.index.freq]

        available_pos_reserve_discharge = None
        available_neg_reserve_discharge = None
        available_pos_reserve_charge = None
        available_neg_reserve_charge = None

        operational_window = {
            "window": {"start": start, "end": end},
            "pos_reserve_discharge": {
                "capacity": available_pos_reserve_discharge,
            },
            "neg_reserve_discharge": {
                "capacity": available_neg_reserve_discharge,
            },
            "pos_reserve_charge": {
                "capacity": available_pos_reserve_charge,
            },
            "neg_reserve_charge": {
                "capacity": available_neg_reserve_charge,
            },
        }

        if available_neg_reserve_discharge < 0:
            logger.error("available_neg_reserve_discharge < 0")

        return operational_window

    def calculate_bids(
        self,
        market_config,
        product_tuple,
    ):
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
        end_excl = end - self.index.freq

        # TODO checks should be at execute_current_dispatch - see powerplant
        # TODO check if resulting power is < max_power
        # if self.outputs["energy"][start:end].max() > self.max_power:
        #     max_pow = self.outputs["energy"][start:end].max()
        #     logger.error(f"{max_pow} greater than {self.max_power} - bidding twice?")

        if product_type == "energy":
            self.set_energy_dispatch_plan(
                dispatch_plan=dispatch_plan,
                start=start,
                end=end,
                product_type=product_type,
            )
        elif product_type in {"capacity_pos", "capacity_neg"}:
            self.outputs[product_type].loc[start:end_excl] += dispatch_plan[
                "total_power"
            ]

    def set_energy_dispatch_plan(
        self,
        dispatch_plan: dict,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ):
        end_excl = end - self.index.freq
        duration = (end - start).total_seconds() / 3600

        if self.outputs["energy"][start:end_excl].min() >= self.min_power_discharge:
            self.set_market_sucess(dispatch_plan, start, end_excl)
            discharged_energy = (
                dispatch_plan["total_power"] * duration / self.efficiency_discharge
            )
            self.current_SOC = self.current_SOC - discharged_energy

        elif self.outputs["energy"][start:end_excl].max() <= -self.min_power_charge:
            self.set_market_sucess(dispatch_plan, start, end_excl)
            # multiplied by -1 because the power when charging is negative
            charged_energy = (
                dispatch_plan["total_power"] * duration * self.efficiency_charge * (-1)
            )
            self.current_SOC = self.current_SOC + charged_energy

        else:
            self.current_status = 0
            self.current_down_time += 1
            self.outputs["energy"].loc[start:end_excl] = 0

            if self.market_success_list[-1] != 0:
                self.mean_market_success = sum(self.market_success_list) / len(
                    self.market_success_list
                )
                self.market_success_list.append(0)

    def set_market_sucess(self, dispatch_plan, start, end):
        self.market_success_list[-1] += 1
        self.current_status = 1  # discharging
        self.current_down_time = 0
        self.outputs["energy"].loc[start:end] += dispatch_plan["total_power"]

    @lru_cache(maxsize=256)
    def calc_marginal_cost(
        self,
        timestep: pd.Timestamp,
        discharge: bool = True,
    ) -> float:
        if discharge:
            variable_cost = (
                self.variable_cost_discharge.at[timestep]
                if type(self.variable_cost_discharge) is pd.Series
                else self.variable_cost_discharge
            )
            efficiency = self.efficiency_discharge

        else:
            variable_cost = (
                self.variable_cost_charge.at[timestep]
                if type(self.variable_cost_charge) is pd.Series
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
                "unit_type": "storage",
            }
        )

        return unit_dict

    def calculate_min_max_power(
        self,
        start,
        current_power_charge,
        current_power_discharge,
        min_SOC,
        max_SOC,
        duration,
    ) -> tuple:
        min_power_discharge = (
            self.min_power_discharge[start]
            if type(self.min_power_discharge) is pd.Series
            else self.min_power_discharge
        )
        min_power_charge = (
            self.min_power_charge[start]
            if type(self.min_power_charge) is pd.Series
            else self.min_power_charge
        )

        max_power_discharge = (
            self.max_power_discharge[start]
            if type(self.max_power_discharge) is pd.Series
            else self.max_power_discharge
        )
        max_power_charge = (
            self.max_power_charge[start]
            if type(self.max_power_charge) is pd.Series
            else self.max_power_charge
        )

        # was charging before
        if self.min_down_time > 0 and self.outputs["energy"] < 0:
            min_power_discharge = 0
            max_power_discharge = 0
        else:
            if self.ramp_down_discharge != -1:
                min_power_discharge = max(
                    current_power_discharge - self.ramp_down_discharge,
                    min_power_discharge,
                )

            if self.ramp_up_discharge != -1:
                max_power_discharge = min(
                    self.ramp_up_discharge + current_power_discharge,
                    max_power_discharge,
                )

        # was discharging before
        if self.min_down_time > 0 and self.outputs["energy"] > 0:
            min_power_charge = 0
            max_power_charge = 0
        else:
            if self.ramp_down_charge < 0:
                min_power_charge = max(
                    current_power_charge - self.ramp_down_charge, min_power_charge
                )

            if self.ramp_up_charge < 0:
                max_power_charge = max(
                    current_power_charge + self.ramp_up_charge, max_power_charge
                )

        # restrict according to min_SOC
        max_power_discharge = min(
            max_power_discharge,
            max(
                0,
                (
                    (self.current_SOC - min_SOC - self.outputs["pos_capacity"][start])
                    * self.efficiency_discharge
                    / duration
                ),
            ),
        )

        # restrict charging according to max_SOC
        max_power_charge = max(
            max_power_charge,
            min(
                0,
                (
                    (self.current_SOC - max_SOC - self.outputs["neg_capacity"][start])
                    / self.efficiency_charge
                    / duration
                ),
            ),
        )
        # pack values to a dict
        min_max_power = {
            "min_power_discharge": min_power_discharge,
            "min_power_charge": min_power_charge,
            "max_power_discharge": max_power_discharge,
            "max_power_charge": max_power_charge,
        }

        # if values are close to zero, set them to zero
        for key, value in min_max_power.items():
            if abs(value) < 1e-3:
                min_max_power[key] = 0

        return min_max_power
