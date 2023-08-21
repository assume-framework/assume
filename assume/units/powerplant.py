import logging
from datetime import datetime, timedelta
from functools import lru_cache

import pandas as pd

from assume.common.base import SupportsMinMax
from assume.common.market_objects import Orderbook

logger = logging.getLogger(__name__)


class PowerPlant(SupportsMinMax):
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
        ramp_up: float = None,
        ramp_down: float = None,
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
        self.ramp_down = max_power if ramp_down is None else ramp_down
        self.ramp_up = max_power if ramp_up is None else ramp_up
        self.min_operating_time = min_operating_time if min_operating_time > 0 else 1
        self.min_down_time = min_down_time if min_down_time > 0 else 1
        self.downtime_hot_start = (
            downtime_hot_start / self.index.freq.delta.total_seconds() / 3600
        )
        self.downtime_warm_start = downtime_warm_start

        self.fixed_cost = fixed_cost
        self.hot_start_cost = hot_start_cost * max_power
        self.warm_start_cost = warm_start_cost * max_power
        self.cold_start_cost = cold_start_cost * max_power

        self.heat_extraction = heat_extraction
        self.max_heat_extraction = max_heat_extraction

        self.location = location

        self.init_marginal_cost()

    def init_marginal_cost(self):
        self.marginal_cost = self.calc_simple_marginal_cost()

    def reset(self):
        """Reset the unit to its initial state."""

        self.current_status = 1
        self.current_down_time = self.min_down_time

        self.outputs["energy"] = pd.Series(0.0, index=self.index)
        # workaround if market schedules do not match
        # for example neg_reserve is required but market did not bid yet
        # it does set a usage in times where no power is used by the market
        # self.outputs["energy"].loc[:] = self.min_power + 0.5 * (
        #    self.max_power - self.min_power
        # )

        self.outputs["heat"] = pd.Series(0.0, index=self.index)
        self.outputs["power_loss"] = pd.Series(0.0, index=self.index)

        self.outputs["capacity_pos"] = pd.Series(0.0, index=self.index)
        self.outputs["capacity_neg"] = pd.Series(0.0, index=self.index)

        self.outputs["profits"] = pd.Series(0.0, index=self.index)
        self.outputs["rewards"] = pd.Series(0.0, index=self.index)
        self.outputs["regrets"] = pd.Series(0.0, index=self.index)

        self.mean_market_success = 0
        self.market_success_list = [0]

    def execute_current_dispatch(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ):
        end_excl = end - self.index.freq
        # TODO ramp down and turn off only for relevant timesteps
        if self.outputs["energy"][start:end_excl].mean() < self.min_power:
            self.outputs["energy"].loc[start:end_excl] = 0
            self.current_status = 0
            self.current_down_time += 1
            if self.market_success_list[-1] != 0:
                self.mean_market_success = sum(self.market_success_list) / len(
                    self.market_success_list
                )
                self.market_success_list.append(0)
        else:
            self.market_success_list[-1] += 1
            self.current_status = 1
            self.current_down_time = 0

        # TODO check if resulting power is < max_power
        # if self.outputs["energy"][start:end_excl].max() > self.max_power:
        #     max_pow = self.outputs["energy"][start:end_excl].max()
        #     logger.error(f"{max_pow} greater than {self.max_power} - bidding twice?")
        return self.outputs["energy"].loc[start:end_excl]

    def calculate_cashflow(self, product_type: str, orderbook: Orderbook):
        for order in orderbook:
            start = order["start_time"]
            end = order["end_time"]
            end_excl = end - self.index.freq

            if isinstance(order["accepted_volume"], dict):
                cashflow = float(
                    order["accepted_price"][i] * order["accepted_volume"][i]
                    for i in order["accepted_volume"].keys()
                )
            else:
                cashflow = float(order["accepted_price"] * order["accepted_volume"])

            hours = (end - start) / timedelta(hours=1)
            self.outputs[f"{product_type}_cashflow"].loc[start:end_excl] += (
                cashflow * hours
            )

    def calc_simple_marginal_cost(
        self,
    ):
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
        does not include ramping
        can be used for arbitrary start times in the future
        """
        end_excl = end - self.index.freq

        # check if unit is currently off and cannot be turned on yet
        # if unit.current_status == 0 and unit.current_down_time < unit.min_down_time:
        #    return 0, 0

        base_load = self.outputs["energy"][start:end_excl]
        heat_demand = self.outputs["heat"][start:end_excl]
        assert heat_demand.min() >= 0

        capacity_neg = self.outputs["capacity_neg"][start:end_excl]
        # needed minimum + capacity_neg - what is already sold is actual minimum
        min_power = self.min_power + capacity_neg - base_load
        # min_power should be at least the heat demand at that time
        min_power = min_power.where(min_power >= heat_demand, heat_demand)

        max_power = (
            self.forecaster.get_availability(self.id)[start:end_excl] * self.max_power
        )
        # provide reserve for capacity_pos
        max_power = max_power - self.outputs["capacity_pos"][start:end_excl]
        # remove what has already been bid
        max_power = max_power - base_load
        # make sure that max_power is > 0 for all timesteps
        max_power = max_power.where(max_power >= 0, 0)

        return min_power, max_power

    def calculate_marginal_cost(self, start: datetime, power: float):
        if self.marginal_cost is not None:
            return self.marginal_cost[start]
        else:
            return self.calc_marginal_cost_with_partial_eff(
                power_output=power,
                timestep=start,
            )

    def as_dict(self) -> dict:
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
