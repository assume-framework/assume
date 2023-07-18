import logging
from functools import lru_cache

import pandas as pd

from assume.strategies import OperationalWindow
from assume.units.base_unit import BaseUnit

logger = logging.getLogger(__name__)


class PowerPlant(BaseUnit):
    def __init__(
        self,
        id: str,
        unit_operator: str,
        technology: str,
        bidding_strategies: dict,
        index: pd.DatetimeIndex,
        max_power: float,
        min_power: float = 0.0,
        availability: pd.Series = None,
        efficiency: float = 1.0,
        fixed_cost: float = 0.0,
        partial_load_eff: bool = False,
        fuel_type: str = "others",
        fuel_price: float or pd.Series = 0.0,
        co2_price: float or pd.Series = 0.0,
        price_forecast: pd.Series = pd.Series(),
        res_demand_forecast: pd.Series = pd.Series(),
        emission_factor: float = 0.0,
        ramp_up: float = -1,
        ramp_down: float = -1,
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
        )

        self.max_power = max_power
        self.min_power = min_power
        self.availability = availability.to_dict() if availability is not None else None
        self.efficiency = efficiency
        self.partial_load_eff = partial_load_eff
        self.fuel_type = fuel_type
        if type(fuel_price) is pd.Series and len(fuel_price) == 1:
            fuel_price = fuel_price.item()
        self.fuel_price = fuel_price
        if type(co2_price) is pd.Series and len(co2_price) == 1:
            co2_price = co2_price.item()
        self.co2_price = co2_price
        self.price_forecast = (
            pd.Series(0.0, index=self.index) if price_forecast.empty else price_forecast
        )
        self.res_demand_forecast = (
            pd.Series(0.0, index=self.index)
            if res_demand_forecast.empty
            else res_demand_forecast
        )
        self.emission_factor = emission_factor

        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        # check ramping enabled
        if self.ramp_down == -1:
            self.ramp_down = max_power
        if self.ramp_up == -1:
            self.ramp_up = max_power
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
        if not self.partial_load_eff and type(self.fuel_price) is not pd.Series:
            fuel_prices = {self.fuel_type: self.fuel_price, "co2": self.co2_price}
            self.marginal_cost = self.calc_simple_marginal_cost(fuel_prices)
        elif not self.partial_load_eff:
            # calculate the marginal cost for the whole time series of fuel prices
            fuel_prices = pd.concat([self.fuel_price, self.co2_price], axis=1)
            # rename columns of fuel_prices to fule_type and co2
            fuel_prices.columns = [self.fuel_type, "co2"]
            self.marginal_cost = pd.DataFrame(
                index=self.index, columns=["marginal_cost"], data=0.0
            )
            self.marginal_cost["marginal_cost"] = fuel_prices.apply(
                self.calc_simple_marginal_cost, axis=1
            )
            self.marginal_cost = self.marginal_cost["marginal_cost"].to_dict()
        else:
            self.marginal_cost = None

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

    def calculate_operational_window(
        self,
        product_type: str,
        product_tuple: tuple,
    ) -> OperationalWindow:
        """Calculate the operation window for the next time step.

        Returns
        -------
        operational_window : dict
            Dictionary containing the operational window for the next time step.
        """

        start, end, only_hours = product_tuple
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        # check if unit is currently off and cannot be turned on yet
        if self.current_status == 0 and self.current_down_time < self.min_down_time:
            return None

        # check if units availability is 0
        if self.availability is not None and self.availability.at[start] == 0:
            return None

        if product_type == "energy":
            return self.calculate_energy_operational_window(start, end)
        elif product_type == "capacity_pos":
            return self.calculate_pos_reserve_operational_window(start, end)
        elif product_type == "capacity_neg":
            return self.calculate_neg_reserve_operational_window(start, end)

    def calculate_energy_operational_window(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> dict:
        current_power = self.get_output_before(start)

        min_power, max_power = self.calculate_min_max_power(start, end)

        # adjust for ramp down speed
        min_power = max(current_power - self.ramp_down, min_power)
        # adjust for ramp up speed
        max_power = min(current_power + self.ramp_up, max_power)

        # adjust min_power if sold negative reserve capacity on control reserve market
        min_power = min_power + self.outputs["capacity_neg"].at[start]

        # adjust max_power if sold positive reserve capacity on control reserve market
        max_power = max_power - self.outputs["capacity_pos"].at[start]

        if self.marginal_cost is None:
            return {
                "window": (start, end),
                "states": {
                    "current_power": {
                        "volume": current_power,
                        "cost": self.calc_marginal_cost_with_partial_eff(
                            power_output=current_power,
                            timestep=start,
                        ),
                    },
                    "min_power": {
                        "volume": min_power,
                        "cost": self.calc_marginal_cost_with_partial_eff(
                            power_output=current_power + min_power,
                            timestep=start,
                        ),
                    },
                    "max_power": {
                        "volume": max_power,
                        "cost": self.calc_marginal_cost_with_partial_eff(
                            power_output=current_power + max_power,
                            timestep=start,
                        ),
                    },
                },
            }

        marginal_cost = (
            self.marginal_cost[start]
            if type(self.marginal_cost) is dict
            else self.marginal_cost
        )
        return {
            "window": (start, end),
            "states": {
                "current_power": {
                    "volume": current_power,
                    "cost": marginal_cost,
                },
                "min_power": {
                    "volume": min_power,
                    "cost": marginal_cost,
                },
                "max_power": {
                    "volume": max_power,
                    "cost": marginal_cost,
                },
            },
        }

    def calculate_pos_reserve_operational_window(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> dict:
        previous_power = self.get_output_before(start)

        min_power, max_power = self.calculate_min_max_power(start, end)

        # should be adjusted to account for ramping speeds
        # and differences between resolutions of energy and reserve markets
        available_pos_reserve = min(max_power - previous_power, self.ramp_up)

        if self.marginal_cost is None:
            marginal_cost = self.calc_marginal_cost_with_partial_eff(
                power_output=self.max_power,
                timestep=start,
            )
        else:
            marginal_cost = (
                self.marginal_cost[start]
                if type(self.marginal_cost) is dict
                else self.marginal_cost
            )

        operational_window = {
            "window": (start, end),
            "states": {
                "pos_reserve": {
                    "volume": available_pos_reserve,
                    "cost": marginal_cost,
                },
            },
        }

        return operational_window

    def calculate_neg_reserve_operational_window(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> dict:
        previous_power = self.get_output_before(start)

        min_power, max_power = self.calculate_min_max_power(start, end)

        # should be adjusted to account for ramping speeds
        # and differences between resolutions of energy and reserve markets
        available_neg_reserve = max(0, min(previous_power - min_power, self.ramp_down))

        if self.marginal_cost is None:
            marginal_cost = self.calc_marginal_cost_with_partial_eff(
                power_output=self.max_power,
                timestep=start,
            )
        else:
            marginal_cost = (
                self.marginal_cost[start]
                if type(self.marginal_cost) is dict
                else self.marginal_cost
            )

        operational_window = {
            "window": (start, end),
            "states": {
                "neg_reserve": {
                    "volume": available_neg_reserve,
                    "cost": marginal_cost,
                },
            },
        }

        if available_neg_reserve < 0:
            logger.error("available_neg_reserve < 0")

        return operational_window

    def execute_current_dispatch(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ):
        end_excl = end - self.index.freq
        if self.outputs["energy"][start:end_excl].min() < self.min_power:
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

    def calculate_cashflow(self, start, end, clearing_price):
        start = start
        end_excl = end - self.index.freq

        self.outputs["cashflow"].loc[start:end_excl] = (
            clearing_price
            * self.outputs["energy"].loc[start:end_excl]
            * (end - start).total_seconds()
            / 3600
        )

    def calc_simple_marginal_cost(
        self,
        fuel_prices: dict,
    ):
        marginal_cost = (
            fuel_prices[self.fuel_type] / self.efficiency
            + fuel_prices["co2"] * self.emission_factor / self.efficiency
            + self.fixed_cost
        )

        return marginal_cost

    @lru_cache(maxsize=256)
    def calc_marginal_cost_with_partial_eff(
        self,
        power_output: float,
        timestep: pd.Timestamp = None,
    ) -> float or pd.Series:
        fuel_price = (
            self.fuel_price.at[timestep]
            if type(self.fuel_price) is pd.Series
            else self.fuel_price
        )
        co2_price = (
            self.co2_price.at[timestep]
            if type(self.co2_price) is pd.Series
            else self.co2_price
        )

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

        marginal_cost = (
            fuel_price / efficiency
            + co2_price * self.emission_factor / efficiency
            + self.fixed_cost
        )

        return marginal_cost

    def calculate_min_max_power(self, start: pd.Timestamp, end: pd.Timestamp):
        end_excl = end - self.index.freq
        base_load = self.outputs["energy"][start:end_excl]
        # check if min_power is a series or a float
        min_power = max(self.min_power - base_load.min(), 0)

        # check if max_power is a series or a float
        max_power = (
            self.availability[start] * self.max_power
            if type(self.availability) is dict
            else self.max_power
        )
        max_power = max(max_power - base_load.max(), 0)

        return min_power, max_power

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
