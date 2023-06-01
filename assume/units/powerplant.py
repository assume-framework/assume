import pandas as pd

from assume.units.base_unit import BaseUnit


class PowerPlant(BaseUnit):
    def __init__(
        self,
        id: str,
        unit_operator: str,
        technology: str,
        bidding_strategies: dict,
        index: pd.DatetimeIndex,
        max_power: float or pd.Series,
        min_power: float or pd.Series = 0.0,
        capacity_factor: float or pd.Series = 1.0,
        efficiency: float = 1,
        partial_load_eff: bool = False,
        fuel_type: str = "others",
        fuel_price: float or pd.Series = 0.0,
        co2_price: float or pd.Series = 0.0,
        price_forecast: pd.Series = pd.Series(),
        res_demand_forecast: pd.Series = pd.Series(),
        emission_factor: float = 0.0,
        ramp_up: float = -1,
        ramp_down: float = -1,
        fixed_cost: float = 0,
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
        **kwargs
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
        self.capacity_factor = capacity_factor
        self.efficiency = efficiency
        self.partial_load_eff = partial_load_eff
        self.fuel_type = fuel_type
        self.fuel_price = fuel_price
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
        self.downtime_hot_start = downtime_hot_start
        self.downtime_warm_start = downtime_warm_start

        self.fixed_cost = fixed_cost
        self.hot_start_cost = hot_start_cost * max_power
        self.warm_start_cost = warm_start_cost * max_power
        self.cold_start_cost = cold_start_cost * max_power

        self.heat_extraction = heat_extraction
        self.max_heat_extraction = max_heat_extraction

        self.location = location

    def reset(self):
        """Reset the unit to its initial state."""

        self.current_status = 1
        self.current_down_time = self.min_down_time

        self.total_power_output = pd.Series(0.0, index=self.index)
        # workaround if market schedules do not match
        # for example neg_reserve is required but market did not bid yet
        # it does set a usage in times where no power is used by the market
        # self.total_power_output.loc[:] = self.min_power + 0.5 * (
        #    self.max_power - self.min_power
        # )

        self.total_heat_output = pd.Series(0.0, index=self.index)
        self.power_loss_chp = pd.Series(0.0, index=self.index)

        self.pos_capacity_reserve = pd.Series(0.0, index=self.index)
        self.neg_capacity_reserve = pd.Series(0.0, index=self.index)

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

        if self.current_status == 0 and self.current_down_time < self.min_down_time:
            return None

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
        current_power = self.total_power_output.at[start - self.index.freq]

        min_power, max_power = self.calculate_min_max_power(timestep=start)

        # adjust for ramp down speed
        min_power = max(current_power - self.ramp_down, min_power)
        # adjust for ramp up speed
        max_power = min(current_power + self.ramp_up, max_power)

        # adjust min_power if sold negative reserve capacity on control reserve market
        min_power = min_power + self.neg_capacity_reserve.at[start]

        # adjust max_power if sold positive reserve capacity on control reserve market
        max_power = max_power - self.pos_capacity_reserve.at[start]

        operational_window = {
            "window": {"start": start, "end": end},
            "current_power": {
                "power": current_power,
                "marginal_cost": self.calc_marginal_cost(
                    power_output=current_power,
                    timestep=start,
                ),
            },
            "min_power": {
                "power": min_power,
                "marginal_cost": self.calc_marginal_cost(
                    power_output=min_power,
                    timestep=start,
                ),
            },
            "max_power": {
                "power": max_power,
                "marginal_cost": self.calc_marginal_cost(
                    power_output=max_power,
                    timestep=start,
                ),
            },
        }

        return operational_window

    def calculate_reserve_operational_window(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> dict:
        current_power = self.total_power_output.at[start - self.index.freq]

        min_power, max_power = self.calculate_min_max_power(timestep=start)

        # should be adjusted to account for ramping speeds
        # and differences between resolutions of energy and reserve markets
        if self.ramp_up != -1:
            available_pos_reserve = min(max_power - current_power, self.ramp_up)
        else:
            available_pos_reserve = max_power - current_power

        available_pos_reserve = min(max_power - current_power, self.ramp_up)
        available_neg_reserve = max(0, min(current_power - min_power, self.ramp_down))

        operational_window = {
            "window": {"start": start, "end": end},
            "pos_reserve": {
                "capacity": available_pos_reserve,
            },
            "neg_reserve": {
                "capacity": available_neg_reserve,
            },
        }

        if available_neg_reserve < 0:
            print("available_neg_reserve < 0")

        return operational_window

    def calculate_bids(
        self,
        product_type,
        product_tuple,
    ):
        return super().calculate_bids(
            product_type=product_type,
            product_tuple=product_tuple,
        )

    def get_dispatch_plan(self, dispatch_plan, time_period):
        if dispatch_plan["total_power"] >= self.min_power:
            self.market_success_list[-1] += 1
            self.current_status = 1
            self.current_down_time = 0
            self.total_power_output.loc[time_period] = dispatch_plan["total_power"]
        else:
            self.current_status = 0
            self.current_down_time += 1
            self.total_power_output.loc[time_period] = 0

            if self.market_success_list[-1] != 0:
                self.mean_market_success = sum(self.market_success_list) / len(
                    self.market_success_list
                )
                self.market_success_list.append(0)

    def calc_marginal_cost(
        self,
        power_output: float,
        timestep: pd.Timestamp,
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

        # Partial load efficiency dependent marginal costs
        if not self.partial_load_eff:
            marginal_cost = (
                fuel_price / self.efficiency
                + co2_price * self.emission_factor / self.efficiency
                + self.fixed_cost
            )

            return marginal_cost

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

    def calculate_min_max_power(self, timestep: pd.Timestamp):
        # check if min_power is a series or a float
        min_power = (
            self.min_power.at[timestep]
            if type(self.min_power) is pd.Series
            else self.min_power
        )

        # check if max_power is a series or a float
        max_power = (
            self.capacity_factor.at[timestep] * self.max_power
            if type(self.capacity_factor) is pd.Series
            else self.capacity_factor * self.max_power
        )

        return min_power, max_power
