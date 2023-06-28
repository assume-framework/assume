import logging
from functools import cache, lru_cache

import pandas as pd

from assume.units.base_unit import BaseUnit

logger = logging.getLogger(__name__)


class RL_PowerPlant(PowerPlant):
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
        self.capacity_factor = capacity_factor
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
        self.downtime_hot_start = downtime_hot_start
        self.downtime_warm_start = downtime_warm_start

        self.fixed_cost = fixed_cost
        self.hot_start_cost = hot_start_cost * max_power
        self.warm_start_cost = warm_start_cost * max_power
        self.cold_start_cost = cold_start_cost * max_power

        self.heat_extraction = heat_extraction
        self.max_heat_extraction = max_heat_extraction

        self.location = location

        if not self.partial_load_eff and type(self.fuel_price) is float:
            fuel_prices = {self.fuel_type: self.fuel_price, "co2": self.co2_price}
            self.marginal_cost = self.calc_simple_marginal_cost(fuel_prices)
        elif not self.partial_load_eff:
            # calculate the marginal cost for the whole time series of fuel prices
            fuel_prices = pd.concat([self.fuel_price, self.co2_price], axis=1)
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

    def calculate_bids(
        self,
        market_config,
        product_tuple,
        observation,
    ):
        """Calculate the bids for the next time step."""

        if self.bidding_strategies[market_config.product_type] == "rl_strategy":
            # define unit wise observations
            # add the total scaled capacity and marginal costs
            # TODO store total capacity and marginal costs some where
            start = world.start
            end = world.end
            now = datetime.utcfromtimestamp(self.context.current_timestamp)
            delta_t = now - start  # in metric of market

            obs_len = 4
            obs = observation.copy()

            # get the marginal cost
            if delta_t < len(self.world.snapshots) - obs_len:
                obs.extend(self.scaled_marginal_cost[delta_t : delta_t + obs_len])
            else:
                obs.extend(self.scaled_marginal_cost[delta_t:])
                obs.extend(
                    self.scaled_marginal_cost[
                        : obs_len - (len(self.world.snapshots) - delta_t)
                    ]
                )

            if delta_t < obs_len:
                obs.extend(
                    self.total_scaled_capacity[
                        len(self.world.snapshots) - obs_len + delta_t :
                    ]
                )
                obs.extend(self.total_scaled_capacity[:delta_t])
            else:
                obs.extend(self.total_scaled_capacity[delta_t - obs_len : delta_t])

            obs = np.array(obs)
            obs = (
                th.tensor(obs, dtype=self.float_type)
                .to(self.device, non_blocking=True)
                .view(-1)
            )

            return self.bidding_strategies[market_config.product_type].step(
                unit=self, observation=obs
            )

        else:
            if market_config.product_type not in self.bidding_strategies:
                return None

            # get operational window for each unit
            operational_window = self.calculate_operational_window(
                product_type=market_config.product_type,
                product_tuple=product_tuple,
            )

            # check if operational window is valid
            if operational_window is None:
                return None

            return self.bidding_strategies[market_config.product_type].calculate_bids(
                unit=self,
                market_config=market_config,
                operational_window=operational_window,
            )

    def set_dispatch_plan(
        self,
        dispatch_plan: dict,
        start: pd.Timestamp,
        end: pd.Timestamp,
        product_type: str,
        clearing_price: float,
    ):
        end_excl = end - self.index.freq

        # based on product type
        if product_type == "energy":
            self.total_power_output.loc[start:end_excl] += dispatch_plan["total_power"]
        elif product_type == "capacity_pos":
            self.pos_capacity_reserve.loc[start:end_excl] += dispatch_plan[
                "total_power"
            ]
        elif product_type == "capacity_neg":
            self.neg_capacity_reserve.loc[start:end_excl] += dispatch_plan[
                "total_power"
            ]

        # TODO add update of bidding strategy here

    def execute_current_dispatch(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ):
        end_excl = end - self.index.freq
        if self.total_power_output[start:end_excl].min() < self.min_power:
            self.total_power_output.loc[start:end_excl] = 0
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
        # if self.total_power_output[start:end_excl].max() > self.max_power:
        #     max_pow = self.total_power_output[start:end_excl].max()
        #     logger.error(f"{max_pow} greater than {self.max_power} - bidding twice?")
        return self.total_power_output.loc[start:end_excl]

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
        base_load = self.total_power_output[start:end]
        # check if min_power is a series or a float
        min_power = (
            self.min_power[start:end].max()
            if type(self.min_power) is pd.Series
            else self.min_power
        )
        min_delta = min_power - base_load.min()

        # check if max_power is a series or a float
        max_power = (
            self.capacity_factor[start:end].min() * self.max_power
            if type(self.capacity_factor) is pd.Series
            else self.capacity_factor * self.max_power
        )
        max_delta = max_power - base_load.max()

        return min_delta, max_delta

    # TODO take obs from unit operator and give it the unit specific observation part
    def create_obs(self, t):
        obs_len = 4
        obs = self.agent.obs.copy()

        # get the marginal cost
        if t < len(self.world.snapshots) - obs_len:
            obs.extend(self.scaled_marginal_cost[t : t + obs_len])
        else:
            obs.extend(self.scaled_marginal_cost[t:])
            obs.extend(
                self.scaled_marginal_cost[: obs_len - (len(self.world.snapshots) - t)]
            )

        if t < obs_len:
            obs.extend(
                self.total_scaled_capacity[len(self.world.snapshots) - obs_len + t :]
            )
            obs.extend(self.total_scaled_capacity[:t])
        else:
            obs.extend(self.total_scaled_capacity[t - obs_len : t])

        obs = np.array(obs)
        obs = (
            th.tensor(obs, dtype=self.float_type)
            .to(self.device, non_blocking=True)
            .view(-1)
        )

        return obs.detach().clone()
