# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import logging
from functools import lru_cache
from typing import TypeAlias

import pandas as pd

from assume.common.fast_pandas import FastIndex, FastSeries
from assume.common.base import BaseUnit
from assume.units.powerplant import PowerPlant
from assume.units.demand import Demand
from assume.units.exchange import Exchange

ForecastIndex: TypeAlias = FastIndex | pd.DatetimeIndex | pd.Series
ForecastSeries: TypeAlias = FastSeries | list | float | pd.Series

log = logging.getLogger(__name__)

def is_renewable(name:str) -> bool:
    return "wind" in name.lower() or "solar" in name.lower()

def _ensure_not_none(
    df: pd.DataFrame | None, index: pd.DatetimeIndex | pd.Series, check_index=False
) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(index=index)
    if check_index and index.freq != df.index.inferred_freq:
        raise ValueError("Forecast frequency does not match index frequency.")
    return df

@lru_cache(max_size=10)
def sort_units(units:list[BaseUnit], market_id:str):
    pps: list[PowerPlant] = []
    demands: list[Demand] = []
    exchanges: list[Exchange] = []

    for unit in units:
        if market_id not in unit.bidding_strategies:
            continue
        if isinstance(unit, PowerPlant):
            pps.append(unit)
        elif isinstance(unit, Demand):
            demands.append(unit)
        elif isinstance(unit, Exchange):
            exchanges.append(unit)

    return pps, demand, exchanges

# @lru_cache(maxsize=10)
def calculate_sum_demand(
    demand_units: list[Demand],
    exchange_units: list[Exchange],
    demand_df: ForecastSeries,
    exchange_df: ForecastSeries,
):
    demand_names = [unit.name for unit in demand_units]
    sum_demand = demand_df[demand_names].sum(axis=1)

    # get exchanges if exchange_units are available
    if exchange_units:  # if not empty
        # get sum of imports as name of exchange_unit_import
        import_units = [f"{unit.name}_import" for unit in exchange_units]
        sum_imports = exchanges[import_units].sum(axis=1)
        # get sum of exports as name of exchange_unit_export
        export_units = [f"{unit.name}_export" for unit in exchange_units]
        sum_exports = exchanges[export_units].sum(axis=1)
        # add imports and exports to the sum_demand
        sum_demand += sum_imports - sum_exports

    return sum_demand

@lru_cache(max_size=1000)
def calculate_naive_price(
    index: ForecastIndex,
    units: dict[str, BaseUnit],
    market_configs: dict[str, dict],
    demand_df: ForecastSeries = None,
    forecast_df: ForecastSeries = None,
    exchange_df: ForecastSeries = None,
) -> dict[str, ForecastSeries]:

    demand_df = _ensure_not_none(demand_df, index)
    forecast_df = _ensure_not_none(demand_df, index)
    exchange_df = _ensure_not_none(demand_df, index)

    price_forecasts: dict[str, pd.Series] = {}

    for market_id, config in self.market_configs.items():
        if config["product_type"] != "energy":
            log.warning(
                f"Price forecast could not be calculated for {market_id}. It can only be calculated for energy-only markets for now."
            )
            continue

        price_forecasts[market_id] = forecast_df.get(f"price_{market_id}")
        if price_forecasts[market_id] is not None:
            # go next if forecast existing
            continue

        # 1. Sort units by type and filter for units with bidding strategy for the given market_id
        powerplant_units, demand_units, exchange_units = sort_units(
            units, market_id
        )

        # 2. Calculate marginal costs for each unit and time step.
        #    The resulting DataFrame has rows = time steps and columns = units.
        #    shape: (index_len, num_pp_units)
        # marginal_costs = powerplants_units.apply(calculate_marginal_cost, axis=1).T
        marginal_costs = pd.Series([unit.marginal_cost for unit in powerplant_units]).T


        # 3. Compute available power for each unit at each time step.
        #    shape: (index_len, num_pp_units)
        # power = self._calc_power()
        power = pd.Series([unit.forecaster.availability * unit.max_power for unit in powerplant_units]).T

        # 4. Process the demand.
        #    Filter demand units with a bidding strategy and sum their forecasts for each time step.
        sum_demand = calculate_sum_demand(demand_units, exchange_units, demand_df, exchange_df)

        # 5. Initialize the price forecast series.
        price_forecast = pd.Series(index=index, data=0.0)

        # 6. Loop over each time step
        for t in index:
            # Get marginal costs and available power for time t (both are Series indexed by unit)
            mc_t = marginal_costs.loc[t]
            power_t = power.loc[t]
            demand_t = sum_demand.loc[t]

            # Sort units by their marginal cost in ascending order for time t.
            sorted_units = mc_t.sort_values().index
            sorted_mc = mc_t.loc[sorted_units]
            sorted_power = power_t.loc[sorted_units]

            # Compute the cumulative sum of available power in the sorted order.
            cumsum_power = sorted_power.cumsum()

            # Find the first unit where the cumulative available power meets or exceeds demand.
            matching_units = cumsum_power[cumsum_power >= demand_t]
            if matching_units.empty:
                # If available capacity is insufficient, set the price to 1000.
                price = 1000.0
            else:
                # The marginal cost of the first unit that meets demand becomes the price.
                price = sorted_mc.loc[matching_units.index[0]]

            price_forecast.loc[t] = price

        price_forecasts[market_id] = price_forecast

    return price_forecasts

@lru_cache(maxsize=100)
def calculate_residual_load(
    units: dict[str, BaseUnit],
    market_configs: dict[str, dict],
    demand_df: ForecastSeries = None,
    forecast_df: ForecastSeries = None,
    exchange_df: ForecastSeries = None,
) -> dict[str, pd.Series]:

    demand_df = _ensure_not_none(demand_df, index)
    forecast_df = _ensure_not_none(demand_df, index)
    exchange_df = _ensure_not_none(demand_df, index)

    residual_loads: dict[str, pd.Series] = {}

    for market_id, config in self.market_configs.items():
        if config["product_type"] != "energy":
            log.warning(
                f"Load forecast could not be calculated for {market_id}. It can only be calculated for energy-only markets for now."
            )
            continue

        residual_loads[market_id] = forecast_df.get(f"residual_load_{market_id}")
        if residual_loads[market_id] is not None:
            # go next if forecast existing
            continue
            
        powerplants_units, demand_units, exchange_units = sort_units(
            units, market_id
        )

        sum_demand = calculate_sum_demand(demand_units, exchange_units, demand_df, exchange_df)

        # shape: (num_pp_units, index_len) -> (index_len)
        vre_feed_in_df = pd.Series([unit.max_power * unit.forecaster.availability
            for unit in powerplants_units if is_renewable(unit.technology)]).sum(axis=0)

        # vre_feed_in_df = self._calc_power(mask).sum(axis=1)
        if vre_feed_in_df.empty:
            vre_feed_in_df = 0
        res_demand_df = sum_demand - vre_feed_in_df

        residual_loads[market_id] = res_demand_df
    return residual_loads

@lru_cache(maxsize=1000)
def preprocess_price(args1,args2):
    # TODO implement preprocess
    # this should allow to remove the whole forecast_initialisation stuff
    # this functions does not evaluate again if it is called by different Forecasts with the same input.
    pass

class UnitForecaster:
    """
    A generalized forecaster for units

    Attributes:
        index (ForecastIndex): the index of all forecast series in this unit
        availability (ForecastSeries): Forecasted availability of a unit
    """

    def __init__(
        self,
        index: ForecastIndex,
        # market_prices: dict[str, ForecastSeries] = None,
        # residual_load: dict[str, ForecastSeries] = None,
        availability: ForecastSeries = 1,
    ):
        if not isinstance(index, FastIndex):
            index = FastIndex(start=index[0], end=index[-1], freq=pd.infer_freq(index))
        # if market_prices is None:
        #     market_prices = {"EOM": 50}  # default value for tests
        # if residual_load is None:
        #     residual_load = {}
        self.index: FastIndex = index
        self.availability: FastSeries = self._to_series(availability)
        # self.price = self._dict_to_series(market_prices)
        # self.residual_load = self._dict_to_series(residual_load)

    def _to_series(self, item: ForecastSeries) -> FastSeries:
        if isinstance(item, FastSeries):
            return item
        return FastSeries(index=self.index, value=item)

    def _dict_to_series(self, d: dict[str, ForecastSeries]) -> dict[str, FastSeries]:
        result: dict[str, FastSeries] = {}
        for key, value in d.items():
            result[key] = self._to_series(value)
        return result

    def preprocess(self, args1, args2):
        """
        """
        #preprocess_price(args1, args2)
        pass

    def initialize(
        self,
        units: dict[str, BaseUnit],
        market_configs: dict[str, dict],
        # fuel_prices: ForecastSeries = None,
        demand_df: ForecastSeries = None,
        forecast_df: ForecastSeries = None,
        exchange_df: ForecastSeries = None,
    ):
        """
        """
        self.price_forecasts = calculate_naive_price(
            self.index,
            units,
            market_configs,
            demand_df,
            forecast_df,
            exchange_df
        )

        self.residual_load = calculate_residual_load(
            units,
            market_configs,
            demand_df,
            forecast_df,
            exchange_df,
        )

    def update(self, args1, args2):
        """
        """
        pass
        # TODO like preprocess


class CustomUnitForecaster(UnitForecaster):
    """
    A more general unit forecaster used e.g. for csv imports

    Attributes:
        index (ForecastIndex): the index of all forecast series in this unit
        **kwargs: The desired attributes for this forecast
    """

    def __init__(
        self,
        index: ForecastIndex,
        **kwargs,
    ):
        super().__init__(index)
        for k, v in kwargs.items():
            if isinstance(v, pd.Series):
                v = self._to_series(v)
            self.__setattr__(k, v)


class DemandForecaster(UnitForecaster):
    """
    A forecaster for demand units

    Attributes:
        index (ForecastIndex): the index of all forecast series in this unit
        market_prices (dict[str, ForecastSeries]): Map of market_id -> forecasted prices
        demand (ForecastSeries): Forecasted demand (must be negative)
        residual_load (dict[str, ForecastSeries]): Map of market_id -> forecasted residual load
        availability (ForecastSeries): Forecasted availability of a unit
    """

    def __init__(
        self,
        index: ForecastIndex,
        # market_prices: dict[str, ForecastSeries] = None,
        demand: ForecastSeries = -100,
        # residual_load: dict[str, ForecastSeries] = None,
        availability: ForecastSeries = 1,
        price_forecasts: ForecastSeries = None,
    ):
        # super().__init__(index, market_prices, residual_load, availability)
        super().__init__(index, availability)
        self.demand = self._to_series(demand)
        if any(self.demand > 0):
            raise ValueError(f"{demand=} must be negative")

    # TODO:
    # initialize
    # preprocess


class PowerplantForecaster(UnitForecaster):
    """
    A forecaster for powerplant units
    Attributes:
        index (ForecastIndex): the index of all forecast series in this unit
        fuel_prices (dict[str, ForecastSeries]): Map of fuel type -> forecasted fuel prices
        market_prices (dict[str, ForecastSeries]): Map of market_id -> forecasted prices
        residual_load (dict[str, ForecastSeries]): Map of market_id -> forecasted residual load
        availability (ForecastSeries): Forecasted availability of a unit
    """

    def __init__(
        self,
        index: ForecastIndex,
        fuel_prices: dict[str, ForecastSeries] = None,
        #market_prices: dict[str, ForecastSeries] = None,
        #residual_load: dict[str, ForecastSeries] = None,
        availability: ForecastSeries = 1,
    ):
        # super().__init__(index, market_prices, residual_load, availability)
        super().__init__(index, availability)
        if fuel_prices is None:
            fuel_prices = {}
        self.fuel_prices = self._dict_to_series(fuel_prices)

    def get_price(self, fuel: str) -> FastSeries:
        if fuel not in self.fuel_prices:
            return self._to_series(0)
        return self.fuel_prices[fuel]

    # TODO:
    # initialize
    # preprocess

class DsmUnitForecaster(UnitForecaster):
    def __init__(
        self,
        index: ForecastIndex,
        # market_prices: dict[str, ForecastSeries] = None,
        fuel_prices=dict[str, ForecastSeries],
        availability: ForecastSeries = 1,
        # congestion_signal: ForecastSeries = 0,
        # residual_load: dict[str, ForecastSeries] = None,
        # renewable_utilisation_signal: ForecastSeries = 0,
        # electricity_price: ForecastSeries = None,
    ):
        #super().__init__(index, market_prices, residual_load, availability)
        super().__init__(index, availability)
        self.fuel_prices = self._dict_to_series(fuel_prices)
        # self.congestion_signal = self._to_series(congestion_signal)
        # self.electricity_price = self._to_series(electricity_price)  # --> market_prices.get("price_EOM")
        # self.renewable_utilisation_signal = self._to_series(
        #    renewable_utilisation_signal
        #)

    def initialize():
        # TODO: preprocess that creates price

        # END TODO:
        self.self.electricity_price = self.prices.get("price_EOM")

class SteelplantForecaster(UnitForecaster):
    """
    A forecaster for steelplant units
    Attributes:
        index (ForecastIndex): the index of all forecast series in this unit
        market_prices (dict[str, ForecastSeries]): Map of market_id -> forecasted prices
        fuel_prices (dict[str, ForecastSeries]): Map of fuel type -> forecasted fuel prices
        availability (ForecastSeries): Forecasted availability of a unit
        congestion_signal (ForecastSeries): Forecasted congestion signal
        residual_load (dict[str, ForecastSeries]): Map of market_id -> forecasted residual load
        renewable_utilisation_signal (ForecastSeries): Forecasted renewable utilisation signal
    """

    def __init__(
        self,
        index: ForecastIndex,
        # market_prices: dict[str, ForecastSeries] = None,
        fuel_prices=dict[str, ForecastSeries],
        availability: ForecastSeries = 1,
        # congestion_signal: ForecastSeries = 0,
        # residual_load: dict[str, ForecastSeries] = None,
        # renewable_utilisation_signal: ForecastSeries = 0,
        # electricity_price: ForecastSeries = None,
    ):
        #super().__init__(index, market_prices, residual_load, availability)
        super().__init__(index, availability)
        self.fuel_prices = self._dict_to_series(fuel_prices)
        # self.congestion_signal = self._to_series(congestion_signal)
        # self.electricity_price = self._to_series(electricity_price)  # --> market_prices.get("price_EOM")
        # self.renewable_utilisation_signal = self._to_series(
        #    renewable_utilisation_signal
        #)

    def get_price(self, fuel: str) -> FastSeries:
        if fuel not in self.fuel_prices:
            return self._to_series(0)
        return self.fuel_prices[fuel]

    # TODO:
    # initialize
    # preprocess


class SteamgenerationForecaster(UnitForecaster):
    """
    A forecaster for steam generation units
    Attributes:
        index (ForecastIndex): the index of all forecast series in this unit
        electricity_price (ForecastSeries): Forecasted electricity price
        fuel_prices (dict[str, ForecastSeries]): Map of fuel type -> forecasted fuel prices
        demand (ForecastSeries): Forecasted electricity demand
        electricity_price_flex (ForecastSeries): Forecasted flexible electricity price
        thermal_demand (ForecastSeries): Forecasted thermal demand
        congestion_signal (ForecastSeries): Forecasted congestion signal
        renewable_utilisation_signal (ForecastSeries): Forecasted renewable utilisation signal
        thermal_storage_schedule (ForecastSeries): Forecasted thermal storage schedule
        residual_load (dict[str, ForecastSeries]): Map of market_id -> forecasted residual load
        availability (ForecastSeries): Forecasted availability of a unit
        market_prices (dict[str, ForecastSeries]): Map of market_id -> forecasted prices
    """

    def __init__(
        self,
        index: ForecastIndex,
        # electricity_price: ForecastSeries,
        fuel_prices: dict[str, ForecastSeries],
        demand: ForecastSeries = 0,
        electricity_price_flex: ForecastSeries = 0,
        thermal_demand: ForecastSeries = 0,
        # congestion_signal: ForecastSeries = 0,
        # renewable_utilisation_signal: ForecastSeries = 0,
        thermal_storage_schedule: ForecastSeries = 0,
        # residual_load: dict[str, ForecastSeries] = None,
        availability: ForecastSeries = 1,
        # market_prices: dict[str, ForecastSeries] = None,
    ):
        # super().__init__(index, market_prices, residual_load, availability)
        super().__init__(index, availability)
        self.demand = demand
        #self.electricity_price = self._to_series(electricity_price)
        self.electricity_price_flex = self._to_series(electricity_price_flex)
        self.fuel_prices = self._dict_to_series(fuel_prices)
        self.thermal_demand = self._to_series(thermal_demand)
        #self.congestion_signal = self._to_series(congestion_signal)
        #self.renewable_utilisation_signal = self._to_series(
        #    renewable_utilisation_signal
        #)
        self.thermal_storage_schedule = self._to_series(thermal_storage_schedule)

    def get_price(self, fuel: str) -> FastSeries:
        if fuel not in self.fuel_prices:
            return self._to_series(0)
        return self.fuel_prices[fuel]


class BuildingForecaster(UnitForecaster):
    """
    A forecaster for building units
    Attributes:
        index (ForecastIndex): the index of all forecast series in this unit
        fuel_prices (dict[str, ForecastSeries]): Map of fuel type -> forecasted fuel prices
        heat_demand (ForecastSeries): Forecasted heat demand
        ev_load_profile (ForecastSeries): Forecasted electric vehicle load profile
        battery_load_profile (ForecastSeries): Forecasted battery load profile
        pv_profile (ForecastSeries): Forecasted photovoltaic profile
        load_profile (ForecastSeries): Forecasted overall load profile
        residual_load (dict[str, ForecastSeries]): Map of market_id -> forecasted residual load
        availability (ForecastSeries): Forecasted availability of a unit
        market_prices (dict[str, ForecastSeries]): Map of market_id -> forecasted prices
        load_profile (ForecastSeries): Forecasted load profile
    """

    def __init__(
        self,
        index: ForecastIndex,
        fuel_prices: dict[str, ForecastSeries],
        heat_demand: ForecastSeries,
        ev_load_profile: ForecastSeries,
        battery_load_profile: ForecastSeries,
        pv_profile: ForecastSeries,
        #electricity_price: ForecastSeries = None,
        #residual_load: dict[str, ForecastSeries] = None,
        availability: ForecastSeries = 1,
        #market_prices: dict[str, ForecastSeries] = None,
        load_profile: ForecastSeries = 0,
    ):
        # super().__init__(index, market_prices, residual_load, availability)
        super().__init__(index, availability)
        self.fuel_prices = self._dict_to_series(fuel_prices)
        self.heat_demand = self._to_series(heat_demand)
        self.ev_load_profile = self._to_series(ev_load_profile)
        self.battery_load_profile = self._to_series(battery_load_profile)
        self.pv_profile = self._to_series(pv_profile)
        self.load_profile = self._to_series(load_profile)
        # self.electricity_price = self._to_series(electricity_price)

    def get_price(self, fuel: str) -> FastSeries:
        if fuel not in self.fuel_prices:
            return self._to_series(0)
        return self.fuel_prices[fuel]


class HydrogenForecaster(UnitForecaster):
    """
    A forecaster for hydrogen units
    Attributes:
        index (ForecastIndex): the index of all forecast series in this unit
        electricity_price (ForecastSeries): Forecasted electricity price
        hydrogen_demand (ForecastSeries): Forecasted hydrogen demand
        seasonal_storage_schedule (ForecastSeries): Forecasted seasonal storage schedule
        residual_load (dict[str, ForecastSeries]): Map of market_id -> forecasted residual load
        availability (ForecastSeries): Forecasted availability of a unit
        market_prices (dict[str, ForecastSeries]): Map of market_id -> forecasted prices
    """

    def __init__(
        self,
        index: ForecastIndex,
        # electricity_price: ForecastSeries,
        hydrogen_demand: ForecastSeries,
        seasonal_storage_schedule: ForecastSeries = 0,
        #residual_load: dict[str, ForecastSeries] = None,
        availability: ForecastSeries = 1,
        #market_prices: dict[str, ForecastSeries] = None,
    ):
        # super().__init__(index, market_prices, residual_load, availability)
        super().__init__(index, availability)
        # self.electricity_price = self._to_series(electricity_price)
        self.hydrogen_demand = self._to_series(hydrogen_demand)
        self.seasonal_storage_schedule = self._to_series(seasonal_storage_schedule)


class ExchangeForecaster(UnitForecaster):
    """
    A forecaster for exchange units
    Attributes:
        index (ForecastIndex): the index of all forecast series in this unit
        volume_import (ForecastSeries): Forecasted import volume
        volume_export (ForecastSeries): Forecasted export volume
        residual_load (dict[str, ForecastSeries]): Map of market_id -> forecasted residual load
        availability (ForecastSeries): Forecasted availability of a unit
        market_prices (dict[str, ForecastSeries]): Map of market_id -> forecasted prices
    """

    def __init__(
        self,
        index: ForecastIndex,
        volume_import: ForecastSeries = 0,
        volume_export: ForecastSeries = 0,
        #residual_load: dict[str, ForecastSeries] = None,
        availability: ForecastSeries = 1,
        #market_prices: dict[str, ForecastSeries] = None,
    ):
        # super().__init__(index, market_prices, residual_load, availability)
        super().__init__(index, availability)
        self.volume_import = self._to_series(volume_import)
        self.volume_export = self._to_series(volume_export)
