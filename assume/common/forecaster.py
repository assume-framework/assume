# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
from typing import TypeAlias

import pandas as pd

from assume.common.fast_pandas import FastIndex, FastSeries

ForecastIndex: TypeAlias = FastIndex | pd.DatetimeIndex | pd.Series
ForecastSeries: TypeAlias = FastSeries | list | float | pd.Series


class UnitForecaster:
    """
    A generalized forecaster for units

    Attributes:
        index (ForecastIndex): the index of all forecast series in this unit
        market_prices (dict[str, ForecastSeries]): Map of market_id -> forecasted prices
        residual_load (dict[str, ForecastSeries]): Map of market_id -> forecasted residual load
        availability (ForecastSeries): Forecasted availability of a unit
    """

    def __init__(
        self,
        index: ForecastIndex,
        market_prices: dict[str, ForecastSeries] = None,
        residual_load: dict[str, ForecastSeries] = None,
        availability: ForecastSeries = 1,
    ):
        if not isinstance(index, FastIndex):
            index = FastIndex(start=index[0], end=index[-1], freq=pd.infer_freq(index))
        if market_prices is None:
            market_prices = {"EOM": 50}  # default value for tests
        if residual_load is None:
            residual_load = {}
        self.index: FastIndex = index
        self.availability: FastSeries = self._to_series(availability)
        self.price = self._dict_to_series(market_prices)
        self.residual_load = self._dict_to_series(residual_load)

    def _to_series(self, item: ForecastSeries) -> FastSeries:
        if isinstance(item, FastSeries):
            return item
        return FastSeries(index=self.index, value=item)

    def _dict_to_series(self, d: dict[str, ForecastSeries]) -> dict[str, FastSeries]:
        result: dict[str, FastSeries] = {}
        for key, value in d.items():
            result[key] = self._to_series(value)
        return result


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
        market_prices: dict[str, ForecastSeries] = None,
        demand: ForecastSeries = -100,
        residual_load: dict[str, ForecastSeries] = None,
        availability: ForecastSeries = 1,
    ):
        super().__init__(index, market_prices, residual_load, availability)
        self.demand = self._to_series(demand)
        if any(self.demand > 0):
            raise ValueError(f"{demand=} must be negative")


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
        market_prices: dict[str, ForecastSeries] = None,
        fuel_prices=dict[str, ForecastSeries],
        availability: ForecastSeries = 1,
        congestion_signal: ForecastSeries = 0,
        residual_load: dict[str, ForecastSeries] = None,
        renewable_utilisation_signal: ForecastSeries = 0,
        electricity_price: ForecastSeries = None,
    ):
        super().__init__(index, market_prices, residual_load, availability)
        self.fuel_prices = self._dict_to_series(fuel_prices)
        self.congestion_signal = self._to_series(congestion_signal)
        self.electricity_price = self._to_series(electricity_price)
        self.renewable_utilisation_signal = self._to_series(
            renewable_utilisation_signal
        )

    def get_price(self, fuel: str) -> FastSeries:
        if fuel not in self.fuel_prices:
            return self._to_series(0)
        return self.fuel_prices[fuel]


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
        market_prices: dict[str, ForecastSeries] = None,
        residual_load: dict[str, ForecastSeries] = None,
        availability: ForecastSeries = 1,
    ):
        super().__init__(index, market_prices, residual_load, availability)
        if fuel_prices is None:
            fuel_prices = {}
        self.fuel_prices = self._dict_to_series(fuel_prices)

    def get_price(self, fuel: str) -> FastSeries:
        if fuel not in self.fuel_prices:
            return self._to_series(0)
        return self.fuel_prices[fuel]


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
        electricity_price: ForecastSeries,
        fuel_prices: dict[str, ForecastSeries],
        demand: ForecastSeries = 0,
        electricity_price_flex: ForecastSeries = 0,
        thermal_demand: ForecastSeries = 0,
        congestion_signal: ForecastSeries = 0,
        renewable_utilisation_signal: ForecastSeries = 0,
        thermal_storage_schedule: ForecastSeries = 0,
        residual_load: dict[str, ForecastSeries] = None,
        availability: ForecastSeries = 1,
        market_prices: dict[str, ForecastSeries] = None,
    ):
        super().__init__(index, market_prices, residual_load, availability)
        self.demand = demand
        self.electricity_price = self._to_series(electricity_price)
        self.electricity_price_flex = self._to_series(electricity_price_flex)
        self.fuel_prices = self._dict_to_series(fuel_prices)
        self.thermal_demand = self._to_series(thermal_demand)
        self.congestion_signal = self._to_series(congestion_signal)
        self.renewable_utilisation_signal = self._to_series(
            renewable_utilisation_signal
        )
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
        electricity_price: ForecastSeries = None,
        residual_load: dict[str, ForecastSeries] = None,
        availability: ForecastSeries = 1,
        market_prices: dict[str, ForecastSeries] = None,
        load_profile: ForecastSeries = 0,
    ):
        super().__init__(index, market_prices, residual_load, availability)
        self.fuel_prices = self._dict_to_series(fuel_prices)
        self.heat_demand = self._to_series(heat_demand)
        self.ev_load_profile = self._to_series(ev_load_profile)
        self.battery_load_profile = self._to_series(battery_load_profile)
        self.pv_profile = self._to_series(pv_profile)
        self.load_profile = self._to_series(load_profile)
        self.electricity_price = self._to_series(electricity_price)

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
        electricity_price: ForecastSeries,
        hydrogen_demand: ForecastSeries,
        seasonal_storage_schedule: ForecastSeries = 0,
        residual_load: dict[str, ForecastSeries] = None,
        availability: ForecastSeries = 1,
        market_prices: dict[str, ForecastSeries] = None,
    ):
        super().__init__(index, market_prices, residual_load, availability)
        self.electricity_price = self._to_series(electricity_price)
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
        residual_load: dict[str, ForecastSeries] = None,
        availability: ForecastSeries = 1,
        market_prices: dict[str, ForecastSeries] = None,
    ):
        super().__init__(index, market_prices, residual_load, availability)
        self.volume_import = self._to_series(volume_import)
        self.volume_export = self._to_series(volume_export)
