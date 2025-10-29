# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd

from assume.common.fast_pandas import FastIndex, FastSeries

type ForecastIndex = FastIndex | pd.DatetimeIndex | pd.Series
type ForecastSeries = FastSeries | list | float | pd.Series


class UnitForecaster:
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


class DemandForecaster(UnitForecaster):
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
    def __init__(
        self,
        index: ForecastIndex,
        market_prices: dict[str, ForecastSeries] = None,
        fuel_prices=dict[str, ForecastSeries],
        availability: ForecastSeries = 1,
        congestion_signal: ForecastSeries = 0,
        residual_load: dict[str, ForecastSeries] = None,
        renewable_utilisation_signal: ForecastSeries = 0,
    ):
        super().__init__(index, market_prices, residual_load, availability)
        self.fuel_prices = self._dict_to_series(fuel_prices)
        self.congestion_signal = self._to_series(congestion_signal)
        self.renewable_utilisation_signal = self._to_series(
            renewable_utilisation_signal
        )

    def get_price(self, fuel: str) -> FastSeries:
        if fuel not in self.fuel_prices:
            return self._to_series(0)
        return self.fuel_prices[fuel]


class PowerplantForecaster(UnitForecaster):
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
    def __init__(
        self,
        index: ForecastIndex,
        demand: ForecastSeries,
        electricity_price: ForecastSeries,
        electricity_price_flex: ForecastSeries,
        fuel_prices: dict[str, ForecastSeries],
        thermal_demand: ForecastSeries,
        congestion_signal: ForecastSeries,
        renewable_utilisation_signal: ForecastSeries,
        thermal_storage_schedule: ForecastSeries,
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
    def __init__(
        self,
        index: ForecastIndex,
        fuel_prices: dict[str, ForecastSeries],
        heat_demand: ForecastSeries,
        ev_load_profile: ForecastSeries,
        battery_load_profile: ForecastSeries,
        pv_profile: ForecastSeries,
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

    def get_price(self, fuel: str) -> FastSeries:
        if fuel not in self.fuel_prices:
            return self._to_series(0)
        return self.fuel_prices[fuel]


class HydrogenForecaster(UnitForecaster):
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
        self.seasonal_storage_schedule = self._dict_to_series(seasonal_storage_schedule)


class ExchangeForecaster(UnitForecaster):
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
