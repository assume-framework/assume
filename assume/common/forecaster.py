# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypeAlias

import pandas as pd

from assume.common.fast_pandas import FastIndex, FastSeries
from assume.common.forecast_algorithms import (
    forecast_algorithms,
    forecast_preprocess_algorithms,
    forecast_update_algorithms,
)
from assume.common.market_objects import MarketConfig

if TYPE_CHECKING:
    from assume.common.base import BaseUnit  # , BaseForecaster

ForecastIndex: TypeAlias = FastIndex | pd.DatetimeIndex | pd.Series
ForecastSeries: TypeAlias = FastSeries | list | float | pd.Series

log = logging.getLogger(__name__)


class UnitForecaster:
    """
    A generalized forecaster to provide units with static input data, such as availability or market price predictions.

    Forecast data is typically provided as time series data, either loaded from a CSV file or calculated
    internally from imported data. This class implements the basic forecaster interface, which relies on three main
    methods to manage the lifecycle of forecasts:

    1. ``preprocess``: Prepares information/forecasts for the initialization and update steps.
    2. ``initialize``: Initializes all forecast timeseries. This should be called once all units are created.
    3. ``update``: Used to change forecasts during runtime. This shall be called during the bid calculation of
       bidding strategies.

    These methods call functions that can be specified in the ``config.yaml`` or the unit CSV files via
    the ``forecast_algorithms`` dictionary. The functions are specified using the pattern ``{prefix}_{forecast_metric}``,
    where ``prefix`` is ``preprocess``, ``update``, or empty for ``initialize``.

    Attributes:
        index (ForecastIndex): The time index for all forecast series in this unit.
        availability (ForecastSeries): Forecasted availability of the unit.
        forecast_algorithms (dict[str, str]): Map specifying the forecast algorithms to use. Keys generally follow
            the format ``{prefix}_{forecast_metric}`` mapping to an ``algorithm_id``.
        price (dict[str, ForecastSeries]): Map of ``market_id`` to forecasted prices. (Initialized from ``market_prices``)
        residual_load (dict[str, ForecastSeries]): Map of ``market_id`` to forecasted residual load.
        preprocess_information (dict): Dictionary storing information prepared during the ``preprocess`` step.
    """

    def __init__(
        self,
        index: ForecastIndex,
        market_prices: dict[str, ForecastSeries] = None,
        residual_load: dict[str, ForecastSeries] = None,
        availability: ForecastSeries = 1,
        forecast_algorithms: dict[str, str] = {},
    ):
        if not isinstance(index, FastIndex):
            index = FastIndex(start=index[0], end=index[-1], freq=pd.infer_freq(index))

        self.index: FastIndex = index
        self.availability: FastSeries = self._to_series(availability)
        self.forecast_algorithms = forecast_algorithms
        if market_prices is None:
            market_prices = {"EOM": 50}  # default value for tests
        if residual_load is None:
            residual_load = {}
        self.price: dict[str, ForecastSeries] = self._dict_to_series(market_prices)
        self.residual_load: dict[str, ForecastSeries] = self._dict_to_series(
            residual_load
        )
        self.preprocess_information = {}
        print(forecast_algorithms)

    def _to_series(self, item: ForecastSeries) -> FastSeries:
        if isinstance(item, FastSeries):
            return item
        return FastSeries(index=self.index, value=item)

    def _dict_to_series(self, d: dict[str, ForecastSeries]) -> dict[str, FastSeries]:
        result: dict[str, FastSeries] = {}
        for key, value in d.items():
            result[key] = self._to_series(value)
        return result

    def preprocess(
        self,
        units: list[BaseUnit],
        market_configs: list[MarketConfig],
        forecast_df: ForecastSeries = None,
        initializing_unit: BaseUnit = None,
    ):
        """
        Applies preprocessing algorithms to the given data.

        This method prepares information and intermediate forecasts required for the
        subsequent `initialize` and `update` steps. The results are stored in the
        `self.preprocess_information` dictionary.

        The specific preprocessing algorithms are configured in `self.forecast_algorithms`
        using the keys `preprocess_price` (default: `price_default`) and
        `preprocess_residual_load` (default: `residual_load_default`).

        Args:
            units (list[BaseUnit]): List of all units in the simulation.
            market_configs (list[MarketConfig]): List of available market configurations.
            forecast_df (ForecastSeries, optional): Dataframe containing explicitly provided forecasts.
            initializing_unit (BaseUnit, optional): The unit that is currently being initialized.
        """

        price_preprocess_algorithm_name = self.forecast_algorithms.get(
            "preprocess_price", "price_default"
        )
        price_preprocess_algorithm = forecast_preprocess_algorithms.get(
            price_preprocess_algorithm_name
        )
        self.preprocess_information["price"] = price_preprocess_algorithm(
            self.index, units, market_configs, forecast_df, initializing_unit
        )

        residual_load_preprocess_algorithm_name = self.forecast_algorithms.get(
            "preprocess_residual_load", "residual_load_default"
        )
        residual_load_preprocess_algorithm = forecast_preprocess_algorithms.get(
            residual_load_preprocess_algorithm_name
        )
        self.preprocess_information["residual_load"] = (
            residual_load_preprocess_algorithm(
                self.index, units, market_configs, forecast_df, initializing_unit
            )
        )

    def initialize(
        self,
        units: list[BaseUnit],
        market_configs: list[MarketConfig],
        forecast_df: ForecastSeries = None,
        initializing_unit: BaseUnit = None,
    ):
        """
        Initializes all forecast metric timeseries for the unit.

        This method should be called exactly once after all units in the simulation are created.
        It first delegates to `preprocess()` to prepare data, then computes the initial
        forecasts (e.g., `price` and `residual_load`).

        The specific forecast algorithms are configured in `self.forecast_algorithms`
        using the keys `price` (default: `price_naive_forecast`) and
        `residual_load` (default: `residual_load_naive_forecast`).

        Args:
            units (list[BaseUnit]): List of all units in the simulation.
            market_configs (list[MarketConfig]): List of available market configurations.
            forecast_df (ForecastSeries, optional): Dataframe containing explicitly provided forecasts.
            initializing_unit (BaseUnit, optional): The unit that is currently being initialized.
        """
        self.preprocess(units, market_configs, forecast_df, initializing_unit)

        # 1. Get price forecast
        price_forecast_algorithm_name = self.forecast_algorithms.get(
            "price", "price_naive_forecast"
        )
        price_forecast_algorithm = forecast_algorithms.get(
            price_forecast_algorithm_name
        )
        if price_forecast_algorithm is not None:  # None if one wants to keep forecasts
            self.price = price_forecast_algorithm(
                self.index,
                units,
                market_configs,
                forecast_df,
                self.preprocess_information["price"],
            )
            self.price = self._dict_to_series(self.price)

        # 2. Get residual load forecast
        residual_load_forecast_algorithm_name = self.forecast_algorithms.get(
            "residual_load", "residual_load_naive_forecast"
        )
        residual_load_forecast_algorithm = forecast_algorithms.get(
            residual_load_forecast_algorithm_name
        )
        if (
            residual_load_forecast_algorithm is not None
        ):  # None if one wants to keep forecasts
            self.residual_load = residual_load_forecast_algorithm(
                self.index,
                units,
                market_configs,
                forecast_df,
                self.preprocess_information["residual_load"],
            )
            self.residual_load = self._dict_to_series(self.residual_load)

    def update(self, *args, **kwargs):
        """
        Dynamically updates the forecast timeseries during runtime.

        This method is designed to be called periodically during the simulation (e.g., during
        the bid calculation of bidding strategies) to change the unit's forecasts based on
        sliding horizons or new runtime data.

        The specific update algorithms are configured in `self.forecast_algorithms`
        using the keys `update_price` (default: `price_default`) and
        `update_residual_load` (default: `residual_load_default`).

        Args:
            *args: Variable length argument list passed to the underlying update algorithms.
            **kwargs: Arbitrary keyword arguments passed to the underlying update algorithms.
        """

        price_update_algorithm_name = self.forecast_algorithms.get(
            "update_price", "price_default"
        )
        price_update_algorithm = forecast_update_algorithms.get(
            price_update_algorithm_name
        )
        self.price = price_update_algorithm(
            self.price, self.preprocess_information["price"], *args, **kwargs
        )
        self.price = self._dict_to_series(self.price)

        residual_load_update_algorithm_name = self.forecast_algorithms.get(
            "update_residual_load", "residual_load_default"
        )
        residual_load_update_algorithm = forecast_update_algorithms.get(
            residual_load_update_algorithm_name
        )
        self.residual_load = residual_load_update_algorithm(
            self.residual_load,
            self.preprocess_information["residual_load"],
            *args,
            **kwargs,
        )
        self.residual_load = self._dict_to_series(self.residual_load)


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
        forecast_algorithms: dict[str, str] = {},
    ):
        # super().__init__(index, market_prices, residual_load, availability)
        super().__init__(
            index=index,
            availability=availability,
            forecast_algorithms=forecast_algorithms,
            market_prices=market_prices,
            residual_load=residual_load,
        )
        self.demand = self._to_series(demand)
        if any(self.demand > 0):
            raise ValueError(f"{demand=} must be negative")


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
        forecast_algorithms: dict[str, str] = {},
    ):
        # super().__init__(index, market_prices, residual_load, availability)
        super().__init__(
            index=index,
            availability=availability,
            forecast_algorithms=forecast_algorithms,
            market_prices=market_prices,
            residual_load=residual_load,
        )
        if fuel_prices is None:
            fuel_prices = {}
        self.fuel_prices = self._dict_to_series(fuel_prices)

    def get_price(self, fuel: str) -> FastSeries:
        if fuel not in self.fuel_prices:
            return self._to_series(0)
        return self.fuel_prices[fuel]


class DsmUnitForecaster(UnitForecaster):
    """
    A forecaster for demand side management (DSM) units.

    This class extends the `UnitForecaster` to include metrics specifically relevant to
    DSM units, such as congestion signals, renewable utilisation signals, and electricity prices.
    It overrides the base interface (`preprocess`, `initialize`, `update`) to prepare and
    maintain forecasts for these additional metrics.

    Attributes:
        index (ForecastIndex): The index of all forecast series in this unit.
        market_prices (dict[str, ForecastSeries]): Map of market_id -> forecasted prices.
        residual_load (dict[str, ForecastSeries]): Map of market_id -> forecasted residual load.
        availability (ForecastSeries): Forecasted availability of a unit.
        congestion_signal (ForecastSeries): Forecasted congestion signal.
        renewable_utilisation_signal (ForecastSeries): Forecasted renewable utilisation signal.
        electricity_price (ForecastSeries): Forecasted electricity price.
        forecast_algorithms (dict[str, str]): Map specifying the forecast algorithms to use.
    """

    def __init__(
        self,
        index: ForecastIndex,
        market_prices: dict[str, ForecastSeries] = None,
        residual_load: dict[str, ForecastSeries] = None,
        availability: ForecastSeries = 1,
        forecast_algorithms: dict[str, str] = {},
        congestion_signal: ForecastSeries = 0.0,
        renewable_utilisation_signal: ForecastSeries = 0.0,
        electricity_price: ForecastSeries = 0.0,
    ):
        super().__init__(
            index=index,
            availability=availability,
            forecast_algorithms=forecast_algorithms,
            market_prices=market_prices,
            residual_load=residual_load,
        )
        self.congestion_signal = self._to_series(congestion_signal)
        self.electricity_price = self._to_series(electricity_price)
        self.renewable_utilisation_signal = self._to_series(
            renewable_utilisation_signal
        )

    def preprocess(
        self,
        units: list[BaseUnit],
        market_configs: list[MarketConfig],
        forecast_df: ForecastSeries = None,
        initializing_unit: BaseUnit = None,
    ):
        """
        Applies preprocessing to the given data for congestion signal and renewable utilization.

        Preprocess information is stored in the `self.preprocess_information` dictionary.

        The specific preprocessing algorithms are configured in `self.forecast_algorithms`
        using the keys `preprocess_congestion_signal` (default: `congestion_signal_default`) and
        `preprocess_renewable_utilisation` (default: `renewable_utilisation_default`).

        Args:
            units (list[BaseUnit]): List of all units in the simulation.
            market_configs (list[MarketConfig]): List of available market configurations.
            forecast_df (ForecastSeries, optional): Dataframe containing explicitly provided forecasts.
            initializing_unit (BaseUnit, optional): The unit that is currently being initialized.
        """

        congestion_signal_preprocess_algorithm_name = self.forecast_algorithms.get(
            "preprocess_congestion_signal", "congestion_signal_default"
        )
        congestion_signal_preprocess_algorithm = forecast_preprocess_algorithms.get(
            congestion_signal_preprocess_algorithm_name
        )
        self.preprocess_information["congestion_signal"] = (
            congestion_signal_preprocess_algorithm(
                self.index,
                units,
                market_configs,
                forecast_df,
                initializing_unit,
            )
        )

        renewable_utilisation_preprocess_algorithm_name = self.forecast_algorithms.get(
            "preprocess_renewable_utilisation", "renewable_utilisation_default"
        )
        renewable_utilisation_preprocess_algorithm = forecast_preprocess_algorithms.get(
            renewable_utilisation_preprocess_algorithm_name
        )
        self.preprocess_information["renewable_utilisation"] = (
            renewable_utilisation_preprocess_algorithm(
                self.index, units, market_configs, forecast_df, initializing_unit
            )
        )

    def initialize(
        self,
        units: list[BaseUnit],
        market_configs: list[MarketConfig],
        forecast_df: ForecastSeries = None,
        initializing_unit: BaseUnit = None,
    ):
        """
        Initializes forecast metric timeseries specific to DSM units.

        This calls the parent class initialization and additionally initializes the
        `electricity_price`, `congestion_signal`, and `renewable_utilisation_signal`.

        The specific forecast algorithms for the new metrics are configured in
        `self.forecast_algorithms` using the keys `congestion_signal`
        (default: `congestion_signal_naive_forecast`) and `renewable_utilisation`
        (default: `renewable_utilisation_naive_forecast`).

        Args:
            units (list[BaseUnit]): List of all units in the simulation.
            market_configs (list[MarketConfig]): List of available market configurations.
            forecast_df (ForecastSeries, optional): Dataframe containing explicitly provided forecasts.
            initializing_unit (BaseUnit, optional): The unit that is currently being initialized.
        """
        # 1. preprocess of price and residual load forecast
        super().preprocess(units, market_configs, forecast_df, initializing_unit)

        # 2. Own preprocess and 3. Initialization and of price and load forecasts
        super().initialize(
            units,
            market_configs,
            forecast_df,
            initializing_unit,
        )

        # 4. Get electricity price forecast
        # TODO how to handle other markets?
        self.electricity_price = self.price.get("EOM")

        # 5. Get congestion signal forecast
        congestion_signal_forecast_algorithm_name = self.forecast_algorithms.get(
            "congestion_signal", "congestion_signal_naive_forecast"
        )
        congestion_signal_forecast_algorithm = forecast_algorithms.get(
            congestion_signal_forecast_algorithm_name
        )
        if (
            congestion_signal_forecast_algorithm is not None
        ):  # None if one wants to keep forecasts
            self.congestion_signal = congestion_signal_forecast_algorithm(
                self.index,
                units,
                market_configs,
                forecast_df,
                self.preprocess_information["congestion_signal"],
            )
            self.congestion_signal = self._dict_to_series(self.congestion_signal)

        # 6. Get renewable utilisation forecast
        renewable_utilisation_forecast_algorithm_name = self.forecast_algorithms.get(
            "renewable_utilisation", "renewable_utilisation_naive_forecast"
        )
        renewable_utilisation_forecast_algorithm = forecast_algorithms.get(
            renewable_utilisation_forecast_algorithm_name
        )
        if (
            renewable_utilisation_forecast_algorithm is not None
        ):  # None if one wants to keep forecasts
            self.renewable_utilisation_signal = (
                renewable_utilisation_forecast_algorithm(
                    self.index,
                    units,
                    market_configs,
                    forecast_df,
                    self.preprocess_information["congestion_signal"],
                )
            )
            self.renewable_utilisation_signal = self._dict_to_series(
                self.renewable_utilisation_signal
            )

    def update(self, *args, **kwargs):
        """
        Dynamically updates the DSM-specific forecast timeseries during runtime.

        Calls the parent class update method and additionally updates the
        `congestion_signal` and `renewable_utilisation_signal` forecasts.

        The specific update algorithms are configured in `self.forecast_algorithms`
        using the keys `update_congestion_signal` (default: `congestion_signal_default`)
        and `update_renewable_utilisation` (default: `renewable_utilisation_default`).

        Args:
            *args: Variable length argument list passed to the underlying update algorithms.
            **kwargs: Arbitrary keyword arguments passed to the underlying update algorithms.
        """

        super().update(*args, **kwargs)

        congestion_signal_update_algorithm_name = self.forecast_algorithms.get(
            "update_congestion_signal", "congestion_signal_default"
        )
        congestion_signal_update_algorithm = forecast_update_algorithms.get(
            congestion_signal_update_algorithm_name
        )
        self.congestion_signal = congestion_signal_update_algorithm(
            self.congestion_signal,
            self.preprocess_information["congestion_signal"],
            *args,
            **kwargs,
        )
        self.congestion_signal = self._dict_to_series(self.congestion_signal)

        renewable_utilisation_update_algorithm_name = self.forecast_algorithms.get(
            "update_renewable_utilisation", "renewable_utilisation_default"
        )
        renewable_utilisation_update_algorithm = forecast_update_algorithms.get(
            renewable_utilisation_update_algorithm_name
        )
        self.renewable_utilisation_signal = renewable_utilisation_update_algorithm(
            self.renewable_utilisation_signal,
            self.preprocess_information["renewable_utilisation"],
            *args,
            **kwargs,
        )
        self.renewable_utilisation_signal = self._dict_to_series(
            self.renewable_utilisation_signal
        )


class SteelplantForecaster(DsmUnitForecaster):
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
        fuel_prices: dict[str, ForecastSeries],
        market_prices: dict[str, ForecastSeries] = None,
        residual_load: dict[str, ForecastSeries] = None,
        availability: ForecastSeries = 1,
        forecast_algorithms: dict[str, str] = {},
        congestion_signal: ForecastSeries = 0.0,
        renewable_utilisation_signal: ForecastSeries = 0.0,
        electricity_price: ForecastSeries = None,
    ):
        # super().__init__(index, market_prices, residual_load, availability)
        super().__init__(
            index=index,
            availability=availability,
            forecast_algorithms=forecast_algorithms,
            market_prices=market_prices,
            residual_load=residual_load,
            congestion_signal=congestion_signal,
            renewable_utilisation_signal=renewable_utilisation_signal,
            electricity_price=electricity_price,
        )
        self.fuel_prices = self._dict_to_series(fuel_prices)

    def get_price(self, fuel: str) -> FastSeries:
        if fuel not in self.fuel_prices:
            return self._to_series(0)
        return self.fuel_prices[fuel]

    def initialize(
        self,
        units: list[BaseUnit],
        market_configs: list[MarketConfig],
        forecast_df: ForecastSeries = None,
        initializing_unit: BaseUnit = None,
    ):
        super().initialize(
            units,
            market_configs,
            forecast_df,
            initializing_unit,
        )

        initializing_unit.electricity_price = self.electricity_price
        initializing_unit.congestion_signal = self.congestion_signal
        initializing_unit.renewable_utilisation_signal = (
            self.renewable_utilisation_signal
        )

        initializing_unit.setup_model()


class SteamgenerationForecaster(DsmUnitForecaster):
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
        fuel_prices: dict[str, ForecastSeries],
        demand: ForecastSeries = 0,
        electricity_price_flex: ForecastSeries = 0,
        thermal_demand: ForecastSeries = 0,
        thermal_storage_schedule: ForecastSeries = 0,
        availability: ForecastSeries = 1,
        forecast_algorithms: dict[str, str] = {},
        market_prices: dict[str, ForecastSeries] = None,
        residual_load: dict[str, ForecastSeries] = None,
        congestion_signal: ForecastSeries = 0.0,
        renewable_utilisation_signal: ForecastSeries = 0.0,
        electricity_price: ForecastSeries = 0.0,
    ):
        super().__init__(
            index=index,
            availability=availability,
            forecast_algorithms=forecast_algorithms,
            market_prices=market_prices,
            residual_load=residual_load,
            congestion_signal=congestion_signal,
            renewable_utilisation_signal=renewable_utilisation_signal,
            electricity_price=electricity_price,
        )
        self.demand = demand
        self.electricity_price_flex = self._to_series(electricity_price_flex)
        self.fuel_prices = self._dict_to_series(fuel_prices)
        self.thermal_demand = self._to_series(thermal_demand)
        self.thermal_storage_schedule = self._to_series(thermal_storage_schedule)

    def get_price(self, fuel: str) -> FastSeries:
        if fuel not in self.fuel_prices:
            return self._to_series(0)
        return self.fuel_prices[fuel]

    def initialize(
        self,
        units: list[BaseUnit],
        market_configs: list[MarketConfig],
        forecast_df: ForecastSeries = None,
        initializing_unit: BaseUnit = None,
    ):
        super().initialize(
            units,
            market_configs,
            forecast_df,
            initializing_unit,
        )

        initializing_unit.electricity_price = self.electricity_price
        initializing_unit.congestion_signal = self.congestion_signal
        initializing_unit.renewable_utilisation_signal = (
            self.renewable_utilisation_signal
        )

        initializing_unit.setup_model()


class BuildingForecaster(DsmUnitForecaster):
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
        forecast_algorithms: dict[str, str] = {},
        market_prices: dict[str, ForecastSeries] = None,
        residual_load: dict[str, ForecastSeries] = None,
        congestion_signal: ForecastSeries = 0.0,
        renewable_utilisation_signal: ForecastSeries = 0.0,
        electricity_price: ForecastSeries = 0.0,
        availability: ForecastSeries = 1,
        load_profile: ForecastSeries = 0,
    ):
        super().__init__(
            index=index,
            availability=availability,
            forecast_algorithms=forecast_algorithms,
            market_prices=market_prices,
            residual_load=residual_load,
            congestion_signal=congestion_signal,
            renewable_utilisation_signal=renewable_utilisation_signal,
            electricity_price=electricity_price,
        )
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

    def initialize(
        self,
        units: list[BaseUnit],
        market_configs: list[MarketConfig],
        forecast_df: ForecastSeries = None,
        initializing_unit: BaseUnit = None,
    ):
        super().initialize(
            units,
            market_configs,
            forecast_df,
            initializing_unit,
        )

        initializing_unit.electricity_price = self.electricity_price
        # initializing_unit.congestion_signal = self.congestion_signal
        # initializing_unit.renewable_utilisation_signal = self.renewable_utilisation_signal

        initializing_unit.setup_model(presolve=True)


class HydrogenForecaster(DsmUnitForecaster):
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
        hydrogen_demand: ForecastSeries,
        forecast_algorithms: dict[str, str] = {},
        seasonal_storage_schedule: ForecastSeries = 0,
        availability: ForecastSeries = 1,
        market_prices: dict[str, ForecastSeries] = None,
        residual_load: dict[str, ForecastSeries] = None,
        electricity_price: ForecastSeries = 0.0,
    ):
        # super().__init__(index, market_prices, residual_load, availability)
        super().__init__(
            index=index,
            availability=availability,
            forecast_algorithms=forecast_algorithms,
            market_prices=market_prices,
            residual_load=residual_load,
            electricity_price=electricity_price,
        )
        # self.electricity_price = self._to_series(electricity_price)
        self.hydrogen_demand = self._to_series(hydrogen_demand)
        self.seasonal_storage_schedule = self._to_series(seasonal_storage_schedule)

    def initialize(
        self,
        units: list[BaseUnit],
        market_configs: list[MarketConfig],
        forecast_df: ForecastSeries = None,
        initializing_unit: BaseUnit = None,
    ):
        super().initialize(
            units,
            market_configs,
            forecast_df,
            initializing_unit,
        )

        initializing_unit.electricity_price = self.electricity_price
        # initializing_unit.congestion_signal = self.congestion_signal
        # initializing_unit.renewable_utilisation_signal = self.renewable_utilisation_signal

        initializing_unit.setup_model()


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
        forecast_algorithms: dict[str, str] = {},
        volume_import: ForecastSeries = 0,
        volume_export: ForecastSeries = 0,
        residual_load: dict[str, ForecastSeries] = None,
        availability: ForecastSeries = 1,
        market_prices: dict[str, ForecastSeries] = None,
    ):
        # super().__init__(index, market_prices, residual_load, availability)
        super().__init__(
            index=index,
            availability=availability,
            forecast_algorithms=forecast_algorithms,
            market_prices=market_prices,
            residual_load=residual_load,
        )
        self.volume_import = self._to_series(volume_import)
        self.volume_export = self._to_series(volume_export)
