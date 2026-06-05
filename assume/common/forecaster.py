# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias

import pandas as pd

from assume.common.exceptions import ValidationError
from assume.common.fast_pandas import FastIndex, FastSeries
from assume.common.market_objects import MarketConfig

if TYPE_CHECKING:
    from assume.common.base import BaseUnit  # , BaseForecaster

ForecastIndex: TypeAlias = FastIndex | pd.DatetimeIndex | pd.Series
ForecastSeries: TypeAlias = FastSeries | list | float | pd.Series

log = logging.getLogger(__name__)


def _ensure_not_none(
    df: pd.DataFrame | None, index: ForecastIndex, check_index=False
) -> pd.DataFrame:
    """Return *df* as-is or create an empty DataFrame with the given *index* if *df* is None."""
    if isinstance(index, FastIndex):
        index = index.as_datetimeindex()

    if df is None:
        return pd.DataFrame(index=index)
    if check_index and index.freq != df.index.inferred_freq:
        raise ValueError("Forecast frequency does not match index frequency.")
    return df


def calculate_base_forecasts(
    index: ForecastIndex,
    units: list[BaseUnit],
    market_configs: list[MarketConfig],
    forecast_algorithm: Callable,
    forecast_df: ForecastSeries = None,
    preprocess_information=None,
    prefix="",
) -> dict[str, ForecastSeries]:
    """Compute per-market forecasts for a single metric (e.g. price, residual_load).

    For each energy market in *market_configs*, returns the corresponding column
    from *forecast_df* if it exists (keyed as ``{prefix}_{market_id}``), otherwise
    falls back to *forecast_algorithm* to calculate the forecast.

    Returns:
        dict[str, ForecastSeries]: Map of ``market_id`` to forecast series.
    """
    # print(prefix, hash(prefix), hash(index), hash(market_configs), hash(units), hash(preprocess_information))
    forecast_df = _ensure_not_none(forecast_df, index)

    forecast: dict[str, pd.Series] = {}

    for config in market_configs:
        market_id = config.market_id
        if config.product_type != "energy":
            log.warning(
                f"Forecast could not be calculated for {market_id}. It can only be calculated for energy-only markets for now."
            )
            continue

        # NOTE: if given forecast_df will currently always prevent other forecast calculations
        forecast[market_id] = forecast_df.get(f"{prefix}_{market_id}")
        if forecast[market_id] is not None:
            # go next if forecast existing
            continue

        forecast[market_id] = forecast_algorithm(
            index,
            units,
            config,
            # forecast_df,
            preprocess_information,
        )
    return forecast


def calculate_node_wise_forecasts(
    index: ForecastIndex,
    units: list[BaseUnit],
    market_configs: list[MarketConfig],
    forecast_algorithm: Callable,
    forecast_df: ForecastSeries = None,
    preprocess_information=None,
    prefix="",
):
    """Compute per-node forecasts for a spatial metric (e.g. congestion_signal, renewable_utilisation).

    Runs *forecast_algorithm* to produce forecasts keyed by node, then overwrites
    individual entries with columns from *forecast_df* (keyed as ``{node}_{prefix}``)
    when both the provided and calculated forecast exist for the same node.

    Returns:
        dict[str, ForecastSeries]: Map of ``{node}_{prefix_alias}`` to forecast series.
    """
    forecast_df = _ensure_not_none(forecast_df, index)

    forecast = forecast_algorithm(
        index,
        units,
        market_configs,
        preprocess_information,
    )

    # FIXME: make prefix coherent to the forecast!!!!
    prefix_alias = "congestion_severity" if prefix == "congestion_signal" else prefix

    buses = (
        list(market_configs)[0]
        .param_dict.get("grid_data", {})
        .get("buses", pd.DataFrame())
    )

    # check if forecast exists in forecast_df for each node
    # if calculated forecast also expects this node: overwrite else ignore
    # NOTE: currently forecast_df will always overwrite calculated forecasts
    for node in buses.index:
        if (
            forecast_df.get(f"{node}_{prefix}") is not None
            and forecast.get(f"{node}_{prefix_alias}") is not None
        ):
            forecast[f"{node}_{prefix_alias}"] = forecast_df[f"{node}_{prefix}"]

    # Also check if there is an aggregated forecasts over all nodes
    if (
        forecast_df.get(f"all_nodes_{prefix}") is not None
        and forecast.get(f"all_nodes_{prefix_alias}") is not None
    ):
        forecast[f"all_nodes_{prefix_alias}"] = forecast_df[f"all_nodes_{prefix}"]

    return forecast


class UnitForecaster:
    """A generalized forecaster to provide units with static input data, such as availability or market price predictions.

    Forecast data is typically provided as time series data, either loaded from a CSV file or calculated
    internally from imported data. This class implements the basic forecaster interface, which relies on three
    lifecycle methods:

    1. ``preprocess``: Prepares intermediate information for the initialization and update steps.
    2. ``initialize``: Computes all forecast timeseries. Should be called once after all units are created.
    3. ``update``: Revises forecasts during runtime (e.g. during bid calculation of bidding strategies).

    Algorithm resolution:
        Each lifecycle method looks up which algorithm to run via two dictionaries:

        - ``forecast_algorithms`` maps a **key** (e.g. ``"price"``, ``"update_price"``) to an
          **algorithm_id** (e.g. ``"price_naive_forecast"``). Keys follow the pattern
          ``{prefix}_{forecast_metric}`` where *prefix* is ``preprocess``, ``update``, or empty
          for ``initialize``.
        - ``_registries`` (populated from ``forecast_registries``) maps each **algorithm_id** to
          the actual callable. It contains three sub-dicts: ``"init"``, ``"preprocess"``, and
          ``"update"``, typically provided by ``get_forecast_registries()`` in ``forecast_algorithms.py``.

    Attributes:
        index (ForecastIndex): The time index for all forecast series in this unit.
        availability (ForecastSeries): Forecasted availability of the unit.
        forecast_algorithms (dict[str, str]): Map of ``{prefix}_{metric}`` keys to algorithm IDs.
        price (dict[str, ForecastSeries]): Map of ``market_id`` to forecasted prices (initialized from ``market_prices``).
        residual_load (dict[str, ForecastSeries]): Map of ``market_id`` to forecasted residual load.
        preprocess_information (dict): Intermediate data prepared during the ``preprocess`` step,
            keyed by metric name (e.g. ``"price"``, ``"residual_load"``).
    """

    def __init__(
        self,
        index: ForecastIndex,
        market_prices: dict[str, ForecastSeries] = None,
        residual_load: dict[str, ForecastSeries] = None,
        availability: ForecastSeries = 1,
        forecast_algorithms: dict[str, str] = {},
        forecast_registries: dict[str, dict] = None,
    ):
        if not isinstance(index, FastIndex):
            index = FastIndex(start=index[0], end=index[-1], freq=pd.infer_freq(index))

        self.index: FastIndex = index
        self.availability: FastSeries = self._to_series(availability)
        if any(self.availability < 0) or any(self.availability > 1):
            raise ValidationError(
                message="Availability must be between 0 and 1", field="availability"
            )
        self.forecast_algorithms = forecast_algorithms
        self._registries = forecast_registries
        if market_prices is None:
            market_prices = {"EOM": 50}  # default value for tests
        if residual_load is None:
            residual_load = {}
        self.price: dict[str, ForecastSeries] = self._dict_to_series(market_prices)
        self.residual_load: dict[str, ForecastSeries] = self._dict_to_series(
            residual_load
        )
        self.preprocess_information = {}

    def _to_series(self, item: ForecastSeries) -> FastSeries:
        """Wrap *item* in a ``FastSeries`` aligned to ``self.index`` (no-op if already one)."""
        if isinstance(item, FastSeries):
            return item
        return FastSeries(index=self.index, value=item)

    def _dict_to_series(self, d: dict[str, ForecastSeries]) -> dict[str, FastSeries]:
        """Apply ``_to_series`` to every value in *d*."""
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
        """Prepare intermediate data needed by ``initialize`` and ``update``.

        Runs the preprocess algorithms for *price* and *residual_load* and stores results
        in ``self.preprocess_information``.

        Algorithm keys (defaults):
            - ``preprocess_price`` (``price_default``)
            - ``preprocess_residual_load`` (``residual_load_default``)

        Args:
            units (list[BaseUnit]): All units in the simulation.
            market_configs (list[MarketConfig]): Available market configurations.
            forecast_df (ForecastSeries, optional): Explicitly provided forecasts (columns override calculated ones).
            initializing_unit (BaseUnit, optional): The unit currently being initialized.
        """

        price_preprocess_algorithm_name = self.forecast_algorithms.get(
            "preprocess_price", "price_default"
        )
        price_preprocess_algorithm = self._registries["preprocess"].get(
            price_preprocess_algorithm_name
        )
        self.preprocess_information["price"] = price_preprocess_algorithm(
            self.index, units, market_configs, forecast_df, initializing_unit
        )

        residual_load_preprocess_algorithm_name = self.forecast_algorithms.get(
            "preprocess_residual_load", "residual_load_default"
        )
        residual_load_preprocess_algorithm = self._registries["preprocess"].get(
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
        """Compute all forecast timeseries for the unit. Call once after all units are created.

        Delegates to ``preprocess()`` first, then computes *price* and *residual_load*
        forecasts via ``calculate_base_forecasts``. If an algorithm resolves to ``None``
        (e.g. ``price_keep_given``), the existing forecast is kept unchanged.

        Algorithm keys (defaults):
            - ``price`` (``price_naive_forecast``)
            - ``residual_load`` (``residual_load_naive_forecast``)

        Args:
            units (list[BaseUnit]): All units in the simulation.
            market_configs (list[MarketConfig]): Available market configurations.
            forecast_df (ForecastSeries, optional): Explicitly provided forecasts (columns override calculated ones).
            initializing_unit (BaseUnit, optional): The unit currently being initialized.
        """
        self.preprocess(units, market_configs, forecast_df, initializing_unit)

        # 1. Get price forecast
        price_forecast_algorithm_name = self.forecast_algorithms.get(
            "price", "price_naive_forecast"
        )
        price_forecast_algorithm = self._registries["init"].get(
            price_forecast_algorithm_name
        )
        if price_forecast_algorithm is not None:  # None means keep existing forecast
            self.price = calculate_base_forecasts(
                self.index,
                units,
                market_configs,
                price_forecast_algorithm,
                forecast_df,
                self.preprocess_information["price"],
                prefix="price",
            )
            self.price = self._dict_to_series(self.price)

        # 2. Get residual load forecast
        residual_load_forecast_algorithm_name = self.forecast_algorithms.get(
            "residual_load", "residual_load_naive_forecast"
        )
        residual_load_forecast_algorithm = self._registries["init"].get(
            residual_load_forecast_algorithm_name
        )
        if (
            residual_load_forecast_algorithm is not None
        ):  # None means keep existing forecast
            self.residual_load = calculate_base_forecasts(
                self.index,
                units,
                market_configs,
                residual_load_forecast_algorithm,
                forecast_df,
                self.preprocess_information["residual_load"],
                prefix="residual_load",
            )
            self.residual_load = self._dict_to_series(self.residual_load)

    def update(self, *args, **kwargs):
        """Revise forecast timeseries during runtime (e.g. during bid calculation).

        Called periodically to adjust forecasts based on sliding horizons or new data.

        Algorithm keys (defaults):
            - ``update_price`` (``price_default``)
            - ``update_residual_load`` (``residual_load_default``)

        Args:
            *args: Passed through to the underlying update algorithms.
            **kwargs: Passed through to the underlying update algorithms.
        """

        price_update_algorithm_name = self.forecast_algorithms.get(
            "update_price", "price_default"
        )
        price_update_algorithm = self._registries["update"].get(
            price_update_algorithm_name
        )
        self.price = price_update_algorithm(
            self.price, self.preprocess_information["price"], *args, **kwargs
        )
        self.price = self._dict_to_series(self.price)

        residual_load_update_algorithm_name = self.forecast_algorithms.get(
            "update_residual_load", "residual_load_default"
        )
        residual_load_update_algorithm = self._registries["update"].get(
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
    """A generic forecaster that sets arbitrary keyword arguments as attributes.

    Used primarily for CSV-imported forecasts. Any ``pd.Series`` values are automatically
    converted to ``FastSeries``.

    Args:
        index (ForecastIndex): The time index for all forecast series.
        **kwargs: Arbitrary forecast attributes to set on the instance.
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
    """Forecaster for demand units.

    Provides price, residual load, and availability forecasts (see :class:`UnitForecaster`)
    plus a demand timeseries specific to demand units.

    Attributes:
        demand (ForecastSeries): Forecasted demand (must be negative).
    """

    def __init__(
        self,
        index: ForecastIndex,
        market_prices: dict[str, ForecastSeries] = None,
        demand: ForecastSeries = -100,
        residual_load: dict[str, ForecastSeries] = None,
        availability: ForecastSeries = 1,
        forecast_algorithms: dict[str, str] = {},
        forecast_registries: dict[str, dict] = None,
    ):
        super().__init__(
            index=index,
            availability=availability,
            forecast_algorithms=forecast_algorithms,
            forecast_registries=forecast_registries,
            market_prices=market_prices,
            residual_load=residual_load,
        )
        self.demand = self._to_series(demand)
        if any(self.demand > 0):
            raise ValidationError(message="demand must be negative", field="demand")


class PowerplantForecaster(UnitForecaster):
    """Forecaster for powerplant units.

    Provides price, residual load, and availability forecasts (see :class:`UnitForecaster`)
    plus per-fuel-type price timeseries.

    Attributes:
        fuel_prices (dict[str, ForecastSeries]): Map of fuel type to forecasted fuel prices.
    """

    def __init__(
        self,
        index: ForecastIndex,
        fuel_prices: dict[str, ForecastSeries] = None,
        market_prices: dict[str, ForecastSeries] = None,
        residual_load: dict[str, ForecastSeries] = None,
        availability: ForecastSeries = 1,
        forecast_algorithms: dict[str, str] = {},
        forecast_registries: dict[str, dict] = None,
    ):
        super().__init__(
            index=index,
            availability=availability,
            forecast_algorithms=forecast_algorithms,
            forecast_registries=forecast_registries,
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
    """Forecaster for demand side management (DSM) units.

    Extends :class:`UnitForecaster` with metrics relevant to DSM units: congestion signals,
    renewable utilisation signals, and electricity prices. Overrides ``preprocess``,
    ``initialize``, and ``update`` to handle these additional forecasts via
    ``calculate_node_wise_forecasts``.

    Attributes:
        congestion_signal (ForecastSeries): Forecasted per-node congestion signal.
        renewable_utilisation_signal (ForecastSeries): Forecasted per-node renewable utilisation.
        electricity_price (ForecastSeries): Forecasted electricity price (derived from EOM price).
    """

    def __init__(
        self,
        index: ForecastIndex,
        market_prices: dict[str, ForecastSeries] = None,
        residual_load: dict[str, ForecastSeries] = None,
        availability: ForecastSeries = 1,
        forecast_algorithms: dict[str, str] = {},
        forecast_registries: dict[str, dict] = None,
        congestion_signal: ForecastSeries = 0.0,
        renewable_utilisation_signal: ForecastSeries = 0.0,
        electricity_price: ForecastSeries = 0.0,
    ):
        super().__init__(
            index=index,
            availability=availability,
            forecast_algorithms=forecast_algorithms,
            forecast_registries=forecast_registries,
            market_prices=market_prices,
            residual_load=residual_load,
        )

        # FIXME: currently default is series while calculations are dict of series
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
        """Prepare intermediate data for DSM-specific metrics (congestion signal, renewable utilisation).

        Results are stored in ``self.preprocess_information``.

        Algorithm keys (defaults):
            - ``preprocess_congestion_signal`` (``congestion_signal_default``)
            - ``preprocess_renewable_utilisation`` (``renewable_utilisation_default``)

        Args:
            units (list[BaseUnit]): All units in the simulation.
            market_configs (list[MarketConfig]): Available market configurations.
            forecast_df (ForecastSeries, optional): Explicitly provided forecasts.
            initializing_unit (BaseUnit, optional): The unit currently being initialized.
        """

        congestion_signal_preprocess_algorithm_name = self.forecast_algorithms.get(
            "preprocess_congestion_signal", "congestion_signal_default"
        )
        congestion_signal_preprocess_algorithm = self._registries["preprocess"].get(
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
        renewable_utilisation_preprocess_algorithm = self._registries["preprocess"].get(
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
        """Initialize all forecast timeseries including DSM-specific metrics.

        Calls parent preprocessing and initialization for price / residual load, then
        additionally computes ``electricity_price`` (from EOM price), ``congestion_signal``,
        and ``renewable_utilisation_signal`` via ``calculate_node_wise_forecasts``.
        Matching columns in *forecast_df* overwrite calculated forecasts.

        Algorithm keys (defaults):
            - ``congestion_signal`` (``congestion_signal_naive_forecast``)
            - ``renewable_utilisation`` (``renewable_utilisation_naive_forecast``)

        Args:
            units (list[BaseUnit]): All units in the simulation.
            market_configs (list[MarketConfig]): Available market configurations.
            forecast_df (ForecastSeries, optional): Explicitly provided forecasts.
            initializing_unit (BaseUnit, optional): The unit currently being initialized.
        """
        # 1. Preprocess price and residual load (parent)
        super().preprocess(units, market_configs, forecast_df, initializing_unit)

        # 2. Own preprocess (congestion_signal, renewable_utilisation)
        #    and 3. Initialize price and residual load forecasts (parent)
        super().initialize(
            units,
            market_configs,
            forecast_df,
            initializing_unit,
        )

        # 4. Derive electricity price from EOM market price
        # TODO: how to handle other markets?
        self.electricity_price = self.price.get("EOM")

        # 5. Get congestion signal forecast
        congestion_signal_forecast_algorithm_name = self.forecast_algorithms.get(
            "congestion_signal", "congestion_signal_naive_forecast"
        )
        congestion_signal_forecast_algorithm = self._registries["init"].get(
            congestion_signal_forecast_algorithm_name
        )
        if (
            congestion_signal_forecast_algorithm is not None
        ):  # None means keep existing forecast
            self.congestion_signal = calculate_node_wise_forecasts(
                self.index,
                units,
                market_configs,
                congestion_signal_forecast_algorithm,
                forecast_df,
                self.preprocess_information["congestion_signal"],
                prefix="congestion_signal",
            )
            self.congestion_signal = self._dict_to_series(self.congestion_signal)

        # 6. Get renewable utilisation forecast
        renewable_utilisation_forecast_algorithm_name = self.forecast_algorithms.get(
            "renewable_utilisation", "renewable_utilisation_naive_forecast"
        )
        renewable_utilisation_forecast_algorithm = self._registries["init"].get(
            renewable_utilisation_forecast_algorithm_name
        )
        if (
            renewable_utilisation_forecast_algorithm is not None
        ):  # None means keep existing forecast
            self.renewable_utilisation_signal = calculate_node_wise_forecasts(
                self.index,
                units,
                market_configs,
                renewable_utilisation_forecast_algorithm,
                forecast_df,
                self.preprocess_information["renewable_utilisation"],
                prefix="renewable_utilisation",
            )
            self.renewable_utilisation_signal = self._dict_to_series(
                self.renewable_utilisation_signal
            )

    def update(self, *args, **kwargs):
        """Update DSM-specific forecast timeseries during runtime.

        Calls parent update for price / residual load, then additionally updates
        ``congestion_signal`` and ``renewable_utilisation_signal``.

        Algorithm keys (defaults):
            - ``update_congestion_signal`` (``congestion_signal_default``)
            - ``update_renewable_utilisation`` (``renewable_utilisation_default``)

        Args:
            *args: Passed through to the underlying update algorithms.
            **kwargs: Passed through to the underlying update algorithms.
        """

        super().update(*args, **kwargs)

        congestion_signal_update_algorithm_name = self.forecast_algorithms.get(
            "update_congestion_signal", "congestion_signal_default"
        )
        congestion_signal_update_algorithm = self._registries["update"].get(
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
        renewable_utilisation_update_algorithm = self._registries["update"].get(
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
    """Forecaster for steelplant units.

    Provides all DSM forecasts (see :class:`DsmUnitForecaster`) plus fuel prices.
    After initialization, DSM signals are copied to the unit and ``setup_model()`` is called.

    Attributes:
        fuel_prices (dict[str, ForecastSeries]): Map of fuel type to forecasted fuel prices.
    """

    def __init__(
        self,
        index: ForecastIndex,
        fuel_prices: dict[str, ForecastSeries],
        market_prices: dict[str, ForecastSeries] = None,
        residual_load: dict[str, ForecastSeries] = None,
        availability: ForecastSeries = 1,
        forecast_algorithms: dict[str, str] = {},
        forecast_registries: dict[str, dict] = None,
        congestion_signal: ForecastSeries = 0.0,
        renewable_utilisation_signal: ForecastSeries = 0.0,
        electricity_price: ForecastSeries = None,
    ):
        super().__init__(
            index=index,
            availability=availability,
            forecast_algorithms=forecast_algorithms,
            forecast_registries=forecast_registries,
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
    """Forecaster for steam generation units.

    Provides all DSM forecasts (see :class:`DsmUnitForecaster`) plus fuel prices and
    thermal-process-specific timeseries. After initialization, DSM signals are copied
    to the unit and ``setup_model()`` is called.

    Attributes:
        fuel_prices (dict[str, ForecastSeries]): Map of fuel type to forecasted fuel prices.
        demand (ForecastSeries): Forecasted electricity demand.
        electricity_price_flex (ForecastSeries): Forecasted flexible electricity price.
        thermal_demand (ForecastSeries): Forecasted thermal demand.
        thermal_storage_schedule (ForecastSeries): Forecasted thermal storage schedule.
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
        forecast_registries: dict[str, dict] = None,
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
            forecast_registries=forecast_registries,
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
    Forecaster for building units.

    Provides all DSM forecasts plus building-specific profiles.
    Supports both aggregate building profiles and arbitrary component-level
    profiles passed through **kwargs.
    """

    def __init__(
        self,
        index: ForecastIndex,
        fuel_prices: dict[str, ForecastSeries],
        heat_demand: ForecastSeries = 0,
        ev_load_profile: ForecastSeries = 0,
        battery_load_profile: ForecastSeries = 0,
        pv_profile: ForecastSeries = 0,
        forecast_algorithms: dict[str, str] = {},
        forecast_registries: dict[str, dict] = None,
        market_prices: dict[str, ForecastSeries] = None,
        residual_load: dict[str, ForecastSeries] = None,
        congestion_signal: ForecastSeries = 0.0,
        renewable_utilisation_signal: ForecastSeries = 0.0,
        electricity_price: ForecastSeries = 0.0,
        electricity_price_flex: ForecastSeries = 0.0,
        availability: ForecastSeries = 1,
        load_profile: ForecastSeries = 0,
        **kwargs,
    ):
        super().__init__(
            index=index,
            availability=availability,
            forecast_algorithms=forecast_algorithms,
            forecast_registries=forecast_registries,
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
        self.electricity_price_flex = self._to_series(electricity_price_flex)

        # store arbitrary building-specific additional profiles
        for key, value in kwargs.items():
            if isinstance(value, pd.Series):
                value = self._to_series(value)
            self.__setattr__(key, value)

    def __getitem__(self, key: str):
        """
        Allow dictionary-style access for arbitrary component-level forecasts,
        e.g.:
            forecaster["building_1_electric_vehicle_1_availability_profile"]
            forecaster["building_1_electric_vehicle_1_range"]
        """
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Forecast '{key}' not found in BuildingForecaster.")

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
        initializing_unit.setup_model(presolve=True)


class HydrogenForecaster(DsmUnitForecaster):
    """Forecaster for hydrogen units.

    Provides all DSM forecasts (see :class:`DsmUnitForecaster`) plus hydrogen-specific
    timeseries. After initialization, electricity price is copied to the unit and
    ``setup_model()`` is called.

    Attributes:
        hydrogen_demand (ForecastSeries): Forecasted hydrogen demand.
        seasonal_storage_schedule (ForecastSeries): Forecasted seasonal storage schedule.
    """

    def __init__(
        self,
        index: ForecastIndex,
        hydrogen_demand: ForecastSeries,
        forecast_algorithms: dict[str, str] = {},
        forecast_registries: dict[str, dict] = None,
        seasonal_storage_schedule: ForecastSeries = 0,
        availability: ForecastSeries = 1,
        market_prices: dict[str, ForecastSeries] = None,
        residual_load: dict[str, ForecastSeries] = None,
        electricity_price: ForecastSeries = 0.0,
    ):
        super().__init__(
            index=index,
            availability=availability,
            forecast_algorithms=forecast_algorithms,
            forecast_registries=forecast_registries,
            market_prices=market_prices,
            residual_load=residual_load,
            electricity_price=electricity_price,
        )
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
    """Forecaster for exchange (import/export) units.

    Provides price, residual load, and availability forecasts (see :class:`UnitForecaster`)
    plus import and export volume timeseries.

    Attributes:
        volume_import (ForecastSeries): Forecasted import volume.
        volume_export (ForecastSeries): Forecasted export volume.
    """

    def __init__(
        self,
        index: ForecastIndex,
        forecast_algorithms: dict[str, str] = {},
        forecast_registries: dict[str, dict] = None,
        volume_import: ForecastSeries = 0,
        volume_export: ForecastSeries = 0,
        residual_load: dict[str, ForecastSeries] = None,
        availability: ForecastSeries = 1,
        market_prices: dict[str, ForecastSeries] = None,
    ):
        super().__init__(
            index=index,
            availability=availability,
            forecast_algorithms=forecast_algorithms,
            forecast_registries=forecast_registries,
            market_prices=market_prices,
            residual_load=residual_load,
        )
        self.volume_import = self._to_series(volume_import)
        self.volume_export = self._to_series(volume_export)
