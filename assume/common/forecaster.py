# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations
import logging
from functools import lru_cache
from typing import TypeAlias, Callable, TYPE_CHECKING
from collections import defaultdict

import pandas as pd
import numpy as np

from assume.common.fast_pandas import FastIndex, FastSeries
from assume.common.market_objects import MarketConfig

if TYPE_CHECKING:
    from assume.common.base import BaseUnit #, BaseForecaster
    from assume.units.powerplant import PowerPlant
    from assume.units.demand import Demand
    from assume.units.exchange import Exchange
    from assume.units.dsm_load_shift import DSMFlex

ForecastIndex: TypeAlias = FastIndex | pd.DatetimeIndex | pd.Series
ForecastSeries: TypeAlias = FastSeries | list | float | pd.Series

log = logging.getLogger(__name__)

def is_renewable(name:str) -> bool:
    return "wind" in name.lower() or "solar" in name.lower()

def _ensure_not_none(
    df: pd.DataFrame | None, index: ForecastIndex, check_index=False
) -> pd.DataFrame:
    if isinstance(index, FastIndex):
        index = index.as_datetimeindex()

    if df is None:
        return pd.DataFrame(index=index)
    if check_index and index.freq != df.index.inferred_freq:
        raise ValueError("Forecast frequency does not match index frequency.")
    return df 

# @lru_cache(maxsize=100)
def calculate_max_power(units, index=None):
    """
    Returns: max available power: shape (num_units, forecast_len)
    """
    return pd.DataFrame([unit.max_power * unit.forecaster.availability for unit in units], index=index)

@lru_cache(maxsize=10)
def sort_units(units:list[BaseUnit], market_id:str | None=None):
    from assume.common.base import BaseUnit #, BaseForecaster
    from assume.units.powerplant import PowerPlant
    from assume.units.demand import Demand
    from assume.units.exchange import Exchange
    from assume.units.storage import Storage
    from assume.units.dsm_load_shift import DSMFlex
    pps: list[PowerPlant] = []
    demands: list[Demand] = []
    storages: list[Storage] = []
    exchanges: list[Exchange] = []
    dsm_units: list[DSMFlex] = []


    for unit in units:
        if market_id is not None and market_id not in unit.bidding_strategies:
            continue
        if isinstance(unit, PowerPlant):
            pps.append(unit)
        elif isinstance(unit, Demand):
            demands.append(unit)
        elif isinstance(unit, Storage):
            storages.append(unit)
        elif isinstance(unit, Exchange):
            exchanges.append(unit)
        elif isinstance(unit, DSMFlex):
            dsm_units.appen(unit)

    return pps, demands, exchanges, storages, dsm_units

# @lru_cache(maxsize=10)
def calculate_sum_demand(
    demand_units: list[Demand],
    exchange_units: list[Exchange],
):
    """
    Returns summed demand at every timestep (incl. imports and exports)
    """
    sum_demand = abs(np.array([unit.forecaster.demand for unit in demand_units])).sum(axis=0)

    # get exchanges if exchange_units are available
    if exchange_units:  # if not empty
        # get sum of imports as name of exchange_unit_import
        sum_imports = abs(np.array([unit.forecaster.volume_import for unit in exchange_units])).sum(axis=0)

        sum_exports = abs(np.array([unit.forecaster.volume_export for unit in exchange_units])).sum(axis=0)
        # add imports and exports to the sum_demand
        sum_demand += sum_imports - sum_exports

    return sum_demand

#@lru_cache(maxsize=1000)
def calculate_naive_price(
    index: ForecastIndex,
    units: list[BaseUnit],
    market_configs: list[MarketConfig],
    forecast_df: ForecastSeries = None,
) -> dict[str, ForecastSeries]:
    """
    Naive price forecast that calculates prices based on merit order with marginal costs.
    Does not take account: Storages, DSM units.
    TODO: further documentation
    """
    if isinstance(index, FastIndex):
        index = index.as_datetimeindex()

    _, demand_units, exchange_units, _, _ = sort_units(
            units
    )
    forecast_df = _ensure_not_none(forecast_df, index)

    price_forecasts: dict[str, pd.Series] = {}

    for config in market_configs:
        market_id = config.market_id
        if config.product_type != "energy":
            log.warning(
                f"Price forecast could not be calculated for {market_id}. It can only be calculated for energy-only markets for now."
            )
            continue

        price_forecasts[market_id] = forecast_df.get(f"price_{market_id}")
        if price_forecasts[market_id] is not None:
            # go next if forecast existing
            continue

        # 1. Sort units by type and filter for units with bidding strategy for the given market_id
        powerplants_units, demand_units, exchange_units, storage_units, dsm_units = sort_units(
            units, market_id
        )

        # 2. Calculate marginal costs for each unit and time step.
        #    The resulting DataFrame has rows = time steps and columns = units.
        #    shape: (index_len, num_pp_units)
        # marginal_costs = powerplants_units.apply(calculate_marginal_cost, axis=1).T
        marginal_costs = pd.DataFrame([unit.marginal_cost for unit in powerplants_units]).T.set_index(index)


        # 3. Compute available power for each unit at each time step.
        #    shape: (index_len, num_pp_units)
        # power = self._calc_power()
        # power = pd.Series([unit.forecaster.availability * unit.max_power for unit in powerplants_units]).T
        power = calculate_max_power(powerplants_units).T.set_index(index)

        # 4. Process the demand.
        #    Filter demand units with a bidding strategy and sum their forecasts for each time step.
        sum_demand = pd.DataFrame(calculate_sum_demand(demand_units, exchange_units), index=index)

        # 5. Initialize the price forecast series.
        price_forecast = pd.Series(index=index, data=0.0)

        # 6. Loop over each time step
        for t in index:
            # Get marginal costs and available power for time t (both are Series indexed by unit)
            mc_t = marginal_costs.loc[t]
            power_t = power.loc[t]
            demand_t = sum_demand.loc[t].item()

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


#@lru_cache(maxsize=100)
def calculate_naive_residual_load(
    index: ForecastIndex,
    units: list[BaseUnit],
    market_configs: list[MarketConfig],
    forecast_df: ForecastSeries = None,
) -> dict[str, ForecastSeries]:

    _, demand_units, exchange_units, _, _ = sort_units(
            units
    )

    forecast_df = _ensure_not_none(forecast_df, index)

    residual_loads: dict[str, pd.Series] = {}

    for config in market_configs:
        market_id = config.market_id
        if config.product_type != "energy":
            log.warning(
                f"Load forecast could not be calculated for {market_id}. It can only be calculated for energy-only markets for now."
            )
            continue

        residual_loads[market_id] = forecast_df.get(f"residual_load_{market_id}")
        if residual_loads[market_id] is not None:
            # go next if forecast existing
            continue
            
        powerplants_units, demand_units, exchange_units, storage_units, dsm_units = sort_units(
            units, market_id
        )

        sum_demand = calculate_sum_demand(demand_units, exchange_units)

        # shape: (num_pp_units, index_len) -> (index_len)
        # vre_feed_in_df = self._calc_power(mask).sum(axis=1)
        renewable_units = [unit for unit in powerplants_units if is_renewable(unit.technology)]
        vre_feed_in_df = pd.DataFrame(calculate_max_power(renewable_units)).sum(axis=0)

        if vre_feed_in_df.empty:
            vre_feed_in_df = 0
        res_demand_df = sum_demand - vre_feed_in_df

        residual_loads[market_id] = res_demand_df
    return residual_loads

def extract_buses_and_lines(market_configs: list[MarketConfig]):
    buses, lines = None, None

    for market_config in market_configs:
        grid_data = market_config.param_dict.get("grid_data")

        if grid_data is None:
            continue

        buses = grid_data.get("buses")
        lines = grid_data.get("lines")
        if buses is not None and lines is not None:
            break
    
    return buses, lines

#@lru_cache(maxsize=100)
def calculate_naive_congestion_forecast(
    index: ForecastIndex,
    units: list[BaseUnit],
    market_configs: list[MarketConfig],
    forecast_df: ForecastSeries = None,
) -> dict[str, ForecastSeries]:
    # Lines and buses should be everywhere the same
    buses, lines = extract_buses_and_lines(market_configs)

    if buses is None or lines is None:
        return None

    powerplants_units, demand_units, exchange_units, storage_units, dsm_units, = sort_units(
        units
    )

    forecast_df = _ensure_not_none(forecast_df, index)

    demand_unit_nodes = {demand.node for demand in demand_units}
    if not all(node in buses.index for node in demand_unit_nodes):
        log.warning(
                "Node-specific congestion signals and renewable utilisation forecasts could not be calculated. "
                "Not all unit nodes are available in buses."
            )
        return None

    # Step 1: Calculate load for each powerplant based on availability factor and max power
    # shape: (forecast_len, num_units)
    power = calculate_max_power(powerplants_units, index=[pp.id for pp in powerplants_units]).T


    # Step 2: Calculate net load for each node (demand - generation)
    net_load_by_node = {}
    
    for node in demand_unit_nodes:
        # Calculate total demand for this node
        node_demand_units = [unit for unit in demand_units if unit.node == node]
        node_demand = calculate_sum_demand(node_demand_units, [],)

        # Calculate total generation for this node by summing powerplant loads
        node_powerplants_units = [unit.id for unit in powerplants_units if unit.node == node]
        node_generation = power[node_powerplants_units].sum(axis=1)

        # Calculate net load (demand - generation)
        net_load_by_node[node] = node_demand - node_generation
    print("load_by_node", net_load_by_node)
    # Step 3: Calculate line-specific congestion severity
    line_congestion_severity = pd.DataFrame(index=index)

    for line_id, line_data in lines.iterrows():
        node1, node2 = line_data["bus0"], line_data["bus1"]
        print(line_id, line_data)
        print(node1, node2)
        s_max_pu = (
            lines.at[line_id, "s_max_pu"]
            if "s_max_pu" in lines.columns
            and not pd.isna(lines.at[line_id, "s_max_pu"])
            else 1.0
        )
        line_capacity = line_data["s_nom"] * s_max_pu

        # Calculate net load for the line as the sum of net loads from both connected nodes
        line_net_load = net_load_by_node[node1] + net_load_by_node[node2]
        print(line_net_load)
        print(line_capacity)
        congestion_severity = line_net_load / line_capacity

        # Store the line-specific congestion severity in DataFrame
        name = f"{line_id}_congestion_severity"
        line_congestion_severity.insert(0, )
    
    print("line_congestion_severity", line_congestion_severity)
    # Step 4: Calculate node-specific congestion signal by aggregating connected lines
    node_congestion_signal = pd.DataFrame(index=index)

    for node in demand_unit_nodes:
        node_congestion_signal[f"{node}_congestion_severity"] = forecast_df.get(f"{node}_congestion_severity")

        if node_congestion_signal[f"{node}_congestion_severity"] is not None:
            # go next if forecast existing
            continue

        # Find all lines connected to this node
        connected_lines = lines[
            (lines["bus0"] == node) | (lines["bus1"] == node)
        ].index

        # Collect all relevant line congestion severities
        relevant_lines = [
            f"{line_id}_congestion_severity" for line_id in connected_lines
        ]

        # Ensure only existing columns are used to avoid KeyError
        relevant_lines = [
            line
            for line in relevant_lines
            if line in line_congestion_severity.columns
        ]

        # Aggregate congestion severities for this node (use max or mean)
        if relevant_lines:
            node_congestion_signal[f"{node}_congestion_severity"] = (
                line_congestion_severity[relevant_lines].max(axis=1)
            )
    print("cs", node_congestion_signal)
    return node_congestion_signal

#@lru_cache(maxsize=100)
def calculate_naive_renewable_utilisation(
    index: ForecastIndex,
    units: list[BaseUnit],
    market_configs: list[MarketConfig],
    forecast_df: ForecastSeries = None,
) -> dict[str, ForecastSeries]:
    forecast_df = _ensure_not_none(forecast_df, index)

    # Lines and buses should be everywhere the same
    buses, lines = extract_buses_and_lines(market_configs)

    if buses is None or lines is None:
        return None

    powerplants_units, demand_units, exchange_units, storage_units, dsm_units = sort_units(
        units
    )

    demand_unit_nodes = {demand.node for demand in demand_units}
    if not all(node in buses.index for node in demand_unit_nodes):
        log.warning(
                "Node-specific congestion signals and renewable utilisation forecasts could not be calculated. "
                "Not all unit nodes are available in buses."
            )
        return None

    # Calculate load for each renewable powerplant based on availability factor and max power
    # shape: (num_pps, forecast_len)
    renewable_units = [unit for unit in powerplants_units if is_renewable(unit.technology)]
    power = calculate_max_power(renewable_units, index=[pp.id for pp in renewable_units])

    renewable_utilisation = pd.DataFrame(index=index)
    
    # Calculate utilisation based on availability and max power for each node
    for node in demand_unit_nodes:
        renewable_utilisation[f"{node}_renewable_utilisation"] = forecast_df.get(f"{node}_renewable_utilisation")

        if renewable_utilisation[f"{node}_renewable_utilisation"] is not None:
            # go next if forecast existing
            continue
        node_renewable_units = [unit.id for unit in renewable_units if unit.node == node]
        utilization = powers[node_renewable_units].sum(axis=0)
        renewable_utilisation[f"{node}_renewable_utilisation"] = utilization

    # Calculate the total renewable utilisation across all nodes if not in forecast_df
    renewable_utilisation["all_nodes_renewable_utilisation"] = forecast_df.get("all_nodes_renewable_utilisation")
    if renewable_utilisation["all_nodes_renewable_utilisation"] is None:
        all_nodes_sum = renewable_utilisation.sum(axis=0)
        renewable_utilisation["all_nodes_renewable_utilisation"] = all_nodes_sum

    return renewable_utilisation


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
        self.residual_load: dict[str, ForecastSeries] = self._dict_to_series(residual_load)

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
        units: list[BaseUnit],
        market_configs: list[MarketConfig],
        # fuel_prices: ForecastSeries = None,
        forecast_df: ForecastSeries = None,
    ):
        """
        """

        # 1. Get price forecast
        price_forecast_algorithm = price_forecast_algorithms.get(
            self.forecast_algorithms.get("price"),
            calculate_naive_price
        )
        if price_forecast_algorithm is not None:  # None if one wants to keep forecasts
            self.price = price_forecast_algorithm(
                self.index,
                units,
                market_configs,
                forecast_df,
            )
        self.price = self._dict_to_series(self.price)

        # 2. Get residual load forecast
        residual_load_forecast_algorithm = residual_load_forecast_algorithms.get(
            self.forecast_algorithms.get("residual_load"),
            calculate_naive_residual_load
        )
        if residual_load_forecast_algorithm is not None:  # None if one wants to keep forecasts
            self.residual_load = residual_load_forecast_algorithm(
                self.index,
                units,
                market_configs,
                forecast_df,
            )
        self.residual_load = self._dict_to_series(self.residual_load)

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
        self.renewable_utilisation_signal = self._to_series(renewable_utilisation_signal)


    def initialize(
        self,
        units: list[BaseUnit],
        market_configs: list[MarketConfig],
        forecast_df: ForecastSeries = None,
    ):
        # Get 1. price and 2. residual load forecast
        super(UnitForecaster).initialize(
            units,
            market_configs,
            forecast_df,
        )

        # 3. Get electricity price forecast
        # TODO how to handle other markets?
        self.electricity_price = self.price_forecast.get("price_EOM")


        # 4. Get congestion signal forecast
        congestion_signal_forecast_algorithm = congestion_signal_forecast_algorithms.get(
            self.forecast_algorithms.get("congestion_signal"),
            calculate_naive_congestion_signal
        )
        if congestion_signal_forecast_algorithm is not None:  # None if one wants to keep forecasts
            self.congestion_signal = congestion_signal_forecast_algorithm(
                self.index,
                units,
                market_configs,
                forecast_df,
            )
        self.congestion_signal = self._dict_to_series(self.congestion_signal)


        # 5. Get renewable utilization forecast
        renewable_utilization_forecast_algorithm = renewable_utilization_forecast_algorithms.get(
            self.forecast_algorithms.get("renewable_utilization"),
            renewable_utilization_congestion_signal
        )
        if congestion_signal_forecast_algorithm is not None:  # None if one wants to keep forecasts
            self.renewable_utilization = renewable_utilization_forecast_algorithm(
                self.index,
                units,
                market_configs,
                forecast_df,
            )
        self.renewable_utilization = self._dict_to_series(self.renewable_utilization)


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
        #super().__init__(index, market_prices, residual_load, availability)
        super().__init__(
            index=index,
            availability=availability,
            forecast_algorithms=forecast_algorithms,
            market_prices=market_prices,
            residual_load=residual_load,
            congestion_signal=congestion_signal,
            renewable_utilisation_signal=renewable_utilisation_signal,
            electricity_price=electricity_price
        )
        self.fuel_prices = self._dict_to_series(fuel_prices)

    def get_price(self, fuel: str) -> FastSeries:
        if fuel not in self.fuel_prices:
            return self._to_series(0)
        return self.fuel_prices[fuel]



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
            electricity_price=electricity_price
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
            electricity_price=electricity_price
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
            electricity_price=electricity_price
        )
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


price_forecast_algorithms: dict[str, Callable] = {
    "price_naive_forecast": calculate_naive_price,
    "default_test": lambda index, *args: {"EOM", FastSeries(index=index, value=50)},
    "keep_given": None,
}

residual_load_forecast_algorithms: dict[str, Callable] = {
    "residual_load_naive_forecast": calculate_naive_residual_load,
    "default_test": lambda *args: {},
    "keep_given": None,
}

congestion_signal_forecast_algorithms: dict[str, Callable] = {
    "congestion_signal_naive_forecast": calculate_naive_congestion_forecast,
    "default_test": lambda index, *args: FastSeries(index=self.index, value=0.0),
    "keep_given": None,
}

renewable_utilization_forecast_algorithms: dict[str, Callable] = {
    "renewable_utilization_naive_forecast": calculate_naive_renewable_utilisation,
    "default_test": lambda index, *args: FastSeries(index=self.index, value=0.0),
    "keep_given": None,
}
