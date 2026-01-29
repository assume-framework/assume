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
from assume.units.dsm_load_shift import DSMFlex

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
    return 

@lru_cache(max_size=100)
def calculate_max_power(units):
    """
    Returns: max available power: shape (num_units, forecast_len)
    """
    return [unit.max_power * unit.forecaster.availability for unit in units]

@lru_cache(max_size=10)
def sort_units(units:list[BaseUnit], market_id:str | None):
    pps: list[PowerPlant] = []
    demands: list[Demand] = []
    exchanges: list[Exchange] = []
    dsm_units: list[DSMFlex] = []

    for unit in units:
        if market_id is not None and market_id not in unit.bidding_strategies:
            continue
        if isinstance(unit, PowerPlant):
            pps.append(unit)
        elif isinstance(unit, Demand):
            demands.append(unit)
        elif isinstance(unit, Exchange):
            exchanges.append(unit)
        elif isinstance(unit, DSMFlex):
            dsm_units.appen(unit)

    return pps, demand, exchanges, dsm_units

@lru_cache(maxsize=10)
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
    """
    Naive price forecast that calculates prices based on merit order with marginal costs.
    Does not take account: Storages, DSM units.
    TODO: further documentation
    """

    demand_df = _ensure_not_none(demand_df, index)
    forecast_df = _ensure_not_none(forecast_df, index)
    exchange_df = _ensure_not_none(exchange_df, index)

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
        powerplant_units, demand_units, exchange_units, dsm_units = sort_units(
            units, market_id
        )

        # 2. Calculate marginal costs for each unit and time step.
        #    The resulting DataFrame has rows = time steps and columns = units.
        #    shape: (index_len, num_pp_units)
        # marginal_costs = powerplants_units.apply(calculate_marginal_cost, axis=1).T
        marginal_costs = pd.DataFrame([unit.marginal_cost for unit in powerplant_units]).T


        # 3. Compute available power for each unit at each time step.
        #    shape: (index_len, num_pp_units)
        # power = self._calc_power()
        # power = pd.Series([unit.forecaster.availability * unit.max_power for unit in powerplant_units]).T
        power = pd.DataFrame(calculate_max_power(powerplant_units)).T

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
    forecast_df = _ensure_not_none(forecast_df, index)
    exchange_df = _ensure_not_none(exchange_df, index)

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
            
        powerplants_units, demand_units, exchange_units, dsm_units = sort_units(
            units, market_id
        )

        sum_demand = calculate_sum_demand(demand_units, exchange_units, demand_df, exchange_df)

        # shape: (num_pp_units, index_len) -> (index_len)
        # vre_feed_in_df = self._calc_power(mask).sum(axis=1)
        renewable_units = [unit for unit in powerplants_units if is_renewable(unit.technology)]
        vre_feed_in_df = pd.DataFrame(calculate_max_power(renewable_units)).sum(axis=0)

        if vre_feed_in_df.empty:
            vre_feed_in_df = 0
        res_demand_df = sum_demand - vre_feed_in_df

        residual_loads[market_id] = res_demand_df
    return residual_loads

def extract_buses_and_lines(market_configs: dict[str, dict]):
    buses, lines = None, None

    for market_config in market_configs:
        grid_data = market_configs.param_dict.get("grid_data")

        if grid_data is None:
            continue

        buses = grid_data.get("buses")
        lines = grid_data.get("lines")
        if buses is not None and lines is not None:
            break
    
    return buses, lines


@lru_cache(maxsize=100)
def calculate_congestion_forecast(
    index: ForecastIndex,
    units: dict[str, BaseUnit],
    market_configs: dict[str, dict],
    forecast_df: ForecastSeries = None,
    demand_df: ForecastSeries = None,
):

    forecast_df = _ensure_not_none(forecast_df, index)
    demand_df = _ensure_not_none(demand_df, index)

    # Lines and buses should be everywhere the same
    buses, lines = extract_buses_and_lines(market_configs)

    if buses is None or lines is None:
        return None

    powerplants_units, demand_units, exchange_units, dsm_units = sort_units(
        units
    )

    demand_unit_nodes = {demand.node for demand in demand_units}
    if not all(node in buses for node in demand_unit_nodes):
        self._logger.warning(
                "Node-specific congestion signals and renewable utilisation forecasts could not be calculated. "
                "Not all unit nodes are available in buses."
            )
        return None

    # Step 1: Calculate load for each powerplant based on availability factor and max power
    # shape: (num_pps, forecast_len)
    power = pd.DataFrame(
        calculate_max_power(powerplant_units),
        index=[pp.name for pp in powerplant_units]
    )

    # Step 2: Calculate net load for each node (demand - generation)
    net_load_by_node = {}
    
    for node in demand_unit_nodes:
        # Calculate total demand for this node
        node_demand_units = [unit for unit in demand_units if unit.node == node]
        node_demand = calculate_sum_demand(node_demand_units, [], demand_df, None)

        # Calculate total generation for this node by summing powerplant loads
        node_powerplant_units = [unit.name for unit in powerplants_units if unit.node == node]
        node_generation = power[node_powerplant_units].sum(axis=1)

        # Calculate net load (demand - generation)
        net_load_by_node[node] = node_demand - node_generation

    # Step 3: Calculate line-specific congestion severity
    line_congestion_severity = pd.DataFrame(index=index)

    for line_id, line_data in lines.iterrows():
        node1, node2 = line_data["bus0"], line_data["bus1"]
        s_max_pu = (
            lines.at[line_id, "s_max_pu"]
            if "s_max_pu" in lines.columns
            and not pd.isna(lines.at[line_id, "s_max_pu"])
            else 1.0
        )
        line_capacity = line_data["s_nom"] * s_max_pu

        # Calculate net load for the line as the sum of net loads from both connected nodes
        line_net_load = net_load_by_node[node1] + net_load_by_node[node2]
        congestion_severity = line_net_load / line_capacity

        # Store the line-specific congestion severity in DataFrame
        line_congestion_severity[f"{line_id}_congestion_severity"] = (
            congestion_severity
        )
    
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

    return node_congestion_signal

@lru_cache(maxsize=100)
def calculate_renewable_utilisation(
    index: ForecastIndex,
    units: dict[str, BaseUnit],
    market_configs: dict[str, dict],
    forecast_df: ForecastSeries = None,
):
    forecast_df = _ensure_not_none(forecast_df, index)

    # Lines and buses should be everywhere the same
    buses, lines = extract_buses_and_lines(market_configs)

    if buses is None or lines is None:
        return None

    powerplants_units, demand_units, exchange_units, dsm_units = sort_units(
        units
    )

    demand_unit_nodes = {demand.node for demand in demand_units}
    if not all(node in buses for node in demand_unit_nodes):
        self._logger.warning(
                "Node-specific congestion signals and renewable utilisation forecasts could not be calculated. "
                "Not all unit nodes are available in buses."
            )
        return None

    # Calculate load for each renewable powerplant based on availability factor and max power
    # shape: (num_pps, forecast_len)
    renewable_units = [unit for unit in powerplants_units if is_renewable(unit.technology)]
    power = pd.DataFrame(
        calculate_max_power(renewable_units),
        index=[pp.name for pp in renewable_units]
    )

    renewable_utilisation = pd.DataFrame(index=index)
    
    # Calculate utilisation based on availability and max power for each node
    for node in demand_unit_nodes:
        renewable_utilisation[f"{node}_renewable_utilisation"] = forecast_df.get(f"{node}_renewable_utilisation")

        if renewable_utilisation[f"{node}_renewable_utilisation"] is not None:
            # go next if forecast existing
            continue
        node_powerplant_units = [unit.name for unit in renewable_units if unit.node == node]
        utilization = powers[node_units].sum(axis=0)
        renewable_utilisation[f"{node}_renewable_utilisation"] = utilization

    # Calculate the total renewable utilisation across all nodes if not in forecast_df
    renewable_utilisation["all_nodes_renewable_utilisation"] = forecast_df.get("all_nodes_renewable_utilisation")
    if renewable_utilisation["all_nodes_renewable_utilisation"] is None:
        all_nodes_sum = renewable_utilisation.sum(axis=0)
        renewable_utilisation["all_nodes_renewable_utilisation"] = all_nodes_sum

    return renewable_utilisation

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
        self.price = calculate_naive_price(
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

    def initialize(
        self,
        units: dict[str, BaseUnit],
        market_configs: dict[str, dict],
        # fuel_prices: ForecastSeries = None,
        demand_df: ForecastSeries = None,
        forecast_df: ForecastSeries = None,
        exchange_df: ForecastSeries = None,
    ):
        # Get default price and residual load forecast
        super(UnitForecaster).initialize(
            units,
            market_configs,
            demand_df,
            forecast_df,
            exchange_df
        )

        # TODO how to handle other markets?
        self.electricity_price = self.price_forecast.get("price_EOM")
        
        self.congestion_signal = calculate_congestion_forecast(
            self.index,
            units,
            market_configs,
            forecast_df,
            demand_df,
        )
        self.renewable_utilization = calculate_renewable_utilisation(
            self.index,
            units,
            market_configs,
            forecast_df,
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
