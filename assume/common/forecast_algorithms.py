# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import pandas as pd

from assume.common.fast_pandas import FastIndex, FastSeries
from assume.common.market_objects import MarketConfig

if TYPE_CHECKING:
    from assume.common.base import BaseUnit
    from assume.units.demand import Demand
    from assume.units.dsm_load_shift import DSMFlex
    from assume.units.exchange import Exchange
    from assume.units.powerplant import PowerPlant
    from assume.units.storage import Storage

ForecastIndex: TypeAlias = FastIndex | pd.DatetimeIndex | pd.Series
ForecastSeries: TypeAlias = FastSeries | list | float | pd.Series

log = logging.getLogger(__name__)


# advanced on: https://discuss.python.org/t/memoizing-based-on-id-to-avoid-implementing-and-computing-hash-method/87701/7
def custom_lru_cache(func_or_None=None, maxsize=128, typed=False, hasher=id):
    """
    Implements a wrap for lru cache that enables to use non-hashable inputs for initialization:
    NOTE: Be careful in using this! If unhashable object changes between calls,
          this might give incorrect (old) results that do not get updated!
      hashifies inputs, than does lru_cache, than turns inputs back to normal and calls function

    wrapper(inputs) -> 'decorated'/cache(hashified_inputs) -> unwrapper(hashified_inputs) -> func(inputs)
    """

    def decorator(func):
        class Hashified:
            def __init__(self, obj):
                self.obj = obj

            def __eq__(self, other):
                return hasher(self.obj) == hasher(other.obj)

            def __hash__(self):
                return hasher(self.obj)

        def unwrapper(*args, **kwargs):
            # returns hashable inputs back to normal state
            return func(
                *(arg.obj if isinstance(arg, Hashified) else arg for arg in args),
                **{
                    name: value.obj if isinstance(value, Hashified) else value
                    for name, value in kwargs.items()
                },
            )

        def wrapper(*args, **kwargs):
            return decorated(
                *(Hashified(arg) if arg.__hash__ is None else arg for arg in args),
                **{
                    name: Hashified(value) if value.__hash__ is None else value
                    for name, value in kwargs.items()
                },
            )

        # wrap lru_cache with hashable inputs and afterwards turn them back to normal
        decorated = lru_cache(maxsize=maxsize, typed=typed)(unwrapper)
        wrapper.cache_info = decorated.cache_info
        wrapper.cache_clear = decorated.cache_clear
        if hasattr(decorated, "cache_parameters"):
            wrapper.cache_parameters = decorated.cache_parameters
        return wrapper

    return decorator if func_or_None is None else decorator(func_or_None)


def is_renewable(name: str) -> bool:
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


def calculate_max_power(units, index=None):
    """
    Returns: max available power: shape (num_units, forecast_len)
    """
    return pd.DataFrame(
        [unit.max_power * unit.forecaster.availability for unit in units], index=index
    )


@custom_lru_cache
def sort_units(units: list[BaseUnit], market_id: str | None = None):
    # FIXME: This is not nice but circumvents circular import for now
    from assume.units.demand import Demand
    from assume.units.dsm_load_shift import DSMFlex
    from assume.units.exchange import Exchange
    from assume.units.powerplant import PowerPlant
    from assume.units.storage import Storage

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
            dsm_units.append(unit)

    return pps, demands, exchanges, storages, dsm_units


def calculate_sum_demand(
    demand_units: list[Demand],
    exchange_units: list[Exchange],
):
    """
    Returns summed demand at every timestep (incl. imports and exports)
    Shape: (num_timesteps,)
    """
    sum_demand = abs(np.array([unit.forecaster.demand for unit in demand_units])).sum(
        axis=0
    )

    # get exchanges if exchange_units are available
    if exchange_units:  # if not empty
        # get sum of imports as name of exchange_unit_import
        sum_imports = abs(
            np.array([unit.forecaster.volume_import for unit in exchange_units])
        ).sum(axis=0)

        sum_exports = abs(
            np.array([unit.forecaster.volume_export for unit in exchange_units])
        ).sum(axis=0)
        # add imports and exports to the sum_demand
        sum_demand += sum_imports - sum_exports

    return sum_demand


@custom_lru_cache
def calculate_naive_price(
    index: ForecastIndex,
    units: list[BaseUnit],
    market_configs: list[MarketConfig],
    forecast_df: ForecastSeries = None,
    preprocess_information=None,
) -> dict[str, ForecastSeries]:
    """
    Naive price forecast that calculates prices based on merit order with marginal costs.
    Does not take account: Storages, DSM units.
    TODO: further documentation
    """
    if isinstance(index, FastIndex):
        index = index.as_datetimeindex()

    _, demand_units, exchange_units, _, _ = sort_units(units)
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
        powerplants_units, demand_units, exchange_units, storage_units, dsm_units = (
            sort_units(units, market_id)
        )

        # 2. Calculate marginal costs for each unit and time step.
        #    The resulting DataFrame has rows = time steps and columns = units.
        #    shape: (index_len, num_pp_units)
        marginal_costs = pd.DataFrame(
            [unit.marginal_cost for unit in powerplants_units]
        ).T.set_index(index)

        # 3. Compute available power for each unit at each time step.
        #    shape: (index_len, num_pp_units)
        power = calculate_max_power(powerplants_units).T.set_index(index)

        # 4. Process the demand.
        #    Filter demand units with a bidding strategy and sum their forecasts for each time step.
        sum_demand = pd.DataFrame(
            calculate_sum_demand(demand_units, exchange_units), index=index
        )

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


@custom_lru_cache
def calculate_naive_residual_load(
    index: ForecastIndex,
    units: list[BaseUnit],
    market_configs: list[MarketConfig],
    forecast_df: ForecastSeries = None,
    preprocess_information=None,
) -> dict[str, ForecastSeries]:
    _, demand_units, exchange_units, _, _ = sort_units(units)

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

        powerplants_units, demand_units, exchange_units, storage_units, dsm_units = (
            sort_units(units, market_id)
        )

        sum_demand = calculate_sum_demand(demand_units, exchange_units)

        # shape: (num_pp_units, index_len) -> (index_len)
        renewable_units = [
            unit for unit in powerplants_units if is_renewable(unit.technology)
        ]
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


@custom_lru_cache
def calculate_naive_congestion_signal(
    index: ForecastIndex,
    units: list[BaseUnit],
    market_configs: list[MarketConfig],
    forecast_df: ForecastSeries = None,
    preprocess_information=None,
) -> dict[str, ForecastSeries]:
    if isinstance(index, FastIndex):
        index = index.as_datetimeindex()

    # Lines and buses should be everywhere the same
    buses, lines = extract_buses_and_lines(market_configs)

    if buses is None or lines is None:
        return {}

    (
        powerplants_units,
        demand_units,
        exchange_units,
        storage_units,
        dsm_units,
    ) = sort_units(units)

    forecast_df = _ensure_not_none(forecast_df, index)

    demand_unit_nodes = {demand.node for demand in demand_units}
    if not all(node in buses.index for node in demand_unit_nodes):
        log.warning(
            "Node-specific congestion signals forecast could not be calculated. "
            "Not all unit nodes are available in buses."
        )
        return {}

    # Step 1: Calculate load for each powerplant based on availability factor and max power
    # shape: (forecast_len, num_units)
    power = calculate_max_power(
        powerplants_units, index=[pp.id for pp in powerplants_units]
    ).T

    # Step 2: Calculate net load for each node (demand - generation)
    net_load_by_node = {}

    for node in demand_unit_nodes:
        # Calculate total demand for this node
        node_demand_units = [unit for unit in demand_units if unit.node == node]
        node_demand = calculate_sum_demand(
            node_demand_units,
            [],
        )

        # Calculate total generation for this node by summing powerplant loads
        node_powerplants_units = [
            unit.id for unit in powerplants_units if unit.node == node
        ]
        node_generation = power[node_powerplants_units].sum(axis=1)

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

        # Store the line-specific congestion severity in DataFrame
        line_congestion_severity[f"{line_id}_congestion_severity"] = (
            line_net_load.values / line_capacity
        )

    # Step 4: Calculate node-specific congestion signal by aggregating connected lines
    node_congestion_signal = pd.DataFrame(index=index)

    for node in demand_unit_nodes:
        congestion_signal = forecast_df.get(f"{node}_congestion_severity")

        if congestion_signal is not None:
            node_congestion_signal[f"{node}_congestion_severity"] = congestion_signal
            # go next if forecast existing
            continue

        # Find all lines connected to this node
        connected_lines = lines[(lines["bus0"] == node) | (lines["bus1"] == node)].index

        # Collect all relevant line congestion severities
        relevant_lines = [
            f"{line_id}_congestion_severity" for line_id in connected_lines
        ]

        # Ensure only existing columns are used to avoid KeyError
        relevant_lines = [
            line for line in relevant_lines if line in line_congestion_severity.columns
        ]

        # Aggregate congestion severities for this node (use max or mean)
        if relevant_lines:
            node_congestion_signal[f"{node}_congestion_severity"] = (
                line_congestion_severity[relevant_lines].max(axis=1)
            )

    return node_congestion_signal


@custom_lru_cache
def calculate_naive_renewable_utilisation(
    index: ForecastIndex,
    units: list[BaseUnit],
    market_configs: list[MarketConfig],
    forecast_df: ForecastSeries = None,
    preprocess_information=None,
) -> dict[str, ForecastSeries]:
    if isinstance(index, FastIndex):
        index = index.as_datetimeindex()

    forecast_df = _ensure_not_none(forecast_df, index)

    # Lines and buses should be everywhere the same
    buses, lines = extract_buses_and_lines(market_configs)

    if buses is None or lines is None:
        return {}

    powerplants_units, demand_units, exchange_units, storage_units, dsm_units = (
        sort_units(units)
    )

    demand_unit_nodes = {demand.node for demand in demand_units}
    if not all(node in buses.index for node in demand_unit_nodes):
        log.warning(
            "Node-specific renewable utilisation forecasts could not be calculated. "
            "Not all unit nodes are available in buses."
        )
        return {}

    # Calculate load for each renewable powerplant based on availability factor and max power
    # shape: (forecast_len, num_pps)
    renewable_units = [
        unit for unit in powerplants_units if is_renewable(unit.technology)
    ]
    power = calculate_max_power(
        renewable_units, index=[pp.id for pp in renewable_units]
    ).T

    renewable_utilisation = pd.DataFrame(index=index)

    # Calculate utilisation based on availability and max power for each node
    for node in demand_unit_nodes:
        utilisation = forecast_df.get(f"{node}_renewable_utilisation")

        if utilisation is not None:
            renewable_utilisation[f"{node}_renewable_utilisation"] = utilisation
            # go next if forecast existing
            continue
        node_renewable_units = [
            unit.id for unit in renewable_units if unit.node == node
        ]
        utilisation = power[node_renewable_units].sum(axis=1)
        renewable_utilisation[f"{node}_renewable_utilisation"] = utilisation.values

    # Calculate the total renewable utilisation across all nodes if not in forecast_df
    all_node_utilisation = forecast_df.get("all_nodes_renewable_utilisation")
    if all_node_utilisation is None:
        all_node_utilisation = renewable_utilisation.sum(axis=1)
        renewable_utilisation["all_nodes_renewable_utilisation"] = (
            all_node_utilisation.values
        )
    else:
        renewable_utilisation["all_nodes_renewable_utilisation"] = (
            all_node_utilisation.values
        )

    return renewable_utilisation


forecast_algorithms = {
    "price_naive_forecast": calculate_naive_price,
    "price_default_test": lambda index, *args: {
        "EOM": FastSeries(index=index, value=50)
    },
    "price_keep_given": None,
    "residual_load_naive_forecast": calculate_naive_residual_load,
    "residual_load_default_test": lambda *args: {},
    "residual_load_keep_given": None,
    "congestion_signal_naive_forecast": calculate_naive_congestion_signal,
    "congestion_signal_default_test": lambda index, *args: FastSeries(
        index=index, value=0.0
    ),
    "congestion_signal_keep_given": None,
    "renewable_utilisation_naive_forecast": calculate_naive_renewable_utilisation,
    "renewable_utilisation_default_test": lambda index, *args: FastSeries(
        index=index, value=0.0
    ),
    "renewable_utilisation_keep_given": None,
}


def default_preprocess(*args, **kwargs):
    return None


def prepare_unit_specific_residual_load_forecasts(
    index: ForecastIndex,
    units: list[BaseUnit],
    market_configs: list[MarketConfig],
    forecast_df: ForecastSeries = None,
    initializing_unit: BaseUnit = None,
):
    unit_name = initializing_unit.id
    preprocess_information = {
        key: forecast_df[key]
        for key in forecast_df.columns
        if unit_name in key and "residual_load" in key
    }

    return preprocess_information


forecast_preprocess_algorithms = {
    "price_default": default_preprocess,
    "residual_load_default": default_preprocess,
    "residual_load_prepare_multiple": prepare_unit_specific_residual_load_forecasts,
    "congestion_signal_default": default_preprocess,
    "renewable_utilisation_default": default_preprocess,
}


def default_update(current_forecast, preprocess_information, *args, **kwargs):
    return current_forecast


def set_preloaded_forecast_by_name(
    current_forecast, preprocess_information, new_forecast_name: str
):
    return preprocess_information[new_forecast_name]


forecast_update_algorithms = {
    "price_default": default_update,
    "residual_load_default": default_update,
    "residual_load_set_preloaded": set_preloaded_forecast_by_name,
    "congestion_signal_default": default_update,
    "renewable_utilisation_default": default_update,
}
