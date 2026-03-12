# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from assume.common.fast_pandas import FastIndex, FastSeries
from assume.common.forecaster import ForecastIndex, ForecastSeries
from assume.common.market_objects import MarketConfig, is_renewable
from assume.common.utils import get_available_products
from assume.markets.clearing_algorithms.simple import PayAsClearRole
from assume.strategies import EnergyHeuristicElasticStrategy
from assume.units.demand import Demand
from assume.units.dsm_load_shift import DSMFlex
from assume.units.exchange import Exchange
from assume.units.powerplant import PowerPlant
from assume.units.storage import Storage

if TYPE_CHECKING:
    from assume.common.base import BaseUnit

log = logging.getLogger(__name__)


def calculate_max_power(units, index=None):
    """
    Returns: max available power: shape (num_units, forecast_len)
    """
    return pd.DataFrame(
        [unit.max_power * unit.forecaster.availability for unit in units], index=index
    )


@lru_cache
def sort_units(units: list[BaseUnit], market_id: str | None = None):
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

    return sum_demand + calculate_exchange_volume(exchange_units)


def calculate_exchange_volume(exchange_units: list[Exchange]):
    """Returns summed exchange volume at every timestep (imports - exports)"""
    sum_demand = 0

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


@lru_cache
def calculate_naive_price_inelastic(
    index: ForecastIndex,
    units: list[BaseUnit],
    config: MarketConfig,
) -> dict[str, ForecastSeries]:
    if isinstance(index, FastIndex):
        index = index.as_datetimeindex()

    # 1. Sort units by type and filter for units with bidding strategy for the given market_id
    powerplants_units, demand_units, exchange_units, _, _ = sort_units(
        units, config.market_id
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

    return price_forecast


@lru_cache
def calculate_naive_price_elastic(
    index: ForecastIndex,
    units: list[BaseUnit],
    config: MarketConfig,
    elastic_demand_units: list[Demand],
) -> dict[str, ForecastSeries]:
    if isinstance(index, FastIndex):
        index = index.as_datetimeindex()

    market_id = config.market_id

    elastic_demand_bids = []
    # 1. Sort units by type and filter for units with bidding strategy for the given market_id
    powerplants_units, demand_units, exchange_units, _, _ = sort_units(units, market_id)

    start = config.opening_hours[0]
    end = start + config.market_products[0].duration

    product_tuples = {(start, end, None)}

    for unit in elastic_demand_units:
        elastic_demand_bids.extend(
            unit.bidding_strategies[market_id].calculate_bids(
                unit,
                config,
                product_tuples=product_tuples,
            )
        )

    # sort all bids by price descending
    all_bids = (
        pd.DataFrame(elastic_demand_bids)
        .sort_values(by="price", ascending=False)
        .reset_index(drop=True)
    )

    elastic_demand_prices = all_bids["price"]
    elastic_demand_volumes = all_bids["volume"]

    # elastic_demand_units = [unit for unit in demand_units
    #                        if isinstance(unit.bidding_strategies[market_id], EnergyHeuristicElasticStrategy)]

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

    # clear the market forecast including elastic demand bids using the PayAsClearRole
    for t in index:
        # get the supply offers
        mc_t = marginal_costs.loc[t]
        power_t = power.loc[t]
        sorted_units = mc_t.sort_values().index
        sorted_mc = mc_t.loc[sorted_units]
        sorted_power = power_t.loc[sorted_units]
        start = t
        end = start + pd.Timedelta(config.market_products[0].duration)
        # Compute the cumulative sum of available power in the sorted order.
        # cumsum_power = sorted_power.cumsum()
        supply_offers = (
            pd.DataFrame(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": None,
                    "node": "node0",  # TODO: ask gugrimm
                    "price": sorted_mc,  # TODO: ask gugrimm if this should be sorted_mc or mc_t
                    "volume": sorted_power,
                }
            )
            .reset_index()
            .rename(columns={"index": "unit_id"})
        )
        # get the demand bids
        demand_t = sum_demand.loc[t]
        demand_bids = (
            pd.DataFrame(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": None,
                    "node": "node0",  # TODO: ask gugrimm
                    "price": elastic_demand_prices,
                    "volume": elastic_demand_volumes,
                }
            )
            .reset_index()
            .rename(columns={"index": "bid_id"})
        )
        # create an orderbook containing all supply offers and demand bids
        orderbook = []
        orderbook.extend(supply_offers.to_dict("records"))
        orderbook.extend(demand_bids.to_dict("records"))
        if demand_t > 0:
            orderbook.append({"price": 3000.0, "volume": demand_t})

        mps = get_available_products(
            config.market_products, pd.Timestamp(start) - pd.Timedelta("1h")
        )
        pac = PayAsClearRole(config)

        accepted, rejected, meta, flows = pac.clear(orderbook, mps)
        price_forecast.loc[t] = meta[0]["price"]

    return price_forecast


@lru_cache
def calculate_naive_price(
    index: ForecastIndex,
    units: list[BaseUnit],
    config: MarketConfig,
    preprocess_information=None,
):
    # 1. Sort units by type and filter for units with bidding strategy for the given market_id
    _, demand_units, _, _, _ = sort_units(units, config.market_id)

    elastic_demand_units = [
        unit
        for unit in demand_units
        if isinstance(
            unit.bidding_strategies[config.market_id], EnergyHeuristicElasticStrategy
        )
    ]

    if len(elastic_demand_units) > 0:
        return calculate_naive_price_elastic(index, units, config, elastic_demand_units)

    return calculate_naive_price_inelastic(index, units, config)


@lru_cache
def calculate_naive_residual_load(
    index: ForecastIndex,
    units: list[BaseUnit],
    config: MarketConfig,
    preprocess_information=None,
) -> dict[str, ForecastSeries]:
    powerplants_units, demand_units, exchange_units, _, _ = sort_units(
        units, config.market_id
    )

    sum_demand = calculate_sum_demand(demand_units, exchange_units)

    # shape: (num_pp_units, index_len) -> (index_len)
    renewable_units = [
        unit for unit in powerplants_units if is_renewable(unit.technology)
    ]
    vre_feed_in_df = calculate_max_power(renewable_units).sum(axis=0)

    if vre_feed_in_df.empty:
        vre_feed_in_df = 0
    res_demand_df = sum_demand - vre_feed_in_df

    return res_demand_df


def extract_buses_and_lines(market_configs: list[MarketConfig]):
    """
    TODO: Recently sure that all market_configs have the same grid data
    """
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


@lru_cache
def calculate_naive_congestion_signal(
    index: ForecastIndex,
    units: list[BaseUnit],
    market_configs: list[MarketConfig],
    preprocess_information=None,
) -> dict[str, ForecastSeries]:
    if isinstance(index, FastIndex):
        index = index.as_datetimeindex()

    # Lines and buses should be everywhere the same
    buses, lines = extract_buses_and_lines(market_configs)

    if buses is None or lines is None:
        return {}

    powerplants_units, demand_units, _, _, _ = sort_units(units)

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


@lru_cache
def calculate_naive_renewable_utilisation(
    index: ForecastIndex,
    units: list[BaseUnit],
    market_configs: list[MarketConfig],
    preprocess_information=None,
) -> dict[str, ForecastSeries]:
    if isinstance(index, FastIndex):
        index = index.as_datetimeindex()

    # Lines and buses should be everywhere the same
    buses, lines = extract_buses_and_lines(market_configs)

    if buses is None or lines is None:
        return {}

    powerplants_units, demand_units, _, _, _ = sort_units(units)

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
        node_renewable_units = [
            unit.id for unit in renewable_units if unit.node == node
        ]
        utilisation = power[node_renewable_units].sum(axis=1)
        renewable_utilisation[f"{node}_renewable_utilisation"] = utilisation.values

    # Calculate the total renewable utilisation across all nodes
    all_node_utilisation = renewable_utilisation.sum(axis=1)
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


def get_forecast_registries() -> dict[str, dict]:
    """Returns the three algorithm registry dicts bundled for injection into forecasters."""
    return {
        "init": forecast_algorithms,
        "preprocess": forecast_preprocess_algorithms,
        "update": forecast_update_algorithms,
    }
