# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from assume.common.fast_pandas import FastIndex, FastSeries
from assume.common.forecaster import ForecastIndex, ForecastSeries
from assume.common.market_objects import MarketConfig, is_renewable
from assume.markets.clearing_algorithms.simple import clear_merit_order
from assume.strategies import EnergyHeuristicElasticStrategy
from assume.units.demand import Demand
from assume.units.dsm_load_shift import DSMFlex
from assume.units.exchange import Exchange
from assume.units.powerplant import PowerPlant
from assume.units.storage import Storage

if TYPE_CHECKING:
    from assume.common.base import BaseUnit

log = logging.getLogger(__name__)


def is_elastic_demand(unit, market_config=None) -> bool:
    """
    Checks whether a unit is an elastic demand.

    .. note::
        There is currently not a clear flag whether some demand is elastic.
        Until then, it is defined via its bidding strategy on a given market.
        If no market given we use the same criterion the Demand class uses itself:
        ``unit.elasticity_model != 0``
    """
    if market_config is not None:
        return isinstance(
            unit.bidding_strategies[market_config.market_id],
            EnergyHeuristicElasticStrategy,
        )

    if isinstance(unit, Demand):
        return unit.elasticity_model != 0

    return False


def calculate_max_power(units, index=None):
    """
    Returns: max available power: shape (num_units, forecast_len)
    """
    return pd.DataFrame(
        [unit.max_power * unit.forecaster.availability for unit in units], index=index
    )


@lru_cache
def sort_units(units: list[BaseUnit], market_id: str | None = None):
    """
    Classify units into powerplants, demands, exchanges, storages, and DSM units.

    If *market_id* is given, only units with a bidding strategy for that market are included.
    """
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
    sum_demand = np.zeros(len(demand_units[0].forecaster.index))

    sum_demand += abs(
        np.array(
            [
                unit.forecaster.demand
                for unit in demand_units
                if not is_elastic_demand(unit)
            ]
        )
    ).sum(axis=0)

    return sum_demand + calculate_exchange_volume(exchange_units)


def calculate_exchange_volume(exchange_units: list[Exchange]):
    """Returns summed exchange volume at every timestep (exports - imports)

    Imports are supply and reduce the demand, exports are demand and increase it.
    """
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
        # imports reduce and exports increase the sum_demand
        sum_demand += sum_exports - sum_imports

    return sum_demand


def _stack_bids(series: list, length: int) -> np.ndarray:
    """
    Stacks per-bid scalars or time series to an array of shape (length, num_bids).

    The array is filled row-contiguous, so that the bids of a timestep can be read without copying.
    """
    bids = np.empty((length, len(series)))
    for k, s in enumerate(series):
        bids[:, k] = np.asarray(s, dtype=float)
    return bids


@lru_cache
def calculate_naive_price(
    index: ForecastIndex,
    units: list[BaseUnit],
    config: MarketConfig,
    preprocess_information=None,
) -> ForecastSeries:
    """
    Naive price forecast with a merit order of the units' own bids, cleared like
    :class:`~assume.markets.clearing_algorithms.simple.PayAsClearRole` and
    :class:`~assume.markets.clearing_algorithms.simple.PayAsBidRole` with
    :func:`~assume.markets.clearing_algorithms.simple.clear_merit_order`.

    Storages and DSM units are ignored in this calculation. Every market mechanism is forecasted
    this way, so for other mechanisms (e.g. complex_clearing or nodal_clearing) block orders,
    linked orders and grid constraints are not considered. Such markets can register their own
    price forecast instead.

    Bids at each timestep:
        - **power plants** offer their available power at marginal cost,
        - **exchanges** offer their imports at ``price_import`` and bid their exports at ``price_export``,
        - **inelastic demands** bid their forecasted demand at their price,
        - **elastic demands** bid the bids of their bidding strategy for the first product,
          which are the same for every timestep.

    For ``pay_as_bid`` the forecast price is the volume-weighted average price of the accepted
    supply, otherwise the clearing price. If nothing is traded, the highest willingness to pay
    sets the price. Without any demand, the cheapest supply sets the price.
    """
    if config.market_mechanism not in ("pay_as_clear", "pay_as_bid"):
        log.info(
            "Price forecast for market '%s' (%s) uses a merit order of simple bids, "
            "block orders and grid constraints are not considered.",
            config.market_id,
            config.market_mechanism,
        )

    if isinstance(index, FastIndex):
        index = index.as_datetimeindex()
    n = len(index)

    # 1. Sort units by type and filter for units with bidding strategy for the given market_id
    powerplant_units, demand_units, exchange_units, _, _ = sort_units(
        units, config.market_id
    )
    elastic_demand_units = [u for u in demand_units if is_elastic_demand(u, config)]
    inelastic_demand_units = [
        u for u in demand_units if not is_elastic_demand(u, config)
    ]

    # 2. Build the bids, shape: (index_len, num_bids)
    supply_price = _stack_bids(
        [unit.marginal_cost for unit in powerplant_units]
        + [unit.price_import for unit in exchange_units],
        n,
    )
    supply_volume = _stack_bids(
        [unit.max_power * unit.forecaster.availability for unit in powerplant_units]
        + [unit.volume_import for unit in exchange_units],
        n,
    )
    demand_price = _stack_bids(
        [unit.price for unit in inelastic_demand_units]
        + [unit.price_export for unit in exchange_units],
        n,
    )
    demand_volume = np.abs(
        _stack_bids(
            [unit.forecaster.demand for unit in inelastic_demand_units]
            + [unit.volume_export for unit in exchange_units],
            n,
        )
    )

    if elastic_demand_units:
        start = config.opening_hours[0]
        end = start + config.market_products[0].duration
        product_tuples = [(start, end, None)]
        elastic_bids = [
            bid
            for unit in elastic_demand_units
            for bid in unit.bidding_strategies[config.market_id].calculate_bids(
                unit, config, product_tuples=product_tuples
            )
        ]
        demand_price = np.hstack(
            [demand_price, _stack_bids([bid["price"] for bid in elastic_bids], n)]
        )
        demand_volume = np.hstack(
            [
                demand_volume,
                np.abs(_stack_bids([bid["volume"] for bid in elastic_bids], n)),
            ]
        )

    # 3. Merit-order dispatch
    pay_as_bid = config.market_mechanism == "pay_as_bid"
    prices = np.zeros(n)
    for t in range(n):
        offered = supply_volume[t] > 0
        demanded = demand_volume[t] > 0
        if not demanded.any():
            # without demand (e.g. imports exceed the demand) the cheapest supply sets the price
            prices[t] = supply_price[t][offered].min() if offered.any() else 0.0
            continue

        clear_price, supply_accepted, demand_accepted, _ = clear_merit_order(
            supply_price[t],
            supply_volume[t],
            demand_price[t],
            demand_volume[t],
        )
        if not demand_accepted.any():
            # no supply is cheaper than the willingness to pay, so nothing is traded
            prices[t] = demand_price[t][demanded].max()
        elif pay_as_bid:
            prices[t] = supply_accepted @ supply_price[t] / supply_accepted.sum()
        else:
            prices[t] = clear_price

    return pd.Series(prices, index=index)


@lru_cache
def calculate_naive_residual_load(
    index: ForecastIndex,
    units: list[BaseUnit],
    config: MarketConfig,
    preprocess_information=None,
) -> dict[str, ForecastSeries]:
    """Compute residual load as total demand minus renewable generation for each timestep.

    NOTE: Elastic demands are ignored in this forecast.
          This will underestimate the residual load if there are elastic demands present.
    """
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
    Extract bus and line DataFrames from the first market config that carries grid data.
    NOTE: Currently all scenario loaders give grid data to all markets so this is maybe overkill
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
    """
    Compute per-node congestion severity signals from net load and line capacities.

    Node congestion forecast resembles::

        max(line congestion of connected lines)
        with line congestion = (demand - supply) / line capacity

    Steps:
        1. **Net load per node** — for each demand node, subtract local generation from
           local demand to obtain the net load timeseries.
        2. **Line congestion severity** — for each transmission line, divide the combined
           net load of its two endpoint nodes by the line's thermal capacity.
        3. **Node aggregation** — for each node, take the maximum congestion severity
           across all connected lines as the node's congestion signal.

    Returns an empty dict if grid data (buses/lines) is unavailable.

    .. note::
        Elastic demands are ignored currently.
    """
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

    # Go on if only elastic demand (as they are ignored)
    if all([is_elastic_demand(unit) for unit in demand_units]):
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
    """
    Compute per-node renewable generation (availability * max_power) and an all-nodes total.

    Returns a DataFrame with columns ``{node}_renewable_utilisation`` for each demand node
    and ``all_nodes_renewable_utilisation`` for the aggregate. Returns an empty dict if
    grid data is unavailable.
    """
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

    if len(renewable_units) == 0:
        return {}

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
