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
from assume.common.utils import get_available_products
from assume.markets.clearing_algorithms.complex_clearing import ComplexClearingRole
from assume.markets.clearing_algorithms.simple import PayAsBidRole, PayAsClearRole
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
    Checks whether a unit as an elastic demand.

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
    """
    Forecast market clearing prices using a merit-order stack against inelastic demand.

    Storages and DSM units are ignored in this calculation.

    Steps:
        1. **Sort units** by type, keeping only those with a bidding strategy for the market.
        2. **Build supply and demand curves** — compute per-unit marginal costs and
           available power, then sum demand (including exchange volumes) for each timestep.
        3. **Merit-order dispatch** — sort supply by ascending marginal cost, stack capacity
           until demand is met, and set the clearing price to the marginal unit's cost
           (defaults to 1000 if capacity is insufficient).
    """
    if isinstance(index, FastIndex):
        index = index.as_datetimeindex()

    # 1. Sort units by type and filter for units with bidding strategy for the given market_id
    powerplants_units, demand_units, exchange_units, _, _ = sort_units(
        units, config.market_id
    )

    # 2. Build supply and demand curves
    # Calculate marginal costs for each unit and time step.
    # The resulting DataFrame has rows = time steps and columns = units.
    # shape: (index_len, num_pp_units)
    marginal_costs = pd.DataFrame(
        [unit.marginal_cost for unit in powerplants_units]
    ).T.set_index(index)

    # Compute available power for each unit at each time step.
    # shape: (index_len, num_pp_units)
    power = calculate_max_power(powerplants_units).T.set_index(index)

    # Process the demand.
    # Filter demand units with a bidding strategy and sum their forecasts for each time step.
    sum_demand = pd.DataFrame(
        calculate_sum_demand(demand_units, exchange_units), index=index
    )

    # 3. Merit-order dispatch
    # Initialize the price forecast series.
    price_forecast = pd.Series(index=index, data=0.0)

    # Loop over each time step
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
            # If available capacity is insufficient, set the price to max willingnes to pay.
            price = max([unit.price[t] for unit in demand_units])
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
    """
    Forecast market clearing prices with price-elastic demand via pay-as-clear matching.

    Storages and DSM units are ignored in this calculation.

    Steps:
        1. **Sort units and collect elastic bids** — classify units by type and compute
           demand bids from elastic demand units for the first product interval.
        2. **Build supply and demand curves** — compute per-unit marginal costs and
           available power, then sum inelastic demand (including exchange volumes).
        3. **Pay-as-clear dispatch** — for each timestep, assemble an orderbook from
           supply offers, elastic demand bids, and the inelastic demand block, then
           clear via ``PayAsClearRole`` to obtain the equilibrium price.
    """
    if isinstance(index, FastIndex):
        index = index.as_datetimeindex()

    market_id = config.market_id

    elastic_demand_bids = []
    # 1. Sort units by type and filter for units with bidding strategy for the given market_id
    powerplants_units, demand_units, exchange_units, _, _ = sort_units(units, market_id)
    inelastic_demand_units = [
        unit for unit in demand_units if unit not in elastic_demand_units
    ]

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

    # 2. Build supply and demand curves
    # Calculate marginal costs for each unit and time step.
    # The resulting DataFrame has rows = time steps and columns = units.
    # shape: (index_len, num_pp_units)
    marginal_costs = pd.DataFrame(
        [unit.marginal_cost for unit in powerplants_units]
    ).T.set_index(index)

    # Compute available power for each unit at each time step.
    # shape: (index_len, num_pp_units)
    power = calculate_max_power(powerplants_units).T.set_index(index)

    # Process the inelastic demand.
    # Filter demand units with a bidding strategy and sum their forecasts for each time step.
    sum_demand = pd.DataFrame(
        calculate_sum_demand(demand_units, exchange_units), index=index
    )

    # 3. Pay-as-clear dispatch
    # Initialize the price forecast series.
    price_forecast = pd.Series(index=index, data=0.0)

    # clear the market forecast including elastic demand bids using the PayAsClearRole
    for t in index:
        # get the supply offers (marginal cost and available power) for time t
        mc_t = marginal_costs.loc[t]
        power_t = power.loc[t]
        start = t
        end = start + pd.Timedelta(config.market_products[0].duration)

        supply_offers = pd.DataFrame(
            {
                "start_time": start,
                "end_time": end,
                "only_hours": None,
                "node": "node0",
                "price": mc_t,
                "volume": power_t,
                "bid_type": "SB",
                "bid_id": [f"{unit.id}_{t}" for unit in powerplants_units],
            }
        )

        # shape of sum_demand: (time_steps, 1)
        demand_t = sum_demand.loc[t][0]

        # get the demand bids
        demand_bids = pd.DataFrame(
            {
                "start_time": start,
                "end_time": end,
                "only_hours": None,
                "node": "node0",
                "price": elastic_demand_prices,
                "volume": elastic_demand_volumes,
                "bid_type": "SB",
                "bid_id": [
                    f"elastic_demand_{t}_{i}" for i in range(len(elastic_demand_prices))
                ],
            }
        )

        # create an orderbook containing all supply offers and demand bids
        orderbook = []
        orderbook.extend(supply_offers.to_dict("records"))
        orderbook.extend(demand_bids.to_dict("records"))
        if demand_t > 0 and len(inelastic_demand_units) > 0:
            inelastic_price_bid = max(
                [unit.price[t] for unit in inelastic_demand_units]
            )
            orderbook.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": None,
                    "node": "node0",
                    "price": inelastic_price_bid,
                    "volume": -demand_t,
                    "bid_type": "SB",
                    "bid_id": f"{inelastic_demand_units[0].id}_{t}",
                }
            )

        cleaned_orderbook = []
        for bid in orderbook:
            if isinstance(bid["volume"], dict):
                if all(volume == 0 for volume in bid["volume"].values()):
                    continue
            elif bid["volume"] == 0:
                continue
            cleaned_orderbook.append(bid)

        mps = get_available_products(
            config.market_products, pd.Timestamp(start) - pd.Timedelta("1h")
        )

        if config.market_mechanism == "pay_as_bid":
            # the forecast price is the volume-weighted average price of matched orders of each timestep
            mechanism = PayAsBidRole(config)
        elif config.market_mechanism == "pay_as_clear":
            mechanism = PayAsClearRole(config)
        elif config.market_mechanism == "complex_clearing":
            mechanism = ComplexClearingRole(config)
        else:
            raise ValueError(
                f"Invalid market mechanism {config.param_dict.get('market_mechanism')}."
            )

        _, _, meta, _ = mechanism.clear(cleaned_orderbook, mps)
        price_forecast.loc[t] = meta[0]["price"]

    return price_forecast


@lru_cache
def calculate_naive_price(
    index: ForecastIndex,
    units: list[BaseUnit],
    config: MarketConfig,
    preprocess_information=None,
):
    """Calculates elastic or inelastic naive price forecast depending on demand unit types."""
    # 1. Sort units by type and filter for units with bidding strategy for the given market_id
    _, demand_units, _, _, _ = sort_units(units, config.market_id)

    elastic_demand_units = {
        unit.id: unit for unit in demand_units if is_elastic_demand(unit, config)
    }

    if len(elastic_demand_units) > 0:
        return calculate_naive_price_elastic(
            index, units, config, elastic_demand_units.values()
        )

    return calculate_naive_price_inelastic(index, units, config)


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


@lru_cache
def _log_once(message: str) -> None:
    log.info(message)


def get_node_to_zone(config: MarketConfig) -> dict[str, str] | None:
    """
    Returns the mapping of the buses of a market to its price zones.

    With a ``zones_identifier`` in the ``param_dict`` of the market, buses are grouped into
    zones (zonal clearing, as in ``ComplexClearingRole``), otherwise every bus is its own zone
    (nodal clearing). Returns None if the market has no grid data (single price zone).
    """
    grid_data = config.param_dict.get("grid_data") or {}
    buses = grid_data.get("buses")
    if buses is None or len(buses) == 0:
        return None
    zones_id = config.param_dict.get("zones_identifier")
    if zones_id and zones_id in buses.columns:
        return buses[zones_id].to_dict()
    return {bus: bus for bus in buses.index}


def get_unit_zone(unit: BaseUnit, config: MarketConfig) -> str | None:
    """Price zone of a unit in a market, None if the market has a single price zone."""
    node_to_zone = get_node_to_zone(config)
    if node_to_zone is None:
        return None
    return node_to_zone.get(unit.node, unit.node)


def _merit_order_clearing(
    supply_price: np.ndarray,
    supply_volume: np.ndarray,
    demand_price: np.ndarray,
    demand_volume: np.ndarray,
) -> float:
    """
    Uniform price of one time step from supply and demand bids (all volumes positive).

    The traded volume is the largest volume at which the price of the supply step does not
    exceed the price of the demand step. The price is set by the bid which is only partially
    accepted at this volume: a supply bid, or a demand bid (e.g. price-sensitive demand or
    scarcity, if the supply is not sufficient). If both marginal bids are fully accepted, the
    last accepted supply bid sets the price (as in the naive forecast). Without demand the price
    is set by the cheapest supply bid.
    """
    supply = supply_volume > 0
    supply_price, supply_volume = supply_price[supply], supply_volume[supply]
    demand = demand_volume > 0
    demand_price, demand_volume = demand_price[demand], demand_volume[demand]

    if len(demand_price) == 0:
        return float(supply_price.min()) if len(supply_price) else 0.0
    if len(supply_price) == 0:
        return float(demand_price.max())

    supply_order = np.argsort(supply_price, kind="stable")
    supply_price = supply_price[supply_order]
    cum_supply = np.cumsum(supply_volume[supply_order])
    demand_order = np.argsort(-demand_price, kind="stable")
    demand_price = demand_price[demand_order]
    cum_demand = np.cumsum(demand_volume[demand_order])

    # the price steps just below each candidate volume decide whether it is traded
    candidates = np.concatenate((cum_supply, cum_demand))
    supply_step = np.searchsorted(cum_supply, candidates - 1e-9)
    demand_step = np.searchsorted(cum_demand, candidates - 1e-9)
    step_supply_price = np.where(
        supply_step < len(supply_price),
        supply_price[np.minimum(supply_step, len(supply_price) - 1)],
        np.inf,
    )
    step_demand_price = np.where(
        demand_step < len(demand_price),
        demand_price[np.minimum(demand_step, len(demand_price) - 1)],
        -np.inf,
    )
    traded = candidates[step_supply_price <= step_demand_price]
    volume = traded.max() if len(traded) else 0.0
    if volume <= 0:
        return float(supply_price[0])

    # marginal bids: the steps which contain the last traded MWh
    supply_step = np.searchsorted(cum_supply, volume - 1e-9)
    demand_step = np.searchsorted(cum_demand, volume - 1e-9)
    supply_partial = cum_supply[supply_step] > volume + 1e-9
    demand_partial = demand_step < len(cum_demand) and (
        cum_demand[demand_step] > volume + 1e-9
    )
    if demand_partial and not supply_partial:
        return float(demand_price[demand_step])
    if not supply_partial and not demand_partial and demand_step + 1 < len(cum_demand):
        # demand bids remain unserved after a fully accepted demand bid: the next demand
        # bid sets the price if the supply is exhausted (scarcity)
        if supply_step + 1 >= len(cum_supply):
            return float(demand_price[demand_step + 1])
    return float(supply_price[supply_step])


@lru_cache
def calculate_zonal_merit_order_prices(
    index: ForecastIndex,
    units: list[BaseUnit],
    config: MarketConfig,
) -> dict[str | None, pd.Series]:
    """
    Forecast the market clearing price of every price zone with a merit order per zone.

    The supply of a zone are the power plants (marginal cost, available power) and the
    imports of the exchange units of the zone, the demand are the demand units (bid price,
    demand) and the exports of the exchange units. Flows between the zones are not
    considered. Storages, DSM units and elastic demand units are ignored.

    Returns:
        dict[str | None, pd.Series]: Price forecast per zone (key None for a market without
        grid data, i.e. a single price zone).
    """
    if isinstance(index, FastIndex):
        index = index.as_datetimeindex()
    num_steps = len(index)

    powerplant_units, demand_units, exchange_units, _, _ = sort_units(
        units, config.market_id
    )
    elastic = [unit for unit in demand_units if is_elastic_demand(unit, config)]
    if elastic:
        log.warning(
            "Elastic demand units are ignored in the zonal merit order price forecast: %s",
            [unit.id for unit in elastic],
        )

    def series(values) -> np.ndarray:
        return np.broadcast_to(np.asarray(values, dtype=float), num_steps)

    # supply and demand bids per zone: lists of (price, volume) arrays over time
    supply: dict = {}
    demand: dict = {}
    for unit in powerplant_units:
        supply.setdefault(get_unit_zone(unit, config), []).append(
            (
                series(unit.marginal_cost),
                series(unit.max_power * series(unit.forecaster.availability)),
            )
        )
    for unit in demand_units:
        if unit in elastic:
            continue
        demand.setdefault(get_unit_zone(unit, config), []).append(
            (series(unit.price), np.abs(series(unit.forecaster.demand)))
        )
    for unit in exchange_units:
        zone = get_unit_zone(unit, config)
        supply.setdefault(zone, []).append(
            (
                series(unit.price_import),
                np.abs(series(unit.forecaster.volume_import)),
            )
        )
        demand.setdefault(zone, []).append(
            (
                series(unit.price_export),
                np.abs(series(unit.forecaster.volume_export)),
            )
        )

    def stack(bids: list) -> tuple[np.ndarray, np.ndarray]:
        if not bids:
            return np.zeros((0, num_steps)), np.zeros((0, num_steps))
        prices, volumes = zip(*bids)
        return np.vstack(prices), np.vstack(volumes)

    prices = {}
    for zone in supply.keys() | demand.keys():
        supply_price, supply_volume = stack(supply.get(zone, []))
        demand_price, demand_volume = stack(demand.get(zone, []))
        prices[zone] = pd.Series(
            [
                _merit_order_clearing(
                    supply_price[:, t],
                    supply_volume[:, t],
                    demand_price[:, t],
                    demand_volume[:, t],
                )
                for t in range(num_steps)
            ],
            index=index,
        )
    return prices


def calculate_zonal_merit_order_price(
    index: ForecastIndex,
    units: list[BaseUnit],
    config: MarketConfig,
    preprocess_information=None,
) -> ForecastSeries:
    """
    Price forecast of the price zone of a unit, from a merit order per zone.

    The zone of the unit has to be determined in the preprocess step
    (``preprocess_price: price_unit_zone``). The prices of all zones are calculated once
    (see :func:`calculate_zonal_merit_order_prices`). Forecasters without a unit (e.g. of unit
    operators) fall back to the naive price forecast of the whole market.
    """
    zone_information = (preprocess_information or {}).get(config.market_id)
    if zone_information is None:
        _log_once(
            "price_zonal_merit_order: forecaster without zone information (forecasters of "
            "unit operators, or units without 'preprocess_price: price_unit_zone') use the "
            "naive price forecast of the whole market"
        )
        return calculate_naive_price(index, units, config, None)
    if zone_information.get("given") is not None:
        return zone_information["given"]

    prices = calculate_zonal_merit_order_prices(index, units, config)
    zone = zone_information["zone"]
    if zone not in prices:
        # no bids in the zone of the unit
        return calculate_naive_price(index, units, config, None)
    return prices[zone]


forecast_algorithms = {
    "price_naive_forecast": calculate_naive_price,
    "price_zonal_merit_order": calculate_zonal_merit_order_price,
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


def prepare_unit_zone(
    index: ForecastIndex,
    units: list[BaseUnit],
    market_configs: list[MarketConfig],
    forecast_df: ForecastSeries = None,
    initializing_unit: BaseUnit = None,
) -> dict[str, dict] | None:
    """
    Determines the price zone of the initializing unit in every market.

    A price forecast of the zone can be given in ``forecast_df`` as column
    ``price_{market_id}_{zone}``, it is then used instead of the calculated one.

    Returns:
        dict[str, dict] | None: ``{market_id: {"zone": zone, "given": series or None}}``,
        None for forecasters without a unit (e.g. of unit operators).
    """
    if initializing_unit is None:
        return None
    zones = {}
    for config in market_configs:
        zone = get_unit_zone(initializing_unit, config)
        given = None
        if forecast_df is not None and zone is not None:
            given = forecast_df.get(f"price_{config.market_id}_{zone}")
        zones[config.market_id] = {"zone": zone, "given": given}
    return zones


forecast_preprocess_algorithms = {
    "price_default": default_preprocess,
    "price_unit_zone": prepare_unit_zone,
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
