# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch as th

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


def calculate_adaptive_merit_order_forecast_inputs(
    index: FastIndex,
    units: tuple[BaseUnit, ...],
    config: MarketConfig,
) -> dict[str, FastSeries]:
    """Calculate the forecast-time inputs for adaptive merit-order correction.

    The existing merit-order price and residual-load algorithms remain the
    baseline. Wind and solar availability are returned separately as
    capacity-weighted factors.
    """
    if config.product_type != "energy":
        raise ValueError("Adaptive merit-order forecasts support energy markets only")
    if config.param_dict.get("grid_data"):
        raise ValueError(
            "Adaptive merit-order forecasts do not support nodal or zonal markets"
        )
    if (
        config.market_mechanism == "pay_as_bid"
        or config.param_dict.get("pricing_mechanism") == "pay_as_bid"
    ):
        raise ValueError(
            "Adaptive merit-order forecasts require one scalar clearing price "
            "per product"
        )

    if not units:
        raise ValueError("Adaptive merit-order forecasts require at least one unit")

    powerplants, _, _, _, _ = sort_units(units, config.market_id)

    def availability_factor(technology: str) -> FastSeries:
        selected_units = []
        for unit in powerplants:
            unit_technology = (
                str(unit.technology).casefold().replace("_", " ").replace("-", " ")
            )
            if technology in unit_technology.split():
                selected_units.append(unit)
        capacity = sum(float(unit.max_power) for unit in selected_units)
        if capacity == 0:
            return FastSeries(index=index, value=0.0)
        available_capacity = calculate_max_power(selected_units).sum(axis=0)
        factor = available_capacity / capacity
        if np.any(factor < 0) or np.any(factor > 1):
            raise ValueError(
                "Renewable availability factors must be between zero and one"
            )
        return FastSeries(index=index, value=factor.to_numpy())

    merit_order_price = calculate_naive_price(index, units, config)
    residual_load = calculate_naive_residual_load(index, units, config)

    return {
        "merit_order_price": FastSeries(index=index, value=merit_order_price),
        "wind_availability_factor": availability_factor("wind"),
        "solar_availability_factor": availability_factor("solar"),
        "residual_load": FastSeries(index=index, value=residual_load),
    }


ADAPTIVE_MERIT_ORDER_FEATURES = (
    # Selected by chronological LASSO screening on example_03, using January
    # through September 2019 for selection and a separate October--December
    # holdout for confirmation. Together they model system tightness, daily
    # persistence, and the weekday/weekend market regime.
    "merit_order_price",
    "wind_availability_factor",
    "solar_availability_factor",
    "residual_load",
    "previous_day_same_hour_residual",
    "previous_day_same_hour_price",
    "weekday",
    "weekend",
)

ADAPTIVE_MERIT_ORDER_SUPPORTED_FEATURES = (
    "merit_order_price",
    "wind_availability_factor",
    "solar_availability_factor",
    "residual_load",
    "previous_day_same_hour_residual",
    "previous_day_same_hour_price",
    # Evaluated but not selected as a default: its incremental contribution
    # was inconsistent once weekday and weekend were included.
    "hour",
    "weekday",
    "weekend",
    # Optional calendar feature. It remains available when a holiday calendar
    # is supplied explicitly; it was not included in the 2019 feature screen.
    "holiday",
)

ADAPTIVE_MERIT_ORDER_SETTINGS = {
    # Number of cleared hourly products collected before the first model fit.
    "minimum_training_samples": 504,
    # Weight retained from one online update to the next; lower values forget
    # older market outcomes more quickly.
    "forgetting_factor": 0.995,
    # L1 selects inputs for the expected-residual equation; L2 stabilises
    # correlated coefficients. The intercept is excluded from both penalties.
    "residual_mean_l1_regularization": 0.01,
    "residual_mean_l2_regularization": 0.001,
    # The same penalties for the log-standard-deviation equation.
    "residual_scale_l1_regularization": 0.01,
    "residual_scale_l2_regularization": 0.001,
    # Predictive distribution used to derive price quantiles.
    "distribution": "gaussian",
    # Johnson SU shape fitting uses only previous standardised forecast errors.
    "johnson_su_history_size": 1008,
    "johnson_su_solver_iterations": 5,
    "johnson_su_learning_rate": 0.01,
    # Inputs for the residual-mean model. A None scale feature list reuses these.
    "features": ADAPTIVE_MERIT_ORDER_FEATURES,
    "scale_features": None,
    # Enable the holiday input only when dates are supplied by a calendar.
    "use_holiday_feature": False,
    "holiday_dates": (),
    # Smallest permitted predicted standard deviation, in price units.
    "sigma_floor": 0.01,
    # Outer Fisher-scoring/IRLS convergence limits for the two Gaussian
    # distributional equations.
    "irls_max_iterations": 25,
    "irls_tolerance": 1e-6,
    # Inner coordinate-descent convergence limits for each weighted fit.
    "solver_max_iterations": 1000,
    "solver_tolerance": 1e-8,
}


def initialize_adaptive_merit_order_feature_state(features, holiday_dates=()) -> dict:
    """Create feature metadata; scaling is fitted on the initial window later."""
    features = tuple(features)
    allowed = set(ADAPTIVE_MERIT_ORDER_SUPPORTED_FEATURES)
    unknown = set(features) - allowed
    if unknown:
        raise ValueError(f"Unknown adaptive merit-order features: {sorted(unknown)}")
    if len(features) != len(set(features)):
        raise ValueError("Adaptive merit-order features must not contain duplicates")

    names = ["intercept"]
    continuous = [False]
    for feature in features:
        if feature == "hour":
            names.extend(("hour_sin", "hour_cos"))
            continuous.extend((True, True))
        elif feature == "weekday":
            names.extend(("weekday_sin", "weekday_cos"))
            continuous.extend((True, True))
        elif feature in (
            "previous_day_same_hour_residual",
            "previous_day_same_hour_price",
        ):
            names.extend((feature, f"{feature}_missing"))
            continuous.extend((True, False))
        else:
            names.append(feature)
            continuous.append(feature not in ("weekend", "holiday"))
    return {
        "features": features,
        "feature_names": tuple(names),
        "continuous": th.tensor(continuous, dtype=th.bool),
        "holiday_dates": frozenset(holiday_dates),
        "means": None,
        "scales": None,
    }


def build_adaptive_merit_order_feature_vector(
    feature_state: dict, inputs: dict
) -> th.Tensor:
    """Build one unscaled forecast-time feature vector."""
    values = [1.0]
    delivery_time = inputs["delivery_time"]
    for feature in feature_state["features"]:
        if feature in (
            "merit_order_price",
            "wind_availability_factor",
            "solar_availability_factor",
            "residual_load",
        ):
            values.append(inputs[feature])
        elif feature in (
            "previous_day_same_hour_residual",
            "previous_day_same_hour_price",
        ):
            value = inputs.get(feature)
            values.extend(
                (float("nan") if value is None else value, float(value is None))
            )
        elif feature == "hour":
            angle = th.tensor(2 * th.pi * delivery_time.hour / 24, dtype=th.float64)
            values.extend((th.sin(angle).item(), th.cos(angle).item()))
        elif feature == "weekday":
            angle = th.tensor(2 * th.pi * delivery_time.weekday() / 7, dtype=th.float64)
            values.extend((th.sin(angle).item(), th.cos(angle).item()))
        elif feature == "weekend":
            values.append(float(delivery_time.weekday() >= 5))
        elif feature == "holiday":
            values.append(
                float(
                    inputs.get("is_holiday", False)
                    or delivery_time.date() in feature_state["holiday_dates"]
                )
            )
    return th.tensor(values, dtype=th.float64)


def fit_adaptive_merit_order_feature_scaling(
    feature_state: dict, rows: list[dict]
) -> None:
    """Fit and freeze continuous-feature scaling on chronological rows."""
    if not rows:
        raise ValueError("At least one feature row is required to fit scaling")
    matrix = th.stack(
        [build_adaptive_merit_order_feature_vector(feature_state, row) for row in rows]
    )
    feature_state["means"] = th.zeros(matrix.shape[1], dtype=th.float64)
    feature_state["scales"] = th.ones(matrix.shape[1], dtype=th.float64)
    for column in th.nonzero(feature_state["continuous"]).flatten().tolist():
        observed = matrix[:, column][th.isfinite(matrix[:, column])]
        if observed.numel():
            feature_state["means"][column] = observed.mean()
            deviation = observed.std(unbiased=False)
            feature_state["scales"][column] = deviation if deviation > 1e-12 else 1.0


def transform_adaptive_merit_order_features(
    feature_state: dict, inputs: dict
) -> th.Tensor:
    """Transform one row with frozen scaling and safe mean imputation."""
    if feature_state["means"] is None or feature_state["scales"] is None:
        raise RuntimeError("Feature scaling must be fitted before transformation")
    result = build_adaptive_merit_order_feature_vector(feature_state, inputs)
    for column in th.nonzero(feature_state["continuous"]).flatten().tolist():
        if not th.isfinite(result[column]):
            result[column] = feature_state["means"][column]
        result[column] = (
            result[column] - feature_state["means"][column]
        ) / feature_state["scales"][column]
    return result


def initialize_online_regularized_regression(
    n_features: int,
    l1: float,
    l2: float,
    forgetting_factor: float,
    max_iterations: int,
    tolerance: float,
) -> dict:
    """Create discounted statistics for an online L1/L2 regression."""
    if n_features <= 0 or l1 < 0 or l2 < 0:
        raise ValueError("Invalid regression dimensions or penalties")
    if not 0 < forgetting_factor <= 1:
        raise ValueError("forgetting_factor must be in (0, 1]")
    return {
        "n_features": n_features,
        "l1": l1,
        "l2": l2,
        "forgetting_factor": forgetting_factor,
        "max_iterations": max_iterations,
        "tolerance": tolerance,
        "gram": th.zeros((n_features, n_features), dtype=th.float64),
        "target_moment": th.zeros(n_features, dtype=th.float64),
        "coefficients": th.zeros(n_features, dtype=th.float64),
        "effective_weight": 0.0,
    }


def solve_online_regularized_regression(model: dict) -> None:
    """Solve weighted L1/L2 regression by warm-start coordinate descent.

    The normalised objective is

    ``0.5 * weighted_squared_error + l1 * |beta| + 0.5 * l2 * beta**2``.

    The intercept in column zero is excluded from both penalties.
    """
    if model["effective_weight"] <= 0:
        raise ValueError("Regression requires a positive effective weight")
    gram = model["gram"] / model["effective_weight"]
    target_moment = model["target_moment"] / model["effective_weight"]
    coefficients = model["coefficients"].clone()
    for _ in range(model["max_iterations"]):
        previous = coefficients.clone()
        for column in range(model["n_features"]):
            diagonal = gram[column, column]
            if diagonal <= 1e-15:
                coefficients[column] = 0
                continue
            partial = target_moment[column] - (
                gram[column] @ coefficients - diagonal * coefficients[column]
            )
            if column == 0:
                coefficients[column] = partial / diagonal
            else:
                thresholded = th.sign(partial) * th.clamp(
                    th.abs(partial) - model["l1"], min=0
                )
                coefficients[column] = thresholded / (diagonal + model["l2"])
        if th.max(th.abs(coefficients - previous)).item() <= model["tolerance"]:
            break
    model["coefficients"] = coefficients


def fit_online_regularized_regression(
    model: dict, features, target, weights=None
) -> None:
    """Fit one weighted regression and retain its sufficient statistics."""
    features = th.as_tensor(features, dtype=th.float64)
    target = th.as_tensor(target, dtype=th.float64)
    if features.shape != (target.numel(), model["n_features"]):
        raise ValueError("features or target have the wrong shape")
    if not th.all(th.isfinite(features)) or not th.all(th.isfinite(target)):
        raise ValueError("Training data must be finite")
    if weights is None:
        weights = th.ones(target.numel(), dtype=th.float64)
    else:
        weights = th.as_tensor(weights, dtype=th.float64)
    if weights.shape != target.shape or not th.all(th.isfinite(weights)):
        raise ValueError("weights have the wrong shape or are not finite")
    if th.any(weights <= 0):
        raise ValueError("Regression weights must be positive")
    model["gram"] = features.T @ (weights[:, None] * features)
    model["target_moment"] = features.T @ (weights * target)
    model["effective_weight"] = weights.sum().item()
    solve_online_regularized_regression(model)


def update_online_regularized_regression(
    model: dict, features, target, weight=1.0, previous_statistics=None
) -> None:
    """Add one weighted observation to discounted sufficient statistics.

    ``previous_statistics`` lets IRLS reconsider the newest observation without
    counting it repeatedly: every outer iteration starts from the same frozen
    pre-outcome statistics and commits only its latest working response.
    """
    features = th.as_tensor(features, dtype=th.float64)
    target = th.as_tensor(target, dtype=th.float64)
    weight = th.as_tensor(weight, dtype=th.float64)
    if features.shape != (model["n_features"],):
        raise ValueError("features have the wrong shape")
    if (
        not th.all(th.isfinite(features))
        or not th.isfinite(target)
        or not th.isfinite(weight)
        or weight <= 0
    ):
        raise ValueError("Online training data must be finite")
    if previous_statistics is None:
        previous_statistics = (
            model["gram"],
            model["target_moment"],
            model["effective_weight"],
        )
    previous_gram, previous_target_moment, previous_effective_weight = (
        previous_statistics
    )
    gamma = model["forgetting_factor"]
    model["gram"] = gamma * previous_gram + weight * th.outer(features, features)
    model["target_moment"] = gamma * previous_target_moment + weight * features * target
    model["effective_weight"] = gamma * previous_effective_weight + weight.item()
    solve_online_regularized_regression(model)


def predict_online_regularized_regression(model: dict, features):
    """Predict from one row or a feature matrix."""
    prediction = th.as_tensor(features, dtype=th.float64) @ model["coefficients"]
    return prediction.item() if prediction.ndim == 0 else prediction


def gaussian_residual_quantile(mean, standard_deviation, probability):
    """Calculate a Gaussian quantile with PyTorch's inverse CDF."""
    if not 0 < probability < 1 or standard_deviation < 0:
        raise ValueError("Invalid probability or standard deviation")
    distribution = th.distributions.Normal(
        th.tensor(0.0, dtype=th.float64),
        th.tensor(1.0, dtype=th.float64),
    )
    return (
        mean
        + standard_deviation
        * distribution.icdf(th.tensor(probability, dtype=th.float64)).item()
    )


def johnson_su_standardised_moments(shape_a, shape_b):
    """Return mean and standard deviation of a Johnson SU variate."""
    shape_a = th.as_tensor(shape_a, dtype=th.float64)
    shape_b = th.as_tensor(shape_b, dtype=th.float64)
    inverse_shape_b = 1 / shape_b
    mean = -th.exp(inverse_shape_b.square() / 2) * th.sinh(shape_a * inverse_shape_b)
    second_moment = (
        th.exp(2 * inverse_shape_b.square()) * th.cosh(2 * shape_a * inverse_shape_b)
        - 1
    ) / 2
    standard_deviation = th.sqrt(th.clamp(second_moment - mean.square(), min=1e-12))
    return mean, standard_deviation


def johnson_su_residual_quantile(
    mean, standard_deviation, shape_a, shape_b, probability
):
    """Calculate a mean-standardised Johnson SU residual quantile."""
    if not 0 < probability < 1 or standard_deviation < 0 or shape_b <= 0:
        raise ValueError("Invalid Johnson SU quantile parameters")
    normal = th.distributions.Normal(
        th.tensor(0.0, dtype=th.float64),
        th.tensor(1.0, dtype=th.float64),
    )
    raw_mean, raw_standard_deviation = johnson_su_standardised_moments(shape_a, shape_b)
    raw_quantile = th.sinh(
        (normal.icdf(th.tensor(probability, dtype=th.float64)) - shape_a) / shape_b
    )
    return (
        mean
        + standard_deviation
        * ((raw_quantile - raw_mean) / raw_standard_deviation).item()
    )


def fit_johnson_su_shape(model: dict) -> None:
    """Update Johnson SU shape from past standardised forecast errors only."""
    errors = model["johnson_su_standardised_errors"]
    if not errors:
        return
    config = model["config"]
    errors = th.tensor(errors[-config["johnson_su_history_size"] :], dtype=th.float64)
    age = th.arange(len(errors) - 1, -1, -1, dtype=th.float64)
    weights = th.tensor(config["forgetting_factor"], dtype=th.float64) ** age
    parameters = th.tensor(
        [model["johnson_su_shape_a"], model["johnson_su_log_shape_b"]],
        dtype=th.float64,
        requires_grad=True,
    )
    iterations = config["johnson_su_solver_iterations"]
    if model["johnson_su_shape"] is None:
        iterations *= 20
    for _ in range(iterations):
        shape_a = th.clamp(parameters[0], min=-5.0, max=5.0)
        log_shape_b = th.clamp(
            parameters[1], min=th.log(th.tensor(0.5)), max=th.log(th.tensor(20.0))
        )
        shape_b = th.exp(log_shape_b)
        raw_mean, raw_standard_deviation = johnson_su_standardised_moments(
            shape_a, shape_b
        )
        raw_errors = raw_mean + raw_standard_deviation * errors
        transformed_errors = shape_a + shape_b * th.asinh(raw_errors)
        log_density = (
            log_shape_b
            - 0.5 * th.log1p(raw_errors.square())
            - 0.5 * transformed_errors.square()
        )
        loss = -(weights * log_density).sum() / weights.sum()
        gradient = th.autograd.grad(loss, parameters)[0]
        parameters = (
            (parameters - config["johnson_su_learning_rate"] * gradient)
            .detach()
            .requires_grad_()
        )
    model["johnson_su_shape_a"] = th.clamp(parameters[0], min=-5.0, max=5.0).item()
    model["johnson_su_log_shape_b"] = th.clamp(
        parameters[1], min=th.log(th.tensor(0.5)), max=th.log(th.tensor(20.0))
    ).item()
    model["johnson_su_shape"] = (
        model["johnson_su_shape_a"],
        th.exp(th.tensor(model["johnson_su_log_shape_b"])).item(),
    )


def initialize_adaptive_merit_order_model(market_id, forecast_inputs, config) -> dict:
    """Create the online state for one market without fitting future data."""
    config = config.copy()
    if config.get("use_holiday_feature"):
        if not config["holiday_dates"]:
            raise ValueError("The holiday feature requires at least one holiday date")
        config["features"] = config["features"] + ("holiday",)
        if config["scale_features"] is not None:
            config["scale_features"] = config["scale_features"] + ("holiday",)
    return {
        "market_id": market_id,
        "forecast_inputs": forecast_inputs,
        "config": config,
        "residual_mean_features": initialize_adaptive_merit_order_feature_state(
            config["features"], config["holiday_dates"]
        ),
        "residual_scale_features": initialize_adaptive_merit_order_feature_state(
            config["scale_features"]
            if config["scale_features"] is not None
            else config["features"],
            config["holiday_dates"],
        ),
        "residual_mean_model": None,
        "residual_scale_model": None,
        "initial_inputs": [],
        "initial_residuals": [],
        "johnson_su_standardised_errors": [],
        "johnson_su_shape_a": 0.0,
        "johnson_su_log_shape_b": th.log(th.tensor(5.0)).item(),
        "johnson_su_shape": None,
        "residual_by_product": {},
        "price_by_product": {},
        "residual_history": [],
        "pending": {},
        "pending_by_product": {},
        "observed_forecast_ids": set(),
        "last_observed_product": None,
        "outcomes": [],
    }


def initialize_adaptive_merit_order_correction(index, units, market) -> dict:
    """Initialize the adaptive model for one market with built-in defaults."""
    units = tuple(units)
    if not units:
        raise ValueError("Adaptive merit-order forecasts require loaded market units")
    config = ADAPTIVE_MERIT_ORDER_SETTINGS.copy()
    return {
        "markets": {
            market.market_id: initialize_adaptive_merit_order_model(
                market.market_id,
                calculate_adaptive_merit_order_forecast_inputs(index, units, market),
                config,
            )
        }
    }


def get_adaptive_merit_order_inputs(model: dict, product_start) -> dict:
    """Collect predictors known when the product forecast is issued."""
    timestamp = pd.Timestamp(product_start)
    lag_time = product_start - pd.Timedelta(days=1)
    return {
        "delivery_time": product_start,
        "merit_order_price": float(
            model["forecast_inputs"]["merit_order_price"].loc[timestamp]
        ),
        "wind_availability_factor": float(
            model["forecast_inputs"]["wind_availability_factor"].loc[timestamp]
        ),
        "solar_availability_factor": float(
            model["forecast_inputs"]["solar_availability_factor"].loc[timestamp]
        ),
        "residual_load": float(
            model["forecast_inputs"]["residual_load"].loc[timestamp]
        ),
        "previous_day_same_hour_residual": model["residual_by_product"].get(lag_time),
        "previous_day_same_hour_price": model["price_by_product"].get(lag_time),
        "is_holiday": product_start.date() in model["config"]["holiday_dates"],
    }


def issue_adaptive_merit_order_correction(
    state, unit_operator_id, market_id, issue_time, products
) -> list[dict]:
    """Issue and freeze adaptive merit-order forecasts before clearing."""
    model = state["markets"].get(market_id)
    if model is None:
        return []
    issued_forecasts = []
    trained = (
        model["residual_mean_model"] is not None
        and model["residual_scale_model"] is not None
    )
    for product_start, _, _ in products:
        forecast_id = (
            f"{unit_operator_id}|{market_id}|{issue_time.isoformat()}|"
            f"{product_start.isoformat()}"
        )
        if forecast_id in model["pending"]:
            issued_forecasts.append(dict(model["pending"][forecast_id]["issued"]))
            continue
        existing = model["pending_by_product"].get(product_start)
        if existing and existing not in model["observed_forecast_ids"]:
            raise ValueError(
                "An adaptive merit-order forecast for "
                f"{product_start!s} is already pending"
            )

        inputs = get_adaptive_merit_order_inputs(model, product_start)
        residual_mean_vector = None
        scale_vector = None
        if trained:
            residual_mean_vector = transform_adaptive_merit_order_features(
                model["residual_mean_features"], inputs
            )
            scale_vector = transform_adaptive_merit_order_features(
                model["residual_scale_features"], inputs
            )
            residual_mean = predict_online_regularized_regression(
                model["residual_mean_model"], residual_mean_vector
            )
            log_sigma = predict_online_regularized_regression(
                model["residual_scale_model"], scale_vector
            )
            log_sigma = th.clamp(
                th.tensor(log_sigma, dtype=th.float64),
                min=th.log(th.tensor(model["config"]["sigma_floor"], dtype=th.float64)),
                max=th.log(th.tensor(1e6, dtype=th.float64)),
            )
            residual_std = max(model["config"]["sigma_floor"], th.exp(log_sigma).item())
            status = "trained"
        elif len(model["residual_history"]) >= 2:
            residual_mean = 0.0
            residual_std = max(
                model["config"]["sigma_floor"],
                th.tensor(model["residual_history"], dtype=th.float64)
                .std(unbiased=True)
                .item(),
            )
            status = "fallback_empirical_uncertainty"
        else:
            residual_mean = 0.0
            residual_std = None
            status = "fallback_no_uncertainty"

        corrected_mean = inputs["merit_order_price"] + residual_mean
        issued = {
            "forecast_id": forecast_id,
            "unit_operator_id": unit_operator_id,
            "market_id": market_id,
            "issue_time": issue_time,
            "product_start": product_start,
            "merit_order_price_forecast": inputs["merit_order_price"],
            "residual_mean_forecast": residual_mean,
            "corrected_price_mean_forecast": corrected_mean,
            "residual_std_forecast": residual_std,
            "price_q10": None,
            "price_q50": corrected_mean,
            "price_q90": None,
            "training_status": status,
            "training_sample_count": len(model["residual_history"]),
        }
        if residual_std is not None:
            if (
                model["config"]["distribution"] == "johnson_su"
                and model["johnson_su_shape"] is not None
            ):
                shape_a, shape_b = model["johnson_su_shape"]
                issued["price_q10"] = johnson_su_residual_quantile(
                    corrected_mean, residual_std, shape_a, shape_b, 0.1
                )
                issued["price_q50"] = johnson_su_residual_quantile(
                    corrected_mean, residual_std, shape_a, shape_b, 0.5
                )
                issued["price_q90"] = johnson_su_residual_quantile(
                    corrected_mean, residual_std, shape_a, shape_b, 0.9
                )
            else:
                issued["price_q10"] = gaussian_residual_quantile(
                    corrected_mean, residual_std, 0.1
                )
                issued["price_q90"] = gaussian_residual_quantile(
                    corrected_mean, residual_std, 0.9
                )
        model["pending"][forecast_id] = {
            "issued": issued.copy(),
            "inputs": inputs,
            "residual_mean_vector": residual_mean_vector,
            "scale_vector": scale_vector,
        }
        model["pending_by_product"][product_start] = forecast_id
        issued_forecasts.append(issued.copy())
    return issued_forecasts


def fit_initial_adaptive_merit_order_models(model: dict) -> None:
    """Fit the initial Gaussian GAMLSS by penalised Fisher scoring/IRLS.

    The residual mean has identity link and working weights ``1 / sigma**2``.
    The standard deviation has a log link, working weight ``2`` and working
    response ``log(sigma) + ((error / sigma)**2 - 1) / 2``. Each weighted
    regression is solved with online-compatible L1/L2 coordinate descent.
    """
    fit_adaptive_merit_order_feature_scaling(
        model["residual_mean_features"], model["initial_inputs"]
    )
    fit_adaptive_merit_order_feature_scaling(
        model["residual_scale_features"], model["initial_inputs"]
    )
    residual_mean_matrix = th.stack(
        [
            transform_adaptive_merit_order_features(
                model["residual_mean_features"], row
            )
            for row in model["initial_inputs"]
        ]
    )
    scale_matrix = th.stack(
        [
            transform_adaptive_merit_order_features(
                model["residual_scale_features"], row
            )
            for row in model["initial_inputs"]
        ]
    )
    residuals = th.tensor(model["initial_residuals"], dtype=th.float64)
    config = model["config"]
    model["residual_mean_model"] = initialize_online_regularized_regression(
        residual_mean_matrix.shape[1],
        config["residual_mean_l1_regularization"],
        config["residual_mean_l2_regularization"],
        config["forgetting_factor"],
        config["solver_max_iterations"],
        config["solver_tolerance"],
    )
    model["residual_scale_model"] = initialize_online_regularized_regression(
        scale_matrix.shape[1],
        config["residual_scale_l1_regularization"],
        config["residual_scale_l2_regularization"],
        config["forgetting_factor"],
        config["solver_max_iterations"],
        config["solver_tolerance"],
    )

    # Stable starting values for the alternating distributional fit.
    fit_online_regularized_regression(
        model["residual_mean_model"], residual_mean_matrix, residuals
    )
    residual_mean = predict_online_regularized_regression(
        model["residual_mean_model"], residual_mean_matrix
    )
    initial_sigma = th.sqrt(th.mean((residuals - residual_mean).square()))
    initial_sigma = th.clamp(initial_sigma, min=config["sigma_floor"])
    model["residual_scale_model"]["coefficients"][0] = th.log(initial_sigma)

    previous_parameters = None
    for _ in range(config["irls_max_iterations"]):
        log_sigma = predict_online_regularized_regression(
            model["residual_scale_model"], scale_matrix
        )
        log_sigma = th.clamp(
            log_sigma,
            min=th.log(th.tensor(config["sigma_floor"], dtype=th.float64)),
            max=th.log(th.tensor(1e6, dtype=th.float64)),
        )
        sigma = th.exp(log_sigma)
        fit_online_regularized_regression(
            model["residual_mean_model"],
            residual_mean_matrix,
            residuals,
            1 / sigma.square(),
        )

        residual_mean = predict_online_regularized_regression(
            model["residual_mean_model"], residual_mean_matrix
        )
        errors = residuals - residual_mean
        log_sigma = predict_online_regularized_regression(
            model["residual_scale_model"], scale_matrix
        )
        log_sigma = th.clamp(
            log_sigma,
            min=th.log(th.tensor(config["sigma_floor"], dtype=th.float64)),
            max=th.log(th.tensor(1e6, dtype=th.float64)),
        )
        sigma = th.exp(log_sigma)
        scale_working_response = log_sigma + ((errors / sigma).square() - 1) / 2
        fit_online_regularized_regression(
            model["residual_scale_model"],
            scale_matrix,
            scale_working_response,
            th.full_like(residuals, 2.0),
        )

        parameters = th.cat(
            (
                model["residual_mean_model"]["coefficients"],
                model["residual_scale_model"]["coefficients"],
            )
        )
        if (
            previous_parameters is not None
            and th.max(th.abs(parameters - previous_parameters)).item()
            <= config["irls_tolerance"]
        ):
            break
        previous_parameters = parameters.clone()

    if model["config"]["distribution"] == "johnson_su":
        initial_standard_deviations = th.exp(
            th.clamp(
                predict_online_regularized_regression(
                    model["residual_scale_model"], scale_matrix
                ),
                min=th.log(th.tensor(config["sigma_floor"], dtype=th.float64)),
                max=th.log(th.tensor(1e6, dtype=th.float64)),
            )
        )
        errors = residuals - predict_online_regularized_regression(
            model["residual_mean_model"], residual_mean_matrix
        )
        model["johnson_su_standardised_errors"] = (
            errors / initial_standard_deviations
        ).tolist()
        fit_johnson_su_shape(model)


def update_adaptive_merit_order_correction(state, market_id, market_meta) -> list[dict]:
    """Link realised prices and update only forecasts issued subsequently."""
    model = state["markets"].get(market_id)
    if model is None:
        return []
    rows = sorted(market_meta, key=lambda row: row["product_start"])
    starts = [row["product_start"] for row in rows]
    if len(starts) != len(set(starts)):
        raise ValueError(
            "Adaptive merit-order forecasts require one scalar clearing price "
            "per product"
        )

    outcomes = []
    for row in rows:
        product_start = row["product_start"]
        price = row["price"]
        if not isinstance(price, int | float) or not th.isfinite(
            th.tensor(price, dtype=th.float64)
        ):
            raise ValueError(
                "Adaptive merit-order forecasts require finite clearing prices"
            )
        forecast_id = model["pending_by_product"].get(product_start)
        if forecast_id is None or forecast_id in model["observed_forecast_ids"]:
            continue
        if (
            model["last_observed_product"] is not None
            and product_start <= model["last_observed_product"]
        ):
            raise ValueError(
                "Adaptive merit-order outcomes must arrive chronologically"
            )

        pending = model["pending"][forecast_id]
        issued = pending["issued"]
        realised_price = float(price)
        realised_residual = realised_price - issued["merit_order_price_forecast"]
        post_forecast_residual = realised_residual - issued["residual_mean_forecast"]
        outcome = issued.copy() | {
            "realised_price": realised_price,
            "realised_residual": realised_residual,
            "post_forecast_residual": post_forecast_residual,
        }
        outcomes.append(outcome)
        model["outcomes"].append(outcome.copy())
        model["observed_forecast_ids"].add(forecast_id)
        model["last_observed_product"] = product_start
        model["price_by_product"][product_start] = realised_price
        model["residual_by_product"][product_start] = realised_residual
        model["residual_history"].append(realised_residual)

        trained = (
            model["residual_mean_model"] is not None
            and model["residual_scale_model"] is not None
        )
        if not trained:
            model["initial_inputs"].append(pending["inputs"])
            model["initial_residuals"].append(realised_residual)
            if (
                len(model["initial_residuals"])
                == model["config"]["minimum_training_samples"]
            ):
                fit_initial_adaptive_merit_order_models(model)
            continue

        residual_mean_vector = pending["residual_mean_vector"]
        scale_vector = pending["scale_vector"]
        if residual_mean_vector is None:
            residual_mean_vector = transform_adaptive_merit_order_features(
                model["residual_mean_features"], pending["inputs"]
            )
        if scale_vector is None:
            scale_vector = transform_adaptive_merit_order_features(
                model["residual_scale_features"], pending["inputs"]
            )
        if model["config"]["distribution"] == "johnson_su":
            model["johnson_su_standardised_errors"].append(
                post_forecast_residual / issued["residual_std_forecast"]
            )
            fit_johnson_su_shape(model)
        # Freeze the pre-outcome statistics. IRLS may reconsider the current
        # working weights repeatedly, but the clearing result is committed once.
        residual_mean_model = model["residual_mean_model"]
        residual_scale_model = model["residual_scale_model"]
        mean_statistics = (
            residual_mean_model["gram"].clone(),
            residual_mean_model["target_moment"].clone(),
            residual_mean_model["effective_weight"],
        )
        scale_statistics = (
            residual_scale_model["gram"].clone(),
            residual_scale_model["target_moment"].clone(),
            residual_scale_model["effective_weight"],
        )
        config = model["config"]
        previous_parameters = None
        for _ in range(config["irls_max_iterations"]):
            log_sigma = predict_online_regularized_regression(
                residual_scale_model, scale_vector
            )
            log_sigma = th.clamp(
                th.tensor(log_sigma, dtype=th.float64),
                min=th.log(th.tensor(config["sigma_floor"], dtype=th.float64)),
                max=th.log(th.tensor(1e6, dtype=th.float64)),
            )
            sigma = th.exp(log_sigma)
            update_online_regularized_regression(
                residual_mean_model,
                residual_mean_vector,
                realised_residual,
                1 / sigma.square(),
                mean_statistics,
            )

            residual_mean = predict_online_regularized_regression(
                residual_mean_model, residual_mean_vector
            )
            error = realised_residual - residual_mean
            log_sigma = predict_online_regularized_regression(
                residual_scale_model, scale_vector
            )
            log_sigma = th.clamp(
                th.tensor(log_sigma, dtype=th.float64),
                min=th.log(th.tensor(config["sigma_floor"], dtype=th.float64)),
                max=th.log(th.tensor(1e6, dtype=th.float64)),
            )
            sigma = th.exp(log_sigma)
            scale_working_response = log_sigma + ((error / sigma).square() - 1) / 2
            update_online_regularized_regression(
                residual_scale_model,
                scale_vector,
                scale_working_response,
                2.0,
                scale_statistics,
            )

            parameters = th.cat(
                (
                    residual_mean_model["coefficients"],
                    residual_scale_model["coefficients"],
                )
            )
            if (
                previous_parameters is not None
                and th.max(th.abs(parameters - previous_parameters)).item()
                <= config["irls_tolerance"]
            ):
                break
            previous_parameters = parameters.clone()
    return outcomes


def evaluate_adaptive_merit_order_forecasts(records) -> dict[str, pd.DataFrame]:
    """Compare merit order, past bias, and adaptive correction."""
    frame = (
        records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    )
    if frame.empty:
        raise ValueError(
            "At least one adaptive merit-order forecast record is required"
        )
    frame["issue_time"] = pd.to_datetime(frame["issue_time"])
    frame["product_start"] = pd.to_datetime(frame["product_start"])
    frame = frame.sort_values(["issue_time", "product_start"]).reset_index(drop=True)
    frame["realised_residual"] = (
        frame["realised_price"] - frame["merit_order_price_forecast"]
    )
    residual_sum = 0.0
    residual_count = 0
    frame["historical_bias"] = 0.0
    for _, rows in frame.groupby("issue_time", sort=True).groups.items():
        frame.loc[rows, "historical_bias"] = (
            residual_sum / residual_count if residual_count else 0
        )
        residuals = frame.loc[rows, "realised_residual"].dropna()
        residual_sum += residuals.sum()
        residual_count += residuals.count()
    frame["constant_historical_bias_forecast"] = (
        frame["merit_order_price_forecast"] + frame["historical_bias"]
    )
    frame["delivery_hour"] = frame["product_start"].dt.hour

    methods = {
        "merit_order_only": "merit_order_price_forecast",
        "constant_historical_bias": "constant_historical_bias_forecast",
        "adaptive_merit_order_correction": "corrected_price_mean_forecast",
    }

    def metric_rows(group):
        result = []
        for method, column in methods.items():
            error = group[column] - group["realised_price"]
            row = {
                "method": method,
                "samples": error.notna().sum(),
                "mae": error.abs().mean(),
                "rmse": error.pow(2).mean() ** 0.5,
                "central_80_coverage": float("nan"),
                "central_80_average_width": float("nan"),
                "pinball_q10": float("nan"),
                "pinball_q50": float("nan"),
                "pinball_q90": float("nan"),
            }
            if method == "adaptive_merit_order_correction":
                interval = group["price_q10"].notna() & group["price_q90"].notna()
                if interval.any():
                    row["central_80_coverage"] = (
                        (
                            group.loc[interval, "realised_price"]
                            >= group.loc[interval, "price_q10"]
                        )
                        & (
                            group.loc[interval, "realised_price"]
                            <= group.loc[interval, "price_q90"]
                        )
                    ).mean()
                    row["central_80_average_width"] = (
                        group.loc[interval, "price_q90"]
                        - group.loc[interval, "price_q10"]
                    ).mean()
                for label, probability in (
                    ("q10", 0.1),
                    ("q50", 0.5),
                    ("q90", 0.9),
                ):
                    valid = group[f"price_{label}"].notna()
                    error_q = (
                        group.loc[valid, "realised_price"]
                        - group.loc[valid, f"price_{label}"]
                    )
                    row[f"pinball_{label}"] = (
                        pd.concat(
                            (
                                probability * error_q,
                                (probability - 1) * error_q,
                            ),
                            axis=1,
                        )
                        .max(axis=1)
                        .mean()
                    )
            result.append(row)
        return result

    by_hour = []
    for hour, group in frame.groupby("delivery_hour"):
        rows = metric_rows(group)
        for row in rows:
            row["delivery_hour"] = hour
        by_hour.extend(rows)
    sample_columns = [
        "issue_time",
        "product_start",
        "merit_order_price_forecast",
        "constant_historical_bias_forecast",
        "corrected_price_mean_forecast",
        "price_q10",
        "price_q90",
        "realised_price",
        "realised_residual",
    ]
    return {
        "summary": pd.DataFrame(metric_rows(frame)),
        "by_delivery_hour": pd.DataFrame(by_hour),
        "samples": frame[sample_columns].head(10),
    }


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
    """Return the forecast algorithm registries bundled with ASSUME."""
    return {
        "init": forecast_algorithms,
        "preprocess": forecast_preprocess_algorithms,
        "update": forecast_update_algorithms,
    }
