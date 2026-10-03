# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rule-based EOM, CRM, and OTC bidding for industrial hybrid DSM units.

Each public strategy below follows the usual ASSUME pattern: calculate bids,
then record market feedback when a cleared order creates a physical commitment.
The small helpers at the top hold the shared physical rules.  The initial
implementation recognises two compatible cement configurations, but the public
strategy names are independent of that first implementation.
"""

from __future__ import annotations

import copy
import math
from datetime import timedelta

import pyomo.environ as pyo

from assume.common.base import MinMaxStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook, Product

# ---------------------------------------------------------------------------
# Shared hybrid-plant rules
# ---------------------------------------------------------------------------


def hybrid_route(unit, strategy_name: str) -> tuple[str, dict, dict | None]:
    """Return the supported heat route and its component configuration.

    The industrial strategy family is intentionally generic.  At present its
    physical rules are implemented for the compatible cement configurations
    only: a direct electric-plus-natural-gas calciner, or E-TES plus a
    natural-gas calciner.  A future industrial unit can extend this one
    explicit function with its own compatible route.
    """
    if unit.technology != "cement_plant":
        raise ValueError(
            "industrial_hybrid strategies currently support compatible cement "
            "hybrid routes only."
        )
    if unit.horizon_mode != "rolling_horizon":
        raise ValueError(
            f"{strategy_name} requires dsm_optimisation_config.horizon_mode="
            "'rolling_horizon'."
        )

    components = getattr(unit, "_orig_components_dict", unit.components)
    calciner = components.get("calciner")
    storage = components.get("thermal_storage")
    if not isinstance(calciner, dict):
        raise ValueError(f"{strategy_name} requires a configured calciner.")
    if calciner.get("fossil_ng_share", 1.0) != 1.0:
        raise ValueError(
            f"{strategy_name} currently requires a natural-gas-only calciner "
            "(fossil_ng_share=1)."
        )

    fuel_type = calciner.get("fuel_type", "electricity").lower()
    if storage is None and fuel_type == "both":
        return "direct", calciner, None
    if (
        isinstance(storage, dict)
        and storage.get("storage_type", "short-term")
        == "short-term_with_generator"
        and fuel_type == "fossil"
    ):
        return "etes", calciner, storage

    raise ValueError(
        f"{strategy_name} supports either a direct electric-plus-natural-gas "
        "calciner without thermal storage, or E-TES plus a natural-gas calciner."
    )


def electric_wtp(unit, timestamp, route: str, calciner, storage) -> float:
    """Value one MWh of electricity as avoided gas and CO2 heat cost."""
    fuel_prices = getattr(unit.forecaster, "fuel_prices", {})
    if "natural_gas" not in fuel_prices or "co2" not in fuel_prices:
        raise ValueError(
            "industrial_hybrid strategies require natural_gas and co2 price series "
            "in fuel_prices_df."
        )

    eta_fossil = float(calciner.get("eta_fossil", 0.90))
    if eta_fossil <= 0:
        raise ValueError("Calciner eta_fossil must be positive.")
    fossil_heat_cost = (
        float(unit.forecaster.get_price("natural_gas").at[timestamp])
        + float(calciner.get("ng_co2_factor", 0.202))
        * float(unit.forecaster.get_price("co2").at[timestamp])
    ) / eta_fossil

    if route == "direct":
        electric_to_heat = float(calciner.get("eta_electric", 0.95))
    else:
        electric_to_heat = (
            float(storage.get("eta_electric", 0.0))
            * float(storage.get("efficiency_charge", 1.0))
            * float(storage.get("efficiency_discharge", 1.0))
        )
    if electric_to_heat <= 0:
        raise ValueError("Electric heat and storage efficiencies must be positive.")
    return fossil_heat_cost * electric_to_heat


def adjust_bid_price(price: float, market_config: MarketConfig) -> float:
    """Apply the market price bounds and price tick to a bid price."""
    price = max(price, market_config.minimum_bid_price)
    if market_config.maximum_bid_price is not None:
        price = min(price, market_config.maximum_bid_price)
    if market_config.price_tick:
        tick = market_config.price_tick
        lower = math.ceil(market_config.minimum_bid_price / tick) * tick
        upper = (
            math.floor(market_config.maximum_bid_price / tick) * tick
            if market_config.maximum_bid_price is not None
            else None
        )
        if upper is not None and lower > upper:
            raise ValueError("Market price bounds contain no valid price tick.")
        price = max(round(price / tick) * tick, lower)
        if upper is not None:
            price = min(price, upper)
    return price


def adjust_bid_volume(volume: float, market_config: MarketConfig) -> float:
    """Apply the market volume cap and tick without overstating physical volume."""
    volume = max(0.0, volume)
    if market_config.maximum_bid_volume is not None:
        volume = min(volume, float(market_config.maximum_bid_volume))
    if market_config.volume_tick:
        volume = (
            int((volume + 1e-12) / market_config.volume_tick)
            * market_config.volume_tick
        )
    return volume


def hours_in_product(unit, start, end, require_complete: bool = False) -> list:
    """Return ASSUME's aligned model timestamps in a market product.

    OTC contracts require every delivery interval to be in the forecast horizon;
    CRM can simply have no available capacity when a product has no intervals.
    """
    last_delivery_time = end - unit.index.freq
    if end <= start or last_delivery_time < start:
        return []
    if require_complete and (
        start not in unit.index or last_delivery_time not in unit.index
    ):
        return []

    first_available_time = max(start, unit.index.start)
    last_available_time = min(last_delivery_time, unit.index.end)
    if first_available_time > last_available_time:
        return []
    return unit.index.get_date_list(first_available_time, last_available_time)


def component_operations_by_step(unit) -> dict[int, dict]:
    """Index the most recent rolling solve's component values by model step."""
    return {
        row["global_t"]: row
        for row in getattr(unit, "_component_operations", [])
        if "global_t" in row
    }


def update_rolling_schedule(
    unit, market_time, force: bool = False, constraint_builder=None
) -> None:
    """Refresh forecasts and the next rolling-operation schedule when required."""
    if getattr(unit.forecaster, "_registries", None) is not None:
        unit.forecaster.update(unit=unit)
    did_reoptimize = unit._check_and_reoptimize_rolling_window(
        market_time, force=force, constraint_builder=constraint_builder
    )
    if not did_reoptimize and unit.optimisation_counter == 0:
        unit.determine_optimal_operation_with_flex()
        unit.optimisation_counter = 1


def accepted_demand_by_hour(orderbook: Orderbook) -> dict:
    """Return cleared demand MW by delivery timestamp."""
    accepted: dict = {}
    for order in orderbook:
        volume = order.get("accepted_volume", 0.0)
        if isinstance(volume, dict):
            for start, value in volume.items():
                accepted[start] = accepted.get(start, 0.0) + max(0.0, -float(value))
        else:
            start = order["start_time"]
            accepted[start] = accepted.get(start, 0.0) + max(
                0.0, -float(volume or 0.0)
            )
    return accepted


def acceptance_fraction(order, offered_volume: float) -> float:
    """Convert an accepted order, including a partial block, to a fraction."""
    accepted = order.get("accepted_volume", 0.0)
    if isinstance(accepted, dict):
        accepted = min((abs(float(value)) for value in accepted.values()), default=0.0)
    submitted_volume = abs(float(order.get("volume", offered_volume)))
    if submitted_volume <= 0:
        return 0.0
    return min(1.0, abs(float(accepted or 0.0)) / submitted_volume)


# ---------------------------------------------------------------------------
# Physical commitments known before EOM bidding
# ---------------------------------------------------------------------------


def otc_profile(unit) -> dict:
    """Physical MW already bought in OTC, indexed by delivery timestamp."""
    return getattr(unit, "industrial_hybrid_otc_procurement", {})


def otc_load_constraint(unit):
    """Require an EOM schedule to consume every accepted physical OTC contract."""
    procurement = otc_profile(unit)
    if not procurement:
        return None

    def add_constraint(model, window_start, window_end):
        constraints = pyo.ConstraintList()
        model.otc_procurement_constraint = constraints
        for local_t in model.time_steps:
            contracted_power = float(procurement.get(unit.index[local_t], 0.0))
            if contracted_power > 0:
                constraints.add(model.total_power_input[local_t] >= contracted_power)

    return add_constraint


def capacity_neg_commitment_constraint(unit, route: str):
    """Keep accepted ``capacity_neg`` commitments feasible in the EOM schedule."""
    commitments = getattr(unit, "industrial_hybrid_capacity_neg_commitments", {})
    if not commitments:
        return None

    def add_constraint(model, window_start, window_end):
        constraints = pyo.ConstraintList()
        model.capacity_neg_commitment_constraint = constraints
        for local_t in model.time_steps:
            timestamp = unit.index[local_t]
            entries = commitments.get(timestamp, [])
            reserved = sum(float(entry["volume"]) for entry in entries)
            if reserved <= 0:
                continue

            if route == "direct":
                potential = min(
                    float(entry["electric_conversion_capacity"])
                    for entry in entries
                    if entry["route"] == "direct"
                )
                constraints.add(
                    model.dsm_blocks["calciner"].power_in[local_t]
                    <= max(0.0, potential - reserved)
                )
                continue

            etes_entries = [entry for entry in entries if entry["route"] == "etes"]
            if not etes_entries:
                continue
            electric_max = min(float(entry["electric_max"]) for entry in etes_entries)
            storage = model.dsm_blocks["thermal_storage"]
            constraints.add(storage.power_in[local_t] <= max(0.0, electric_max - reserved))
            for entry in etes_entries:
                remaining_steps = sum(
                    1 for point in entry["hours"] if point > timestamp
                )
                if remaining_steps > 0:
                    constraints.add(
                        storage.soc[local_t]
                        <= float(entry["max_soc"])
                        - remaining_steps
                        * float(entry["volume"])
                        * float(entry["eta_electric"])
                        / float(entry["capacity"])
                    )

    return add_constraint


# ---------------------------------------------------------------------------
# EOM
# ---------------------------------------------------------------------------


class IndustrialHybridEomStrategy(MinMaxStrategy):
    """Bid the rolling electricity schedule of a compatible industrial hybrid unit."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending_bid_contexts: dict[tuple[str, str], dict] = {}

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        if market_config.product_type != "energy":
            raise ValueError("industrial_hybrid_eom is an energy-market strategy.")
        route, calciner, storage = hybrid_route(unit, "industrial_hybrid_eom")
        if not product_tuples:
            return []
        commit_steps = unit._parse_duration_to_steps(unit._rh_commit)
        expected_starts = [
            product_tuples[0][0] + step * unit.index.freq
            for step in range(commit_steps)
        ]
        if [product[0] for product in product_tuples] != expected_starts:
            raise ValueError(
                "industrial_hybrid_eom requires EOM products that exactly cover the "
                "next rolling commit horizon."
            )

        # OTC awards must be consumed and CRM awards must remain callable.
        constraint_builders = [
            builder
            for builder in (
                otc_load_constraint(unit),
                capacity_neg_commitment_constraint(unit, route),
            )
            if builder is not None
        ]
        constraint_builder = None
        if constraint_builders:

            def add_physical_commitments(model, window_start, window_end):
                for builder in constraint_builders:
                    builder(model, window_start, window_end)

            constraint_builder = add_physical_commitments
        initial_states = copy.deepcopy(
            unit._rh_init_states
            if unit._rh_init_states is not None
            else unit._collect_init_states()
        )
        update_rolling_schedule(
            unit,
            product_tuples[0][0],
            force=constraint_builder is not None,
            constraint_builder=constraint_builder,
        )

        rows = component_operations_by_step(unit)
        otc_power = otc_profile(unit)
        bid_context = {
            "route": route,
            "calciner": calciner,
            "storage": storage,
            "initial_states": initial_states,
            "products": {},
        }
        bids: Orderbook = []
        for start, end, only_hours in product_tuples:
            global_t = unit.index._get_idx_from_date(start)
            row = rows.get(global_t, {})
            planned_power = max(0.0, float(unit.opt_power_requirement.at[start]))
            contracted_power = min(planned_power, float(otc_power.get(start, 0.0)))
            flexible_power = float(
                row.get(
                    "calciner_power_input"
                    if route == "direct"
                    else "thermal_storage_power_input",
                    0.0,
                )
            )
            planned_production = float(
                getattr(unit, "_rh_full_horizon_production", [0.0] * len(unit.index))[
                    global_t
                ]
            )
            bid_context["products"][start] = {
                "global_t": global_t,
                "power": planned_power,
                "flexible_power": max(0.0, flexible_power),
                "production": max(0.0, planned_production),
                "storage_discharge": float(row.get("thermal_storage_discharge", 0.0)),
            }

            # An accepted OTC contract already procures this part of the load.
            residual_power = max(0.0, planned_power - contracted_power)
            if residual_power > 0:
                bids.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": only_hours,
                        "price": adjust_bid_price(
                            electric_wtp(unit, start, route, calciner, storage),
                            market_config,
                        ),
                        "volume": -residual_power,
                        "node": unit.node,
                    }
                )

        self.pending_bid_contexts[(unit.id, market_config.market_id)] = bid_context
        return self.remove_empty_bids(bids)

    def on_market_feedback(
        self, unit: SupportsMinMax, market_config: MarketConfig, orderbook: Orderbook
    ) -> None:
        """Apply the deterministic gas-fallback rule after EOM clearing."""
        bid_context = self.pending_bid_contexts.pop(
            (unit.id, market_config.market_id), None
        )
        if bid_context is None:
            return

        # Re-optimisation: after every EOM result, Pyomo searches all possible new
        # schedules. It can find a better economic response, but is much more
        # computationally expensive and harder to interpret.
        accepted_eom = accepted_demand_by_hour(orderbook)
        accepted_otc = otc_profile(unit)
        route = bid_context["route"]
        calciner = bid_context["calciner"]
        storage = bid_context["storage"]

        storage_energy = None
        if route == "etes":
            initial_soc = bid_context["initial_states"].get(
                "thermal_storage", {}
            ).get(
                "soc", storage.get("initial_soc", 1.0)
            )
            storage_energy = float(initial_soc) * float(storage["capacity"])

        for start, item in bid_context["products"].items():
            planned_power = item["power"]
            flexible_power = min(item["flexible_power"], planned_power)
            base_power = max(0.0, planned_power - flexible_power)
            bought_power = min(
                planned_power,
                accepted_eom.get(start, 0.0) + float(accepted_otc.get(start, 0.0)),
            )

            supplied_base = min(base_power, bought_power)
            production_share = 1.0 if base_power == 0 else supplied_base / base_power
            electric_flexible = min(
                flexible_power * production_share,
                max(0.0, bought_power - supplied_base),
            )
            missing_flexible = flexible_power * production_share - electric_flexible

            gas_fallback = 0.0
            if route == "direct":
                gas_fallback = (
                    missing_flexible * float(calciner.get("eta_electric", 0.95))
                ) / float(calciner.get("eta_fossil", 0.90))
            else:
                retained_energy = storage_energy * (
                    1.0 - float(storage.get("storage_loss_rate", 0.0))
                )
                charged_heat = (
                    electric_flexible
                    * float(storage["eta_electric"])
                    * float(storage.get("efficiency_charge", 1.0))
                )
                target_discharge = item["storage_discharge"] * production_share
                actual_discharge = min(
                    target_discharge,
                    (retained_energy + charged_heat)
                    * float(storage.get("efficiency_discharge", 1.0)),
                )
                gas_fallback = (
                    target_discharge - actual_discharge
                ) / float(calciner.get("eta_fossil", 0.90))
                storage_energy = max(
                    0.0,
                    retained_energy
                    + charged_heat
                    - actual_discharge / float(storage.get("efficiency_discharge", 1.0)),
                )
                unit.outputs["thermal_storage_soc"].at[start] = (
                    storage_energy / float(storage["capacity"])
                )

            unit.outputs["gas_fallback"].at[start] = gas_fallback
            unit.outputs["unserved_clinker"].at[start] = item["production"] * (
                1.0 - production_share
            )
            if hasattr(unit, "_rh_full_horizon_production"):
                unit._rh_full_horizon_production[item["global_t"]] = (
                    item["production"] * production_share
                )

        if route == "etes" and unit._rh_init_states is not None:
            unit._rh_init_states.setdefault("thermal_storage", {})["soc"] = (
                storage_energy / float(storage["capacity"])
            )


# ---------------------------------------------------------------------------
# Negative CRM capacity
# ---------------------------------------------------------------------------


class IndustrialHybridCapacityNegStrategy(MinMaxStrategy):
    """Offer firm additional electric demand on a ``capacity_neg`` product."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending_bid_contexts: dict[tuple[str, str], dict] = {}

    @staticmethod
    def validate_capacity_products(unit, product_tuples: list[Product]) -> None:
        if not product_tuples:
            return
        first_start = min(product[0] for product in product_tuples)
        commit_end = (
            first_start
            + unit._parse_duration_to_steps(unit._rh_commit) * unit.index.freq
        )
        for start, end, only_hours in product_tuples:
            if end <= start or start < first_start or end > commit_end:
                raise ValueError(
                    "industrial_hybrid_capacity_neg requires each capacity product "
                    "to be delivered entirely inside the next rolling commit window."
                )

    @staticmethod
    def direct_available_capacity_neg(
        unit, calciner, rows, start, end
    ) -> tuple[float, dict]:
        eta_electric = float(calciner.get("eta_electric", 0.0))
        if eta_electric <= 0:
            return 0.0, {}
        potential: dict = {}
        available_capacity_neg: list[float] = []
        for timestamp in hours_in_product(unit, start, end):
            row = rows.get(unit.index._get_idx_from_date(timestamp), {})
            clinker = max(0.0, float(row.get("calciner_clinker_output", 0.0)))
            heat = clinker * float(calciner.get("specific_heat_demand", 0.0))
            potential[timestamp] = heat / eta_electric
            available_capacity_neg.append(
                max(0.0, potential[timestamp] - float(otc_profile(unit).get(timestamp, 0.0)))
            )
        return (
            min(available_capacity_neg) if available_capacity_neg else 0.0,
            potential,
        )

    @staticmethod
    def etes_available_capacity_neg(
        storage, projected_soc, start, end
    ) -> tuple[float, float]:
        hours = (end - start) / timedelta(hours=1)
        capacity = float(storage["capacity"])
        eta_electric = float(storage.get("eta_electric", 0.0))
        efficiency_charge = float(storage.get("efficiency_charge", 1.0))
        if hours <= 0 or capacity <= 0 or eta_electric <= 0 or efficiency_charge <= 0:
            return 0.0, projected_soc
        electric_max = float(
            storage.get(
                "max_power", float(storage["max_power_charge"]) / eta_electric
            )
        )
        max_soc = float(storage.get("max_soc", 1.0))
        space_limited = (
            max(0.0, max_soc - projected_soc)
            * capacity
            / (hours * eta_electric * efficiency_charge)
        )
        offered = min(electric_max, space_limited)
        projected_soc = min(
            max_soc,
            projected_soc
            + offered * hours * eta_electric * efficiency_charge / capacity,
        )
        return offered, projected_soc

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        if market_config.product_type != "capacity_neg":
            raise ValueError(
                "industrial_hybrid_capacity_neg is a capacity_neg-market strategy."
            )
        route, calciner, storage = hybrid_route(
            unit, "industrial_hybrid_capacity_neg"
        )
        self.validate_capacity_products(unit, product_tuples)
        if not product_tuples:
            return []

        # Capacity is calculated from the normal forecast schedule. If accepted,
        # its reservation is imposed in one later EOM rolling solve.
        update_rolling_schedule(unit, min(product[0] for product in product_tuples))
        rows = component_operations_by_step(unit)
        projected_soc = float(
            (unit._rh_init_states or {}).get("thermal_storage", {}).get(
                "soc", storage.get("initial_soc", 1.0) if storage else 0.0
            )
        )
        bid_context = {"products": {}}
        bids: Orderbook = []
        for start, end, only_hours in sorted(product_tuples, key=lambda product: product[0]):
            hours = hours_in_product(unit, start, end)
            if not hours:
                continue
            if route == "direct":
                available_capacity_neg, electric_conversion_capacity = (
                    self.direct_available_capacity_neg(
                        unit, calciner, rows, start, end
                    )
                )
                product_context = {
                    "electric_conversion_capacity": electric_conversion_capacity
                }
            else:
                available_capacity_neg, projected_soc = self.etes_available_capacity_neg(
                    storage, projected_soc, start, end
                )
                product_context = {
                    "electric_max": float(
                        storage.get(
                            "max_power",
                            float(storage["max_power_charge"])
                            / float(storage["eta_electric"]),
                        )
                    ),
                    "capacity": float(storage["capacity"]),
                    "eta_electric": float(storage["eta_electric"]),
                    "max_soc": float(storage.get("max_soc", 1.0)),
                }

            volume = adjust_bid_volume(available_capacity_neg, market_config)
            opportunity_cost = sum(
                max(
                    0.0,
                    electric_wtp(unit, timestamp, route, calciner, storage)
                    - float(unit.forecaster.electricity_price.at[timestamp]),
                )
                * (unit.index.freq / timedelta(hours=1))
                for timestamp in hours
            ) / ((end - start) / timedelta(hours=1))
            bid_context["products"][start] = {
                "volume": volume,
                "route": route,
                "hours": hours,
                **product_context,
            }
            if volume > 0:
                bids.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": only_hours,
                        "price": adjust_bid_price(opportunity_cost, market_config),
                        "volume": volume,
                        "node": unit.node,
                    }
                )

        self.pending_bid_contexts[(unit.id, market_config.market_id)] = bid_context
        return self.remove_empty_bids(bids)

    def on_market_feedback(
        self, unit: SupportsMinMax, market_config: MarketConfig, orderbook: Orderbook
    ) -> None:
        """Store each accepted CRM block as an hourly physical reservation."""
        bid_context = self.pending_bid_contexts.pop(
            (unit.id, market_config.market_id), None
        )
        if bid_context is None:
            return

        commitments = getattr(
            unit, "industrial_hybrid_capacity_neg_commitments", {}
        ).copy()
        for order in orderbook:
            item = bid_context["products"].get(order.get("start_time"))
            if item is None or item["volume"] <= 0:
                continue
            awarded = item["volume"] * acceptance_fraction(order, item["volume"])
            if awarded <= 0:
                continue
            for timestamp in item["hours"]:
                entry = {key: value for key, value in item.items() if key != "volume"}
                entry["volume"] = awarded
                if item["route"] == "direct":
                    entry["electric_conversion_capacity"] = item[
                        "electric_conversion_capacity"
                    ][timestamp]
                commitments.setdefault(timestamp, []).append(entry)
        unit.industrial_hybrid_capacity_neg_commitments = commitments


# ---------------------------------------------------------------------------
# Long-term OTC procurement
# ---------------------------------------------------------------------------


class IndustrialHybridOtcStrategy(MinMaxStrategy):
    """Buy firm physical electricity in advance on a pay-as-bid OTC market."""

    def __init__(self, *args, auxiliary_security_value=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.auxiliary_security_value = auxiliary_security_value
        self.pending_bid_contexts: dict[tuple[str, str], dict] = {}

    @staticmethod
    def firm_production(unit, timestamp) -> float:
        """Use scheduled production, or the per-interval demand forecast early on."""
        production = getattr(unit, "_rh_full_horizon_production", None)
        try:
            scheduled = float(production[unit.index._get_idx_from_date(timestamp)])
            if scheduled > 0:
                return scheduled
        except (IndexError, KeyError, TypeError, ValueError):
            pass

        # Long-term products can extend beyond the first rolling solve. For the
        # initial cement implementation, clinker demand is the firm forecast floor.
        demand = getattr(unit.forecaster, "clinker_demand", None)
        try:
            return max(0.0, float(demand.at[timestamp]))
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return 0.0

    @classmethod
    def auxiliary_security_volume(cls, unit, calciner, hours) -> float:
        return min(
            cls.firm_production(unit, timestamp)
            * float(calciner.get("specific_electricity_aux", 0.0))
            for timestamp in hours
        )

    @classmethod
    def direct_process_volume(cls, unit, calciner, hours) -> float:
        eta_electric = float(calciner.get("eta_electric", 0.0))
        if eta_electric <= 0:
            return 0.0
        return min(
            cls.firm_production(unit, timestamp)
            * float(calciner.get("specific_heat_demand", 0.0))
            / eta_electric
            for timestamp in hours
        )

    @staticmethod
    def process_bid_ceiling(unit, hours, route, calciner, storage) -> float:
        """Average the lower of forecast EOM price and electric WTP over a block."""
        weighted_value = 0.0
        total_hours = 0.0
        for timestamp in hours:
            duration = unit.index.freq / timedelta(hours=1)
            eom_price = float(unit.forecaster.electricity_price.at[timestamp])
            electric_wtp_value = electric_wtp(
                unit, timestamp, route, calciner, storage
            )
            if not math.isfinite(eom_price) or not math.isfinite(electric_wtp_value):
                raise ValueError(
                    "industrial_hybrid_otc requires finite price forecasts for "
                    "every delivery interval."
                )
            weighted_value += min(eom_price, electric_wtp_value) * duration
            total_hours += duration
        return weighted_value / total_hours if total_hours else 0.0

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        if market_config.product_type != "energy":
            raise ValueError("industrial_hybrid_otc is an energy-market strategy.")
        if self.auxiliary_security_value is None:
            raise ValueError(
                "industrial_hybrid_otc requires bidding_strategy_params."
                "auxiliary_security_value."
            )
        if not math.isfinite(float(self.auxiliary_security_value)):
            raise ValueError(
                "industrial_hybrid_otc auxiliary_security_value must be finite."
            )
        route, calciner, storage = hybrid_route(unit, "industrial_hybrid_otc")

        bid_context = {"products": {}}
        bids: Orderbook = []
        for start, end, only_hours in product_tuples:
            hours = hours_in_product(unit, start, end, require_complete=True)
            if not hours:
                continue
            try:
                process_ceiling = self.process_bid_ceiling(
                    unit, hours, route, calciner, storage
                )
            except (KeyError, IndexError, TypeError, ValueError):
                # A physical long-term contract is not bid without full forecasts.
                continue

            tranches = {
                "auxiliary": {
                    "volume": adjust_bid_volume(
                        self.auxiliary_security_volume(unit, calciner, hours),
                        market_config,
                    ),
                    "price": adjust_bid_price(
                        float(self.auxiliary_security_value), market_config
                    ),
                    "hours": hours,
                }
            }
            if route == "direct":
                tranches["direct_process"] = {
                    "volume": adjust_bid_volume(
                        self.direct_process_volume(unit, calciner, hours),
                        market_config,
                    ),
                    "price": adjust_bid_price(process_ceiling, market_config),
                    "hours": hours,
                }

            bid_context["products"][start] = tranches
            for tranche, item in tranches.items():
                if item["volume"] > 0:
                    bids.append(
                        {
                            "start_time": start,
                            "end_time": end,
                            "only_hours": only_hours,
                            "price": item["price"],
                            "volume": -item["volume"],
                            "industrial_hybrid_tranche": tranche,
                            "node": unit.node,
                        }
                    )

        self.pending_bid_contexts[(unit.id, market_config.market_id)] = bid_context
        return self.remove_empty_bids(bids)

    def on_market_feedback(
        self, unit: SupportsMinMax, market_config: MarketConfig, orderbook: Orderbook
    ) -> None:
        """Persist accepted OTC MW, including partial block acceptance, by hour."""
        bid_context = self.pending_bid_contexts.pop(
            (unit.id, market_config.market_id), None
        )
        if bid_context is None:
            return

        procurement = otc_profile(unit).copy()
        for order in orderbook:
            item = bid_context["products"].get(order.get("start_time"), {}).get(
                order.get("industrial_hybrid_tranche")
            )
            if item is None or item["volume"] <= 0:
                continue
            accepted_power = item["volume"] * acceptance_fraction(order, item["volume"])
            if accepted_power <= 0:
                continue
            for timestamp in item["hours"]:
                procurement[timestamp] = procurement.get(timestamp, 0.0) + accepted_power
        unit.industrial_hybrid_otc_procurement = procurement
