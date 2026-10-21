# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Generic sector-coupled ASSUME unit.

The unit is a physical parent system that coordinates several internal DSM
components while exposing either:

1. one aggregate market position, or
2. component-level market positions linked to the physical parent.

The first implementation contains presets for:
    - heat systems:
        heat_pump + boiler + thermal_storage
    - hydrogen systems:
        electrolyser + hydrogen_seasonal_storage + fuel cell

Additional sectors can be added by supplying ``carrier_balances`` instead of
creating another parent-unit class.

Important modelling principle
-----------------------------
The internal components are never optimized independently. Carrier balances,
storage states and all component constraints remain inside one parent Pyomo
optimization.

Positive ``total_power_input`` means physical electricity consumption.
Electricity-market demand therefore uses the opposite sign.
"""

from __future__ import annotations

import copy
from typing import Any

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

from assume.common.base import SupportsMinMax
from assume.units.dsm_load_shift import DSMFlex


class SectorCoupledSystem(DSMFlex, SupportsMinMax):
    """
    Generic multi-component, sector-coupled parent unit.

    Parameters
    ----------
    system_type : str
        Preset physical configuration. Currently ``"heat"`` and
        ``"hydrogen"`` are provided.
    carrier_balances : dict, optional
        Custom carrier-balance description. If given, it overrides the preset
        balances and allows additional future sectors without defining a new
        ASSUME unit class.
    market_representation : {"aggregate", "component"}
        ``"aggregate"`` exposes the parent unit only.
        ``"component"`` exposes child market IDs linked by ``parent_unit_id``.
    absolute_demands : dict[str, float], optional
        Optional horizon-wide demand constraints per carrier.
    enforce_terminal_soc : bool
        If True, storage SOC is returned to its initial/reference terminal
        level in full-horizon operation.
    """

    _component_market_tolerance = 1e-7

    SYSTEM_PRESETS: dict[str, dict[str, Any]] = {
        "heat": {
            "primary_carrier": "heat",
            "balances": {
                "heat": {
                    "demand_attr": "heat_demand",
                    "sense": "eq",
                    "required_prefixes": ["heat_pump"],
                    "allowed_prefixes": [
                        "heat_pump",
                        "boiler",
                        "thermal_storage",
                    ],
                    "producers": [
                        {"prefix": "heat_pump", "variable": "heat_out"},
                        {"prefix": "boiler", "variable": "heat_out"},
                    ],
                    "storages": [
                        {
                            "prefix": "thermal_storage",
                            "charge": "charge",
                            "discharge": "discharge",
                            "soc": "soc",
                        }
                    ],
                }
            },
        },
        "hydrogen": {
            "primary_carrier": "hydrogen",
            "balances": {
                "hydrogen": {
                    "demand_attr": "hydrogen_demand",
                    "sense": "eq",
                    "required_prefixes": ["electrolyser"],
                    "allowed_prefixes": [
                        "electrolyser",
                        "hydrogen_seasonal_storage",
                        "fuel_cell",
                    ],
                    "producers": [
                        {
                            "prefix": "electrolyser",
                            "variable": "hydrogen_out",
                        }
                    ],
                    "consumers": [
                        {
                            "prefix": "fuel_cell",
                            "variable": "hydrogen_in",
                        }
                    ],
                    "storages": [
                        {
                            "prefix": "hydrogen_seasonal_storage",
                            "charge": "charge",
                            "discharge": "discharge",
                            "soc": "soc",
                        }
                    ],
                }
            },
        },
    }

    def _component_power_expr(self, m, t):
        """
        Net electrical exchange of the sector-coupled system.

        Positive = electricity consumption
        Negative = electricity generation/export
        """
        expr = 0

        for name in m.dsm_blocks:
            block = m.dsm_blocks[name]

            if hasattr(block, "power_in"):
                expr += block.power_in[t]

            if hasattr(block, "power_out"):
                expr -= block.power_out[t]

        return expr

    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        forecaster,
        components: dict[str, dict] | None = None,
        system_type: str = "heat",
        carrier_balances: dict[str, dict[str, Any]] | None = None,
        primary_carrier: str | None = None,
        technology: str = "sector_coupled_system",
        objective: str = "min_variable_cost",
        flexibility_measure: str = "cost_based_load_shift",
        cost_tolerance: float = 10,
        node: str = "node0",
        location: tuple[float, float] = (0.0, 0.0),
        market_representation: str = "aggregate",
        absolute_demands: dict[str, float] | None = None,
        demand: float = 0.0,
        enforce_terminal_soc: bool = True,
        **kwargs,
    ):
        if components is None:
            components = {}

        if carrier_balances is None:
            if system_type not in self.SYSTEM_PRESETS:
                raise ValueError(
                    f"Unknown system_type '{system_type}'. "
                    f"Use one of {list(self.SYSTEM_PRESETS)} or provide carrier_balances."
                )
            preset = copy.deepcopy(self.SYSTEM_PRESETS[system_type])
            carrier_balances = preset["balances"]
            if primary_carrier is None:
                primary_carrier = preset["primary_carrier"]
        elif primary_carrier is None:
            if not carrier_balances:
                raise ValueError("carrier_balances must not be empty.")
            primary_carrier = next(iter(carrier_balances))

        if market_representation not in {"aggregate", "component"}:
            raise ValueError(
                "market_representation must be 'aggregate' or 'component'."
            )

        self.system_type = system_type
        self.carrier_balances = copy.deepcopy(carrier_balances)
        self.primary_carrier = primary_carrier
        self.market_representation = market_representation
        self.enforce_terminal_soc = enforce_terminal_soc
        self.objective = objective
        self.flexibility_measure = flexibility_measure
        self.cost_tolerance = cost_tolerance

        self.absolute_demands = dict(absolute_demands or {})
        if demand and primary_carrier not in self.absolute_demands:
            self.absolute_demands[primary_carrier] = float(demand)

        self._validate_component_configuration(components)

        self.carrier_demands: dict[str, Any] = {}
        for carrier, spec in self.carrier_balances.items():
            demand_attr = spec.get("demand_attr")
            if not demand_attr:
                raise ValueError(f"Carrier '{carrier}' requires a demand_attr.")
            if not hasattr(forecaster, demand_attr):
                raise ValueError(
                    f"Forecaster for '{id}' does not provide '{demand_attr}' "
                    f"required by carrier '{carrier}'."
                )
            self.carrier_demands[carrier] = getattr(forecaster, demand_attr)

        # Existing long-term H2 storage uses an external seasonal schedule.
        if system_type == "hydrogen" and hasattr(
            forecaster, "seasonal_storage_schedule"
        ):
            for name, cfg in components.items():
                if (
                    name.startswith("hydrogen_seasonal_storage")
                    and isinstance(cfg, dict)
                    and cfg.get("storage_type", "short-term") == "long-term"
                ):
                    cfg["storage_schedule_profile"] = (
                        forecaster.seasonal_storage_schedule
                    )

        super().__init__(
            id=id,
            unit_operator=unit_operator,
            technology=technology,
            components=components,
            bidding_strategies=bidding_strategies,
            forecaster=forecaster,
            node=node,
            location=location,
            **kwargs,
        )

        self.storage_components = self._matching_storage_components(
            self.components
        )

    # Configuration helpers
    @staticmethod
    def _matches_prefix(component_name: str, prefix: str) -> bool:
        return component_name == prefix or component_name.startswith(
            f"{prefix}_"
        )

    def _names_matching_prefix(
        self,
        prefix: str,
        components: dict | None = None,
    ) -> list[str]:
        source = self.components if components is None else components
        return [
            name
            for name in source
            if self._matches_prefix(name, prefix)
        ]

    def _matching_storage_components(
        self,
        components: dict | None = None,
    ) -> list[str]:
        names: list[str] = []
        source = self.components if components is None else components

        for spec in self.carrier_balances.values():
            for storage in spec.get("storages", []):
                prefix = storage["prefix"]
                for name in source:
                    if self._matches_prefix(name, prefix) and name not in names:
                        names.append(name)
        return names

    def _validate_component_configuration(
        self,
        components: dict[str, dict],
    ) -> None:
        required_prefixes: set[str] = set()
        allowed_prefixes: set[str] = set()

        for spec in self.carrier_balances.values():
            required_prefixes.update(spec.get("required_prefixes", []))
            allowed_prefixes.update(spec.get("allowed_prefixes", []))
            allowed_prefixes.update(
                term["prefix"] for term in spec.get("producers", [])
            )
            allowed_prefixes.update(
                term["prefix"] for term in spec.get("storages", [])
            )

        for required in required_prefixes:
            if not any(
                self._matches_prefix(name, required)
                for name in components
            ):
                raise ValueError(
                    f"SectorCoupledSystem requires at least one "
                    f"'{required}' component."
                )

        if allowed_prefixes:
            for name in components:
                if not any(
                    self._matches_prefix(name, prefix)
                    for prefix in allowed_prefixes
                ):
                    raise ValueError(
                        f"Component '{name}' is not valid for "
                        f"system_type='{self.system_type}'. "
                        f"Allowed prefixes: {sorted(allowed_prefixes)}"
                    )

    def _component_output_variable(
        self,
        component_name: str,
    ) -> str | None:
        for spec in self.carrier_balances.values():
            for term in spec.get("producers", []):
                if self._matches_prefix(component_name, term["prefix"]):
                    return term["variable"]
        return None

    def _storage_spec(
        self,
        component_name: str,
    ) -> dict[str, Any] | None:
        for spec in self.carrier_balances.values():
            for storage in spec.get("storages", []):
                if self._matches_prefix(component_name, storage["prefix"]):
                    return storage
        return None

    # Pyomo physical model
    def define_parameters(self):
        self.model.electricity_price = pyo.Param(
            self.model.time_steps,
            initialize={
                t: value
                for t, value in enumerate(
                    self._values_for_model(self.forecaster.electricity_price)
                )
            },
        )

        for carrier, series in self.carrier_demands.items():
            self.model.add_component(
                f"{carrier}_demand",
                pyo.Param(
                    self.model.time_steps,
                    initialize={
                        t: value
                        for t, value in enumerate(
                            self._values_for_model(series)
                        )
                    },
                ),
            )

        for carrier, value in self.absolute_demands.items():
            self.model.add_component(
                f"absolute_{carrier}_demand",
                pyo.Param(initialize=float(value)),
            )

        # Fuel prices currently required by Boiler components.
        raw_components = self.components
        boiler_fuel_types = {
            cfg.get("fuel_type", "electricity")
            for name, cfg in raw_components.items()
            if self._matches_prefix(name, "boiler")
            and isinstance(cfg, dict)
        }

        if "natural_gas" in boiler_fuel_types:
            self.model.natural_gas_price = pyo.Param(
                self.model.time_steps,
                initialize={
                    t: value
                    for t, value in enumerate(
                        self._values_for_model(
                            self.forecaster.get_price("natural gas")
                        )
                    )
                },
            )

        if "hydrogen_gas" in boiler_fuel_types:
            self.model.hydrogen_gas_price = pyo.Param(
                self.model.time_steps,
                initialize={
                    t: value
                    for t, value in enumerate(
                        self._values_for_model(
                            self.forecaster.get_price("hydrogen")
                        )
                    )
                },
            )

    def define_variables(self):
        self.model.total_power_input = pyo.Var(
            self.model.time_steps,
            within=pyo.Reals,
        )
        self.model.variable_cost = pyo.Var(
            self.model.time_steps,
            within=pyo.Reals,
        )

        for carrier, value in self.absolute_demands.items():
            if value and value > 0:
                self.model.add_component(
                    f"{carrier}_served",
                    pyo.Var(
                        self.model.time_steps,
                        within=pyo.NonNegativeReals,
                    ),
                )

    def _carrier_supply_expr(self, m, carrier: str, t):
        spec = self.carrier_balances[carrier]
        expr = 0

        for producer in spec.get("producers", []):
            for name in self._names_matching_prefix(producer["prefix"]):
                block = m.dsm_blocks[name]
                variable = producer["variable"]
                if not hasattr(block, variable):
                    raise AttributeError(
                        f"Component '{name}' does not expose '{variable}' "
                        f"required by carrier '{carrier}'."
                    )
                expr += getattr(block, variable)[t]


        for consumer in spec.get("consumers", []):
            for name in self._names_matching_prefix(consumer["prefix"]):
                block = m.dsm_blocks[name]
                variable = consumer["variable"]

                if not hasattr(block, variable):
                    raise AttributeError(
                        f"Component '{name}' does not expose '{variable}' "
                        f"required by carrier '{carrier}'."
                    )

                expr -= getattr(block, variable)[t]


        for storage in spec.get("storages", []):
            for name in self._names_matching_prefix(storage["prefix"]):
                block = m.dsm_blocks[name]
                expr += getattr(
                    block, storage.get("discharge", "discharge")
                )[t]
                expr -= getattr(
                    block, storage.get("charge", "charge")
                )[t]

        return expr

    def initialize_process_sequence(self):
        self.model.carrier_balance_constraints = pyo.ConstraintList()

        for carrier, spec in self.carrier_balances.items():
            demand_param = getattr(self.model, f"{carrier}_demand")
            absolute = float(self.absolute_demands.get(carrier, 0.0))

            for t in self.model.time_steps:
                supply = self._carrier_supply_expr(self.model, carrier, t)

                if absolute > 0:
                    served = getattr(self.model, f"{carrier}_served")[t]
                    self.model.carrier_balance_constraints.add(
                        supply == served
                    )
                    continue

                sense = spec.get("sense", "eq")
                if sense == "eq":
                    self.model.carrier_balance_constraints.add(
                        supply == demand_param[t]
                    )
                elif sense == "ge":
                    self.model.carrier_balance_constraints.add(
                        supply >= demand_param[t]
                    )
                else:
                    raise ValueError(
                        f"Unsupported balance sense '{sense}' "
                        f"for carrier '{carrier}'."
                    )

    def define_constraints(self):
        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
            return (
                m.total_power_input[t]
                == self._component_power_expr(m, t)
            )

        @self.model.Constraint(self.model.time_steps)
        def variable_cost_constraint(m, t):
            cost_terms = [
                m.dsm_blocks[name].operating_cost[t]
                for name in m.dsm_blocks
                if hasattr(m.dsm_blocks[name], "operating_cost")
            ]
            if not cost_terms:
                return m.variable_cost[t] == 0
            return m.variable_cost[t] == pyo.quicksum(cost_terms)

        self.model.absolute_carrier_demand_constraints = pyo.ConstraintList()
        for carrier, value in self.absolute_demands.items():
            if not value or value <= 0:
                continue
            served = getattr(self.model, f"{carrier}_served")
            absolute_param = getattr(
                self.model, f"absolute_{carrier}_demand"
            )
            self.model.absolute_carrier_demand_constraints.add(
                pyo.quicksum(
                    served[t] for t in self.model.time_steps
                )
                >= absolute_param
            )

        if (
            self.enforce_terminal_soc
            and self.horizon_mode == "full_horizon"
            and self.storage_components
        ):
            last_t = self.model.time_steps.last()
            self.model.sector_terminal_soc = pyo.ConstraintList()

            for name in self.storage_components:
                block = self.model.dsm_blocks[name]
                if hasattr(block, "soc") and hasattr(block, "initial_soc"):
                    self.model.sector_terminal_soc.add(
                        block.soc[last_t] == block.initial_soc
                    )

    # Market/reference helpers
    def _component_market_id(self, component_name: str) -> str:
        component = self.components[component_name]
        kwargs = getattr(component, "kwargs", {}) or {}
        market_id = kwargs.get("market_unit_id")

        if market_id is not None and not pd.isna(market_id):
            return str(market_id)

        return f"{self.id}_{component_name}"

    def get_component_market_ids(self) -> dict[str, str]:
        return {
            name: self._component_market_id(name)
            for name in self.components
        }

    def _index_position(self, start) -> int:
        for pos, timestamp in enumerate(self.index):
            if timestamp == start:
                return pos
        raise KeyError(
            f"Timestamp {start!r} is not contained in "
            f"SectorCoupledSystem '{self.id}'."
        )

    @staticmethod
    def _value_at(
        series: pd.Series | None,
        timestamp,
        default: float = 0.0,
    ) -> float:
        if series is None:
            return float(default)
        try:
            return float(series.at[timestamp])
        except (KeyError, AttributeError, TypeError):
            return float(default)

    def _extract_component_trajectory(
        self,
        instance: pyo.ConcreteModel,
    ) -> dict[str, dict[str, pd.Series]]:
        result: dict[str, dict[str, pd.Series]] = {}

        for component_name in self.components:
            block = instance.dsm_blocks[component_name]
            variable_series: dict[str, pd.Series] = {}

            for variable in block.component_objects(pyo.Var, active=True):
                if not variable.is_indexed():
                    continue

                values = []
                usable = True
                for t in instance.time_steps:
                    try:
                        values.append(float(pyo.value(variable[t])))
                    except (KeyError, ValueError, TypeError):
                        usable = False
                        break

                if usable and len(values) == len(self.index):
                    variable_series[variable.local_name] = pd.Series(
                        values,
                        index=list(self.index),
                        dtype=float,
                    )

            result[component_name] = variable_series

        return result

    def _store_market_reference(
        self,
        instance: pyo.ConcreteModel,
    ) -> None:
        self.da_power = pd.Series(
            [
                float(pyo.value(instance.total_power_input[t]))
                for t in instance.time_steps
            ],
            index=list(self.index),
            dtype=float,
        )

        self.da_component_state = self._extract_component_trajectory(
            instance
        )

        self.da_component_power = {}
        self.da_storage_soc = {}
        self.da_storage_charge = {}
        self.da_storage_discharge = {}

        for name, state in self.da_component_state.items():
            if "power_in" in state:
                self.da_component_power[name] = state["power_in"]
            if "soc" in state:
                self.da_storage_soc[name] = state["soc"]
            if "charge" in state:
                self.da_storage_charge[name] = state["charge"]
            if "discharge" in state:
                self.da_storage_discharge[name] = state["discharge"]

        self._sector_redispatch_cache = {}

    def prepare_market_state(self, force: bool = False) -> None:
        """Solve and store the physically consistent cost-optimal reference."""
        if (
            not force
            and hasattr(self, "da_component_state")
            and hasattr(self, "da_power")
        ):
            return

        instance = self.model.create_instance()
        instance = self.switch_to_opt(instance)

        results = self.solver.solve(instance, options={}, tee=False)

        if not (
            results.solver.status == SolverStatus.ok
            and results.solver.termination_condition
            == TerminationCondition.optimal
        ):
            raise RuntimeError(
                f"Reference solve failed for SectorCoupledSystem "
                f"'{self.id}'. status={results.solver.status}, "
                f"termination={results.solver.termination_condition}"
            )

        self._store_market_reference(instance)

    def _reference_component_value(
        self,
        component_name: str,
        variable_name: str,
        start,
        default: float = 0.0,
    ) -> float:
        self.prepare_market_state()
        series = (
            self.da_component_state
            .get(component_name, {})
            .get(variable_name)
        )
        return self._value_at(series, start, default)

    @staticmethod
    def _snapshot_value(
        snapshot: dict[str, dict[str, float]],
        component_name: str,
        variable_name: str,
        default: float = 0.0,
    ) -> float:
        return float(
            snapshot.get(component_name, {}).get(variable_name, default)
        )

    def _extract_component_snapshot(
        self,
        instance: pyo.ConcreteModel,
        target_t: int,
    ) -> dict[str, dict[str, float]]:
        snapshot: dict[str, dict[str, float]] = {}

        for component_name in self.components:
            block = instance.dsm_blocks[component_name]
            values: dict[str, float] = {}

            for variable in block.component_objects(pyo.Var, active=True):
                if not variable.is_indexed():
                    continue
                try:
                    values[variable.local_name] = float(
                        pyo.value(variable[target_t])
                    )
                except (KeyError, ValueError, TypeError):
                    continue

            snapshot[component_name] = values

        return snapshot

    # State-aware redispatch envelope
    def _fix_committed_past_to_reference(
        self,
        instance: pyo.ConcreteModel,
        target_t: int,
    ) -> None:
        if target_t <= 0:
            return

        for component_name, states in self.da_component_state.items():
            block = instance.dsm_blocks[component_name]

            for variable_name, series in states.items():
                if not hasattr(block, variable_name):
                    continue

                variable = getattr(block, variable_name)
                if not variable.is_indexed():
                    continue

                for t in instance.time_steps:
                    if t >= target_t:
                        break
                    try:
                        value = float(series.iloc[t])
                        lb, ub = variable[t].lb, variable[t].ub
                        if lb is not None:
                            value = max(value, lb)
                        if ub is not None:
                            value = min(value, ub)
                        variable[t].fix(value)
                    except (
                        KeyError,
                        ValueError,
                        TypeError,
                        AttributeError,
                    ):
                        continue

        for t in instance.time_steps:
            if t >= target_t:
                break
            instance.total_power_input[t].fix(float(self.da_power.iloc[t]))

    def _add_reference_terminal_soc(
        self,
        instance: pyo.ConcreteModel,
    ) -> None:
        if not self.storage_components:
            return

        final_t = list(instance.time_steps)[-1]
        instance.market_terminal_soc = pyo.ConstraintList()

        for name in self.storage_components:
            reference_soc = (
                self.da_component_state
                .get(name, {})
                .get("soc")
            )
            if reference_soc is None:
                continue

            block = instance.dsm_blocks[name]
            if not hasattr(block, "soc"):
                continue

            instance.market_terminal_soc.add(
                block.soc[final_t]
                == float(reference_soc.iloc[final_t])
            )

    def _solve_redispatch_extreme(
        self,
        start,
        sense: str,
    ) -> tuple[float, dict[str, dict[str, float]]]:
        if sense not in {"min", "max"}:
            raise ValueError("sense must be either 'min' or 'max'.")

        self.prepare_market_state()

        cache = getattr(self, "_sector_redispatch_cache", {})
        key = (start, sense)

        if key in cache:
            return cache[key]

        target_t = self._index_position(start)

        instance = self.model.create_instance()
        instance = self.switch_to_opt(instance)

        if hasattr(instance, "obj_rule_opt"):
            instance.obj_rule_opt.deactivate()

        self._fix_committed_past_to_reference(instance, target_t)
        self._add_reference_terminal_soc(instance)

        total_cost = pyo.quicksum(
            instance.variable_cost[t]
            for t in instance.time_steps
        )
        epsilon = 1e-9

        if sense == "min":
            expression = (
                instance.total_power_input[target_t]
                + epsilon * total_cost
            )
            objective_sense = pyo.minimize
        else:
            expression = (
                instance.total_power_input[target_t]
                - epsilon * total_cost
            )
            objective_sense = pyo.maximize

        instance.sector_market_bound_objective = pyo.Objective(
            expr=expression,
            sense=objective_sense,
        )

        results = self.solver.solve(instance, options={}, tee=False)

        if not (
            results.solver.status == SolverStatus.ok
            and results.solver.termination_condition
            == TerminationCondition.optimal
        ):
            raise RuntimeError(
                f"{sense} redispatch-bound solve failed for "
                f"'{self.id}' at {start}. "
                f"status={results.solver.status}, "
                f"termination={results.solver.termination_condition}"
            )

        power = float(
            pyo.value(instance.total_power_input[target_t])
        )
        snapshot = self._extract_component_snapshot(
            instance,
            target_t,
        )

        cache[key] = (power, snapshot)
        self._sector_redispatch_cache = cache

        return power, snapshot

    def get_redispatch_power_bounds(
        self,
        start,
    ) -> tuple[float, float]:
        """
        Return min/max feasible physical electricity consumption.

        The complete parent system is re-optimized; carrier balances,
        storage states and all component constraints remain active.
        """
        tolerance = self._component_market_tolerance

        self.prepare_market_state()

        min_consumption, _ = self._solve_redispatch_extreme(
            start, "min"
        )
        max_consumption, _ = self._solve_redispatch_extreme(
            start, "max"
        )
        base_consumption = float(self.da_power.at[start])

        if min_consumption > base_consumption + tolerance:
            raise RuntimeError(
                f"Reference consumption of '{self.id}' at {start} "
                f"is below calculated minimum: "
                f"base={base_consumption}, min={min_consumption}."
            )

        if base_consumption > max_consumption + tolerance:
            raise RuntimeError(
                f"Reference consumption of '{self.id}' at {start} "
                f"is above calculated maximum: "
                f"base={base_consumption}, max={max_consumption}."
            )

        if abs(min_consumption) < tolerance:
            min_consumption = 0.0
        if abs(max_consumption) < tolerance:
            max_consumption = 0.0

        return float(min_consumption), float(max_consumption)

    # Market positions
    def get_component_eom_positions(
        self,
        start,
    ) -> list[dict[str, Any]]:
        """Return direct electrical EOM positions per internal component."""
        self.prepare_market_state()
        positions = []

        for component_name in self.components:
            power = self._reference_component_value(
                component_name,
                "power_in",
                start,
                0.0,
            )

            positions.append(
                {
                    "unit_id": self._component_market_id(component_name),
                    "parent_unit_id": self.id,
                    "component_type": component_name,
                    "node": self.node,
                    "volume": -float(power),
                    "physical_power": float(power),
                }
            )

        summed_power = sum(
            item["physical_power"] for item in positions
        )
        parent_power = float(self.da_power.at[start])

        if abs(summed_power - parent_power) > 1e-6:
            raise RuntimeError(
                f"Component electrical powers of '{self.id}' "
                f"do not sum to total_power_input at {start}: "
                f"components={summed_power}, parent={parent_power}."
            )

        return positions

    def _sector_flexibility_weights(
        self,
        start,
        min_snapshot: dict[str, dict[str, float]],
        max_snapshot: dict[str, dict[str, float]],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Heuristic physical attribution of parent electrical flexibility.

        This attribution is an accounting decomposition, not an independent
        feasibility claim for each child component.
        """
        tolerance = self._component_market_tolerance
        up_weights: dict[str, float] = {}
        down_weights: dict[str, float] = {}

        for name in self.components:
            storage = self._storage_spec(name)

            if storage is not None:
                charge_var = storage.get("charge", "charge")
                discharge_var = storage.get("discharge", "discharge")

                base_net = (
                    self._reference_component_value(
                        name, discharge_var, start, 0.0
                    )
                    - self._reference_component_value(
                        name, charge_var, start, 0.0
                    )
                )
                min_net = (
                    self._snapshot_value(
                        min_snapshot, name, discharge_var, 0.0
                    )
                    - self._snapshot_value(
                        min_snapshot, name, charge_var, 0.0
                    )
                )
                max_net = (
                    self._snapshot_value(
                        max_snapshot, name, discharge_var, 0.0
                    )
                    - self._snapshot_value(
                        max_snapshot, name, charge_var, 0.0
                    )
                )

                up_weights[name] = max(0.0, min_net - base_net)
                down_weights[name] = max(0.0, base_net - max_net)
                continue

            output_var = self._component_output_variable(name)
            if output_var is None:
                continue

            base_power = self._reference_component_value(
                name, "power_in", start, 0.0
            )
            min_power = self._snapshot_value(
                min_snapshot, name, "power_in", 0.0
            )
            max_power = self._snapshot_value(
                max_snapshot, name, "power_in", 0.0
            )

            # Direct electrical components are handled by the fallback below.
            if max(
                abs(base_power),
                abs(min_power),
                abs(max_power),
            ) > tolerance:
                continue

            base_output = self._reference_component_value(
                name, output_var, start, 0.0
            )
            min_output = self._snapshot_value(
                min_snapshot, name, output_var, 0.0
            )
            max_output = self._snapshot_value(
                max_snapshot, name, output_var, 0.0
            )

            up_weights[name] = max(
                0.0, min_output - base_output
            )
            down_weights[name] = max(
                0.0, base_output - max_output
            )

        return up_weights, down_weights

    @staticmethod
    def _allocate_total_by_weights(
        total: float,
        weights: dict[str, float],
        names: list[str],
        tolerance: float,
    ) -> dict[str, float]:
        allocation = {name: 0.0 for name in names}
        positive = {
            name: max(0.0, float(weights.get(name, 0.0)))
            for name in names
        }
        weight_sum = sum(positive.values())

        if total <= tolerance or weight_sum <= tolerance:
            return allocation

        for name, weight in positive.items():
            allocation[name] = total * weight / weight_sum

        residual = total - sum(allocation.values())
        if abs(residual) > tolerance:
            best = max(positive, key=positive.get)
            allocation[best] += residual

        return allocation

    def get_component_redispatch_positions(
        self,
        start,
    ) -> list[dict[str, Any]]:
        """Decompose the physically feasible parent RD envelope over components."""
        tolerance = self._component_market_tolerance
        self.prepare_market_state()

        base_consumption = float(self.da_power.at[start])
        min_consumption, min_snapshot = self._solve_redispatch_extreme(
            start, "min"
        )
        max_consumption, max_snapshot = self._solve_redispatch_extreme(
            start, "max"
        )

        up_total = max(
            0.0, base_consumption - min_consumption
        )
        down_total = max(
            0.0, max_consumption - base_consumption
        )

        names = list(self.components)

        up_weights, down_weights = self._sector_flexibility_weights(
            start,
            min_snapshot,
            max_snapshot,
        )

        up_allocation = self._allocate_total_by_weights(
            up_total,
            up_weights,
            names,
            tolerance,
        )
        down_allocation = self._allocate_total_by_weights(
            down_total,
            down_weights,
            names,
            tolerance,
        )

        # Fallback: attribute unexplained flexibility to direct changes in
        # electrical input.
        if (
            up_total > tolerance
            and sum(up_allocation.values()) <= tolerance
        ):
            direct_up = {}
            for name in names:
                base_power = self._reference_component_value(
                    name, "power_in", start, 0.0
                )
                min_power = self._snapshot_value(
                    min_snapshot, name, "power_in", 0.0
                )
                direct_up[name] = max(
                    0.0, base_power - min_power
                )
            up_allocation = self._allocate_total_by_weights(
                up_total,
                direct_up,
                names,
                tolerance,
            )

        if (
            down_total > tolerance
            and sum(down_allocation.values()) <= tolerance
        ):
            direct_down = {}
            for name in names:
                base_power = self._reference_component_value(
                    name, "power_in", start, 0.0
                )
                max_power = self._snapshot_value(
                    max_snapshot, name, "power_in", 0.0
                )
                direct_down[name] = max(
                    0.0, max_power - base_power
                )
            down_allocation = self._allocate_total_by_weights(
                down_total,
                direct_down,
                names,
                tolerance,
            )

        def add_residual(
            total: float,
            allocation: dict[str, float],
        ) -> None:
            residual = total - sum(allocation.values())
            if residual <= tolerance:
                return

            for name in names:
                base_power = self._reference_component_value(
                    name, "power_in", start, 0.0
                )
                if base_power > tolerance:
                    allocation[name] += residual
                    return

            if names:
                allocation[names[0]] += residual

        add_residual(up_total, up_allocation)
        add_residual(down_total, down_allocation)

        records = []

        for name in names:
            baseline_power = self._reference_component_value(
                name,
                "power_in",
                start,
                0.0,
            )
            signed_volume = -baseline_power

            up_cap = max(
                0.0, up_allocation.get(name, 0.0)
            )
            down_cap = max(
                0.0, down_allocation.get(name, 0.0)
            )

            max_power = signed_volume + up_cap
            min_power = signed_volume - down_cap
            p_nom = max(
                abs(signed_volume),
                abs(min_power),
                abs(max_power),
                tolerance,
            )

            records.append(
                {
                    "unit_id": self._component_market_id(name),
                    "parent_unit_id": self.id,
                    "component_type": name,
                    "node": self.node,
                    "volume": float(signed_volume),
                    "min_power": float(min_power),
                    "max_power": float(max_power),
                    "p_nom": float(p_nom),
                    "is_flexible_consumer": True,
                    "parent_volume": -base_consumption,
                    "parent_min_power": -max_consumption,
                    "parent_max_power": -min_consumption,
                    "upward_flexibility": float(up_cap),
                    "downward_flexibility": float(down_cap),
                }
            )

        child_baseline = sum(
            item["volume"] for item in records
        )
        child_up = sum(
            item["max_power"] - item["volume"]
            for item in records
        )
        child_down = sum(
            item["volume"] - item["min_power"]
            for item in records
        )

        if abs(child_baseline + base_consumption) > 1e-6:
            raise RuntimeError(
                f"Component RD baseline of '{self.id}' "
                "does not reproduce parent baseline."
            )
        if abs(child_up - up_total) > 1e-6:
            raise RuntimeError(
                f"Component upward RD headroom of '{self.id}' "
                "does not reproduce parent headroom."
            )
        if abs(child_down - down_total) > 1e-6:
            raise RuntimeError(
                f"Component downward RD headroom of '{self.id}' "
                "does not reproduce parent headroom."
            )

        return records

    # Public market interface used by generic strategies
    def get_eom_market_positions(
        self,
        start,
    ) -> list[dict[str, Any]]:
        self.prepare_market_state()

        if self.market_representation == "component":
            return self.get_component_eom_positions(start)

        physical_power = float(self.da_power.at[start])
        return [
            {
                "unit_id": self.id,
                "parent_unit_id": self.id,
                "component_type": "aggregate",
                "node": self.node,
                "volume": -physical_power,
                "physical_power": physical_power,
            }
        ]

    def get_redispatch_market_positions(
        self,
        start,
    ) -> list[dict[str, Any]]:
        self.prepare_market_state()

        if self.market_representation == "component":
            return self.get_component_redispatch_positions(start)

        base = float(self.da_power.at[start])
        min_consumption, max_consumption = (
            self.get_redispatch_power_bounds(start)
        )

        signed_volume = -base
        min_power = -max_consumption
        max_power = -min_consumption
        p_nom = max(
            abs(signed_volume),
            abs(min_power),
            abs(max_power),
            self._component_market_tolerance,
        )

        return [
            {
                "unit_id": self.id,
                "parent_unit_id": self.id,
                "component_type": "aggregate",
                "node": self.node,
                "volume": signed_volume,
                "min_power": min_power,
                "max_power": max_power,
                "p_nom": p_nom,
                "is_flexible_consumer": True,
                "parent_volume": signed_volume,
                "parent_min_power": min_power,
                "parent_max_power": max_power,
                "upward_flexibility": base - min_consumption,
                "downward_flexibility": max_consumption - base,
            }
        ]
