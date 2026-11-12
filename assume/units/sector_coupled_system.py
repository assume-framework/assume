# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
from typing import Any

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

from assume.common.base import SupportsMinMax
from assume.units.dsm_load_shift import DSMFlex


def _flows(variable, *prefixes):
    return [{"prefix": p, "variable": variable} for p in prefixes]


def _storage(prefix):
    return dict(prefix=prefix, charge="charge", discharge="discharge", soc="soc")


def _heat_balance(required=("heat_pump",)):
    return {
        "demand_attr": "heat_demand",
        "sense": "eq",
        "required_prefixes": list(required),
        "allowed_prefixes": ["heat_pump", "boiler", "thermal_storage", "chp"],
        "producers": _flows("heat_out", "heat_pump", "boiler", "chp"),
        "storages": [_storage("thermal_storage")],
    }


def _hydrogen_balance(include_chp=False):
    allowed_prefixes = ["electrolyser", "hydrogen_seasonal_storage", "fuel_cell"]

    consumers = _flows("hydrogen_in", "fuel_cell")

    if include_chp:
        allowed_prefixes.append("chp")
        consumers += _flows("hydrogen_in", "chp")

    return {
        "demand_attr": "hydrogen_demand",
        "sense": "eq",
        "required_prefixes": ["electrolyser"],
        "allowed_prefixes": allowed_prefixes,
        "producers": _flows("hydrogen_out", "electrolyser"),
        "consumers": consumers,
        "storages": [_storage("hydrogen_seasonal_storage")],
    }


class SectorCoupledSystem(DSMFlex, SupportsMinMax):
    """
    Generic sector-coupled ASSUME unit.

    The unit is a physical parent system that coordinates several internal DSM
    components while exposing either:

    1. one aggregate market position, or
    2. component-level market positions linked to the physical parent.

    Presets (``system_type``):
        - ``heat``: heat_pump + boiler + chp + thermal_storage
        - ``hydrogen``: electrolyser + hydrogen_seasonal_storage + fuel_cell
        - ``heat_hydrogen``: both balances; an H2-fired CHP draws its hydrogen
        from the internal H2 balance and supplies the heat balance

    Additional sectors can be added by supplying ``carrier_balances`` instead of
    creating another parent-unit class.

    Important modelling principle:
    The internal components are never optimized independently. Carrier balances,
    storage states and all component constraints remain inside one parent Pyomo
    optimization.

    Carrier balances:
    The default heat and hydrogen presets use exact carrier balances. For hydrogen,
    production must be allocated to hydrogen demand, storage charging, or another
    explicitly modelled hydrogen-consuming component. Unmodelled hydrogen surplus
    is not permitted.

    Electrical sign convention:
    ``total_power_input`` represents the net electrical grid exchange of the
    sector-coupled parent:

    - positive: electricity import / consumption
    - negative: electricity export / generation

    Negative values are required for systems containing electricity-producing
    components such as fuel cells and CHPs.
    """

    _component_market_tolerance = 1e-7

    SYSTEM_PRESETS: dict[str, dict[str, Any]] = {
        "heat": {
            "primary_carrier": "heat",
            "balances": {"heat": _heat_balance()},
        },
        "hydrogen": {
            "primary_carrier": "hydrogen",
            "balances": {"hydrogen": _hydrogen_balance()},
        },
        "heat_hydrogen": {
            "primary_carrier": "heat",
            "balances": {
                "heat": _heat_balance(required=["chp"]),
                "hydrogen": _hydrogen_balance(include_chp=True),
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
                    f"Unknown system_type '{system_type}'. Use one of "
                    f"{list(self.SYSTEM_PRESETS)} or provide carrier_balances."
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

        # Long-term H2 storage uses an external seasonal schedule.
        if system_type in {"hydrogen", "heat_hydrogen"} and hasattr(
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

        # Store solved rolling windows so redispatch can retrieve the
        # DA reference that originally committed a requested timestep.
        self._rh_market_references: dict[int, dict[str, Any]] = {}

        self.storage_components = self._matching_storage_components(self.components)

    # Configuration helpers
    @staticmethod
    def _matches_prefix(component_name: str, prefix: str) -> bool:
        return component_name == prefix or component_name.startswith(f"{prefix}_")

    def _names_matching_prefix(
        self,
        prefix: str,
        components: dict | None = None,
    ) -> list[str]:
        source = self.components if components is None else components
        return [name for name in source if self._matches_prefix(name, prefix)]

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
                term["prefix"] for term in spec.get("consumers", [])
            )
            allowed_prefixes.update(term["prefix"] for term in spec.get("storages", []))

        for required in required_prefixes:
            if not any(self._matches_prefix(name, required) for name in components):
                raise ValueError(
                    f"SectorCoupledSystem requires at least one '{required}' component."
                )

        if allowed_prefixes:
            for name in components:
                if not any(
                    self._matches_prefix(name, prefix) for prefix in allowed_prefixes
                ):
                    raise ValueError(
                        f"Component '{name}' is not valid for "
                        f"system_type='{self.system_type}'. "
                        f"Allowed prefixes: {sorted(allowed_prefixes)}"
                    )

        # An H2-fired CHP has no external fuel price. It must therefore deliver
        # heat into the heat balance AND draw hydrogen from the hydrogen balance,
        # so both carriers are coupled in one model. Systems without an
        # H2-fired CHP are not affected by this check.
        h2_chps = [
            name
            for name, cfg in components.items()
            if self._matches_prefix(name, "chp")
            and isinstance(cfg, dict)
            and str(cfg.get("fuel_type", "natural_gas")).strip().lower()
            == "hydrogen_gas"
        ]
        if h2_chps:

            def chp_term(carrier: str, role: str, variable: str) -> bool:
                terms = self.carrier_balances.get(carrier, {}).get(role, [])
                return any(
                    term["prefix"] == "chp" and term["variable"] == variable
                    for term in terms
                )

            if not (
                chp_term("heat", "producers", "heat_out")
                and chp_term("hydrogen", "consumers", "hydrogen_in")
            ):
                raise ValueError(
                    f"H2-fired CHP {h2_chps} needs a heat balance with 'chp' "
                    "producing 'heat_out' and a hydrogen balance with 'chp' "
                    "consuming 'hydrogen_in', e.g. system_type='heat_hydrogen'."
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
                        for t, value in enumerate(self._values_for_model(series))
                    },
                ),
            )

        for carrier, value in self.absolute_demands.items():
            self.model.add_component(
                f"absolute_{carrier}_demand",
                pyo.Param(initialize=float(value)),
            )

        # External fuel prices. Natural gas is needed by gas-fired boilers and CHPs;
        # hydrogen only by H2-fired boilers (H2-fired CHP uses internally balanced H2).
        # Defaults match the component classes: Boiler -> electricity,
        # CHP -> natural_gas.
        def fuel_types(prefix, default):
            return {
                cfg.get("fuel_type", default)
                for name, cfg in self.components.items()
                if isinstance(cfg, dict) and self._matches_prefix(name, prefix)
            }

        def price_param(commodity):
            prices = self._values_for_model(self.forecaster.get_price(commodity))
            return pyo.Param(self.model.time_steps, initialize=dict(enumerate(prices)))

        boiler_fuels = fuel_types("boiler", default="electricity")
        chp_fuels = fuel_types("chp", default="natural_gas")

        if "natural_gas" in boiler_fuels | chp_fuels:
            self.model.natural_gas_price = price_param("natural gas")
        if "hydrogen_gas" in boiler_fuels:
            self.model.hydrogen_gas_price = price_param("hydrogen")

    def define_variables(self):
        self.model.total_power_input = pyo.Var(
            self.model.time_steps,
            within=pyo.Reals,
            doc=(
                "Net electrical exchange of the sector-coupled system. "
                "Positive values denote electricity import/consumption; "
                "negative values denote electricity export/generation."
            ),
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
                expr += getattr(block, storage.get("discharge", "discharge"))[t]
                expr -= getattr(block, storage.get("charge", "charge"))[t]

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
                    self.model.carrier_balance_constraints.add(supply == served)
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
                        f"Unsupported balance sense '{sense}' for carrier '{carrier}'."
                    )

    def define_constraints(self):
        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
            return m.total_power_input[t] == self._component_power_expr(m, t)

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
            absolute_param = getattr(self.model, f"absolute_{carrier}_demand")
            self.model.absolute_carrier_demand_constraints.add(
                pyo.quicksum(served[t] for t in self.model.time_steps) >= absolute_param
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
        return {name: self._component_market_id(name) for name in self.components}

    def _index_position(self, start) -> int:
        for pos, timestamp in enumerate(self.index):
            if timestamp == start:
                return pos
        raise KeyError(
            f"Timestamp {start!r} is not contained in SectorCoupledSystem '{self.id}'."
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

    def _on_rolling_window_solved(
        self,
        instance: pyo.ConcreteModel,
        window_start: int,
        window_end: int,
        commit_end: int,
        saved_index,
    ) -> None:
        """
        Capture the solved rolling-horizon window as the market reference.

        The complete solved look-ahead window is retained in
        ``_rh_market_reference`` so redispatch can later be evaluated around the
        exact same physical trajectory and initial state.

        Only the committed part of the rolling window is merged into
        ``da_power`` and ``da_component_state``.  Look-ahead-only timesteps are
        provisional and must not overwrite the chronological DA reference until
        they are actually committed by a later rolling window.
        """
        if self.horizon_mode != "rolling_horizon":
            return

        full_index = list(saved_index)
        window_index = full_index[window_start:window_end]
        n_commit = commit_end - window_start
        committed_index = window_index[:n_commit]

        if len(window_index) != len(list(instance.time_steps)):
            raise RuntimeError(
                f"Rolling-window reference length mismatch for '{self.id}': "
                f"window has {len(window_index)} timestamps but solved instance "
                f"has {len(list(instance.time_steps))} time steps."
            )

        window_power = pd.Series(
            [
                float(pyo.value(instance.total_power_input[t]))
                for t in instance.time_steps
            ],
            index=window_index,
            dtype=float,
        )

        # During the rolling solve self.index is the window index, so this
        # extracts a physically consistent trajectory on exactly window_index.
        window_component_state = self._extract_component_trajectory(instance)

        # Keep the whole current look-ahead solution.  This is deliberately
        # separate from da_* because only the committed part is authoritative.
        self._rh_market_reference = {
            "window_start": int(window_start),
            "window_end": int(window_end),
            "commit_end": int(commit_end),
            "index": pd.Index(window_index),
            "committed_index": pd.Index(committed_index),
            "power": window_power.copy(deep=True),
            "component_state": {
                name: {
                    variable: series.copy(deep=True)
                    for variable, series in states.items()
                }
                for name, states in window_component_state.items()
            },
            # The solved Pyomo instance is no longer mutated after this hook;
            # retain it so a later rolling-aware redispatch solve can clone/reuse the
            # exact DA-window model instead of rebuilding a full-horizon model.
            "instance": instance,
        }

        # Retain this solved rolling window so redispatch can still refer to it
        # after a later DA window has been optimized.
        self._rh_market_references[int(window_start)] = self._rh_market_reference

        # Merge committed parent power into the chronological DA reference.
        existing_power = getattr(self, "da_power", None)
        if isinstance(existing_power, pd.Series):
            self.da_power = existing_power.reindex(full_index).astype(float)
        else:
            self.da_power = pd.Series(float("nan"), index=full_index, dtype=float)

        # Anything at/after commit_end is still provisional in rolling mode.
        # Invalidate an older full-horizon/cache value so it cannot be mistaken
        # for an already committed DA reference.
        if commit_end < len(full_index):
            self.da_power.iloc[commit_end:] = float("nan")

        if committed_index:
            self.da_power.loc[committed_index] = window_power.iloc[:n_commit].to_numpy()

        # Merge committed component variables into the chronological reference.
        if not isinstance(getattr(self, "da_component_state", None), dict):
            self.da_component_state = {}

        for component_name, states in window_component_state.items():
            target_states = self.da_component_state.setdefault(component_name, {})

            for variable_name, window_series in states.items():
                existing_series = target_states.get(variable_name)

                if isinstance(existing_series, pd.Series):
                    full_series = existing_series.reindex(full_index).astype(float)
                else:
                    full_series = pd.Series(float("nan"), index=full_index, dtype=float)

                if commit_end < len(full_index):
                    full_series.iloc[commit_end:] = float("nan")

                if committed_index:
                    full_series.loc[committed_index] = window_series.iloc[
                        :n_commit
                    ].to_numpy()

                target_states[variable_name] = full_series

        self._rebuild_market_reference_views()

    def _rebuild_market_reference_views(self) -> None:
        """Rebuild derived market-reference views from ``da_component_state``."""
        self.da_component_power = {}
        self.da_storage_soc = {}
        self.da_storage_charge = {}
        self.da_storage_discharge = {}

        for name, state in self.da_component_state.items():
            power_in = state.get("power_in")
            power_out = state.get("power_out")

            if power_in is not None or power_out is not None:
                if power_in is None:
                    self.da_component_power[name] = -power_out
                elif power_out is None:
                    self.da_component_power[name] = power_in
                else:
                    self.da_component_power[name] = power_in - power_out

            if "soc" in state:
                self.da_storage_soc[name] = state["soc"]

            if "charge" in state:
                self.da_storage_charge[name] = state["charge"]

            if "discharge" in state:
                self.da_storage_discharge[name] = state["discharge"]

        # Any cached RD envelope was calculated around the old reference.
        self._sector_redispatch_cache = {}

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

    def _store_market_reference(self, instance: pyo.ConcreteModel) -> None:
        self.da_power = pd.Series(
            [
                float(pyo.value(instance.total_power_input[t]))
                for t in instance.time_steps
            ],
            index=list(self.index),
            dtype=float,
        )

        self.da_component_state = self._extract_component_trajectory(instance)

        self._rebuild_market_reference_views()

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
            and results.solver.termination_condition == TerminationCondition.optimal
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
        series = self.da_component_state.get(component_name, {}).get(variable_name)
        return self._value_at(series, start, default)

    def _reference_component_net_power(
        self,
        component_name: str,
        start,
    ) -> float:
        power_in = self._reference_component_value(
            component_name,
            "power_in",
            start,
            0.0,
        )
        power_out = self._reference_component_value(
            component_name,
            "power_out",
            start,
            0.0,
        )

        return power_in - power_out

    @staticmethod
    def _snapshot_value(
        snapshot: dict[str, dict[str, float]],
        component_name: str,
        variable_name: str,
        default: float = 0.0,
    ) -> float:
        return float(snapshot.get(component_name, {}).get(variable_name, default))

    @staticmethod
    def _snapshot_component_net_power(
        snapshot: dict[str, dict[str, float]],
        component_name: str,
    ) -> float:
        """
        Return the component's net electrical grid exchange from a
        solved component snapshot.

        Positive values denote electricity consumption/import.
        Negative values denote electricity generation/export.
        """
        values = snapshot.get(component_name, {})

        power_in = float(values.get("power_in", 0.0))
        power_out = float(values.get("power_out", 0.0))

        return power_in - power_out

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
                    values[variable.local_name] = float(pyo.value(variable[target_t]))
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
            reference_soc = self.da_component_state.get(name, {}).get("soc")
            if reference_soc is None:
                continue

            block = instance.dsm_blocks[name]
            if not hasattr(block, "soc"):
                continue

            instance.market_terminal_soc.add(
                block.soc[final_t] == float(reference_soc.iloc[final_t])
            )

    def _rolling_reference_for_start(
        self,
        start,
    ) -> tuple[dict[str, Any], int]:
        """
        Return the rolling-window reference and local timestep for ``start``.

        Redispatch is evaluated against the rolling DA window that originally
        committed the requested timestep. This allows redispatch to use an older
        DA reference even after a newer rolling window has already been solved.

        A timestep that exists only in the provisional look-ahead part of a solved
        window is rejected until it has actually been committed.
        """
        global_t = self._index_position(start)

        reference = None
        references = getattr(self, "_rh_market_references", {})

        # First look for the window that actually committed this timestep.
        for stored_window_start in sorted(references, reverse=True):
            candidate = references[stored_window_start]

            candidate_start = int(candidate["window_start"])
            candidate_commit_end = int(candidate["commit_end"])

            if candidate_start <= global_t < candidate_commit_end:
                reference = candidate
                break

        # Backward-compatible fallback to the current/latest reference.
        if reference is None:
            candidate = getattr(self, "_rh_market_reference", None)

            if isinstance(candidate, dict):
                candidate_start = int(candidate["window_start"])
                candidate_commit_end = int(candidate["commit_end"])

                if candidate_start <= global_t < candidate_commit_end:
                    reference = candidate

        # No committed reference was found. Check whether the timestep is known
        # but still belongs only to a provisional look-ahead region.
        if reference is None:
            provisional_reference = None

            for stored_window_start in sorted(references, reverse=True):
                candidate = references[stored_window_start]

                candidate_start = int(candidate["window_start"])
                candidate_end = int(candidate["window_end"])
                candidate_commit_end = int(candidate["commit_end"])

                if (
                    candidate_start <= global_t < candidate_end
                    and global_t >= candidate_commit_end
                ):
                    provisional_reference = candidate
                    break

            if provisional_reference is None:
                candidate = getattr(self, "_rh_market_reference", None)

                if isinstance(candidate, dict):
                    candidate_start = int(candidate["window_start"])
                    candidate_end = int(candidate["window_end"])
                    candidate_commit_end = int(candidate["commit_end"])

                    if (
                        candidate_start <= global_t < candidate_end
                        and global_t >= candidate_commit_end
                    ):
                        provisional_reference = candidate

            if provisional_reference is not None:
                window_start = int(provisional_reference["window_start"])
                commit_end = int(provisional_reference["commit_end"])

                raise RuntimeError(
                    f"Redispatch timestamp {start} for '{self.id}' is not committed "
                    f"yet. Active committed range is "
                    f"[{window_start}:{commit_end})."
                )

            raise RuntimeError(
                f"No committed rolling-horizon market reference is available "
                f"for SectorCoupledSystem '{self.id}' at {start}."
            )

        window_start = int(reference["window_start"])
        window_end = int(reference["window_end"])
        commit_end = int(reference["commit_end"])

        # Defensive consistency check.
        if not (window_start <= global_t < window_end):
            raise RuntimeError(
                f"Redispatch timestamp {start} for '{self.id}' is outside "
                f"the selected rolling window [{window_start}:{window_end})."
            )

        if global_t >= commit_end:
            raise RuntimeError(
                f"Redispatch timestamp {start} for '{self.id}' is not committed "
                f"in the selected rolling window. Committed range is "
                f"[{window_start}:{commit_end})."
            )

        target_local = global_t - window_start

        return reference, target_local

    def _fix_rolling_past_to_reference(
        self, instance: pyo.ConcreteModel, target_local: int, reference: dict[str, Any]
    ) -> None:
        """
        Fix local timesteps before the RD target to the rolling DA trajectory.

        The rolling-window model already starts from the correct carried initial
        state.  Fixing only the local past preserves chronology while leaving the
        RD target and future look-ahead timesteps free to re-optimise.
        """
        if target_local <= 0:
            return

        component_state = reference["component_state"]

        for component_name, states in component_state.items():
            if component_name not in instance.dsm_blocks:
                continue

            block = instance.dsm_blocks[component_name]

            for variable_name, series in states.items():
                if not hasattr(block, variable_name):
                    continue

                variable = getattr(block, variable_name)
                if not variable.is_indexed():
                    continue

                for local_t in range(target_local):
                    if local_t not in variable:
                        continue

                    value = float(series.iloc[local_t])
                    if pd.isna(value):
                        continue

                    lb = variable[local_t].lb
                    ub = variable[local_t].ub

                    if lb is not None:
                        value = max(value, lb)
                    if ub is not None:
                        value = min(value, ub)

                    variable[local_t].fix(value)

        rolling_power = reference["power"]
        for local_t in range(target_local):
            value = float(rolling_power.iloc[local_t])
            if not pd.isna(value):
                instance.total_power_input[local_t].fix(value)

    def _add_rolling_reference_terminal_soc(
        self, instance: pyo.ConcreteModel, reference: dict[str, Any]
    ) -> None:
        """
        Preserve the DA storage value at the end of the current look-ahead window.

        RD may change the target hour and re-optimise subsequent hours, but it
        must return each storage to the SOC of the original rolling DA solution
        at the end of that same look-ahead window.
        """
        if not self.storage_components:
            return

        final_local = list(instance.time_steps)[-1]
        instance.market_terminal_soc = pyo.ConstraintList()

        for name in self.storage_components:
            reference_soc = reference["component_state"].get(name, {}).get("soc")
            if reference_soc is None or len(reference_soc) == 0:
                continue

            terminal_soc = float(reference_soc.iloc[-1])
            if pd.isna(terminal_soc):
                continue

            if name not in instance.dsm_blocks:
                continue

            block = instance.dsm_blocks[name]
            if not hasattr(block, "soc"):
                continue

            instance.market_terminal_soc.add(block.soc[final_local] == terminal_soc)

    def _solve_redispatch_extreme(
        self, start, sense: str
    ) -> tuple[float, dict[str, dict[str, float]]]:
        if sense not in {"min", "max"}:
            raise ValueError("sense must be either 'min' or 'max'.")

        self.prepare_market_state()

        cache = getattr(self, "_sector_redispatch_cache", {})

        if self.horizon_mode == "rolling_horizon":
            reference, target_t = self._rolling_reference_for_start(start)

            key = (
                start,
                sense,
                int(reference["window_start"]),
                int(reference["commit_end"]),
            )
            if key in cache:
                return cache[key]

            # Start from the exact Pyomo instance that generated the rolling DA
            # trajectory.  Clone it so the stored reference is never mutated by
            # redispatch-bound calculations.
            instance = reference["instance"].clone()

            self._fix_rolling_past_to_reference(instance, target_t, reference)
            self._add_rolling_reference_terminal_soc(instance, reference)

        else:
            key = (start, sense)
            if key in cache:
                return cache[key]

            target_t = self._index_position(start)

            instance = self.model.create_instance()
            instance = self.switch_to_opt(instance)

            self._fix_committed_past_to_reference(instance, target_t)
            self._add_reference_terminal_soc(instance)

        # The RD bound introduces its own objective.  Deactivate whichever
        # optimization objective is active in the cloned/fresh reference model.
        for objective in instance.component_objects(pyo.Objective, active=True):
            objective.deactivate()

        total_cost = pyo.quicksum(
            instance.variable_cost[t] for t in instance.time_steps
        )
        epsilon = 1e-9

        if sense == "min":
            expression = instance.total_power_input[target_t] + epsilon * total_cost
            objective_sense = pyo.minimize
        else:
            expression = instance.total_power_input[target_t] - epsilon * total_cost
            objective_sense = pyo.maximize

        instance.sector_market_bound_objective = pyo.Objective(
            expr=expression, sense=objective_sense
        )

        results = self.solver.solve(instance, options={}, tee=False)

        if not (
            results.solver.status == SolverStatus.ok
            and results.solver.termination_condition == TerminationCondition.optimal
        ):
            raise RuntimeError(
                f"{sense} redispatch-bound solve failed for "
                f"'{self.id}' at {start}. "
                f"status={results.solver.status}, "
                f"termination={results.solver.termination_condition}"
            )

        power = float(pyo.value(instance.total_power_input[target_t]))
        snapshot = self._extract_component_snapshot(instance, target_t)

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

        min_consumption, _ = self._solve_redispatch_extreme(start, "min")
        max_consumption, _ = self._solve_redispatch_extreme(start, "max")
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
            power = self._reference_component_net_power(
                component_name,
                start,
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

        summed_power = sum(item["physical_power"] for item in positions)
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

                base_net = self._reference_component_value(
                    name, discharge_var, start, 0.0
                ) - self._reference_component_value(name, charge_var, start, 0.0)
                min_net = self._snapshot_value(
                    min_snapshot, name, discharge_var, 0.0
                ) - self._snapshot_value(min_snapshot, name, charge_var, 0.0)
                max_net = self._snapshot_value(
                    max_snapshot, name, discharge_var, 0.0
                ) - self._snapshot_value(max_snapshot, name, charge_var, 0.0)

                up_weights[name] = max(0.0, min_net - base_net)
                down_weights[name] = max(0.0, base_net - max_net)
                continue

            # Prefer direct electrical changes whenever the component exchanges
            # electricity with the grid.
            base_power = self._reference_component_net_power(name, start)
            min_power = self._snapshot_component_net_power(min_snapshot, name)
            max_power = self._snapshot_component_net_power(max_snapshot, name)

            if max(abs(base_power), abs(min_power), abs(max_power)) > tolerance:
                # Upward redispatch means reducing net consumption or increasing
                # generation, i.e. moving towards a lower net electrical exchange
                up_weights[name] = max(0.0, base_power - min_power)

                # Downward redispatch means increasing net consumption or reducing
                # generation.
                down_weights[name] = max(0.0, max_power - base_power)
                continue

            # Components without direct electrical exchange can still influence
            # electrical flexibility indirectly through the coupled carrier balance
            output_var = self._component_output_variable(name)
            if output_var is None:
                continue

            base_output = self._reference_component_value(name, output_var, start, 0.0)
            min_output = self._snapshot_value(min_snapshot, name, output_var, 0.0)
            max_output = self._snapshot_value(max_snapshot, name, output_var, 0.0)

            up_weights[name] = max(0.0, min_output - base_output)
            down_weights[name] = max(0.0, base_output - max_output)

        return up_weights, down_weights

    @staticmethod
    def _allocate_total_by_weights(
        total: float,
        weights: dict[str, float],
        names: list[str],
        tolerance: float,
    ) -> dict[str, float]:
        allocation = {name: 0.0 for name in names}
        positive = {name: max(0.0, float(weights.get(name, 0.0))) for name in names}
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
        min_consumption, min_snapshot = self._solve_redispatch_extreme(start, "min")
        max_consumption, max_snapshot = self._solve_redispatch_extreme(start, "max")

        up_total = max(0.0, base_consumption - min_consumption)
        down_total = max(0.0, max_consumption - base_consumption)

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
        if up_total > tolerance and sum(up_allocation.values()) <= tolerance:
            direct_up = {}
            for name in names:
                base_power = self._reference_component_net_power(name, start)
                min_power = self._snapshot_component_net_power(min_snapshot, name)
                direct_up[name] = max(0.0, base_power - min_power)
            up_allocation = self._allocate_total_by_weights(
                up_total,
                direct_up,
                names,
                tolerance,
            )

        if down_total > tolerance and sum(down_allocation.values()) <= tolerance:
            direct_down = {}
            for name in names:
                base_power = self._reference_component_net_power(name, start)
                max_power = self._snapshot_component_net_power(max_snapshot, name)
                direct_down[name] = max(0.0, max_power - base_power)
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
                base_power = self._reference_component_net_power(name, start)

                if abs(base_power) > tolerance:
                    allocation[name] += residual
                    return

            if names:
                allocation[names[0]] += residual

        add_residual(up_total, up_allocation)
        add_residual(down_total, down_allocation)

        records = []

        for name in names:
            baseline_power = self._reference_component_net_power(
                name,
                start,
            )
            signed_volume = -baseline_power

            up_cap = max(0.0, up_allocation.get(name, 0.0))
            down_cap = max(0.0, down_allocation.get(name, 0.0))

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

        child_baseline = sum(item["volume"] for item in records)
        child_up = sum(item["max_power"] - item["volume"] for item in records)
        child_down = sum(item["volume"] - item["min_power"] for item in records)

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
        min_consumption, max_consumption = self.get_redispatch_power_bounds(start)

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
