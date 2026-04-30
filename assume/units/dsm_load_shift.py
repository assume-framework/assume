# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy
import logging
from collections.abc import Callable
from datetime import datetime

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import (
    SolverFactory,
    SolverStatus,
    TerminationCondition,
)

from assume.common.fast_pandas import FastIndex, FastSeries
from assume.common.utils import get_supported_solver_pyomo
from assume.units.dst_components import demand_side_technologies

logger = logging.getLogger(__name__)


class DSMFlex:
    # Mapping of flexibility measures to their respective functions
    flexibility_map: dict[str, Callable[[pyo.ConcreteModel], None]] = {
        "electricity_price_signal": lambda self, model: self.electricity_price_signal(
            model
        ),
        "cost_based_load_shift": lambda self, model: self.cost_based_flexibility(model),
        "congestion_management_flexibility": lambda self,
        model: self.grid_congestion_management(model),
        "symmetric_flexible_block": lambda self, model: self.symmetric_flexible_block(
            model
        ),
        "peak_load_shifting": lambda self, model: self.peak_load_shifting_flexibility(
            model
        ),
        "renewable_utilisation": lambda self, model: self.renewable_utilisation(
            model,
        ),
    }
    big_M = 10000000

    # Rolling-horizon extensibility hooks (override in subclasses)
    # Set _demand_attr_suffix to the instance attribute that holds total demand
    # (e.g. "steel_demand" → self.steel_demand).  None means no demand tracking.
    _demand_attr_suffix: str | None = None
    # Per-block extraction schema used by _extract_component_operations.
    # Format: {block_name: (pwr_attr, out_attr, pwr_col_name, out_col_name)}
    _component_schema: dict = {}
    # Additional time-series attributes (full-horizon length) that must be
    # saved and sliced to the window during _solve_rolling_horizon_next_window.
    _extra_price_attrs: list = []

    def __init__(self, components, **kwargs):
        # Extract rolling-horizon optimisation config before passing **kwargs up the MRO.
        dsm_opt = kwargs.pop("dsm_optimisation_config", None)
        if not isinstance(dsm_opt, dict):
            dsm_opt = {}

        super().__init__(**kwargs)

        self.components = components
        self.solver = SolverFactory(get_supported_solver_pyomo())

        # Rolling-horizon settings (populated from config; default is full horizon)
        self.horizon_mode = dsm_opt.get("horizon_mode", "full_horizon")
        self._rh_look_ahead = dsm_opt.get("look_ahead_horizon")  # e.g. "72h"
        self._rh_commit = dsm_opt.get("commit_horizon")  # e.g. "24h"
        self._rh_step = dsm_opt.get("rolling_step")  # e.g. "24h"

        # Rolling-horizon state tracking (for per-market-round re-optimization)
        self._rh_window_start_idx = 0  # Current window start index in full horizon
        self._rh_last_market_request_step = (
            None  # Track which step the last market request was for
        )
        self._rh_optimized_until_step = (
            0  # How far we've optimized (in full horizon steps)
        )

        if self.horizon_mode == "rolling_horizon":
            if not all([self._rh_look_ahead, self._rh_commit, self._rh_step]):
                raise ValueError(
                    "Rolling horizon mode requires look_ahead_horizon, "
                    "commit_horizon, and rolling_step to be specified."
                )

    def initialize_components(self):
        """
        Initializes the DSM components by creating and adding blocks to the model.

        This method iterates over the provided components, instantiates their corresponding classes,
        and adds the respective blocks to the Pyomo model.

        Args:
            components (dict[str, dict]): A dictionary where each key is a technology name and
                                        the value is a dictionary of parameters for the respective technology.
                                        Each technology is mapped to a corresponding class in `demand_side_technologies`.

        The method:
        - Looks up the corresponding class for each technology in `demand_side_technologies`.
        - Instantiates the class by passing the required parameters.
        - Adds the resulting block to the model under the `dsm_blocks` attribute.
        """
        components = self.components.copy()
        self.model.dsm_blocks = pyo.Block(list(components.keys()))

        for technology, component_data in components.items():
            if technology in demand_side_technologies:
                # Get the class from the dictionary mapping (adjust `demand_side_technologies` to hold classes)
                component_class = demand_side_technologies[technology]

                # Instantiate the component with the required parameters (unpack the component_data dictionary)
                component_instance = component_class(
                    time_steps=self.model.time_steps, **component_data
                )
                # Add the component to the components dictionary
                self.components[technology] = component_instance

                # Add the component's block to the model
                component_instance.add_to_model(
                    self.model, self.model.dsm_blocks[technology]
                )

    def setup_model(self, presolve=True):
        # Initialize the Pyomo model
        # along with optimal and flexibility constraints
        # and the objective functions

        self.optimisation_counter = 0

        # Snapshot the component *dicts* (including any time-series injected by the
        # subclass __init__, e.g. pv_profile) before initialize_components() replaces
        # them with Pyomo component instances.  This snapshot is used by the rolling-
        # horizon solver to rebuild a fresh model for each look-ahead window.
        self._orig_components_dict = copy.deepcopy(self.components)

        self.model = pyo.ConcreteModel()
        self.define_sets()
        self.define_parameters()
        self.define_variables()

        self.initialize_components()
        self.initialize_process_sequence()

        self.define_constraints()
        self.define_objective_opt()

        # Solve the model to determine the optimal operation without flexibility
        # and store the results to be used in the flexibility mode later
        if presolve:
            self.determine_optimal_operation_without_flex(switch_flex_off=False)

        # Modify the model to include the flexibility measure constraints
        # as well as add a new objective function to the model
        # to maximize the flexibility measure
        if self.flexibility_measure in DSMFlex.flexibility_map:
            DSMFlex.flexibility_map[self.flexibility_measure](self, self.model)
        else:
            raise ValueError(f"Unknown flexibility measure: {self.flexibility_measure}")

    def define_sets(self) -> None:
        """
        Defines the sets for the Pyomo model.
        """
        self.model.time_steps = pyo.Set(
            initialize=[idx for idx, _ in enumerate(self.index)]
        )

    def define_objective_opt(self):
        """
        Defines the objective for the optimization model.

        Args:
            model (pyomo.ConcreteModel): The Pyomo model.
        """
        if self.objective == "min_variable_cost":

            @self.model.Objective(sense=pyo.minimize)
            def obj_rule_opt(m):
                """
                Minimizes the total variable cost over all time steps.
                """
                total_variable_cost = pyo.quicksum(
                    self.model.variable_cost[t] for t in self.model.time_steps
                )

                return total_variable_cost

        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    def electricity_price_signal(self, model):
        """
        Determine the optimal operation using a new electricity price signal.
        """
        # Delete the existing electricity_price component
        model.del_component(model.electricity_price)

        # Add the updated electricity_price component explicitly
        model.add_component(
            "electricity_price",
            pyo.Param(
                model.time_steps,
                initialize={
                    t: value
                    for t, value in enumerate(self.forecaster.electricity_price_flex)
                },
            ),
        )

        @self.model.Objective(sense=pyo.minimize)
        def obj_rule_flex(m):
            """
            Maximizes the load shift over all time steps.
            """
            total_variable_cost = pyo.quicksum(
                self.model.variable_cost[t] for t in self.model.time_steps
            )

            return total_variable_cost

    def cost_based_flexibility(self, model):
        """
        Modify the optimization model to include constraints for flexibility within cost tolerance.
        """

        model.cost_tolerance = pyo.Param(initialize=(self.cost_tolerance))
        model.total_cost = pyo.Param(initialize=0.0, mutable=True)

        # Variables
        model.load_shift_pos = pyo.Var(model.time_steps, within=pyo.NonNegativeIntegers)
        model.load_shift_neg = pyo.Var(model.time_steps, within=pyo.NonNegativeIntegers)
        model.shift_indicator = pyo.Var(model.time_steps, within=pyo.Binary)

        @model.Constraint()
        def total_cost_upper_limit(m):
            return pyo.quicksum(
                m.variable_cost[t] for t in m.time_steps
            ) <= m.total_cost * (1 + (m.cost_tolerance / 100))

        @model.Constraint(model.time_steps)
        def flex_constraint_upper(m, t):
            return m.load_shift_pos[t] <= (1 - m.shift_indicator[t]) * self.big_M

        @model.Constraint(model.time_steps)
        def flex_constraint_lower(m, t):
            return m.load_shift_neg[t] <= m.shift_indicator[t] * self.big_M

        @model.Constraint(model.time_steps)
        def total_power_input_constraint_with_flex(m, t):
            # Apply constraints based on the technology type
            if self.technology == "hydrogen_plant":
                # Hydrogen plant constraint
                return (
                    m.total_power_input[t] + m.load_shift_pos[t] - m.load_shift_neg[t]
                    == self.model.dsm_blocks["electrolyser"].power_in[t]
                )
            elif self.technology == "steel_plant":
                # Steel plant constraint with conditional electrolyser inclusion
                if self.has_electrolyser:
                    return (
                        m.total_power_input[t]
                        + m.load_shift_pos[t]
                        - m.load_shift_neg[t]
                        == self.model.dsm_blocks["electrolyser"].power_in[t]
                        + self.model.dsm_blocks["eaf"].power_in[t]
                        + self.model.dsm_blocks["dri_plant"].power_in[t]
                    )
                else:
                    return (
                        m.total_power_input[t]
                        + m.load_shift_pos[t]
                        - m.load_shift_neg[t]
                        == self.model.dsm_blocks["eaf"].power_in[t]
                        + self.model.dsm_blocks["dri_plant"].power_in[t]
                    )

            elif self.technology == "building":
                total_power_input = m.inflex_demand[t]
                if self.has_heatpump:
                    total_power_input += self.model.dsm_blocks["heat_pump"].power_in[t]
                if self.has_boiler:
                    total_power_input += self.model.dsm_blocks["boiler"].power_in[t]
                if self.has_ev:
                    total_power_input += (
                        self.model.dsm_blocks["electric_vehicle"].charge[t]
                        - self.model.dsm_blocks["electric_vehicle"].discharge[t]
                    )
                if self.has_battery_storage:
                    total_power_input += (
                        self.model.dsm_blocks["generic_storage"].charge[t]
                        - self.model.dsm_blocks["generic_storage"].discharge[t]
                    )
                if self.has_pv:
                    total_power_input -= self.model.dsm_blocks["pv_plant"].power[t]

                return (
                    m.total_power_input[t] + m.load_shift_pos[t] - m.load_shift_neg[t]
                    == total_power_input
                )

            elif self.technology == "steam_generator_plant":
                total_power = 0
                if self.has_heatpump:
                    total_power += self.model.dsm_blocks["heat_pump"].power_in[t]
                if self.has_boiler:
                    boiler = self.components["boiler"]
                    if boiler.fuel_type == "electricity":
                        total_power += self.model.dsm_blocks["boiler"].power_in[t]
                return (
                    m.total_power_input[t] + m.load_shift_pos[t] - m.load_shift_neg[t]
                    == total_power
                )

        @self.model.Objective(sense=pyo.maximize)
        def obj_rule_flex(m):
            """
            Maximizes the load shift over all time steps.
            """

            maximise_load_shift = pyo.quicksum(
                m.load_shift_pos[t] for t in m.time_steps
            )

            return maximise_load_shift

    def symmetric_flexible_block(self, model):
        """
        Cost-based CRM flexibility (rolling 4-hour blocks) for steam_generator_plant.
        Optimizes a feasible operational profile (sum of tech variables), calculates
        up/down 4-hour CRM blocks, and enforces a total cost limit.
        Also stores static plant-wide max and min capacity as attributes for later use.
        """
        block_length = 4
        min_bid_MW = 1
        time_steps = list(sorted(model.time_steps))
        T = len(time_steps)
        possible_starts = [time_steps[i] for i in range(T - block_length + 1)]

        # ---- STATIC PLANT-WIDE MAX/MIN CAPACITY ----
        max_plant_capacity = 0
        min_plant_capacity = 0
        if self.has_heatpump:
            max_plant_capacity += model.dsm_blocks["heat_pump"].max_power
            min_plant_capacity += model.dsm_blocks["heat_pump"].min_power
        if self.has_boiler:
            boiler = self.components["boiler"]
            if boiler.fuel_type == "electricity":
                max_plant_capacity += model.dsm_blocks["boiler"].max_power
                min_plant_capacity += model.dsm_blocks["boiler"].min_power

        # Save as attributes on the unit for use in bidding strategy, etc.
        self.max_plant_capacity = max_plant_capacity
        self.min_plant_capacity = min_plant_capacity

        # ---- FLEXIBILITY BLOCK VARIABLES ----
        model.block_up = pyo.Var(possible_starts, within=pyo.NonNegativeReals)
        model.block_down = pyo.Var(possible_starts, within=pyo.NonNegativeReals)
        model.block_is_bid_up = pyo.Var(possible_starts, within=pyo.Binary)
        model.block_is_bid_down = pyo.Var(possible_starts, within=pyo.Binary)

        # ConstraintLists to hold per-block constraints
        model.block_up_window = pyo.ConstraintList()
        model.block_down_window = pyo.ConstraintList()

        for t in possible_starts:
            for offset in range(block_length):
                tau = time_steps[time_steps.index(t) + offset]
                total_power = 0
                if self.has_heatpump:
                    total_power += model.dsm_blocks["heat_pump"].power_in[tau]
                if self.has_boiler:
                    boiler = self.components["boiler"]
                    if boiler.fuel_type == "electricity":
                        total_power += model.dsm_blocks["boiler"].power_in[tau]
                # Use the stored plant-wide capacities here if you want,
                # or (if you allow for time-varying max/min in the future) keep the local max/min per tau

                # Block up/down window constraints (use static min/max plant capacity here)
                model.block_up_window.add(
                    model.block_up[t] <= self.max_plant_capacity - total_power
                )
                model.block_down_window.add(
                    model.block_down[t] <= total_power - self.min_plant_capacity
                )

        @model.Constraint(possible_starts)
        def block_bid_logic_up(m, t):
            return m.block_up[t] >= min_bid_MW * m.block_is_bid_up[t]

        @model.Constraint(possible_starts)
        def block_bid_logic_down(m, t):
            return m.block_down[t] >= min_bid_MW * m.block_is_bid_down[t]

        @model.Constraint(possible_starts)
        def symmetric_block(m, t):
            return m.block_is_bid_up[t] + m.block_is_bid_down[t] <= 1

        # ---- COST TOLERANCE ----
        model.cost_tolerance = pyo.Param(initialize=self.cost_tolerance)
        model.total_cost = pyo.Param(initialize=0.0, mutable=True)

        @model.Constraint()
        def total_cost_upper_limit(m):
            return pyo.quicksum(
                m.variable_cost[t] for t in m.time_steps
            ) <= m.total_cost * (1 + (m.cost_tolerance / 100))

        # ---- OBJECTIVE: MAXIMIZE CRM UPWARD FLEXIBILITY ----
        @model.Objective(sense=pyo.maximize)
        def obj_rule_flex(m):
            return sum(m.block_up[t] for t in possible_starts)

    def grid_congestion_management(self, model):
        """
        Adjust load shifting based directly on grid congestion signals to enable
        congestion-responsive flexibility.

        Args:
            model (pyomo.ConcreteModel): The Pyomo model being optimized.
        """

        # Generate the congestion indicator dictionary based on the threshold
        congestion_indicator_dict = {
            i: int(value > self.congestion_threshold)
            for i, value in enumerate(self.congestion_signal)
        }

        # Define the cost tolerance parameter
        model.cost_tolerance = pyo.Param(initialize=(self.cost_tolerance))
        model.total_cost = pyo.Param(initialize=0.0, mutable=True)
        # Define congestion_indicator as a fixed parameter with matching indices
        model.congestion_indicator = pyo.Param(
            model.time_steps, initialize=congestion_indicator_dict
        )

        # Variables
        model.load_shift_pos = pyo.Var(model.time_steps, within=pyo.NonNegativeIntegers)
        model.load_shift_neg = pyo.Var(model.time_steps, within=pyo.NonNegativeIntegers)
        model.shift_indicator = pyo.Var(model.time_steps, within=pyo.Binary)

        # Constraint to manage total cost upper limit with cost tolerance
        @model.Constraint()
        def total_cost_upper_limit(m):
            return pyo.quicksum(
                m.variable_cost[t] for t in m.time_steps
            ) <= m.total_cost * (1 + (m.cost_tolerance / 100))

        @model.Constraint(model.time_steps)
        def flex_constraint_upper(m, t):
            return m.load_shift_pos[t] <= (1 - m.shift_indicator[t]) * self.big_M

        @model.Constraint(model.time_steps)
        def flex_constraint_lower(m, t):
            return m.load_shift_neg[t] <= m.shift_indicator[t] * self.big_M

        # Power input constraint with flexibility based on congestion
        @model.Constraint(model.time_steps)
        def total_power_input_constraint_with_flex(m, t):
            if self.technology == "steel_plant":
                if self.has_electrolyser:
                    return (
                        m.total_power_input[t]
                        + m.load_shift_pos[t]
                        - m.load_shift_neg[t]
                        == self.model.dsm_blocks["electrolyser"].power_in[t]
                        + self.model.dsm_blocks["eaf"].power_in[t]
                        + self.model.dsm_blocks["dri_plant"].power_in[t]
                    )
                else:
                    return (
                        m.total_power_input[t]
                        + m.load_shift_pos[t]
                        - m.load_shift_neg[t]
                        == self.model.dsm_blocks["eaf"].power_in[t]
                        + self.model.dsm_blocks["dri_plant"].power_in[t]
                    )

            elif self.technology == "hydrogen_plant":
                # Hydrogen plant constraint
                return (
                    m.total_power_input[t] + m.load_shift_pos[t] - m.load_shift_neg[t]
                    == self.model.dsm_blocks["electrolyser"].power_in[t]
                )
            elif self.technology == "steam_generator_plant":
                total_power = 0
                if self.has_heatpump:
                    total_power += self.model.dsm_blocks["heat_pump"].power_in[t]
                if self.has_boiler:
                    boiler = self.components["boiler"]
                    if boiler.fuel_type == "electricity":
                        total_power += self.model.dsm_blocks["boiler"].power_in[t]
                return (
                    m.total_power_input[t] + m.load_shift_pos[t] - m.load_shift_neg[t]
                    == total_power
                )

        @self.model.Objective(sense=pyo.maximize)
        def obj_rule_flex(m):
            """
            Maximizes the load shift over all time steps.
            """
            maximise_load_shift = pyo.quicksum(
                m.load_shift_neg[t] * m.congestion_indicator[t] for t in m.time_steps
            )

            return maximise_load_shift

    def peak_load_shifting_flexibility(self, model):
        """
        Implements constraints for peak load shifting flexibility by identifying peak periods
        and allowing load shifts from peak to off-peak periods within a cost tolerance.

        Args:
            model (pyomo.ConcreteModel): The Pyomo model being optimized.
        """

        max_load = max(self.opt_power_requirement)

        peak_load_cap_value = max_load * (
            self.peak_load_cap / 100
        )  # E.g., 10% threshold
        # Add peak_threshold_value as a Param on the model so it can be accessed elsewhere
        model.peak_load_cap_value = pyo.Param(initialize=peak_load_cap_value)

        # Parameters
        model.cost_tolerance = pyo.Param(initialize=self.cost_tolerance)
        model.total_cost = pyo.Param(initialize=0.0, mutable=True)

        # Variables for load shifting
        model.load_shift_pos = pyo.Var(model.time_steps, within=pyo.NonNegativeIntegers)
        model.load_shift_neg = pyo.Var(model.time_steps, within=pyo.NonNegativeIntegers)
        model.shift_indicator = pyo.Var(model.time_steps, within=pyo.Binary)

        peak_periods = {
            t
            for t in model.time_steps
            if self.opt_power_requirement.iloc[t] > peak_load_cap_value
        }
        model.peak_indicator = pyo.Param(
            model.time_steps,
            initialize={t: int(t in peak_periods) for t in model.time_steps},
        )

        @model.Constraint()
        def total_cost_upper_limit(m):
            return pyo.quicksum(
                m.variable_cost[t] for t in m.time_steps
            ) <= m.total_cost * (1 + (m.cost_tolerance / 100))

        @model.Constraint(model.time_steps)
        def flex_constraint_upper(m, t):
            return m.load_shift_pos[t] <= (1 - m.shift_indicator[t]) * self.big_M

        @model.Constraint(model.time_steps)
        def flex_constraint_lower(m, t):
            return m.load_shift_neg[t] <= m.shift_indicator[t] * self.big_M

        # Power input constraint with flexibility based on congestion
        @model.Constraint(model.time_steps)
        def total_power_input_constraint_with_peak_shift(m, t):
            if self.has_electrolyser:
                return (
                    m.total_power_input[t] + m.load_shift_pos[t] - m.load_shift_neg[t]
                    == m.dsm_blocks["electrolyser"].power_in[t]
                    + m.dsm_blocks["eaf"].power_in[t]
                    + m.dsm_blocks["dri_plant"].power_in[t]
                )
            else:
                return (
                    m.total_power_input[t] + m.load_shift_pos[t] - m.load_shift_neg[t]
                    == m.dsm_blocks["eaf"].power_in[t]
                    + m.dsm_blocks["dri_plant"].power_in[t]
                )

        @model.Constraint(model.time_steps)
        def peak_threshold_constraint(m, t):
            """
            Ensures that the power input during peak periods does not exceed the peak threshold value.
            """
            if self.has_electrolyser:
                return (
                    m.dsm_blocks["electrolyser"].power_in[t]
                    + m.dsm_blocks["eaf"].power_in[t]
                    + m.dsm_blocks["dri_plant"].power_in[t]
                    <= peak_load_cap_value
                )
            else:
                return (
                    m.dsm_blocks["eaf"].power_in[t]
                    + m.dsm_blocks["dri_plant"].power_in[t]
                    <= peak_load_cap_value
                )

        @self.model.Objective(sense=pyo.maximize)
        def obj_rule_flex(m):
            """
            Maximizes the load shift over all time steps.
            """
            maximise_load_shift = pyo.quicksum(
                m.load_shift_neg[t] * m.peak_indicator[t] for t in m.time_steps
            )

            return maximise_load_shift

    def renewable_utilisation(self, model):
        """
        Implements flexibility based on the renewable utilisation signal. The normalized renewable intensity
        signal indicates the periods with high renewable availability, allowing the steel plant to adjust
        its load flexibly in response.

        Args:
            model (pyomo.ConcreteModel): The Pyomo model being optimized.
        """
        # Normalize renewable utilization signal between 0 and 1
        renewable_signal_normalised = (
            self.renewable_utilisation_signal - self.renewable_utilisation_signal.min()
        ) / (
            self.renewable_utilisation_signal.max()
            - self.renewable_utilisation_signal.min()
        )
        # Add normalized renewable signal as a model parameter
        model.renewable_signal = pyo.Param(
            model.time_steps,
            initialize={
                t: renewable_signal_normalised.iloc[t] for t in model.time_steps
            },
        )

        model.cost_tolerance = pyo.Param(initialize=self.cost_tolerance)
        model.total_cost = pyo.Param(initialize=0.0, mutable=True)

        # Variables for load flexibility based on renewable intensity
        model.load_shift_pos = pyo.Var(model.time_steps, within=pyo.NonNegativeIntegers)
        model.load_shift_neg = pyo.Var(model.time_steps, within=pyo.NonNegativeIntegers)
        model.shift_indicator = pyo.Var(model.time_steps, within=pyo.Binary)

        # Constraint to manage total cost upper limit with cost tolerance
        @model.Constraint()
        def total_cost_upper_limit(m):
            return pyo.quicksum(
                m.variable_cost[t] for t in m.time_steps
            ) <= m.total_cost * (1 + (m.cost_tolerance / 100))

        @model.Constraint(model.time_steps)
        def flex_constraint_upper(m, t):
            return m.load_shift_pos[t] <= (1 - m.shift_indicator[t]) * self.big_M

        @model.Constraint(model.time_steps)
        def flex_constraint_lower(m, t):
            return m.load_shift_neg[t] <= m.shift_indicator[t] * self.big_M

        # Power input constraint integrating flexibility
        @model.Constraint(model.time_steps)
        def total_power_input_constraint_flex(m, t):
            if self.has_electrolyser:
                return (
                    m.total_power_input[t] + m.load_shift_pos[t] - m.load_shift_neg[t]
                    == m.dsm_blocks["electrolyser"].power_in[t]
                    + m.dsm_blocks["eaf"].power_in[t]
                    + m.dsm_blocks["dri_plant"].power_in[t]
                )
            else:
                return (
                    m.total_power_input[t] + m.load_shift_pos[t] - m.load_shift_neg[t]
                    == m.dsm_blocks["eaf"].power_in[t]
                    + m.dsm_blocks["dri_plant"].power_in[t]
                )

        @self.model.Objective(sense=pyo.maximize)
        def obj_rule_flex(m):
            """
            Maximizes the load increase over all time steps based on renewable surplus.
            """
            maximise_renewable_utilisation = pyo.quicksum(
                m.load_shift_pos[t] * m.renewable_signal[t] for t in m.time_steps
            )

            return maximise_renewable_utilisation

    # ------------------------------------------------------------------
    # Rolling-horizon helpers
    # ------------------------------------------------------------------

    def _parse_duration_to_steps(self, duration_str: str) -> int:
        """Convert a duration string such as '24h' or '72h' to an integer
        number of time steps, based on the current ``self.index.freq``."""
        duration_seconds = pd.to_timedelta(duration_str).total_seconds()
        step_seconds = self.index.freq.total_seconds()
        return int(duration_seconds / step_seconds)

    def _collect_series_attrs_for_window(
        self, window_start: int, window_end: int, full_len: int
    ) -> dict:
        """Replace every :class:`FastSeries` attribute whose length matches the
        full simulation horizon with the corresponding numpy slice for the
        current window.

        Returns a dict of ``{attr_name: original_FastSeries}`` that is used by
        :meth:`_restore_series_attrs` to undo the substitution afterwards.
        """
        saved: dict = {}
        for attr_name, val in list(self.__dict__.items()):
            if isinstance(val, FastSeries) and len(val) == full_len:
                saved[attr_name] = val
                # Replace with a plain numpy slice; enumerate() works on ndarrays
                # so define_parameters() calls enumerate(self.attr) still work.
                setattr(self, attr_name, val.data[window_start:window_end])
        return saved

    def _restore_series_attrs(self, saved_attrs: dict) -> None:
        """Restore unit-level attributes previously saved by
        :meth:`_collect_series_attrs_for_window`."""
        for attr_name, val in saved_attrs.items():
            setattr(self, attr_name, val)

    def _deep_slice_component_data(
        self,
        comp_data,
        window_start: int,
        window_end: int,
        full_len: int,
    ):
        """Recursively copy *comp_data*, replacing any :class:`FastSeries` values
        of length *full_len* with a new :class:`FastSeries` aligned to the current
        window index (``self.index`` must already be the window :class:`FastIndex`
        at the time this is called).

        Dicts are recursed into; all other types are returned unchanged.
        """
        if isinstance(comp_data, dict):
            return {
                k: self._deep_slice_component_data(
                    v, window_start, window_end, full_len
                )
                for k, v in comp_data.items()
            }
        if isinstance(comp_data, FastSeries) and len(comp_data) == full_len:
            window_data = comp_data.data[window_start:window_end]
            return FastSeries(index=self.index, value=window_data)
        return comp_data

    def _prepare_window_components(
        self,
        window_start: int,
        window_end: int,
        full_len: int,
        init_states: dict,
        remaining_demand: float = None,
    ) -> dict:
        """Build a fresh dict-of-dicts for the components in the current window.

        Concretely this:
        * deep-copies ``self._orig_components_dict`` (preserving dict structure),
        * slices any nested :class:`FastSeries` to the window length,
        * patches ``initial_soc`` for stateful storage components using the
          state carried over from the previous committed period,
        * patches operational_status, start_up, shut_down states for generation units,
        * patches remaining_demand for steel plants (cumulative demand tracking).

        ``self.index`` must already be the window :class:`FastIndex` at call time.
        """
        window_comps: dict = {}
        for tech_name, comp_data in self._orig_components_dict.items():
            if isinstance(comp_data, dict):
                sliced = self._deep_slice_component_data(
                    comp_data, window_start, window_end, full_len
                )
                # Patch carried-over state variables for this technology
                if tech_name in init_states and isinstance(
                    init_states[tech_name], dict
                ):
                    state_dict = init_states[tech_name]
                    # Patch SoC if present (storage components)
                    if "soc" in state_dict:
                        sliced["initial_soc"] = state_dict["soc"]
                    # Patch operational status if present (generation/load units)
                    if "operational_status" in state_dict:
                        sliced["initial_operational_status"] = state_dict[
                            "operational_status"
                        ]
                    # Note: start_up and shut_down are derived from operational_status transitions,
                    # so they don't need to be explicitly patched
                window_comps[tech_name] = sliced
            else:
                window_comps[tech_name] = comp_data

        # Patch remaining_demand for units that track cumulative production
        if remaining_demand is not None and self._has_demand_tracking:
            # Find the first dict-type component block to carry the value in
            for _blk, _blk_data in window_comps.items():
                if isinstance(_blk_data, dict):
                    window_comps["_remaining_demand"] = remaining_demand
                    break

        return window_comps

    # ------------------------------------------------------------------
    # Rolling-horizon shared helpers
    # ------------------------------------------------------------------

    @property
    def _has_demand_tracking(self) -> bool:
        """True when this unit tracks a cumulative production demand across windows."""
        return self._demand_attr_suffix is not None

    def _primary_output_expr(self, m, t):
        """Return the Pyomo expression for the unit's primary production output at step *t*.

        Override in subclasses whose rolling-horizon demand tracking is based on
        a physical output other than total electrical power input
        (e.g. SteelPlant tracks tonnes of steel rather than MWh consumed).
        """
        return m.total_power_input[t]

    def _collect_init_states(self) -> dict:
        """Collect initial states (SoC, operational_status) from the original component dicts."""
        init_states: dict = {}
        for tech, data in self._orig_components_dict.items():
            if isinstance(data, dict):
                state: dict = {}
                if "initial_soc" in data:
                    state["soc"] = data["initial_soc"]
                if "initial_operational_status" in data:
                    state["operational_status"] = data["initial_operational_status"]
                if state:
                    init_states[tech] = state
        return init_states

    def _update_init_states(
        self, instance, commit_local: int, init_states: dict
    ) -> None:
        """Update *init_states* in-place with end-of-commit values from the solved instance."""
        for tech_name in list(init_states.keys()):
            try:
                block = instance.dsm_blocks[tech_name]
                state: dict = {}
                if hasattr(block, "soc"):
                    state["soc"] = pyo.value(block.soc[commit_local])
                if hasattr(block, "operational_status"):
                    state["operational_status"] = int(
                        round(pyo.value(block.operational_status[commit_local]))
                    )
                if state:
                    init_states[tech_name] = state
            except (KeyError, AttributeError):
                pass

    def _detect_operation_strategy(self, N: int, saved_attrs: dict) -> tuple:
        """Detect the operation strategy for the current unit.

        Checks for unit-ID-prefixed attributes in precedence order:
        *profile_guided* → *min_demand* → *cost_optimized* (default).

        Returns ``(strategy, full_horizon_load_profile, full_horizon_min_demand)``.
        Each non-``None`` sequence is padded/truncated to exactly *N* steps and
        its original value is preserved in *saved_attrs* for later restoration.

        When ``_demand_attr_suffix`` is ``None`` (no demand tracking) the method
        always returns ``("cost_optimized", None, None)``.
        """
        if not self._has_demand_tracking:
            return "cost_optimized", None, None

        unit_id = str(getattr(self, "id", None))

        def _fit_to_horizon(attr_name: str, pad_value=None):
            if not (attr_name and hasattr(self, attr_name)):
                return None
            val = getattr(self, attr_name)
            if val is None:
                return None
            try:
                lst = list(val)
            except TypeError:
                return None
            if not lst:
                return None
            if len(lst) < N:
                lst.extend(
                    [lst[-1] if pad_value is None else pad_value] * (N - len(lst))
                )
            else:
                lst = lst[:N]
            if attr_name not in saved_attrs:
                saved_attrs[attr_name] = val
            return lst

        profile = _fit_to_horizon(
            f"{unit_id}_normalized_load_profile" if unit_id else None
        )
        if profile is not None:
            return "profile_guided", profile, None

        demand = _fit_to_horizon(
            f"{unit_id}_{self._demand_attr_suffix}"
            if (unit_id and self._demand_attr_suffix)
            else None,
            pad_value=0.0,
        )
        if demand is not None:
            return "min_demand", None, demand

        return "cost_optimized", None, None

    def _add_window_demand_constraints(
        self,
        strategy: str,
        remaining_demand: float,
        full_horizon_load_profile,
        full_horizon_min_demand,
        window_start: int,
        commit_end: int,
    ) -> None:
        """Add demand-related constraints to the current window model.

        Uses ``_primary_output_expr`` so subclasses can track any physical output
        (e.g. steel tonnes, hydrogen kg) rather than just electrical energy.
        """
        # Upper-bound on total window production (skipped for min_demand strategy)
        if strategy != "min_demand":
            self.model.window_demand_limit = pyo.Param(
                initialize=remaining_demand, mutable=True
            )
            self.model.window_demand_con = pyo.Constraint(
                rule=lambda m: sum(
                    self._primary_output_expr(m, t) for t in m.time_steps
                )
                <= m.window_demand_limit
            )

        if (
            strategy == "profile_guided"
            and full_horizon_load_profile is not None
            and remaining_demand > 0
        ):
            remaining_profile = (
                list(full_horizon_load_profile[window_start:])
                if hasattr(full_horizon_load_profile, "__getitem__")
                else list(full_horizon_load_profile)[window_start:]
            )
            commit_profile = (
                list(full_horizon_load_profile[window_start:commit_end])
                if hasattr(full_horizon_load_profile, "__getitem__")
                else list(full_horizon_load_profile)[window_start:commit_end]
            )
            remaining_sum = sum(remaining_profile) if remaining_profile else 1.0
            commit_sum = sum(commit_profile) if commit_profile else 0.0
            if remaining_sum > 0:
                fraction = commit_sum / remaining_sum
                min_commit = (
                    remaining_demand * fraction * (1.0 - self.load_profile_deviation)
                )
                n_commit = commit_end - window_start
                self.model.min_commit_production = pyo.Param(
                    initialize=min_commit, mutable=True
                )
                self.model.window_min_commit_con = pyo.Constraint(
                    rule=lambda m: sum(
                        self._primary_output_expr(m, t) for t in range(n_commit)
                    )
                    >= m.min_commit_production
                )
                logger.info(
                    "[RH-STRATEGY] profile_guided: min_commit=%.1f MWh over %d steps",
                    min_commit,
                    n_commit,
                )

        elif strategy == "min_demand" and full_horizon_min_demand is not None:
            n_steps = commit_end - window_start
            min_window = (
                list(full_horizon_min_demand[window_start:commit_end])
                if hasattr(full_horizon_min_demand, "__getitem__")
                else list(full_horizon_min_demand)[window_start:commit_end]
            )
            min_window = (min_window + [0.0] * n_steps)[:n_steps]
            self.model.min_hourly_demand_param = pyo.Param(
                range(n_steps), initialize=dict(enumerate(min_window))
            )
            self.model.min_hourly_demand_con = pyo.Constraint(
                range(n_steps),
                rule=lambda m, t: self._primary_output_expr(m, t)
                >= m.min_hourly_demand_param[t],
            )
            logger.info(
                "[RH-STRATEGY] min_demand: per-hour constraints over %d steps", n_steps
            )

        else:
            logger.info("[RH-STRATEGY] cost_optimized: pure cost minimisation")

    def _add_profile_soft_constraints(
        self, window_start: int, window_end: int, full_horizon_load_profile
    ) -> bool:
        """Add soft profile-tracking deviation variables and a penalised objective.

        Returns ``True`` if the constraints were added successfully.
        """
        try:
            window_len = window_end - window_start
            window_profile = (
                list(full_horizon_load_profile[window_start:window_end])
                if hasattr(full_horizon_load_profile, "__getitem__")
                else list(full_horizon_load_profile)[window_start:window_end]
            )
            if len(window_profile) < window_len and window_profile:
                window_profile.extend(
                    [window_profile[-1]] * (window_len - len(window_profile))
                )
            window_profile = window_profile[:window_len]
            if (
                not window_profile
                or max(window_profile) == 0
                or sum(window_profile) == 0
            ):
                return False

            max_power = getattr(self, "max_power", 922.0)
            targets = {
                t: window_profile[t] * max_power for t in range(len(window_profile))
            }
            self.model.profile_dev_pos = pyo.Var(
                self.model.time_steps, within=pyo.NonNegativeReals
            )
            self.model.profile_dev_neg = pyo.Var(
                self.model.time_steps, within=pyo.NonNegativeReals
            )
            self.model.profile_target = pyo.Param(
                self.model.time_steps, initialize=targets, default=0.0
            )

            @self.model.Constraint(self.model.time_steps)
            def profile_deviation_con(m, t):
                return (
                    m.total_power_input[t]
                    == m.profile_target[t] + m.profile_dev_pos[t] - m.profile_dev_neg[t]
                )

            penalty = 10.0 / max(self.load_profile_deviation, 0.01)
            self.model.del_component(self.model.obj_rule_opt)

            @self.model.Objective(sense=pyo.minimize)
            def obj_rule_opt(m):
                return pyo.quicksum(
                    m.variable_cost[t] for t in m.time_steps
                ) + penalty * pyo.quicksum(
                    m.profile_dev_pos[t] + m.profile_dev_neg[t] for t in m.time_steps
                )

            logger.info(
                "[LOAD-PROFILE-RH] Window [%d:%d]: soft constraint active "
                "(penalty=%.1f, deviation=%.0%%)",
                window_start,
                window_end,
                penalty,
                self.load_profile_deviation * 100,
            )
            return True
        except Exception as e:
            logger.warning(
                "[LOAD-PROFILE-RH] Failed to add profile soft constraint: %s. "
                "Continuing without.",
                e,
            )
            return False

    def _solve_with_profile_fallback(
        self, load_profile_added: bool, window_start: int, window_end: int
    ):
        """Create instance and solve; retry without profile penalty if infeasible.

        Returns ``(instance, results)``.
        """
        _profile_comps = [
            "profile_deviation_con",
            "profile_dev_pos",
            "profile_dev_neg",
            "profile_target",
            "obj_rule_opt",
        ]

        def _strip_profile_and_plain_obj():
            for c in _profile_comps:
                if hasattr(self.model, c):
                    self.model.del_component(c)

            @self.model.Objective(sense=pyo.minimize)
            def obj_rule_opt(m):
                return pyo.quicksum(m.variable_cost[t] for t in m.time_steps)

        instance = self.model.create_instance()
        try:
            results = self.solver.solve(instance, options={})
        except RuntimeError as e:
            if not load_profile_added:
                raise
            logger.warning(
                "[LOAD-PROFILE-RH] Solver error in window [%d:%d] with profile: %.80s. "
                "Re-solving without.",
                window_start,
                window_end,
                str(e),
            )
            _strip_profile_and_plain_obj()
            instance = self.model.create_instance()
            results = self.solver.solve(instance, options={})
            load_profile_added = False

        if (
            load_profile_added
            and results.solver.status == SolverStatus.ok
            and results.solver.termination_condition == TerminationCondition.infeasible
        ):
            logger.warning(
                "[LOAD-PROFILE-RH] Profile infeasible in window [%d:%d]; re-solving without.",
                window_start,
                window_end,
            )
            _strip_profile_and_plain_obj()
            instance = self.model.create_instance()
            results = self.solver.solve(instance, options={})

        return instance, results

    def _log_solver_status(self, results, window_start: int, window_end: int) -> None:
        """Log the solver termination status for a window."""
        tc = results.solver.termination_condition
        if (
            results.solver.status == SolverStatus.ok
            and tc == TerminationCondition.optimal
        ):
            logger.debug("Window [%d:%d] solved optimally.", window_start, window_end)
        elif tc == TerminationCondition.infeasible:
            logger.warning(
                "Window [%d:%d] infeasible — committed values remain zero.",
                window_start,
                window_end,
            )
        else:
            logger.warning(
                "Window [%d:%d] solver status: %s / %s",
                window_start,
                window_end,
                results.solver.status,
                tc,
            )

    def _extract_component_operations(
        self,
        instance,
        window_start: int,
        commit_end: int,
        saved_index,
    ) -> None:
        """Append per-component operational data for the committed window steps."""
        if not hasattr(self, "_component_operations"):
            self._component_operations = []

        _block_schema = self._component_schema

        for local_t in range(commit_end - window_start):
            global_t = window_start + local_t
            row: dict = {
                "global_t": global_t,
                "timestamp": str(saved_index[global_t]),
            }
            try:
                if hasattr(instance, "dsm_blocks"):
                    for blk, (
                        pwr_attr,
                        out_attr,
                        pwr_key,
                        out_key,
                    ) in _block_schema.items():
                        row[pwr_key] = 0.0
                        row[out_key] = 0.0
                        if blk not in instance.dsm_blocks:
                            continue
                        b = instance.dsm_blocks[blk]
                        if hasattr(b, pwr_attr) and local_t in getattr(b, pwr_attr):
                            row[pwr_key] = float(
                                pyo.value(getattr(b, pwr_attr)[local_t])
                            )
                        if hasattr(b, out_attr) and local_t in getattr(b, out_attr):
                            row[out_key] = float(
                                pyo.value(getattr(b, out_attr)[local_t])
                            )
            except Exception as ex:
                logger.debug(
                    "Could not extract component data for t=%d: %s", local_t, ex
                )
            self._component_operations.append(row)

    def _solve_rolling_horizon_opt(self) -> None:
        """Perform the cost-optimal solve in rolling-horizon fashion.

        The full horizon is split into overlapping look-ahead windows.  For each
        window a fresh Pyomo model is built and solved; the first *commit_horizon*
        steps are written to the full-horizon result arrays and the component
        states at the end of that period are carried forward to the next window.
        For steel plants, cumulative remaining demand is decremented each window.

        After all windows the full-horizon ``opt_power_requirement``,
        ``variable_cost_series``, and ``total_cost`` are populated.
        """
        N = len(self.index)
        freq = self.index.freq

        look_ahead_steps = self._parse_duration_to_steps(self._rh_look_ahead)
        commit_steps = self._parse_duration_to_steps(self._rh_commit)
        rolling_steps = self._parse_duration_to_steps(self._rh_step)

        logger.info(
            "[RH-ENGINE] Horizon=%d steps, look-ahead=%d, commit=%d, roll=%d",
            N,
            look_ahead_steps,
            commit_steps,
            rolling_steps,
        )

        opt_power: list = [0.0] * N
        var_cost: list = [0.0] * N
        self._rh_full_horizon_production = [0.0] * N

        saved_index = self.index
        saved_model = self.model
        saved_components = self.components

        init_states = self._collect_init_states()

        remaining_demand = None
        if self._has_demand_tracking:
            remaining_demand = getattr(self, self._demand_attr_suffix)
            logger.info("[RH-DEMAND] Initial remaining_demand=%.2f", remaining_demand)

        window_start = 0
        window_num = 0
        while window_start < N:
            window_end = min(window_start + look_ahead_steps, N)
            commit_end = min(window_start + commit_steps, N)

            logger.debug(
                "RH window %d: [%d:%d], commit [%d:%d]",
                window_num,
                window_start,
                window_end,
                window_start,
                commit_end,
            )

            self.index = FastIndex(
                start=saved_index[window_start],
                end=saved_index[window_end - 1],
                freq=freq,
            )
            saved_attrs = self._collect_series_attrs_for_window(
                window_start, window_end, N
            )
            self.components = self._prepare_window_components(
                window_start, window_end, N, init_states, remaining_demand
            )

            self.model = pyo.ConcreteModel()
            self.define_sets()
            self.define_parameters()
            self.define_variables()
            self.initialize_components()
            self.initialize_process_sequence()
            self.define_constraints()

            if remaining_demand is not None and self._has_demand_tracking:
                self.model.demand_limit = pyo.Param(
                    initialize=remaining_demand, mutable=True
                )
                self.model.demand_upper_bound = pyo.Constraint(
                    rule=lambda m: sum(
                        self._primary_output_expr(m, t) for t in m.time_steps
                    )
                    <= m.demand_limit
                )

            self.define_objective_opt()
            instance = self.model.create_instance()
            results = self.solver.solve(instance, options={})
            self._log_solver_status(results, window_start, window_end)

            n_commit = commit_end - window_start
            window_production = 0.0
            for local_t in range(n_commit):
                global_t = window_start + local_t
                power_val = pyo.value(instance.total_power_input[local_t])
                opt_power[global_t] = power_val
                var_cost[global_t] = pyo.value(instance.variable_cost[local_t])
                window_production += power_val
                self._rh_full_horizon_production[global_t] = power_val

            self._update_init_states(instance, n_commit - 1, init_states)

            if remaining_demand is not None and self._has_demand_tracking:
                remaining_demand -= window_production
                logger.debug(
                    "[RH-DEMAND] Committed=%.2f, remaining=%.2f",
                    window_production,
                    remaining_demand,
                )

            self._restore_series_attrs(saved_attrs)
            window_start += rolling_steps
            window_num += 1

        self.index = saved_index
        self.model = saved_model
        self.components = saved_components

        self.opt_power_requirement = FastSeries(index=self.index, value=opt_power)
        self.total_cost = sum(var_cost)
        self.variable_cost_series = FastSeries(index=self.index, value=var_cost)

        logger.info(
            "[RH-COMPLETE] %d windows | total_cost=%.2f | total_production=%.2f",
            window_num,
            self.total_cost,
            sum(opt_power),
        )

    # ------------------------------------------------------------------
    # /Rolling-horizon helpers
    # ------------------------------------------------------------------

    def _check_and_reoptimize_rolling_window(self, current_time) -> bool:
        """Check whether to re-optimise for the next rolling window.

        Called by the bidding strategy each time it generates bids.
        Returns ``True`` if re-optimisation was performed.
        """
        if self.horizon_mode != "rolling_horizon":
            return False

        try:
            current_step = self.index._get_idx_from_date(current_time)
        except (KeyError, ValueError, AttributeError, TypeError) as e:
            logger.debug(
                "Could not map %s to index step: %s. Skipping re-optimisation.",
                current_time,
                e,
            )
            return False

        if current_step >= self._rh_optimized_until_step:
            logger.info(
                "[RH-MARKET-TRIGGER] %s | step=%d >= opt_until=%d | "
                "re-optimising window for %s",
                current_time,
                current_step,
                self._rh_optimized_until_step,
                self.id,
            )
            self._solve_rolling_horizon_next_window(current_step)
            return True

        return False

    def _solve_rolling_horizon_next_window(self, current_step: int) -> None:
        """Optimise the next rolling window starting from *current_step*.

        Called after each market round to re-optimise for the next window only,
        using the current component states as the initial condition.
        """
        N = len(self.index)
        freq = self.index.freq

        if not hasattr(self, "_rh_full_horizon_production"):
            self._rh_full_horizon_production = [0.0] * N

        look_ahead_steps = self._parse_duration_to_steps(self._rh_look_ahead)
        commit_steps = self._parse_duration_to_steps(self._rh_commit)

        window_start = current_step
        window_end = min(window_start + look_ahead_steps, N)
        commit_end = min(window_start + commit_steps, N)

        if window_start >= N:
            return

        logger.info(
            "[RH-WINDOW-START] %s | window=[%d:%d], commit=[%d:%d]",
            self.id,
            window_start,
            window_end,
            window_start,
            commit_end,
        )

        saved_index = self.index
        saved_model = self.model if hasattr(self, "model") else None
        saved_components = self.components

        init_states = self._collect_init_states()

        self.index = FastIndex(
            start=saved_index[window_start],
            end=saved_index[window_end - 1],
            freq=freq,
        )
        saved_attrs = self._collect_series_attrs_for_window(window_start, window_end, N)

        # Detect operation strategy; preserve full-horizon sequences in saved_attrs
        operation_strategy, full_horizon_load_profile, full_horizon_min_demand = (
            self._detect_operation_strategy(N, saved_attrs)
        )
        if operation_strategy != "cost_optimized":
            logger.info("[RH-SETUP] strategy=%s", operation_strategy)

        # Save full-horizon time-series attributes not yet captured by
        # _collect_series_attrs_for_window (subclass-specific, e.g. commodity prices)
        for attr_name in self._extra_price_attrs:
            if attr_name not in saved_attrs and hasattr(self, attr_name):
                val = getattr(self, attr_name)
                try:
                    if len(val) == N and hasattr(val, "__getitem__"):
                        saved_attrs[attr_name] = val
                        setattr(self, attr_name, val[window_start:window_end])
                except (TypeError, AttributeError):
                    pass

        _pending_opr_updates: dict = {}
        window_production = 0.0

        try:
            remaining_demand = None
            if self._has_demand_tracking:
                produced_so_far = sum(
                    float(v)
                    for v in self._rh_full_horizon_production[:current_step]
                    if v and v > 0
                )
                total_demand = getattr(self, self._demand_attr_suffix)
                remaining_demand = max(0.0, total_demand - produced_so_far)
                logger.info(
                    "[RH-WINDOW-DEMAND] global=%.2f, produced=%.2f, remaining=%.2f",
                    total_demand,
                    produced_so_far,
                    remaining_demand,
                )

            self.components = self._prepare_window_components(
                window_start, window_end, N, init_states, remaining_demand
            )

            self.model = pyo.ConcreteModel()
            self.define_sets()
            self.define_parameters()
            self.define_variables()
            self.initialize_components()
            self.initialize_process_sequence()
            self.define_constraints()

            if remaining_demand is not None and self._has_demand_tracking:
                self._add_window_demand_constraints(
                    operation_strategy,
                    remaining_demand,
                    full_horizon_load_profile,
                    full_horizon_min_demand,
                    window_start,
                    commit_end,
                )

            self.define_objective_opt()

            load_profile_added = False
            if (
                self._has_demand_tracking
                and operation_strategy == "profile_guided"
                and full_horizon_load_profile is not None
            ):
                load_profile_added = self._add_profile_soft_constraints(
                    window_start, window_end, full_horizon_load_profile
                )

            instance, results = self._solve_with_profile_fallback(
                load_profile_added, window_start, window_end
            )
            self._log_solver_status(results, window_start, window_end)

            n_commit = commit_end - window_start
            for local_t in range(n_commit):
                global_t = window_start + local_t
                power_val = pyo.value(instance.total_power_input[local_t])
                _pending_opr_updates[global_t] = power_val

                try:
                    prod_val = pyo.value(self._primary_output_expr(instance, local_t))
                except (KeyError, AttributeError, TypeError):
                    prod_val = power_val

                window_production += prod_val
                if hasattr(self, "_rh_full_horizon_production"):
                    self._rh_full_horizon_production[global_t] = prod_val

            logger.info("[RH-MARKET] Window production: %.2f MWh", window_production)

            if self._has_demand_tracking:
                self._extract_component_operations(
                    instance, window_start, commit_end, saved_index
                )

            self._update_init_states(instance, n_commit - 1, init_states)

        finally:
            self._restore_series_attrs(saved_attrs)
            self.index = saved_index
            if saved_model is not None:
                self.model = saved_model
            self.components = saved_components

        for global_t, power_val in _pending_opr_updates.items():
            try:
                self.opt_power_requirement[saved_index[global_t]] = power_val
            except (IndexError, KeyError, AttributeError, TypeError) as e:
                logger.debug(
                    "Could not update opt_power_requirement[%d]: %s", global_t, e
                )

        self._rh_window_start_idx = window_start
        self._rh_optimized_until_step = commit_end
        logger.info(
            "[RH-WINDOW-COMPLETE] production=%.2f MWh | opt_until=%d | states=%s",
            window_production,
            commit_end,
            list(init_states.keys()),
        )

    # ------------------------------------------------------------------

    def determine_optimal_operation_without_flex(self, switch_flex_off=True):
        """
        Determines the optimal operation of the steel plant without considering flexibility.

        Note: Rolling horizon optimization is disabled during forecaster initialization
        (switch_flex_off=False) as it can cause infeasibility during model setup.
        Rolling horizon will be used for market time re-optimization later.
        """
        # During forecaster initialization, always use full-horizon method
        # Rolling horizon will be used later during market time re-optimization
        if not switch_flex_off:
            pass
        elif self.horizon_mode == "rolling_horizon":
            logger.info(
                "[ROLLING-HORIZON] %s: Starting rolling-horizon optimization "
                "(look_ahead=%s, commit=%s, step=%s)",
                self.id,
                self._rh_look_ahead,
                self._rh_commit,
                self._rh_step,
            )
            try:
                self._solve_rolling_horizon_opt()
                logger.info(
                    "[ROLLING-HORIZON] %s: Rolling-horizon optimization complete",
                    self.id,
                )
                return
            except Exception as rh_error:
                logger.warning(
                    "[ROLLING-HORIZON] Rolling horizon optimization failed: %s. "
                    "Falling back to full-horizon optimization.",
                    type(rh_error).__name__,
                )
                # Fall through to full-horizon solve below

        # create an instance of the model
        instance = self.model.create_instance()
        # switch the instance to the optimal mode by deactivating the flexibility constraints and objective
        if switch_flex_off:
            instance = self.switch_to_opt(instance)

        # solve the instance
        results = self.solver.solve(instance, options={})

        # Check solver status and termination condition
        if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal
        ):
            logger.debug("The model was solved optimally.")
            # Display the Objective Function Value
            objective_value = instance.obj_rule_opt()
            logger.debug("The value of the objective function is %s.", objective_value)

        elif results.solver.termination_condition == TerminationCondition.infeasible:
            logger.warning("The model is infeasible - check constraints.")
            if switch_flex_off and self._has_demand_tracking:
                total_demand = getattr(self, self._demand_attr_suffix)
                logger.warning(
                    "  Full-horizon solve is INFEASIBLE with demand constraint of %.2f",
                    total_demand,
                )
            return

        else:
            logger.debug("Solver Status: %s", results.solver.status)
            logger.debug(
                "Termination Condition: %s", results.solver.termination_condition
            )

        opt_power_requirement = [
            pyo.value(instance.total_power_input[t]) for t in instance.time_steps
        ]
        self.opt_power_requirement = FastSeries(
            index=self.index, value=opt_power_requirement
        )

        # CRITICAL: If rolling horizon is enabled, populate the full-horizon accumulator
        # This ensures the first market call has correct production history
        if self.horizon_mode == "rolling_horizon":
            if not hasattr(self, "_rh_full_horizon_production"):
                self._rh_full_horizon_production = [0.0] * len(self.index)

            # Track primary production output (subclass may override _primary_output_expr)
            try:
                production_schedule = [
                    pyo.value(self._primary_output_expr(instance, t))
                    for t in instance.time_steps
                ]
            except (KeyError, AttributeError):
                production_schedule = opt_power_requirement

            for i, prod_val in enumerate(production_schedule):
                self._rh_full_horizon_production[i] = prod_val

            # Log the full-horizon production profile
            total_full_horizon = sum(production_schedule)
            logger.info(
                f"[RH-SETUP] Full-horizon solve complete: Total production = {total_full_horizon:.2f} MWh, "
                f"First 24 hours = {sum(production_schedule[:24]):.2f} MWh"
            )
            logger.debug(
                f"[RH-SETUP] Full-horizon production schedule (first 24h): {[f'{v:.1f}' for v in production_schedule[:24]]}"
            )

        self.total_cost = sum(
            instance.variable_cost[t].value for t in instance.time_steps
        )

        # Variable cost series
        variable_cost = [
            pyo.value(instance.variable_cost[t]) for t in instance.time_steps
        ]
        self.variable_cost_series = FastSeries(index=self.index, value=variable_cost)

    def determine_optimal_operation_with_flex(self):
        """
        Determines the optimal operation of the steel plant without considering flexibility.
        """
        # create an instance of the model
        instance = self.model.create_instance()
        # switch the instance to the flexibility mode by deactivating the optimal constraints and objective
        instance = self.switch_to_flex(instance)

        # solve the instance with error handling
        try:
            results = self.solver.solve(instance)
        except RuntimeError as e:
            # If solver fails, log warning and continue with current solution
            logger.warning(
                f"[FLEX-PHASE] Solver raised RuntimeError during flexibility phase: {str(e)[:80]}... "
                "Continuing with current optimal operation without flexibility."
            )
            return

        # Check solver status and termination condition
        if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal
        ):
            logger.debug("The model was solved optimally.")

            # Display the Objective Function Value
            objective_value = instance.obj_rule_flex()
            logger.debug("The value of the objective function is %s.", objective_value)

        elif results.solver.termination_condition == TerminationCondition.infeasible:
            logger.debug("The model is infeasible.")

        else:
            logger.debug("Solver Status: ", results.solver.status)
            logger.debug(
                "Termination Condition: ", results.solver.termination_condition
            )

        if self.flexibility_measure == "electricity_price_signal":
            flex_power_requirement = [
                pyo.value(instance.total_power_input[t]) for t in instance.time_steps
            ]
            self.flex_power_requirement = FastSeries(
                index=self.index, value=flex_power_requirement
            )
        elif self.flexibility_measure == "symmetric_flexible_block":
            adjusted_total_power_input = []
            for t in instance.time_steps:
                total_power = 0.0
                for tech_name, tech_block in instance.dsm_blocks.items():
                    if hasattr(tech_block, "power_in"):
                        total_power += pyo.value(tech_block.power_in[t])
                adjusted_total_power_input.append(total_power)

            self.flex_power_requirement = FastSeries(
                index=self.index, value=adjusted_total_power_input
            )

            # Now assign to your series
            self.flex_power_requirement = FastSeries(
                index=self.index, value=adjusted_total_power_input
            )

        else:
            # Compute adjusted total power input with load shift applied
            adjusted_total_power_input = []
            for t in instance.time_steps:
                # Calculate the load-shifted value of total_power_input
                adjusted_power = (
                    instance.total_power_input[t].value
                    + instance.load_shift_pos[t].value
                    - instance.load_shift_neg[t].value
                )
                adjusted_total_power_input.append(adjusted_power)

            # Assign this list to flex_power_requirement as a pandas Series
            self.flex_power_requirement = FastSeries(
                index=self.index, value=adjusted_total_power_input
            )

        # Variable cost series
        flex_variable_cost = [
            instance.variable_cost[t].value for t in instance.time_steps
        ]
        self.flex_variable_cost_series = FastSeries(
            index=self.index, value=flex_variable_cost
        )

    def switch_to_opt(self, instance):
        """
        Switches the instance to solve a cost based optimisation problem by deactivating the flexibility constraints and objective.

        Args:
            instance (pyomo.ConcreteModel): The instance of the Pyomo model.

        Returns:
            pyomo.ConcreteModel: The modified instance with flexibility constraints and objective deactivated.
        """

        instance.obj_rule_flex.deactivate()

        # Deactivate flexibility constraints if they exist
        if hasattr(instance, "total_cost_upper_limit"):
            instance.total_cost_upper_limit.deactivate()

        if hasattr(instance, "peak_load_shift_constraint"):
            instance.peak_load_shift_constraint.deactivate()

        # if hasattr(instance, "total_power_input_constraint_with_flex"):
        instance.total_power_input_constraint_with_flex.deactivate()

        return instance

    def switch_to_flex(self, instance):
        """
        Switches the instance to flexibility mode by deactivating few constraints and objective function.

        Args:
            instance (pyomo.ConcreteModel): The instance of the Pyomo model.

        Returns:
            pyomo.ConcreteModel: The modified instance with optimal constraints and objective deactivated.
        """
        # deactivate the optimal constraints and objective
        instance.obj_rule_opt.deactivate()
        instance.total_power_input_constraint.deactivate()

        # fix values of model.total_power_input
        for t in instance.time_steps:
            # Handle both pandas Series and numpy arrays
            if hasattr(self.opt_power_requirement, "iloc"):
                power_value = self.opt_power_requirement.iloc[t]
            else:
                power_value = self.opt_power_requirement[t]
            instance.total_power_input[t].fix(power_value)
        instance.total_cost = self.total_cost

        return instance

    def calculate_marginal_cost(self, start: datetime, power: float) -> float:
        """
        Calculates the marginal cost of operating the building unit at a specific time and power level.

        The marginal cost represents the additional cost incurred by increasing the power output by one unit.

        Args:
            start (datetime): The start time of the dispatch period.
            power (float): The power output level of the unit during the dispatch.

        Returns:
            float: The marginal cost of the unit for the given power level. Returns 0 if there is no power requirement.
        """
        # Initialize marginal cost
        marginal_cost = 0
        epsilon = 1e-3

        if power > epsilon:
            marginal_cost = abs(self.variable_cost_series.at[start] / power)
        return marginal_cost

    def as_dict(self) -> dict:
        """
        Returns the attributes of the unit as a dictionary, including specific attributes.

        Returns:
            dict: The attributes of the unit as a dictionary.
        """
        # Assuming unit_dict is a dictionary that you want to save to the database
        components_list = [component for component in self.model.dsm_blocks.keys()]

        # Convert the list to a delimited string
        components_string = ",".join(components_list)

        unit_dict = super().as_dict()
        unit_dict.update(
            {
                "unit_type": "demand",
                "components": components_string,
            }
        )

        return unit_dict
