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
from assume.common.utils import get_supported_solver
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

    def __init__(self, components, **kwargs):
        # Extract rolling-horizon optimisation config before passing **kwargs up the MRO.
        dsm_opt = kwargs.pop("dsm_optimisation_config", None)
        if not isinstance(dsm_opt, dict):
            dsm_opt = {}

        super().__init__(**kwargs)

        self.components = components
        self.solver = SolverFactory(get_supported_solver())

        # Rolling-horizon settings (populated from config; default is full horizon)
        self._horizon_mode = dsm_opt.get("horizon_mode", "full_horizon")
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

        if self._horizon_mode == "rolling_horizon":
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

        # For steel plants, patch the cumulative remaining_demand if provided
        if remaining_demand is not None and self.technology == "steel_plant":
            if "eaf" in window_comps and isinstance(window_comps["eaf"], dict):
                # Store remaining_demand for later use by define_parameters
                window_comps["_remaining_demand"] = remaining_demand

        return window_comps

    def _solve_rolling_horizon_opt(self) -> None:
        """Perform the cost-optimal solve in rolling-horizon fashion.

        The full simulation horizon is split into overlapping look-ahead
        windows.  For each window:

        1. A fresh Pyomo model is built for the window's time steps only.
        2. The model is solved.
        3. The first *commit_horizon* time steps of the solution are written
           to the full-horizon result arrays.
        4. The state of charge (SoC) and dynamic operational states (on/off status,
           startup/shutdown flags) of components at the end of the committed period
           are passed forward as the initial condition for the next window.
        5. For steel plants, the cumulative remaining demand is tracked and decremented
           based on committed production, ensuring correct multi-window demand fulfillment.

        After all windows the full-horizon ``opt_power_requirement``,
        ``variable_cost_series``, and ``total_cost`` are set exactly as the
        non-rolling path does.
        """
        N = len(self.index)
        freq = self.index.freq

        look_ahead_steps = self._parse_duration_to_steps(self._rh_look_ahead)
        commit_steps = self._parse_duration_to_steps(self._rh_commit)
        rolling_steps = self._parse_duration_to_steps(self._rh_step)

        print(
            f"  [RH-ENGINE] Horizon length: {N} steps, "
            f"Look-ahead: {look_ahead_steps} steps, "
            f"Commit: {commit_steps} steps, Roll: {rolling_steps} steps"
        )

        # Pre-allocate full-horizon result buffers
        opt_power: list[float] = [0.0] * N
        var_cost: list[float] = [0.0] * N

        # CRITICAL: Initialize persistent full-horizon accumulator for multi-window tracking
        # This survives across windows so Window 2+ can see production from earlier windows
        self._rh_full_horizon_production = [0.0] * N

        # Save originals so we can restore them after the loop
        saved_index = self.index
        saved_model = self.model
        saved_components = self.components  # component *instances* at this point

        # Collect initial states from the original component dicts.
        # Each tech maps to a dict of state variables (soc, operational_status, etc.)
        init_states: dict[str, dict[str, float]] = {}
        for tech, data in self._orig_components_dict.items():
            if isinstance(data, dict):
                tech_state = {}
                if "initial_soc" in data:
                    tech_state["soc"] = data["initial_soc"]
                if "initial_operational_status" in data:
                    tech_state["operational_status"] = data[
                        "initial_operational_status"
                    ]
                if tech_state:  # Only add if there's at least one state variable
                    init_states[tech] = tech_state

        # Initialize cumulative demand tracking for steel plants
        remaining_demand = None
        if self.technology == "steel_plant":
            remaining_demand = self.steel_demand
            print(
                f"  [RH-DEMAND] Initialized remaining_demand = {remaining_demand:.2f}"
            )

        window_start = 0
        window_num = 0
        while window_start < N:
            window_end = min(window_start + look_ahead_steps, N)
            commit_end = min(window_start + commit_steps, N)

            print(
                f"  [RH-WINDOW {window_num}] [{window_start}:{window_end}] "
                f"(commit: [{window_start}:{commit_end}])"
            )
            if remaining_demand is not None:
                print(f"    [RH-DEMAND] Remaining: {remaining_demand:.2f}")

            logger.debug(
                "Rolling-horizon window [%d:%d], committing [%d:%d]",
                window_start,
                window_end,
                window_start,
                commit_end,
            )

            # --- Set up the window "environment" on self ---
            self.index = FastIndex(
                start=saved_index[window_start],
                end=saved_index[window_end - 1],
                freq=freq,
            )

            # Slice all unit-level FastSeries attrs to the window length
            saved_attrs = self._collect_series_attrs_for_window(
                window_start, window_end, N
            )

            # Build component dicts with window-sliced time-series and
            # carried-forward initial states (including remaining_demand for steel plants)
            self.components = self._prepare_window_components(
                window_start, window_end, N, init_states, remaining_demand
            )

            # --- Build and solve the window model ---
            self.model = pyo.ConcreteModel()
            self.define_sets()
            self.define_parameters()
            self.define_variables()
            self.initialize_components()
            self.initialize_process_sequence()
            self.define_constraints()

            # Add demand limit constraint for steel plants in rolling horizon
            if remaining_demand is not None and self.technology == "steel_plant":
                # Add as a Pyomo Parameter first (before create_instance)
                self.model.demand_limit = pyo.Param(
                    initialize=remaining_demand, mutable=True
                )

                def demand_upper_bound_rule(m):
                    """Upper bound: total production <= remaining demand"""
                    return (
                        sum(m.total_power_input[t] for t in m.time_steps)
                        <= m.demand_limit
                    )

                self.model.demand_upper_bound = pyo.Constraint(
                    rule=demand_upper_bound_rule
                )
                print(f"      [RH-CONSTRAINT] Demand limit: {remaining_demand:.2f} MWh")

            self.define_objective_opt()

            instance = self.model.create_instance()

            try:
                # Debug: Print model statistics before solving
                num_vars = sum(1 for _ in instance.component_data_objects(pyo.Var))
                num_cons = sum(
                    1 for _ in instance.component_data_objects(pyo.Constraint)
                )
                print(
                    f"      [RH-MODEL] Variables: {num_vars} | Constraints: {num_cons}"
                )

                results = self.solver.solve(instance, options={})
            except RuntimeError as solve_error:
                logger.error(
                    f"[RH-ERROR] Solver RuntimeError in window [{window_start}:{window_end}]: {str(solve_error)[:200]}"
                )
                print("      [RH-ERROR] RuntimeError from solver:")
                print(f"      {str(solve_error)[:300]}")
                # For now, re-raise to see the full error
                raise
            except Exception as solve_error:
                logger.error(
                    f"[RH-ERROR] Unexpected solver error in window [{window_start}:{window_end}]: {type(solve_error).__name__}"
                )
                print(
                    f"      [RH-ERROR] Unexpected error: {type(solve_error).__name__}"
                )
                raise

            if (results.solver.status == SolverStatus.ok) and (
                results.solver.termination_condition == TerminationCondition.optimal
            ):
                logger.debug(
                    "Rolling window [%d:%d] solved optimally.", window_start, window_end
                )
            elif (
                results.solver.termination_condition == TerminationCondition.infeasible
            ):
                logger.warning(
                    "Rolling window [%d:%d] is infeasible — committed values will remain zero.",
                    window_start,
                    window_end,
                )
            else:
                logger.warning(
                    "Rolling window [%d:%d] solver status: %s / %s",
                    window_start,
                    window_end,
                    results.solver.status,
                    results.solver.termination_condition,
                )

            # --- Extract committed results ---
            n_commit = commit_end - window_start
            window_production = 0.0
            for local_t in range(n_commit):
                global_t = window_start + local_t
                power_val = pyo.value(instance.total_power_input[local_t])
                opt_power[global_t] = power_val
                var_cost[global_t] = pyo.value(instance.variable_cost[local_t])
                window_production += power_val
                # CRITICAL: Update the persistent full-horizon accumulator
                # This allows subsequent windows to see cumulative production
                self._rh_full_horizon_production[global_t] = power_val

            # --- Transfer end-of-commit state to the next window ---
            commit_local = n_commit - 1  # last committed step (0-based in window)

            # Extract all dynamic states from the solved instance
            for tech_name in list(init_states.keys()):
                try:
                    block = instance.dsm_blocks[tech_name]
                    state_vars = {}

                    # Extract SoC if present
                    if hasattr(block, "soc"):
                        state_vars["soc"] = pyo.value(block.soc[commit_local])

                    # Extract operational status if present
                    if hasattr(block, "operational_status"):
                        state_vars["operational_status"] = int(
                            round(pyo.value(block.operational_status[commit_local]))
                        )

                    # Note: start_up and shut_down are decision variables that derive from
                    # operational_status transitions, so we don't need to carry them explicitly.
                    # The next window's initial_operational_status will be set to the commit-end
                    # status, and start_up/shut_down will be recomputed in the next window.

                    if state_vars:
                        init_states[tech_name] = state_vars
                except (KeyError, AttributeError):
                    pass

            # --- Update cumulative demand tracking for steel plants ---
            if remaining_demand is not None and self.technology == "steel_plant":
                remaining_demand -= window_production
                print(
                    f"    [RH-DEMAND] Committed production: {window_production:.2f}, Remaining after: {remaining_demand:.2f}"
                )

            # Restore the unit-level attrs for the next iteration / final model
            self._restore_series_attrs(saved_attrs)

            window_start += rolling_steps
            window_num += 1

        # --- Restore the full-horizon environment ---
        self.index = saved_index
        self.model = (
            saved_model  # restore the full-horizon model (used by flex measure)
        )
        self.components = saved_components  # restore component instances

        # Store rolling-horizon results (same API as the full-horizon path)
        self.opt_power_requirement = FastSeries(index=self.index, value=opt_power)
        self.total_cost = sum(var_cost)
        self.variable_cost_series = FastSeries(index=self.index, value=var_cost)

        total_production = sum(opt_power)
        print(
            f"  [RH-COMPLETE] Processed {window_num} windows, "
            f"Total cost: {self.total_cost:.2f}, "
            f"Total production: {total_production:.2f}, "
            f"Avg power: {sum(opt_power) / len(opt_power):.2f}"
        )
        if remaining_demand is not None:
            print(f"  [RH-DEMAND] Final remaining: {remaining_demand:.2f}")

    # ------------------------------------------------------------------
    # /Rolling-horizon helpers
    # ------------------------------------------------------------------

    def _check_and_reoptimize_rolling_window(self, current_time) -> bool:
        """
        Check if we need to re-optimize for the next rolling window based on the current market request time.

        This method is called by the bidding strategy each time it needs to generate bids.
        If the current market time is beyond the last optimized window, it re-optimizes for the next window.

        Args:
            current_time (datetime): The current market request time

        Returns:
            bool: True if re-optimization was performed, False otherwise
        """
        if self._horizon_mode != "rolling_horizon":
            return False

        # Find the step index corresponding to current_time
        try:
            current_step = self.index._get_idx_from_date(current_time)
        except (KeyError, ValueError, AttributeError, TypeError) as e:
            logger.debug(
                f"Could not map current_time {current_time} to index: {e}. Skipping re-optimization check"
            )
            return False

        # Check if we need to re-optimize
        if current_step >= self._rh_optimized_until_step:
            msg = (
                f"\n{'=' * 100}\n"
                f"[RH-MARKET-TRIGGER] {current_time} | Step {current_step} >= Optimized until {self._rh_optimized_until_step}\n"
                f"[RH-MARKET-TRIGGER] Re-optimizing for next rolling window for unit: {self.id}\n"
                f"{'=' * 100}"
            )
            print(msg)
            logger.info(msg)
            self._solve_rolling_horizon_next_window(current_step)
            return True

        return False

    def _solve_rolling_horizon_next_window(self, current_step: int) -> None:
        """
        Optimize for just the next rolling window starting from current_step.

        This is called after each market round and optimizes only for the next window,
        using the current state (SoC, operational status) as the starting point.

        Args:
            current_step (int): The current step index in the full horizon
        """
        N = len(self.index)
        freq = self.index.freq

        # CRITICAL: Initialize full-horizon accumulator on first market call if not already done
        if not hasattr(self, "_rh_full_horizon_production"):
            self._rh_full_horizon_production = [0.0] * N
            logger.debug(
                f"[RH-INIT] Initialized full-horizon accumulator for {N} steps"
            )

        look_ahead_steps = self._parse_duration_to_steps(self._rh_look_ahead)
        commit_steps = self._parse_duration_to_steps(self._rh_commit)
        rolling_steps = self._parse_duration_to_steps(self._rh_step)

        window_start = current_step
        window_end = min(window_start + look_ahead_steps, N)
        commit_end = min(window_start + commit_steps, N)

        if window_start >= N:
            logger.debug(
                f"[RH-MARKET] Already at end of horizon ({window_start} >= {N}), no re-optimization needed"
            )
            return

        msg = (
            f"\n[RH-WINDOW-START] Unit: {self.id} | Window: [{window_start}:{window_end}] "
            f"(commit: [{window_start}:{commit_end}]) | Total horizon steps: {N}"
        )
        print(msg)
        logger.info(msg)

        # Save originals so we can restore them after this window
        saved_index = self.index
        saved_model = self.model if hasattr(self, "model") else None
        saved_components = self.components

        # Pre-allocate result buffer for this window
        opt_power_window = [0.0] * (commit_end - window_start)
        var_cost_window = [0.0] * (commit_end - window_start)

        # Collect current states (SoC, operational_status) for all techs
        init_states: dict[str, dict[str, float]] = {}
        for tech, data in self._orig_components_dict.items():
            if isinstance(data, dict):
                tech_state = {}
                if "initial_soc" in data:
                    tech_state["soc"] = data["initial_soc"]
                if "initial_operational_status" in data:
                    tech_state["operational_status"] = data[
                        "initial_operational_status"
                    ]
                if tech_state:
                    init_states[tech] = tech_state

        # --- Set up the window "environment" on self ---
        self.index = FastIndex(
            start=saved_index[window_start],
            end=saved_index[window_end - 1],
            freq=freq,
        )

        # Slice all unit-level FastSeries attrs to the window length
        saved_attrs = self._collect_series_attrs_for_window(window_start, window_end, N)

        # For steel plants, explicitly save and slice price attributes and strategy-specific data
        # (they might be FastSeries that got sliced by _collect_series_attrs_for_window)
        full_horizon_load_profile = None
        full_horizon_min_demand = None
        operation_strategy = "cost_optimized"  # Default strategy

        if self.technology == "steel_plant":
            unit_id = str(getattr(self, "id", None))

            # Strategy Detection: Check for ID-prefixed attributes
            # Strategy 1: Normalized load profile (profile-guided)
            profile_attr = f"{unit_id}_normalized_load_profile" if unit_id else None
            if profile_attr and hasattr(self, profile_attr):
                normalized_load_profile_val = getattr(self, profile_attr)
                if normalized_load_profile_val is not None:
                    operation_strategy = "profile_guided"
                    if hasattr(normalized_load_profile_val, "__len__"):
                        try:
                            profile_len = len(normalized_load_profile_val)
                            if profile_len == N or profile_len == N - 1:
                                full_horizon_load_profile = normalized_load_profile_val
                                if profile_attr not in saved_attrs:
                                    saved_attrs[profile_attr] = (
                                        normalized_load_profile_val
                                    )
                                    logger.info(
                                        f"[RH-SETUP] Strategy=profile_guided: Saved full-horizon profile ({len(full_horizon_load_profile)} steps, N={N})"
                                    )
                            elif profile_len < N:
                                profile_list = list(normalized_load_profile_val)
                                if len(profile_list) > 0:
                                    profile_list.extend(
                                        [profile_list[-1]] * (N - len(profile_list))
                                    )
                                    full_horizon_load_profile = profile_list
                                    saved_attrs[profile_attr] = (
                                        full_horizon_load_profile
                                    )
                                    logger.info(
                                        f"[RH-SETUP] Strategy=profile_guided: Padded profile from {profile_len} to {N}"
                                    )
                            else:
                                full_horizon_load_profile = list(
                                    normalized_load_profile_val
                                )[:N]
                                saved_attrs[profile_attr] = full_horizon_load_profile
                                logger.warning(
                                    f"[RH-SETUP] Strategy=profile_guided: Truncated profile from {profile_len} to {N}"
                                )
                        except (TypeError, AttributeError) as e:
                            logger.warning(f"[RH-SETUP] Error processing profile: {e}")

            # Strategy 2: Hourly minimum steel demand (min-demand)
            demand_attr = f"{unit_id}_steel_demand" if unit_id else None
            if (
                demand_attr
                and hasattr(self, demand_attr)
                and operation_strategy == "cost_optimized"
            ):
                steel_demand_val = getattr(self, demand_attr)
                if steel_demand_val is not None:
                    operation_strategy = "min_demand"
                    if hasattr(steel_demand_val, "__len__"):
                        try:
                            demand_len = len(steel_demand_val)
                            if demand_len == N or demand_len == N - 1:
                                full_horizon_min_demand = steel_demand_val
                                if demand_attr not in saved_attrs:
                                    saved_attrs[demand_attr] = steel_demand_val
                                    logger.info(
                                        f"[RH-SETUP] Strategy=min_demand: Saved hourly minimums ({len(full_horizon_min_demand)} steps)"
                                    )
                            elif demand_len < N:
                                demand_list = list(steel_demand_val)
                                demand_list.extend([0.0] * (N - len(demand_list)))
                                full_horizon_min_demand = demand_list
                                saved_attrs[demand_attr] = full_horizon_min_demand
                                logger.info(
                                    f"[RH-SETUP] Strategy=min_demand: Padded demands from {demand_len} to {N}"
                                )
                            else:
                                full_horizon_min_demand = list(steel_demand_val)[:N]
                                logger.warning(
                                    f"[RH-SETUP] Strategy=min_demand: Truncated demands from {demand_len} to {N}"
                                )
                        except (TypeError, AttributeError) as e:
                            logger.warning(
                                f"[RH-SETUP] Error processing min demand: {e}"
                            )

            if operation_strategy != "cost_optimized":
                logger.info(f"[RH-SETUP] Operating strategy: {operation_strategy}")

            # Also save and slice price attributes
            price_attrs = [
                "electricity_price",
                "hydrogen_price",
                "natural_gas_price",
                "steel_price",
                "iron_ore_price",
                "lime_price",
                "co2_price",
            ]
            for attr_name in price_attrs:
                if hasattr(self, attr_name):
                    val = getattr(self, attr_name)
                    # Only save if not already saved by _collect_series_attrs_for_window
                    if attr_name not in saved_attrs:
                        # Check if it's array-like with length matching full horizon
                        try:
                            if len(val) == N:
                                saved_attrs[attr_name] = val
                                # Slice it for the window
                                if hasattr(val, "__getitem__"):  # array-like
                                    sliced = val[window_start:window_end]
                                    setattr(self, attr_name, sliced)
                        except (TypeError, AttributeError):
                            pass  # Not array-like, skip

        # Pending opt_power_requirement updates — applied AFTER _restore_series_attrs
        # so we write into the fully-restored FastSeries (not the numpy window slice).
        _pending_opr_updates: dict[int, float] = {}
        window_production = (
            0.0  # initialise here so finally block can reference it safely
        )

        try:
            # Get remaining demand for steel plants (if applicable)
            remaining_demand = None
            if self.technology == "steel_plant":
                total_production_so_far = 0.0
                if hasattr(self, "_rh_full_horizon_production"):
                    for i in range(current_step):
                        if i < len(self._rh_full_horizon_production):
                            val = self._rh_full_horizon_production[i]
                            total_production_so_far += (
                                float(val) if val is not None and val > 0 else 0.0
                            )
                remaining_demand = max(0, self.steel_demand - total_production_so_far)
                msg = (
                    f"[RH-WINDOW-DEMAND] Global demand: {self.steel_demand:.2f} | "
                    f"Produced so far (steps 0-{current_step}): {total_production_so_far:.2f} | "
                    f"Remaining for this window: {remaining_demand:.2f}"
                )
                print(msg)
                logger.info(msg)

            # Build component dicts with window-sliced time-series and carried-forward states
            self.components = self._prepare_window_components(
                window_start, window_end, N, init_states, remaining_demand
            )

            # --- Build and solve the window model ---
            self.model = pyo.ConcreteModel()
            self.define_sets()
            self.define_parameters()
            self.define_variables()
            self.initialize_components()
            self.initialize_process_sequence()
            self.define_constraints()

            # Add demand constraint for steel plants in rolling horizon window
            if remaining_demand is not None and self.technology == "steel_plant":
                # For min_demand strategy: per-hour minimums are the sole demand driver.
                # WARNING: Component minimum operating levels may force production above per-hour minimums!
                # If min_demand << component_min_power, the optimizer will be forced to overproduce.
                if operation_strategy == "min_demand":
                    pass  # window_demand_con intentionally omitted; only per-hour constraints apply
                else:
                    self.model.window_demand_limit = pyo.Param(
                        initialize=remaining_demand, mutable=True
                    )

                    def window_demand_rule(m):
                        return (
                            sum(
                                m.dsm_blocks["eaf"].steel_output[t]
                                for t in m.time_steps
                            )
                            <= m.window_demand_limit
                        )

                    self.model.window_demand_con = pyo.Constraint(
                        rule=window_demand_rule
                    )

                # --- Strategy 1: Profile-guided (soft constraints + commit-window minimum) ---
                if (
                    operation_strategy == "profile_guided"
                    and full_horizon_load_profile is not None
                    and remaining_demand > 0
                ):
                    # Calculate what fraction of the remaining profile falls in the COMMIT window
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

                    remaining_profile_sum = (
                        sum(remaining_profile) if remaining_profile else 1.0
                    )
                    commit_profile_sum = sum(commit_profile) if commit_profile else 0.0

                    if remaining_profile_sum > 0:
                        profile_fraction = commit_profile_sum / remaining_profile_sum
                        # Minimum production for COMMITTED hours: profile-proportional share × (1 - deviation)
                        min_commit_production = (
                            remaining_demand
                            * profile_fraction
                            * (1.0 - self.load_profile_deviation)
                        )
                        n_commit_steps = commit_end - window_start

                        self.model.min_commit_production = pyo.Param(
                            initialize=min_commit_production, mutable=True
                        )

                        # Constraint applies ONLY to committed timesteps (not full look-ahead)
                        def window_min_commit_rule(m):
                            return (
                                sum(
                                    m.dsm_blocks["eaf"].steel_output[t]
                                    for t in range(n_commit_steps)
                                )
                                >= m.min_commit_production
                            )

                        self.model.window_min_commit_con = pyo.Constraint(
                            rule=window_min_commit_rule
                        )
                        logger.info(
                            f"[RH-STRATEGY] profile_guided: min_commit={min_commit_production:.1f} MWh over {n_commit_steps} steps "
                            f"(profile_fraction={profile_fraction:.2f}, max_total={remaining_demand:.1f} MWh)"
                        )

                # --- Strategy 2: Min-demand (per-hour minimum constraints) ---
                # The per-hour minimums are the sole demand target for this strategy.
                # No global remaining-demand cap is applied (see above).
                elif (
                    operation_strategy == "min_demand"
                    and full_horizon_min_demand is not None
                ):
                    n_steps = commit_end - window_start
                    min_demand_window = (
                        list(full_horizon_min_demand[window_start:commit_end])
                        if hasattr(full_horizon_min_demand, "__getitem__")
                        else list(full_horizon_min_demand)[window_start:commit_end]
                    )

                    # Pad to match window size if needed
                    while len(min_demand_window) < n_steps:
                        min_demand_window.append(0.0)
                    min_demand_window = min_demand_window[:n_steps]

                    min_val = min(min_demand_window) if min_demand_window else 0
                    max_val = max(min_demand_window) if min_demand_window else 0
                    total_min = sum(min_demand_window)
                    logger.info(
                        f"[RH-DEBUG] min_demand_window: n_steps={n_steps}, min={min_val:.2f}, max={max_val:.2f}, values={min_demand_window[:5]}..."
                    )

                    # Check component constraints from original dict
                    eaf_max = (
                        self._orig_components_dict.get("eaf", {}).get("max_power", 0)
                        if hasattr(self, "_orig_components_dict")
                        else 0
                    )
                    dri_min = (
                        self._orig_components_dict.get("dri_plant", {}).get(
                            "min_power", 0
                        )
                        if hasattr(self, "_orig_components_dict")
                        else 0
                    )
                    dri_max = (
                        self._orig_components_dict.get("dri_plant", {}).get(
                            "max_power", 0
                        )
                        if hasattr(self, "_orig_components_dict")
                        else 0
                    )
                    logger.info(
                        f"[RH-DEBUG] Component constraints: EAF_max={eaf_max:.2f}, DRI_min={dri_min:.2f}, DRI_max={dri_max:.2f}, per-hour_demand={max_val:.2f}"
                    )

                    if dri_min > 0 and max_val < dri_min:
                        logger.warning(
                            f"[RH-WARNING] min_demand ({max_val:.2f} MWh/h) is LESS than DRI_min_power ({dri_min:.2f} MWh/h). "
                            f"Optimizer will run DRI at minimum, forcing production to ~{dri_min * 1.09:.2f} MWh/h (approx steel equivalent). "
                            f"Consider: (A) Increase min_demand to {dri_min:.0f}+ MWh/h, or (B) Reduce DRI min_power in unit specs."
                        )

                    # Add per-hour minimum demand constraint
                    self.model.min_hourly_demand_param = pyo.Param(
                        range(n_steps),
                        initialize={i: min_demand_window[i] for i in range(n_steps)},
                    )

                    def min_hourly_demand_rule(m, t):
                        return (
                            m.dsm_blocks["eaf"].steel_output[t]
                            >= m.min_hourly_demand_param[t]
                        )

                    self.model.min_hourly_demand_con = pyo.Constraint(
                        range(n_steps), rule=min_hourly_demand_rule
                    )
                    logger.info(
                        f"[RH-STRATEGY] min_demand: Added per-hour minimums over {n_steps} steps, "
                        f"total minimum={total_min:.1f} MWh, per-hour range=[{min_val:.2f}, {max_val:.2f}]"
                    )

                # --- Strategy 3: Cost-optimized (no profile-based or demand-based constraints) ---
                else:
                    logger.info(
                        "[RH-STRATEGY] cost_optimized: No additional constraints (pure cost minimization)"
                    )

            self.define_objective_opt()

            # --- Apply soft constraints for profile-guided strategy ---
            load_profile_constraints_added = False
            if (
                self.technology == "steel_plant"
                and operation_strategy == "profile_guided"
                and full_horizon_load_profile is not None
            ):
                try:
                    # Extract window profile from FULL-HORIZON version
                    window_len = window_end - window_start
                    window_profile_raw = (
                        list(full_horizon_load_profile[window_start:window_end])
                        if hasattr(full_horizon_load_profile, "__getitem__")
                        else list(full_horizon_load_profile)[window_start:window_end]
                    )

                    # Pad/truncate to match window length
                    if len(window_profile_raw) < window_len and window_profile_raw:
                        window_profile_raw.extend(
                            [window_profile_raw[-1]]
                            * (window_len - len(window_profile_raw))
                        )
                    elif len(window_profile_raw) > window_len:
                        window_profile_raw = window_profile_raw[:window_len]

                    if len(window_profile_raw) > 0:
                        profile_sum = sum(window_profile_raw)
                        profile_max_val = max(window_profile_raw)

                        # Target total_power_input at each timestep:
                        # Use profile values directly as fraction of plant max_power
                        # Profile normalized (0-1) × max_power → target MW per hour
                        max_power = getattr(self, "max_power", 922.0)

                        if profile_max_val > 0 and profile_sum > 0:
                            target_production = {}
                            for t_idx in range(len(window_profile_raw)):
                                # Scale profile to power: profile[t] × max_power
                                target_production[t_idx] = (
                                    window_profile_raw[t_idx] * max_power
                                )

                            # Add deviation variables (positive and negative)
                            self.model.profile_dev_pos = pyo.Var(
                                self.model.time_steps, within=pyo.NonNegativeReals
                            )
                            self.model.profile_dev_neg = pyo.Var(
                                self.model.time_steps, within=pyo.NonNegativeReals
                            )

                            # Target production parameter
                            self.model.profile_target = pyo.Param(
                                self.model.time_steps,
                                initialize=target_production,
                                default=0.0,
                            )

                            # Deviation tracking: total_power_input[t] = target[t] + dev_pos[t] - dev_neg[t]
                            # Using total_power_input (continuous) instead of steel_output (batch-constrained)
                            @self.model.Constraint(self.model.time_steps)
                            def profile_deviation_con(m, t):
                                return (
                                    m.total_power_input[t]
                                    == m.profile_target[t]
                                    + m.profile_dev_pos[t]
                                    - m.profile_dev_neg[t]
                                )

                            # Penalty weight: controls profile adherence vs cost optimization
                            # deviation parameter inverts to penalty strength:
                            #   deviation=0.1 → strict (weight=100), deviation=0.9 → loose (weight ~11)
                            # Multiplied by 10 to make penalty meaningful relative to variable costs
                            penalty_weight = 10.0 / max(
                                self.load_profile_deviation, 0.01
                            )

                            # Delete existing objective and replace with penalized version
                            self.model.del_component(self.model.obj_rule_opt)

                            @self.model.Objective(sense=pyo.minimize)
                            def obj_rule_opt(m):
                                total_cost = pyo.quicksum(
                                    m.variable_cost[t] for t in m.time_steps
                                )
                                profile_penalty = penalty_weight * pyo.quicksum(
                                    m.profile_dev_pos[t] + m.profile_dev_neg[t]
                                    for t in m.time_steps
                                )
                                return total_cost + profile_penalty

                            load_profile_constraints_added = True

                            # Log profile targets
                            target_vals = [
                                target_production.get(t, 0)
                                for t in range(min(5, window_len))
                            ]
                            logger.info(
                                f"[LOAD-PROFILE-RH] Window {window_start}-{window_end}: "
                                f"Soft constraint active (penalty_weight={penalty_weight:.1f}, "
                                f"deviation={self.load_profile_deviation:.0%}, max_power={max_power})"
                            )
                            target_vals = [
                                target_production.get(t, 0)
                                for t in range(min(5, window_len))
                            ]
                            logger.info(
                                f"  Target power first 5h: {[f'{v:.0f}' for v in target_vals]} MWh"
                            )
                except Exception as e:
                    logger.warning(
                        f"[LOAD-PROFILE-RH] Failed to add soft profile constraint: {e}. Continuing without."
                    )

            instance = self.model.create_instance()

            # Solve — with load-profile fallback if infeasible
            try:
                results = self.solver.solve(instance, options={})
            except RuntimeError as e:
                if load_profile_constraints_added and self.technology == "steel_plant":
                    logger.warning(
                        f"[LOAD-PROFILE-RH] Solver error with profile penalty in window {window_start}-{window_end}: {str(e)[:80]}. "
                        "Removing profile penalty and re-solving."
                    )
                    # Remove profile-related components and re-solve with plain cost objective
                    for comp_name in [
                        "profile_deviation_con",
                        "profile_dev_pos",
                        "profile_dev_neg",
                        "profile_target",
                        "obj_rule_opt",
                    ]:
                        if hasattr(self.model, comp_name):
                            self.model.del_component(comp_name)

                    # Re-add plain objective
                    @self.model.Objective(sense=pyo.minimize)
                    def obj_rule_opt(m):
                        return pyo.quicksum(m.variable_cost[t] for t in m.time_steps)

                    instance = self.model.create_instance()
                    results = self.solver.solve(instance, options={})
                    load_profile_constraints_added = False
                else:
                    raise

            # Secondary infeasibility check (for solvers that don't raise)
            if (
                results.solver.status == SolverStatus.ok
                and results.solver.termination_condition
                == TerminationCondition.infeasible
                and load_profile_constraints_added
                and self.technology == "steel_plant"
            ):
                logger.warning(
                    f"[LOAD-PROFILE-RH] Profile penalty infeasible in window {window_start}-{window_end}; re-solving without."
                )
                for comp_name in [
                    "profile_deviation_con",
                    "profile_dev_pos",
                    "profile_dev_neg",
                    "profile_target",
                    "obj_rule_opt",
                ]:
                    if hasattr(self.model, comp_name):
                        self.model.del_component(comp_name)

                @self.model.Objective(sense=pyo.minimize)
                def obj_rule_opt(m):
                    return pyo.quicksum(m.variable_cost[t] for t in m.time_steps)

                instance = self.model.create_instance()
                results = self.solver.solve(instance, options={})

            if (results.solver.status == SolverStatus.ok) and (
                results.solver.termination_condition == TerminationCondition.optimal
            ):
                logger.debug("[RH-WINDOW-SOLVED] Optimal solution found")
            elif (
                results.solver.termination_condition == TerminationCondition.infeasible
            ):
                logger.warning("[RH-WINDOW-SOLVED] Window is infeasible")
            else:
                logger.warning(
                    "[RH-WINDOW-SOLVED] Solver status: %s / %s",
                    results.solver.status,
                    results.solver.termination_condition,
                )

            # --- Extract committed results ---
            n_commit = commit_end - window_start
            for local_t in range(n_commit):
                global_t = window_start + local_t
                power_val = pyo.value(instance.total_power_input[local_t])
                cost_val = pyo.value(instance.variable_cost[local_t])
                opt_power_window[local_t] = power_val
                var_cost_window[local_t] = cost_val

                if self.technology == "steel_plant":
                    try:
                        prod_val = pyo.value(
                            instance.dsm_blocks["eaf"].steel_output[local_t]
                        )
                    except (KeyError, AttributeError, TypeError):
                        prod_val = power_val
                else:
                    prod_val = power_val

                window_production += prod_val

                # Queue opt_power_requirement update for after restore
                _pending_opr_updates[global_t] = power_val

                if hasattr(self, "_rh_full_horizon_production"):
                    self._rh_full_horizon_production[global_t] = prod_val

            logger.info(
                f"[RH-MARKET] Committed production for this window: {window_production:.2f} MWh"
            )

            # --- Component-level data extraction ---
            if self.technology == "steel_plant":
                if not hasattr(self, "_component_operations"):
                    self._component_operations = []
                try:
                    for local_t in range(n_commit):
                        global_t = window_start + local_t
                        timestamp = saved_index[global_t]
                        eaf_power = eaf_steel = dri_power = dri_output = elec_power = (
                            h2_prod
                        ) = 0.0
                        try:
                            if hasattr(instance, "dsm_blocks"):
                                if "eaf" in instance.dsm_blocks:
                                    b = instance.dsm_blocks["eaf"]
                                    if hasattr(b, "power_in") and local_t in b.power_in:
                                        eaf_power = float(
                                            pyo.value(b.power_in[local_t])
                                        )
                                    if (
                                        hasattr(b, "steel_output")
                                        and local_t in b.steel_output
                                    ):
                                        eaf_steel = float(
                                            pyo.value(b.steel_output[local_t])
                                        )
                                if "dri_plant" in instance.dsm_blocks:
                                    b = instance.dsm_blocks["dri_plant"]
                                    if hasattr(b, "power_in") and local_t in b.power_in:
                                        dri_power = float(
                                            pyo.value(b.power_in[local_t])
                                        )
                                    if (
                                        hasattr(b, "dri_output")
                                        and local_t in b.dri_output
                                    ):
                                        dri_output = float(
                                            pyo.value(b.dri_output[local_t])
                                        )
                                if "electrolyser" in instance.dsm_blocks:
                                    b = instance.dsm_blocks["electrolyser"]
                                    if hasattr(b, "power_in") and local_t in b.power_in:
                                        elec_power = float(
                                            pyo.value(b.power_in[local_t])
                                        )
                                    if (
                                        hasattr(b, "hydrogen_out")
                                        and local_t in b.hydrogen_out
                                    ):
                                        h2_prod = float(
                                            pyo.value(b.hydrogen_out[local_t])
                                        )
                        except Exception as ex:
                            logger.debug(
                                f"Could not extract component data for t={local_t}: {ex}"
                            )
                        self._component_operations.append(
                            {
                                "global_t": global_t,
                                "timestamp": str(timestamp),
                                "eaf_power_input": eaf_power,
                                "eaf_steel_output": eaf_steel,
                                "dri_power_input": dri_power,
                                "dri_output": dri_output,
                                "electrolyser_power": elec_power,
                                "hydrogen_prod": h2_prod,
                            }
                        )
                except Exception as ex:
                    logger.debug(f"Error extracting component data: {ex}")

            # --- Transfer end-of-commit state to next window ---
            commit_local = n_commit - 1
            for tech_name in list(init_states.keys()):
                try:
                    block = instance.dsm_blocks[tech_name]
                    state_vars = {}
                    if hasattr(block, "soc"):
                        state_vars["soc"] = pyo.value(block.soc[commit_local])
                    if hasattr(block, "operational_status"):
                        state_vars["operational_status"] = int(
                            round(pyo.value(block.operational_status[commit_local]))
                        )
                    if state_vars:
                        init_states[tech_name] = state_vars
                except (KeyError, AttributeError):
                    pass

        finally:
            # --- Always restore unit-level attrs and full-horizon environment ---
            # This runs even if an exception was raised, preventing FastSeries
            # attributes (e.g. opt_power_requirement) from staying as numpy slices.
            self._restore_series_attrs(saved_attrs)
            self.index = saved_index
            if saved_model is not None:
                self.model = saved_model
            self.components = saved_components

        # --- Apply queued opt_power_requirement updates to the restored FastSeries ---
        for global_t, power_val in _pending_opr_updates.items():
            try:
                timestamp = saved_index[global_t]
                self.opt_power_requirement[timestamp] = power_val
            except (IndexError, KeyError, AttributeError, TypeError) as e:
                logger.debug(f"Could not update opt_power_requirement[{global_t}]: {e}")

        # Update tracking variables
        self._rh_window_start_idx = window_start
        self._rh_optimized_until_step = commit_end
        msg = (
            f"[RH-WINDOW-COMPLETE] Window optimization complete | "
            f"Committed production: {window_production:.2f} MWh | "
            f"Optimized until step: {commit_end} | "
            f"States carried over: {list(init_states.keys())}\n"
            f"{'=' * 100}"
        )
        print(msg)
        logger.info(msg)

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
            # Forecaster setup phase - always use full horizon
            # (rolling horizon will be triggered during market re-optimization)
            pass
        elif self._horizon_mode == "rolling_horizon":
            # Market optimization phase with rolling horizon - use it here
            print(
                f"\n[ROLLING-HORIZON] {self.id}: Starting rolling-horizon optimization "
                f"(look_ahead={self._rh_look_ahead}, commit={self._rh_commit}, step={self._rh_step})"
            )
            try:
                self._solve_rolling_horizon_opt()
                print(
                    f"[ROLLING-HORIZON] {self.id}: Rolling-horizon optimization complete\n"
                )
                return
            except Exception as rh_error:
                logger.warning(
                    f"[ROLLING-HORIZON] Rolling horizon optimization failed: {type(rh_error).__name__}. "
                    f"Falling back to full-horizon optimization."
                )
                print(
                    f"\n[ROLLING-HORIZON] {self.id}: Rolling-horizon optimization failed."
                )
                print("  Falling back to full-horizon method.\n")
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
            if switch_flex_off and self.technology == "steel_plant":
                logger.warning(
                    f"  Steel plant full-horizon solve is INFEASIBLE with demand constraint of {self.steel_demand} MWh"
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
        if self._horizon_mode == "rolling_horizon":
            if not hasattr(self, "_rh_full_horizon_production"):
                self._rh_full_horizon_production = [0.0] * len(self.index)

            # For steel plants, track steel_output; for others track power_input
            if self.technology == "steel_plant":
                try:
                    production_schedule = [
                        pyo.value(instance.dsm_blocks["eaf"].steel_output[t])
                        for t in instance.time_steps
                    ]
                except (KeyError, AttributeError):
                    production_schedule = opt_power_requirement
            else:
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
