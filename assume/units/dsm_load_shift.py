# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from collections.abc import Callable
from datetime import datetime

import matplotlib as mpl
import pandas as pd
import pyomo.environ as pyo
from matplotlib import pyplot as plt

from pyomo.opt import (
    SolverFactory,
    SolverStatus,
    TerminationCondition,
)

from assume.common.fast_pandas import FastSeries
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
        super().__init__(**kwargs)

        self.components = components

        self.initialize_solver()

    def initialize_solver(self, solver=None):
        # Define a solver
        solver = get_supported_solver(solver)
        if solver == "gurobi":
            self.solver_options = {"LogToConsole": 0, "OutputFlag": 0}
        elif solver == "appsi_highs":
            self.solver_options = {"output_flag": False, "log_to_console": False}
        else:
            self.solver_options = {}
        self.solver = SolverFactory(solver)

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
                    t: value for t, value in enumerate(self.electricity_price_flex)
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
                m.load_shift_neg[t] for t in m.time_steps
            )

            return maximise_load_shift

    def symmetric_flexible_block(self, model):
        """
        Cost-based CRM flexibility (rolling 4-hour blocks).
        Bids must be in integer multiples of `min_bid_MW` (e.g., 1 MW).
        Feasibility is enforced hour-by-hour via headroom/footroom constraints.
        """
        block_length = 4
        min_bid_MW = 1.0
        time_steps = list(sorted(model.time_steps))
        T = len(time_steps)
        possible_starts = [time_steps[i] for i in range(T - block_length + 1)]

        # ---- STATIC PLANT-WIDE MAX/MIN CAPACITY ----
        max_plant_capacity = 0.0
        min_plant_capacity = 0.0
        if self.has_heatpump:
            max_plant_capacity += model.dsm_blocks["heat_pump"].max_power
            min_plant_capacity += model.dsm_blocks["heat_pump"].min_power
        if self.has_heat_resistor:
            max_plant_capacity += model.dsm_blocks["heat_resistor"].max_power
            min_plant_capacity += model.dsm_blocks["heat_resistor"].min_power
        if self.has_boiler:
            boiler = self.components["boiler"]
            if boiler.fuel_type == "electricity":
                max_plant_capacity += model.dsm_blocks["boiler"].max_power
                min_plant_capacity += model.dsm_blocks["boiler"].min_power

        self.max_plant_capacity = max_plant_capacity
        self.min_plant_capacity = min_plant_capacity

        # Conservative big-M for integer counts
        M_blocks = int(pyo.ceil(max_plant_capacity / min_bid_MW))

        # ---- VARIABLES ----
        # integer number of min-bid units; enforces 1‑MW step size automatically
        model.n_blocks_up = pyo.Var(possible_starts, within=pyo.NonNegativeIntegers)
        model.n_blocks_down = pyo.Var(possible_starts, within=pyo.NonNegativeIntegers)

        # actual continuous volumes linked to the integer counts
        model.block_up = pyo.Var(possible_starts, within=pyo.NonNegativeReals)
        model.block_down = pyo.Var(possible_starts, within=pyo.NonNegativeReals)

        # direction binaries (only to enforce “up or down, not both”)
        model.block_is_bid_up = pyo.Var(possible_starts, within=pyo.Binary)
        model.block_is_bid_down = pyo.Var(possible_starts, within=pyo.Binary)

        # link to 1‑MW increments
        @model.Constraint(possible_starts)
        def block_up_step(m, t):
            return m.block_up[t] == min_bid_MW * m.n_blocks_up[t]

        @model.Constraint(possible_starts)
        def block_down_step(m, t):
            return m.block_down[t] == min_bid_MW * m.n_blocks_down[t]

        # activate counts only if that direction is chosen
        @model.Constraint(possible_starts)
        def up_count_active(m, t):
            return m.n_blocks_up[t] <= M_blocks * m.block_is_bid_up[t]

        @model.Constraint(possible_starts)
        def down_count_active(m, t):
            return m.n_blocks_down[t] <= M_blocks * m.block_is_bid_down[t]

        # only one direction per 4h block
        @model.Constraint(possible_starts)
        def symmetric_block(m, t):
            return m.block_is_bid_up[t] + m.block_is_bid_down[t] <= 1

        # ---- PER‑HOUR FEASIBILITY OVER THE 4‑H WINDOW ----
        model.block_up_window = pyo.ConstraintList()
        model.block_down_window = pyo.ConstraintList()

        for t0 in possible_starts:
            i0 = time_steps.index(t0)
            for k in range(block_length):
                tau = time_steps[i0 + k]

                total_power = 0.0
                if self.has_heatpump:
                    total_power += model.dsm_blocks["heat_pump"].power_in[tau]
                if self.has_heat_resistor:
                    total_power += model.dsm_blocks["heat_resistor"].power_in[tau]
                if self.has_boiler:
                    boiler = self.components["boiler"]
                    if boiler.fuel_type == "electricity":
                        total_power += model.dsm_blocks["boiler"].power_in[tau]

                # headroom / footroom in each hour limit the 4‑h block volume
                model.block_up_window.add(
                    model.block_up[t0] <= self.max_plant_capacity - total_power
                )
                model.block_down_window.add(
                    model.block_down[t0] <= total_power - self.min_plant_capacity
                )

        # ---- COST TOLERANCE ----
        model.cost_tolerance = pyo.Param(initialize=self.cost_tolerance)
        model.total_cost = pyo.Param(initialize=0.0, mutable=True)

        @model.Constraint()
        def total_cost_upper_limit(m):
            return pyo.quicksum(
                m.variable_cost[t] for t in m.time_steps
            ) <= m.total_cost * (1 + (m.cost_tolerance / 100.0))

        # ---- OBJECTIVE (example: maximize upward flexibility) ----
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
        print(peak_load_cap_value)
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
            if self.technology == "steel_plant":
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

        @model.Constraint(model.time_steps)
        def peak_threshold_constraint(m, t):
            """
            Ensures that the power input during peak periods does not exceed the peak threshold value.
            """
            if self.technology == "steel_plant":
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

            elif self.technology == "steam_generator_plant":
                total_power = 0
                if self.has_heatpump:
                    total_power += self.model.dsm_blocks["heat_pump"].power_in[t]
                if self.has_boiler:
                    boiler = self.components["boiler"]
                    if boiler.fuel_type == "electricity":
                        total_power += self.model.dsm_blocks["boiler"].power_in[t]
                return total_power <= peak_load_cap_value

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
        model.load_shift_pos = pyo.Var(model.time_steps, within=pyo.NonNegativeReals)
        model.load_shift_neg = pyo.Var(model.time_steps, within=pyo.NonNegativeReals)
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
            if self.technology == "steel_plant":
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
            elif self.technology == "steam_generator_plant":
                total_power = 0
                if self.has_heatpump:
                    total_power += self.model.dsm_blocks["heat_pump"].power_in[t]
                if self.has_boiler:
                    boiler = self.components["boiler"]
                    if boiler.fuel_type == "electricity":
                        total_power += self.model.dsm_blocks["boiler"].power_in[t]
                if self.has_heat_resistor:
                    total_power += self.model.dsm_blocks["heat_resistor"].power_in[t]
                return (
                    m.total_power_input[t] + m.load_shift_pos[t] - m.load_shift_neg[t]
                    == total_power
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

    def determine_optimal_operation_without_flex(self, switch_flex_off=True):
        """
        Determines the optimal operation of the steel plant without considering flexibility.
        """
        # create an instance of the model
        instance = self.model.create_instance()
        # switch the instance to the optimal mode by deactivating the flexibility constraints and objective
        if switch_flex_off:
            instance = self.switch_to_opt(instance)
        # solve the instance
        results = self.solver.solve(instance, options=self.solver_options)

        # Check solver status and termination condition
        if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal
        ):
            logger.debug("The model was solved optimally.")

            # Display the Objective Function Value
            objective_value = instance.obj_rule_opt()
            logger.debug("The value of the objective function is %s.", objective_value)

        elif results.solver.termination_condition == TerminationCondition.infeasible:
            logger.debug("The model is infeasible.")

        else:
            logger.debug("Solver Status: ", results.solver.status)
            logger.debug(
                "Termination Condition: ", results.solver.termination_condition
            )

        self.total_cost = sum(
            instance.variable_cost[t].value for t in instance.time_steps
        )
        print(f"Total variable cost: {self.total_cost:.2f}")

        opt_power_requirement = [
            pyo.value(instance.total_power_input[t]) for t in instance.time_steps
        ]
        self.opt_power_requirement = FastSeries(
            index=self.index, value=opt_power_requirement
        )

        self.total_cost = sum(
            instance.variable_cost[t].value for t in instance.time_steps
        )

        # Variable cost series
        variable_cost = [
            pyo.value(instance.variable_cost[t]) for t in instance.time_steps
        ]
        self.variable_cost_series = FastSeries(index=self.index, value=variable_cost)
        ###################################Plots and Dashboard excecution###############################################
        self.render_sankey_timeseries(instance, html_path="./outputs/sankey_timeseries.html",
                                step_stride=1, # use >1 to downsample frames if you have many timesteps
                                min_flow=1e-6  # hide tiny links for readability
                                )
        self.render_emissions_analytics(instance,
        baseline_instance=None,                     # optional: pass a "no-TES" or BAU instance for comparison
        html_path="./outputs/emissions_analytics.html",
        elec_ef_param_name="electricity_emission_factor"  # tCO2/MWh_e on the model (optional)
    )
        self.render_sankey_timeseries_1(instance, html_path="./outputs/sankey_timeseries_!.html",
                                step_stride=1, # use >1 to downsample frames if you have many timesteps
                                min_flow=1e-6  # hide tiny links for readability
                                )
        self.render_cement_dashboard(instance, html_path="./outputs/cement_dashboard.html")
        self.animated_tes_sankey(instance, html_path="./outputs/cement_tes_playback.html")
        self.render_storage_analytics(instance, html_path="./outputs/storage_analytics.html")

        # # ------------------- PLOT (preheater + calciner + kiln + TES + H2 routing) -------------------
        # mpl.rcParams.update(
        #     {
        #         "font.size": 13,
        #         "font.family": "serif",
        #         "axes.titlesize": 15,
        #         "axes.labelsize": 13,
        #         "legend.fontsize": 12,
        #         "lines.linewidth": 2,
        #         "axes.grid": True,
        #         "grid.linestyle": "--",
        #         "grid.alpha": 0.7,
        #         "figure.dpi": 120,
        #     }
        # )

        # time_steps = list(instance.time_steps)

        # # helper: pyo.value() -> 0.0 if var is uninitialized (common when that fuel path isn't used)
        # def safe_value(v):
        #     try:
        #         return pyo.value(v)
        #     except Exception:
        #         return 0.0

        # def series_or_none(var_block_attr):
        #     # returns list of values if attribute exists on the block, else None
        #     if not var_block_attr:
        #         return None
        #     return [safe_value(var_block_attr[t]) for t in instance.time_steps]

        # total_cumulative_thermal_output = sum(
        #     safe_value(instance.cumulative_thermal_output[t])
        #     for t in instance.time_steps
        # )
        # print(f"Total cumulative thermal output: {total_cumulative_thermal_output:.2f}")

        # # --- grab blocks if present
        # ph = (
        #     instance.dsm_blocks["preheater"]
        #     if "preheater" in instance.dsm_blocks
        #     else None
        # )
        # cc = (
        #     instance.dsm_blocks["calciner"]
        #     if "calciner" in instance.dsm_blocks
        #     else None
        # )
        # rk = instance.dsm_blocks["kiln"] if "kiln" in instance.dsm_blocks else None
        # el = (
        #     instance.dsm_blocks["electrolyser"]
        #     if "electrolyser" in instance.dsm_blocks
        #     else None
        # )
        # ts = (
        #     instance.dsm_blocks["thermal_storage"]
        #     if "thermal_storage" in instance.dsm_blocks
        #     else None
        # )

        # # --- inputs/outputs (safe to read regardless of mode)
        # preheater_power_in = (
        #     series_or_none(getattr(ph, "power_in", None)) if ph else None
        # )
        # preheater_ng_in = (
        #     series_or_none(getattr(ph, "natural_gas_in", None)) if ph else None
        # )
        # preheater_coal_in = series_or_none(getattr(ph, "coal_in", None)) if ph else None
        # preheater_heat_out = (
        #     series_or_none(getattr(ph, "heat_out", None)) if ph else None
        # )
        # preheater_wh_in = (
        #     series_or_none(getattr(ph, "external_heat_in", None)) if ph else None
        # )

        # calciner_power_in = (
        #     series_or_none(getattr(cc, "power_in", None)) if cc else None
        # )
        # calciner_ng_in = (
        #     series_or_none(getattr(cc, "natural_gas_in", None)) if cc else None
        # )
        # calciner_coal_in = series_or_none(getattr(cc, "coal_in", None)) if cc else None
        # calciner_h2_in = (
        #     series_or_none(getattr(cc, "hydrogen_in", None)) if cc else None
        # )
        # calciner_heat_out = (
        #     series_or_none(getattr(cc, "heat_out", None)) if cc else None
        # )
        # # NEW: effective heat into the calciner (after TES charge/discharge)
        # calciner_eff_in = (
        #     series_or_none(getattr(cc, "effective_heat_in", None)) if cc else None
        # )

        # kiln_power_in = series_or_none(getattr(rk, "power_in", None)) if rk else None
        # kiln_ng_in = series_or_none(getattr(rk, "natural_gas_in", None)) if rk else None
        # kiln_coal_in = series_or_none(getattr(rk, "coal_in", None)) if rk else None
        # kiln_h2_in = series_or_none(getattr(rk, "hydrogen_in", None)) if rk else None
        # kiln_heat_out = series_or_none(getattr(rk, "heat_out", None)) if rk else None

        # # Electrolyser
        # el_power = series_or_none(getattr(el, "power_in", None)) if el else None
        # el_h2 = series_or_none(getattr(el, "hydrogen_out", None)) if el else None

        # # Thermal Storage (TES)
        # ts_charge = (
        #     series_or_none(getattr(ts, "charge", None)) if ts else None
        # )  # [MWth]
        # ts_discharge = (
        #     series_or_none(getattr(ts, "discharge", None)) if ts else None
        # )  # [MWth]
        # ts_soc = series_or_none(getattr(ts, "soc", None)) if ts else None  # [MWhth]
        # ts_power_in = (
        #     series_or_none(getattr(ts, "power_in", None)) if ts else None
        # )  # [MWe]

        # # demand (per-step) if available (material demand; used in CSV, optional to plot elsewhere)
        # demand_series = (
        #     [
        #         safe_value(instance.clinker_demand_per_time_step[t])
        #         for t in instance.time_steps
        #     ]
        #     if hasattr(instance, "clinker_demand_per_time_step")
        #     else None
        # )

        # # H2 routing (from electrolyser to units)
        # h2_to_cc = (
        #     [safe_value(instance.h2_to_calciner[t]) for t in instance.time_steps]
        #     if hasattr(instance, "h2_to_calciner")
        #     else None
        # )
        # h2_to_rk = (
        #     [safe_value(instance.h2_to_kiln[t]) for t in instance.time_steps]
        #     if hasattr(instance, "h2_to_kiln")
        #     else None
        # )

        # # prices
        # electricity_price = (
        #     [safe_value(instance.electricity_price[t]) for t in instance.time_steps]
        #     if hasattr(instance, "electricity_price")
        #     else None
        # )
        # natural_gas_price = (
        #     [safe_value(instance.natural_gas_price[t]) for t in instance.time_steps]
        #     if hasattr(instance, "natural_gas_price")
        #     else None
        # )
        # coal_price = (
        #     [safe_value(instance.coal_price[t]) for t in instance.time_steps]
        #     if hasattr(instance, "coal_price")
        #     else None
        # )
        # hydrogen_price = (
        #     [safe_value(instance.hydrogen_price[t]) for t in instance.time_steps]
        #     if hasattr(instance, "hydrogen_price")
        #     else None
        # )

        # fig, axs = plt.subplots(
        #     2, 1, figsize=(10, 8), sharex=True, constrained_layout=True
        # )

        # # ---------- Top: Unit Inputs & Energy Prices ----------
        # def plot_if_nonzero(ax, x, y, label, color, style="-", eps=1e-9):
        #     if y is None:
        #         return None
        #     if any(abs(v) > eps for v in y):
        #         return ax.plot(x, y, label=label, color=color, linestyle=style)[0]
        #     return None

        # # Preheater inputs
        # h1 = plot_if_nonzero(
        #     axs[0], time_steps, preheater_power_in, "Preheater Power [MWₑ]", "C1"
        # )
        # h2 = plot_if_nonzero(
        #     axs[0], time_steps, preheater_ng_in, "Preheater NG [MWₜₕ]", "C0"
        # )
        # h3 = plot_if_nonzero(
        #     axs[0], time_steps, preheater_coal_in, "Preheater Coal [MWₜₕ]", "C7"
        # )
        # # Waste heat to preheater
        # h8 = plot_if_nonzero(
        #     axs[0], time_steps, preheater_wh_in, "WH → Preheater [MWₜₕ]", "C15", ":"
        # )

        # # Calciner inputs
        # h4 = plot_if_nonzero(
        #     axs[0], time_steps, calciner_power_in, "Calciner Power [MWₑ]", "C5"
        # )
        # h5 = plot_if_nonzero(
        #     axs[0], time_steps, calciner_ng_in, "Calciner NG [MWₜₕ]", "C6"
        # )
        # h6 = plot_if_nonzero(
        #     axs[0], time_steps, calciner_coal_in, "Calciner Coal [MWₜₕ]", "C8"
        # )
        # h7 = plot_if_nonzero(
        #     axs[0], time_steps, calciner_h2_in, "Calciner H₂ [MWₕ₂]", "C9"
        # )

        # # Kiln inputs
        # h9 = plot_if_nonzero(
        #     axs[0], time_steps, kiln_power_in, "Kiln Power [MWₑ]", "C11"
        # )
        # h10 = plot_if_nonzero(axs[0], time_steps, kiln_ng_in, "Kiln NG [MWₜₕ]", "C12")
        # h11 = plot_if_nonzero(
        #     axs[0], time_steps, kiln_coal_in, "Kiln Coal [MWₜₕ]", "C13"
        # )
        # h12 = plot_if_nonzero(axs[0], time_steps, kiln_h2_in, "Kiln H₂ [MWₕ₂]", "C14")

        # # Electrolyser + H2 routing
        # h_elP = plot_if_nonzero(
        #     axs[0], time_steps, el_power, "Electrolyser Power [MWₑ]", "C4", "--"
        # )
        # h_elH = plot_if_nonzero(
        #     axs[0], time_steps, el_h2, "Electrolyser H₂ [MWₕ₂]", "C10", "--"
        # )
        # h_rtC = plot_if_nonzero(
        #     axs[0], time_steps, h2_to_cc, "H₂ → Calciner [MWₕ₂]", "C17", ":"
        # )
        # h_rtK = plot_if_nonzero(
        #     axs[0], time_steps, h2_to_rk, "H₂ → Kiln [MWₕ₂]", "C18", ":"
        # )

        # # TES electric heater
        # h_tsp = plot_if_nonzero(
        #     axs[0], time_steps, ts_power_in, "TES Heater Power [MWₑ]", "C19", "-."
        # )

        # axs[0].set_ylabel("Inputs")
        # axs[0].set_title("Unit Inputs, H₂ Routing & Prices")
        # axs[0].grid(True, which="both", axis="both")

        # # secondary axis for prices
        # ln2 = []
        # axp = axs[0].twinx()
        # if electricity_price:
        #     ln2 += axp.plot(
        #         time_steps,
        #         electricity_price,
        #         label="Elec Price [€/MWhₑ]",
        #         color="C2",
        #         linestyle="--",
        #     )
        # if natural_gas_price:
        #     ln2 += axp.plot(
        #         time_steps,
        #         natural_gas_price,
        #         label="NG Price [€/MWhₜₕ]",
        #         color="C3",
        #         linestyle="--",
        #     )
        # if coal_price:
        #     ln2 += axp.plot(
        #         time_steps,
        #         coal_price,
        #         label="Coal Price [€/MWhₜₕ]",
        #         color="C6",
        #         linestyle="-.",
        #     )
        # if hydrogen_price:
        #     ln2 += axp.plot(
        #         time_steps,
        #         hydrogen_price,
        #         label="H₂ Price [€/MWhₜₕ]",
        #         color="C8",
        #         linestyle=":",
        #     )
        # axp.set_ylabel("Energy Price", color="gray")
        # axp.tick_params(axis="y", labelcolor="gray")

        # # extend legend lines
        # lines = [
        #     l
        #     for l in [
        #         h1,
        #         h2,
        #         h3,
        #         h4,
        #         h5,
        #         h6,
        #         h7,
        #         h8,
        #         h9,
        #         h10,
        #         h11,
        #         h12,
        #         h_elP,
        #         h_elH,
        #         h_rtC,
        #         h_rtK,
        #         h_tsp,
        #     ]
        #     if l is not None
        # ] + ln2
        # labels = [l.get_label() for l in lines]
        # if lines:
        #     axs[0].legend(lines, labels, loc="upper left", frameon=True)

        # # ---------- Bottom: Thermal Storage Operation ----------
        # axs[1].clear()
        # pl0 = plot_if_nonzero(
        #     axs[1], time_steps, ts_charge, "Storage Charge [MWₜₕ]", "C2", "-"
        # )
        # pl1 = plot_if_nonzero(
        #     axs[1], time_steps, ts_discharge, "Storage Discharge [MWₜₕ]", "C3", "--"
        # )

        # # SOC as filled area if available
        # if ts_soc and any(abs(v) > 1e-9 for v in ts_soc):
        #     axs[1].fill_between(
        #         time_steps, ts_soc, 0, alpha=0.25, label="Storage SOC [MWhₜₕ]"
        #     )

        # axs[1].set_ylabel("MWₜₕ / MWhₜₕ")
        # axs[1].set_title("Thermal Storage Operation")
        # axs[1].grid(True, which="both", axis="both")
        # # Build legend items that exist
        # bottom_lines = [l for l in [pl0, pl1] if l is not None]
        # if bottom_lines:
        #     axs[1].legend(loc="upper right", frameon=True)

        # axs[1].set_xlabel("Time Step")

        # plt.tight_layout()
        # plt.show()
        # # plt.savefig("./examples/outputs/opt_cement_operation.png", dpi=300, bbox_inches="tight")

        # # ------------- CSV dataframe -------------
        # def nz_or_none(s):
        #     return s if (s and any(abs(v) > 1e-9 for v in s)) else None

        # df = pd.DataFrame(
        #     {
        #         "Time Step": time_steps,
        #         "PH Power [MW_e]": nz_or_none(preheater_power_in),
        #         "PH NG [MW_th]": nz_or_none(preheater_ng_in),
        #         "PH Coal [MW_th]": nz_or_none(preheater_coal_in),
        #         "PH Heat Out [MW_th]": nz_or_none(preheater_heat_out),
        #         "PH WH In [MW_th]": nz_or_none(preheater_wh_in),
        #         "CC Power [MW_e]": nz_or_none(calciner_power_in),
        #         "CC NG [MW_th]": nz_or_none(calciner_ng_in),
        #         "CC Coal [MW_th]": nz_or_none(calciner_coal_in),
        #         "CC H2 [MW_th]": nz_or_none(calciner_h2_in),
        #         "CC Heat Out [MW_th]": nz_or_none(calciner_heat_out),
        #         "CC Effective Heat In [MW_th]": nz_or_none(
        #             calciner_eff_in
        #         ),  # NEW (post‑TES)
        #         "Kiln Power [MW_e]": nz_or_none(kiln_power_in),
        #         "Kiln NG [MW_th]": nz_or_none(kiln_ng_in),
        #         "Kiln Coal [MW_th]": nz_or_none(kiln_coal_in),
        #         "Kiln H2 [MW_th]": nz_or_none(kiln_h2_in),
        #         "Kiln Heat Out [MW_th]": nz_or_none(kiln_heat_out),
        #         "Electrolyser Power [MW_e]": nz_or_none(el_power),
        #         "Electrolyser Hydrogen [MW_h2]": nz_or_none(el_h2),
        #         "H2 to Calciner [MW_h2]": nz_or_none(h2_to_cc),
        #         "H2 to Kiln [MW_h2]": nz_or_none(h2_to_rk),
        #         "TES Charge [MW_th]": nz_or_none(ts_charge),
        #         "TES Discharge [MW_th]": nz_or_none(ts_discharge),
        #         "TES SOC [MWh_th]": nz_or_none(ts_soc),
        #         "TES Heater Power [MW_e]": nz_or_none(ts_power_in),
        #         "Clinker Demand [t/h]": demand_series if demand_series else None,
        #     }
        # )
        # df.to_csv("./examples/outputs/opt_cement_timeseries.csv", index=False)

    def determine_optimal_operation_with_flex(self):
        """
        Determines the optimal operation of the steel plant without considering flexibility.
        """
        # create an instance of the model
        instance = self.model.create_instance()
        # switch the instance to the flexibility mode by deactivating the optimal constraints and objective
        instance = self.switch_to_flex(instance)
        # solve the instance
        results = self.solver.solve(instance, options=self.solver_options)

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

        if hasattr(instance, "total_power_input_constraint_with_flex"):
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
            instance.total_power_input[t].fix(self.opt_power_requirement.iloc[t])
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
