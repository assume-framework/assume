# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from collections.abc import Callable
from datetime import datetime

import pyomo.environ as pyo
from pyomo.opt import (
    SolverFactory,
    SolverStatus,
    TerminationCondition,
    check_available_solvers,
)

from assume.common.fast_pandas import FastSeries
from assume.units.dst_components import get_technology_class

SOLVERS = ["appsi_highs", "gurobi", "glpk", "cbc", "cplex"]

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
        solvers = check_available_solvers(*SOLVERS)

        # raise an error if no solver is available
        if not solvers:
            raise ValueError(
                f"None of {SOLVERS} are available. Install one of them to proceed."
            )

        solver = solver if solver in solvers else solvers[0]
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
        """
        components = self.components.copy()
        self.model.dsm_blocks = pyo.Block(list(components.keys()))

        for technology, component_data in components.items():
            try:
                # Dynamically get the class for the technology
                component_class = get_technology_class(technology)

                # Instantiate the component with the required parameters
                component_instance = component_class(
                    time_steps=self.model.time_steps, **component_data
                )

                # Add the component to the components dictionary
                self.components[technology] = component_instance

                # Optional external range binding
                range_param_name = f"{technology}_range"
                external_range = getattr(self.model, range_param_name, None)

                if external_range is not None:
                    component_instance.add_to_model(
                        self.model,
                        self.model.dsm_blocks[technology],
                        external_range=external_range,
                    )
                else:
                    component_instance.add_to_model(
                        self.model, self.model.dsm_blocks[technology]
                    )

            except ValueError as e:
                logger.error(f"Error initializing component {technology}: {e}")
                raise

    def setup_model(self, presolve=True):
        # Initialize the Pyomo model
        # along with optimal and flexibility constraints
        # and the objective functions

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
        self.model.evs = pyo.Set(
        initialize=[
            ev for ev in self.components if ev.startswith("electric_vehicle")
        ],
        ordered=True,
        doc="Set of electric vehicles"
        )

        self.model.charging_stations = pyo.Set(
        initialize=[
            cs for cs in self.components if cs.startswith("charging_station")
        ],
        ordered=True,
        doc="Set of charging stations"
        )
        # self.model.evs = pyo.Set(initialize=[ev for ev in self.components if ev.startswith("electric_vehicle")])
        # self.model.charging_stations = pyo.Set(initialize=[cs for cs in self.components if cs.startswith("charging_station")])
        # self.model.evs = pyo.Set(
        #     initialize=[
        #         ev for ev in self.components if ev.startswith("electric_vehicle")
        #     ],
        #     ordered=True,
        # )

        # self.model.charging_stations = pyo.Set(
        #     initialize=[
        #         cs
        #         for cs, _ in sorted(
        #             (
        #                 (cs, self.components[cs]["max_power"])
        #                 for cs in self.components
        #                 if cs.startswith("charging_station")
        #             ),
        #             key=lambda x: -x[1],  # descending by max_power
        #         )
        #     ],
        #     ordered=True,
        # )

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
                # Building components (e.g., heat_pump, boiler, pv_plant, generic_storage)
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

            elif self.technology == "bus_depot":
                cs_discharge = sum(
                    m.dsm_blocks[cs].discharge[t]
                    for cs in m.dsm_blocks
                    if cs.startswith("charging_station")
                )
                return (
                    m.total_power_input[t] + m.load_shift_pos[t] - m.load_shift_neg[t]
                    == cs_discharge
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
                # Building components (e.g., heat_pump, boiler, pv_plant, generic_storage)
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

            elif self.technology == "bus_depot":
                cs_discharge = sum(
                    m.dsm_blocks[cs].discharge[t]
                    for cs in m.dsm_blocks
                    if cs.startswith("charging_station")
                )
                return (
                    m.total_power_input[t] + m.load_shift_pos[t] - m.load_shift_neg[t]
                    == cs_discharge
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
                # Building components (e.g., heat_pump, boiler, pv_plant, generic_storage)
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

            elif self.technology == "bus_depot":
                cs_discharge = sum(
                    m.dsm_blocks[cs].discharge[t]
                    for cs in m.dsm_blocks
                    if cs.startswith("charging_station")
                )
                return (
                    m.total_power_input[t] + m.load_shift_pos[t] - m.load_shift_neg[t]
                    == cs_discharge
                )

        @model.Constraint(model.time_steps)
        def peak_threshold_constraint(m, t):
            """
            Ensures that the power input during peak periods does not exceed the peak threshold value.
            """
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
                        self.model.dsm_blocks["electrolyser"].power_in[t]
                        + self.model.dsm_blocks["eaf"].power_in[t]
                        + self.model.dsm_blocks["dri_plant"].power_in[t]
                        <= peak_load_cap_value
                    )
                else:
                    return (
                        m.dsm_blocks["eaf"].power_in[t]
                        + m.dsm_blocks["dri_plant"].power_in[t]
                        <= peak_load_cap_value
                    )

            elif self.technology == "bus_depot":
                cs_discharge = sum(
                    m.dsm_blocks[cs].discharge[t]
                    for cs in m.dsm_blocks
                    if cs.startswith("charging_station")
                )
                return cs_discharge <= peak_load_cap_value

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

        # Power input constraint with flexibility based on congestion
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
                # Building components (e.g., heat_pump, boiler, pv_plant, generic_storage)
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

            elif self.technology == "bus_depot":
                cs_discharge = sum(
                    m.dsm_blocks[cs].discharge[t]
                    for cs in m.dsm_blocks
                    if cs.startswith("charging_station")
                )
                return (
                    m.total_power_input[t] + m.load_shift_pos[t] - m.load_shift_neg[t]
                    == cs_discharge
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
    # PLOTTING SECTION - Technology specific plots
        if self.technology == "bus_depot":
            self._plot_bus_depot_optimization(instance)
        elif self.technology == "steel_plant":
            self._plot_steel_plant_optimization(instance)
        elif self.technology == "building":
            self._plot_building_optimization(instance)
        # Add more technology-specific plots as needed

    def _plot_bus_depot_optimization(self, instance):
        """
        Bus depot specific plotting logic
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
        
        if not (hasattr(instance, 'evs') and hasattr(instance, 'charging_stations')):
            logger.warning("Bus depot plotting requires EVs and charging stations")
            return
        
        time_steps = list(instance.time_steps)
        evs = list(instance.evs)
        charging_stations = list(instance.charging_stations)
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 12))
        
        # 1. EV Status Matrix (Idle/Queue/Charging)
        status_matrix = np.zeros((len(evs), len(time_steps)))
        
        for i, ev in enumerate(evs):
            for j, t in enumerate(time_steps):
                # Get availability
                ev_availability = getattr(instance, f"{ev}_availability", None)
                if ev_availability is None or pyo.value(ev_availability[t]) == 0:
                    status_matrix[i, j] = 0  # Not available (driving)
                    continue
                
                # Check if charging
                is_charging = False
                for cs in charging_stations:
                    if pyo.value(instance.is_assigned[ev, cs, t]) > 0.5:
                        status_matrix[i, j] = 2  # Charging
                        is_charging = True
                        break
                
                # Check if in queue
                if not is_charging and hasattr(instance, 'in_queue'):
                    if pyo.value(instance.in_queue[ev, t]) > 0.5:
                        status_matrix[i, j] = 1  # In queue
        
        # Custom colormap: 0=Gray(Driving), 1=Yellow(Queue), 2=Green(Charging)
        colors = ['lightgray', 'yellow', 'green']
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        
        im = ax1.imshow(status_matrix, aspect='auto', cmap=cmap, vmin=0, vmax=2)
        ax1.set_yticks(np.arange(len(evs)))
        ax1.set_yticklabels(evs)
        ax1.set_xlabel('Time Steps')
        ax1.set_title('EV Status (Gray=Driving, Yellow=Queue, Green=Charging)')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='lightgray', label='Driving'),
            mpatches.Patch(color='yellow', label='In Queue'),
            mpatches.Patch(color='green', label='Charging')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # 2. Charging Station Utilization
        for cs in charging_stations:
            cs_discharge = []
            cs_max_power = pyo.value(instance.dsm_blocks[cs].max_power)
            
            for t in time_steps:
                discharge = pyo.value(instance.dsm_blocks[cs].discharge[t])
                cs_discharge.append(discharge)
            
            ax2.plot(time_steps, cs_discharge, label=f'{cs}', linewidth=2)
            ax2.axhline(y=cs_max_power, color='gray', linestyle='--', alpha=0.5)
        
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Discharge Power (kW)')
        ax2.set_title('Charging Station Power Output')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. EV Charging Power
        for ev in evs[:5]:  # Limit to first 5 EVs for clarity
            if ev in instance.dsm_blocks:
                ev_charge = [pyo.value(instance.dsm_blocks[ev].charge[t]) for t in time_steps]
                ax3.plot(time_steps, ev_charge, label=ev, alpha=0.8)
        
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Charging Power (kW)')
        ax3.set_title('EV Charging Power (First 5 EVs)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. EV State of Charge
        for ev in evs[:5]:  # Limit to first 5 EVs for clarity
            if ev in instance.dsm_blocks and hasattr(instance.dsm_blocks[ev], 'soc'):
                soc = [pyo.value(instance.dsm_blocks[ev].soc[t]) for t in time_steps]
                max_capacity = pyo.value(instance.dsm_blocks[ev].max_capacity)
                soc_percentage = [s/max_capacity * 100 for s in soc]
                ax4.plot(time_steps, soc_percentage, label=ev, marker='o', markersize=3)
        
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('State of Charge (%)')
        ax4.set_title('EV Battery Status (First 5 EVs)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 105])
        
        # 5. Queue Length Over Time
        queue_length = []
        for t in time_steps:
            if hasattr(instance, 'in_queue'):
                count = sum(pyo.value(instance.in_queue[ev, t]) for ev in evs)
            else:
                count = 0
            queue_length.append(count)
        
        ax5.bar(time_steps, queue_length, color='orange', alpha=0.7)
        ax5.set_xlabel('Time Steps')
        ax5.set_ylabel('Number of EVs Waiting')
        ax5.set_title('Queue Length Over Time')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Marginal Cost Analysis
        electricity_prices = [pyo.value(instance.electricity_price[t]) for t in time_steps]
        total_power = [pyo.value(instance.total_power_input[t]) for t in time_steps]
        
        # Calculate marginal cost (electricity price * power consumption)
        marginal_costs = [price * power for price, power in zip(electricity_prices, total_power)]
        
        # Create dual y-axis plot
        ax6_twin = ax6.twinx()
        
        # Plot electricity price on left axis
        line1 = ax6.plot(time_steps, electricity_prices, 'b-', label='Electricity Price (€/MWh)', linewidth=2)
        ax6.set_xlabel('Time Steps')
        ax6.set_ylabel('Electricity Price (€/MWh)', color='b')
        ax6.tick_params(axis='y', labelcolor='b')
        
        # Plot marginal cost on right axis
        line2 = ax6_twin.plot(time_steps, marginal_costs, 'r-', label='Marginal Cost (€)', linewidth=2)
        ax6_twin.set_ylabel('Marginal Cost (€)', color='r')
        ax6_twin.tick_params(axis='y', labelcolor='r')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc='upper left')
        
        ax6.set_title('Electricity Price and Marginal Cost')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        self._print_bus_depot_summary(instance, evs, charging_stations, time_steps)

    def _print_bus_depot_summary(self, instance, evs, charging_stations, time_steps):
        """Print optimization summary for bus depot"""
        print("\n" + "="*60)
        print("BUS DEPOT OPTIMIZATION SUMMARY")
        print("="*60)
        
        total_cost = sum(pyo.value(instance.variable_cost[t]) for t in time_steps)
        print(f"Total Cost: ${total_cost:.2f}")
        
        ##avg_power = np.mean([pyo.value(instance.total_power_input[t]) for t in time_steps])
        max_power = max([pyo.value(instance.total_power_input[t]) for t in time_steps])
        ##print(f"Average Power: {avg_power:.2f} kW")
        print(f"Peak Power: {max_power:.2f} kW")
        
        # CS Utilization
        print("\nCharging Station Utilization:")
        for cs in charging_stations:
            total_energy = sum(
                pyo.value(instance.dsm_blocks[cs].discharge[t]) for t in time_steps
            )
            max_possible = pyo.value(instance.dsm_blocks[cs].max_power) * len(time_steps)
            utilization = (total_energy / max_possible * 100) if max_possible > 0 else 0
            print(f"  {cs}: {total_energy:.1f} kWh delivered, {utilization:.1f}% capacity utilization")
        
        print("="*60 + "\n")
        """
        Plots the states of electric vehicles (Charging / Queued / Idle) over time.
        """
        
        # # Plot
        # time_steps = list(instance.time_steps)

        # # Plot EVs and Charging Stations
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # for block_name in instance.dsm_blocks:
        #     block = instance.dsm_blocks[block_name]

        #     if block_name.startswith("electric_vehicle"):
        #         ev_charge = [pyo.value(block.charge[t]) for t in time_steps]
        #         ev_discharge = [pyo.value(block.discharge[t]) for t in time_steps]
        #         ev_soc = [pyo.value(block.soc[t]) for t in time_steps]

        #         ax1.plot(
        #             time_steps,
        #             ev_charge,
        #             label=f"{block_name} Charge",
        #             linestyle="solid",
        #         )
        #         ax1.plot(
        #             time_steps,
        #             [-v for v in ev_discharge],
        #             label=f"{block_name} Discharge",
        #             linestyle="dashed",
        #         )
        #         ax1.plot(
        #             time_steps, ev_soc, label=f"{block_name} SOC", linestyle="dotted"
        #         )

        #     elif block_name.startswith("charging_station"):
        #         cs_discharge = [pyo.value(block.discharge[t]) for t in time_steps]
        #         ax2.plot(
        #             time_steps,
        #             cs_discharge,
        #             label=f"{block_name} Discharge",
        #             linestyle="solid",
        #         )

        # ax1.set_title("Electric Vehicle Charging Behavior")
        # ax1.set_ylabel("Power / SOC")
        # ax1.legend()

        # ax2.set_title("Charging Station Discharge Behavior")
        # ax2.set_xlabel("Time Steps")
        # ax2.set_ylabel("Discharge Power")
        # ax2.legend()

        # plt.tight_layout()
        # plt.show()

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
        # Deactivate the flexibility objective if it exists
        # if hasattr(instance, "obj_rule_flex"):
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
