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

SOLVERS = ["gurobi","appsi_highs",  "glpk", "cbc", "cplex"]

logger = logging.getLogger(__name__)


class DSMFlex:
    # Unified constants for all DSM flexibility implementations
    BIG_M = 1e6
    EPSILON = 1e-3
    
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
        "bidirectional_prosumer_renewable_discharge": lambda self, model: self.bidirectional_prosumer_renewable_discharge(
            model,
        ),
        "symmetric_flexible_block": lambda self, model: self.symmetric_flexible_block(
            model
        ),
    }

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
            
        elif self.objective == "max_net_income":

            @self.model.Objective(sense=pyo.maximize)
            def obj_rule_opt(m):
                """
                Maximizes the total net income over all time steps with emergency incentive and FCR revenue.
                """
                net_income = pyo.quicksum(
                    self.model.net_income[t] for t in self.model.time_steps
                )
                
                # Add emergency incentive calculation (incentive*emergency_signal*ev_discharge)
            ##    emergency_incentive = 0
            ##    if hasattr(self.model, 'emergency_signal') and hasattr(self.model, 'incentive'):
            ##        for t in self.model.time_steps:
            ##            for ev in getattr(self.model, 'evs', []):
            ##                if ev in self.model.dsm_blocks and hasattr(self.model.dsm_blocks[ev], 'discharge'):
            ##                    emergency_incentive += (
            ##                        self.model.incentive[t] * 
            ##                        self.model.emergency_signal[t] * 
            ##                        self.model.dsm_blocks[ev].discharge[t]
            ##                    )

                # Add FCR revenue if available (for prosumer depots)
             ##   fcr_revenue = 0
            ##    if hasattr(self.model, 'fcr_revenue'):
             ##       fcr_revenue = self.model.fcr_revenue
                
                return net_income ##+ emergency_incentive + fcr_revenue

        elif self.objective == "bidirectional_prosumer_emergency_discharge":

            @self.model.Objective(sense=pyo.maximize)
            def obj_rule_opt(m):
                """
                Maximizes emergency discharge for bidirectional prosumer operation with emergency incentive and FCR revenue.
                Similar to net income but optimized for discharge scenarios.
                """
                # Use the same net income calculation as a base
                net_income = pyo.quicksum(
                    self.model.net_income[t] for t in self.model.time_steps
                )
                
                # Add emergency incentive calculation (incentive*emergency_signal*ev_discharge)
                emergency_incentive = 0
                if hasattr(self.model, 'emergency_signal') and hasattr(self.model, 'incentive'):
                    for t in self.model.time_steps:
                        for ev in getattr(self.model, 'evs', []):
                            if ev in self.model.dsm_blocks and hasattr(self.model.dsm_blocks[ev], 'discharge'):
                                emergency_incentive += (
                                    self.model.incentive[t] * 
                                    self.model.emergency_signal[t] * 
                                    self.model.dsm_blocks[ev].discharge[t]
                                )

                # Add FCR revenue if available (for prosumer depots)
               ## fcr_revenue = 0
              ##  if hasattr(self.model, 'fcr_revenue'):
               ##     fcr_revenue = self.model.fcr_revenue
                
                return net_income + emergency_incentive #### + fcr_revenue

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
            return m.load_shift_pos[t] <= (1 - m.shift_indicator[t]) * self.BIG_M

        @model.Constraint(model.time_steps)
        def flex_constraint_lower(m, t):
            return m.load_shift_neg[t] <= m.shift_indicator[t] * self.BIG_M

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
            return m.load_shift_pos[t] <= (1 - m.shift_indicator[t]) * self.BIG_M

        @model.Constraint(model.time_steps)
        def flex_constraint_lower(m, t):
            return m.load_shift_neg[t] <= m.shift_indicator[t] * self.BIG_M

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
            return m.load_shift_pos[t] <= (1 - m.shift_indicator[t]) * self.BIG_M

        @model.Constraint(model.time_steps)
        def flex_constraint_lower(m, t):
            return m.load_shift_neg[t] <= m.shift_indicator[t] * self.BIG_M

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
            return m.load_shift_pos[t] <= (1 - m.shift_indicator[t]) * self.BIG_M

        @model.Constraint(model.time_steps)
        def flex_constraint_lower(m, t):
            return m.load_shift_neg[t] <= m.shift_indicator[t] * self.BIG_M

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
                ev_charge = sum(
                    m.dsm_blocks[ev].discharge[t]
                    for ev in m.dsm_blocks
                    if ev.startswith("electric_vehicle")
                )
                return (
                    m.total_power_input[t] + m.load_shift_pos[t] - m.load_shift_neg[t]
                    == ev_charge
                )

        @self.model.Objective(sense=pyo.maximize)
        def obj_rule_flex(m):
            """
            Maximizes the load increase over all time steps based on renewable surplus.
            """
            maximise_renewable_utilisation = pyo.quicksum(
                m.load_shift_pos[t] * m.renewable_signal[t] for t in m.time_steps
            )
           ## maximise_feedin_utilisation = pyo.quicksum(
            ##    m.load_shift_neg[t] * (1-m.renewable_signal[t]) for t in m.time_steps
            ##)

            return maximise_renewable_utilisation  ##+maximise_feedin_utilisation

    def bidirectional_prosumer_renewable_discharge(self, model):
        """
        Implements bidirectional+prosumer flexibility that discharges when renewable intensity is low.
        This measure allows the system to act as both consumer and prosumer, discharging stored energy
        when renewable availability is insufficient to meet demand efficiently.
        
        Currently implemented for bus_depot technology.

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
        
        # Invert the signal so low renewable intensity = high discharge signal
        discharge_signal_normalised = 1 - renewable_signal_normalised
        
        # Add normalized discharge signal as a model parameter
        model.discharge_signal = pyo.Param(
            model.time_steps,
            initialize={
                t: discharge_signal_normalised.iloc[t] for t in model.time_steps
            },
        )

        model.cost_tolerance = pyo.Param(initialize=self.cost_tolerance)
        model.total_cost = pyo.Param(initialize=0.0, mutable=True)

        # Variables for bidirectional flexibility with reasonable bounds
        model.load_shift_pos = pyo.Var(model.time_steps, within=pyo.NonNegativeReals)  # Max 1 kW
        model.load_shift_neg = pyo.Var(model.time_steps, within=pyo.NonNegativeReals)  # Max 1 kW  
        model.shift_indicator = pyo.Var(model.time_steps, within=pyo.Binary)
        
        # Additional variable for prosumer discharge capability with bounds
        model.prosumer_discharge = pyo.Var(model.time_steps, within=pyo.NonNegativeReals)  # Max 0.5 kW

        # Constraint to manage total cost upper limit with cost tolerance
        @model.Constraint()
        def total_cost_upper_limit(m):
            return pyo.quicksum(
                m.variable_cost[t] for t in m.time_steps
            ) <= m.total_cost * (1 + (m.cost_tolerance / 100))

        @model.Constraint(model.time_steps)
        def flex_constraint_upper(m, t):
            return m.load_shift_pos[t] <= (1 - m.shift_indicator[t]) * self.BIG_M

        @model.Constraint(model.time_steps)
        def flex_constraint_lower(m, t):
            return m.load_shift_neg[t] <= m.shift_indicator[t] * self.BIG_M

        # Power input constraint with bidirectional flexibility
        @model.Constraint(model.time_steps)
        def total_power_input_constraint_with_flex(m, t):
            if self.technology == "bus_depot":
                # For bus depot, the flexibility affects the power balance
                # Load shift positive = increase load, Load shift negative = reduce load  
                # Prosumer discharge = inject power back to grid (only for prosumer=Yes EVs)
                return (
                    m.total_power_input[t] == 
                    sum(m.dsm_blocks[ev].discharge[t] for ev in m.dsm_blocks if ev.startswith("electric_vehicle"))
                    + m.load_shift_pos[t] - m.load_shift_neg[t] - m.prosumer_discharge[t]
                )
            else:
                # For other technologies, basic flexibility constraint
                return (
                    m.total_power_input[t] + m.load_shift_pos[t] - m.load_shift_neg[t] == 
                    m.total_power_input[t]
                )

        # Prosumer constraint: only prosumer=Yes EVs can use prosumer_discharge
        @model.Constraint(model.time_steps)
        def prosumer_discharge_constraint(m, t):
            # Check if this unit is a prosumer-enabled EV
            if hasattr(self, 'is_prosumer') and self.is_prosumer and self.technology.startswith("electric_vehicle"):
                # Allow prosumer discharge up to max bound (already set in variable definition)
                return pyo.Constraint.Skip  # No additional constraint needed
            else:
                # Force prosumer_discharge to zero for non-prosumer units
                return m.prosumer_discharge[t] == 0

        @self.model.Objective(sense=pyo.maximize)
        def obj_rule_flex(m):
            """
            Maximizes discharge when renewable intensity is low (bidirectional+prosumer behavior)
            Same as renewable_utilisation but adds prosumer discharge capability
            """
            # Sum all EV discharges during low renewable periods
            total_ev_discharge_during_low_re = pyo.quicksum(
                sum(m.dsm_blocks[ev].discharge[t] for ev in m.dsm_blocks if ev.startswith("electric_vehicle")) 
                * m.discharge_signal[t] 
                for t in m.time_steps
            )
            
            # Add prosumer discharge during low renewable periods (only for prosumer=Yes)
            prosumer_discharge_during_low_re = pyo.quicksum(
                m.prosumer_discharge[t] * m.discharge_signal[t] for t in m.time_steps
            )
            
            # Also include load reduction during low RE periods
            load_reduction_during_low_re = pyo.quicksum(
                m.load_shift_neg[t] * m.discharge_signal[t] for t in m.time_steps
            )

            return total_ev_discharge_during_low_re + prosumer_discharge_during_low_re + load_reduction_during_low_re

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
        
        # Calculate and print total electricity demand for opt mode
        total_opt_demand = sum(opt_power_requirement)
        print(f"Total Electricity Demand (OPT mode): {total_opt_demand:.2f} MW")
    # PLOTTING SECTION - Technology specific plots
        if self.technology == "bus_depot":
            self._plot_bus_depot_optimization(instance)
        elif self.technology == "steel_plant":
            self._plot_steel_plant_optimization(instance)
        elif self.technology == "building":
            self._plot_building_optimization(instance)
        # Add more technology-specific plots as needed

    def _save_individual_plots_without_flex(self, instance, evs, charging_stations, time_steps):
        """
        Save each subplot as an individual PNG file - WITHOUT FLEX version
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
        import pyomo.environ as pyo

        # 1. EV Status Matrix
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        status_matrix = np.zeros((len(evs), len(time_steps)))

        for i, ev in enumerate(evs):
            for j, t in enumerate(time_steps):
                ev_availability = getattr(instance, f"{ev}_availability", None)
                if ev_availability is None or pyo.value(ev_availability[t]) == 0:
                    status_matrix[i, j] = 0
                    continue

                is_charging = False
                for cs in charging_stations:
                    if pyo.value(instance.is_assigned[ev, cs, t]) > 0.5:
                        status_matrix[i, j] = 2
                        is_charging = True
                        break

                if not is_charging and hasattr(instance, 'in_queue'):
                    if pyo.value(instance.in_queue[ev, t]) > 0.5:
                        status_matrix[i, j] = 1

        colors = ['lightgray', 'yellow', 'green']
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        im = ax1.imshow(status_matrix, aspect='auto', cmap=cmap, vmin=0, vmax=2)
        ax1.set_yticks(np.arange(len(evs)))
        ax1.set_yticklabels(evs)
        ax1.set_xlabel('Time Steps')
        ax1.set_title('EV Status (Gray=Driving, Yellow=Queue, Green=Charging) - WITHOUT FLEX')
        legend_elements = [
            mpatches.Patch(color='lightgray', label='Driving'),
            mpatches.Patch(color='yellow', label='In Queue'),
            mpatches.Patch(color='green', label='Charging')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.tight_layout()
        filename1 = 'bus_depot_1_ev_status_without_flex.png'
        plt.savefig(filename1, dpi=300, bbox_inches='tight')
        print(f"Individual plot saved: {filename1}")
        plt.close()

        # 2. Charging Station Utilization
        fig2, ax2 = plt.subplots(figsize=(10, 6))
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
        ax2.set_title('Charging Station Power Output - WITHOUT FLEX')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        filename2 = 'bus_depot_2_charging_station_power_without_flex.png'
        plt.savefig(filename2, dpi=300, bbox_inches='tight')
        print(f"Individual plot saved: {filename2}")
        plt.close()

        # 3. EV Charging and Discharging Power
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        for ev in evs[:5]:
            if ev in instance.dsm_blocks:
                ev_charge = [pyo.value(instance.dsm_blocks[ev].charge[t]) for t in time_steps]
                ev_discharge = [pyo.value(instance.dsm_blocks[ev].discharge[t]) for t in time_steps]
                ev_usage = [pyo.value(instance.dsm_blocks[ev].usage[t]) for t in time_steps]

                ax3.plot(time_steps, ev_charge, label=f'{ev} Charge', color='blue', alpha=0.8)
                ax3.plot(time_steps, [-d for d in ev_discharge], label=f'{ev} Discharge', color='red', alpha=0.8, linestyle='--')
                ax3.plot(time_steps, [-u for u in ev_usage], label=f'{ev} Usage', color='orange', alpha=0.8, linestyle=':')

        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Power (kW)')
        ax3.set_title('EV Power: Charge (Blue), Discharge (Red), Usage (Orange) - WITHOUT FLEX')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        filename3 = 'bus_depot_3_ev_power_without_flex.png'
        plt.savefig(filename3, dpi=300, bbox_inches='tight')
        print(f"Individual plot saved: {filename3}")
        plt.close()

        # 4. EV State of Charge
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        for ev in evs[:5]:
            if ev in instance.dsm_blocks and hasattr(instance.dsm_blocks[ev], 'soc'):
                soc = [pyo.value(instance.dsm_blocks[ev].soc[t]) for t in time_steps]
                max_capacity = pyo.value(instance.dsm_blocks[ev].max_capacity)
                soc_percentage = [s/max_capacity * 100 for s in soc]
                ax4.plot(time_steps, soc_percentage, label=ev, marker='o', markersize=3)

        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('State of Charge (%)')
        ax4.set_title('EV Battery Status (First 5 EVs) - WITHOUT FLEX')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 105])
        plt.tight_layout()
        filename4 = 'bus_depot_4_ev_soc_without_flex.png'
        plt.savefig(filename4, dpi=300, bbox_inches='tight')
        print(f"Individual plot saved: {filename4}")
        plt.close()

        # 5. Queue Length Over Time
        fig5, ax5 = plt.subplots(figsize=(10, 6))
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
        ax5.set_title('Queue Length Over Time - WITHOUT FLEX')
        ax5.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        filename5 = 'bus_depot_5_queue_length_without_flex.png'
        plt.savefig(filename5, dpi=300, bbox_inches='tight')
        print(f"Individual plot saved: {filename5}")
        plt.close()

        # 6. Financial Analysis
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        electricity_prices = [pyo.value(instance.electricity_price[t]) for t in time_steps]
        total_power = [pyo.value(instance.total_power_input[t]) for t in time_steps]
        variable_costs = [pyo.value(instance.variable_cost[t]) for t in time_steps]
        variable_revenues = [pyo.value(instance.variable_rev[t]) for t in time_steps]
        net_incomes = [pyo.value(instance.net_income[t]) for t in time_steps]

        ax6_twin = ax6.twinx()

        line1 = ax6.plot(time_steps, variable_costs, 'r-', label='Variable Cost (€)', linewidth=2)
        line2 = ax6.plot(time_steps, variable_revenues, 'g-', label='Variable Revenue (€)', linewidth=2)
        line3 = ax6.plot(time_steps, net_incomes, 'purple', label='Net Income (€)', linewidth=2, linestyle='--')

        ax6.set_xlabel('Time Steps')
        ax6.set_ylabel('Financial Metrics (€)', color='black')
        ax6.tick_params(axis='y', labelcolor='black')

        line4 = ax6_twin.plot(time_steps, electricity_prices, 'b-', label='Electricity Price (€/MWh)', linewidth=1, alpha=0.7)
        ax6_twin.set_ylabel('Electricity Price (€/MWh)', color='b')
        ax6_twin.tick_params(axis='y', labelcolor='b')

        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc='upper left')

        ax6.set_title('Financial Analysis: Cost, Revenue, Net Income & Price - WITHOUT FLEX')
        ax6.grid(True, alpha=0.3)
        plt.tight_layout()
        filename6 = 'bus_depot_6_financial_analysis_without_flex.png'
        plt.savefig(filename6, dpi=300, bbox_inches='tight')
        print(f"Individual plot saved: {filename6}")
        plt.close()

    def _plot_bus_depot_optimization(self, instance):
        """
        Bus depot specific plotting logic
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
        import pandas as pd
        import os
        
        if not (hasattr(instance, 'evs') and hasattr(instance, 'charging_stations')):
            logger.warning("Bus depot plotting requires EVs and charging stations")
            return
        
        time_steps = list(instance.time_steps)
        evs = list(instance.evs)
        charging_stations = list(instance.charging_stations)
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Bus Depot Optimization Results - WITHOUT FLEX', fontsize=16, fontweight='bold')
        
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
        ax1.set_title('EV Status (Gray=Driving, Yellow=Queue, Green=Charging) - WITHOUT FLEX')
        
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
        ax2.set_title('Charging Station Power Output - WITHOUT FLEX')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. EV Charging and Discharging Power
        for ev in evs[:5]:  # Limit to first 5 EVs for clarity
            if ev in instance.dsm_blocks:
                ev_charge = [pyo.value(instance.dsm_blocks[ev].charge[t]) for t in time_steps]
                ev_discharge = [pyo.value(instance.dsm_blocks[ev].discharge[t]) for t in time_steps]
                ev_usage = [pyo.value(instance.dsm_blocks[ev].usage[t]) for t in time_steps]
                
                # Plot charging as positive values (blue)
                ax3.plot(time_steps, ev_charge, label=f'{ev} Charge', color='blue', alpha=0.8)
                # Plot discharging as negative values (red)  
                ax3.plot(time_steps, [-d for d in ev_discharge], label=f'{ev} Discharge', color='red', alpha=0.8, linestyle='--')
                # Plot usage as negative values (orange)
                ax3.plot(time_steps, [-u for u in ev_usage], label=f'{ev} Usage', color='orange', alpha=0.8, linestyle=':')
        
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Power (kW)')
        ax3.set_title('EV Power: Charge (Blue), Discharge (Red), Usage (Orange) - WITHOUT FLEX')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 4. EV State of Charge
        for ev in evs[:5]:  # Limit to first 5 EVs for clarity
            if ev in instance.dsm_blocks and hasattr(instance.dsm_blocks[ev], 'soc'):
                soc = [pyo.value(instance.dsm_blocks[ev].soc[t]) for t in time_steps]
                max_capacity = pyo.value(instance.dsm_blocks[ev].max_capacity)
                soc_percentage = [s/max_capacity * 100 for s in soc]
                ax4.plot(time_steps, soc_percentage, label=ev, marker='o', markersize=3)
        
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('State of Charge (%)')
        ax4.set_title('EV Battery Status (First 5 EVs) - WITHOUT FLEX')
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
        ax5.set_title('Queue Length Over Time - WITHOUT FLEX')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Financial Analysis: Cost, Revenue, and Net Income
        electricity_prices = [pyo.value(instance.electricity_price[t]) for t in time_steps]
        total_power = [pyo.value(instance.total_power_input[t]) for t in time_steps]
        variable_costs = [pyo.value(instance.variable_cost[t]) for t in time_steps]
        variable_revenues = [pyo.value(instance.variable_rev[t]) for t in time_steps]
        net_incomes = [pyo.value(instance.net_income[t]) for t in time_steps]
        
        # Calculate marginal cost (electricity price * power consumption)
        marginal_costs = [price * power for price, power in zip(electricity_prices, total_power)]
        
        # Create dual y-axis plot
        ax6_twin = ax6.twinx()
        
        # Plot financial metrics on left axis
        line1 = ax6.plot(time_steps, variable_costs, 'r-', label='Variable Cost (€)', linewidth=2)
        line2 = ax6.plot(time_steps, variable_revenues, 'g-', label='Variable Revenue (€)', linewidth=2)
        line3 = ax6.plot(time_steps, net_incomes, 'purple', label='Net Income (€)', linewidth=2, linestyle='--')
        
        ax6.set_xlabel('Time Steps')
        ax6.set_ylabel('Financial Metrics (€)', color='black')
        ax6.tick_params(axis='y', labelcolor='black')
        
        # Plot electricity price on right axis
        line4 = ax6_twin.plot(time_steps, electricity_prices, 'b-', label='Electricity Price (€/MWh)', linewidth=1, alpha=0.7)
        ax6_twin.set_ylabel('Electricity Price (€/MWh)', color='b')
        ax6_twin.tick_params(axis='y', labelcolor='b')
        
        # Combine legends
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc='upper left')
        
        ax6.set_title('Financial Analysis: Cost, Revenue, Net Income & Price - WITHOUT FLEX')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()

        # Save combined plot
        combined_filename = 'bus_depot_combined_without_flex.png'
        plt.savefig(combined_filename, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved: {combined_filename}")

        plt.show()

        # Save individual plots
        self._save_individual_plots_without_flex(instance, evs, charging_stations, time_steps)

        # Export graph data to CSV files
        try:
            self._export_bus_depot_graphs_to_csv(instance, evs, charging_stations, time_steps)
        except PermissionError:
            print("CSV files are open in another program. Close them and run again to export data.")

        # Print summary
        self._print_bus_depot_summary(instance, evs, charging_stations, time_steps)

    def _export_bus_depot_graphs_to_csv(self, instance, evs, charging_stations, time_steps):
        """
        Export all bus depot graph data to CSV files - WITHOUT FLEX version
        Creates two detailed CSV files: one for EVs and one for Charging Stations
        """
        import pandas as pd
        import numpy as np
        import os

        base_path = os.getcwd()

        # ========== ELECTRIC VEHICLES DETAILED CSV ==========
        ev_detailed_data = []

        for ev in evs:
            if ev not in instance.dsm_blocks:
                continue

            for t in time_steps:
                # Basic information
                row = {
                    'EV_ID': ev,
                    'Time_Step': t,
                }

                # Operational Behaviour
                ev_availability = getattr(instance, f"{ev}_availability", None)
                if ev_availability is not None:
                    row['Availability'] = pyo.value(ev_availability[t])
                else:
                    row['Availability'] = 0

                # Status (Driving/Queue/Charging)
                if row['Availability'] == 0:
                    row['Status'] = 'Driving'
                else:
                    is_charging = False
                    for cs in charging_stations:
                        if pyo.value(instance.is_assigned[ev, cs, t]) > 0.5:
                            row['Status'] = 'Charging'
                            row['Assigned_Station'] = cs
                            is_charging = True
                            break

                    if not is_charging:
                        if hasattr(instance, 'in_queue') and pyo.value(instance.in_queue[ev, t]) > 0.5:
                            row['Status'] = 'In_Queue'
                        else:
                            row['Status'] = 'Idle'

                # Power flows
                row['Charge_Power_kW'] = pyo.value(instance.dsm_blocks[ev].charge[t])
                row['Discharge_Power_kW'] = pyo.value(instance.dsm_blocks[ev].discharge[t])
                row['Usage_Power_kW'] = pyo.value(instance.dsm_blocks[ev].usage[t])

                # State of Charge
                if hasattr(instance.dsm_blocks[ev], 'soc'):
                    soc = pyo.value(instance.dsm_blocks[ev].soc[t])
                    max_capacity = pyo.value(instance.dsm_blocks[ev].max_capacity)
                    row['SOC_kWh'] = soc
                    row['SOC_Percentage'] = (soc / max_capacity * 100) if max_capacity > 0 else 0
                    row['Max_Capacity_kWh'] = max_capacity

                # Net power (positive for charging, negative for discharging/usage)
                row['Net_Power_kW'] = row['Charge_Power_kW'] - row['Discharge_Power_kW'] - row['Usage_Power_kW']

                ev_detailed_data.append(row)

        ev_df = pd.DataFrame(ev_detailed_data)
        ev_csv_path = os.path.join(base_path, "ev_operational_details_without_flex.csv")
        ev_df.to_csv(ev_csv_path, index=False)
        print(f"EV Operational Details exported to: {ev_csv_path}")

        # ========== CHARGING STATIONS DETAILED CSV ==========
        cs_detailed_data = []

        # Get electricity price for marginal cost calculation
        electricity_prices = {t: pyo.value(instance.electricity_price[t]) for t in time_steps}

        for cs in charging_stations:
            if cs not in instance.dsm_blocks:
                continue

            cs_max_power = pyo.value(instance.dsm_blocks[cs].max_power)

            for t in time_steps:
                row = {
                    'Station_ID': cs,
                    'Time_Step': t,
                }

                # Discharge (power output to EVs)
                discharge = pyo.value(instance.dsm_blocks[cs].discharge[t])
                row['Discharge_Power_kW'] = discharge
                row['Max_Power_kW'] = cs_max_power
                row['Utilization_Percentage'] = (discharge / cs_max_power * 100) if cs_max_power > 0 else 0

                # Cost and Revenue per time step
                electricity_price = electricity_prices[t]
                row['Electricity_Price_EUR_per_MWh'] = electricity_price

                # Convert kW to MWh for cost calculation (assuming 1 hour time step)
                energy_mwh = discharge / 1000.0  # kW to MW, 1 hour duration
                row['Cost_per_Time_Step_EUR'] = energy_mwh * electricity_price

                # Marginal cost (cost per kWh delivered)
                row['Marginal_Cost_EUR_per_kWh'] = electricity_price / 1000.0 if electricity_price > 0 else 0

                # Count assigned EVs
                assigned_evs = sum(1 for ev in evs if pyo.value(instance.is_assigned[ev, cs, t]) > 0.5)
                row['Assigned_EVs_Count'] = assigned_evs

                cs_detailed_data.append(row)

        cs_df = pd.DataFrame(cs_detailed_data)
        cs_csv_path = os.path.join(base_path, "charging_station_operational_details_without_flex.csv")
        cs_df.to_csv(cs_csv_path, index=False)
        print(f"Charging Station Operational Details exported to: {cs_csv_path}")

        # ========== LEGACY EXPORTS (keep for compatibility) ==========
        # Graph 1: EV Status Matrix (Idle/Queue/Charging)
        status_matrix = np.zeros((len(evs), len(time_steps)))

        for i, ev in enumerate(evs):
            for j, t in enumerate(time_steps):
                ev_availability = getattr(instance, f"{ev}_availability", None)
                if ev_availability is None or pyo.value(ev_availability[t]) == 0:
                    status_matrix[i, j] = 0
                    continue

                is_charging = False
                for cs in charging_stations:
                    if pyo.value(instance.is_assigned[ev, cs, t]) > 0.5:
                        status_matrix[i, j] = 2
                        is_charging = True
                        break

                if not is_charging and hasattr(instance, 'in_queue'):
                    if pyo.value(instance.in_queue[ev, t]) > 0.5:
                        status_matrix[i, j] = 1

        status_df = pd.DataFrame(status_matrix, index=evs, columns=[f'timestep_{t}' for t in time_steps])
        csv_path1 = os.path.join(base_path, "ev_status_matrix.csv")
        status_df.to_csv(csv_path1)
        print(f"EV Status Matrix exported to: {csv_path1}")

        # Graph 2: Charging Station Power Output
        cs_data = {}
        for cs in charging_stations:
            cs_discharge = []
            for t in time_steps:
                discharge = pyo.value(instance.dsm_blocks[cs].discharge[t])
                cs_discharge.append(discharge)
            cs_data[cs] = cs_discharge

        cs_df_legacy = pd.DataFrame(cs_data, index=[f'timestep_{t}' for t in time_steps])
        csv_path2 = os.path.join(base_path, "charging_station_power.csv")
        cs_df_legacy.to_csv(csv_path2)
        print(f"Charging Station Power exported to: {csv_path2}")

        # Graph 3: EV Charging and Discharging Power
        ev_charge_data = {}
        ev_discharge_data = {}
        ev_usage_data = {}
        for ev in evs:
            if ev in instance.dsm_blocks:
                ev_charge = [pyo.value(instance.dsm_blocks[ev].charge[t]) for t in time_steps]
                ev_discharge = [pyo.value(instance.dsm_blocks[ev].discharge[t]) for t in time_steps]
                ev_usage = [pyo.value(instance.dsm_blocks[ev].usage[t]) for t in time_steps]
                ev_charge_data[f'{ev}_charge'] = ev_charge
                ev_discharge_data[f'{ev}_discharge'] = ev_discharge
                ev_usage_data[f'{ev}_usage'] = ev_usage

        if ev_charge_data or ev_discharge_data or ev_usage_data:
            ev_power_data = {**ev_charge_data, **ev_discharge_data, **ev_usage_data}
            ev_power_df = pd.DataFrame(ev_power_data, index=[f'timestep_{t}' for t in time_steps])
            csv_path3 = os.path.join(base_path, "ev_charging_power.csv")
            ev_power_df.to_csv(csv_path3)
            print(f"EV Charging and Discharging Power exported to: {csv_path3}")

        # Graph 4: Financial Analysis Data
        total_power = [pyo.value(instance.total_power_input[t]) for t in time_steps]
        variable_costs = [pyo.value(instance.variable_cost[t]) for t in time_steps]
        variable_revenues = [pyo.value(instance.variable_rev[t]) for t in time_steps]
        net_incomes = [pyo.value(instance.net_income[t]) for t in time_steps]
        marginal_costs = [electricity_prices[t] * total_power[i] for i, t in enumerate(time_steps)]

        financial_df = pd.DataFrame({
            'electricity_price': [electricity_prices[t] for t in time_steps],
            'total_power': total_power,
            'variable_cost': variable_costs,
            'variable_revenue': variable_revenues,
            'net_income': net_incomes,
            'marginal_cost': marginal_costs
        }, index=[f'timestep_{t}' for t in time_steps])

        csv_path4 = os.path.join(base_path, "financial_analysis.csv")
        financial_df.to_csv(csv_path4)
        print(f"Financial Analysis exported to: {csv_path4}")

        print(f"\n[OK] All CSV files exported to {base_path}")
        print(f"  - NEW: {ev_csv_path}")
        print(f"  - NEW: {cs_csv_path}")

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
            print("Flex objective value:", objective_value)

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
                # For bus depot technology, sum charging station discharges
                if self.technology == "bus_depot":
                    for cs in getattr(instance, 'charging_stations', []):
                        if cs in instance.dsm_blocks and hasattr(instance.dsm_blocks[cs], 'discharge'):
                            total_power += pyo.value(instance.dsm_blocks[cs].discharge[t])
                else:
                    # For other technologies, use power_in if available
                    for tech_name, tech_block in instance.dsm_blocks.items():
                        if hasattr(tech_block, "power_in"):
                            total_power += pyo.value(tech_block.power_in[t])
                adjusted_total_power_input.append(total_power)

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

        # Calculate and print total electricity demand for flex mode
        if hasattr(self, 'flex_power_requirement'):
            total_flex_demand = self.flex_power_requirement.sum()
            print(f"Total Electricity Demand (FLEX mode): {total_flex_demand:.2f} MW")
        
        # Variable cost series
        flex_variable_cost = [
            instance.variable_cost[t].value for t in instance.time_steps
        ]
        self.flex_variable_cost_series = FastSeries(
            index=self.index, value=flex_variable_cost
        )
        
        # Calculate total cost for flex mode
        self.total_cost = sum(
            instance.variable_cost[t].value for t in instance.time_steps
        )
        
        # PLOTTING SECTION - Technology specific plots with flex
        if self.technology == "bus_depot":
            self._plot_bus_depot_optimization_with_flex(instance)
            # Add FCR plotting if prosumer enabled
            if hasattr(self, 'is_prosumer') and self.is_prosumer:
                try:
                    self.plot_bus_depot_fcr(instance, save_path="bus_depot_fcr_results.png", show=False)
                    print("FCR plot exported to: bus_depot_fcr_results.png")
                except Exception as e:
                    print(f"FCR plotting failed: {e}")
        elif self.technology == "steel_plant":
            self._plot_steel_plant_optimization_with_flex(instance)
        elif self.technology == "building":
            self._plot_building_optimization_with_flex(instance)
        # Add more technology-specific plots as needed

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

    def _save_individual_plots_with_flex(self, instance, evs, charging_stations, time_steps):
        """
        Save each subplot as an individual PNG file - WITH FLEX version
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
        import pyomo.environ as pyo

        # 1. EV Status Matrix
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        status_matrix = np.zeros((len(evs), len(time_steps)))

        for i, ev in enumerate(evs):
            for j, t in enumerate(time_steps):
                ev_availability = getattr(instance, f"{ev}_availability", None)
                if ev_availability is None or pyo.value(ev_availability[t]) == 0:
                    status_matrix[i, j] = 0
                    continue

                is_charging = False
                for cs in charging_stations:
                    if pyo.value(instance.is_assigned[ev, cs, t]) > 0.5:
                        status_matrix[i, j] = 2
                        is_charging = True
                        break

                if not is_charging and hasattr(instance, 'in_queue'):
                    if pyo.value(instance.in_queue[ev, t]) > 0.5:
                        status_matrix[i, j] = 1

        colors = ['lightgray', 'yellow', 'green']
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        im = ax1.imshow(status_matrix, aspect='auto', cmap=cmap, vmin=0, vmax=2)
        ax1.set_yticks(np.arange(len(evs)))
        ax1.set_yticklabels(evs)
        ax1.set_xlabel('Time Steps')
        ax1.set_title('EV Status (Gray=Driving, Yellow=Queue, Green=Charging) - WITH FLEX')
        legend_elements = [
            mpatches.Patch(color='lightgray', label='Driving'),
            mpatches.Patch(color='yellow', label='In Queue'),
            mpatches.Patch(color='green', label='Charging')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.tight_layout()
        filename1 = 'bus_depot_1_ev_status_with_flex.png'
        plt.savefig(filename1, dpi=300, bbox_inches='tight')
        print(f"Individual plot saved: {filename1}")
        plt.close()

        # 2. Charging Station Utilization
        fig2, ax2 = plt.subplots(figsize=(10, 6))
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
        ax2.set_title('Charging Station Power Output - WITH FLEX')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        filename2 = 'bus_depot_2_charging_station_power_with_flex.png'
        plt.savefig(filename2, dpi=300, bbox_inches='tight')
        print(f"Individual plot saved: {filename2}")
        plt.close()

        # 3. EV Charging and Discharging Power
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        for ev in evs[:5]:
            if ev in instance.dsm_blocks:
                ev_charge = [pyo.value(instance.dsm_blocks[ev].charge[t]) for t in time_steps]
                ev_discharge = [pyo.value(instance.dsm_blocks[ev].discharge[t]) for t in time_steps]
                ev_usage = [pyo.value(instance.dsm_blocks[ev].usage[t]) for t in time_steps]

                ax3.plot(time_steps, ev_charge, label=f'{ev} Charge', color='blue', alpha=0.8)
                ax3.plot(time_steps, [-d for d in ev_discharge], label=f'{ev} Discharge', color='red', alpha=0.8, linestyle='--')
                ax3.plot(time_steps, [-u for u in ev_usage], label=f'{ev} Usage', color='orange', alpha=0.8, linestyle=':')

        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Power (kW)')
        ax3.set_title('EV Power: Charge (Blue), Discharge (Red), Usage (Orange) - WITH FLEX')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        filename3 = 'bus_depot_3_ev_power_with_flex.png'
        plt.savefig(filename3, dpi=300, bbox_inches='tight')
        print(f"Individual plot saved: {filename3}")
        plt.close()

        # 4. EV State of Charge
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        for ev in evs[:5]:
            if ev in instance.dsm_blocks and hasattr(instance.dsm_blocks[ev], 'soc'):
                soc = [pyo.value(instance.dsm_blocks[ev].soc[t]) for t in time_steps]
                max_capacity = pyo.value(instance.dsm_blocks[ev].max_capacity)
                soc_percentage = [s/max_capacity * 100 for s in soc]
                ax4.plot(time_steps, soc_percentage, label=ev, marker='o', markersize=3)

        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('State of Charge (%)')
        ax4.set_title('EV Battery Status (First 5 EVs) - WITH FLEX')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 105])
        plt.tight_layout()
        filename4 = 'bus_depot_4_ev_soc_with_flex.png'
        plt.savefig(filename4, dpi=300, bbox_inches='tight')
        print(f"Individual plot saved: {filename4}")
        plt.close()

        # 5. Load Shift Analysis
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        if hasattr(instance, 'load_shift_pos') and hasattr(instance, 'load_shift_neg'):
            load_shift_pos = [pyo.value(instance.load_shift_pos[t]) for t in time_steps]
            load_shift_neg = [pyo.value(instance.load_shift_neg[t]) for t in time_steps]
            net_shift = [pos - neg for pos, neg in zip(load_shift_pos, load_shift_neg)]

            ax5.plot(time_steps, load_shift_pos, label='Load Shift Positive', color='green', linewidth=2)
            ax5.plot(time_steps, [-neg for neg in load_shift_neg], label='Load Shift Negative', color='red', linewidth=2)
            ax5.plot(time_steps, net_shift, label='Net Load Shift', color='purple', linewidth=2, linestyle='--')
            ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

            ax5.set_xlabel('Time Steps')
            ax5.set_ylabel('Load Shift (kW)')
            ax5.set_title('Load Shifting Analysis - WITH FLEX')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            # Fallback to queue length if load shift variables not available
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
            ax5.set_title('Queue Length Over Time - WITH FLEX')
            ax5.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filename5 = 'bus_depot_5_load_shift_with_flex.png'
        plt.savefig(filename5, dpi=300, bbox_inches='tight')
        print(f"Individual plot saved: {filename5}")
        plt.close()

        # 6. Financial Analysis
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        electricity_prices = [pyo.value(instance.electricity_price[t]) for t in time_steps]
        total_power = [pyo.value(instance.total_power_input[t]) for t in time_steps]
        variable_costs = [pyo.value(instance.variable_cost[t]) for t in time_steps]
        variable_revenues = [pyo.value(instance.variable_rev[t]) for t in time_steps]
        net_incomes = [pyo.value(instance.net_income[t]) for t in time_steps]

        ax6_twin = ax6.twinx()

        line1 = ax6.plot(time_steps, variable_costs, 'r-', label='Variable Cost (€)', linewidth=2)
        line2 = ax6.plot(time_steps, variable_revenues, 'g-', label='Variable Revenue (€)', linewidth=2)
        line3 = ax6.plot(time_steps, net_incomes, 'purple', label='Net Income (€)', linewidth=2, linestyle='--')

        ax6.set_xlabel('Time Steps')
        ax6.set_ylabel('Financial Metrics (€)', color='black')
        ax6.tick_params(axis='y', labelcolor='black')

        line4 = ax6_twin.plot(time_steps, electricity_prices, 'b-', label='Electricity Price (€/MWh)', linewidth=1, alpha=0.7)
        ax6_twin.set_ylabel('Electricity Price (€/MWh)', color='b')
        ax6_twin.tick_params(axis='y', labelcolor='b')

        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc='upper left')

        ax6.set_title('Financial Analysis: Cost, Revenue, Net Income & Price - WITH FLEX')
        ax6.grid(True, alpha=0.3)
        plt.tight_layout()
        filename6 = 'bus_depot_6_financial_analysis_with_flex.png'
        plt.savefig(filename6, dpi=300, bbox_inches='tight')
        print(f"Individual plot saved: {filename6}")
        plt.close()

    def _plot_bus_depot_optimization_with_flex(self, instance):
        """
        Bus depot specific plotting logic for WITH FLEX mode
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
        import pandas as pd
        import os
        
        if not (hasattr(instance, 'evs') and hasattr(instance, 'charging_stations')):
            logger.warning("Bus depot plotting requires EVs and charging stations")
            return
        
        time_steps = list(instance.time_steps)
        evs = list(instance.evs)
        charging_stations = list(instance.charging_stations)
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Bus Depot Optimization Results - WITH FLEX', fontsize=16, fontweight='bold')
        
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
        ax1.set_title('EV Status (Gray=Driving, Yellow=Queue, Green=Charging) - WITH FLEX')
        
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
        ax2.set_title('Charging Station Power Output - WITH FLEX')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. EV Charging and Discharging Power
        for ev in evs[:5]:  # Limit to first 5 EVs for clarity
            if ev in instance.dsm_blocks:
                ev_charge = [pyo.value(instance.dsm_blocks[ev].charge[t]) for t in time_steps]
                ev_discharge = [pyo.value(instance.dsm_blocks[ev].discharge[t]) for t in time_steps]
                ev_usage = [pyo.value(instance.dsm_blocks[ev].usage[t]) for t in time_steps]
                
                # Plot charging as positive values (blue)
                ax3.plot(time_steps, ev_charge, label=f'{ev} Charge', color='blue', alpha=0.8)
                # Plot discharging as negative values (red)  
                ax3.plot(time_steps, [-d for d in ev_discharge], label=f'{ev} Discharge', color='red', alpha=0.8, linestyle='--')
                # Plot usage as negative values (orange)
                ax3.plot(time_steps, [-u for u in ev_usage], label=f'{ev} Usage', color='orange', alpha=0.8, linestyle=':')
        
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Power (kW)')
        ax3.set_title('EV Power: Charge (Blue), Discharge (Red), Usage (Orange) - WITH FLEX')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 4. EV State of Charge
        for ev in evs[:5]:  # Limit to first 5 EVs for clarity
            if ev in instance.dsm_blocks and hasattr(instance.dsm_blocks[ev], 'soc'):
                soc = [pyo.value(instance.dsm_blocks[ev].soc[t]) for t in time_steps]
                max_capacity = pyo.value(instance.dsm_blocks[ev].max_capacity)
                soc_percentage = [s/max_capacity * 100 for s in soc]
                ax4.plot(time_steps, soc_percentage, label=ev, marker='o', markersize=3)
        
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('State of Charge (%)')
        ax4.set_title('EV Battery Status (First 5 EVs) - WITH FLEX')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 105])
        
        # 5. Load Shift Analysis (NEW - specific to flex mode)
        if hasattr(instance, 'load_shift_pos') and hasattr(instance, 'load_shift_neg'):
            load_shift_pos = [pyo.value(instance.load_shift_pos[t]) for t in time_steps]
            load_shift_neg = [pyo.value(instance.load_shift_neg[t]) for t in time_steps]
            net_shift = [pos - neg for pos, neg in zip(load_shift_pos, load_shift_neg)]
            
            ax5.plot(time_steps, load_shift_pos, label='Load Shift Positive', color='green', linewidth=2)
            ax5.plot(time_steps, [-neg for neg in load_shift_neg], label='Load Shift Negative', color='red', linewidth=2)
            ax5.plot(time_steps, net_shift, label='Net Load Shift', color='purple', linewidth=2, linestyle='--')
            ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            ax5.set_xlabel('Time Steps')
            ax5.set_ylabel('Load Shift (kW)')
            ax5.set_title('Load Shifting Analysis - WITH FLEX')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            # Fallback to queue length if load shift variables not available
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
            ax5.set_title('Queue Length Over Time - WITH FLEX')
            ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Financial Analysis: Cost, Revenue, and Net Income
        electricity_prices = [pyo.value(instance.electricity_price[t]) for t in time_steps]
        total_power = [pyo.value(instance.total_power_input[t]) for t in time_steps]
        variable_costs = [pyo.value(instance.variable_cost[t]) for t in time_steps]
        variable_revenues = [pyo.value(instance.variable_rev[t]) for t in time_steps]
        net_incomes = [pyo.value(instance.net_income[t]) for t in time_steps]
        
        # Calculate marginal cost (electricity price * power consumption)
        marginal_costs = [price * power for price, power in zip(electricity_prices, total_power)]
        
        # Create dual y-axis plot
        ax6_twin = ax6.twinx()
        
        # Plot financial metrics on left axis
        line1 = ax6.plot(time_steps, variable_costs, 'r-', label='Variable Cost (€)', linewidth=2)
        line2 = ax6.plot(time_steps, variable_revenues, 'g-', label='Variable Revenue (€)', linewidth=2)
        line3 = ax6.plot(time_steps, net_incomes, 'purple', label='Net Income (€)', linewidth=2, linestyle='--')
        
        ax6.set_xlabel('Time Steps')
        ax6.set_ylabel('Financial Metrics (€)', color='black')
        ax6.tick_params(axis='y', labelcolor='black')
        
        # Plot electricity price on right axis
        line4 = ax6_twin.plot(time_steps, electricity_prices, 'b-', label='Electricity Price (€/MWh)', linewidth=1, alpha=0.7)
        ax6_twin.set_ylabel('Electricity Price (€/MWh)', color='b')
        ax6_twin.tick_params(axis='y', labelcolor='b')
        
        # Combine legends
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc='upper left')
        
        ax6.set_title('Financial Analysis: Cost, Revenue, Net Income & Price - WITH FLEX')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save combined plot
        combined_filename = 'bus_depot_combined_with_flex.png'
        plt.savefig(combined_filename, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved: {combined_filename}")

        plt.show()

        # Save individual plots
        self._save_individual_plots_with_flex(instance, evs, charging_stations, time_steps)

        # Export graph data to CSV files with FLEX suffix
        try:
            self._export_bus_depot_graphs_to_csv_with_flex(instance, evs, charging_stations, time_steps)
        except PermissionError:
            print("CSV files are open in another program. Close them and run again to export data.")

        # Print summary
        self._print_bus_depot_summary_with_flex(instance, evs, charging_stations, time_steps)

    def _export_bus_depot_graphs_to_csv_with_flex(self, instance, evs, charging_stations, time_steps):
        """
        Export all bus depot graph data to CSV files - WITH FLEX version
        Creates two detailed CSV files: one for EVs and one for Charging Stations
        """
        import pandas as pd
        import numpy as np
        import os

        base_path = os.getcwd()

        # ========== ELECTRIC VEHICLES DETAILED CSV (WITH FLEX) ==========
        ev_detailed_data = []

        for ev in evs:
            if ev not in instance.dsm_blocks:
                continue

            for t in time_steps:
                # Basic information
                row = {
                    'EV_ID': ev,
                    'Time_Step': t,
                }

                # Operational Behaviour
                ev_availability = getattr(instance, f"{ev}_availability", None)
                if ev_availability is not None:
                    row['Availability'] = pyo.value(ev_availability[t])
                else:
                    row['Availability'] = 0

                # Status (Driving/Queue/Charging)
                if row['Availability'] == 0:
                    row['Status'] = 'Driving'
                else:
                    is_charging = False
                    for cs in charging_stations:
                        if pyo.value(instance.is_assigned[ev, cs, t]) > 0.5:
                            row['Status'] = 'Charging'
                            row['Assigned_Station'] = cs
                            is_charging = True
                            break

                    if not is_charging:
                        if hasattr(instance, 'in_queue') and pyo.value(instance.in_queue[ev, t]) > 0.5:
                            row['Status'] = 'In_Queue'
                        else:
                            row['Status'] = 'Idle'

                # Power flows
                row['Charge_Power_kW'] = pyo.value(instance.dsm_blocks[ev].charge[t])
                row['Discharge_Power_kW'] = pyo.value(instance.dsm_blocks[ev].discharge[t])
                row['Usage_Power_kW'] = pyo.value(instance.dsm_blocks[ev].usage[t])

                # State of Charge
                if hasattr(instance.dsm_blocks[ev], 'soc'):
                    soc = pyo.value(instance.dsm_blocks[ev].soc[t])
                    max_capacity = pyo.value(instance.dsm_blocks[ev].max_capacity)
                    row['SOC_kWh'] = soc
                    row['SOC_Percentage'] = (soc / max_capacity * 100) if max_capacity > 0 else 0
                    row['Max_Capacity_kWh'] = max_capacity

                # Net power (positive for charging, negative for discharging/usage)
                row['Net_Power_kW'] = row['Charge_Power_kW'] - row['Discharge_Power_kW'] - row['Usage_Power_kW']

                ev_detailed_data.append(row)

        ev_df = pd.DataFrame(ev_detailed_data)
        ev_csv_path = os.path.join(base_path, "ev_operational_details_with_flex.csv")
        ev_df.to_csv(ev_csv_path, index=False)
        print(f"EV Operational Details (WITH FLEX) exported to: {ev_csv_path}")

        # ========== CHARGING STATIONS DETAILED CSV (WITH FLEX) ==========
        cs_detailed_data = []

        # Get electricity price for marginal cost calculation
        electricity_prices = {t: pyo.value(instance.electricity_price[t]) for t in time_steps}

        for cs in charging_stations:
            if cs not in instance.dsm_blocks:
                continue

            cs_max_power = pyo.value(instance.dsm_blocks[cs].max_power)

            for t in time_steps:
                row = {
                    'Station_ID': cs,
                    'Time_Step': t,
                }

                # Discharge (power output to EVs)
                discharge = pyo.value(instance.dsm_blocks[cs].discharge[t])
                row['Discharge_Power_kW'] = discharge
                row['Max_Power_kW'] = cs_max_power
                row['Utilization_Percentage'] = (discharge / cs_max_power * 100) if cs_max_power > 0 else 0

                # Cost and Revenue per time step
                electricity_price = electricity_prices[t]
                row['Electricity_Price_EUR_per_MWh'] = electricity_price

                # Convert kW to MWh for cost calculation (assuming 1 hour time step)
                energy_mwh = discharge / 1000.0  # kW to MW, 1 hour duration
                row['Cost_per_Time_Step_EUR'] = energy_mwh * electricity_price

                # Marginal cost (cost per kWh delivered)
                row['Marginal_Cost_EUR_per_kWh'] = electricity_price / 1000.0 if electricity_price > 0 else 0

                # Revenue per time step (if applicable - V2G revenue)
                # In flex mode, there might be revenue from V2G discharge
                row['Revenue_per_Time_Step_EUR'] = 0  # Placeholder - can be calculated if V2G revenue exists

                # Count assigned EVs
                assigned_evs = sum(1 for ev in evs if pyo.value(instance.is_assigned[ev, cs, t]) > 0.5)
                row['Assigned_EVs_Count'] = assigned_evs

                cs_detailed_data.append(row)

        cs_df = pd.DataFrame(cs_detailed_data)
        cs_csv_path = os.path.join(base_path, "charging_station_operational_details_with_flex.csv")
        cs_df.to_csv(cs_csv_path, index=False)
        print(f"Charging Station Operational Details (WITH FLEX) exported to: {cs_csv_path}")

        # ========== LEGACY EXPORTS (keep for compatibility) ==========
        # Graph 1: EV Status Matrix (Idle/Queue/Charging)
        status_matrix = np.zeros((len(evs), len(time_steps)))

        for i, ev in enumerate(evs):
            for j, t in enumerate(time_steps):
                ev_availability = getattr(instance, f"{ev}_availability", None)
                if ev_availability is None or pyo.value(ev_availability[t]) == 0:
                    status_matrix[i, j] = 0
                    continue

                is_charging = False
                for cs in charging_stations:
                    if pyo.value(instance.is_assigned[ev, cs, t]) > 0.5:
                        status_matrix[i, j] = 2
                        is_charging = True
                        break

                if not is_charging and hasattr(instance, 'in_queue'):
                    if pyo.value(instance.in_queue[ev, t]) > 0.5:
                        status_matrix[i, j] = 1

        status_df = pd.DataFrame(status_matrix, index=evs, columns=[f'timestep_{t}' for t in time_steps])
        csv_path1 = os.path.join(base_path, "ev_status_matrix_with_flex.csv")
        status_df.to_csv(csv_path1)
        print(f"EV Status Matrix (WITH FLEX) exported to: {csv_path1}")

        # Graph 2: Charging Station Power Output
        cs_data = {}
        for cs in charging_stations:
            cs_discharge = []
            for t in time_steps:
                discharge = pyo.value(instance.dsm_blocks[cs].discharge[t])
                cs_discharge.append(discharge)
            cs_data[cs] = cs_discharge

        cs_df_legacy = pd.DataFrame(cs_data, index=[f'timestep_{t}' for t in time_steps])
        csv_path2 = os.path.join(base_path, "charging_station_power_with_flex.csv")
        cs_df_legacy.to_csv(csv_path2)
        print(f"Charging Station Power (WITH FLEX) exported to: {csv_path2}")

        # Graph 3: EV Charging and Discharging Power
        ev_charge_data = {}
        ev_discharge_data = {}
        ev_usage_data = {}
        for ev in evs:
            if ev in instance.dsm_blocks:
                ev_charge = [pyo.value(instance.dsm_blocks[ev].charge[t]) for t in time_steps]
                ev_discharge = [pyo.value(instance.dsm_blocks[ev].discharge[t]) for t in time_steps]
                ev_usage = [pyo.value(instance.dsm_blocks[ev].usage[t]) for t in time_steps]
                ev_charge_data[f'{ev}_charge'] = ev_charge
                ev_discharge_data[f'{ev}_discharge'] = ev_discharge
                ev_usage_data[f'{ev}_usage'] = ev_usage

        if ev_charge_data or ev_discharge_data or ev_usage_data:
            ev_power_data = {**ev_charge_data, **ev_discharge_data, **ev_usage_data}
            ev_power_df = pd.DataFrame(ev_power_data, index=[f'timestep_{t}' for t in time_steps])
            csv_path3 = os.path.join(base_path, "ev_charging_power_with_flex.csv")
            ev_power_df.to_csv(csv_path3)
            print(f"EV Charging and Discharging Power (WITH FLEX) exported to: {csv_path3}")

        # Graph 4: Load Shift Analysis (NEW for flex mode)
        if hasattr(instance, 'load_shift_pos') and hasattr(instance, 'load_shift_neg'):
            load_shift_pos = [pyo.value(instance.load_shift_pos[t]) for t in time_steps]
            load_shift_neg = [pyo.value(instance.load_shift_neg[t]) for t in time_steps]
            net_shift = [pos - neg for pos, neg in zip(load_shift_pos, load_shift_neg)]

            load_shift_df = pd.DataFrame({
                'load_shift_positive': load_shift_pos,
                'load_shift_negative': load_shift_neg,
                'net_load_shift': net_shift
            }, index=[f'timestep_{t}' for t in time_steps])

            csv_path4 = os.path.join(base_path, "load_shift_analysis_with_flex.csv")
            load_shift_df.to_csv(csv_path4)
            print(f"Load Shift Analysis (WITH FLEX) exported to: {csv_path4}")

        # Graph 5: Financial Analysis Data
        total_power = [pyo.value(instance.total_power_input[t]) for t in time_steps]
        variable_costs = [pyo.value(instance.variable_cost[t]) for t in time_steps]
        variable_revenues = [pyo.value(instance.variable_rev[t]) for t in time_steps]
        net_incomes = [pyo.value(instance.net_income[t]) for t in time_steps]
        marginal_costs = [electricity_prices[t] * total_power[i] for i, t in enumerate(time_steps)]

        financial_df = pd.DataFrame({
            'electricity_price': [electricity_prices[t] for t in time_steps],
            'total_power': total_power,
            'variable_cost': variable_costs,
            'variable_revenue': variable_revenues,
            'net_income': net_incomes,
            'marginal_cost': marginal_costs
        }, index=[f'timestep_{t}' for t in time_steps])

        csv_path5 = os.path.join(base_path, "financial_analysis_with_flex.csv")
        financial_df.to_csv(csv_path5)
        print(f"Financial Analysis (WITH FLEX) exported to: {csv_path5}")

        print(f"\n[OK] All CSV files (WITH FLEX) exported to {base_path}")
        print(f"  - NEW: {ev_csv_path}")
        print(f"  - NEW: {cs_csv_path}")

    def _print_bus_depot_summary_with_flex(self, instance, evs, charging_stations, time_steps):
        """Print optimization summary for bus depot WITH FLEX"""
        print("\n" + "="*70)
        print("BUS DEPOT OPTIMIZATION SUMMARY - WITH FLEX")
        print("="*70)
        
        total_cost = sum(pyo.value(instance.variable_cost[t]) for t in time_steps)
        print(f"Total Cost: ${total_cost:.2f}")
        
        max_power = max([pyo.value(instance.total_power_input[t]) for t in time_steps])
        print(f"Peak Power: {max_power:.2f} kW")
        
        # Load shift summary
        if hasattr(instance, 'load_shift_pos') and hasattr(instance, 'load_shift_neg'):
            total_pos_shift = sum(pyo.value(instance.load_shift_pos[t]) for t in time_steps)
            total_neg_shift = sum(pyo.value(instance.load_shift_neg[t]) for t in time_steps)
            print(f"Total Positive Load Shift: {total_pos_shift:.2f} kW")
            print(f"Total Negative Load Shift: {total_neg_shift:.2f} kW")
            print(f"Net Load Shift: {total_pos_shift - total_neg_shift:.2f} kW")
        
        # CS Utilization
        print("\nCharging Station Utilization:")
        for cs in charging_stations:
            total_energy = sum(
                pyo.value(instance.dsm_blocks[cs].discharge[t]) for t in time_steps
            )
            max_possible = pyo.value(instance.dsm_blocks[cs].max_power) * len(time_steps)
            utilization = (total_energy / max_possible * 100) if max_possible > 0 else 0
            print(f"  {cs}: {total_energy:.1f} kWh delivered, {utilization:.1f}% capacity utilization")
        
        print("="*70 + "\n")

    def symmetric_flexible_block(self, model):
        """
        Cost-based CRM flexibility (rolling 4-hour blocks) for charging stations.
        Bids must be in integer multiples of `min_bid_MW` (e.g., 1 MW).
        Feasibility is enforced hour-by-hour via headroom/footroom constraints.
        """
        block_length = 4
        min_bid_MW = 1.0
        time_steps = list(sorted(model.time_steps))
        T = len(time_steps)
        possible_starts = [time_steps[i] for i in range(T - block_length + 1)]

        # ---- STATIC CHARGING STATION MAX/MIN CAPACITY ----
        max_plant_capacity = 0.0
        min_plant_capacity = 0.0
        
        # For charging stations (bus depot technology)
        if self.technology == "bus_depot":
            for cs in getattr(model, 'charging_stations', []):
                if cs in model.dsm_blocks:
                    max_plant_capacity += model.dsm_blocks[cs].max_power
                    min_plant_capacity += getattr(model.dsm_blocks[cs], 'min_power', 0)
        
        self.max_plant_capacity = max_plant_capacity
        self.min_plant_capacity = min_plant_capacity

        # Conservative big-M for integer counts
        M_blocks = int(pyo.ceil(max_plant_capacity / min_bid_MW)) if max_plant_capacity > 0 else 1

        # ---- VARIABLES ----
        # integer number of min-bid units; enforces 1‑MW step size automatically
        model.n_blocks_up = pyo.Var(possible_starts, within=pyo.NonNegativeIntegers)
        model.n_blocks_down = pyo.Var(possible_starts, within=pyo.NonNegativeIntegers)

        # actual continuous volumes linked to the integer counts
        model.block_up = pyo.Var(possible_starts, within=pyo.NonNegativeReals)
        model.block_down = pyo.Var(possible_starts, within=pyo.NonNegativeReals)

        # direction binaries (only to enforce "up or down, not both")
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
                # For charging stations (bus depot technology)
                if self.technology == "bus_depot":
                    for cs in getattr(model, 'charging_stations', []):
                        if cs in model.dsm_blocks and hasattr(model.dsm_blocks[cs], 'discharge'):
                            total_power += model.dsm_blocks[cs].discharge[tau]

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

        # ---- OBJECTIVE (maximize upward flexibility) ----
        @model.Objective(sense=pyo.maximize)
        def obj_rule_flex(m):
            return sum(m.block_up[t] for t in possible_starts)

