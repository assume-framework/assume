# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from collections.abc import Callable

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import (
    SolverFactory,
    SolverStatus,
    TerminationCondition,
    check_available_solvers,
)

from assume.common.fast_pandas import FastSeries
from assume.units.dst_components import demand_side_technologies

SOLVERS = ["gurobi", "appsi_highs", "glpk", "cbc", "cplex"]

logger = logging.getLogger(__name__)


class DSMFlex:
    # Mapping of flexibility measures to their respective functions
    flexibility_map: dict[str, Callable[[pyo.ConcreteModel], None]] = {
        "cost_based_load_shift": lambda self, model: self.cost_based_flexibility(model),
        "electricity_price_signal_based_flexibility": lambda self,
        model: self.electricity_price_signal_based_flexibility(model),
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
        solver = solver if solver in solvers else solvers[0]
        if solver == "gurobi":
            self.solver_options = {"LogToConsole": 0, "OutputFlag": 0}
        elif solver == "appsi_highs":
            self.solver_options = {"output_flag": False, "log_to_console": False}
        else:
            self.solver_options = {}
        self.solver = SolverFactory(solver)

        self.initialize_solver()

    def initialize_solver(self, solver=None):
        # Define a solver
        solvers = check_available_solvers(*SOLVERS)
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
            elif self.technology == "cement_plant":
                power_input = 0
                if self.has_raw_mill:
                    power_input += self.model.dsm_blocks["raw_material_mill"].power_in[
                        t
                    ]
                if self.has_cement_mill:
                    power_input += self.model.dsm_blocks["cement_mill"].power_in[t]
                if self.has_clinker_system:
                    power_input += self.model.dsm_blocks["clinker_system"].power_in[t]
                if self.has_ccs_system:
                    power_input += self.model.dsm_blocks["ccs_system"].power_in[t]
                if self.has_electrolyser:
                    power_input += self.model.dsm_blocks["electrolyser"].power_in[t]
                return (
                    m.total_power_input[t] + m.load_shift_pos[t] - m.load_shift_neg[t]
                    == power_input
                )

    def electricity_price_signal_based_flexibility(self, model):
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
            if self.has_electrolyser:
                return (
                    m.total_power_input[t] + m.load_shift_pos[t] - m.load_shift_neg[t]
                    == self.model.dsm_blocks["electrolyser"].power_in[t]
                    + self.model.dsm_blocks["eaf"].power_in[t]
                    + self.model.dsm_blocks["dri_plant"].power_in[t]
                )
            else:
                return (
                    m.total_power_input[t] + m.load_shift_pos[t] - m.load_shift_neg[t]
                    == self.model.dsm_blocks["eaf"].power_in[t]
                    + self.model.dsm_blocks["dri_plant"].power_in[t]
                )

    def peak_load_shifting_flexibility(self, model):
        """
        Implements constraints for peak load shifting flexibility by identifying peak periods
        and allowing load shifts from peak to off-peak periods within a cost tolerance.

        Args:
            model (pyomo.ConcreteModel): The Pyomo model being optimized.
        """

        max_load = max(self.opt_power)

        peak_load_cap_value = max_load * (
            self.peak_load_cap / 100
        )  # E.g., 10% threshold
        # Add peak_threshold_value as a Param on the model so it can be accessed elsewhere
        model.peak_load_cap_value = pyo.Param(initialize=peak_load_cap_value)

        # Parameters
        # model.cost_tolerance = pyo.Param(initialize=self.cost_tolerance)
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
            if self.technology == "cement_plant":
                power_input = 0
                if self.has_raw_mill:
                    power_input += self.model.dsm_blocks["raw_material_mill"].power_in[
                        t
                    ]
                if self.has_cement_mill:
                    power_input += self.model.dsm_blocks["cement_mill"].power_in[t]
                if self.has_clinker_system:
                    power_input += self.model.dsm_blocks["clinker_system"].power_in[t]
                if self.has_ccs_system:
                    power_input += self.model.dsm_blocks["ccs_system"].power_in[t]
                if self.has_electrolyser:
                    power_input += self.model.dsm_blocks["electrolyser"].power_in[t]
                return (
                    m.total_power_input[t] + m.load_shift_pos[t] - m.load_shift_neg[t]
                    == power_input
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
            
            if self.technology == "cement_plant":
                power_input = 0
                if self.has_raw_mill:
                    power_input += self.model.dsm_blocks["raw_material_mill"].power_in[
                        t
                    ]
                if self.has_cement_mill:
                    power_input += self.model.dsm_blocks["cement_mill"].power_in[t]
                if self.has_clinker_system:
                    power_input += self.model.dsm_blocks["clinker_system"].power_in[t]
                if self.has_ccs_system:
                    power_input += self.model.dsm_blocks["ccs_system"].power_in[t]
                if self.has_electrolyser:
                    power_input += self.model.dsm_blocks["electrolyser"].power_in[t]
                return (
                     power_input <= peak_load_cap_value
                )

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

    def determine_optimal_operation_without_flex(self, switch_flex_off=True):
        """
        Determines the optimal operation of the steel plant without considering flexibility.
        """
        # create an instance of the model
        instance = self.model.create_instance()
        # switch the instance to the optimal mode by deactivating the flexibility constraints and objective
        if switch_flex_off:
            instance = self.switch_to_opt(instance)
        if switch_flex_off:
            instance = self.switch_to_opt(instance)
        # solve the instance
        # self.solver_options["mipgap"] = 0.1
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

        # Extract time series data for variable cost and total power input
        # time_steps = list(instance.time_steps)
        # variable_cost_series = [
        #     pyo.value(instance.variable_cost[t]) for t in time_steps
        # ]
        # total_power_input_series = [
        #     pyo.value(instance.total_power_input[t]) for t in time_steps
        # ]

        # # Save time series data to attributes
        # self.opt_power_requirement = FastSeries(
        #     index=self.index, value=total_power_input_series
        # )
        # self.variable_cost_series = FastSeries(
        #     index=self.index, value=variable_cost_series
        # )

        # # Save to Excel
        # data = {
        #     "Time Step": time_steps,
        #     "Variable Cost": variable_cost_series,
        #     "Total Power Input": total_power_input_series,
        # }
        # df = pd.DataFrame(data)
        # df.to_excel("./examples/outputs/opt_power_requirement.xlsx", index=False)
        # logger.debug(
        #     f"Time series data saved to {"./examples/outputs/opt_power_requirement.xlsx"}"
        # )

        # # Calculate total cost
        # self.total_cost = sum(variable_cost_series)

        # # Extract power input for raw material mill, clinker system, and cement mill

        # electrolyser_power_in = (
        #     [
        #         pyo.value(instance.dsm_blocks["electrolyser"].power_in[t])
        #         for t in instance.time_steps
        #     ]
        #     if "electrolyser" in instance.dsm_blocks
        #     else None
        # )

        # clinker_system_power_in = (
        #     [
        #         pyo.value(instance.dsm_blocks["clinker_system"].power_in[t])
        #         for t in instance.time_steps
        #     ]
        #     if "clinker_system" in instance.dsm_blocks
        #     else None
        # )

        # cement_mill_power_in = (
        #     [
        #         pyo.value(instance.dsm_blocks["cement_mill"].power_in[t])
        #         for t in instance.time_steps
        #     ]
        #     if "cement_mill" in instance.dsm_blocks
        #     else None
        # )

        # # ccs_system_power_in = (
        # #     [
        # #         pyo.value(instance.dsm_blocks["ccs_system"].power_in[t])
        # #         for t in instance.time_steps
        # #     ]
        # #     if "ccs_system" in instance.dsm_blocks
        # #     else None
        # # )
        # storage_charge = (
        #     [
        #         pyo.value(instance.dsm_blocks["hydrogen_buffer_storage"].charge[t])
        #         for t in instance.time_steps
        #     ]
        #     if "hydrogen_buffer_storage" in instance.dsm_blocks
        #     else None
        # )

        # storage_discharge = (
        #     [
        #         -pyo.value(instance.dsm_blocks["hydrogen_buffer_storage"].discharge[t])
        #         for t in instance.time_steps
        #     ]
        #     if "hydrogen_buffer_storage" in instance.dsm_blocks
        #     else None
        # )

        # storage_soc = (
        #     [
        #         pyo.value(instance.dsm_blocks["hydrogen_buffer_storage"].soc[t])
        #         for t in instance.time_steps
        #     ]
        #     if "hydrogen_buffer_storage" in instance.dsm_blocks
        #     else None
        # )

        # # Plot the power input
        # time_steps = range(len(instance.time_steps))
        # plt.figure(figsize=(10, 20))

        # if electrolyser_power_in:
        #     # Middle subplot: Clinker System
        #     plt.subplot(4, 1, 1)
        #     plt.plot(
        #         time_steps,
        #         electrolyser_power_in,
        #         label="electrolyser Power input",
        #         color="green",
        #     )
        #     plt.title("electrolyser power input")
        #     plt.xlabel("Time Steps")
        #     plt.ylabel("Power (MW)")
        #     plt.legend()

        # if clinker_system_power_in:
        #     # Bottom subplot: Cement Mill
        #     plt.subplot(4, 1, 2)
        #     plt.plot(
        #         time_steps,
        #         clinker_system_power_in,
        #         label="clinker_system_power_in",
        #         color="blue",
        #     )
        #     plt.title("clinker_system_power_in")
        #     plt.xlabel("Time Steps")
        #     plt.ylabel("Power (MW)")
        #     plt.legend()

        # if cement_mill_power_in:
        #     # Top subplot: Raw Material Mill
        #     plt.subplot(4, 1, 3)
        #     plt.plot(
        #         time_steps,
        #         cement_mill_power_in,
        #         label="cement_mill_power_in",
        #         color="orange",
        #     )
        #     plt.title("cement_mill_power_in")
        #     plt.xlabel("Time Steps")
        #     plt.ylabel("Power (MW)")
        #     plt.legend()

        # if storage_charge and storage_discharge and storage_soc:
        #     # Subplot 4: Hydrogen Storage
        #     plt.subplot(4, 1, 4)
        #     plt.plot(
        #         time_steps,
        #         storage_charge,
        #         label="Charge",
        #         color="green",
        #         linestyle="solid",
        #     )
        #     plt.plot(
        #         time_steps,
        #         [-x for x in storage_discharge],  # Invert discharge for clarity
        #         label="Discharge",
        #         color="red",
        #         linestyle="solid",
        #     )
        #     plt.fill_between(
        #         time_steps,
        #         storage_soc,
        #         alpha=0.3,
        #         label="State of Charge (SOC)",
        #         color="blue",
        #     )
        #     plt.title("Clinker Buffer Storage")
        #     plt.xlabel("Time Steps")
        #     plt.ylabel("Clinker buffer (units)")
        #     plt.legend()

        # if ccs_system_power_in:
        #     # Top subplot: Raw Material Mill
        #     plt.subplot(4, 1, 3)
        #     plt.plot(
        #         time_steps,
        #         ccs_system_power_in,
        #         label="ccs_system_power_in",
        #         color="black",
        #     )
        #     plt.title("ccs_system_power_in")
        #     plt.xlabel("Time Steps")
        #     plt.ylabel("Power (MW)")
        #     plt.legend()

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

        self.solver_options["mipgap"] = 0.1  # Allows up to 10% deviation from optimal
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

        if self.flexibility_measure == "electricity_price_signal_based_flexibility":
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

        # Save to Excel
        time_steps = list(instance.time_steps)
        data = {
            "Time Step": time_steps,
            "opt_power": self.opt_power,
            "flex_power": self.flex_power_requirement,
        }
        df = pd.DataFrame(data)
        df.to_excel(
            "./examples/inputs/paper/output/opt_power_requirement.xlsx",
            index=False,
        )
        logger.debug(
            f"Time series data saved to {"./examples/inputs/paper/output/opt_power_requirement.xlsx"}"
        )

    def switch_to_opt(self, instance):
        """
        Switches the instance to solve a cost based optimisation problem by deactivating the flexibility constraints and objective.

        Args:
            instance (pyomo.ConcreteModel): The instance of the Pyomo model.

        Returns:
            pyomo.ConcreteModel: The modified instance with flexibility constraints and objective deactivated.
        """
        if self.flexibility_measure == "electricity_price_signal_based_flexibility":
            instance.obj_rule_flex.deactivate()
            return instance

        else:
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
        if self.flexibility_measure == "electricity_price_signal_based_flexibility":
            instance.obj_rule_opt.deactivate()
            return instance
        else:
            instance.obj_rule_opt.deactivate()
            instance.total_power_input_constraint.deactivate()

            # fix values of model.total_power_input
            for t in instance.time_steps:
                instance.total_power_input[t].fix(self.opt_power_requirement.iloc[t])
            instance.total_cost = self.total_cost

            return instance
