# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

from assume.common.fast_pandas import FastSeries
from assume.units.dst_components import demand_side_technologies

logger = logging.getLogger(__name__)


class DSMFlex:
    def __init__(self, components, **kwargs):
        super().__init__(**kwargs)

        self.components = components

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

    def switch_to_opt(self, instance):
        """
        Switches the instance to solve a cost based optimisation problem by deactivating the flexibility constraints and objective.

        Args:
            instance (pyomo.ConcreteModel): The instance of the Pyomo model.

        Returns:
            pyomo.ConcreteModel: The modified instance with flexibility constraints and objective deactivated.
        """
        # deactivate the flexibility constraints and objective
        instance.obj_rule_flex.deactivate()

        instance.total_cost_upper_limit.deactivate()
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

    def determine_optimal_operation_without_flex(self):
        """
        Determines the optimal operation of the steel plant without considering flexibility.
        """
        # create an instance of the model
        instance = self.model.create_instance()
        # switch the instance to the optimal mode by deactivating the flexibility constraints and objective
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

        flex_power_requirement = [
            pyo.value(instance.total_power_input[t]) for t in instance.time_steps
        ]
        self.flex_power_requirement = FastSeries(
            index=self.index, value=flex_power_requirement
        )

        # Variable cost series
        flex_variable_cost = [
            instance.variable_cost[t].value for t in instance.time_steps
        ]
        self.flex_variable_cost_series = FastSeries(
            index=self.index, value=flex_variable_cost
        )

    def flexibility_cost_tolerance(self, model):
        """
        Modify the optimization model to include constraints for flexibility within cost tolerance.
        """

        model.cost_tolerance = pyo.Param(initialize=(self.cost_tolerance))
        model.total_cost = pyo.Param(initialize=0.0, mutable=True)

        # Variables
        model.load_shift = pyo.Var(model.time_steps, within=pyo.Reals)

        @model.Constraint(model.time_steps)
        def total_cost_upper_limit(m, t):
            return sum(
                model.variable_cost[t] for t in model.time_steps
            ) <= model.total_cost * (1 + (model.cost_tolerance / 100))

        @model.Constraint(model.time_steps)
        def total_power_input_constraint_with_flex(m, t):
            # Apply constraints based on the technology type
            if self.technology == "hydrogen_plant":
                # Hydrogen plant constraint
                return (
                    m.total_power_input[t] - m.load_shift[t]
                    == self.model.dsm_blocks["electrolyser"].power_in[t]
                )
            elif self.technology == "steel_plant":
                # Steel plant constraint with conditional electrolyser inclusion
                if self.has_electrolyser:
                    return (
                        m.total_power_input[t] - m.load_shift[t]
                        == self.model.dsm_blocks["electrolyser"].power_in[t]
                        + self.model.dsm_blocks["eaf"].power_in[t]
                        + self.model.dsm_blocks["dri_plant"].power_in[t]
                    )
                else:
                    return (
                        m.total_power_input[t] - m.load_shift[t]
                        == self.model.dsm_blocks["eaf"].power_in[t]
                        + self.model.dsm_blocks["dri_plant"].power_in[t]
                    )
