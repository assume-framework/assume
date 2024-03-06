# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from typing import Dict, List

import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import value
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

import assume.common.flexibility as flex
from assume.common.base import SupportsMinMax
from assume.common.forecasts import OperationStatesForecaster, Forecaster, CsvForecaster
from assume.units.dst_components import (
    DriPlant,
    DRIStorage,
    ElectricArcFurnace,
    Electrolyser,
    GenericStorage,
)

logger = logging.getLogger(__name__)

# Mapping of component type identifiers to their respective classes
dst_components = {
    "electrolyser": Electrolyser,
    "h2storage": GenericStorage,
    "dri_plant": DriPlant,
    "dri_storage": DRIStorage,
    "eaf": ElectricArcFurnace,
}


class SteelPlant(SupportsMinMax):
    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        technology: str = "steel_plant",
        node: str = "bus0",
        index: pd.DatetimeIndex = None,
        location: tuple[float, float] = (0.0, 0.0),
        components: Dict[str, Dict] = None,
        objective: str = None,
        demand: float = 0,
        cost_tolerance: float = 10,
        **kwargs,
    ):
        super().__init__(
            id=id,
            unit_operator=unit_operator,
            technology=technology,
            bidding_strategies=bidding_strategies,
            index=index,
            node=node,
            **kwargs,
        )

        self.natural_gas_price = self.forecaster["fuel_price_natural_gas"]
        self.electricity_price = self.forecaster["price_EOM"]
        self.iron_ore_price = self.forecaster["iron_ore_price"]
        self.steel_demand = demand
        self.steel_price = self.forecaster.get_price('steel')
        self.dri_price = self.forecaster["dri_price"]

        self.location = location
        self.objective = objective
        self.cost_tolerance = cost_tolerance
        self.components = {}

        self.model = pyo.ConcreteModel()
        self.define_sets()
        self.define_parameters()

        # Initialize components based on the selected technology configuration
        self.initialize_components(components)
        self.initialize_process_sequence()
        self.define_variables()
        if self.objective == "max_flexibility":
            flex.flexibility_cost_tolerance(self)
        self.define_constraints()
        self.define_objective()

        self.power_requirement = None

    def initialize_components(self, components: Dict[str, Dict]):
        for technology, component_data in components.items():
            component_id = f"{self.id}_{technology}"
            if technology in dst_components:
                component_class = dst_components[technology]
                component_instance = component_class(
                    model=self.model, id=component_id, **component_data
                )

                # Call the add_to_model method for each component
                component_instance.add_to_model(self.model, self.model.time_steps)
                self.components[technology] = component_instance

    def initialize_process_sequence(self):
        # Assuming the presence of 'h2storage' indicates the desire for dynamic flow management
        has_h2storage = "h2storage" in self.components

        # Constraint for direct hydrogen flow from Electrolyser to dri plant
        @self.model.Constraint(self.model.time_steps)
        def direct_hydrogen_flow_constraint(m, t):
            # This constraint allows part of the hydrogen produced by the dri plant to go directly to the EAF
            # The actual amount should ensure that it does not exceed the capacity or demand of the EAF
            if has_h2storage:
                return (
                    self.components["electrolyser"].b.hydrogen_out[t]
                    + self.components["h2storage"].b.discharge[t]
                    == self.components["dri_plant"].b.hydrogen_in[t]
                    + self.components["h2storage"].b.charge[t]
                )
            else:
                return (
                    self.components["electrolyser"].b.hydrogen_out[t]
                    >= self.components["dri_plant"].b.hydrogen_in[t]
                )

        # Assuming the presence of dristorage' indicates the desire for dynamic flow management
        has_dristorage = "dri_storage" in self.components

        # Constraint for direct hydrogen flow from Electrolyser to dri plant
        @self.model.Constraint(self.model.time_steps)
        def direct_dri_flow_constraint(m, t):
            # This constraint allows part of the dri produced by the dri plant to go directly to the dri storage
            # The actual amount should ensure that it does not exceed the capacity or demand of the EAF
            if has_dristorage:
                return (
                    self.components["dri_plant"].b.dri_output[t]
                    + self.components["dri_storage"].b.discharge_dri[t]
                    == self.components["eaf"].b.dri_input[t]
                    + self.components["dri_storage"].b.charge_dri[t]
                )
            else:
                return (
                    self.components["dri_plant"].b.dri_output[t]
                    == self.components["eaf"].b.dri_input[t]
                )

        # Constraint for material flow from dri plant to Electric Arc Furnace
        @self.model.Constraint(self.model.time_steps)
        def shaft_to_arc_furnace_material_flow_constraint(m, t):
            return (
                self.components["dri_plant"].b.dri_output[t]
                == self.components["eaf"].b.dri_input[t]
            )

    def define_sets(self) -> None:
        self.model.time_steps = pyo.Set(
            initialize=[idx for idx, _ in enumerate(self.index)]
        )

    def define_parameters(self):
        self.model.electricity_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.electricity_price)},
        )
        self.model.natural_gas_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.natural_gas_price)},
        )
        self.model.iron_ore_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.iron_ore_price)},
        )
        self.model.dri_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.dri_price)},
        )
        self.model.steel_demand = pyo.Param(initialize=self.steel_demand)
        self.model.steel_price = pyo.Param(initialize=self.steel_price.mean(), within=pyo.NonNegativeReals)

    def define_variables(self):
        self.model.total_power_input = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        ) 
        self.model.variable_cost = pyo.Var(

            self.model.time_steps, within=pyo.NonNegativeReals
        )

    def define_constraints(self):
        @self.model.Constraint(self.model.time_steps)
        def dri_output_association_constraint(m, t):
            return sum(self.components["eaf"].b.steel_output[t] for t in self.model.time_steps) >= self.model.steel_demand

        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
            return (
                m.total_power_input[t]
                == self.components["electrolyser"].b.power_in[t] + self.components["eaf"].b.power_eaf[t]
            )
        
        @self.model.Constraint(self.model.time_steps)
        def cost_per_time_step(m, t):
            return self.model.variable_cost[t] == \
                self.components["electrolyser"].b.start_cost[t] + \
                    self.components["electrolyser"].b.electricity_cost[t] + \
                    self.components["dri_plant"].b.dri_operating_cost[t] + \
                    self.components["eaf"].b.eaf_operating_cost[t] + \
                    self.iron_ore_price.iat[t] * self.components["dri_plant"].b.iron_ore_in[t]
            
    def define_objective(self):
        if self.objective == "min_variable_cost":

            @self.model.Objective(sense=pyo.minimize)
            def obj_rule(m):
                # Sum up the variable cost over all time steps
                total_variable_cost = sum(self.model.variable_cost[t] for t in self.model.time_steps)
                
                return total_variable_cost
                
        elif self.objective == "max_flexibility":
            @self.model.Objective(sense=pyo.maximize)
            def obj_rule(m):
                maximise_flexibility = sum(
                    self.model.positive_flex[t] 
                    +  self.model.negetive_flex[t]
                    for t in self.model.time_steps
                )
                return maximise_flexibility

        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    def determine_optimal_operation_without_flex(self):
        """
        Determines the optimal operation of the steel plant without considering flexibility.
        """
        # Create a solver
        solver = SolverFactory("gurobi")

        results = solver.solve(self.model, tee=False)  # , tee=True

        # Check solver status and termination condition
        if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal
        ):
            logger.debug("The model was solved optimally.")

            # Display the Objective Function Value
            objective_value = self.model.obj_rule()
            logger.debug(f"The value of the objective function is {objective_value}.")

        elif results.solver.termination_condition == TerminationCondition.infeasible:
            logger.debug("The model is infeasible.")

        else:
            logger.debug("Solver Status: ", results.solver.status)
            logger.debug(
                "Termination Condition: ", results.solver.termination_condition
            )

        temp = self.model.total_power_input.get_values()
        self.power_requirement = pd.Series(index=self.index, data=0.0)
        for i, date in enumerate(self.index):
            self.power_requirement.loc[date] = temp[i]

        total_power_input = {
            t: (
                value(self.model.total_power_input[t])
            )
            for t in self.model.time_steps
        }

        # Calculate the total cost of the optimal operation
        total_variable_costs = {
            t: (
                value(self.model.variable_cost[t])
            )
            for t in self.model.time_steps
        }

        # Instantiate an OperationStatesForecaster object
        operation_states_forecaster = OperationStatesForecaster(index=self.index)

        for time_step, total_power_input in total_power_input.items():
            unit_id = self.id
            prefixed_power = {f"{unit_id}_power": [total_power_input]}
            operation_states_forecaster.set_operation_states(time_step, pd.DataFrame(prefixed_power, index=[0]))

        # Set the total cost data for each unit with unit ID as prefix
        for time_step, total_variable_cost in total_variable_costs.items():
            unit_id = self.id
            prefixed_total_variable_cost = {f"{unit_id}_variable_cost": [total_variable_cost]}
            operation_states_forecaster.set_operation_states(time_step, pd.DataFrame(prefixed_total_variable_cost, index=[0]))

        # Save the operation states data to a CSV file
        operation_states_forecaster.save_operation_states(path="C:\\Manish_REPO\\ASSUME\\examples\\inputs\\example_04")

    def determine_optimal_operation_with_flex(self):
        """
        Determines the optimal operation of the steel plant without considering flexibility.
        """
        # Create a solver
        solver = SolverFactory("gurobi")

        results = solver.solve(self.model, tee=False)  # , tee=True

        # Check solver status and termination condition
        if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal
        ):
            logger.debug("The model was solved optimally.")

            # Display the Objective Function Value
            objective_value = self.model.obj_rule()
            logger.debug(f"The value of the objective function is {objective_value}.")

        elif results.solver.termination_condition == TerminationCondition.infeasible:
            logger.debug("The model is infeasible.")

        else:
            logger.debug("Solver Status: ", results.solver.status)
            logger.debug(
                "Termination Condition: ", results.solver.termination_condition
            )

        temp = self.model.total_power_input.get_values()
        self.power_requirement = pd.Series(index=self.index, data=0.0)
        for i, date in enumerate(self.index):
            self.power_requirement.loc[date] = temp[i]

        # Collect total power input, positive flex, and negative flex values
        total_power_input = {t: value(self.model.total_power_input[t]) for t in self.model.time_steps}
        positive_flex = {t: value(self.model.positive_flex[t]) for t in self.model.time_steps}
        negetive_flex = {t: value(self.model.negetive_flex[t]) for t in self.model.time_steps}
        variable_costs = {t: value(self.model.variable_cost[t]) for t in self.model.time_steps}

        # Instantiate an OperationStatesForecaster object
        operation_states_forecaster = OperationStatesForecaster(index=self.index)

        # Loop over time steps
        for time_step in self.model.time_steps:
            unit_id = self.id
            power_input = total_power_input[time_step]
            power_positive_flex = -positive_flex[time_step]
            power_negative_flex = negetive_flex[time_step]
            total_variable_cost = variable_costs[time_step]

            threshold = 1e-10
            # Adjust negative flex to zero if it's below the threshold
            if power_negative_flex < threshold:
                power_negative_flex = 0

            # Calculate total energy consumption for the current time step
            total_power_consumption = (
                value(self.components["electrolyser"].b.power_in[time_step])
                + value(self.components["eaf"].b.power_eaf[time_step])
            )

            # Calculate marginal cost per unit of energy
            if total_power_consumption > 0:
                marginal_cost_per_unit_power = total_variable_cost / total_power_consumption
            else:
                marginal_cost_per_unit_power = 0  # Avoid division by zero

            # Set the operation states data including total power input, positive flex, and negative flex
            prefixed_power = {f"{unit_id}_power": [power_input]}
            prefixed_positive_flex = {f"{unit_id}_positive_flex": [power_positive_flex]}
            prefixed_negative_flex = {f"{unit_id}_negative_flex": [power_negative_flex]}

            # Save marginal costs in the operation states
            operation_states_forecaster.set_operation_states(
                time_step,
                pd.DataFrame(
                    {f"{unit_id}_marginal_cost": [marginal_cost_per_unit_power]},
                    index=[0],
                ),
            )

            operation_states_data = pd.DataFrame(
                {**prefixed_power, **prefixed_positive_flex, **prefixed_negative_flex},
                index=[0],
            )

            operation_states_forecaster.set_operation_states(time_step, operation_states_data)

        # Save the operation states data to a CSV file
        operation_states_forecaster.save_operation_states_with_flex(
            path="C:\\Manish_REPO\\ASSUME\\examples\\inputs\\example_04"
        )

    def calculate_marginal_cost(self, start: pd.Timestamp, power: float) -> float:
        """
        Calculate the marginal cost of the unit returns the marginal cost of the unit based on the provided time and power.

        Args:
            start (pandas.Timestamp): The start time of the dispatch.
            power (float): The power output of the unit.

        Returns:
            float: the marginal cost of the unit for the given power.
        """
        for t in self.model.time_steps:
            # Calculate total variable costs for the current time step
            total_variable_costs = (
                value(self.components["electrolyser"].b.start_cost[t])
                + value(self.components["electrolyser"].b.electricity_cost[t])
                + value(self.components["dri_plant"].b.dri_operating_cost[t])
                + value(self.components["eaf"].b.eaf_operating_cost[t])
                + value(self.iron_ore_price.iat[t] * self.components["dri_plant"].b.iron_ore_in[t])
            )
            
            # Calculate total energy consumption for the current time step
            total_energy_consumption = (
                value(self.components["electrolyser"].b.power_in[t])
                + value(self.components["eaf"].b.power_eaf[t])
            )
            
            # Calculate marginal cost per unit of energy
            if total_energy_consumption > 0:
                marginal_cost_per_unit_energy = total_variable_costs / total_energy_consumption
            else:
                marginal_cost_per_unit_energy = 0  # Avoid division by zero
        
            return marginal_cost_per_unit_energy
        # return self.electricity_price.at[start]

    def as_dict(self) -> dict:
        """
        Returns the attributes of the unit as a dictionary, including specific attributes.

        Returns:
            dict: The attributes of the unit as a dictionary.
        """
        # Assuming unit_dict is a dictionary that you want to save to the database
        components_list = [component for component in self.components.keys()]

        # Convert the list to a delimited string
        components_string = ",".join(components_list)

        unit_dict = super().as_dict()
        unit_dict.update(
            {
                "unit_type": "steel_plant",
                "components": components_string,
            }
        )

        return unit_dict
