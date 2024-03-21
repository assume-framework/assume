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
        self.iron_ore_price = self.forecaster.get_price('iron_ore')
        self.steel_demand = demand #/ 672
        self.steel_price = self.forecaster.get_price('steel')
        self.lime_co2_factor = self.forecaster.get_price('lime_co2_factor')
        self.lime_price = self.forecaster.get_price('lime')
        self.co2_price = self.forecaster.get_price('co2')

        self.recalculated_power = self.forecaster[f"{self.id}_recalculated_power"]

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
        if self.objective == "recalculate":
            flex.recalculate_with_accepted_offers(self)
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
        self.model.steel_demand = pyo.Param(initialize=self.steel_demand)
        self.model.steel_price = pyo.Param(initialize=self.steel_price.mean(), within=pyo.NonNegativeReals)
        self.model.lime_co2_factor = pyo.Param(initialize=self.lime_co2_factor.mean(), within=pyo.NonNegativeReals)
        self.model.co2_price = pyo.Param(initialize=self.co2_price.mean(), within=pyo.NonNegativeReals)
        self.model.lime_price = pyo.Param(initialize=self.lime_price.mean(), within=pyo.NonNegativeReals)
        self.model.iron_ore_price = pyo.Param(initialize=self.iron_ore_price.mean(), within=pyo.NonNegativeReals)

    def define_variables(self):
        self.model.total_power_input = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        ) 
        self.model.variable_cost = pyo.Var(

            self.model.time_steps, within=pyo.NonNegativeReals
        )

    def define_constraints(self):
        @self.model.Constraint(self.model.time_steps)
        def steel_output_association_constraint(m, t):
            return sum(self.components["eaf"].b.steel_output[t] for t in self.model.time_steps) == self.model.steel_demand
        
        # @self.model.Constraint(self.model.time_steps)
        # def steel_output_association_constraint(m, t):
        #     return self.components["eaf"].b.steel_output[t] == self.model.steel_demand

        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
            if self.objective == 'max_flexibility':
                return pyo.Constraint.Skip
            else:
                return (
                    m.total_power_input[t]
                    == self.components["electrolyser"].b.power_in[t] + \
                        self.components["eaf"].b.power_eaf[t] + self.components["dri_plant"].b.power_dri[t]
                )
        
        @self.model.Constraint(self.model.time_steps)
        def cost_per_time_step(m, t):
            return self.model.variable_cost[t] == \
                self.components["electrolyser"].b.start_cost[t] + \
                    self.components["electrolyser"].b.electricity_cost[t] + \
                    self.components["dri_plant"].b.dri_operating_cost[t] + \
                    self.components["eaf"].b.eaf_operating_cost[t]
            
    def define_objective(self):
        if self.objective == "min_variable_cost" or "recalculate":

            @self.model.Objective(sense=pyo.minimize)
            def obj_rule(m):
                # Sum up the variable cost over all time steps
                total_variable_cost = sum(self.model.variable_cost[t] for t in self.model.time_steps)
                
                return total_variable_cost
                
        elif self.objective == "max_flexibility":
            @self.model.Objective(sense=pyo.maximize)
            def obj_rule(m):
                maximise_flexibility = sum(m.load_shift[t] for t in self.model.time_steps)
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
        
        total_variable_costs = {
            t: (
                value(self.model.variable_cost[t])
            )
            for t in self.model.time_steps
        }

        # Instantiate an OperationStatesForecaster object
        operation_states_forecaster = OperationStatesForecaster(index=self.index)

        for time_step, power_input in total_power_input.items():
            unit_id = self.id
            prefixed_total_power_input = {f"{unit_id}_power": [power_input]}
            operation_states_forecaster.set_operation_states(time_step, pd.DataFrame(prefixed_total_power_input, index=[0]))
     
        # Extract power values for electrolyser, eaf, and dri for each time step
        for t in self.model.time_steps:
            # Electrolyser power input
            electrolyser_power_input_value = value(self.components["electrolyser"].b.power_in[t])
            prefixed_electrolyser_power_input = {f"{unit_id}_electrolyser": electrolyser_power_input_value}
            operation_states_forecaster.set_operation_states(t, pd.DataFrame(prefixed_electrolyser_power_input, index=[0]))
            
            # DRI power input
            dri_power_input_value = value(self.components["dri_plant"].b.power_dri[t])
            prefixed_dri_power_input = {f"{unit_id}_dri": dri_power_input_value}
            operation_states_forecaster.set_operation_states(t, pd.DataFrame(prefixed_dri_power_input, index=[0]))
            
            # EAF power input
            eaf_power_input_value = value(self.components["eaf"].b.power_eaf[t])
            prefixed_eaf_power_input = {f"{unit_id}_eaf": eaf_power_input_value}
            operation_states_forecaster.set_operation_states(t, pd.DataFrame(prefixed_eaf_power_input, index=[0]))

            # EAF steel output
            eaf_steel_output_value = value(self.components["eaf"].b.steel_output[t])
            prefixed_eaf_steel_output = {f"{unit_id}_steel_output": eaf_steel_output_value}
            operation_states_forecaster.set_operation_states(t, pd.DataFrame(prefixed_eaf_steel_output, index=[0]))

            if "dri_storage" in self.components:

                # DRI Storage
                dri_charge_value = value(self.components["dri_storage"].b.charge_dri[t])
                prefixed_dri_charge_value = {f"{unit_id}_dri_charge": dri_charge_value}
                operation_states_forecaster.set_operation_states(t, pd.DataFrame(prefixed_dri_charge_value, index=[0]))

                dri_discharge_value = value(self.components["dri_storage"].b.discharge_dri[t])
                prefixed_dri_discharge_value = {f"{unit_id}_dri_discharge": dri_discharge_value}
                operation_states_forecaster.set_operation_states(t, pd.DataFrame(prefixed_dri_discharge_value, index=[0]))

                dri_soc_value = value(self.components["dri_storage"].b.soc_dri[t])
                prefixed_dri_soc_value = {f"{unit_id}_dri_soc": dri_soc_value}
                operation_states_forecaster.set_operation_states(t, pd.DataFrame(prefixed_dri_soc_value, index=[0]))

            if "h2storage" in self.components:

                # H2 Storage
                charge_value = value(self.components["h2storage"].b.charge[t])
                prefixed_charge_value = {f"{unit_id}_h2_charge": charge_value}
                operation_states_forecaster.set_operation_states(t, pd.DataFrame(prefixed_charge_value, index=[0]))

                discharge_value = value(self.components["h2storage"].b.discharge[t])
                prefixed_discharge_value = {f"{unit_id}_h2_discharge": discharge_value}
                operation_states_forecaster.set_operation_states(t, pd.DataFrame(prefixed_discharge_value, index=[0]))

                soc_value = value(self.components["h2storage"].b.soc[t])
                prefixed_soc_value = {f"{unit_id}_h2_soc": soc_value}
                operation_states_forecaster.set_operation_states(t, pd.DataFrame(prefixed_soc_value, index=[0]))

        # Set the total cost data for each unit with unit ID as prefix
        for time_step, total_variable_cost in total_variable_costs.items():
            unit_id = self.id
            prefixed_total_variable_cost = {f"{unit_id}_variable_cost": [total_variable_cost]}
            operation_states_forecaster.set_operation_states(time_step, pd.DataFrame(prefixed_total_variable_cost, index=[0]))

        # Save the operation states data to a CSV file
        operation_states_forecaster.save_operation_states(path="C:\\Manish_REPO\\ASSUME\\examples\\inputs\\example_04", unit=unit_id)


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
            print(f"The objective value is: {objective_value}")

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
        load_shifts = {t: value(self.model.load_shift[t]) for t in self.model.time_steps}
        # Calculate total energy consumption for the current time step

        variable_costs = {t: value(self.model.variable_cost[t]) for t in self.model.time_steps}

        # Instantiate an OperationStatesForecaster object
        operation_states_forecaster = OperationStatesForecaster(index=self.index)

        # Loop over time steps
        for time_step in self.model.time_steps:
            unit_id = self.id
            total_variable_cost = variable_costs[time_step]
            eaf_steel_output_value = value(self.components["eaf"].b.steel_output[time_step])

            # Calculate total energy consumption for the current time step
            total_power_consumption = (
                value(self.components["electrolyser"].b.power_in[time_step])
                + value(self.components["eaf"].b.power_eaf[time_step])
                + value(self.components["dri_plant"].b.power_dri[time_step])
            )

            # Calculate marginal cost per unit of energy
            if total_power_consumption > 0:
                marginal_cost_per_unit_power = total_variable_cost / total_power_consumption
            else:
                marginal_cost_per_unit_power = 0  # Avoid division by zero

            positive_reserve = (value(self.model.prev_power[time_step])- total_power_consumption)
            negetive_reserve = (total_power_consumption - value(self.model.prev_power[time_step]))

            threshold = 1e-10
            # Adjust negative flex to zero if it's below the threshold

            if positive_reserve < threshold:
                positive_reserve = 0
            if negetive_reserve < threshold:
                negetive_reserve = 0
            
            prefixed_power = {f"{unit_id}_power": [total_power_consumption]}
            prefixed_variable_cost = {f"{unit_id}_total_variable_cost": [total_variable_cost]}
            prefixed_pos_res = {f"{unit_id}_pos_res": [positive_reserve]}
            prefixed_neg_res = {f"{unit_id}_neg_res": [negetive_reserve]}
            # EAF steel output
            prefixed_eaf_steel_output = {f"{unit_id}_steel_output": eaf_steel_output_value}

            

            # Save marginal costs in the operation states
            operation_states_forecaster.set_operation_states(
                time_step,
                pd.DataFrame(
                    {f"{unit_id}_marginal_cost": [marginal_cost_per_unit_power]},
                    index=[0],
                ),
            )

            if "dri_storage" in self.components:

                # DRI Storage
                dri_charge_value = value(self.components["dri_storage"].b.charge_dri[time_step])
                prefixed_dri_charge_value = {f"{unit_id}_dri_charge": dri_charge_value}

                dri_discharge_value = value(self.components["dri_storage"].b.discharge_dri[time_step])
                prefixed_dri_discharge_value = {f"{unit_id}_dri_discharge": dri_discharge_value}

                dri_soc_value = value(self.components["dri_storage"].b.soc_dri[time_step])
                prefixed_dri_soc_value = {f"{unit_id}_dri_soc": dri_soc_value}

            if "h2storage" in self.components:

                # H2 Storage
                charge_value = value(self.components["h2storage"].b.charge[time_step])
                prefixed_charge_value = {f"{unit_id}_h2_charge": charge_value}

                discharge_value = value(self.components["h2storage"].b.discharge[time_step])
                prefixed_discharge_value = {f"{unit_id}_h2_discharge": discharge_value}

                soc_value = value(self.components["h2storage"].b.soc[time_step])
                prefixed_soc_value = {f"{unit_id}_h2_soc": soc_value}

            operation_states_data = pd.DataFrame(
                {**prefixed_power, **prefixed_variable_cost, **prefixed_pos_res, **prefixed_neg_res, **prefixed_eaf_steel_output,
                 **prefixed_dri_charge_value, **prefixed_dri_discharge_value,**prefixed_dri_soc_value,
                 **prefixed_charge_value, **prefixed_discharge_value, **prefixed_soc_value},
                index=[0],
            )

            operation_states_forecaster.set_operation_states(time_step, operation_states_data)

        # Save the operation states data to a CSV file
        operation_states_forecaster.save_operation_states_with_flex(
            path="C:\\Manish_REPO\\ASSUME\\examples\\inputs\\example_04", unit=unit_id)

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
            )
            
            # Calculate total energy consumption for the current time step
            total_energy_consumption = (
                value(self.components["electrolyser"].b.power_in[t])
                + value(self.components["eaf"].b.power_eaf[t])
                + + value(self.components["dri_plant"].b.power_dri[t])
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
