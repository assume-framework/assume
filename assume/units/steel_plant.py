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
from assume.common.forecasts import OperationStatesForecaster
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
        steel_demand: float = 2000,
        tolerance_percentage: float = 2.0,
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
        self.steel_demand = steel_demand
        self.dri_price = self.forecaster["dri_price"]
        self.holiday = self.forecaster["holiday"]

        self.location = location
        self.objective = objective
        self.tolerance_percentage = tolerance_percentage
        self.components = {}

        self.model = pyo.ConcreteModel()
        self.define_sets()
        self.define_parameters()

        # Initialize components based on the selected technology configuration
        self.initialize_components(components)
        self.initialize_process_sequence()

        self.define_variables()
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
                    >= self.components["dri_plant"].b.hydrogen_in[t]
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
                    >= self.components["eaf"].b.dri_input[t]
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

        self.model.holiday = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.holiday)},
        )

    def define_variables(self):
        self.model.total_power_input = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )

    def define_constraints(self):
        @self.model.Constraint(self.model.time_steps)
        def dri_output_association_constraint(m, t):
            return sum(self.components["eaf"].b.steel_output[t] for t in self.model.time_steps) >= self.model.steel_demand #self.steel_demand.iat[t]

        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
            return (
                m.total_power_input[t]
                == self.components["electrolyser"].b.power_in[t]
                + self.components["eaf"].b.power_eaf[t]
            )
    
        if self.objective == "max_flexibility":
            flex.modify_model_for_flexibility(self, id=self.id)

            
    def define_objective(self):
        if self.objective == "maximize_marginal_profit":

            @self.model.Objective(sense=pyo.maximize)
            def obj_rule(m):
                total_revenue = sum(
                    self.dri_price.iat[t] * m.aggregated_dri_output[t]
                    for t in m.time_steps
                )

                total_costs = sum(
                    self.electricity_price.iat[t]
                    * self.components["electrolyser"].b.power_in[t]
                    +
                    # self.hydrogen_price[t] * self.components['electrolyser'].b.hydrogen_out[t] +
                    self.iron_ore_price.iat[t]
                    * self.components["dri_plant"].b.iron_ore_in[t]
                    for t in m.time_steps
                )

                return total_revenue - total_costs

        elif self.objective == "min_total_cost":

            @self.model.Objective(sense=pyo.minimize)
            def obj_rule(m):
                total_costs = sum(
                    self.components["electrolyser"].b.start_cost[t]
                    + self.components["electrolyser"].b.electricity_cost[t]
                    + self.components["dri_plant"].b.dri_operating_cost[t]
                    + self.components["eaf"].b.eaf_operating_cost[t]
                    + self.iron_ore_price.iat[t]
                    * self.components["dri_plant"].b.iron_ore_in[t]
                    for t in m.time_steps
                )
                return total_costs
        elif self.objective == "max_flexibility":
            @self.model.Objective(sense=pyo.maximize)
            def obj_rule(m):
                flexibility_utilisation = sum(
                    self.model.max_ramp_up_electrolyser[t] 
                    +  self.model.max_ramp_down_electrolyser[t] 
                    +  self.model.max_ramp_up_eaf[t] 
                    +  self.model.max_ramp_down_eaf[t] for t in self.model.time_steps)

        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    def determine_optimal_operation(self):
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
        # Determine the optimal operation of the steel plant
        # optimal_operation = self.run_optimization()

        temp = self.model.total_power_input.get_values()
        self.power_requirement = pd.Series(index=self.index, data=0.0)
        for i, date in enumerate(self.index):
            self.power_requirement.loc[date] = temp[i]

        # Calculate the total cost of the optimal operation
        total_cost = sum(
        value(self.components["electrolyser"].b.start_cost[t])
                + value(self.components["electrolyser"].b.electricity_cost[t])
                + value(self.components["dri_plant"].b.dri_operating_cost[t])
                + value(self.components["eaf"].b.eaf_operating_cost[t])
                + value(self.iron_ore_price.iat[t] * self.components["dri_plant"].b.iron_ore_in[t])
                for t in self.model.time_steps
            )

         # Save the total cost using the OperationStatesForecaster method
        operation_states_forecaster = OperationStatesForecaster(index=self.index)
        operation_states_forecaster.save_total_cost(total_cost, self.id, path="C:\\Manish_REPO\\ASSUME\\examples\\inputs\\example_04")

        # Calculate the operational state of the Electric Arc Furnace
        eaf_operational_state = self.components["eaf"].calculate_operational_state(
            self.model.time_steps
        )

        # Calculate the operational state of the Electrolyser
        electrolyser_operational_state = self.components["electrolyser"].calculate_operational_state(
            self.model.time_steps
        )

        # Extract actual values from the Pyomo expressions and variables
        for time_step, state in eaf_operational_state.items():
            
            eaf_operational_state[time_step]["power"] = value(
                state["power"]
            )

        for time_step, state in electrolyser_operational_state.items():
            electrolyser_operational_state[time_step]["power"] = value(
                state["power"]
            )

        # Instantiate an OperationStatesForecaster object
        operation_states_forecaster = OperationStatesForecaster(index=self.index)

        # Set the operational state data for each unit
        for time_step, state_data in eaf_operational_state.items():
            unit_id = self.components["eaf"].id  # Get the ID of the Electric Arc Furnace
            prefixed_state_data = {f"{unit_id}_{key}": [value] for key, value in state_data.items()}
            operation_states_forecaster.set_operation_states(time_step, pd.DataFrame(prefixed_state_data, index=[0]))

        for time_step, state_data in electrolyser_operational_state.items():
            unit_id = self.components["electrolyser"].id  # Get the ID of the Electrolyser
            prefixed_state_data = {f"{unit_id}_{key}": [value] for key, value in state_data.items()}
            operation_states_forecaster.set_operation_states(time_step, pd.DataFrame(prefixed_state_data, index=[0]))

        # Save the operation states data to a CSV file
        operation_states_forecaster.save_operation_states(path="C:\\Manish_REPO\\ASSUME\\examples\\inputs\\example_04")

        results_dfs = []

        # Iterate over each time step
        for t in self.model.time_steps:
            steel_out = self.components['eaf'].b.steel_output[t].value
            power_in_eaf = self.components['eaf'].b.power_eaf[t].value
            power_in_electrolyser = self.components['electrolyser'].b.power_in[t].value
            hydrogen_out_electrolyser = self.components['electrolyser'].b.hydrogen_out[t].value
            natural_gas_in = self.components['dri_plant'].b.natural_gas_in[t].value

            dri_in_eaf = self.components['eaf'].b.dri_input[t].value
            dri_charge = self.components['dri_storage'].b.charge_dri[t].value
            dri_discharge = self.components['dri_storage'].b.discharge_dri[t].value
            dri_SOC = self.components['dri_storage'].b.soc_dri[t].value
            
            H2_in_dri = self.components['dri_plant'].b.hydrogen_in[t].value
            H2_charge = self.components['h2storage'].b.charge[t].value
            H2_discharge = self.components['h2storage'].b.discharge[t].value
            H2_SOC = self.components['h2storage'].b.soc[t].value
            # shutdown_electrolyser = self.components['electrolyser'].b.shut_down[t].value
            # in_operation_electrolyser = self.components['electrolyser'].b.in_operation[t].value
            # start_up_electrolyser = self.components['electrolyser'].b.start_up[t].value

            # Create a DataFrame for the current time step
            df = pd.DataFrame({
                'Time Step': [t],
                'Steel': [steel_out],
                'El. eaf': [power_in_eaf],
                'El. elec.': [power_in_electrolyser],
                'H2 Prod.': [hydrogen_out_electrolyser],
                'NH2_in.': [natural_gas_in],
                'DRI Charge': [dri_charge],
                'DRI Discharge': [dri_discharge],
                'DRI SOC': [dri_SOC],
                'DRI In': [dri_in_eaf],

                'H2 Charge': [H2_charge],
                'H2 Discharge': [H2_discharge],
                'H2 SOC': [H2_SOC],
                'H2 In': [H2_in_dri],
                # 'Shutdown': [shutdown_electrolyser],
                # 'On': [in_operation_electrolyser],
                # 'Start Up': [start_up_electrolyser]
            })

            # Append the DataFrame to the list
            results_dfs.append(df)

        # Concatenate all DataFrames in the list into a single DataFrame
        results_df = pd.concat(results_dfs, ignore_index=True)

        # Display the entire table
        print(results_df)

        self.total_cost = total_cost

    def calculate_marginal_cost(self, start: pd.Timestamp, power: float) -> float:
        """
        Calculate the marginal cost of the unit returns the marginal cost of the unit based on the provided time and power.

        Args:
            start (pandas.Timestamp): The start time of the dispatch.
            power (float): The power output of the unit.

        Returns:
            float: the marginal cost of the unit for the given power.
        """
        return self.electricity_price.at[start]

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
