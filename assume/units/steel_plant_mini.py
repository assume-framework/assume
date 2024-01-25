import logging
from typing import Dict, List

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

from assume.common.base import SupportsMinMax
from assume.units.dst_components import Electrolyser, ShaftFurnace, ElectricArcFurnace

logger = logging.getLogger(__name__)

# Mapping of component type identifiers to their respective classes
dst_components = {
    "electrolyser": Electrolyser,
    "shaft_furnace": ShaftFurnace,
}

# Define possible technology configurations
technology_configurations = {
    'electrolyser_shaftFurnace': [
        ('electrolyser', 'hydrogen_out', 'shaft_furnace', 'hydrogen_in'),
    ],
    # Add other configurations as needed
}

class SteelPlant(SupportsMinMax):
    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        technology: str = "steel_plant",
        plant_type: str = "electrolyser_shaftFurnace",
        node: str = "bus0",
        index: pd.DatetimeIndex = None,
        location: tuple[float, float] = (0.0, 0.0),
        components: Dict[str, Dict] = None,
        objective: str = None,

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

        
        self.hydrogen_price = self.forecaster["hydrogen_price"]
        self.electricity_price = self.forecaster["price_EOM"]
        self.iron_ore_price = self.forecaster["iron_ore_price"]
        self.dri_demand = self.forecaster["dri_demand"]
        self.dri_price = self.forecaster["dri_price"]

        self.location = location

        self.objective = objective

        self.components = {}

        # Validate and set the plant_type attribute
        if plant_type not in technology_configurations:
            raise ValueError(f"Plant type '{plant_type}' is not recognized.")
        self.plant_type = plant_type

        self.create_model()
        
        # Initialize components based on the selected technology configuration
        self.initialize_components(components)
        self.initialize_process_sequence(plant_type)
        
        self.define_constraints()

        print(self.model.display())

    # def calculate_min_max_power(self, start: pd.Timestamp, end: pd.Timestamp):
    #     # Calculate aggregated power_in from all components with power_in variable
    #     aggregated_power_in = sum(
    #         component.component_block.power_in[t] 
    #         for component in self.components.values() 
    #         if hasattr(component.component_block, 'power_in') 
    #         for t in self.model.time_steps
    #     )

    #     return aggregated_power_in


    def create_model(self):
        print("Creating Master Model for SteelPlant")
        self.model = pyo.ConcreteModel()
        self.define_sets()
        self.define_parameters()
        self.define_variables()
        self.define_objective()

    def initialize_components(self, components):
        for component_id, component_data in components.items():
            component_technology = component_data["technology"]  # Use "technology"
            if component_technology in dst_components:
                component_class = dst_components[component_technology]
                component_instance = component_class(model=self.model, id=component_id, **component_data)

                 # Call the add_to_model method for each component
                component_instance.add_to_model(self.model, self.model.time_steps)
                self.components[component_id] = component_instance


    def initialize_process_sequence(self, plant_type):
        # Use plant_type to determine the process sequence
        if plant_type not in technology_configurations:
            raise ValueError(f"Plant type '{plant_type}' is not recognized.")

        if plant_type in technology_configurations:
            process_sequence = technology_configurations[plant_type]

            for connection in process_sequence:
                source_component_id, source_output, target_component_id, target_input = connection

                if source_component_id in self.components and target_component_id in self.components:
                    source_component = self.components[source_component_id]
                    target_component = self.components[target_component_id]

                    # Define the process connection based on the updated configuration
                    if source_output == 'hydrogen_out' and target_input == 'hydrogen_in':
                        # Add a constraint to enforce the connection
                        @self.model.Constraint(self.model.time_steps)
                        def hydrogen_transfer_constraint(m, t):
                            return source_component.component_block.hydrogen_out[t] == target_component.component_block.hydrogen_in[t]

    def define_sets(self) -> None:
        # self.model.time_steps = pyo.Set(initialize=range(len(self.index)))
        self.model.time_steps = pyo.Set(
            initialize=[idx for idx, _ in enumerate(self.index)]
        )

    def define_parameters(self):
        self.model.electricity_price = pyo.Param(self.model.time_steps, initialize={t: value for t, value in enumerate(self.electricity_price)})
        self.model.hydrogen_price = pyo.Param(self.model.time_steps, initialize={t: value for t, value in enumerate(self.hydrogen_price)})
        self.model.iron_ore_price = pyo.Param(self.model.time_steps, initialize={t: value for t, value in enumerate(self.iron_ore_price)})
        self.model.dri_price = pyo.Param(self.model.time_steps, initialize={t: value for t, value in enumerate(self.dri_price)})

        self.model.dri_demand = pyo.Param(self.model.time_steps, initialize={t: value for t, value in enumerate(self.dri_demand)})
        
    def define_variables(self):
        # Define variables specific to the SteelPlant
        self.model.aggregated_power_in = pyo.Var(self.model.time_steps, within=pyo.NonNegativeReals)
        self.model.aggregated_hydrogen_in = pyo.Var(self.model.time_steps, within=pyo.NonNegativeReals)
        self.model.aggregated_iron_ore_in = pyo.Var(self.model.time_steps, within=pyo.NonNegativeReals)
        self.model.aggregated_dri_output = pyo.Var(self.model.time_steps, within=pyo.NonNegativeReals)
        # Define other variables relevant to your steel plant here
        
    def define_constraints(self):

        #  Add the constraint to aggregate power_in from all components
        @self.model.Constraint(self.model.time_steps)
        def aggregate_power_constraint(m, t):
            return m.aggregated_power_in[t] == sum(
                component.component_block.power_in[t]
                for component in self.components.values()
                if hasattr(component.component_block, 'power_in')
            )
        
        @self.model.Constraint(self.model.time_steps)
        def aggregate_iron_ore_constraint(m, t):
            return m.aggregated_iron_ore_in[t] == sum(
                component.component_block.iron_ore_in[t]
                for component in self.components.values()
                if hasattr(component.component_block, 'iron_ore_in')
            )
        
        @self.model.Constraint(self.model.time_steps)
        def aggregate_hydrogen_constraint(m, t):
            return m.aggregated_hydrogen_in[t] == sum(
                component.component_block.power_in[t]
                for component in self.components.values()
                if hasattr(component.component_block, 'hydrogen_in')
            )
        
        @self.model.Constraint(self.model.time_steps)
        def aggregate_dri_out_constraint(m, t):
            return m.aggregated_dri_output[t] == sum(
                component.component_block.dri_output[t]
                for component in self.components.values()
                if hasattr(component.component_block, 'dri_output')
            )
        
        @self.model.Constraint(self.model.time_steps)
        def dri_output_association_constraint(m, t):
                    return m.aggregated_dri_output[t] == self.dri_demand[t]
        
        # for component_instance in self.components.values():
        #     if isinstance(component_instance, ShaftFurnace):
        #         # Add constraints specific to the ShaftFurnace component

        #         @self.model.Constraint(self.model.time_steps)
        #         def dri_output_association_constraint(m, t):
        #             return m.dri_output[t] == m.dri_demand[t]
                
         

    def define_objective(self):
        if self.objective == "maximize_marginal_profit":
            @self.model.Objective(sense=pyo.maximize)
            def obj_rule(m):
                total_revenue = sum(
                    self.dri_price[t] * m.aggregated_dri_output[t]
                    for t in m.time_steps
                )
                
                total_costs = sum(
                    self.electricity_price[t] * m.aggregated_power_in[t] +
                    self.hydrogen_price[t] * m.aggregated_hydrogen_in[t] +
                    self.iron_ore_price[t] * m.aggregated_iron_ore_in[t]
                    for t in m.time_steps
                )
                
                return total_revenue - total_costs
        elif self.objective == "minimize_marginal_cost":
            @self.model.Objective(sense=pyo.minimize)
            def obj_rule(m):
                total_costs = sum(
                    self.electricity_price[t] * m.aggregated_power_in[t] +
                    self.hydrogen_price[t] * m.aggregated_hydrogen_in[t] +
                    self.iron_ore_price[t] * m.aggregated_iron_ore_in[t]
                    for t in m.time_steps
                )
                return total_costs
        else:
            raise ValueError(f"Unknown objective: {self.objective}")



    def run_optimization(self):
        # Create a solver
        solver = SolverFactory("gurobi")
        self.model.pprint()
        # Solve the model
        # solver.solve(self.model, tee=True)
        results = solver.solve(self.model, tee=True)  # , tee=True
        # print(results)
        # self.model.display()

        # Check solver status and termination condition
        if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal
        ):
            print("The model was solved optimally.")

            # Display the Objective Function Value
            objective_value = pyo.value(self.model.objective)
            print(f"The value of the objective function is {objective_value}.")

        elif results.solver.termination_condition == TerminationCondition.infeasible:
            print("The model is infeasible.")

        else:
            print("Solver Status: ", results.solver.status)
            print("Termination Condition: ", results.solver.termination_condition)

