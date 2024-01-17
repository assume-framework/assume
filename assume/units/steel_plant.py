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
    "electric_arc_furnace": ElectricArcFurnace
}

# Define possible technology configurations
technology_configurations = {
    'electrolyser_shaftFurnace_EAF': [
        ('electrolyser', 'hydrogen_out', 'shaft_furnace', 'hydrogen_in'),
        ('shaft_furnace', 'dri_output', 'electric_arc_furnace', 'dri_input')
    ],
    'blastFurnace_basicOxygenFurnace': [
        ('blast_furnace', 'iron_output', 'basic_oxygen_furnace', 'iron_input')
    ],
    'electrolyser_storage_shaftFurnace_EAF': [
        ('electrolyser', 'hydrogen_out', 'hydrogen_storage', 'hydrogen_in'),
        ('electrolyser', 'hydrogen_out', 'shaft_furnace', 'direct_hydrogen_input'),
        ('hydrogen_storage', 'hydrogen_out', 'shaft_furnace', 'stored_hydrogen_in'),
        ('shaft_furnace', 'dri_output', 'electric_arc_furnace', 'dri_input')
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
        plant_type: str = 'electrolyser_shaftFurnace_EAF',
        node: str = "bus0",
        index: pd.DatetimeIndex = None,
        location: tuple[float, float] = (0.0, 0.0),
        components: Dict[str, Dict] = None,
        objective: str = None,

        # hydrogen_price: float = None,
        # electricity_price: float = None,
        # natural_gas_price: float = None,
        # iron_ore_price: float = None,
        # steel_price: float = None,
        # steel_demand: float = None,

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

        self.hydrogen_demand = self.forecaster[f"{self.id}_hydrogen_demand"]
        self.hydrogen_price = self.forecaster["hydrogen_price"]
        self.electricity_price = self.forecaster["price_EOM"]
        self.natural_gas_price = self.forecaster["fuel_price_naturalgas"]
        self.iron_ore_price = self.forecaster["iron_ore_price"]
        self.steel_price = self.forecaster["steel_price"]
        self.steel_demand = self.forecaster["steel_demand"]

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

    def create_model(self):
        print("Creating Master Model for SteelPlant")
        self.model = pyo.ConcreteModel()
        self.define_sets()
        self.define_parameters()
        self.define_variables()
        self.define_objective()

    def initialize_components(self, components):
        for component_id, component_info in components.items():
            component_type = component_info["technology"]
            if component_type not in dst_components:
                raise ValueError(f"Component type '{component_type}' not recognized.")

            component_class = dst_components[component_type]
            
            self.components[component_id] = component_class(self.model, component_id, **component_info)
            self.components[component_id].add_to_model(self.model, self.model.time_steps)

    def initialize_process_sequence(self, plant_type):
        # Use plant_type to determine the process sequence
        if plant_type not in technology_configurations:
            raise ValueError(f"Plant type '{plant_type}' is not recognized.")

        sequence = technology_configurations[plant_type]
        for unit_type in sequence:
            if unit_type in self.components:
                self.initialize_unit_sequence(self.components[unit_type])

    def initialize_unit_sequence(self, technology_choice):
        process_sequence = technology_configurations[technology_choice]
        for process_link in process_sequence:
            source_unit, source_output, target_unit, target_input = process_link

            @self.model.Constraint(self.model.time_steps)
            def process_flow_constraint(m, t):
                if target_unit == 'shaft_furnace' and source_unit == 'electrolyser':
                    # Constraint for hydrogen flow from electrolyser to shaft furnace
                    return m.shaft_furnace.hydrogen_input_from_electrolyser[t] == getattr(m.components[source_unit], source_output)[t] * m.use_hydrogen_from_electrolyser[t]
                elif target_unit == 'shaft_furnace' and source_unit == 'hydrogen_storage':
                    # Constraint for hydrogen flow from storage to shaft furnace
                    return m.shaft_furnace.hydrogen_input_from_storage[t] == getattr(m.components[source_unit], source_output)[t] * m.use_hydrogen_from_storage[t]
                else:
                    # Standard process flow constraint
                    return getattr(m.components[source_unit], source_output)[t] == getattr(m.components[target_unit], target_input)[t]

            constraint_name = f"flow_from_{source_unit}_to_{target_unit}"
            self.model.add_component(constraint_name, process_flow_constraint)



    def define_sets(self) -> None:
        # self.model.time_steps = pyo.Set(initialize=range(len(self.index)))
        self.model.time_steps = pyo.Set(
            initialize=[idx for idx, _ in enumerate(self.index)]
        )

    def define_parameters(self):
        self.model.electricity_price = pyo.Param(self.model.time_steps, 
                                                 initialize={t: self.electricity_price if isinstance(self.electricity_price, (float, int)) 
                                                             else self.electricity_price[t] for t in self.model.time_steps})
        
        self.model.hydrogen_price = pyo.Param(self.model.time_steps, 
                                                 initialize={t: self.hydrogen_price if isinstance(self.hydrogen_price, (float, int)) 
                                                             else self.hydrogen_price[t] for t in self.model.time_steps})
        
        self.model.natural_gas_price = pyo.Param(self.model.time_steps, 
                                                 initialize={t: self.natural_gas_price if isinstance(self.natural_gas_price, (float, int)) 
                                                             else self.natural_gas_price[t] for t in self.model.time_steps})
        
        self.model.iron_ore_price = pyo.Param(self.model.time_steps, 
                                                 initialize={t: self.iron_ore_price if isinstance(self.iron_ore_price, (float, int)) 
                                                             else self.iron_ore_price[t] for t in self.model.time_steps})
        
        self.model.steel_price = pyo.Param(self.model.time_steps, 
                                                 initialize={t: self.steel_price if isinstance(self.steel_price, (float, int)) 
                                                             else self.steel_price[t] for t in self.model.time_steps})
        
        self.model.steel_demand = pyo.Param(self.model.time_steps, 
                                                 initialize={t: self.steel_demand if isinstance(self.steel_demand, (float, int)) 
                                                             else self.steel_demand[t] for t in self.model.time_steps})
        
    def define_variables(self):
        
        # Binary decision variables for hydrogen source selection
        self.model.use_hydrogen_from_electrolyser = pyo.Var(self.model.time_steps, within=pyo.Binary)
        # self.model.hydrogen_input_from_electrolyser = pyo.Var(self.model.time_steps, within=pyo.NonNegativeReals)
        self.model.use_hydrogen_from_storage = pyo.Var(self.model.time_steps, within=pyo.Binary)
        # self.model.hydrogen_input_from_storage = pyo.Var(self.model.time_steps, within=pyo.NonNegativeReals)

        self.model.hydrogen_input_from_electrolyser = pyo.Var(self.model.time_steps, within=pyo.NonNegativeReals)


        self.model.power_in = pyo.Var(self.model.time_steps, within=pyo.NonNegativeReals)
        self.model.hydrogen_in = pyo.Var(self.model.time_steps, within=pyo.NonNegativeReals)
        self.model.natural_gas_in = pyo.Var(self.model.time_steps, within=pyo.NonNegativeReals)
        self.model.iron_ore_in = pyo.Var(self.model.time_steps, within=pyo.NonNegativeReals)

        # Function to initialize the steel_output variable based on the outputs of different components
        def initialize_steel_output(m, t):
            steel_output = 0
            if 'electric_arc_furnace' in self.components:
                steel_output += self.components['electric_arc_furnace'].steel_output[t]
            if 'basic_oxygen_furnace' in self.components:  # Placeholder for future implementation
                steel_output += self.components['basic_oxygen_furnace'].steel_output[t]
            # Add other component outputs as needed
            return steel_output

        # Define the steel_output variable with the initialization rule
        self.model.steel_output = pyo.Var(self.model.time_steps, within=pyo.NonNegativeReals, 
                                          initialize=initialize_steel_output)


    def define_constraints(self):

        # Check if steel demand is a scalar (float or int)
        if isinstance(self.steel_demand, (float, int)):
            # Constraint for scalar steel demand - total steel output should match total demand over all time steps
            @self.model.Constraint()
            def total_steel_demand_constraint(m):
                return sum(m.steel_output[t] for t in m.time_steps) == m.steel_demand

        else:  # Assuming steel demand is a time series
            # Constraint for time series steel demand - steel output should match demand at each time step
            @self.model.Constraint(self.model.time_steps)
            def steel_demand_constraint(m, t):
                return m.steel_output[t] == m.steel_demand[t]

    def calculate_min_max_power(self, start: pd.Timestamp, end: pd.Timestamp):
        aggregated_power = 0
        total_cost = 0

        for t in pd.date_range(start=start, end=end, freq='H'):
            hour_index = t.hour  # Convert timestamp to hour index

            # print(f"Hour Index: {hour_index}")

            # Aggregating power input and calculating cost for Electrolyser
            if 'electrolyser' in self.components:
                electrolyser_power_in = self.components['electrolyser'].component_block.power_in[hour_index].value
                if electrolyser_power_in is not None:
                    aggregated_power += electrolyser_power_in
                    electrolyser_cost = (electrolyser_power_in * self.model.electricity_price[hour_index].value +
                                        self.components['electrolyser'].component_block.hydrogen_in[hour_index].value * self.model.hydrogen_price[hour_index].value)
                    total_cost += electrolyser_cost
                # else:
                #     print(f"Warning: electrolyser Power Input is None for hour {hour_index}")

            # Calculating cost for Shaft Furnace (assuming it doesn't use power_in)
            if 'shaft_furnace' in self.components:
                natural_gas_input = self.components['shaft_furnace'].component_block.natural_gas_in[hour_index]
                if natural_gas_input.value is not None:
                    shaft_furnace_cost = (natural_gas_input.value * self.model.natural_gas_price[hour_index].value +
                                        self.components['shaft_furnace'].component_block.hydrogen_in[hour_index].value * self.model.hydrogen_price[hour_index].value +
                                        self.components['shaft_furnace'].component_block.iron_ore_in[hour_index].value * self.model.iron_ore_price[hour_index].value)
                    total_cost += shaft_furnace_cost

            # Aggregating power input and calculating cost for Electric Arc Furnace
            if 'electric_arc_furnace' in self.components:
                eaf_power_input = self.components['electric_arc_furnace'].component_block.power_in[hour_index]
                if eaf_power_input.value is not None:
                    eaf_power_in = eaf_power_input.value
                    aggregated_power += eaf_power_in
                    electric_arc_furnace_cost = eaf_power_in * self.model.electricity_price[hour_index].value
                    total_cost += electric_arc_furnace_cost

            # Add other component power and cost calculations as needed

        # Calculate marginal cost (total cost divided by total power, if power is not zero)
        if aggregated_power > 0:
            marginal_cost = total_cost / aggregated_power
        else:
            marginal_cost = 0

        return aggregated_power, marginal_cost



    def define_objective(self):
    # Helper function to calculate total costs
        def total_costs(m):
            return sum(m.electricity_price[t] * m.power_in[t] +
                    m.hydrogen_price[t] * m.hydrogen_in[t] +
                    m.natural_gas_price[t] * m.natural_gas_in[t] +
                    m.iron_ore_price[t] * m.iron_ore_in[t]
                    for t in m.time_steps)

        # Helper function to calculate total revenue
        def total_revenue(m):
            return sum(m.steel_price[t] * m.steel_output[t] for t in m.time_steps)

        # Define the objective function based on the specified objective
        if self.objective == "maximize_marginal_profit":
            self.model.objective = pyo.Objective(expr=total_revenue(self.model) - total_costs(self.model), sense=pyo.maximize)
        elif self.objective == "minimize_marginal_cost":
            self.model.objective = pyo.Objective(expr=total_costs(self.model), sense=pyo.minimize)

    def run_optimization(self):
        # Create a solver
        solver = SolverFactory("gurobi")

        print("Model Components Before Optimization:")
        self.model.pprint()

        for t in self.model.time_steps:
            self.model.power_in[t] = 0  

        for t in self.model.time_steps:
            variable_name = f"power_in[{t}]"
            variable_value = self.model.power_in[t].value
            print(f"{variable_name} = {variable_value}")

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
            objective_value = self.model.obj_rule()
            print(f"The value of the objective function is {objective_value}.")

        elif results.solver.termination_condition == TerminationCondition.infeasible:
            print("The model is infeasible.")

        else:
            print("Solver Status: ", results.solver.status)
            print("Termination Condition: ", results.solver.termination_condition)

