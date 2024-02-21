import logging
from typing import Dict, List
import assume.common.flexibility as flex

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

from assume.common.base import SupportsMinMax
from assume.units.dst_components import Electrolyser, DriPlant, ElectricArcFurnace, GenericStorage, DRIStorage

logger = logging.getLogger(__name__)

# Mapping of component type identifiers to their respective classes
dst_components = {
    "electrolyser": Electrolyser,
    "h2storage": GenericStorage,
    "dri_plant": DriPlant,
    "dri_storage": DRIStorage,
    "eaf": ElectricArcFurnace
}

class SteelPlant(SupportsMinMax):
    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        technology: str = "steel_plant",
        plant_type: str = "electrolyser_h2storage_dri_eaf",
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
        self.natural_gas_price = self.forecaster["fuel_price_natural_gas"]
        self.electricity_price = self.forecaster["price_EOM"]
        self.iron_ore_price = self.forecaster["iron_ore_price"]
        self.steel_demand = self.forecaster["steel_demand"]
        self.dri_price = self.forecaster["dri_price"]

        self.location = location
        self.objective = objective

        self.components = {}
        self.plant_type = plant_type

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

    def initialize_components(self, components):
        for component_id, component_data in components.items():
            component_technology = component_data["technology"]
            if component_technology in dst_components:
                component_class = dst_components[component_technology]
                component_instance = component_class(model=self.model, id=component_id, **component_data)

                 # Call the add_to_model method for each component
                component_instance.add_to_model(self.model, self.model.time_steps)
                self.components[component_id] = component_instance


    def initialize_process_sequence(self):
        # Assuming the presence of 'h2storage' indicates the desire for dynamic flow management
        has_h2storage = 'h2storage' in self.components

        # Constraint for direct hydrogen flow from Electrolyser to dri plant
        @self.model.Constraint(self.model.time_steps)
        def direct_hydrogen_flow_constraint(m, t):
            # This constraint allows part of the hydrogen produced by the dri plant to go directly to the EAF
            # The actual amount should ensure that it does not exceed the capacity or demand of the EAF
            if has_h2storage:
                return self.components['electrolyser'].b.hydrogen_out[t] + self.components['h2storage'].b.discharge[t] >= self.components['dri_plant'].b.hydrogen_in[t] + self.components['h2storage'].b.charge[t]
            else: 
                return self.components['electrolyser'].b.hydrogen_out[t] >= self.components['dri_plant'].b.hydrogen_in[t]
            
        # Assuming the presence of dristorage' indicates the desire for dynamic flow management
        has_dristorage = 'dri_storage' in self.components

        # Constraint for direct hydrogen flow from Electrolyser to dri plant
        @self.model.Constraint(self.model.time_steps)
        def direct_dri_flow_constraint(m, t):
            # This constraint allows part of the dri produced by the dri plant to go directly to the dri storage
            # The actual amount should ensure that it does not exceed the capacity or demand of the EAF
            if has_dristorage:
                return self.components['dri_plant'].b.dri_output[t] + self.components['dri_storage'].b.discharge_dri[t] >= self.components['eaf'].b.dri_input[t] + self.components['dri_storage'].b.charge_dri[t]
            else: 
                return self.components['dri_plant'].b.dri_output[t] == self.components['eaf'].b.dri_input[t]
            
        # Constraint for material flow from dri plant to Electric Arc Furnace
        @self.model.Constraint(self.model.time_steps)
        def shaft_to_arc_furnace_material_flow_constraint(m, t):
            return self.components['dri_plant'].b.dri_output[t] == self.components['eaf'].b.dri_input[t]

    def define_sets(self) -> None:
        self.model.time_steps = pyo.Set(
            initialize=[idx for idx, _ in enumerate(self.index)]
        )

    def define_parameters(self):
        self.model.electricity_price = pyo.Param(self.model.time_steps, initialize={t: value for t, value in enumerate(self.electricity_price)})
        self.model.natural_gas_price = pyo.Param(self.model.time_steps, initialize={t: value for t, value in enumerate(self.natural_gas_price)})
        self.model.iron_ore_price = pyo.Param(self.model.time_steps, initialize={t: value for t, value in enumerate(self.iron_ore_price)})
        self.model.dri_price = pyo.Param(self.model.time_steps, initialize={t: value for t, value in enumerate(self.dri_price)})

        self.model.steel_demand = pyo.Param(self.model.time_steps, initialize={t: value for t, value in enumerate(self.steel_demand)})
        
    def define_variables(self):
        self.model.total_power_input = pyo.Var(self.model.time_steps, within=pyo.NonNegativeReals)
        
    def define_constraints(self):
        @self.model.Constraint(self.model.time_steps)
        def dri_output_association_constraint(m, t):
            return self.components['eaf'].b.steel_output[t] >= self.steel_demand[t]
        
        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
            return m.total_power_input[t] == self.components['electrolyser'].b.power_in[t] + self.components['eaf'].b.power_eaf[t]

    def define_objective(self):
        if self.objective == "maximize_marginal_profit":
            @self.model.Objective(sense=pyo.maximize)
            def obj_rule(m):
                total_revenue = sum(
                    self.dri_price[t] * m.aggregated_dri_output[t]
                    for t in m.time_steps
                )
                
                total_costs = sum(
                    self.electricity_price[t] * self.components['electrolyser'].b.power_in[t] + 
                    # self.hydrogen_price[t] * self.components['electrolyser'].b.hydrogen_out[t] +
                    self.iron_ore_price[t] * self.components['dri_plant'].b.iron_ore_in[t]
                    for t in m.time_steps
                )
                
                return total_revenue - total_costs
        elif self.objective == "minimize_marginal_cost":
            @self.model.Objective(sense=pyo.minimize)
            def obj_rule(m):
                total_costs = sum(
                    self.components['electrolyser'].b.start_cost[t] + 
                    self.components['electrolyser'].b.electricity_cost[t] +
                    self.components['dri_plant'].b.dri_operating_cost[t] +
                    self.components['eaf'].b.eaf_operating_cost[t] +
                    self.iron_ore_price[t] * self.components['dri_plant'].b.iron_ore_in[t]
                    for t in m.time_steps
                )
                return total_costs
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    def run_optimization(self):
        # Create a solver
        solver = SolverFactory("gurobi")

        print("Model Components Before Optimization:")
        # self.model.pprint()
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

        temp = self.model.total_power_input.get_values()
        self.power_requirement = pd.Series(index=self.index, data=0.0)
        for i, date in enumerate(self.index):
            self.power_requirement.loc[date] = temp[i]

def determine_optimal_operation(self):
        """
        Determines the optimal operation of the steel plant without considering flexibility.
        """
        optimal_operation = self.run_optimization()
        return optimal_operation

