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
        # time_steps=None,
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
        self.plant_type =plant_type

        self.location = location

        self.objective = objective
        self.components = {}  # Store created components
        # self.index = time_steps if time_steps is not None else []  # Use the provided time_steps argument or initialize as an empty list


        # Combine variables, parameters, constraints, and objectives
        self.model_steel_plant = pyo.ConcreteModel()
        # Initialize components based on the list passed
        self.define_sets()
        self.initialize_components(components=components)
        # Define parameters
        
        self.define_parameters()
        # Define components
        self.define_variables()
        self.define_constraints()
        self.connect_components()
        self.define_objective()

        # Combine variables, parameters, constraints, and objectives
        # self.model_steel_plant = pyo.ConcreteModel()

    def initialize_components(self, components: Dict[str, Dict]):
        if self.components is None:
            logger.warning("No components specified for building")
            return

        for i, (unit_type, tech_params) in enumerate(components.items()):
            unit_name = f"{unit_type.lower()}"
            component_class = dst_components.get(unit_type)

            if component_class is None:
                raise ValueError(f"Unknown unit type: {unit_type}")

            unit_block = pyo.Block()
            self.model_steel_plant.add_component(unit_name, unit_block)
            new_component = component_class(
                id=self.id,
                time_steps=self.model_steel_plant.time_steps,
                model=unit_block,
                **tech_params,  # Pass all parameters from the unit_params dictionary
            )

            # Store the component in the dictionary with its name as the key
            self.components[unit_name] = new_component

    def define_sets(self) -> None:
        self.model_steel_plant.time_steps = pyo.Set(
            initialize=[idx for idx, _ in enumerate(self.index)]
        )

    def define_parameters(self):
        self.model_steel_plant.electricity_price = pyo.Param(
            self.model_steel_plant.time_steps, initialize={t: value for t, value in enumerate(self.electricity_price)})
        self.model_steel_plant.hydrogen_price = pyo.Param(
            self.model_steel_plant.time_steps, initialize={t: value for t, value in enumerate(self.hydrogen_price)})
        self.model_steel_plant.iron_ore_price = pyo.Param(
            self.model_steel_plant.time_steps, initialize={t: value for t, value in enumerate(self.iron_ore_price)})
        self.model_steel_plant.dri_price = pyo.Param(
            self.model_steel_plant.time_steps, initialize={t: value for t, value in enumerate(self.dri_price)})
        self.model_steel_plant.dri_demand = pyo.Param(
            self.model_steel_plant.time_steps, initialize={t: value for t, value in enumerate(self.dri_demand)})

    def define_variables(self):
        self.model_steel_plant.some_variable = pyo.Var(within=pyo.NonNegativeReals)

    def define_constraints(self):
        # @self.model_steel_plant.Constraint(self.model_steel_plant.time_steps)
        # def main_hydrogen_association(m, t):
        #     return self.components["electrolyser"].model.hydrogen_out[t] == self.components["shaft_furnace"].model.hydrogen_in[t]
        
        @self.model_steel_plant.Constraint(self.model_steel_plant.time_steps)
        def main_constraint_rule(model, t):
            return self.components["shaft_furnace"].model.dri_output[t] == model.dri_demand[t]
        
    def connect_components(self):
        # Connect components if plant_type is "electrolyser_shaftFurnace"
        if self.plant_type == "electrolyser_shaftFurnace":
            @self.model_steel_plant.Constraint(self.model_steel_plant.time_steps)
            def hydrogen_connection_rule(m, t):
                return self.components["electrolyser"].m.hydrogen_out[t] == self.components["shaft_furnace"].m.hydrogen_in[t]

        
    def define_objective(self):
        def total_cost_rule(model):
            return sum(
                self.model_steel_plant.electricity_price[t] * self.components["electrolyser"].model.power_in[t] +
                self.model_steel_plant.hydrogen_price[t] * self.components["electrolyser"].model.hydrogen_out[t] +
                self.model_steel_plant.iron_ore_price[t] * self.components["shaft_furnace"].model.iron_ore_in[t]
                for t in self.model_steel_plant.time_steps
            )

        self.model_steel_plant.objective = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)


    def run_optimization(self):
        solver = SolverFactory('gurobi')
        self.model_steel_plant.pprint()
        # Solve the model
        self.results = solver.solve(self.model_steel_plant, tee=True)

        # Check solver status and termination condition
        if (self.results.solver.status == SolverStatus.ok) and (
            self.results.solver.termination_condition == TerminationCondition.optimal
        ):
            logger.debug("The model was solved optimally.")

        elif (
            self.results.solver.termination_condition == TerminationCondition.infeasible
        ):
            logger.debug("The model is infeasible.")

        else:
            logger.debug("Solver Status: ", self.results.solver.status)
            logger.debug(
                "Termination Condition: ", self.results.solver.termination_condition
            )


