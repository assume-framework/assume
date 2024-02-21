import logging
from typing import Dict, List

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

from assume.common.base import BaseUnit
from assume.units.dst_components import Electrolyser, DriPlant, ElectricArcFurnace

logger = logging.getLogger(__name__)

# SOLVERS = ["gurobi", "glpk"]

dst_components = {
    "electrolyser": Electrolyser,
    "dri_plant": DriPlant,
    "electric_arc_furnace": ElectricArcFurnace
}


class Plant(BaseUnit):
    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        technology: str = "electrolyser_plant",
        node: str = "bus0",
        index: pd.DatetimeIndex = None,
        location: tuple[float, float] = (0.0, 0.0),
        # storage_list: List = None,  # List of storage units
        components: Dict[str, Dict] = None,
        objective: str = "minimize_cost",
        hydrogen_price: float = None,
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

        # self.storage_list = storage_list
        self.components = {}
        # self.storage_units = {}

        self.hydrogen_demand = self.forecaster[f"{self.id}_hydrogen_demand"]
        self.electricity_price = self.forecaster["price_EOM"]
        self.hydrogen_price = self.forecaster["hydrogen_price"]
        self.objective = objective

        self.location = location

        self.create_model()
        self.initialize_components(components=components)
        self.define_constraints()

        print(self.model.display())

    def create_model(self):
        print("Creating Master Model for Plant")
        self.model = pyo.ConcreteModel()
        self.define_sets()
        self.define_parameters()
        self.define_variables()
        self.define_objective()

    # Initialize units based on the list passed
    def initialize_components(self, components: Dict[str, Dict] = None):
        if self.components is None:
            logger.warning("No components specified for plant")
            return

        for i, (unit_type, tech_params) in enumerate(components.items()):
            unit_name = f"{unit_type.lower()}_{i+1}"
            component_class = dst_components.get(unit_type)
            if component_class is None:
                raise ValueError(f"Unknown unit type: {unit_type}")

            unit_block = pyo.Block()
            self.model.add_component(unit_name, unit_block)

            new_component = component_class(
                id=self.id,
                model=self.model,
                **tech_params,
            )

            new_component.add_to_model(
                unit_block=unit_block,
                time_steps=self.model.time_steps,
            )
            self.components[unit_name] = new_component

    def define_sets(self) -> None:
        self.model.time_steps = pyo.Set(
            initialize=[idx for idx, _ in enumerate(self.index)]
        )

    def define_parameters(self) -> None:
        self.model.hydrogen_demand = pyo.Param(
            self.model.time_steps, initialize=dict(enumerate(self.hydrogen_demand))
        )
        
        self.model.electricity_price = pyo.Param(
            self.model.time_steps,
            initialize=dict(enumerate(self.electricity_price)),
        )
        self.model.hydrogen_price = pyo.Param(
            self.model.time_steps,
             initialize=dict(enumerate(self.hydrogen_price))
        )

    def define_variables(self):
        self.model.aggregated_power_in = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.aggregated_hydrogen_production = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.aggregated_startup_cost = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.revenue = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )

    def define_constraints(self):

        @self.model.Constraint(self.model.time_steps)
        def aggregate_power_in_constraint(m, t):
            return m.aggregated_power_in[t] == sum(
                getattr(m, unit_name).power_in[t]
                for unit_name in self.components.keys()
            )

        @self.model.Constraint(self.model.time_steps)
        def aggregate_hydrogen_production_constraint(m, t):
            return m.aggregated_hydrogen_production[t] == sum(
                getattr(m, unit_name).hydrogen_output[t]
                for unit_name in self.components.keys()
            )

        @self.model.Constraint(self.model.time_steps)
        def revenue_calculation(m, t):
            return m.revenue[t] == m.aggregated_hydrogen_production[t] * m.hydrogen_price[t]

        # @self.model.Constraint(self.model.time_steps)
        # def aggregate_startup_cost_constraint(m, t):
        #     return m.aggregated_startup_cost[t] == sum(
        #         getattr(m, unit_name).startup_cost_incurred[t]
        #         for unit_name in self.components.keys()
        #     )

        @self.model.Constraint(self.model.time_steps)
        def demand_integration(m, t):
            return m.aggregated_hydrogen_production[t] >= m.hydrogen_demand[t]

    def define_objective(self):
        if self.objective == "minimize_cost":

            @self.model.Objective(sense=pyo.minimize)
            def obj_rule(m):
                return sum(
                    m.aggregated_power_in[t] * self.electricity_price[t]
                    for t in self.model.time_steps
                ) + sum(
                m.aggregated_startup_cost[t]
                for t in m.time_steps
            )

        elif self.objective == "maximize_comfort":
            # Define your comfort metric here
            pass
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    def run_optimization(self):
        # Create a solver
        solver = SolverFactory("gurobi")

        # Solve the model
        solver.solve(self.model, tee=True)
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