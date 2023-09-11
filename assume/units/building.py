import logging
from typing import Dict, List

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import (
    SolverFactory,
    SolverStatus,
    TerminationCondition,
    check_available_solvers,
)

from assume.common.base import BaseUnit
from assume.units.dst_components import AirConditioner, HeatPump, Storage

logger = logging.getLogger(__name__)

SOLVERS = ["gurobi", "glpk"]

dst_components = {
    "heat_pump": HeatPump,
    "air_conditioner": AirConditioner,
    "thermal_storage": Storage,
    "electrical_storage": Storage,
}


class Building(BaseUnit):
    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        technology: str = "building",
        node: str = "bus0",
        index: pd.DatetimeIndex = None,
        location: tuple[float, float] = (0.0, 0.0),
        storage_list: List = None,  # List of storage units
        components: Dict[str, Dict] = None,
        objective: str = "minimize_cost",
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

        self.storage_list = storage_list
        self.components = {}
        self.storage_units = {}

        self.heating_demand = self.forecaster[f"{self.id}_heating"]
        self.cooling_demand = self.forecaster[f"{self.id}_cooling"]
        self.electricity_price = self.forecaster.get_electricity_price(EOM="EOM")
        self.objective = objective

        self.location = location

        self.create_model()
        self.initialize_components(components=components)
        self.define_constraints()

        self.results = None

    def create_model(self):
        self.model = pyo.ConcreteModel()
        self.define_sets()
        self.define_parameters()
        self.define_variables()
        self.define_objective()

    def define_sets(self) -> None:
        self.model.time_steps = pyo.Set(
            initialize=[idx for idx, _ in enumerate(self.index)]
        )

    def define_parameters(self) -> None:
        self.model.heating_demand = pyo.Param(
            self.model.time_steps, initialize=dict(enumerate(self.heating_demand))
        )
        self.model.cooling_demand = pyo.Param(
            self.model.time_steps, initialize=dict(enumerate(self.cooling_demand))
        )
        self.model.electricity_price = pyo.Param(
            self.model.time_steps,
            initialize=dict(enumerate(self.electricity_price)),
        )

    def define_variables(self):
        self.model.aggregated_power_in = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.aggregated_heat_out = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.aggregated_cool_out = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )

    def define_objective(self):
        if self.objective == "minimize_cost":

            @self.model.Objective(sense=pyo.minimize)
            def obj_rule(m):
                return sum(
                    m.aggregated_power_in[t] * self.electricity_price[t]
                    for t in self.model.time_steps
                )

        elif self.objective == "maximize_comfort":
            # Define your comfort metric here
            pass
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    # Initialize units based on the list passed
    def initialize_components(self, components: Dict[str, Dict] = None):
        if self.components is None:
            logger.warning("No components specified for building")
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

    def initialize_storages_from_list(self, storage_list):
        for storage_type in storage_list:
            # Get the specific parameters for this storage type
            storage_params = self.unit_parameters.get(storage_type, {})

            # Initialize a Storage object based on its type and parameters
            storage_unit = Storage(type=storage_type, **storage_params)

            # Add the initialized storage unit to self.storage_units
            self.storage_units[storage_type] = storage_unit

            # Connect the storage unit based on its type
            if storage_type == "thermal":
                self.connect_thermal_units(storage_unit)
            elif storage_type == "electrical":
                self.connect_electrical_units(storage_unit)

    def define_constraints(self):
        @self.model.Constraint(self.model.time_steps)
        def aggregate_power_in_constraint(m, t):
            return m.aggregated_power_in[t] == sum(
                getattr(m, unit_name).power_in[t]
                for unit_name in self.components.keys()
            )

        @self.model.Constraint(self.model.time_steps)
        def aggregate_heat_out_constraint(m, t):
            return m.aggregated_heat_out[t] == sum(
                getattr(m, unit_name).heat_out[t]
                for unit_name in self.components.keys()
                if hasattr(getattr(m, unit_name), "heat_out")
            )

        @self.model.Constraint(self.model.time_steps)
        def aggregate_cool_out_constraint(m, t):
            return m.aggregated_cool_out[t] == sum(
                getattr(m, unit_name).cool_out[t]
                for unit_name in self.components.keys()
                if hasattr(getattr(m, unit_name), "cool_out")
            )

        @self.model.Constraint(self.model.time_steps)
        def heat_balance(m, t):
            return m.aggregated_heat_out[t] == m.heating_demand[t]

        @self.model.Constraint(self.model.time_steps)
        def cool_balance(m, t):
            return m.aggregated_cool_out[t] == self.cooling_demand[t]

    def connect_thermal_units(self, storage_unit):
        for unit_name, unit in self.components.items():
            if hasattr(unit, "heat_out"):
                # Add constraint or equation relating unit's heat_out and storage_unit's energy_in
                # This might be part of your optimization model
                ...

    def connect_electrical_units(self, storage_unit):
        for unit_name, unit in self.components.items():
            if hasattr(unit, "power_in"):
                # Add constraint or equation relating unit's power_in, storage_unit's power_in, and grid's power_out
                # This would also be part of your optimization model
                ...

    def run_optimization(self):
        # Create a solver
        solvers = check_available_solvers(*SOLVERS)
        if len(solvers) < 1:
            raise Exception(f"None of {SOLVERS} are available")

        solver = SolverFactory(solvers[0])

        # Solve the model
        self.results = solver.solve(self.model, tee=False)

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

    def calculate_marginal_cost(self, start, **kwargs):
        return self.electricity_price[start]
