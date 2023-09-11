import pyomo.environ as pyo
from pyomo.environ import *
import pandas as pd
from typing import List, Optional, Dict, Union
from assume.common.base import BaseUnit
from assume.units.dst_components import HeatPump, AirConditioner, Storage
from assume.common.market_objects import Product
from assume.common.forecasts import CsvForecaster, Forecaster
from importlib import import_module


class Building(BaseUnit):
    def __init__(
        self,
        id: str,
        unit_operator: str,
        technology: Union[str, List[str]],
        bidding_strategies: dict,
        node: str = "bus0",
        index: pd.DatetimeIndex = None,
        location: tuple[float, float] = (0.0, 0.0),
        storage_list: List = None,  # List of storage units
        unit_params: Dict[str, Dict] = None,
        unit_params_dict: dict = None,
        objective: str = "minimize_cost",
        forecaster: CsvForecaster = None,
        heating_demand: Optional[List[float]] = None,
        cooling_demand: Optional[List[float]] = None,
        **kwargs,
    ):
        print("Initializing Building")

        if isinstance(technology, str):
            technology = [tech.strip() for tech in technology.split(',')]

        super().__init__(
            id=id,
            unit_operator=unit_operator,
            technology=technology,
            bidding_strategies=bidding_strategies,
            index=index,
            node=node,
            forecaster=forecaster,
        )

        self.unit_list = technology
        self.storage_list = storage_list
        self.units = {}
        self.storage_units = {}
        # self.time_steps = pd.date_range(start, end, freq='H')
        self.storage_list = storage_list
        self.unit_params = unit_params_dict if unit_params_dict is not None else {}
        # self.unit_params = unit_params_dict if unit_params_dict is not None else {}
        self.heating_demand = forecaster['heating_demand']
        self.cooling_demand = forecaster['cooling_demand']
        self.electricity_price = forecaster.get_electricity_price(EOM='EOM')
        self.objective = objective

        self.create_model()

        # Initialize units based on the list passed
        # self.initialize_units_from_list(technology)
        self.initialize_units_from_list(self.unit_list, self.unit_params)
        self.define_constraints()

    def create_model(self):
        print("Creating Master Model for Building")
        self.model = ConcreteModel()
        self.define_sets()
        self.define_parameters()
        self.define_variables()
        # self.define_constraints()
        self.define_objective()

    def reset(self):
        """Reset the unit to its initial state."""
        pass

    # Initialize units based on the list passed
    def initialize_units_from_list(self, unit_names: list, unit_params: Dict[str, Dict]):
        """
        Initialize units from a dictionary of parameters.

        :param unit_names: List of unit names (e.g., technology types)
        :param unit_params: Dictionary of dictionaries containing technology-specific parameters
        """

        print("Initializing Units from Dictionary")

        for unit_name in unit_names:
            tech_params = unit_params.get(unit_name, {})
            unit_class = globals().get(unit_name)

            if unit_class is not None:
                # Add the unit name to the model's set of units
                self.model.units.add(unit_name)

                # Create a new Block for this unit
                unit_block = Block()
                self.model.add_component(unit_name, unit_block)

                # Merge global unit_params and technology-specific params
                merged_params = {**tech_params, **{'model': unit_block}}

                # Create and initialize the unit
                new_unit = unit_class(id=self.id, unit_operator=self.unit_operator,
                                      technology=unit_name, **merged_params)

                # Add the unit to the Building's Block and to the dictionary of units
                new_unit.add_to_model(
                    unit_block, self.model.units, self.model.time_steps
                )
                self.units[unit_name] = new_unit

                print(f"Units index set after initialization: {list(self.model.units)}")  # Debugging line
            else:
                print(f"Warning: No class found for unit name {unit_name}")

    # def initialize_units_from_list(self, technology):
    #     print("Initializing Units from List")
    #     for unit_name in technology:
    #         # unit_params = self.unit_params.get(unit_name, {})
    #         unit_class = globals().get(unit_name)
    #
    #         if unit_class is not None:
    #             self.model.units.add(unit_name)
    #             unit_block = Block()
    #             self.model.add_component(unit_name, unit_block)
    #             unit_params = self.unit_params
    #             unit_params["model"] = unit_block
    #             new_unit = unit_class(id, **unit_params)
    #             new_unit.add_to_model(
    #                 unit_block, self.model.units, self.model.time_steps
    #             )
    #             self.units[unit_name] = new_unit
    #             # Append the unit's master variables and constraints to the building's master lists
    #             print(
    #                 f"Units index set after initialization: {list(self.model.units)}"
    #             )  # Debugging line
    #         else:
    #             print(f"Warning: No class found for unit name {unit_name}")

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

    def connect_thermal_units(self, storage_unit):
        for unit_name, unit in self.units.items():
            if hasattr(unit, "heat_out"):
                # Add constraint or equation relating unit's heat_out and storage_unit's energy_in
                # This might be part of your optimization model
                ...

    def connect_electrical_units(self, storage_unit):
        for unit_name, unit in self.units.items():
            if hasattr(unit, "power_in"):
                # Add constraint or equation relating unit's power_in, storage_unit's power_in, and grid's power_out
                # This would also be part of your optimization model
                ...

    def define_sets(self) -> None:
        self.model.time_steps = Set(initialize=[idx for idx, _ in enumerate(self.index)])
        self.model.units = Set(ordered=True, initialize=self.units.keys())

    def define_parameters(self) -> None:
        self.model.heating_demand = Param(self.model.time_steps,
                                          initialize={idx: v for idx, v in enumerate(self.heating_demand)})
        self.model.cooling_demand = Param(self.model.time_steps,
                                          initialize={idx: v for idx, v in enumerate(self.cooling_demand)})
        self.model.electricity_price = Param(self.model.time_steps,
                                             initialize={idx: v for idx, v in enumerate(self.electricity_price)})

    def define_variables(self):

        self.model.aggregated_power_in = Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.aggregated_heat_out = Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.aggregated_cool_out = Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )

    def define_constraints(self):
        @self.model.Constraint(self.model.time_steps)
        def aggregate_power_in_constraint(m, t):
            return m.aggregated_power_in[t] == sum(
                getattr(m, unit_name).power_in[unit_name, t]
                for unit_name in self.units.keys()
            )

        @self.model.Constraint(self.model.time_steps)
        def aggregate_heat_out_constraint(m, t):
            return m.aggregated_heat_out[t] == sum(
                getattr(m, unit_name).heat_out[unit_name, t]
                for unit_name in self.units.keys()
                if hasattr(getattr(m, unit_name), "heat_out")
            )

        @self.model.Constraint(self.model.time_steps)
        def aggregate_cool_out_constraint(m, t):
            return m.aggregated_cool_out[t] == sum(
                getattr(m, unit_name).cool_out[unit_name, t]
                for unit_name in self.units.keys()
                if hasattr(getattr(m, unit_name), "cool_out")
            )

        @self.model.Constraint(self.model.time_steps)
        def heat_balance(m, t):
            return m.aggregated_heat_out[t] == m.heating_demand[t]

        @self.model.Constraint(self.model.time_steps)
        def cool_balance(m, t):
            return m.aggregated_cool_out[t] == self.cooling_demand[t]

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

    def run_optimization(self):

        # Create a solver
        solver = SolverFactory('gurobi')

        # Solve the model
        solver.solve(self.model, tee=True)
        results = solver.solve(self.model, tee=True)  # , tee=True
        # print(results)
        self.model.display()

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
