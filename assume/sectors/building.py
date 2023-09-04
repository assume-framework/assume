import pyomo.environ as pyo
from pyomo.environ import *
import pandas as pd
from typing import List, Optional, Dict
from assume.units.base_unit import BaseUnit
from assume.units.dst_units import HeatPump, AirConditioner, Storage
from importlib import import_module


class Building(BaseUnit):
    def __init__(
        self,
        id: str,
        unit_operator: str,
        technology: str,
        bidding_strategies: dict,
        node: str = "bus0",
        index: pd.DatetimeIndex = None,
        time_steps: pd.DatetimeIndex = None,
        location: tuple[float, float] = (0.0, 0.0),
        unit_list: List = None,  # List of units
        storage_list: List = None,  # List of storage units
        unit_parameters: dict = None,
        objective: str = "minimize_cost",
        heating_demand: Optional[List[float]] = None,
        cooling_demand: Optional[List[float]] = None,
        electricity_price: List[float] = None,
        **kwargs,
    ):
        print("Initializing Building")
        super().__init__(
            id=id,
            unit_operator=unit_operator,
            technology=technology,
            bidding_strategies=bidding_strategies,
            index=index,
            node=node,
        )

        self.unit_list = unit_list
        self.storage_list = storage_list
        self.units = {}
        self.storage_units = {}
        self.time_steps = time_steps
        self.storage_list = storage_list
        self.unit_parameters = unit_parameters
        self.heating_demand = heating_demand
        self.cooling_demand = cooling_demand
        self.electricity_price = electricity_price
        self.objective = objective

        self.create_model()

        # Initialize units based on the list passed
        self.initialize_units_from_list(unit_list)
        self.define_constraints()

    def create_model(self):
        print("Creating Master Model for Building")
        self.model = ConcreteModel()
        self.define_sets()
        self.define_parameters()
        self.define_variables()
        # self.define_constraints()
        self.define_objective()

    # Initialize units based on the list passed
    def initialize_units_from_list(self, unit_list):
        print("Initializing Units from List")
        print(f"Unit list received: {unit_list}")
        for unit_name in unit_list:
            unit_params = self.unit_parameters.get(unit_name, {})
            unit_class = globals().get(unit_name)

            if unit_class is not None:
                self.model.units.add(unit_name)
                unit_block = Block()
                self.model.add_component(unit_name, unit_block)

                unit_params["model"] = unit_block
                new_unit = unit_class(**unit_params)
                new_unit.add_to_model(
                    unit_block, self.model.units, self.model.time_steps
                )
                self.units[unit_name] = new_unit
                # Append the unit's master variables and constraints to the building's master lists
                print(
                    f"Units index set after initialization: {list(self.model.units)}"
                )  # Debugging line
            else:
                print(f"Warning: No class found for unit name {unit_name}")

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
        self.model.time_steps = Set(
            initialize=[idx for idx, _ in enumerate(self.time_steps)]
        )
        self.model.units = Set(ordered=True, initialize=self.units.keys())

    def define_parameters(self) -> None:
        self.model.heating_demand = Param(
            self.model.time_steps,
            initialize={idx: v for idx, v in enumerate(self.heating_demand)},
        )
        self.model.cooling_demand = Param(
            self.model.time_steps,
            initialize={idx: v for idx, v in enumerate(self.cooling_demand)},
        )

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
        print("Defining Building Constraints")
        print(
            f"Units index set just before constraint definition: {list(self.model.units)}"
        )
        print(
            f"Time steps index set just before constraint definition: {list(self.model.time_steps)}"
        )

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
