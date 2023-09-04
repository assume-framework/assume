import logging
import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import *


class HeatPump:
    def __init__(
        self, model, id, technology, max_power, min_power, ramp_up, ramp_down, cop
    ):
        self.model = model
        self.id = id
        self.max_power = max_power
        self.min_power = min_power
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.cop = cop

    def add_to_model(self, unit_block, units, time_steps):
        self.model = unit_block
        self.define_parameters(unit_block)
        self.define_variables(unit_block, units, time_steps)
        self.define_constraints(unit_block, units, time_steps)

    def define_parameters(self, unit_block) -> None:
        unit_block.max_power = Param(initialize=self.max_power)
        unit_block.min_power = Param(initialize=self.min_power)
        unit_block.ramp_up = Param(initialize=self.ramp_up)
        unit_block.ramp_down = Param(initialize=self.ramp_down)
        unit_block.cop = Param(initialize=self.cop)  # Coefficient of Performance

    def define_variables(self, unit_block, units, time_steps):
        unit_block.heat_out = Var(units, time_steps, within=pyo.NonNegativeReals)
        unit_block.power_in = Var(units, time_steps, within=pyo.NonNegativeReals)

    def define_constraints(self, unit_block, units, time_steps) -> None:
        # Heat output bounds
        @unit_block.Constraint(units, time_steps)
        def p_output_lower_bound(m, u, t):
            return m.power_in[u, t] >= m.min_power

        @unit_block.Constraint(units, time_steps)
        def p_output_upper_bound(m, u, t):
            return m.power_in[u, t] <= m.max_power

        # Ramp up/down constraints
        @unit_block.Constraint(units, time_steps)
        def ramp_up_constraint(m, u, t):
            if t == 0:
                return Constraint.Skip
            return m.heat_out[u, t] - m.heat_out[u, t - 1] <= m.ramp_up

        @unit_block.Constraint(units, time_steps)
        def ramp_down_constraint(m, u, t):
            if t == 0:
                return Constraint.Skip
            return m.heat_out[u, t - 1] - m.heat_out[u, t] <= m.ramp_down

        # COP constraint
        @unit_block.Constraint(units, time_steps)
        def cop_constraint(m, u, t):
            return m.power_in[u, t] == m.heat_out[u, t] / m.cop

    def define_objective_h(self):
        # Define the objective function specific to HeatPump if needed
        pass


class AirConditioner:
    def __init__(
        self,
        id: str,
        technology: str,
        model,
        max_power: float = None,
        min_power: float = None,
        ramp_up: float = None,  # Ramp-up rate per time step
        ramp_down: float = None,  # Ramp-down rate per time step
        cooling_factor: float = None,  # Power to cooling conversion factor
        **kwargs,
    ):
        self.model = model
        self.id = id
        self.max_power = max_power
        self.min_power = min_power
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.cooling_factor = cooling_factor

    def add_to_model(self, unit_block, units, time_steps):
        self.model = unit_block
        self.define_parameters(unit_block)
        self.define_variables(unit_block, units, time_steps)
        self.define_constraints(unit_block, units, time_steps)

    def define_parameters(self, unit_block) -> None:
        unit_block.max_power = Param(initialize=self.max_power)
        unit_block.min_power = Param(initialize=self.min_power)
        unit_block.ramp_up = Param(initialize=self.ramp_up)
        unit_block.ramp_down = Param(initialize=self.ramp_down)
        unit_block.cooling_factor = Param(
            initialize=self.cooling_factor
        )  # Add the cooling factor parameter

    def define_variables(self, unit_block, units, time_steps):
        unit_block.cool_out = Var(units, time_steps, within=pyo.NonNegativeReals)
        unit_block.power_in = Var(units, time_steps, within=pyo.NonNegativeReals)

    def define_constraints(self, unit_block, units, time_steps) -> None:
        # Heat output bounds
        @unit_block.Constraint(units, time_steps)
        def p_output_lower_bound(m, u, t):
            return m.cool_out[u, t] >= m.min_power

        @unit_block.Constraint(units, time_steps)
        def p_output_upper_bound(m, u, t):
            return m.cool_out[u, t] <= m.max_power

        # Ramp up/down constraints
        @unit_block.Constraint(units, time_steps)
        def ramp_up_constraint(m, u, t):
            if t == 0:
                return Constraint.Skip
            return m.cool_out[u, t] - m.cool_out[u, t - 1] <= m.ramp_up

        @unit_block.Constraint(units, time_steps)
        def ramp_down_constraint(m, u, t):
            if t == 0:
                return Constraint.Skip
            return m.cool_out[u, t - 1] - m.cool_out[u, t] <= m.ramp_down

        # Cooling factor constraint
        @unit_block.Constraint(units, time_steps)
        def cooling_factor_constraint(m, u, t):
            return m.power_in[u, t] == m.cool_out[u, t] / m.cooling_factor

    def define_objective(self):
        # Define the objective function specific to HeatPump if needed
        pass


class Storage:
    def __init__(
        self,
        id: str,
        model,
        unit_operator: str,
        technology: str,
        bidding_strategies: dict,
        storage_type: str,  # Either 'heat' or 'electricity'
        max_capacity: float = None,
        min_capacity: float = None,
        charge_efficiency: float = None,
        discharge_efficiency: float = None,
        **kwargs,
    ):
        self.model = model
        self.id = id
        self.storage_type = storage_type
        self.max_capacity = max_capacity
        self.min_capacity = min_capacity
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency

    def add_to_model(self, unit_storage_block, storage_units, time_steps):
        self.model = unit_storage_block
        self.define_parameters(unit_storage_block)
        self.define_variables(unit_storage_block, storage_units, time_steps)
        self.define_constraints(unit_storage_block, storage_units, time_steps)

    def define_parameters(self, unit_storage_block) -> None:
        unit_storage_block.max_capacity = Param(initialize=self.max_capacity)
        unit_storage_block.min_capacity = Param(initialize=self.min_capacity)
        unit_storage_block.charge_efficiency = Param(initialize=self.charge_efficiency)
        unit_storage_block.discharge_efficiency = Param(
            initialize=self.discharge_efficiency
        )

    def define_variables(self, unit_storage_block, storage_units, time_steps):
        unit_storage_block.state_of_charge = Var(
            storage_units, time_steps, domain=pyo.NonNegativeReals
        )
        unit_storage_block.energy_in = Var(
            storage_units, time_steps, domain=pyo.NonNegativeReals
        )
        unit_storage_block.energy_out = Var(
            storage_units, time_steps, domain=pyo.NonNegativeReals
        )

    def define_constraints(self, unit_storage_block, storage_units, time_steps) -> None:
        @unit_storage_block.Constraint(storage_units, time_steps)
        def soc_upper_limit(m, u, t):
            return m.state_of_charge[u, t] <= m.max_capacity

        @unit_storage_block.Constraint(storage_units, time_steps)
        def soc_lower_limit(m, u, t):
            return m.min_capacity <= m.state_of_charge[u, t]

        @unit_storage_block.Constraint(storage_units, time_steps)
        def charge_discharge_balance(m, u, t):
            if t == 0:
                return Constraint.Skip
            return (
                m.state_of_charge[u, t]
                == m.state_of_charge[u, t - 1]
                + m.charge_efficiency * m.energy_in[u, t]
                - m.energy_out[u, t] / m.discharge_efficiency
            )

        @unit_storage_block.Constraint(storage_units, time_steps)
        def energy_in_out_relation(m, u, t):
            if self.storage_type == "heat":
                return (
                    m.energy_in[u, t] == m.heat_out[u, t]
                )  # Replace with appropriate heat source
            elif self.storage_type == "electricity":
                return (
                    m.energy_in[t] == m.power_out[t]
                )  # Replace with appropriate power source

        # Additional constraints specific to the storage technology
