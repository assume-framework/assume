import logging

import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import *


class HeatPump:
    def __init__(
        self,
        model,
        id,
        max_power,
        min_power,
        ramp_up,
        ramp_down,
        cop,
        **kwargs,
    ):
        self.model = model
        self.id = id
        self.max_power = max_power
        self.min_power = min_power
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.cop = cop

    def add_to_model(self, unit_block, time_steps):
        self.component_block = unit_block
        self.define_parameters()
        self.define_variables(time_steps)
        self.define_constraints(time_steps)

    def define_parameters(self) -> None:
        self.component_block.max_power = Param(initialize=self.max_power)
        self.component_block.min_power = Param(initialize=self.min_power)
        self.component_block.ramp_up = Param(initialize=self.ramp_up)
        self.component_block.ramp_down = Param(initialize=self.ramp_down)
        self.component_block.cop = Param(
            initialize=self.cop
        )  # Coefficient of Performance

    def define_variables(self, time_steps):
        self.component_block.heat_out = Var(time_steps, within=pyo.NonNegativeReals)
        self.component_block.power_in = Var(time_steps, within=pyo.NonNegativeReals)

    def define_constraints(self, time_steps) -> None:
        # Heat output bounds
        @self.component_block.Constraint(time_steps)
        def p_output_lower_bound(m, t):
            return m.power_in[t] >= m.min_power

        @self.component_block.Constraint(time_steps)
        def p_output_upper_bound(m, t):
            return m.power_in[t] <= m.max_power

        # Ramp up/down constraints
        @self.component_block.Constraint(time_steps)
        def ramp_up_constraint(m, t):
            if t == 0:
                return Constraint.Skip
            return m.heat_out[t] - m.heat_out[t - 1] <= m.ramp_up

        @self.component_block.Constraint(time_steps)
        def ramp_down_constraint(m, t):
            if t == 0:
                return Constraint.Skip
            return m.heat_out[t - 1] - m.heat_out[t] <= m.ramp_down

        # COP constraint
        @self.component_block.Constraint(time_steps)
        def cop_constraint(m, t):
            return m.power_in[t] == m.heat_out[t] / m.cop

    def define_objective_h(self):
        # Define the objective function specific to HeatPump if needed
        pass


class AirConditioner:
    def __init__(
        self,
        id: str,
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

    def add_to_model(self, unit_block, time_steps):
        self.component_block = unit_block
        self.define_parameters()
        self.define_variables(time_steps)
        self.define_constraints(time_steps)

    def define_parameters(self) -> None:
        self.component_block.max_power = Param(initialize=self.max_power)
        self.component_block.min_power = Param(initialize=self.min_power)
        self.component_block.ramp_up = Param(initialize=self.ramp_up)
        self.component_block.ramp_down = Param(initialize=self.ramp_down)
        self.component_block.cooling_factor = Param(
            initialize=self.cooling_factor
        )  # Add the cooling factor parameter

    def define_variables(self, time_steps):
        self.component_block.cool_out = Var(time_steps, within=pyo.NonNegativeReals)
        self.component_block.power_in = Var(time_steps, within=pyo.NonNegativeReals)

    def define_constraints(self, time_steps) -> None:
        # Heat output bounds
        @self.component_block.Constraint(time_steps)
        def p_output_lower_bound(m, t):
            return m.cool_out[t] >= m.min_power

        @self.component_block.Constraint(time_steps)
        def p_output_upper_bound(m, t):
            return m.cool_out[t] <= m.max_power

        # Ramp up/down constraints
        @self.component_block.Constraint(time_steps)
        def ramp_up_constraint(m, t):
            if t == 0:
                return Constraint.Skip
            return m.cool_out[t] - m.cool_out[t - 1] <= m.ramp_up

        @self.component_block.Constraint(time_steps)
        def ramp_down_constraint(m, t):
            if t == 0:
                return Constraint.Skip
            return m.cool_out[t - 1] - m.cool_out[t] <= m.ramp_down

        # Cooling factor constraint
        @self.component_block.Constraint(time_steps)
        def cooling_factor_constraint(m, t):
            return m.power_in[t] == m.cool_out[t] / m.cooling_factor

    def define_objective(self):
        # Define the objective function specific to HeatPump if needed
        pass


class Storage:
    def __init__(
        self,
        id: str,
        model,
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

class Electrolyser:
    def __init__(
        self,
        model,
        id,
        max_capacity,
        min_load, 
        standby_power,
        startup_cost,
        **kwargs
    ):
        self.model = model
        self.id = id

        self.max_capacity = max_capacity
        self.min_load = min_load
        self.standby_power = standby_power
        self.startup_cost = startup_cost

        # Define breakpoints and slopes for the piecewise expression here
        self.segments = [1, 2, 3, 4]
        self.breakpoints = [7.84, 16, 52.25]
        self.slopes = [24.51, -92.16, 17.93, 13.10]

    def add_to_model(self, unit_block, time_steps):
        self.component_block = unit_block
        self.define_parameters()
        self.define_variables(time_steps)
        self.define_constraints(time_steps)

    def define_parameters(self):
        self.component_block.max_capacity = Param(initialize=self.max_capacity)
        self.component_block.min_load = Param(initialize=self.min_load)
        self.component_block.standby_power = Param(initialize=self.standby_power)

        self.component_block.startup_cost = Param(initialize=self.startup_cost)

    def define_variables(self, time_steps):
        self.component_block.power_in = Var(time_steps, within=NonNegativeReals)
        self.component_block.hydrogen_produced = Var(time_steps, within=NonNegativeReals)

        self.component_block.is_online = Var(time_steps, within=Binary)
        self.component_block.is_offline = Var(time_steps, within=Binary)
        self.component_block.is_standby = Var(time_steps, within=Binary)
        self.component_block.is_startup = Var(time_steps, within=Binary)

        self.component_block.startup_cost_incurred = Var(time_steps, within=NonNegativeReals)

    def define_constraints(self, time_steps):
        @self.component_block.Constraint(time_steps)
        def operational_states(m, t):
            return m.is_online[t] + m.is_offline[t] + m.is_standby[t] == 1
        
        # Startup condition
        @self.component_block.Constraint(time_steps)
        def startup_condition(m, t):
            if t == 0:
                return Constraint.Skip
            return m.is_startup[t] >= m.is_online[t] - m.is_online[t-1] - m.is_standby[t-1]

        # Startup cost incurred
        @self.component_block.Constraint(time_steps)
        def startup_cost_constraint(m, t):
            return m.startup_cost_incurred[t] == m.startup_cost * m.is_startup[t]

        @self.component_block.Constraint(time_steps)
        def upper_bound(m, t):
            return m.power_in[t] <= m.max_capacity * m.is_online[t] + m.standby_power * m.is_standby[t]

        @self.component_block.Constraint(time_steps)
        def lower_bound(m, t):
            return m.power_in[t] >= m.min_load * m.is_online[t] + m.standby_power * m.is_standby[t]

        @self.component_block.Constraint(time_steps)
        def hydrogen_production(m, t):
            return m.hydrogen_produced[t] == Piecewise(
                ((self.slopes[0] * m.power_in[t]) + (self.slopes[1] * m.power_in[t]) + (self.slopes[2] * m.power_in[t]), m.power_in[t] <= self.breakpoints[0]),
                ((self.slopes[1] * m.power_in[t]) + (self.slopes[2] * m.power_in[t]), self.breakpoints[0] < m.power_in[t] <= self.breakpoints[1]),
                (self.slopes[2] * m.power_in[t], self.breakpoints[1] < m.power_in[t] <= self.breakpoints[2]),
                (0, True)  # Default value if none of the conditions are met
            )

    