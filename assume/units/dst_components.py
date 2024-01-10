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
        max_power, 
        min_power, 
        ramp_up, 
        ramp_down,
        min_operating_time, 
        min_down_time, 
        downtime_hot_start, 
        downtime_warm_start,
        efficiency, 
        compressor_power, 
        **kwargs
    ):
        self.model = model
        self.id = id
        self.max_power = max_power
        self.min_power = min_power
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.min_operating_time = min_operating_time
        self.min_down_time = min_down_time
        self.downtime_hot_start = downtime_hot_start
        self.downtime_warm_start = downtime_warm_start
        self.efficiency = efficiency
        self.compressor_power = compressor_power

    def add_to_model(self, unit_block, time_steps):
        self.component_block = unit_block
        self.define_parameters()
        self.define_variables(time_steps)
        self.define_constraints(time_steps)

    def define_parameters(self):
        self.component_block.max_power = Param(initialize=self.max_power)
        self.component_block.min_power = Param(initialize=self.min_power)
        self.component_block.ramp_up = Param(initialize=self.ramp_up)
        self.component_block.ramp_down = Param(initialize=self.ramp_down)
        self.component_block.min_operating_time = Param(initialize=self.min_operating_time)
        self.component_block.min_down_time = Param(initialize=self.min_down_time)
        self.component_block.efficiency = Param(initialize=self.efficiency)
        self.component_block.compressor_power = Param(initialize=self.compressor_power)

    def define_variables(self, time_steps):
        self.component_block.power_in = Var(time_steps, within=pyo.NonNegativeReals)
        self.component_block.hydrogen_production = Var(time_steps, within=pyo.NonNegativeReals)
        self.component_block.operational_status = Var(time_steps, within=Binary)

    def define_constraints(self, time_steps):
        # Power bounds constraints
        @self.component_block.Constraint(time_steps)
        def power_upper_bound(m, t):
            return m.power_in[t] <= m.max_power * m.operational_status[t]

        @self.component_block.Constraint(time_steps)
        def power_lower_bound(m, t):
            return m.power_in[t] >= m.min_power * m.operational_status[t]

        # Ramp-up constraint
        @self.component_block.Constraint(time_steps)
        def ramp_up_constraint(m, t):
            if t == 0:
                return Constraint.Skip
            return m.power_in[t] - m.power_in[t-1] <= m.ramp_up

        # Ramp-down constraint
        @self.component_block.Constraint(time_steps)
        def ramp_down_constraint(m, t):
            if t == 0:
                return Constraint.Skip
            return m.power_in[t-1] - m.power_in[t] <= m.ramp_down

        # Minimum operating time constraint
        @self.component_block.Constraint(time_steps)
        def min_operating_time_constraint(m, t):
            if t >= m.min_operating_time:
                return sum(m.operational_status[i] for i in range(t-m.min_operating_time, t)) >= m.min_operating_time * m.operational_status[t]
            else:
                return Constraint.Skip

        # Minimum downtime constraint
        @self.component_block.Constraint(time_steps)
        def min_down_time_constraint(m, t):
            if t >= m.min_down_time:
                return sum(1 - m.operational_status[i] for i in range(t-m.min_down_time, t)) >= m.min_down_time * (1 - m.operational_status[t])
            else:
                return Constraint.Skip

        # Power consumption equation
        @self.component_block.Constraint(time_steps)
        def power_in_equation(m, t):
            return m.power_in[t] == m.hydrogen_output[t] / m.efficiency + m.compressor_power

class ShaftFurnace:
    def __init__(self, 
                 model, 
                 id, 
                 max_iron_ore_throughput, 
                 max_natural_gas_input, 
                 max_hydrogen_input, 
                 ramp_up, 
                 ramp_down, 
                 downtime_hot_start, 
                 downtime_warm_start, 
                 efficiency, 
                 **kwargs):
        
        self.model = model
        self.id = id
        # Operational parameters
        self.max_iron_ore_throughput = max_iron_ore_throughput
        self.max_natural_gas_input = max_natural_gas_input
        self.max_hydrogen_input = max_hydrogen_input
        # Additional operational characteristics
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.downtime_hot_start = downtime_hot_start
        self.downtime_warm_start = downtime_warm_start
        self.efficiency = efficiency
        # Reducing gas inlets
        self.natural_gas_inlet = False
        self.hydrogen_inlet = False

         # New attributes for dual hydrogen source handling
        self.max_direct_hydrogen_input = max_hydrogen_input  # Maximum direct hydrogen input from electrolyser
        self.max_stored_hydrogen_input = max_hydrogen_input  # Maximum stored hydrogen input from storage

    def add_to_model(self, unit_block, time_steps):
        self.component_block = unit_block
        self.define_parameters()
        self.define_variables(time_steps)
        self.define_constraints(time_steps)

    def configure_reducing_gases(self, reducing_gas_specification):
        self.natural_gas_inlet = reducing_gas_specification.get('natural_gas', False)
        self.hydrogen_inlet = reducing_gas_specification.get('hydrogen', False)

    def add_to_model(self, unit_block, time_steps):
        self.component_block = unit_block
        self.define_parameters()
        self.define_variables(time_steps)
        self.define_constraints(time_steps)

    def define_parameters(self):
        self.component_block.max_iron_ore_throughput = Param(initialize=self.max_iron_ore_throughput)
        self.component_block.max_natural_gas_input = Param(initialize=self.max_natural_gas_input)
        self.component_block.max_hydrogen_input = Param(initialize=self.max_hydrogen_input)
        self.component_block.ramp_up = Param(initialize=self.ramp_up)
        self.component_block.ramp_down = Param(initialize=self.ramp_down)
        self.component_block.downtime_hot_start = Param(initialize=self.downtime_hot_start)
        self.component_block.downtime_warm_start = Param(initialize=self.downtime_warm_start)
        self.component_block.efficiency = Param(initialize=self.efficiency)
        self.component_block.dri_production_efficiency = Param(initialize=self.dri_production_efficiency)

    def define_variables(self, time_steps):
        self.component_block.iron_ore_in = Var(time_steps, within=NonNegativeReals)
        self.component_block.natural_gas_in = Var(time_steps, within=NonNegativeReals)
        self.component_block.hydrogen_in = Var(time_steps, within=NonNegativeReals)
        self.component_block.dri_produced = Var(time_steps, within=NonNegativeReals)
        self.component_block.operational_status = Var(time_steps, within=Binary)
        self.component_block.dri_output = Var(time_steps, within=NonNegativeReals)

        # Variables for hydrogen input from both electrolyser and storage
        self.component_block.direct_hydrogen_in = Var(time_steps, within=NonNegativeReals)
        self.component_block.stored_hydrogen_in = Var(time_steps, within=NonNegativeReals)
        self.component_block.total_hydrogen_in = Var(time_steps, within=NonNegativeReals)

    def function_of_hydrogen(self, hydrogen_input):
        # Assuming a simple linear relationship
        # Replace with more accurate representation as needed
        efficiency_factor = self.dri_production_efficiency  # example efficiency factor
        return hydrogen_input * efficiency_factor

    def define_constraints(self, time_steps):
        # Constraint for iron ore throughput
        @self.component_block.Constraint(time_steps)
        def iron_ore_throughput_constraint(m, t):
            return m.iron_ore_in[t] <= m.max_iron_ore_throughput * m.operational_status[t]

        # Constraint for natural gas input
        @self.component_block.Constraint(time_steps)
        def natural_gas_input_constraint(m, t):
            return m.natural_gas_in[t] <= m.max_natural_gas_input * m.operational_status[t]

        # Constraint for direct hydrogen input from electrolyser
        @self.component_block.Constraint(time_steps)
        def direct_hydrogen_input_constraint(m, t):
            return m.direct_hydrogen_in[t] <= m.max_direct_hydrogen_input * m.operational_status[t]

        # Constraint for stored hydrogen input from hydrogen storage
        @self.component_block.Constraint(time_steps)
        def stored_hydrogen_input_constraint(m, t):
            return m.stored_hydrogen_in[t] <= m.max_stored_hydrogen_input * m.operational_status[t]

        # Constraint for total hydrogen input being the sum of direct and stored hydrogen inputs
        @self.component_block.Constraint(time_steps)
        def total_hydrogen_input_constraint(m, t):
            return m.total_hydrogen_in[t] == m.direct_hydrogen_in[t] + m.stored_hydrogen_in[t]

        # Constraint for DRI output
        @self.component_block.Constraint(time_steps)
        def dri_output_constraint(m, t):
            # Assuming a linear relationship between hydrogen input and DRI output
            # Adjust the formula as needed to match your process characteristics
            return m.dri_output[t] == m.iron_ore_in[t] * m.dri_production_efficiency * self.function_of_hydrogen(m.total_hydrogen_in[t])

class ElectricArcFurnace:
    def __init__(self, 
                 model, 
                 id, 
                 max_power_input, 
                 max_dri_input, 
                 max_scrap_input, 
                 efficiency, 
                 ramp_up, 
                 ramp_down, 
                 downtime_hot_start, 
                 downtime_warm_start, 
                 **kwargs):
        
        self.model = model
        self.id = id
        # Operational parameters
        self.max_power_input = max_power_input
        self.max_dri_input = max_dri_input
        self.max_scrap_input = max_scrap_input
        self.efficiency = efficiency
        # Additional operational characteristics
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.downtime_hot_start = downtime_hot_start
        self.downtime_warm_start = downtime_warm_start

    def add_to_model(self, unit_block, time_steps):
        self.component_block = unit_block
        self.define_parameters()
        self.define_variables(time_steps)
        self.define_constraints(time_steps)

    def define_parameters(self):
        self.component_block.max_power_input = Param(initialize=self.max_electricity_input)
        self.component_block.max_dri_input = Param(initialize=self.max_dri_input)
        self.component_block.max_scrap_input = Param(initialize=self.max_scrap_input)
        self.component_block.efficiency = Param(initialize=self.efficiency)
        self.component_block.ramp_up = Param(initialize=self.ramp_up)
        self.component_block.ramp_down = Param(initialize=self.ramp_down)
        self.component_block.downtime_hot_start = Param(initialize=self.downtime_hot_start)
        self.component_block.downtime_warm_start = Param(initialize=self.downtime_warm_start)

    def define_variables(self, time_steps):
        self.component_block.power_in = Var(time_steps, within=NonNegativeReals)
        self.component_block.dri_input = Var(time_steps, within=NonNegativeReals)
        self.component_block.scrap_input = Var(time_steps, within=NonNegativeReals)
        self.component_block.steel_output = Var(time_steps, within=NonNegativeReals)
        self.component_block.operational_status = Var(time_steps, within=Binary)

    def define_constraints(self, time_steps):
        @self.component_block.Constraint(time_steps)
        def electricity_input_constraint(m, t):
            return m.power_in[t] <= m.max_power_input * m.operational_status[t]

        @self.component_block.Constraint(time_steps)
        def dri_input_constraint(m, t):
            return m.dri_input[t] <= m.max_dri_input * m.operational_status[t]

        @self.component_block.Constraint(time_steps)
        def scrap_input_constraint(m, t):
            return m.scrap_input[t] <= m.max_scrap_input * m.operational_status[t]

        @self.component_block.Constraint(time_steps)
        def steel_output_constraint(m, t):
            # This constraint defines the steel output based on inputs and efficiency
            return m.steel_output[t] == (m.dri_input[t] + m.scrap_input[t]) * m.efficiency

    # Additional methods for processing, ramping up/down, etc.
  
    