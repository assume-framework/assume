import logging
from collections import defaultdict

import pandas as pd
from assume.common.forecasts import Forecaster
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
        self.b = unit_block
        self.define_parameters()
        self.define_variables(time_steps)
        self.define_constraints(time_steps)

    def define_parameters(self) -> None:
        self.b.max_power = Param(initialize=self.max_power)
        self.b.min_power = Param(initialize=self.min_power)
        self.b.ramp_up = Param(initialize=self.ramp_up)
        self.b.ramp_down = Param(initialize=self.ramp_down)
        self.b.cop = Param(
            initialize=self.cop
        )  # Coefficient of Performance

    def define_variables(self, time_steps):
        self.b.heat_out = Var(time_steps, within=pyo.NonNegativeReals)
        self.b.power_in = Var(time_steps, within=pyo.NonNegativeReals)

    def define_constraints(self, time_steps) -> None:
        # Heat output bounds
        @self.b.Constraint(time_steps)
        def p_output_lower_bound(b, t):
            return b.power_in[t] >= b.min_power

        @self.b.Constraint(time_steps)
        def p_output_upper_bound(b, t):
            return b.power_in[t] <= b.max_power

        # Ramp up/down constraints
        @self.b.Constraint(time_steps)
        def ramp_up_constraint(b, t):
            if t == 0:
                return Constraint.Skip
            return b.heat_out[t] - b.heat_out[t - 1] <= b.ramp_up

        @self.b.Constraint(time_steps)
        def ramp_down_constraint(b, t):
            if t == 0:
                return Constraint.Skip
            return b.heat_out[t - 1] - b.heat_out[t] <= b.ramp_down

        # COP constraint
        @self.b.Constraint(time_steps)
        def cop_constraint(b, t):
            return b.power_in[t] == b.heat_out[t] / b.cop

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
        self.b = unit_block
        self.define_parameters()
        self.define_variables(time_steps)
        self.define_constraints(time_steps)

    def define_parameters(self) -> None:
        self.b.max_power = Param(initialize=self.max_power)
        self.b.min_power = Param(initialize=self.min_power)
        self.b.ramp_up = Param(initialize=self.ramp_up)
        self.b.ramp_down = Param(initialize=self.ramp_down)
        self.b.cooling_factor = Param(
            initialize=self.cooling_factor
        )  # Add the cooling factor parameter

    def define_variables(self, time_steps):
        self.b.cool_out = Var(time_steps, within=pyo.NonNegativeReals)
        self.b.power_in = Var(time_steps, within=pyo.NonNegativeReals)

    def define_constraints(self, time_steps) -> None:
        # Heat output bounds
        @self.b.Constraint(time_steps)
        def p_output_lower_bound(b, t):
            return b.cool_out[t] >= b.min_power

        @self.b.Constraint(time_steps)
        def p_output_upper_bound(b, t):
            return b.cool_out[t] <= b.max_power

        # Ramp up/down constraints
        @self.b.Constraint(time_steps)
        def ramp_up_constraint(b, t):
            if t == 0:
                return Constraint.Skip
            return b.cool_out[t] - b.cool_out[t - 1] <= b.ramp_up

        @self.b.Constraint(time_steps)
        def ramp_down_constraint(b, t):
            if t == 0:
                return Constraint.Skip
            return b.cool_out[t - 1] - b.cool_out[t] <= b.ramp_down

        # Cooling factor constraint
        @self.b.Constraint(time_steps)
        def cooling_factor_constraint(b, t):
            return b.power_in[t] == b.cool_out[t] / b.cooling_factor

    def define_objective(self):
        # Define the objective function specific to HeatPump if needed
        pass

class Electrolyser:
    def __init__(
        self, 
        model, 
        id, 
        rated_power, 
        min_power, 
        ramp_up, 
        ramp_down,
        min_operating_time, 
        min_down_time,
        efficiency,
        start_price,
        fuel_type,
        **kwargs
    ):
        self.model = model
        self.id = id
        self.fuel_type = fuel_type
        self.rated_power = rated_power
        self.min_power = min_power
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.min_operating_time = min_operating_time
        self.min_down_time = min_down_time
        self.efficiency = efficiency
        self.start_price = start_price

    def add_to_model(self, unit_block, time_steps):
        
        self.b = unit_block
        self.define_parameters()
        self.define_variables(time_steps)
        self.define_constraints(time_steps)

    def define_parameters(self):
        self.b.rated_power = Param(initialize=self.rated_power)
        self.b.min_power = Param(initialize=self.min_power)
        self.b.ramp_up = Param(initialize=self.ramp_up)
        self.b.ramp_down = Param(initialize=self.ramp_down)
        self.b.min_operating_time = Param(initialize=self.min_operating_time)
        self.b.min_down_time = Param(initialize=self.min_down_time)
        self.b.efficiency = Param(initialize=self.efficiency)

        self.b.start_price= Param(initialize=self.start_price)

    def define_variables(self, time_steps):
        self.b.power_in = Var(time_steps, within=pyo.NonNegativeReals)
        self.b.hydrogen_out = Var(time_steps, within=pyo.NonNegativeReals)
        self.b.in_operation = Var(time_steps, within=Boolean)
        self.b.start_up = Var(time_steps, within=Boolean)
        self.b.shut_down = Var(time_steps, within=Boolean)

        self.b.electricity_cost = Var(time_steps, within=pyo.NonNegativeReals)
        self.b.start_cost = Var(time_steps, within=pyo.NonNegativeReals) 


    def define_constraints(self, time_steps):
        
        # Power bounds constraints
        @self.b.Constraint(time_steps)
        def power_upper_bound(b, t):
            return b.power_in[t] <= b.rated_power * b.in_operation[t]

        @self.b.Constraint(time_steps)
        def power_lower_bound(b, t):
            return b.power_in[t] >= b.min_power * b.in_operation[t]

        # Ramp-up constraint
        @self.b.Constraint(time_steps)
        def ramp_up_constraint(b, t):
            if t == 0:
                return Constraint.Skip
            return b.power_in[t] - b.power_in[t-1] <= b.ramp_up #* b.in_operation[t-1]

        # Ramp-down constraint
        @self.b.Constraint(time_steps)
        def ramp_down_constraint(b, t):
            if t == 0:
                return Constraint.Skip
            return b.power_in[t-1] - b.power_in[t] <= b.ramp_down #* b.in_operation[t]

        # Plant running constraint
        @self.b.Constraint(time_steps)
        def in_operation_rule(b, t):
            if t == 0:
                return b.in_operation[t] ==  b.start_up[t]
            return b.in_operation[t] - b.in_operation[t-1] ==  b.start_up[t] - b.shut_down[t]

        # Switch on and off constraints
        @self.b.Constraint(time_steps)
        def start_up_off_rule(b, t):
            return b.start_up[t] + b.shut_down[t] <= 1
        
        @self.model.Constraint(time_steps)
        def shut_down_logic(b, t):
         return b.shut_down[t] >= - b.power_in[t]
    
        @self.b.Constraint(time_steps)
        def min_operating_time_constraint(b, t):
            if t == 0:
                return Constraint.Skip  # No constraint for the first time step
            # Calculate the time step duration dynamically
            delta_t = t - (t-1)
            # Convert minimum operating time to the time unit of your choice
            min_operating_time_units = int(self.min_operating_time / delta_t)

            if t < min_operating_time_units:
                return b.start_up[t] == 1
            else:
                return sum(b.start_up[t-i] for i in range(min_operating_time_units)) >= min_operating_time_units * b.start_up[t]
            
        @self.b.Constraint(time_steps)
        def min_downtime_constraint(b, t):
            if t == 0:
                return Constraint.Skip  # No constraint for the first time step
            # Calculate the time step duration dynamically
            delta_t = t - (t-1)
            # Convert minimum downtime to the time unit of your choice
            min_downtime_units = int(self.min_down_time / delta_t)
            if t < min_downtime_units:
                return b.shut_down[t] == 1
            else:
                return sum(1 - b.in_operation[t-i] for i in range(min_downtime_units)) >= min_downtime_units * b.shut_down[t] 
        
        @self.b.Constraint(time_steps)
        def operating_cost_with_el_price(b, t):
            return b.electricity_cost[t] == b.power_in[t] * self.model.electricity_price[t] 
        
        # Efficiency constraint
        @self.b.Constraint(time_steps)
        def power_in_equation(b, t):
            return  b.power_in[t] == b.hydrogen_out[t] / b.efficiency 
        
        # Switch on and off constraints
        @self.b.Constraint(time_steps)
        def start_up_cost(b, t):
            return b.start_cost[t] == b.start_up[t] * b.start_price
        
class DriPlant:
    def __init__(self, 
                 model, 
                 id, 
                 max_iron_ore_throughput, 
                 max_natural_gas_input, 
                 max_hydrogen_input, 
                 efficiency,
                 specific_hydrogen_consumption,
                 specific_natural_gas_consumption,
                 fuel_type,
                 ramp_up,
                 ramp_down,
                 min_operating_time,
                 min_down_time,
                 **kwargs):
        
        self.model = model
        self.id = id
        self.fuel_type = fuel_type
        # Operational parameters
        self.max_iron_ore_throughput = max_iron_ore_throughput
        self.max_natural_gas_input = max_natural_gas_input
        self.max_hydrogen_input = max_hydrogen_input
        # Additional operational characteristics
        self.efficiency = efficiency
        self.specific_hydrogen_consumption = specific_hydrogen_consumption
        self.specific_natural_gas_consumption = specific_natural_gas_consumption
        # Flexibility parameters
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.min_operating_time = min_operating_time
        self.min_down_time = min_down_time

    def add_to_model(self, unit_block, time_steps):
        self.b = unit_block
        self.define_parameters()
        self.define_variables(time_steps)
        self.define_constraints(time_steps)

    def define_parameters(self):
        self.b.max_iron_ore_throughput = Param(initialize=self.max_iron_ore_throughput)
        self.b.max_natural_gas_input = Param(initialize=self.max_natural_gas_input)
        self.b.max_hydrogen_input = Param(initialize=self.max_hydrogen_input)
        self.b.efficiency_dri = Param(initialize=self.efficiency)
        self.b.specific_hydrogen_consumption = Param(initialize=self.specific_hydrogen_consumption)
        self.b.specific_natural_gas_consumption = Param(initialize=self.specific_natural_gas_consumption)
        # Flexibility parameters
        self.b.ramp_up_dri = Param(initialize=self.ramp_up)
        self.b.ramp_down_dri = Param(initialize=self.ramp_down)
        self.b.min_operating_time_dri = Param(initialize=self.min_operating_time)
        self.b.min_down_time_dri = Param(initialize=self.min_down_time)

    def define_variables(self, time_steps):
        self.b.iron_ore_in = Var(time_steps, within=NonNegativeReals)
        self.b.natural_gas_in = Var(time_steps, within=NonNegativeReals)
        self.b.dri_operating_cost = Var(time_steps, within=NonNegativeReals)
        self.b.hydrogen_in = Var(time_steps, within=NonNegativeReals)
        self.b.operational_status = Var(time_steps, within=Binary)
        self.b.dri_output = Var(time_steps, within=NonNegativeReals)

    def function_of_hydrogen(self, hydrogen_in):
        # Assuming a simple linear relationship
        # Replace with more accurate representation as needed
        efficiency_factor = self.dri_production_efficiency  # example efficiency factor
        return hydrogen_in * efficiency_factor
    
    def define_constraints(self, time_steps):
        @self.b.Constraint(time_steps)
        def iron_ore_throughput_constraint(b, t):
            return b.iron_ore_in[t] <= b.max_iron_ore_throughput

        @self.b.Constraint(time_steps)
        def hydrogen_input_constraint(b, t):
            if self.fuel_type == "hydrogen" or self.fuel_type == "both":
                return b.hydrogen_in[t] <= b.max_hydrogen_input
            else:
                return Constraint.Skip

        @self.b.Constraint(time_steps)
        def natural_gas_input_constraint(b, t):
            if self.fuel_type == "natural_gas" or self.fuel_type == "both":
                return b.natural_gas_in[t] <= b.max_natural_gas_input
            else:
                return Constraint.Skip
            
        @self.b.Constraint(time_steps)
        def iron_ore_constraint(b, t):
            return b.iron_ore_in[t] == b.dri_output[t] /  b.efficiency_dri

        @self.b.Constraint(time_steps)
        def dri_output_constraint(b, t):
            if self.fuel_type == "hydrogen":
                return b.dri_output[t] == b.hydrogen_in[t] * b.specific_hydrogen_consumption
            elif self.fuel_type == "natural_gas":
                return b.dri_output[t] == b.natural_gas_in[t] * b.specific_natural_gas_consumption
            elif self.fuel_type == "both":
                return b.dri_output[t] == b.hydrogen_in[t] * b.specific_hydrogen_consumption + b.natural_gas_in[t] * b.specific_natural_gas_consumption
            

        # Flexibility constraints
        @self.b.Constraint(time_steps)
        def ramp_up_dri_constraint(b, t):
            if t == 0:
                return Constraint.Skip
            else:
                return b.dri_output[t] - b.dri_output[t-1] <= b.ramp_up
        
        @self.b.Constraint(time_steps)
        def ramp_down_dri_constraint(b, t):
            if t == 0:
                return Constraint.Skip
            else:
                return b.dri_output[t-1] - b.dri_output[t] <= b.ramp_down
        
        @self.b.Constraint(time_steps)
        def min_operating_time_dri__constraint(b, t):
            if t == 0:
                return pyo.Constraint.Skip

            # Calculate the time step duration dynamically
            delta_t = t - (t-1)
            # Convert minimum operating time to the time unit of your choice
            min_operating_time_units = int(self.min_operating_time / delta_t)

            if t < min_operating_time_units:
                # Ensure that the cumulative sum of DRI production over the past min_operating_time_units time steps is at least min_operating_time_units times the production at time step t
                return sum(b.dri_output[i] for i in range(t - min_operating_time_units + 1, t + 1)) >= min_operating_time_units * b.dri_output[t]
            else:
                return sum(b.dri_output[i] for i in range(t - min_operating_time_units + 1, t + 1)) >= min_operating_time_units * b.dri_output[t]

        @self.b.Constraint(time_steps)
        def min_down_time_dri_constraint(b, t):
            if t == 0:
                return pyo.Constraint.Skip  # No constraint for the first time step

            # Calculate the time step duration dynamically
            delta_t = t - (t-1)
            # Convert minimum downtime to the time unit of your choice
            min_downtime_units = int(self.min_down_time / delta_t)

            if t < min_downtime_units:
                # Ensure that the cumulative sum of DRI production over the past min_downtime_units time steps is at least min_downtime_units times the production at time step t
                return sum(b.dri_output[t-i] for i in range(min_downtime_units)) >= min_downtime_units * b.dri_output[t]
            else:
                return sum(b.dri_output[t-i] for i in range(min_downtime_units)) >= min_downtime_units * b.dri_output[t]
            
        # Operational cost
        @self.b.Constraint(time_steps)
        def dri_operating_cost_constraint(b, t):
            # This constraint defines the steel output based on inputs and efficiency
            return b.dri_operating_cost[t] == b.natural_gas_in[t] * self.model.natural_gas_price[t]

class ElectricArcFurnace:
    def __init__(self, 
                 model, 
                 id, 
                 rated_power, 
                 min_power,
                 max_dri_input,
                 specific_electricity_demand, 
                 specific_dri_demand,
                 ramp_up,
                 ramp_down,
                 min_operating_time,
                 min_down_time,
                 **kwargs):
        
        self.model = model
        self.id = id
        # Operational parameters
        self.rated_power = rated_power
        self.min_power = min_power
        self.max_dri_input = max_dri_input
        # Additional operational characteristics
        self.specific_electricity_demand = specific_electricity_demand
        self.specific_dri_demand = specific_dri_demand
        # Flexibility parameters
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.min_operating_time = min_operating_time
        self.min_down_time = min_down_time

    def add_to_model(self, unit_block, time_steps):
        self.b = unit_block
        self.define_parameters()
        self.define_variables(time_steps)
        self.define_constraints(time_steps)

    def define_parameters(self):
        self.b.rated_power_eaf = Param(initialize=self.rated_power)
        self.b.min_power_eaf = Param(initialize=self.min_power)
        self.b.max_dri_input = Param(initialize=self.max_dri_input)
        self.b.specific_electricity_demand = Param(initialize=self.specific_electricity_demand)
        self.b.specific_dri_demand = Param(initialize=self.specific_dri_demand)
        # Flexibility parameters
        self.b.ramp_up_eaf = pyo.Param(initialize=self.ramp_up)
        self.b.ramp_down_eaf = pyo.Param(initialize=self.ramp_down)
        self.b.min_operating_time_eaf = pyo.Param(initialize=self.min_operating_time)
        self.b.min_down_time_eaf = pyo.Param(initialize=self.min_down_time)

    def define_variables(self, time_steps):
        self.b.power_eaf = Var(time_steps, within=NonNegativeReals)
        self.b.dri_input = Var(time_steps, within=NonNegativeReals)
        self.b.steel_output = Var(time_steps, within=NonNegativeReals)
        self.b.eaf_operating_cost = Var(time_steps, within=NonNegativeReals)

    def define_constraints(self, time_steps):

        # Power bounds constraints
        @self.b.Constraint(time_steps)
        def electricity_input_upper_bound(b, t):
            return b.power_eaf[t] <= b.rated_power_eaf

        @self.b.Constraint(time_steps)
        def electricity_input_lower_bound(b, t):
            return b.power_eaf[t] >= b.min_power_eaf
        
        @self.b.Constraint(time_steps)
        def steel_output_dri_relation(b, t):
            # This constraint defines the steel output based on inputs and efficiency
            return b.dri_input[t] ==  b.steel_output[t] * b.specific_dri_demand
        
        @self.b.Constraint(time_steps)
        def steel_output_power_relation(b, t):
            # This constraint defines the steel output based on inputs and efficiency
            return b.power_eaf[t] == b.steel_output[t] * b.specific_electricity_demand
        
        # Flexibility constraints
        @self.b.Constraint(time_steps)
        def ramp_up_eaf_constraint(b, t):
            if t == 0:
                return pyo.Constraint.Skip
            return b.power_eaf[t] - b.power_eaf[t-1] <= b.ramp_up_eaf

        @self.b.Constraint(time_steps)
        def ramp_down_eaf_constraint(b, t):
            if t == 0:
                return pyo.Constraint.Skip
            return b.power_eaf[t-1] - b.power_eaf[t] <= b.ramp_down_eaf

        @self.b.Constraint(time_steps)
        def min_operating_time_eaf_constraint(b, t):
            if t == 0:
                return pyo.Constraint.Skip  # No constraint for the first time step
            # Calculate the time step duration dynamically
            delta_t = t - (t-1)
            # Convert minimum operating time to the time unit of your choice
            min_operating_time_units = int(self.min_operating_time / delta_t)
            
            if t < min_operating_time_units:
                # Ensure that the cumulative sum of DRI production over the past min_operating_time_units time steps is at least min_operating_time_units times the production at time step t
                return sum(b.steed_output[i] for i in range(t - min_operating_time_units + 1, t + 1)) >= min_operating_time_units * b.steel_output[t]
            else:
                return sum(b.steel_output[i] for i in range(t - min_operating_time_units + 1, t + 1)) >= min_operating_time_units * b.steel_output[t]

        @self.b.Constraint(time_steps)
        def min_down_time_eaf_constraint(b, t):
            if t == 0:
                return pyo.Constraint.Skip  # No constraint for the first time step
            # Calculate the time step duration dynamically
            delta_t = t - (t-1)
            # Convert minimum downtime to the time unit of your choice
            min_downtime_units = int(self.min_down_time / delta_t)
            
            if t < min_downtime_units:
                return sum(b.steel_output[t-i] for i in range(min_downtime_units)) >= min_downtime_units * b.steel_output[t]
            else:
                return sum(b.steel_output[t-i] for i in range(min_downtime_units)) >= min_downtime_units * b.steel_output[t]
            
        # operational cost
        @self.b.Constraint(time_steps)
        def eaf_operating_cost_cosntraint(b, t):
            # This constraint defines the steel output based on inputs and efficiency
            return b.eaf_operating_cost[t] == b.power_eaf[t] * self.model.electricity_price[t]
        
class GenericStorage:
    def __init__(self, 
                 model, 
                 id,
                 max_capacity, 
                 min_capacity, 
                 initial_soc, 
                 storage_loss_rate, 
                 charge_loss_rate, 
                 discharge_loss_rate,
                 **kwargs
                 ):
        self.model = model
        self.id = id
        self.max_capacity = max_capacity
        self.min_capacity = min_capacity
        self.initial_soc = initial_soc
        self.storage_loss_rate = storage_loss_rate
        self.charge_loss_rate = charge_loss_rate
        self.discharge_loss_rate = discharge_loss_rate

    def add_to_model(self, unit_block, time_steps):
        self.b = unit_block
        self.define_parameters()
        self.define_variables(time_steps)
        self.define_constraints(time_steps)

    def define_parameters(self):
        self.b.max_capacity = Param(initialize=self.max_capacity)
        self.b.min_capacity = Param(initialize=self.min_capacity)
        self.b.initial_soc = Param(initialize=self.initial_soc)
        self.b.storage_loss_rate = Param(initialize=self.storage_loss_rate)
        self.b.charge_loss_rate = Param(initialize=self.charge_loss_rate)
        self.b.discharge_loss_rate = Param(initialize=self.discharge_loss_rate)

    def define_variables(self, time_steps):
        self.b.soc = Var(time_steps, within=NonNegativeReals) 
        self.b.uniformity_indicator = Var(time_steps, within=Binary)

         # Define the variables for power and hydrogen
        self.b.charge = Var(time_steps, within=NonNegativeReals)
        self.b.discharge = Var(time_steps, within=NonNegativeReals)

    def define_constraints(self, time_steps):
        @self.b.Constraint(time_steps)
        def storage_min_capacity_constraint(b, t):
            return b.soc[t] >= b.min_capacity

        @self.b.Constraint(time_steps)
        def storage_max_capacity_constraint(b, t):
            return b.soc[t] <= b.max_capacity
        
        @self.b.Constraint(time_steps)
        def energy_in_max_capacity_constraint(b, t):
            return b.charge[t] <= b.max_capacity #* b.uniformity_indicator[t]

        @self.b.Constraint(time_steps)
        def energy_out_max_capacity_constraint(b, t):
            return  b.discharge[t] <= b.max_capacity #* (1 - b.uniformity_indicator[t])

        @self.b.Constraint(time_steps)
        def energy_in_uniformity_constraint(b, t):
            return b.charge[t] <= b.max_capacity * b.uniformity_indicator[t]

        @self.b.Constraint(time_steps)
        def energy_out_uniformity_constraint(b, t):
            return  b.discharge[t] <= b.max_capacity * (1 - b.uniformity_indicator[t])

        @self.b.Constraint(self.model.time_steps)
        def storage_capacity_change_constraint(b, t):
            return b.soc[t] == (
                ((b.soc[t - 1] if t > 0 else b.initial_soc) * (1 - b.storage_loss_rate)) +
                ((1 - b.charge_loss_rate) * b.charge[t]) -
                ((1 + b.discharge_loss_rate) * b.discharge[t])
            )
        
class DRIStorage:
    def __init__(self, 
                 model, 
                 id,
                 max_capacity, 
                 min_capacity, 
                 initial_soc, 
                 storage_loss_rate, 
                 charge_loss_rate, 
                 discharge_loss_rate,
                 **kwargs
                 ):
        self.model = model
        self.id = id
        self.max_capacity = max_capacity
        self.min_capacity = min_capacity
        self.initial_soc = initial_soc
        self.storage_loss_rate = storage_loss_rate
        self.charge_loss_rate = charge_loss_rate
        self.discharge_loss_rate = discharge_loss_rate

    def add_to_model(self, unit_block, time_steps):
        self.b = unit_block
        self.define_parameters()
        self.define_variables(time_steps)
        self.define_constraints(time_steps)

    def define_parameters(self):
        self.b.max_capacity_dri = Param(initialize=self.max_capacity)
        self.b.min_capacity_dri = Param(initialize=self.min_capacity)
        self.b.initial_soc_dri = Param(initialize=self.initial_soc)
        self.b.storage_loss_rate_dri = Param(initialize=self.storage_loss_rate)
        self.b.charge_loss_rate_dri = Param(initialize=self.charge_loss_rate)
        self.b.discharge_loss_rate_dri = Param(initialize=self.discharge_loss_rate)

    def define_variables(self, time_steps):
        self.b.soc_dri = Var(time_steps, within=NonNegativeReals) 
        self.b.uniformity_indicator_dri = Var(time_steps, within=Binary)

         # Define the variables for power and hydrogen
        self.b.charge_dri = Var(time_steps, within=NonNegativeReals)
        self.b.discharge_dri = Var(time_steps, within=NonNegativeReals)

    def define_constraints(self, time_steps):
        @self.b.Constraint(time_steps)
        def storage_min_capacity_dri_constraint(b, t):
            return b.soc_dri[t] >= b.min_capacity_dri

        @self.b.Constraint(time_steps)
        def storage_max_capacity_dri_constraint(b, t):
            return b.soc_dri[t] <= b.max_capacity_dri
        
        @self.b.Constraint(time_steps)
        def energy_in_max_capacity_dri_constraint(b, t):
            return b.charge_dri[t] <= b.max_capacity_dri #* b.uniformity_indicator[t]

        @self.b.Constraint(time_steps)
        def energy_out_max_capacity_dri_constraint(b, t):
            return  b.discharge_dri[t] <= b.max_capacity_dri #* (1 - b.uniformity_indicator[t])

        @self.b.Constraint(time_steps)
        def energy_in_uniformity_dri_constraint(b, t):
            return b.charge_dri[t] <= b.max_capacity_dri * b.uniformity_indicator_dri[t]

        @self.b.Constraint(time_steps)
        def energy_out_uniformity_dri_constraint(b, t):
            return  b.discharge_dri[t] <= b.max_capacity_dri * (1 - b.uniformity_indicator_dri[t])

        @self.b.Constraint(self.model.time_steps)
        def storage_capacity_change_dri_constraint(b, t):
            return b.soc_dri[t] == (
                ((b.soc_dri[t - 1] if t > 0 else b.initial_soc_dri) * (1 - b.storage_loss_rate_dri)) +
                ((1 - b.charge_loss_rate_dri) * b.charge_dri[t]) -
                ((1 + b.discharge_loss_rate_dri) * b.discharge_dri[t])
            )

  
    