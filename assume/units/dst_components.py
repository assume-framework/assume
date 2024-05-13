# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime

import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import *

# Industrial Units


class Electrolyser:
    """
    Represents an electrolyser unit used for hydrogen production through electrolysis.

    Attributes:
        model (pyomo.ConcreteModel): The Pyomo model where the electrolyser unit will be added.
        id (str): Identifier for the electrolyser unit.
        rated_power (float): The rated power capacity of the electrolyser (in kW).
        min_power (float): The minimum power required for operation (in kW).
        ramp_up (float): The maximum rate at which the electrolyser can increase its power output (in kW/hr).
        ramp_down (float): The maximum rate at which the electrolyser can decrease its power output (in kW/hr).
        min_operating_time (float): The minimum duration the electrolyser must operate continuously (in hours).
        min_down_time (float): The minimum downtime required between operating cycles (in hours).
        efficiency (float): The efficiency of the electrolysis process.
        fuel_type (str): The type of fuel used by the electrolyser unit.
    """

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
        fuel_type,
        **kwargs,
    ):
        """
        Initializes the Electrolyser object with the provided parameters.

        Args:
            model (pyomo.ConcreteModel): The Pyomo model where the electrolyser unit will be added.
            id (str): Identifier for the electrolyser unit.
            rated_power (float): The rated power capacity of the electrolyser (in kW).
            min_power (float): The minimum power required for operation (in kW).
            ramp_up (float): The maximum rate at which the electrolyser can increase its power output (in kW/hr).
            ramp_down (float): The maximum rate at which the electrolyser can decrease its power output (in kW/hr).
            min_operating_time (float): The minimum duration the electrolyser must operate continuously (in hours).
            min_down_time (float): The minimum downtime required between operating cycles (in hours).
            efficiency (float): The efficiency of the electrolysis process.
            fuel_type (str): The type of fuel used by the electrolyser unit.
            **kwargs: Additional keyword arguments.
        """
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

    def add_to_model(self, unit_block, time_steps):
        """
        Adds the electrolyser unit to the Pyomo model.

        Args:
            unit_block (pyomo.Block): The Pyomo block where the electrolyser will be added.
            time_steps (list): List of time steps for which the model will be defined.
        """
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

    def define_variables(self, time_steps):
        self.b.power_in = Var(time_steps, within=pyo.NonNegativeReals)
        self.b.hydrogen_out = Var(time_steps, within=pyo.NonNegativeReals)
        self.b.electricity_cost = Var(time_steps, within=pyo.NonNegativeReals)

    def define_constraints(self, time_steps):
        # Power bounds constraints
        @self.b.Constraint(time_steps)
        def power_upper_bound(b, t):
            """
            Ensures that the power input to the electrolyser does not exceed its rated power capacity.

            """
            return b.power_in[t] <= b.rated_power

        @self.b.Constraint(time_steps)
        def power_lower_bound(b, t):
            """
            Ensures that the power input to the electrolyser does not fall below the minimum required power.

            """
            return b.power_in[t] >= b.min_power

        # Ramp-up constraint
        @self.b.Constraint(time_steps)
        def ramp_up_constraint(b, t):
            """
            Limits the rate at which the power input to the electrolyser can increase.

            """
            if t == 0:
                return Constraint.Skip
            return b.power_in[t] - b.power_in[t - 1] <= b.ramp_up

        # Ramp-down constraint
        @self.b.Constraint(time_steps)
        def ramp_down_constraint(b, t):
            """
            Limits the rate at which the power input to the electrolyser can decrease.

            """
            if t == 0:
                return Constraint.Skip
            return b.power_in[t - 1] - b.power_in[t] <= b.ramp_down

        @self.b.Constraint(time_steps)
        def min_operating_time_electrolyser_constraint(b, t):
            """
            Ensures that the electrolyser operates continuously for a minimum duration.

            """
            if t == 0:
                return pyo.Constraint.Skip

            # Calculate the time step duration dynamically
            delta_t = t - (t - 1)
            # Convert minimum operating time to the time unit of your choice
            min_operating_time_units = int(self.min_operating_time / delta_t)

            if t < min_operating_time_units:
                # Ensure that the cumulative sum of DRI production over the past min_operating_time_units time steps is at least min_operating_time_units times the production at time step t
                return (
                    sum(
                        b.power_in[i]
                        for i in range(t - min_operating_time_units + 1, t + 1)
                    )
                    >= min_operating_time_units * b.power_in[t]
                )
            else:
                return (
                    sum(
                        b.power_in[i]
                        for i in range(t - min_operating_time_units + 1, t + 1)
                    )
                    >= min_operating_time_units * b.power_in[t]
                )

        @self.b.Constraint(time_steps)
        def min_downtime_electrolyser_constraint(b, t):
            """
            Ensures that the electrolyser has a minimum downtime between operating cycles.

            """
            if t == 0:
                return pyo.Constraint.Skip  # No constraint for the first time step

            # Calculate the time step duration dynamically
            delta_t = t - (t - 1)
            # Convert minimum downtime to the time unit of your choice
            min_downtime_units = int(self.min_down_time / delta_t)

            if t < min_downtime_units:
                # Ensure that the cumulative sum of DRI production over the past min_downtime_units time steps is at least min_downtime_units times the production at time step t
                return (
                    sum(b.power_in[t - i] for i in range(min_downtime_units))
                    >= min_downtime_units * b.power_in[t]
                )
            else:
                return (
                    sum(b.hydrogen_out[t - i] for i in range(min_downtime_units))
                    >= min_downtime_units * b.power_in[t]
                )

        # Efficiency constraint
        @self.b.Constraint(time_steps)
        def power_in_equation(b, t):
            """
            Relates the power input to the hydrogen output based on the efficiency of the electrolysis process.

            """
            return b.power_in[t] == b.hydrogen_out[t] / b.efficiency

        @self.b.Constraint(time_steps)
        def operating_cost_with_el_price(b, t):
            """
            Calculates the operating cost of the electrolyser based on the electricity price.

            """
            return (
                b.electricity_cost[t] == b.power_in[t] * self.model.electricity_price[t]
            )


class DriPlant:
    def __init__(
        self,
        model,
        id,
        specific_hydrogen_consumption,
        specific_natural_gas_consumption,
        specific_electricity_consumption,
        specific_iron_ore_consumption,
        rated_power,
        min_power,
        fuel_type,
        ramp_up,
        ramp_down,
        min_operating_time,
        min_down_time,
        **kwargs,
    ):
        self.model = model
        self.id = id
        self.fuel_type = fuel_type
        self.min_power = min_power
        self.rated_power = rated_power
        self.specific_hydrogen_consumption = specific_hydrogen_consumption
        self.specific_natural_gas_consumption = specific_natural_gas_consumption
        self.specific_electricity_consumption = specific_electricity_consumption
        self.specific_iron_ore_consumption = specific_iron_ore_consumption
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
        self.b.specific_hydrogen_consumption = Param(
            initialize=self.specific_hydrogen_consumption
        )
        self.b.specific_natural_gas_consumption = Param(
            initialize=self.specific_natural_gas_consumption
        )
        self.b.specific_electricity_consumption_dri = Param(
            initialize=self.specific_electricity_consumption
        )
        self.b.specific_iron_ore_consumption = Param(
            initialize=self.specific_iron_ore_consumption
        )
        # Flexibility parameters
        self.b.min_power_dri = Param(initialize=self.min_power)
        self.b.rated_power_dri = Param(initialize=self.rated_power)
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
        self.b.power_dri = Var(time_steps, within=NonNegativeReals)

    def define_constraints(self, time_steps):
        @self.b.Constraint(time_steps)
        def dri_power_lower_bound(b, t):
            return b.power_dri[t] >= b.min_power_dri

        @self.b.Constraint(time_steps)
        def dri_power_upper_bound(b, t):
            return b.power_dri[t] <= b.rated_power_dri

        @self.b.Constraint(time_steps)
        def dri_output_constraint(b, t):
            if self.fuel_type == "hydrogen":
                return (
                    b.dri_output[t]
                    == b.hydrogen_in[t] / b.specific_hydrogen_consumption
                )
            elif self.fuel_type == "natural_gas":
                return (
                    b.dri_output[t]
                    == b.natural_gas_in[t] / b.specific_natural_gas_consumption
                )
            elif self.fuel_type == "both":
                return b.dri_output[t] == (
                    b.hydrogen_in[t] / b.specific_hydrogen_consumption
                ) + (b.natural_gas_in[t] / b.specific_natural_gas_consumption)

        @self.b.Constraint(time_steps)
        def dri_output_electricity_constraint(b, t):
            return (
                b.power_dri[t]
                == b.dri_output[t] * b.specific_electricity_consumption_dri
            )

        @self.b.Constraint(time_steps)
        def iron_ore_constraint(b, t):
            return b.iron_ore_in[t] == b.dri_output[t] * b.specific_iron_ore_consumption

        # Flexibility constraints
        @self.b.Constraint(time_steps)
        def ramp_up_dri_constraint(b, t):
            if t == 0:
                return Constraint.Skip
            else:
                return b.power_dri[t] - b.power_dri[t - 1] <= b.ramp_up_dri

        @self.b.Constraint(time_steps)
        def ramp_down_dri_constraint(b, t):
            if t == 0:
                return Constraint.Skip
            else:
                return b.power_dri[t - 1] - b.power_dri[t] <= b.ramp_down_dri

        @self.b.Constraint(time_steps)
        def min_operating_time_dri__constraint(b, t):
            if t == 0:
                return pyo.Constraint.Skip

            # Calculate the time step duration dynamically
            delta_t = t - (t - 1)
            # Convert minimum operating time to the time unit of your choice
            min_operating_time_units = int(self.min_operating_time / delta_t)

            if t < min_operating_time_units:
                # Ensure that the cumulative sum of DRI production over the past min_operating_time_units time steps is at least min_operating_time_units times the production at time step t
                return (
                    sum(
                        b.power_dri[i]
                        for i in range(t - min_operating_time_units + 1, t + 1)
                    )
                    >= min_operating_time_units * b.power_dri[t]
                )
            else:
                return (
                    sum(
                        b.power_dri[i]
                        for i in range(t - min_operating_time_units + 1, t + 1)
                    )
                    >= min_operating_time_units * b.power_dri[t]
                )

        @self.b.Constraint(time_steps)
        def min_down_time_dri_constraint(b, t):
            if t == 0:
                return pyo.Constraint.Skip  # No constraint for the first time step

            # Calculate the time step duration dynamically
            delta_t = t - (t - 1)
            # Convert minimum downtime to the time unit of your choice
            min_downtime_units = int(self.min_down_time / delta_t)

            if t < min_downtime_units:
                # Ensure that the cumulative sum of DRI production over the past min_downtime_units time steps is at least min_downtime_units times the production at time step t
                return (
                    sum(b.power_dri[t - i] for i in range(min_downtime_units))
                    >= min_downtime_units * b.power_dri[t]
                )
            else:
                return (
                    sum(b.dri_output[t - i] for i in range(min_downtime_units))
                    >= min_downtime_units * b.power_dri[t]
                )

        # Operational cost
        @self.b.Constraint(time_steps)
        def dri_operating_cost_constraint(b, t):
            # This constraint defines the steel output based on inputs and efficiency
            return (
                b.dri_operating_cost[t]
                == b.natural_gas_in[t] * self.model.natural_gas_price[t]
                + b.power_dri[t] * self.model.electricity_price[t]
                + b.iron_ore_in[t] * self.model.iron_ore_price
            )


class ElectricArcFurnace:
    """
    Represents an Electric Arc Furnace (EAF) in a steel production process.

    Parameters:
    - model: A Pyomo ConcreteModel object representing the optimization model.
    - id: A unique identifier for the ElectricArcFurnace instance.
    - rated_power: The rated power capacity of the electric arc furnace (in MW).
    - min_power: The minimum power requirement of the electric arc furnace (in MW).
    - specific_electricity_consumption: The specific electricity consumption of the electric arc furnace (in kWh per ton of steel produced).
    - specific_dri_demand: The specific demand for Direct Reduced Iron (DRI) in the electric arc furnace (in tons per ton of steel produced).
    - specific_lime_demand: The specific demand for lime in the electric arc furnace (in tons per ton of steel produced).
    - ramp_up: The ramp-up rate of the electric arc furnace (in MW per hour).
    - ramp_down: The ramp-down rate of the electric arc furnace (in MW per hour).
    - min_operating_time: The minimum operating time requirement for the electric arc furnace (in hours).
    - min_down_time: The minimum downtime requirement for the electric arc furnace (in hours).

    Methods:
    - add_to_model(unit_block, time_steps): Adds the ElectricArcFurnace instance to the optimization model.
    - define_parameters(): Defines the parameters for the ElectricArcFurnace instance.
    - define_variables(time_steps): Defines the decision variables for the ElectricArcFurnace instance.
    - define_constraints(time_steps): Defines the constraints for the ElectricArcFurnace instance.
    """

    def __init__(
        self,
        model,
        id,
        rated_power,
        min_power,
        specific_electricity_consumption,
        specific_dri_demand,
        specific_lime_demand,
        ramp_up,
        ramp_down,
        min_operating_time,
        min_down_time,
        **kwargs,
    ):
        self.model = model
        self.id = id
        # Operational parameters
        self.rated_power = rated_power
        self.min_power = min_power
        # Additional operational characteristics
        self.specific_electricity_consumption = specific_electricity_consumption
        self.specific_dri_demand = specific_dri_demand
        self.specific_lime_demand = specific_lime_demand
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
        self.b.specific_electricity_consumption_eaf = Param(
            initialize=self.specific_electricity_consumption
        )
        self.b.specific_dri_demand = Param(initialize=self.specific_dri_demand)
        self.b.specific_lime_demand = Param(initialize=self.specific_lime_demand)
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
        self.b.emission_eaf = Var(time_steps, within=NonNegativeReals)
        self.b.lime_demand = Var(time_steps, within=NonNegativeReals)

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
            return b.steel_output[t] == b.dri_input[t] / b.specific_dri_demand

        @self.b.Constraint(time_steps)
        def steel_output_power_relation(b, t):
            # This constraint defines the steel output based on inputs and efficiency
            return (
                b.power_eaf[t]
                == b.steel_output[t] * b.specific_electricity_consumption_eaf
            )

        @self.b.Constraint(time_steps)
        def eaf_lime_demand(b, t):
            return b.lime_demand[t] == b.steel_output[t] * b.specific_lime_demand

        @self.b.Constraint(time_steps)
        def eaf_co2_emission(b, t):
            return b.emission_eaf[t] == b.lime_demand[t] * self.model.lime_co2_factor

        # Flexibility constraints
        @self.b.Constraint(time_steps)
        def ramp_up_eaf_constraint(b, t):
            if t == 0:
                return pyo.Constraint.Skip
            return b.power_eaf[t] - b.power_eaf[t - 1] <= b.ramp_up_eaf

        @self.b.Constraint(time_steps)
        def ramp_down_eaf_constraint(b, t):
            if t == 0:
                return pyo.Constraint.Skip
            return b.power_eaf[t - 1] - b.power_eaf[t] <= b.ramp_down_eaf

        @self.b.Constraint(time_steps)
        def min_operating_time_eaf_constraint(b, t):
            if t == 0:
                return pyo.Constraint.Skip  # No constraint for the first time step
            # Calculate the time step duration dynamically
            delta_t = t - (t - 1)
            # Convert minimum operating time to the time unit of your choice
            min_operating_time_units = int(self.min_operating_time / delta_t)

            if t < min_operating_time_units:
                # Ensure that the cumulative sum of DRI production over the past min_operating_time_units time steps is at least min_operating_time_units times the production at time step t
                return (
                    sum(
                        b.steed_output[i]
                        for i in range(t - min_operating_time_units + 1, t + 1)
                    )
                    >= min_operating_time_units * b.steel_output[t]
                )
            else:
                return (
                    sum(
                        b.steel_output[i]
                        for i in range(t - min_operating_time_units + 1, t + 1)
                    )
                    >= min_operating_time_units * b.steel_output[t]
                )

        @self.b.Constraint(time_steps)
        def min_down_time_eaf_constraint(b, t):
            if t == 0:
                return pyo.Constraint.Skip  # No constraint for the first time step
            # Calculate the time step duration dynamically
            delta_t = t - (t - 1)
            # Convert minimum downtime to the time unit of your choice
            min_downtime_units = int(self.min_down_time / delta_t)

            if t < min_downtime_units:
                return (
                    sum(b.steel_output[t - i] for i in range(min_downtime_units))
                    >= min_downtime_units * b.steel_output[t]
                )
            else:
                return (
                    sum(b.steel_output[t - i] for i in range(min_downtime_units))
                    >= min_downtime_units * b.steel_output[t]
                )

        # operational cost
        @self.b.Constraint(time_steps)
        def eaf_operating_cost_cosntraint(b, t):
            # This constraint defines the steel output based on inputs and efficiency
            return (
                b.eaf_operating_cost[t]
                == b.power_eaf[t] * self.model.electricity_price[t]
                + b.emission_eaf[t] * self.model.co2_price
                + b.lime_demand[t] * self.model.lime_price
            )


class GenericStorage:
    """
    Represents a generic energy storage unit.

    Attributes:
        model: A Pyomo ConcreteModel object representing the optimization model.
        id (str): A unique identifier for the storage unit.
        max_capacity (float): The maximum storage capacity of the unit.
        min_capacity (float): The minimum storage capacity of the unit.
        initial_soc (float): The initial state of charge (SOC) of the storage unit.
        storage_loss_rate (float): The rate of energy loss due to storage inefficiency.
        charge_loss_rate (float): The rate of energy loss during charging.
        discharge_loss_rate (float): The rate of energy loss during discharging.

    Methods:
        add_to_model(unit_block, time_steps):
            Adds the storage unit to the optimization model.
        define_parameters():
            Defines the parameters of the storage unit in the optimization model.
        define_variables(time_steps):
            Defines the variables of the storage unit in the optimization model.
        define_constraints(time_steps):
            Defines the constraints of the storage unit in the optimization model.
    """

    def __init__(
        self,
        model,
        id,
        max_capacity,
        min_capacity,
        initial_soc,
        storage_loss_rate,
        charge_loss_rate,
        discharge_loss_rate,
        **kwargs,
    ):
        """
        Initializes a GenericStorage object with the specified parameters.

        Args:
            model: A Pyomo ConcreteModel object representing the optimization model.
            id (str): A unique identifier for the storage unit.
            max_capacity (float): The maximum storage capacity of the unit.
            min_capacity (float): The minimum storage capacity of the unit.
            initial_soc (float): The initial state of charge (SOC) of the storage unit.
            storage_loss_rate (float): The rate of energy loss due to storage inefficiency.
            charge_loss_rate (float): The rate of energy loss during charging.
            discharge_loss_rate (float): The rate of energy loss during discharging.
            **kwargs: Additional keyword arguments.
        """
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
            return b.charge[t] <= b.max_capacity

        @self.b.Constraint(time_steps)
        def energy_out_max_capacity_constraint(b, t):
            return b.discharge[t] <= b.max_capacity

        @self.b.Constraint(time_steps)
        def energy_in_uniformity_constraint(b, t):
            return b.charge[t] <= b.max_capacity * b.uniformity_indicator[t]

        @self.b.Constraint(time_steps)
        def energy_out_uniformity_constraint(b, t):
            return b.discharge[t] <= b.max_capacity * (1 - b.uniformity_indicator[t])

        @self.b.Constraint(time_steps)
        def storage_capacity_change_constraint(b, t):
            return b.soc[t] == (
                ((b.soc[t - 1] if t > 0 else b.initial_soc) * (1 - b.storage_loss_rate))
                + ((1 - b.charge_loss_rate) * b.charge[t])
                - ((1 + b.discharge_loss_rate) * b.discharge[t])
            )


class DRIStorage:
    """
    Represents a Direct Reduced Iron (DRI) storage unit.

    Parameters:
    - model: A Pyomo ConcreteModel object representing the optimization model.
    - id: A unique identifier for the DRI storage unit.
    - max_capacity: The maximum capacity of the DRI storage unit.
    - min_capacity: The minimum capacity of the DRI storage unit.
    - initial_soc: The initial state of charge (SOC) of the DRI storage unit.
    - storage_loss_rate: The rate of DRI loss due to storage over time.
    - charge_loss_rate: The rate of DRI loss during charging.
    - discharge_loss_rate: The rate of DRI loss during discharging.

    Constraints:
    - storage_min_capacity_dri_constraint: Ensures the SOC of the DRI storage unit stays above the minimum capacity.
    - storage_max_capacity_dri_constraint: Ensures the SOC of the DRI storage unit stays below the maximum capacity.
    - energy_in_max_capacity_dri_constraint: Limits the charging of the DRI storage unit to its maximum capacity.
    - energy_out_max_capacity_dri_constraint: Limits the discharging of the DRI storage unit to its maximum capacity.
    - energy_in_uniformity_dri_constraint: Ensures uniformity in charging the DRI storage unit.
    - energy_out_uniformity_dri_constraint: Ensures uniformity in discharging the DRI storage unit.
    - storage_capacity_change_dri_constraint: Defines the change in SOC of the DRI storage unit over time.
    """

    def __init__(
        self,
        model,
        id,
        max_capacity,
        min_capacity,
        initial_soc,
        storage_loss_rate,
        charge_loss_rate,
        discharge_loss_rate,
        **kwargs,
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
            return b.charge_dri[t] <= b.max_capacity_dri

        @self.b.Constraint(time_steps)
        def energy_out_max_capacity_dri_constraint(b, t):
            return b.discharge_dri[t] <= b.max_capacity_dri

        @self.b.Constraint(time_steps)
        def energy_in_uniformity_dri_constraint(b, t):
            return b.charge_dri[t] <= b.max_capacity_dri * b.uniformity_indicator_dri[t]

        @self.b.Constraint(time_steps)
        def energy_out_uniformity_dri_constraint(b, t):
            return b.discharge_dri[t] <= b.max_capacity_dri * (
                1 - b.uniformity_indicator_dri[t]
            )

        @self.b.Constraint(time_steps)
        def storage_capacity_change_dri_constraint(b, t):
            return b.soc_dri[t] == (
                (
                    (b.soc_dri[t - 1] if t > 0 else b.initial_soc_dri)
                    * (1 - b.storage_loss_rate_dri)
                )
                + ((1 - b.charge_loss_rate_dri) * b.charge_dri[t])
                - ((1 + b.discharge_loss_rate_dri) * b.discharge_dri[t])
            )
