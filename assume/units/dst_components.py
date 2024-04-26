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
            return b.power_in[t] <= b.rated_power * b.in_operation[t]

        @self.b.Constraint(time_steps)
        def power_lower_bound(b, t):
            """
            Ensures that the power input to the electrolyser does not fall below the minimum required power.

            """
            return b.power_in[t] >= b.min_power * b.in_operation[t]

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
