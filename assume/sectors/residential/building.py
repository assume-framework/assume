from assume.units.heatpump import HeatPump
from assume.strategies.dr_strategies.dr_programmes import RealTimeTariffDR
import pyomo.environ as pyo
from pyomo.environ import *
from enum import Enum
import numpy as np
from numpy import ndarray


class Building:
    def __init__(
        self,
        building,
        heat_pump_model,
        load_profile,
        electricity_prices,
        demand_response_program
    ):
        super().__init__(sector=building)
        self.heat_pump_model = heat_pump_model
        self.load_profile = load_profile
        self.electricity_prices = electricity_prices
        self.demand_response_program = demand_response_program

        # Parameters
        self.model.load_profile = Param(self.model.time_steps)
        self.model.electricity_price = Param(self.model.time_steps)
        self.model.real_time_price = Param(self.model.time_steps)

    def enable_demand_response_program(self, demand_response_program):
        # Set the demand response program for the building
        self.demand_response_program = demand_response_program

    def custom_objective_rule(self, model):
        # Define your custom objective function here
        self.model.obj = Objective(expr=(sum(model.real_time_price[t] * (
            sum(model.power_hp[i, t] for i in model.heat_pumps)) for t in model.time_steps)), sense=minimize)

    def optimize(self, demand_response_program=None):
        # Set the load profile for the heat pump
        self.heat_pump_model.set_heat_pump_load_profile(model=self.model,load_profile=self.load_profile)

        # Add heat pump constraints to the optimization model
        self.heat_pump_model.add_heat_pump_constraints()

        # Add demand response program constraints to the optimization model (if applicable)
        if self.demand_response_program:
            # If demand response program is enabled, add its constraints and update objective function
            self.demand_response_program.add_real_time_tariff_constraints(self.heat_pump_model)
            self.demand_response_program.update_objective_function(self.heat_pump_model)

        # Perform optimization
        optimal_consumption = self.heat_pump_model.optimize()

        return optimal_consumption
