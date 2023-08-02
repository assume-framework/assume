import logging
import pyomo.environ as pyo
from pyomo.environ import *
import pandas as pd
from enum import Enum
import numpy as np
from numpy import ndarray
from typing import List
from abc import ABC, abstractmethod

from assume.units.base_unit import BaseUnit
logger = logging.getLogger(__name__)


class DST(ABC, BaseUnit):
    def __init__(
            self,
            id: str,
            unit_operator: str,
            technology: str,
            bidding_strategies: dict,
            location: tuple[float, float] = (0.0, 0.0),
            node: str = "bus0",
            index: pd.DatetimeIndex = None,
            time_steps: pd.DatetimeIndex = None,
            **kwargs,
    ):
        super().__init__(
            id=id,
            technology=technology,
            unit_operator=unit_operator,
            node=node,
            bidding_strategies=bidding_strategies,
            index=index,
        )

    def create_model(self, model) -> None:
        self.model = model
        self.define_sets()
        self.define_parameters()
        self.define_variables()
        self.define_constraints()
        self.define_objective()

    def define_sets(self) -> None:
        pass

    def define_parameters(self) -> None:
        pass

    def define_variables(self) -> None:
        pass

    def define_constraints(self) -> None:
        pass

    def define_objective(self) -> None:
        pass


class HeatPump(DST):
    def __init__(
        self,
        id: str,
        unit_operator: str,
        technology: str,
        sector,
        bidding_strategies: dict,
        #max_power: float or pd.Series,
        #min_power: float or pd.Series,
        #ramp_up_rate: float or pd.Series,
        #ramp_down_rate: float or pd.Series,
        #heat_pump_type: str,
        #source_temp: float or pd.Series = 15,
        #sink_temp: float or pd.Series = 35,
        #real_time_prices: float or pd.Series = 35,
        #electricity_prices: float or pd.Series = 40,
        index: pd.DatetimeIndex = None,
        time_steps: pd.DatetimeIndex = None,
        location: tuple[float, float] = (0.0, 0.0),
        node: str = "bus0",
        **kwargs,
    ):
        super().__init__(
            id=id,
            technology=technology,
            unit_operator=unit_operator,
            node=node,
            bidding_strategies=bidding_strategies,
            index=index,
        )
        self.model = ConcreteModel()
        self.sector = sector

    def add_model_parameters(
            self,
            # ramp_up_rate: float or pd.Series = None,
            # ramp_down_rate: float or pd.Series = None,
            time_steps: pd.DatetimeIndex = None,
            heat_pump: List[int] = list(),
            max_power: float or pd.Series = None,
            min_power: float or pd.Series = None,
            heat_pump_type: str = None,
            source_temp: float or pd.Series = None,
            sink_temp: float or pd.Series = None,

            real_time_prices: float or pd.Series = None,
            electricity_prices: float or pd.Series = None,
            load_profile: float or pd.Series = None,
    ):
        self.time_steps = time_steps
        self.heat_pump = heat_pump
        self.max_power = max_power
        self.min_power = min_power
        #self.ramp_up_rate = ramp_up_rate
        #self.ramp_down_rate = ramp_down_rate
        self.heat_pump_type = heat_pump_type
        self.source_temp = source_temp
        self.sink_temp = sink_temp
        self.electricity_prices = electricity_prices
        self.real_time_prices = real_time_prices
        self.load_profile = load_profile
        return self

    # Define sets
    def define_sets(self):
        # self.model.time_steps = Set(initialize={idx: v for idx, v in enumerate(self.time_steps)})
        # self.model.time_steps = Set(initialize=self.time_steps)
        self.model.time_steps = Set(initialize=[idx for idx, _ in enumerate(self.time_steps)])
        # self.model.heat_pumps = Set(within=pyo.NonNegativeIntegers)
        self.model.heat_pumps = Set(ordered=True, initialize=self.heat_pump)

    # Define parameters
    def define_parameters(self) -> None:
        self.model.max_power = Param(initialize=self.max_power)
        self.model.min_power = Param(initialize=self.min_power)
        # self.model.ramp_up_rate = Param(self.model.heat_pumps, initialize=self.ramp_down_rate)
        # self.model.ramp_down_rate = Param(self.model.heat_pumps, initialize=self.ramp_down_rate)
        self.sink_temp = Param(self.model.time_steps, initialize=self.sink_temp)
        self.source_temp = Param(self.model.time_steps, initialize=self.source_temp)
        # self.model.heat_pump_COP = Param(initialize=se)
        self.model.electricity_prices = Param(self.model.time_steps,
                                              initialize={idx: v for idx, v in enumerate(self.electricity_prices)})
        self.model.load_profile = Param(self.model.time_steps,
                                              initialize={idx: v for idx, v in enumerate(self.load_profile)})
        # self.model.real_time_prices = Param(self.model.time_steps, initialize=self.real_time_prices)

        # self.model.electricity_prices = Param(self.model.time_steps, initialize=self.electricity_prices)
        # self.model.load_profile = Param(self.model.time_steps, initialize=self.load_profile)

    # Define variables
    def define_variables(self) -> None:
        self.model.power_in = Var(self.model.heat_pumps, self.model.time_steps, domain=pyo.NonNegativeReals)
        self.model.power_out = Var(self.model.heat_pumps, self.model.time_steps, domain=pyo.NonNegativeReals)
        # self.model.s_on = Var(self.model.time_steps, domain=pyo.Binary)
        # self.model.s_start = Var(self.model.time_steps, domain=pyo.Binary)
        # self.model.s_stop = Var(self.model.time_steps, domain=pyo.Binary)
        self.model.c = Var(self.model.heat_pumps, self.model.time_steps, domain=pyo.Reals)

    # Define constraints
    def define_constraints(self):

        # @self.model.Constraint(self.model.time_steps, self.model.heat_pumps)
        # def s_on_power_out_relation(m, t, h):
        #     return m.s_on[t, h] == (1 if value(m.power_out[t, h]) > 0 else 0)
        #
        # # Constraint to set s_start[t] to 1 and s_stop[t] to 0 if s_on[t - 1] is 0
        # @self.model.Constraint(self.model.time_steps)
        # def s_start_s_stop_s_on_relation(m, t):
        #     return m.s_start[t] + m.s_stop[t] + (1 - m.s_on[t - 1] if t > 0 else 0) <= 1
        #
        # # Constraints to maintain the relations between s_on, s_start, and s_stop
        # @self.model.Constraint(self.model.time_steps)
        # def s_on_s_start_relation(m, t):
        #     return m.s_on[t] - (m.s_on[t - 1] if t > 0 else 0) == m.s_start[t] - m.s_stop[t]
        #
        # @self.model.Constraint(self.model.time_steps)
        # def s_on_s_start_relation(m, t):
        #     return m.s_on[t] - m.s_on[t - 1] == m.s_start[t] - m.s_stop[t] if t > 0 else Constraint.Skip
        #
        # @self.model.Constraint(self.model.time_steps, self.model.heat_pumps)
        # def s_start_s_stop_relation(m, t):  # pylint: disable=W0612
        #     return m.s_start[t] + m.s_stop[t] <= 1

        @self.model.Constraint(self.model.time_steps, self.model.heat_pumps)
        def p_output_lower_bound(h, t):  # pylint: disable=W0612
            return self.model.power_out[h, t] >= self.model.min_power[h]  # * m.s_on[t]

        @self.model.Constraint(self.model.time_steps, self.model.heat_pumps)
        def p_output_upper_bound(h, t):  # pylint: disable=W0612
            return self.model.power_out[h, t] <= self.model.max_power[h] # * m.s_on[t]

        @self.model.Constraint(self.model.time_steps, self.model.heat_pumps)
        def energy_balance_rule(h, t):
            return self.model.power_out[h, t] >= self.model.load_profile[t] # ==

        @self.model.Constraint(self.model.time_steps, self.model.heat_pumps)
        def p_input_lower_bound(h, t):  # pylint: disable=W0612
            return self.model.power_in[h, t] == self.model.power_out[h, t]  # / 3  COP

        @self.model.Constraint(self.model.time_steps, self.model.heat_pumps)
        def cost_def(h, t):
            return self.model.c[h, t] == self.model.power_out[h, t] * self.model.electricity_prices[t]

        # self.model.ramp_up_constraint = Constraint(self.model.heat_pumps, self.model.time_steps,
        #                                               rule=self.ramp_up_constraint_hp_rule)
        # self.model.ramp_down_constraint_hp = Constraint(self.model.heat_pumps, self.model.time_steps,
        #                                                 rule=self.ramp_down_constraint_hp_rule)
        # self.model.energy_balance_hp = Constraint(self.model.heat_pumps, self.model.time_steps,
        #                                           rule=self.energy_balance_hp_constraint_rule)

    def define_objective(self):
        # objective function
        self.model.obj = Objective(
            expr=sum(self.model.c[t, h] for t in self.model.time_steps for h in self.model.heat_pumps), sense=minimize)

    # def calculate_delta_t(self, timestamp):
    #     # Calculate temperature difference between the source and sink temperatures for the heat pump
    #     source_temp = self.source_temp[timestamp] if isinstance(self.source_temp, pd.Series) else self.source_temp
    #     sink_temp = self.sink_temp[timestamp] if isinstance(self.sink_temp, pd.Series) else self.sink_temp
    #
    #     delta_t = sink_temp - source_temp
    #
    #     return delta_t
    #
    # def calculate_cop(self, timestamp, heat_pump_type):
    #     # Calculate the Coefficient of Performance (COP) of the heat pump based on the temperature difference
    #     source_temp = self.source_temp[timestamp] if isinstance(self.source_temp, pd.Series) else self.source_temp
    #     sink_temp = self.sink_temp[timestamp] if isinstance(self.sink_temp, pd.Series) else self.sink_temp
    #
    #     delta_t = sink_temp - source_temp
    #
    #     if heat_pump_type == "ASHP":
    #         cop = 6.81 + 0.121 * delta_t + 0.000630 * delta_t**2
    #     elif heat_pump_type == "GSHP":
    #         cop = 8.77 + 0.150 * delta_t + 0.000734 * delta_t**2
    #     else:
    #         raise ValueError("Invalid heat pump type. Must be either 'ASHP' or 'GSHP'")
    #
    #     return cop

    # def ramp_up_rule(self, m, t):
    #     # Ramp-up limit constraint for each time step
    #     if t == m.time_steps.first():
    #         return pyo.Constraint.Skip  # Skip constraint for the first time step
    #     else:
    #         max_power_capacity = self.max_power  # Maximum rated capacity of the heat pump
    #         ramp_up_limit = max_power_capacity * (self.ramp_up_rate / 100.0)  # Ramp-up limit as a percentage of max_power
    #         return m.power_hp[self.id, t] - m.power_hp[self.id, t - 1] <= ramp_up_limit
    #
    # def ramp_down_rule(self, m, t):
    #     # Ramp-down limit constraint for each time step
    #     if t == m.time_steps.first():
    #         return pyo.Constraint.Skip  # Skip constraint for the first time step
    #     else:
    #         max_power_capacity = self.max_power  # Maximum rated capacity of the heat pump
    #         ramp_down_limit = max_power_capacity * (self.ramp_down_rate / 100.0)  # Ramp-down limit as a percentage of max_power
    #         return m.power_hp[self.id, t - 1] - m.power_hp[self.id, t] <= ramp_down_limit