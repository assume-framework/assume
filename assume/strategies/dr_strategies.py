import pyomo.environ as pyo
from pyomo.environ import *


class RealTimeTariffDR:
    def __init__(self, real_time_prices):
        self.real_time_prices = real_time_prices

    def add_real_time_tariffs(self, model):
        model.model.real_time_price = Param(
            model.model.time_steps, initialize=self.real_time_prices
        )

    def update_objective_function(self, model):
        def _objective_rule(m):
            return sum(
                m.real_time_price[t] * (sum(m.power_hp[i, t] for i in m.heat_pumps))
                for t in m.time_steps
            )
            # +
            # sum(m.power_hs[i, t] for i in m.heat_storages) +
            # sum(m.power_ev[i, t] for i in m.electric_vehicles) +
            # sum(m.power_el[i, t] for i in m.electrolysers))
            # for t in m.time_steps)

        model.model.obj = pyo.Objective(rule=_objective_rule)

    def enable_demand_response_program(self, model):
        # Add constraints and update objective function to enable real-time tariff-based demand response
        self.add_real_time_tariffs(model)
        self.update_objective_function(model)
