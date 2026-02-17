# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import numpy as np

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, check_available_solvers
from pyomo.opt.base.solvers import SolverFactoryClass

from assume.common.base import MinMaxChargeStrategy, SupportsMinMaxCharge
from assume.common.market_objects import MarketConfig, Order, Orderbook, Product

from assume.common.utils import parse_duration


class StorageEnergyOptimizationStrategy(MinMaxChargeStrategy):
    '''
    Storage energy optimization strategy for multiple markets
    - currently only EOM
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ths strategy must know the markets it should bid into
        # how?
        # self.markets = markets
        # self.model = pyo.ConcreteModel('storage')
        self.foresight = parse_duration(kwargs.get("eom_foresight", "12h"))
        # self.foresight = parse_duration('24h') # h # could be stated either in config or units.csv
        # timestep in datetime
        self.time_step = parse_duration('1h')  # h # should be related to market time resolution
        # Create solver instance once and reuse it to prevent file descriptor leaks
        self.solver_instance = SolverFactory('gurobi')

    def calculate_bids(self, unit, market_config, product_tuples, **kwargs):
        if market_config.market_id == "CRM_pos":
            raise NotImplementedError("CRM_pos not implemented for storage EOM strategy")
        elif market_config.market_id == "CRM_neg":
            raise NotImplementedError("CRM_neg not implemented for storage EOM strategy")
        elif market_config.market_id == "EOM":
            bids = self.calculate_EOM_bids(
                unit, market_config, product_tuples, **kwargs
            )
        return bids
    
    def optimize(self, unit, market_config, product_tuples, **kwargs):
        '''
        Optimize storage operation over foresight horizon
        '''
        # if foresight horizon is longer than available products, adjust foresight
        # no actually this should not be done
        # but make sure everything works at end of simulation
        # if len(product_tuples) * self.time_step < self.foresight:
        #     self.foresight = len(product_tuples) * self.time_step

        # Define model
        model = pyo.ConcreteModel('storage')
        time_indices = pd.date_range(product_tuples[0][0], product_tuples[0][0] + self.foresight, freq=self.time_step)
        # model.T = pyo.RangeSet(0, time_indices)
        model.T = pyo.Set(initialize=time_indices,
                          doc="timesteps")
        
        # Variables
        model.power_charge = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, -1*unit.max_power_charge))
        model.power_discharge = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, unit.max_power_discharge))
        model.soc = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(unit.min_soc, unit.max_soc))
        #model.bid_price_charge = pyo.Var(model.T, domain=pyo.Reals, bounds=(market_config.min_price, market_config.max_price))
        #model.bid_price_discharge = pyo.Var(model.T, domain=pyo.Reals, bounds=(market_config.min_price, market_config.max_price))
        model.hourly_profit = pyo.Var(model.T, domain=pyo.Reals)

        # Price forecast
        #price_t = [unit.forecaster.price['EOM'].at[i] for i in time_indices]
        price_t = unit.forecaster.price['EOM'].as_pd_series().loc[time_indices]
        msv_t = unit.forecaster.price['EOM_msv'].as_pd_series().loc[time_indices]
        #model.price = pyo.Param(model.T, initialize=lambda m, t: price_t[t])

        # Objective
        def hourly_profit_rule(model, t):
            expr = model.hourly_profit[t] == ((model.power_discharge[t] - model.power_charge[t]) * price_t[t] * 1 #self.time_step
                                              #- unit.variable_cost_discharge * model.power_discharge[t]
                                              #- unit.variable_cost_charge * model.power_charge[t]
                                              )
            return expr
        
        model.hourly_profit_rule = pyo.Constraint(model.T, rule=hourly_profit_rule)

        def objective_rule(model):
            return pyo.quicksum(model.hourly_profit[t] for t in model.T)
        
        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

        # Constraints
        time_indices_list = list(time_indices)
        def soc_rule(m, t):
            t_idx = time_indices_list.index(t)
            if t_idx == 0: # first timestep
                return m.soc[t] == unit.initial_soc + ((model.power_charge[t] * unit.efficiency_charge
                                                        - model.power_discharge[t] / unit.efficiency_discharge)
                                                         * 1 # self.time_step
                                                         / unit.capacity
                )
            else:
                t_prev = time_indices_list[t_idx - 1]
                return m.soc[t] == m.soc[t_prev] + ((model.power_charge[t] * unit.efficiency_charge
                                                     - model.power_discharge[t] / unit.efficiency_discharge)
                                                       * 1 # self.time_step
                                                      / unit.capacity
                )
        model.soc_rule = pyo.Constraint(model.T, rule=soc_rule)
                
        # Solve model
        results = self.solver_instance.solve(model, tee=False)
        # pyo.check_termination_condition(results)
        # could be checked with
        # results.solver.termination_condition == pyo.TerminationCondition.optimal

        power_discharge = [model.power_discharge[t].value for t in model.T]
        power_charge = [model.power_charge[t].value for t in model.T]
        #price_discharge = [model.bid_price_discharge[t] for t in model.T]
        #price_charge = [model.bid_price_charge[t] for t in model.T]
        #price_discharge = [market_config.minimum_bid_price for t in model.T]
        #price_charge = [market_config.maximum_bid_price for t in model.T]
        #price_discharge = [msv_t[t] for t in model.T]
        #price_charge = [msv_t[t] for t in model.T]
        price_discharge = [price_t[t] - 0.1 for t in model.T]
        price_charge = [price_t[t] + 0.1 for t in model.T]

        # close the model
        del model
        return power_discharge, power_charge, price_discharge, price_charge
    
    def calculate_EOM_bids(self,
                           unit,
                           market_config,
                           product_tuples,
                           **kwargs,
                           ) -> Orderbook:
        '''
        Calculate bids for energy-only market using storage energy optimization
        '''
        start_all = product_tuples[0][0]
        end_all = product_tuples[-1][1]
        power_discharge, power_charge, price_discharge, price_charge = self.optimize(
            unit, market_config, product_tuples, **kwargs
        )

        bids = []
        for i, product in enumerate(product_tuples):
            start = product[0]
            end = product[1]
            bid_discharge = None
            bid_charge = None
            if power_discharge[i] > 0:
                bid_discharge = price_discharge[i]
                order: Order = {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": product_tuples[i][2],
                    "price": bid_discharge,
                    "volume": power_discharge[i],
                    "bid_type": "SB",
                    "node": unit.node,
                }
                bids.append(order)
            elif power_charge[i] > 0:
                bid_charge = price_charge[i]
                order: Order = {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": product_tuples[i][2],
                    "price": bid_charge,
                    "volume": -power_charge[i],
                    "bid_type": "SB",
                    "node": unit.node,
                }
                bids.append(order)
            else:
                continue
            unit.outputs['eom_discharge_bids'].loc[start] = bid_discharge
            unit.outputs['eom_charge_bids'].loc[start] = bid_charge
        return bids