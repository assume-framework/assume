#!/usr/bin/env python3
"""Manually create and solve a minimal depot model to find infeasibility"""

import pyomo.environ as pyo

# Minimal test: PV + EV + CS without battery
model = pyo.ConcreteModel()

# Time steps
model.time_steps = pyo.Set(initialize=[0, 1, 2])

# Variables
model.grid_power = pyo.Var(model.time_steps, within=pyo.NonNegativeReals)
model.grid_feedin = pyo.Var(model.time_steps, within=pyo.NonNegativeReals)  # Prosumer
model.pv_used = pyo.Var(model.time_steps, within=pyo.NonNegativeReals)
model.pv_curtail = pyo.Var(model.time_steps, within=pyo.NonNegativeReals)
model.cs_discharge = pyo.Var(model.time_steps, within=pyo.NonNegativeReals)  # To EV
model.cs_charge = pyo.Var(model.time_steps, within=pyo.NonNegativeReals)     # From EV (V2G)

# Parameters
pv_gen = {0: 0.05, 1: 0.05, 2: 0.05}  # 50W PV
cs_demand = {0: 0.02, 1: 0.02, 2: 0.02}  # 20W EV charging need

model.pv_generation = pyo.Param(model.time_steps, initialize=pv_gen)

# Constraints
@model.Constraint(model.time_steps)
def pv_balance_constraint(m, t):
    """PV generation = PV used + PV curtailed"""
    return m.pv_generation[t] == m.pv_used[t] + m.pv_curtail[t]

@model.Constraint(model.time_steps)
def electricity_balance_constraint(m, t):
    """Grid + PV = CS discharge + Grid feedin"""
    return m.grid_power[t] + m.pv_used[t] == m.cs_discharge[t] + m.grid_feedin[t]

# Objective: minimize cost
@model.Objective(sense=pyo.minimize)
def cost_objective(m):
    return sum(m.grid_power[t] for t in m.time_steps)

# Solve
solver = pyo.SolverFactory('appsi_highs')
result = solver.solve(model, tee=True)

print("\n" + "=" * 80)
print("SOLUTION STATUS:", result.solver.termination_condition)
print("=" * 80)

if result.solver.termination_condition == pyo.TerminationCondition.optimal:
    print("\n✅ MODEL IS FEASIBLE!")
    for t in model.time_steps:
        print(f"\nTime {t}:")
        print(f"  PV gen: {pyo.value(model.pv_generation[t]):.4f}")
        print(f"  PV used: {pyo.value(model.pv_used[t]):.4f}")
        print(f"  PV curtail: {pyo.value(model.pv_curtail[t]):.4f}")
        print(f"  Grid import: {pyo.value(model.grid_power[t]):.4f}")
        print(f"  Grid feedin: {pyo.value(model.grid_feedin[t]):.4f}")
        print(f"  CS discharge: {pyo.value(model.cs_discharge[t]):.4f}")
else:
    print("\n❌ MODEL IS INFEASIBLE!")
    print("Check constraints for contradictions")
