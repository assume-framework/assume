import pyomo.environ as pe

marginal_costs = {
    "Wind": 0,
    "Coal": 30,
    "Gas": 60,
    "Oil": 80,
}

capacities = {
    "Coal": 35000,
    "Wind": 3000,
    "Gas": 8000,
    "Oil": 2000
}

load = 42000

m = pe.ConcreteModel()
m.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

m.S = pe.Set(initialize=capacities.keys())

m.g = pe.Var(m.S, within=pe.NonNegativeReals)

m.cost = pe.Objective(expr=sum(marginal_costs[s] * m.g[s] for s in m.S))

m.cost.pprint()

@m.Constraint(m.S)
def generator_limit(m, s):
    return m.g[s] <= capacities[s]

m.generator_limit.pprint()

m.energy_balance = pe.Constraint(expr=sum(m.g[s] for s in m.S) == load)

m.energy_balance.pprint()

pe.SolverFactory('gurobi').solve(m).write()