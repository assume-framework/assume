import pyomo.environ as pyo

# Electrolyser class
class Electrolyser:
    def __init__(self, time_steps, max_power, min_power, efficiency, compressor_power, ramp_up, ramp_down,):
        self.model = pyo.Block()
        self.time_steps = time_steps
        self.max_power = max_power
        self.min_power = min_power
        self.efficiency = efficiency
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.compressor_power = compressor_power
        self.define_params()
        self.define_variables()
        self.define_constraints()

    def define_params(self):
        # Parameters for Electrolyser
        self.model.max_power = pyo.Param(self.time_steps, initialize=self.max_power)
        self.model.min_power = pyo.Param(self.time_steps, initialize=self.min_power)
        self.model.efficiency = pyo.Param(self.time_steps, initialize=self.efficiency)
        self.model.min_operating_time = pyo.Param(initialize=self.min_operating_time)
        self.model.min_down_time = pyo.Param(initialize=self.min_down_time)

    def define_variables(self):
        # Variables for Electrolyser
        self.model.power_in = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        self.model.hydrogen_out = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        self.model.operational_status = pyo.Var(self.time_steps, within=pyo.Binary)
        self.model.in_operation = pyo.Var(self.time_steps, within=pyo.Boolean)
        self.model.start_up = pyo.Var(self.time_steps, within=pyo.Binary)
        self.model.shut_down = pyo.Var(self.time_steps, within=pyo.Binary)

        self.model.operating_cost = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        self.model.start_cost = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)

    def define_constraints(self):
        @self.model.Constraint(time_steps)
        def power_upper_bound(m, t):
            return m.power_in[t] <= m.max_power[t] * m.in_operation[t]

        @self.model.Constraint(time_steps)
        def power_lower_bound(m, t):
            return m.power_in[t] >= m.min_power[t] * m.in_operation[t]

        @self.model.Constraint(time_steps)
        def ramp_up_constraint(m, t):
            if t == 0:
                return pyo.Constraint.Skip
            return m.power_in[t] - m.power_in[t - 1] <= m.ramp_up

        @self.model.Constraint(time_steps)
        def ramp_down_constraint(m, t):
            if t == 0:
                return pyo.Constraint.Skip
            return m.power_in[t - 1] - m.power_in[t] <= m.ramp_down

        @self.model.Constraint(time_steps)
        def power_in_equation(m, t):
            return m.power_in[t] == m.hydrogen_out[t] / m.efficiency[t]

        @self.model.Constraint(time_steps)
        def in_operation_rule(m, t):
            if t == 0:
                return m.in_operation[t] == m.start_up[t]
            return m.in_operation[t] - m.in_operation[t - 1] == m.start_up[t] - m.shut_down[t]

        @self.model.Constraint(time_steps)
        def start_up_off_rule(m, t):
            return m.start_up[t] + m.shut_down[t] <= 1

        @self.model.Constraint(time_steps)
        def shut_down_logic(m, t):
            return m.shut_down[t] >= (1 - m.in_operation[t])

        @self.model.Constraint(time_steps)
        def min_operating_time_constraint(m, t):
            if t == 0:
                return pyo.Constraint.Skip
            delta_t = t - (t - 1)
            min_operating_time_units = int(m.min_operating_time / delta_t)
            if t < min_operating_time_units:
                return m.start_up[t] == 1
            else:
                return sum(m.start_up[t - i] for i in range(min_operating_time_units)) >= min_operating_time_units * m.start_up[t]

        @self.model.Constraint(time_steps)
        def min_downtime_constraint(m, t):
            if t == 0:
                return pyo.Constraint.Skip
            delta_t = t - (t - 1)
            min_downtime_units = int(m.min_down_time / delta_t)
            if t < min_downtime_units:
                return m.shut_down[t] == 1
            else:
                return sum(1 - m.in_operation[t - i] for i in range(min_downtime_units)) >= min_downtime_units * m.shut_down[t]

        @self.model.Constraint(time_steps)
        def operating_cost_function(m, t):
            return m.operating_cost[t] == m.in_operation[t] * m.electricity_price

        @self.model.Constraint(time_steps)
        def start_up_cost(m, t):
            return m.start_cost[t] == m.start_up[t] * m.start_price
# ShaftFurnace class
class ShaftFurnace:
    def __init__(self, time_steps, max_iron_ore_throughput, max_hydrogen_input, dri_production_efficiency, specific_hydrogen_consumption):
        self.model = pyo.Block()
        self.time_steps = time_steps
        self.max_iron_ore_throughput = max_iron_ore_throughput
        self.max_hydrogen_input = max_hydrogen_input
        self.dri_production_efficiency = dri_production_efficiency
        self.specific_hydrogen_consumption = specific_hydrogen_consumption
        self.define_params()
        self.define_variables()
        self.define_constraints()

    def define_params(self):
        # Parameters for ShaftFurnace
        self.model.max_iron_ore_throughput = pyo.Param(self.time_steps, initialize=self.max_iron_ore_throughput)
        self.model.max_hydrogen_input = pyo.Param(self.time_steps, initialize=self.max_hydrogen_input)
        self.model.dri_production_efficiency = pyo.Param(self.time_steps, initialize=self.dri_production_efficiency)
        self.model.specific_hydrogen_consumption = pyo.Param(self.time_steps, initialize=self.specific_hydrogen_consumption)

    def define_variables(self):
        # Variables for ShaftFurnace
        self.model.iron_ore_in = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        self.model.hydrogen_in = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        self.model.dri_output = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)

    def define_constraints(self):
        # Constraints for ShaftFurnace
        self.model.iron_ore_constraint = pyo.Constraint(self.time_steps,
                                                        rule=lambda model, t: model.iron_ore_in[t] <= model.max_iron_ore_throughput[t])
        self.model.hydrogen_constraint = pyo.Constraint(self.time_steps,
                                                        rule=lambda model, t: model.hydrogen_in[t] <= model.max_hydrogen_input[t])
        self.model.dri_production = pyo.Constraint(self.time_steps,
                                                    rule=lambda model, t: model.hydrogen_in[t] == model.dri_output[t] * model.specific_hydrogen_consumption[t])
        self.model.dri_production = pyo.Constraint(self.time_steps,
                                                    rule=lambda model, t: model.iron_ore_in[t] == model.dri_output[t] /  model.dri_production_efficiency[t])

# SteelPlant class
class SteelPlant:
    def __init__(self, time_steps, electrolyser_params, 
                 shaft_furnace_params, dri_demand, 
                 electricity_price, hydrogen_price, iron_ore_price):
        self.model_steel_plant = pyo.ConcreteModel()
        self.time_steps = time_steps
        self.electrolyser_params = electrolyser_params
        self.shaft_furnace_params = shaft_furnace_params
        self.dri_demand = dri_demand
        self.electricity_price = electricity_price
        self.hydrogen_price = hydrogen_price
        self.iron_ore_price = iron_ore_price
        self.define_sets()
        self.initialize_components()
        self.define_params()
        self.define_variables()
        self.define_constraints()
        self.define_objective()

    def define_sets(self):
        # Define sets for the model
        self.model_steel_plant.time_steps = pyo.Set(initialize=self.time_steps)

    def initialize_components(self):
        # Initialize components as subcomponents
        self.electrolyser = Electrolyser(self.model_steel_plant.time_steps, **self.electrolyser_params)
        self.shaft_furnace = ShaftFurnace(self.model_steel_plant.time_steps, **self.shaft_furnace_params)
       
        # Attach the Block objects to the model_steel_plant
        self.model_steel_plant.electrolyser = self.electrolyser.model
        self.model_steel_plant.shaft_furnace = self.shaft_furnace.model

    def define_params(self):
        # Parameters for SteelPlant
        self.model_steel_plant.electricity_price = pyo.Param(self.model_steel_plant.time_steps, initialize=self.electricity_price)
        self.model_steel_plant.hydrogen_price = pyo.Param(self.model_steel_plant.time_steps, initialize=self.hydrogen_price)
        self.model_steel_plant.iron_ore_price = pyo.Param(self.model_steel_plant.time_steps, initialize=self.iron_ore_price)
        self.model_steel_plant.dri_demand = pyo.Param(self.model_steel_plant.time_steps, initialize=self.dri_demand)

    def define_variables(self):
        pass

    def define_constraints(self):
        # Constraints for SteelPlant
        def hydrogen_match_rule(model, t):
            # Hydrogen match constraint between Electrolyser and ShaftFurnace
            return model.electrolyser.hydrogen_out[t] == model.shaft_furnace.hydrogen_in[t]

        self.model_steel_plant.hydrogen_demand = pyo.Constraint(self.model_steel_plant.time_steps, rule=hydrogen_match_rule)

        def demand_match_constraint_rule(model, t):
            # DRI Demand match constraint
            return model.shaft_furnace.dri_output[t] == model.dri_demand[t]

        self.model_steel_plant.main_constraint = pyo.Constraint(self.model_steel_plant.time_steps, rule=demand_match_constraint_rule)


    def define_objective(self):
        # Objective function to minimize total cost
        def total_cost_rule(model):
            electricity_cost = sum(model.electricity_price[t] * model.electrolyser.power_in[t] for t in model.time_steps)
            hydrogen_cost = sum(model.hydrogen_price[t] * model.shaft_furnace.hydrogen_in[t] for t in model.time_steps)
            iron_ore_cost = sum(model.iron_ore_price[t] * model.shaft_furnace.iron_ore_in[t] for t in model.time_steps)
            return electricity_cost + hydrogen_cost + iron_ore_cost

        self.model_steel_plant.objective = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

    def run_optimization(self):
        # Solve the optimization problem
        solver = pyo.SolverFactory('gurobi')
        results = solver.solve(self.model_steel_plant, tee=True)

        if results.solver.status == pyo.SolverStatus.ok and results.solver.termination_condition == pyo.TerminationCondition.optimal:
            calculated_objective = pyo.value(self.model_steel_plant.objective)
            return results, calculated_objective
        else:
            return results, None

# Example input data
time_steps = [1, 2, 3, 4, 5]
electrolyser_params = {
    'max_power': {1: 500, 2: 500, 3: 500, 4: 500, 5: 500},
    'min_power': {1: 100, 2: 100, 3: 100, 4: 100, 5: 100},
    'ramp_up': {1: 100, 2: 100, 3: 100, 4: 100, 5: 100},
    'ramp_down': {1: 100, 2: 100, 3: 100, 4: 100, 5: 100},

    'efficiency': {1: 0.9, 2: 0.9, 3: 0.9, 4: 0.9, 5: 0.9},
    'compressor_power': {1: 10, 2: 10, 3: 10, 4: 10, 5: 10}
}
shaft_furnace_params = {
    'max_iron_ore_throughput': {1: 1000, 2: 1000, 3: 1000, 4: 1000, 5: 1000},
    'max_hydrogen_input': {1: 100, 2: 100, 3: 100, 4: 100, 5: 100},
    'dri_production_efficiency': {1: 0.95, 2: 0.95, 3: 0.95, 4: 0.95, 5: 0.95},
    'specific_hydrogen_consumption': {1: 0.9, 2: 0.9, 3: 0.9, 4: 0.9, 5: 0.9}
}
dri_demand = {1: 90, 2: 95, 3: 85, 4: 0, 5: 100}
electricity_price = {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1}
hydrogen_price = {1: 2.0, 2: 2.0, 3: 2.0, 4: 2.0, 5: 2.0}
iron_ore_price = {1: 50.0, 2: 50.0, 3: 50.0, 4: 50.0, 5: 50.0}

# Create the SteelPlant instance
steel_plant = SteelPlant(time_steps, electrolyser_params, shaft_furnace_params, dri_demand, electricity_price, hydrogen_price, iron_ore_price)

# Run the optimization and print results
results, objective = steel_plant.run_optimization()

if objective is not None:
    print(f"Optimal Objective Value: {objective:.2f}")
    for t in time_steps:
        electrolyser = steel_plant.model_steel_plant.electrolyser
        shaft_furnace = steel_plant.model_steel_plant.shaft_furnace

        print(f"Time Step {t}:")
        print(f"  DRI Demand[{t}] = {pyo.value(steel_plant.dri_demand[t]):.2f}")
        print(f"  Hydrogen In[{t}] = {pyo.value(shaft_furnace.hydrogen_in[t]):.2f}")
        print(f"  DRI Output[{t}] = {pyo.value(shaft_furnace.dri_output[t]):.2f}")
        print(f"  Power In[{t}] = {pyo.value(electrolyser.power_in[t]):.2f}")
        print(f"  Hydrogen Out[{t}] = {pyo.value(electrolyser.hydrogen_out[t]):.2f}")
        print(f"  Iron Ore In[{t}] = {pyo.value(shaft_furnace.iron_ore_in[t]):.2f}")
        print(f"  Operational_status[{t}] = {pyo.value(electrolyser.operational_status[t]):.2f}")
else:
    print("Optimization did not converge to an optimal solution.")
