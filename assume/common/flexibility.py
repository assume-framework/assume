import pyomo.environ as pyo
import pandas as pd

def flexibility_cost_tolerance(self):
    """
    Modify the optimization model to include constraints for flexibility within cost tolerance.
    """     

    self.prev_power = self.forecaster[f"{self.id}_power"]
    self.prev_variable_cost = self.forecaster[f"{self.id}_variable_cost"]

    # Parameters
    self.model.prev_variable_cost = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.prev_variable_cost)}
        )
        
    self.model.cost_tolerance = pyo.Param(
            initialize=(self.cost_tolerance)
        )

    self.model.prev_power = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.prev_power)}
        )
    
    # Variables
    self.model.positive_flex = pyo.Var(self.model.time_steps, within=pyo.NonNegativeReals)
    self.model.negetive_flex = pyo.Var(self.model.time_steps, within=pyo.NonNegativeReals)

    self.model.ramp_up_power = pyo.Var(self.model.time_steps, within=pyo.NonNegativeReals)
    self.model.ramp_down_power = pyo.Var(self.model.time_steps, within=pyo.NonNegativeReals)
    
    self.model.upper_cost_limit = pyo.Var(within=pyo.NonNegativeReals)  
    self.model.flex_switch = pyo.Var(self.model.time_steps, within=pyo.Boolean) 

    # Calculate the upper limit of the total cost that the steel plant can bear with flexibility

    @self.model.Constraint()
    def determine_upper_cost_limit(m):
        return self.model.upper_cost_limit == sum(self.model.prev_variable_cost[t] 
                                                    for t in self.model.time_steps) + (self.model.cost_tolerance / 100) * sum(self.model.prev_variable_cost[t]                                                                                                                        for t in self.model.time_steps)

    @self.model.Constraint(self.model.time_steps)
    def total_cost_upper_limit(m, t):
        return sum(self.model.variable_cost[t] for t in self.model.time_steps) <= self.model.upper_cost_limit

    # Ramp up power electrolyser

    @self.model.Constraint(self.model.time_steps)
    def ramp_up_flex_min_bound(m, t):
        return m.ramp_up_power[t] >= m.prev_power[t]
    
    # Negetive flexibility

    @self.model.Constraint(self.model.time_steps)
    def negetive_flex_electrolyser_constraint(m, t):
        return m.negetive_flex[t] == m.ramp_up_power[t] * (1 - m.flex_switch[t]) - m.prev_power[t] 
    
    # Ramp down power
    
    @self.model.Constraint(self.model.time_steps)
    def ramp_down_flex_max_bound(m, t):
        return m.ramp_down_power[t] <= m.prev_power[t]

    @self.model.Constraint(self.model.time_steps)
    def positive_flex_constraint(m, t):
        return m.positive_flex[t] == (m.prev_power[t] - m.ramp_down_power[t]) * m.flex_switch[t]
    
    @self.model.Constraint(self.model.time_steps)
    def total_power_flex_relation_constrint(m, t):
        return m.total_power_input[t] == m.ramp_up_power[t] + m.ramp_down_power[t]

# Load data from CSV file
df = pd.read_csv('C:\\Manish_REPO\\ASSUME\\examples\\inputs\\example_04\\accepted_offers.csv')

prefixes = set(col.split('_')[0] for col in df.columns if '_' in col)

# Iterate over each prefix
for prefix in prefixes:
    # Identify columns with the current prefix for 'power' and 'accepted'
    power_col = f'{prefix}_power'
    accepted_col = f'{prefix}_accepted'
    
    # Perform subtraction only if both 'power' and 'accepted' columns exist
    if power_col in df.columns and accepted_col in df.columns:
        df[f'{prefix}_recalculated_power'] = df[power_col] + df[accepted_col]

# Save the DataFrame to a new CSV file, overwriting if columns with the same name already exist
df.to_csv('C:\\Manish_REPO\\ASSUME\\examples\\inputs\\example_04\\accepted_offers.csv', index=False)

    