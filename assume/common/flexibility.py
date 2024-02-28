import pyomo.environ as pyo
import pandas as pd

def modify_model_for_flexibility(self, id):
    """
    Modify the optimization model to include constraints for flexibility within cost tolerance.
    """     
    # Load values from operation_states.csv and total_cost.csv
    operation_states_df = pd.read_csv("C:\\Manish_REPO\\ASSUME\\examples\\inputs\\example_04\\operation_states.csv", index_col=0)
    total_cost_df = pd.read_csv("C:\\Manish_REPO\\ASSUME\\examples\\inputs\\example_04\\total_cost.csv")

    # Get the ID for the plant
    component_id = self.components[id].id

    # Variables

    self.model.max_ramp_up_electrolyser = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
    
    self.model.max_ramp_down_electrolyser = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
    
    self.model.max_ramp_up_eaf = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
    
    self.model.max_ramp_down_eaf = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
    
    # Parameters

    self.model.prev_power_in_electrolyser = pyo.Param(
        self.model.time_steps,
        initialize={t: operation_states_df.at[t, f'{component_id}_power'] for t in self.model.time_steps},
    )

    self.model.prev_power_in_electrolyser = pyo.Param(
        self.model.time_steps,
        initialize={t: operation_states_df.at[t, f'{component_id}_power'] for t in self.model.time_steps},
    )

    # Set the total cost
    self.model.total_cost = pyo.Param(
        initialize=total_cost_df.at[0, component_id]
    )   

    # Calculate the upper limit of the total cost that the steel plant can bear with flexibility
    upper_cost_limit = self.model.total_cost + (self.model.tolerance_percentage / 100) * self.model.total_cost

    # Add constraints to limit the total cost within the upper cost limit
    self.model.total_cost_upper_limit = pyo.Constraint(expr=sum(
        self.components["electrolyser"].b.start_cost[t]
        + self.components["electrolyser"].b.electricity_cost[t]
        + self.components["dri_plant"].b.dri_operating_cost[t]
        + self.components["eaf"].b.eaf_operating_cost[t]
        + self.iron_ore_price.iat[t] * self.components["dri_plant"].b.iron_ore_in[t]
        for t in self.model.time_steps) <= upper_cost_limit)

    # Add constraints for electrolyser ramp-up and ramp-down flexibility
    self.model.electrolyser_ramp_up_constraint = pyo.Constraint(
        self.model.time_steps,
        rule=lambda m, t: self.components["electrolyser"].b.power_in[t]
                        <= self.model.prev_power_in_electrolyser[t] + (self.components["electrolyser"].b.rated_power - self.model.prev_power_in_electrolyser[t])
    )
    self.model.electrolyser_ramp_down_constraint = pyo.Constraint(
        self.model.time_steps,
        rule=lambda m, t: self.components["electrolyser"].b.power_in[t]
                        >= self.model.prev_power_in_electrolyser[t] - (self.model.prev_power_in_electrolyser[t] - self.components["electrolyser"].b.min_power)
    )

    # Add constraints for EAF ramp-up and ramp-down flexibility
    @self.model.Constraint(self.model.time_steps)
    def max_ramp_up_electrolyser_constraint(m, t):
            return (
                m.max_ramp_up_electrolyser[t]
                == self.components["eaf"].b.rated_power_eaf - self.model.prev_power_in_eaf[t]
            )
    @self.model.Constraint(self.model.time_steps)
    def max_ramp_down_electrolyser_constraint(m, t):
            return (
            m.max_ramp_up_electrolyser[t]
                == self.components["electrolyser"].b.power_in[t] - self.components["electrolyser"].b.min_power
            )

    
        # max_ramp_down_electrolyser = {
        #     t: self.components["electrolyser"].b.power_in[t] - self.components["electrolyser"].b.min_power
        #     for t in self.model.time_steps
        # }
        # max_ramp_up_eaf = {
        #     t: self.components["eaf"].b.rated_power_eaf - self.components["eaf"].b.power_eaf[t]
        #     for t in self.model.time_steps
        # }
        # max_ramp_down_eaf = {
        #     t: self.components["eaf"].b.power_eaf[t] - self.components["eaf"].b.min_power_eaf
        #     for t in self.model.time_steps
        # }
    self.model.eaf_ramp_up_constraint = pyo.Constraint(
        self.model.time_steps,
        rule=lambda m, t: self.components["eaf"].b.power_eaf[t]
                        <=  self.model.prev_power_in_eaf[t] + (self.components["eaf"].b.rated_power_eaf - self.model.prev_power_in_eaf[t])
    )
    self.model.eaf_ramp_down_constraint = pyo.Constraint(
        self.model.time_steps,
        rule=lambda m, t: self.components["eaf"].b.power_eaf[t]
                        >=  self.model.prev_power_in_eaf[t] - (self.model.prev_power_in_eaf[t] - self.components["eaf"].b.min_power_eaf)
    )


        