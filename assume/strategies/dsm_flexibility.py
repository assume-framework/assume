# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo


def flexibility_cost_tolerance(self):
    """
    Modify the optimization model to include constraints for flexibility within cost tolerance.
    """

    self.prev_power = self.forecaster[f"{self.id}_power"]
    self.prev_variable_cost = self.forecaster[f"{self.id}_variable_cost"]

    # Parameters
    self.model.prev_variable_cost = pyo.Param(
        self.model.time_steps,
        initialize={t: value for t, value in enumerate(self.prev_variable_cost)},
    )

    self.model.cost_tolerance = pyo.Param(initialize=(self.cost_tolerance))

    self.model.prev_power = pyo.Param(
        self.model.time_steps,
        initialize={t: value for t, value in enumerate(self.prev_power)},
    )

    # Variables
    self.model.load_shift = pyo.Var(self.model.time_steps, within=pyo.Reals)

    self.model.upper_cost_limit = pyo.Var(within=pyo.NonNegativeReals)
    self.model.flex_switch = pyo.Var(self.model.time_steps, within=pyo.Binary)

    # Calculate the upper limit of the total cost that the steel plant can bear with flexibility

    @self.model.Constraint()
    def determine_upper_cost_limit(m):
        return self.model.upper_cost_limit == sum(
            self.model.prev_variable_cost[t] for t in self.model.time_steps
        ) + (self.model.cost_tolerance / 100) * sum(
            self.model.prev_variable_cost[t] for t in self.model.time_steps
        )

    @self.model.Constraint(self.model.time_steps)
    def total_cost_upper_limit(m, t):
        return (
            sum(self.model.variable_cost[t] for t in self.model.time_steps)
            <= self.model.upper_cost_limit
        )

    @self.model.Constraint(self.model.time_steps)
    def total_power_flex_relation_constraint(m, t):
        return (
            m.prev_power[t] - m.load_shift[t]
            == self.components["electrolyser"].b.power_in[t]
            + self.components["eaf"].b.power_eaf[t]
            + self.components["dri_plant"].b.power_dri[t]
        )


def recalculate_with_accepted_offers(self):
    self.reference_power = self.forecaster[f"{self.id}_power"]
    self.accepted_pos_capacity = self.forecaster[
        f"{self.id}_power_steelneg"
    ]  # _accepted_pos_res

    # Parameters
    self.model.reference_power = pyo.Param(
        self.model.time_steps,
        initialize={t: value for t, value in enumerate(self.reference_power)},
    )

    self.model.accepted_pos_capacity = pyo.Param(
        self.model.time_steps,
        initialize={t: value for t, value in enumerate(self.accepted_pos_capacity)},
    )

    # Variables
    self.model.capacity_upper_bound = pyo.Var(
        self.model.time_steps, within=pyo.NonNegativeReals
    )

    # Constraints
    @self.model.Constraint(self.model.time_steps)
    def capacity_upper_bound_constraint(m, t):
        return (
            m.capacity_upper_bound[t]
            == m.reference_power[t] - m.accepted_pos_capacity[t]
        )

    @self.model.Constraint(self.model.time_steps)
    def total_power_upper_limit(m, t):
        if m.accepted_pos_capacity[t] > 0:
            return m.total_power_input[t] <= m.capacity_upper_bound[t]
        else:
            return pyo.Constraint.Skip


# # Load data from CSV file
# df = pd.read_csv('C:\\Manish_REPO\\ASSUME\\examples\\inputs\\example_04\\accepted_offers.csv')

# prefixes = set(col.split('_')[0] for col in df.columns if '_' in col)

# # Iterate over each prefix
# for prefix in prefixes:
#     # Identify columns with the current prefix for 'power' and 'accepted'
#     power_col = f'{prefix}_power'
#     accepted_col = f'{prefix}_accepted'

#     # Perform subtraction only if both 'power' and 'accepted' columns exist
#     if power_col in df.columns and accepted_col in df.columns:
#         df[f'{prefix}_recalculated_power'] = df[power_col] + df[accepted_col]

# # Save the DataFrame to a new CSV file, overwriting if columns with the same name already exist
# df.to_csv('C:\\Manish_REPO\\ASSUME\\examples\\inputs\\example_04\\accepted_offers.csv', index=False)
