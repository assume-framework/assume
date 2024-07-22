# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pyomo.environ as pyo


class DSMFlex:
    def flexibility_cost_tolerance(self, model):
        """
        Modify the optimization model to include constraints for flexibility within cost tolerance.
        """

        model.cost_tolerance = pyo.Param(initialize=(self.cost_tolerance))
        model.total_cost = pyo.Param(initialize=0.0, mutable=True)

        # Variables
        model.load_shift = pyo.Var(model.time_steps, within=pyo.Reals)

        @model.Constraint(model.time_steps)
        def total_cost_upper_limit(m, t):
            return sum(
                model.variable_cost[t] for t in model.time_steps
            ) <= model.total_cost * (1 + (model.cost_tolerance / 100))

        @model.Constraint(model.time_steps)
        def total_power_input_constraint_with_flex(m, t):
            return (
                m.total_power_input[t] - m.load_shift[t]
                == self.model.dsm_blocks["electrolyser"].power_in[t]
                + self.model.dsm_blocks["eaf"].power_eaf[t]
                + self.model.dsm_blocks["dri_plant"].power_dri[t]
            )

    def recalculate_with_accepted_offers(self, model):
        self.reference_power = self.forecaster[f"{self.id}_power"]
        self.accepted_pos_capacity = self.forecaster[
            f"{self.id}_power_steelneg"
        ]  # _accepted_pos_res

        # Parameters
        model.reference_power = pyo.Param(
            model.time_steps,
            initialize={t: value for t, value in enumerate(self.reference_power)},
        )

        model.accepted_pos_capacity = pyo.Param(
            model.time_steps,
            initialize={t: value for t, value in enumerate(self.accepted_pos_capacity)},
        )

        # Variables
        model.capacity_upper_bound = pyo.Var(
            model.time_steps, within=pyo.NonNegativeReals
        )

        # Constraints
        @model.Constraint(model.time_steps)
        def capacity_upper_bound_constraint(m, t):
            return (
                m.capacity_upper_bound[t]
                == m.reference_power[t] - m.accepted_pos_capacity[t]
            )

        @model.Constraint(model.time_steps)
        def total_power_upper_limit(m, t):
            if m.accepted_pos_capacity[t] > 0:
                return m.total_power_input[t] <= m.capacity_upper_bound[t]
            else:
                return pyo.Constraint.Skip
