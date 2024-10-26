# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pyomo.environ as pyo

from assume.units.dst_components import demand_side_technologies


class DSMFlex:
    def __init__(self, components, **kwargs):
        super().__init__(**kwargs)

        self.components = components

    def initialize_components(self):
        """
        Initializes the DSM components by creating and adding blocks to the model.

        This method iterates over the provided components, instantiates their corresponding classes,
        and adds the respective blocks to the Pyomo model.

        Args:
            components (dict[str, dict]): A dictionary where each key is a technology name and
                                        the value is a dictionary of parameters for the respective technology.
                                        Each technology is mapped to a corresponding class in `demand_side_technologies`.

        The method:
        - Looks up the corresponding class for each technology in `demand_side_technologies`.
        - Instantiates the class by passing the required parameters.
        - Adds the resulting block to the model under the `dsm_blocks` attribute.
        """
        components = self.components.copy()
        self.model.dsm_blocks = pyo.Block(list(components.keys()))

        for technology, component_data in components.items():
            if technology in demand_side_technologies:
                # Get the class from the dictionary mapping (adjust `demand_side_technologies` to hold classes)
                component_class = demand_side_technologies[technology]

                # Instantiate the component with the required parameters (unpack the component_data dictionary)
                component_instance = component_class(
                    time_steps=self.model.time_steps, **component_data
                )
                # Add the component to the components dictionary
                self.components[technology] = component_instance

                # Add the component's block to the model
                component_instance.add_to_model(
                    self.model, self.model.dsm_blocks[technology]
                )

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
            if self.has_electrolyser:
                return (
                    m.total_power_input[t] - m.load_shift[t]
                    == self.model.dsm_blocks["electrolyser"].power_in[t]
                    + self.model.dsm_blocks["eaf"].power_in[t]
                    + self.model.dsm_blocks["dri_plant"].power_in[t]
                )
            else:
                return (
                    m.total_power_input[t] - m.load_shift[t]
                    == self.model.dsm_blocks["eaf"].power_in[t]
                    + self.model.dsm_blocks["dri_plant"].power_in[t]
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
