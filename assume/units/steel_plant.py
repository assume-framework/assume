# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from typing import Dict, List

import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import value
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

import assume.common.flexibility as flex
from assume.common.base import SupportsMinMax
from assume.units.dst_components import (
    DriPlant,
    DRIStorage,
    ElectricArcFurnace,
    Electrolyser,
    GenericStorage,
)

logger = logging.getLogger(__name__)

# Mapping of component type identifiers to their respective classes
dst_components = {
    "electrolyser": Electrolyser,
    "h2storage": GenericStorage,
    "dri_plant": DriPlant,
    "dri_storage": DRIStorage,
    "eaf": ElectricArcFurnace,
}


class SteelPlant(SupportsMinMax):
    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        technology: str = "steel_plant",
        node: str = "bus0",
        index: pd.DatetimeIndex = None,
        location: tuple[float, float] = (0.0, 0.0),
        components: Dict[str, Dict] = None,
        objective: str = None,
        **kwargs,
    ):
        super().__init__(
            id=id,
            unit_operator=unit_operator,
            technology=technology,
            bidding_strategies=bidding_strategies,
            index=index,
            node=node,
            **kwargs,
        )

        self.natural_gas_price = self.forecaster["fuel_price_natural_gas"]
        self.electricity_price = self.forecaster["price_EOM"]
        self.iron_ore_price = self.forecaster["iron_ore_price"]
        self.steel_demand = self.forecaster["steel_demand"]
        self.dri_price = self.forecaster["dri_price"]

        self.location = location
        self.objective = objective

        self.components = {}

        self.model = pyo.ConcreteModel()
        self.define_sets()
        self.define_parameters()

        # Initialize components based on the selected technology configuration
        self.initialize_components(components)
        self.initialize_process_sequence()

        self.define_variables()
        self.define_constraints()
        self.define_objective()

        self.power_requirement = None

    def initialize_components(self, components: Dict[str, Dict]):
        for technology, component_data in components.items():
            component_id = f"{self.id}_{technology}"
            if technology in dst_components:
                component_class = dst_components[technology]
                component_instance = component_class(
                    model=self.model, id=component_id, **component_data
                )

                # Call the add_to_model method for each component
                component_instance.add_to_model(self.model, self.model.time_steps)
                self.components[technology] = component_instance

    def initialize_process_sequence(self):
        # Assuming the presence of 'h2storage' indicates the desire for dynamic flow management
        has_h2storage = "h2storage" in self.components

        # Constraint for direct hydrogen flow from Electrolyser to dri plant
        @self.model.Constraint(self.model.time_steps)
        def direct_hydrogen_flow_constraint(m, t):
            # This constraint allows part of the hydrogen produced by the dri plant to go directly to the EAF
            # The actual amount should ensure that it does not exceed the capacity or demand of the EAF
            if has_h2storage:
                return (
                    self.components["electrolyser"].b.hydrogen_out[t]
                    + self.components["h2storage"].b.discharge[t]
                    >= self.components["dri_plant"].b.hydrogen_in[t]
                    + self.components["h2storage"].b.charge[t]
                )
            else:
                return (
                    self.components["electrolyser"].b.hydrogen_out[t]
                    >= self.components["dri_plant"].b.hydrogen_in[t]
                )

        # Assuming the presence of dristorage' indicates the desire for dynamic flow management
        has_dristorage = "dri_storage" in self.components

        # Constraint for direct hydrogen flow from Electrolyser to dri plant
        @self.model.Constraint(self.model.time_steps)
        def direct_dri_flow_constraint(m, t):
            # This constraint allows part of the dri produced by the dri plant to go directly to the dri storage
            # The actual amount should ensure that it does not exceed the capacity or demand of the EAF
            if has_dristorage:
                return (
                    self.components["dri_plant"].b.dri_output[t]
                    + self.components["dri_storage"].b.discharge_dri[t]
                    >= self.components["eaf"].b.dri_input[t]
                    + self.components["dri_storage"].b.charge_dri[t]
                )
            else:
                return (
                    self.components["dri_plant"].b.dri_output[t]
                    == self.components["eaf"].b.dri_input[t]
                )

        # Constraint for material flow from dri plant to Electric Arc Furnace
        @self.model.Constraint(self.model.time_steps)
        def shaft_to_arc_furnace_material_flow_constraint(m, t):
            return (
                self.components["dri_plant"].b.dri_output[t]
                == self.components["eaf"].b.dri_input[t]
            )

    def define_sets(self) -> None:
        self.model.time_steps = pyo.Set(
            initialize=[idx for idx, _ in enumerate(self.index)]
        )

    def define_parameters(self):
        self.model.electricity_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.electricity_price)},
        )
        self.model.natural_gas_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.natural_gas_price)},
        )
        self.model.iron_ore_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.iron_ore_price)},
        )
        self.model.dri_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.dri_price)},
        )

        self.model.steel_demand = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.steel_demand)},
        )

    def define_variables(self):
        self.model.total_power_input = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )

    def define_constraints(self):
        @self.model.Constraint(self.model.time_steps)
        def dri_output_association_constraint(m, t):
            return self.components["eaf"].b.steel_output[t] >= self.steel_demand.iat[t]

        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
            return (
                m.total_power_input[t]
                == self.components["electrolyser"].b.power_in[t]
                + self.components["eaf"].b.power_eaf[t]
            )

    def define_objective(self):
        if self.objective == "maximize_marginal_profit":

            @self.model.Objective(sense=pyo.maximize)
            def obj_rule(m):
                total_revenue = sum(
                    self.dri_price.iat[t] * m.aggregated_dri_output[t]
                    for t in m.time_steps
                )

                total_costs = sum(
                    self.electricity_price.iat[t]
                    * self.components["electrolyser"].b.power_in[t]
                    +
                    # self.hydrogen_price[t] * self.components['electrolyser'].b.hydrogen_out[t] +
                    self.iron_ore_price.iat[t]
                    * self.components["dri_plant"].b.iron_ore_in[t]
                    for t in m.time_steps
                )

                return total_revenue - total_costs

        elif self.objective == "minimize_marginal_cost":

            @self.model.Objective(sense=pyo.minimize)
            def obj_rule(m):
                total_costs = sum(
                    self.components["electrolyser"].b.start_cost[t]
                    + self.components["electrolyser"].b.electricity_cost[t]
                    + self.components["dri_plant"].b.dri_operating_cost[t]
                    + self.components["eaf"].b.eaf_operating_cost[t]
                    + self.iron_ore_price.iat[t]
                    * self.components["dri_plant"].b.iron_ore_in[t]
                    for t in m.time_steps
                )
                return total_costs

        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    def run_optimization(self):
        # Create a solver
        solver = SolverFactory("gurobi")

        results = solver.solve(self.model, tee=False)  # , tee=True

        # Check solver status and termination condition
        if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal
        ):
            logger.debug("The model was solved optimally.")

            # Display the Objective Function Value
            objective_value = self.model.obj_rule()
            logger.debug(f"The value of the objective function is {objective_value}.")

        elif results.solver.termination_condition == TerminationCondition.infeasible:
            logger.debug("The model is infeasible.")

        else:
            logger.debug("Solver Status: ", results.solver.status)
            logger.debug(
                "Termination Condition: ", results.solver.termination_condition
            )

    def determine_optimal_operation(self):
        """
        Determines the optimal operation of the steel plant without considering flexibility.
        """
        # Determine the optimal operation of the steel plant
        optimal_operation = self.run_optimization()

        temp = self.model.total_power_input.get_values()
        self.power_requirement = pd.Series(index=self.index, data=0.0)
        for i, date in enumerate(self.index):
            self.power_requirement.loc[date] = temp[i]

        # Calculate the total cost of the optimal operation
        total_cost = sum(
            self.components["electrolyser"].b.start_cost[t]
            + self.components["electrolyser"].b.electricity_cost[t]
            + self.components["dri_plant"].b.dri_operating_cost[t]
            + self.components["eaf"].b.eaf_operating_cost[t]
            + self.iron_ore_price.iat[t] * self.components["dri_plant"].b.iron_ore_in[t]
            for t in self.model.time_steps
        )
        # Calculate the operational state of the Electric Arc Furnace
        eaf_operational_state = self.components["eaf"].calculate_operational_state(
            self.model.time_steps
        )
        # Extract actual values from the Pyomo expressions and variables
        for time_step, state in eaf_operational_state.items():
            eaf_operational_state[time_step]["capacity_utilization"] = value(
                state["capacity_utilization"]
            )
            eaf_operational_state[time_step]["ramp_up_capacity"] = value(
                state["ramp_up_capacity"]
            )
            eaf_operational_state[time_step]["ramp_down_capacity"] = value(
                state["ramp_down_capacity"]
            )
            eaf_operational_state[time_step]["is_operating"] = value(
                state["is_operating"]
            )
            # eaf_operational_state[time_step]['is_min_operating_time_met'] = value(state['is_min_operating_time_met'])
            # eaf_operational_state[time_step]['is_min_down_time_met'] = value(state['is_min_down_time_met'])

        return optimal_operation, eaf_operational_state, total_cost

    def determine_flexibility(self, eaf_operational_state, tolerance):
        """
        Determines the optimal maximum negative and positive flexibility of the steel plant
        considering a tolerance on the total cost.
        """
        # Extract operational state parameters from eaf_operational_state
        ramp_up_capacity = max(
            eaf_operational_state[t]["ramp_up_capacity"] for t in self.model.time_steps
        )
        ramp_down_capacity = max(
            eaf_operational_state[t]["ramp_down_capacity"]
            for t in self.model.time_steps
        )

        # Determine the maximum negative flexibility
        max_negative_flexibility = (1 - tolerance) * sum(
            self.components["electrolyser"].b.start_cost[t]
            + self.components["electrolyser"].b.electricity_cost[t]
            + self.components["dri_plant"].b.dri_operating_cost[t]
            + self.components["eaf"].b.eaf_operating_cost[t]
            + self.iron_ore_price.iat[t] * self.components["dri_plant"].b.iron_ore_in[t]
            for t in self.model.time_steps
        )

        # Determine the maximum positive flexibility
        max_positive_flexibility = (1 + tolerance) * sum(
            self.components["electrolyser"].b.start_cost[t]
            + self.components["electrolyser"].b.electricity_cost[t]
            + self.components["dri_plant"].b.dri_operating_cost[t]
            + self.components["eaf"].b.eaf_operating_cost[t]
            + self.iron_ore_price.iat[t] * self.components["dri_plant"].b.iron_ore_in[t]
            for t in self.model.time_steps
        )

        # Apply the flexibility constraints on ramp_up_capacity and ramp_down_capacity
        adjusted_ramp_up_capacity = ramp_up_capacity + max_positive_flexibility
        adjusted_ramp_down_capacity = ramp_down_capacity + max_negative_flexibility

        # Run optimization again with adjusted flexibility parameters
        optimal_operation = self.run_optimization()

        # Capture the operational state during flexibility provision
        adjusted_eaf_operational_state = self.components[
            "eaf"
        ].calculate_operational_state(self.model.time_steps)

        return (
            adjusted_ramp_up_capacity,
            adjusted_ramp_down_capacity,
            adjusted_eaf_operational_state,
            optimal_operation,
        )

    def calculate_marginal_cost(self, start: pd.Timestamp, power: float) -> float:
        """
        Calculate the marginal cost of the unit returns the marginal cost of the unit based on the provided time and power.

        Args:
            start (pandas.Timestamp): The start time of the dispatch.
            power (float): The power output of the unit.

        Returns:
            float: the marginal cost of the unit for the given power.
        """
        return self.electricity_price.at[start]

    def as_dict(self) -> dict:
        """
        Returns the attributes of the unit as a dictionary, including specific attributes.

        Returns:
            dict: The attributes of the unit as a dictionary.
        """
        # Assuming unit_dict is a dictionary that you want to save to the database
        components_list = [component for component in self.components.keys()]

        # Convert the list to a delimited string
        components_string = ",".join(components_list)

        unit_dict = super().as_dict()
        unit_dict.update(
            {
                "unit_type": "steel_plant",
                "components": components_string,
            }
        )

        return unit_dict
