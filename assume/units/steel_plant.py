# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from typing import Dict

import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import value
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

from assume.common.base import SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook
from assume.common.utils import get_products_index
from assume.strategies.dsm_load_shift import flexibility_cost_tolerance
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
        node: str = "node0",
        index: pd.DatetimeIndex = None,
        location: tuple[float, float] = (0.0, 0.0),
        components: Dict[str, Dict] = None,
        objective: str = None,
        flexibility_measure: str = "max_load_shift",
        demand: float = 0,
        cost_tolerance: float = 10,
        **kwargs,
    ):
        super().__init__(
            id=id,
            unit_operator=unit_operator,
            technology=technology,
            bidding_strategies=bidding_strategies,
            index=index,
            node=node,
            location=location,
            **kwargs,
        )

        self.natural_gas_price = self.forecaster["fuel_price_natural_gas"]
        self.electricity_price = self.forecaster["price_EOM"]
        self.iron_ore_price = self.forecaster.get_price("iron_ore")
        self.steel_demand = demand
        self.steel_price = self.forecaster.get_price("steel")
        self.lime_co2_factor = self.forecaster.get_price("lime_co2_factor")
        self.lime_price = self.forecaster.get_price("lime")
        self.co2_price = self.forecaster.get_price("co2")

        self.recalculated_power = self.forecaster[f"{self.id}_recalculated_power"]

        self.location = location
        self.objective = objective
        self.flexibility_measure = flexibility_measure
        self.cost_tolerance = cost_tolerance
        self.components = {}

        # Main Model part
        self.model = pyo.ConcreteModel()
        self.define_sets()
        self.define_parameters()

        # Model opt
        self.model_opt = self.model.clone()
        self.model_opt.model
        self.initialize_components(components)
        self.initialize_process_sequence()
        self.define_variables()
        self.define_constraints()
        self.define_objective_opt()

        # Model flex
        # self.model_flex = self.model.create_instance()
        # if self.flexibility_measure == "max_load_shift":
        #     flexibility_cost_tolerance(self)
        # self.define_objective_flex()

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
                    == self.components["dri_plant"].b.hydrogen_in[t]
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
                    == self.components["eaf"].b.dri_input[t]
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
        self.model.steel_demand = pyo.Param(initialize=self.steel_demand)
        self.model.steel_price = pyo.Param(
            initialize=self.steel_price.mean(), within=pyo.NonNegativeReals
        )
        self.model.lime_co2_factor = pyo.Param(
            initialize=self.lime_co2_factor.mean(), within=pyo.NonNegativeReals
        )
        self.model.co2_price = pyo.Param(
            initialize=self.co2_price.mean(), within=pyo.NonNegativeReals
        )
        self.model.lime_price = pyo.Param(
            initialize=self.lime_price.mean(), within=pyo.NonNegativeReals
        )
        self.model.iron_ore_price = pyo.Param(
            initialize=self.iron_ore_price.mean(), within=pyo.NonNegativeReals
        )

    def define_variables(self):
        self.model.total_power_input = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.variable_cost = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )

    def define_constraints(self):
        @self.model.Constraint(self.model.time_steps)
        def steel_output_association_constraint(m, t):
            return (
                sum(
                    self.components["eaf"].b.steel_output[t]
                    for t in self.model.time_steps
                )
                == self.model.steel_demand
            )

        # @self.model.Constraint(self.model.time_steps)
        # def steel_output_association_constraint(m, t):
        #     return self.components["eaf"].b.steel_output[t] == self.model.steel_demand

        # @self.model.Constraint(self.model.time_steps)
        # def total_power_input_constraint(m, t):
        #     if self.flexibility_measure == "max_load_shift":
        #         return pyo.Constraint.Skip
        #     else:
        #         return (
        #             m.total_power_input[t]
        #             == self.components["electrolyser"].b.power_in[t]
        #             + self.components["eaf"].b.power_eaf[t]
        #             + self.components["dri_plant"].b.power_dri[t]
        #         )

        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
            return (
                m.total_power_input[t]
                == self.components["electrolyser"].b.power_in[t]
                + self.components["eaf"].b.power_eaf[t]
                + self.components["dri_plant"].b.power_dri[t]
            )

        @self.model.Constraint(self.model.time_steps)
        def cost_per_time_step(m, t):
            return (
                m.variable_cost[t]
                == self.components["electrolyser"].b.electrolyser_operating_cost[t]
                + self.components["dri_plant"].b.dri_operating_cost[t]
                + self.components["eaf"].b.eaf_operating_cost[t]
            )

    def define_objective_opt(self):
        if self.objective == "min_variable_cost" or "recalculate":

            @self.model_opt.Objective(sense=pyo.minimize)
            def obj_rule_opt(m):
                # Sum up the variable cost over all time steps
                total_variable_cost = sum(
                    self.model.variable_cost[t] for t in self.model.time_steps
                )

                return total_variable_cost

        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    def define_objective_flex(self):
        if self.flexibility_measure == "max_load_shift":

            @self.model.Objective(sense=pyo.maximize)
            def obj_rule_flex(m):
                maximise_load_shift = sum(
                    m.load_shift[t] for t in self.model.time_steps
                )
                return maximise_load_shift

        else:
            raise ValueError(f"Unknown objective: {self.flexibility_measure}")

    def determine_optimal_operation_without_flex(self):
        """
        Determines the optimal operation of the steel plant without considering flexibility.
        """
        # Create a solver
        solver = SolverFactory("gurobi")

        results = solver.solve(self.model_opt, tee=False)  # , tee=True

        # Check solver status and termination condition
        if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal
        ):
            logger.debug("The model was solved optimally.")

            # Display the Objective Function Value
            objective_value = self.model_opt.obj_rule_opt()
            logger.debug(f"The value of the objective function is {objective_value}.")

        elif results.solver.termination_condition == TerminationCondition.infeasible:
            logger.debug("The model is infeasible.")

        else:
            logger.debug("Solver Status: ", results.solver.status)
            logger.debug(
                "Termination Condition: ", results.solver.termination_condition
            )

        temp = self.model_opt.total_power_input.get_values()
        self.power_requirement = pd.Series(index=self.index, data=0.0)
        for i, date in enumerate(self.index):
            self.power_requirement.loc[date] = temp[i]

    def determine_optimal_operation_with_flex(self):
        """
        Determines the optimal operation of the steel plant without considering flexibility.
        """
        # Create a solver
        solver = SolverFactory("gurobi")

        results = solver.solve(self.model_flex, tee=False)  # , tee=True

        # Check solver status and termination condition
        if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal
        ):
            logger.debug("The model was solved optimally.")

            # Display the Objective Function Value
            objective_value = self.model.obj_rule_flex()
            logger.debug(f"The value of the objective function is {objective_value}.")
            print(f"The objective value is: {objective_value}")

        elif results.solver.termination_condition == TerminationCondition.infeasible:
            logger.debug("The model is infeasible.")

        else:
            logger.debug("Solver Status: ", results.solver.status)
            logger.debug(
                "Termination Condition: ", results.solver.termination_condition
            )

        temp = self.model.total_power_input.get_values()
        self.power_requirement = pd.Series(index=self.index, data=0.0)
        for i, date in enumerate(self.index):
            self.power_requirement.loc[date] = temp[i]

    def set_dispatch_plan(
        self,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ) -> None:
        """
        Adds the dispatch plan from the current market result to the total dispatch plan and calculates the cashflow.

        Args:
            marketconfig (MarketConfig): The market configuration.
            orderbook (Orderbook): The orderbook.
        """
        products_index = get_products_index(orderbook)

        # max_power = (
        #     self.forecaster.get_availability(self.id)[products_index] * self.max_power
        # )

        product_type = marketconfig.product_type
        for order in orderbook:
            start = order["start_time"]
            end = order["end_time"]
            end_excl = end - self.index.freq
            if isinstance(order["accepted_volume"], dict):
                self.outputs[product_type].loc[start:end_excl] += [
                    order["accepted_volume"][key]
                    for key in order["accepted_volume"].keys()
                ]
            else:
                self.outputs[product_type].loc[start:end_excl] += order[
                    "accepted_volume"
                ]

        self.calculate_cashflow(product_type, orderbook)

        for start in products_index:
            current_power = self.outputs[product_type][start]

            # previous_power = self.get_output_before(start)
            # op_time = self.get_operation_time(start)

            # current_power = self.calculate_ramp(op_time, previous_power, current_power)

            # if current_power > 0:
            #     current_power = min(current_power, max_power[start])
            #     current_power = max(current_power, self.min_power)

            self.outputs[product_type][start] = current_power

        self.bidding_strategies[marketconfig.market_id].calculate_reward(
            unit=self,
            marketconfig=marketconfig,
            orderbook=orderbook,
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
        for t in self.model.time_steps:
            # Calculate total variable costs for the current time step
            total_variable_costs = (
                +value(self.components["electrolyser"].b.electrolyser_operating_cost[t])
                + value(self.components["dri_plant"].b.dri_operating_cost[t])
                + value(self.components["eaf"].b.eaf_operating_cost[t])
            )

            # Calculate total energy consumption for the current time step
            total_energy_consumption = (
                value(self.components["electrolyser"].b.power_in[t])
                + value(self.components["eaf"].b.power_eaf[t])
                + +value(self.components["dri_plant"].b.power_dri[t])
            )

            # Calculate marginal cost per unit of energy
            if total_energy_consumption > 0:
                marginal_cost_per_unit_energy = (
                    total_variable_costs / total_energy_consumption
                )
            else:
                marginal_cost_per_unit_energy = 0  # Avoid division by zero

            return marginal_cost_per_unit_energy
        # return self.electricity_price.at[start]

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
