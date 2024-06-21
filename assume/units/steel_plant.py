# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from typing import Dict

import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import value
from pyomo.opt import (
    SolverFactory,
    SolverStatus,
    TerminationCondition,
    check_available_solvers,
)

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

SOLVERS = ["glpk", "gurobi", "cbc", "cplex"]

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
        self.dsm_components = {}

        # Main Model part
        self.model = pyo.ConcreteModel()
        self.define_sets(self.model)
        self.define_parameters(self.model)
        self.initialize_components(components, self.model)
        self.initialize_process_sequence(self.model)

        self.define_variables(self.model)
        self.define_constraints(self.model)
        self.define_objective_opt(self.model)

        if self.flexibility_measure == "max_load_shift":
            flexibility_cost_tolerance(self, self.model)
        self.define_objective_flex(self.model)

        solvers = check_available_solvers(*SOLVERS)
        if len(solvers) < 1:
            raise Exception(f"None of {SOLVERS} are available")
        self.solver = SolverFactory(solvers[0])

        self.opt_power_requirement = None
        self.flex_power_requirement = None

    def switch_to_opt(self, instance):
        # deactivate the flexibility constraints and objective
        instance.obj_rule_flex.deactivate()

        instance.total_cost_upper_limit.deactivate()
        instance.total_power_input_constraint_with_flex.deactivate()

        return instance

    def switch_to_flex(self, instance):
        # deactivate the optimal constraints and objective
        instance.obj_rule_opt.deactivate()
        instance.total_power_input_constraint.deactivate()

        # fix values of model.total_power_input
        for t in instance.time_steps:
            instance.total_power_input[t].fix(self.opt_power_requirement[t])
        instance.total_cost = self.total_cost

        return instance

    def initialize_components(self, components: Dict[str, Dict], model):
        for technology, component_data in components.items():
            component_id = f"{self.id}_{technology}"
            if technology in dst_components:
                component_class = dst_components[technology]
                component_instance = component_class(
                    model=model, id=component_id, **component_data
                )

                # Call the add_to_model method for each component
                component_instance.add_to_model(model, model.time_steps)
                self.dsm_components[technology] = component_instance

    def initialize_process_sequence(self, model):
        # Assuming the presence of 'h2storage' indicates the desire for dynamic flow management
        has_h2storage = "h2storage" in self.dsm_components

        # Constraint for direct hydrogen flow from Electrolyser to dri plant
        @model.Constraint(model.time_steps)
        def direct_hydrogen_flow_constraint(m, t):
            # This constraint allows part of the hydrogen produced by the dri plant to go directly to the EAF
            # The actual amount should ensure that it does not exceed the capacity or demand of the EAF
            if has_h2storage:
                return (
                    self.dsm_components["electrolyser"].b.hydrogen_out[t]
                    + self.dsm_components["h2storage"].b.discharge[t]
                    == self.dsm_components["dri_plant"].b.hydrogen_in[t]
                    + self.dsm_components["h2storage"].b.charge[t]
                )
            else:
                return (
                    self.dsm_components["electrolyser"].b.hydrogen_out[t]
                    >= self.dsm_components["dri_plant"].b.hydrogen_in[t]
                )

        # Assuming the presence of dristorage' indicates the desire for dynamic flow management
        has_dristorage = "dri_storage" in self.dsm_components

        # Constraint for direct hydrogen flow from Electrolyser to dri plant
        @model.Constraint(model.time_steps)
        def direct_dri_flow_constraint(m, t):
            # This constraint allows part of the dri produced by the dri plant to go directly to the dri storage
            # The actual amount should ensure that it does not exceed the capacity or demand of the EAF
            if has_dristorage:
                return (
                    self.dsm_components["dri_plant"].b.dri_output[t]
                    + self.dsm_components["dri_storage"].b.discharge_dri[t]
                    == self.dsm_components["eaf"].b.dri_input[t]
                    + self.dsm_components["dri_storage"].b.charge_dri[t]
                )
            else:
                return (
                    self.dsm_components["dri_plant"].b.dri_output[t]
                    == self.dsm_components["eaf"].b.dri_input[t]
                )

        # Constraint for material flow from dri plant to Electric Arc Furnace
        @model.Constraint(model.time_steps)
        def shaft_to_arc_furnace_material_flow_constraint(m, t):
            return (
                self.dsm_components["dri_plant"].b.dri_output[t]
                == self.dsm_components["eaf"].b.dri_input[t]
            )

    def define_sets(self, model) -> None:
        model.time_steps = pyo.Set(initialize=[idx for idx, _ in enumerate(self.index)])

    def define_parameters(self, model):
        model.electricity_price = pyo.Param(
            model.time_steps,
            initialize={t: value for t, value in enumerate(self.electricity_price)},
        )
        model.natural_gas_price = pyo.Param(
            model.time_steps,
            initialize={t: value for t, value in enumerate(self.natural_gas_price)},
        )
        model.steel_demand = pyo.Param(initialize=self.steel_demand)
        model.steel_price = pyo.Param(
            initialize=self.steel_price.mean(), within=pyo.NonNegativeReals
        )
        model.lime_co2_factor = pyo.Param(
            initialize=self.lime_co2_factor.mean(), within=pyo.NonNegativeReals
        )
        model.co2_price = pyo.Param(
            initialize=self.co2_price.mean(), within=pyo.NonNegativeReals
        )
        model.lime_price = pyo.Param(
            initialize=self.lime_price.mean(), within=pyo.NonNegativeReals
        )
        model.iron_ore_price = pyo.Param(
            initialize=self.iron_ore_price.mean(), within=pyo.NonNegativeReals
        )

    def define_variables(self, model):
        model.total_power_input = pyo.Var(model.time_steps, within=pyo.NonNegativeReals)
        model.variable_cost = pyo.Var(model.time_steps, within=pyo.NonNegativeReals)

    def define_constraints(self, model):
        @model.Constraint(model.time_steps)
        def steel_output_association_constraint(m, t):
            return (
                sum(
                    self.dsm_components["eaf"].b.steel_output[t]
                    for t in model.time_steps
                )
                == model.steel_demand
            )

        # @self.model.Constraint(self.model.time_steps)
        # def steel_output_association_constraint(m, t):
        #     return self.dsm_components["eaf"].b.steel_output[t] == self.model.steel_demand

        # @self.model.Constraint(self.model.time_steps)
        # def total_power_input_constraint(m, t):
        #     if self.flexibility_measure == "max_load_shift":
        #         return pyo.Constraint.Skip
        #     else:
        #         return (
        #             m.total_power_input[t]
        #             == self.dsm_components["electrolyser"].b.power_in[t]
        #             + self.dsm_components["eaf"].b.power_eaf[t]
        #             + self.dsm_components["dri_plant"].b.power_dri[t]
        #         )

        @model.Constraint(model.time_steps)
        def total_power_input_constraint(m, t):
            return (
                m.total_power_input[t]
                == self.dsm_components["electrolyser"].b.power_in[t]
                + self.dsm_components["eaf"].b.power_eaf[t]
                + self.dsm_components["dri_plant"].b.power_dri[t]
            )

        @model.Constraint(model.time_steps)
        def cost_per_time_step(m, t):
            return (
                model.variable_cost[t]
                == self.dsm_components["electrolyser"].b.electrolyser_operating_cost[t]
                + self.dsm_components["dri_plant"].b.dri_operating_cost[t]
                + self.dsm_components["eaf"].b.eaf_operating_cost[t]
            )

    def define_objective_opt(self, model):
        if self.objective == "min_variable_cost" or "recalculate":

            @model.Objective(sense=pyo.minimize)
            def obj_rule_opt(m):
                # Sum up the variable cost over all time steps
                total_variable_cost = sum(
                    model.variable_cost[t] for t in model.time_steps
                )

                return total_variable_cost

        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    def define_objective_flex(self, model):
        if self.flexibility_measure == "max_load_shift":

            @model.Objective(sense=pyo.maximize)
            def obj_rule_flex(m):
                maximise_load_shift = sum(m.load_shift[t] for t in model.time_steps)
                return maximise_load_shift

        else:
            raise ValueError(f"Unknown objective: {self.flexibility_measure}")

    def determine_optimal_operation_without_flex(self):
        """
        Determines the optimal operation of the steel plant without considering flexibility.
        """
        # create an instance of the model
        instance = self.model.create_instance()
        # switch the instance to the optimal mode by deactivating the flexibility constraints and objective
        instance = self.switch_to_opt(instance)
        # solve the instance
        results = self.solver.solve(instance, tee=False)  # , tee=True

        # Check solver status and termination condition
        if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal
        ):
            logger.debug("The model was solved optimally.")

            # Display the Objective Function Value
            objective_value = instance.obj_rule_opt()
            logger.debug(f"The value of the objective function is {objective_value}.")

        elif results.solver.termination_condition == TerminationCondition.infeasible:
            logger.debug("The model is infeasible.")

        else:
            logger.debug("Solver Status: ", results.solver.status)
            logger.debug(
                "Termination Condition: ", results.solver.termination_condition
            )

        temp = instance.total_power_input.get_values()
        self.opt_power_requirement = pd.Series(data=temp, index=self.index)

        self.total_cost = sum(
            instance.variable_cost[t].value for t in instance.time_steps
        )

    def determine_optimal_operation_with_flex(self):
        """
        Determines the optimal operation of the steel plant without considering flexibility.
        """
        # create an instance of the model
        instance = self.model.create_instance()
        # switch the instance to the flexibility mode by deactivating the optimal constraints and objective
        instance = self.switch_to_flex(instance)
        # solve the instance
        results = self.solver.solve(instance, tee=False)  # , tee=True

        # Check solver status and termination condition
        if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal
        ):
            logger.debug("The model was solved optimally.")

            # Display the Objective Function Value
            objective_value = instance.obj_rule_flex()
            logger.debug(f"The value of the objective function is {objective_value}.")
            print(f"The objective value is: {objective_value}")

        elif results.solver.termination_condition == TerminationCondition.infeasible:
            logger.debug("The model is infeasible.")

        else:
            logger.debug("Solver Status: ", results.solver.status)
            logger.debug(
                "Termination Condition: ", results.solver.termination_condition
            )

        temp = instance.total_power_input.get_values()
        self.flex_power_requirement = pd.Series(data=temp, index=self.index)

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
                +value(
                    self.dsm_components["electrolyser"].b.electrolyser_operating_cost[t]
                )
                + value(self.dsm_components["dri_plant"].b.dri_operating_cost[t])
                + value(self.dsm_components["eaf"].b.eaf_operating_cost[t])
            )

            # Calculate total energy consumption for the current time step
            total_energy_consumption = (
                value(self.dsm_components["electrolyser"].b.power_in[t])
                + value(self.dsm_components["eaf"].b.power_eaf[t])
                + +value(self.dsm_components["dri_plant"].b.power_dri[t])
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
        components_list = [component for component in self.dsm_components.keys()]

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
