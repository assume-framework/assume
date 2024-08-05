# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

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
from assume.units.dsm_load_shift import DSMFlex
from assume.units.dst_components import (
    create_driplant,
    create_dristorage,
    create_electric_arc_furnance,
    create_electrolyser,
    create_hydrogen_storage,
)

SOLVERS = ["gurobi", "glpk", "cbc", "cplex"]

logger = logging.getLogger(__name__)

# Mapping of component type identifiers to their respective classes
dst_components = {
    "electrolyser": create_electrolyser,
    "h2storage": create_hydrogen_storage,
    "dri_plant": create_driplant,
    "dri_storage": create_dristorage,
    "eaf": create_electric_arc_furnance,
}


class SteelPlant(SupportsMinMax, DSMFlex):
    """
    The SteelPlant class represents a steel plant unit in the energy system.

    Args:
        id (str): The unique identifier of the unit.
        unit_operator (str): The operator of the unit.
        bidding_strategies (dict): The bidding strategies of the unit.
        technology (str): The technology of the unit.
        node (str): The node of the unit.
        index (pd.DatetimeIndex): The index for the data of the unit.
        location (tuple[float, float]): The location of the unit.
        components (dict[str, dict]): The components of the unit such as Electrolyser, DRI Plant, DRI Storage, and Electric Arc Furnace.
        objective (str): The objective of the unit, e.g. minimize variable cost ("min_variable_cost").
        flexibility_measure (str): The flexibility measure of the unit, e.g. maximum load shift ("max_load_shift").
        demand (float): The demand of the unit - the amount of steel to be produced.
        cost_tolerance (float): The cost tolerance of the unit - the maximum cost that can be tolerated when shifting the load.
    """

    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        technology: str = "steel_plant",
        node: str = "node0",
        index: pd.DatetimeIndex = None,
        location: tuple[float, float] = (0.0, 0.0),
        components: dict[str, dict] = None,
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

        self.objective = objective
        self.flexibility_measure = flexibility_measure
        self.cost_tolerance = cost_tolerance

        # Main Model part
        self.model = pyo.ConcreteModel()
        self.define_sets()
        self.define_parameters()
        self.initialize_components(components)
        self.initialize_process_sequence()

        self.define_variables()
        self.define_constraints()
        self.define_objective_opt()

        if self.flexibility_measure == "max_load_shift":
            self.flexibility_cost_tolerance(self.model)
        self.define_objective_flex()

        solvers = check_available_solvers(*SOLVERS)
        if len(solvers) < 1:
            raise Exception(f"None of {SOLVERS} are available")
        self.solver = SolverFactory(solvers[0])

        self.opt_power_requirement = None
        self.flex_power_requirement = None

    def switch_to_opt(self, instance):
        """
        Switches the instance to solve a cost based optimisation problem by deactivating the flexibility constraints and objective.

        Args:
            instance (pyomo.ConcreteModel): The instance of the Pyomo model.

        Returns:
            pyomo.ConcreteModel: The modified instance with flexibility constraints and objective deactivated.
        """
        # deactivate the flexibility constraints and objective
        instance.obj_rule_flex.deactivate()

        instance.total_cost_upper_limit.deactivate()
        instance.total_power_input_constraint_with_flex.deactivate()

        return instance

    def switch_to_flex(self, instance):
        """
        Switches the instance to flexibility mode by deactivating few constraints and objective function.

        Args:
            instance (pyomo.ConcreteModel): The instance of the Pyomo model.

        Returns:
            pyomo.ConcreteModel: The modified instance with optimal constraints and objective deactivated.
        """
        # deactivate the optimal constraints and objective
        instance.obj_rule_opt.deactivate()
        instance.total_power_input_constraint.deactivate()

        # fix values of model.total_power_input
        for t in instance.time_steps:
            instance.total_power_input[t].fix(self.opt_power_requirement.iloc[t])
        instance.total_cost = self.total_cost

        return instance

    def initialize_components(self, components: dict[str, dict]):
        """
        Initializes the components of the steel plant.

        Args:
            components (dict[str, dict]): The components of the steel plant.
            model (pyomo.ConcreteModel): The Pyomo model.
        """
        self.model.dsm_blocks = pyo.Block(list(components.keys()))
        for technology, component_data in components.items():
            if technology in dst_components:
                factory_method = dst_components[technology]
                self.model.dsm_blocks[technology].transfer_attributes_from(
                    factory_method(
                        self.model, time_steps=self.model.time_steps, **component_data
                    )
                )

    def initialize_process_sequence(self):
        """
        Initializes the process sequence and constraints for the steel plant. Here, the components/ technologies are connected to establish a process for steel production
        """
        # Assuming the presence of 'h2storage' indicates the desire for dynamic flow management
        has_h2storage = "h2storage" in self.model.dsm_blocks.keys()

        # Constraint for direct hydrogen flow from Electrolyser to dri plant
        @self.model.Constraint(self.model.time_steps)
        def direct_hydrogen_flow_constraint(m, t):
            """
            Ensures the direct hydrogen flow from the electrolyser to the DRI plant or storage.
            """
            # This constraint allows part of the hydrogen produced by the dri plant to go directly to the EAF
            # The actual amount should ensure that it does not exceed the capacity or demand of the EAF
            if has_h2storage:
                return (
                    self.model.dsm_blocks["electrolyser"].hydrogen_out[t]
                    + self.model.dsm_blocks["h2storage"].discharge[t]
                    == self.model.dsm_blocks["dri_plant"].hydrogen_in[t]
                    + self.model.dsm_blocks["h2storage"].charge[t]
                )
            else:
                return (
                    self.model.dsm_blocks["electrolyser"].hydrogen_out[t]
                    >= self.model.dsm_blocks["dri_plant"].hydrogen_in[t]
                )

        # Assuming the presence of dristorage' indicates the desire for dynamic flow management
        has_dristorage = "dri_storage" in self.model.dsm_blocks.keys()

        # Constraint for direct hydrogen flow from Electrolyser to dri plant
        @self.model.Constraint(self.model.time_steps)
        def direct_dri_flow_constraint(m, t):
            """
            Ensures the direct DRI flow from the DRI plant to the EAF or DRI storage.
            """
            # This constraint allows part of the dri produced by the dri plant to go directly to the dri storage
            # The actual amount should ensure that it does not exceed the capacity or demand of the EAF
            if has_dristorage:
                return (
                    self.model.dsm_blocks["dri_plant"].dri_output[t]
                    + self.model.dsm_blocks["dri_storage"].discharge_dri[t]
                    == self.model.dsm_blocks["eaf"].dri_input[t]
                    + self.model.dsm_blocks["dri_storage"].charge_dri[t]
                )
            else:
                return (
                    self.model.dsm_blocks["dri_plant"].dri_output[t]
                    == self.model.dsm_blocks["eaf"].dri_input[t]
                )

        # Constraint for material flow from dri plant to Electric Arc Furnace
        @self.model.Constraint(self.model.time_steps)
        def shaft_to_arc_furnace_material_flow_constraint(m, t):
            """
            Ensures the material flow from the DRI plant to the Electric Arc Furnace.
            """
            return (
                self.model.dsm_blocks["dri_plant"].dri_output[t]
                == self.model.dsm_blocks["eaf"].dri_input[t]
            )

    def define_sets(self) -> None:
        """
        Defines the sets for the Pyomo model.
        """
        self.model.time_steps = pyo.Set(
            initialize=[idx for idx, _ in enumerate(self.index)]
        )

    def define_parameters(self):
        """
        Defines the parameters for the Pyomo model.
        """
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
        """
        Defines the variables for the Pyomo model.
        """
        self.model.total_power_input = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.variable_cost = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )

    def define_constraints(self):
        @self.model.Constraint(self.model.time_steps)
        def steel_output_association_constraint(m, t):
            """
            Ensures the steel output meets the steel demand across all time steps.

            This constraint sums the steel output from the Electric Arc Furnace (EAF) over all time steps
            and ensures that it equals the steel demand. This is useful when the steel demand is to be met
            by the total production over the entire time horizon.
            """
            return (
                sum(
                    self.model.dsm_blocks["eaf"].steel_output[t]
                    for t in self.model.time_steps
                )
                == self.model.steel_demand
            )

        """
        The following commented constraint ensures the steel output meets the steel demand for each time step.

        This constraint directly compares the steel output of the Electric Arc Furnace (EAF) at each individual
        time step to the steel demand. This is useful when the steel demand needs to be met at each specific time step,
        rather than over the entire time horizon.
        """
        # @self.model.Constraint(self.model.time_steps)
        # def steel_output_association_constraint(m, t):
        #     return self.model.dsm_blocks["eaf"].steel_output[t] == self.model.steel_demand

        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
            """
            Ensures the total power input is the sum of power inputs of all components.
            """
            return (
                m.total_power_input[t]
                == self.model.dsm_blocks["electrolyser"].power_in[t]
                + self.model.dsm_blocks["eaf"].power_eaf[t]
                + self.model.dsm_blocks["dri_plant"].power_dri[t]
            )

        @self.model.Constraint(self.model.time_steps)
        def cost_per_time_step(m, t):
            """
            Calculates the variable cost per time step.
            """
            return (
                self.model.variable_cost[t]
                == self.model.dsm_blocks["electrolyser"].electrolyser_operating_cost[t]
                + self.model.dsm_blocks["dri_plant"].dri_operating_cost[t]
                + self.model.dsm_blocks["eaf"].eaf_operating_cost[t]
            )

    def define_objective_opt(self):
        """
        Defines the objective for the optimization model.

        Args:
            model (pyomo.ConcreteModel): The Pyomo model.
        """
        if self.objective == "min_variable_cost" or "recalculate":

            @self.model.Objective(sense=pyo.minimize)
            def obj_rule_opt(m):
                """
                Minimizes the total variable cost over all time steps.
                """
                total_variable_cost = sum(
                    self.model.variable_cost[t] for t in self.model.time_steps
                )

                return total_variable_cost

        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    def define_objective_flex(self):
        """
        Defines the flexibility objective for the optimization model.

        Args:
            model (pyomo.ConcreteModel): The Pyomo model.
        """
        if self.flexibility_measure == "max_load_shift":

            @self.model.Objective(sense=pyo.maximize)
            def obj_rule_flex(m):
                """
                Maximizes the load shift over all time steps.
                """
                maximise_load_shift = sum(
                    m.load_shift[t] for t in self.model.time_steps
                )
                return maximise_load_shift

        else:
            raise ValueError(f"Unknown objective: {self.flexibility_measure}")

    def calculate_optimal_operation_if_needed(self):
        if (
            self.opt_power_requirement is not None
            and self.flex_power_requirement is None
            and self.flexibility_measure == "max_load_shift"
        ):
            self.determine_optimal_operation_with_flex()

        if self.opt_power_requirement is None and self.objective == "min_variable_cost":
            self.determine_optimal_operation_without_flex()

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
        self.opt_power_requirement = pd.Series(data=temp)
        self.opt_power_requirement.index = self.index

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

        elif results.solver.termination_condition == TerminationCondition.infeasible:
            logger.debug("The model is infeasible.")

        else:
            logger.debug("Solver Status: ", results.solver.status)
            logger.debug(
                "Termination Condition: ", results.solver.termination_condition
            )

        temp = instance.total_power_input.get_values()
        self.flex_power_requirement = pd.Series(data=temp)
        self.flex_power_requirement.index = self.index

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
                    self.model.dsm_blocks["electrolyser"].electrolyser_operating_cost[t]
                )
                + value(self.model.dsm_blocks["dri_plant"].dri_operating_cost[t])
                + value(self.model.dsm_blocks["eaf"].eaf_operating_cost[t])
            )

            # Calculate total energy consumption for the current time step
            total_energy_consumption = (
                value(self.model.dsm_blocks["electrolyser"].power_in[t])
                + value(self.model.dsm_blocks["eaf"].power_eaf[t])
                + +value(self.model.dsm_blocks["dri_plant"].power_dri[t])
            )

            # Calculate marginal cost per unit of energy
            if total_energy_consumption > 0:
                marginal_cost_per_unit_energy = (
                    total_variable_costs / total_energy_consumption
                )
            else:
                marginal_cost_per_unit_energy = 0  # Avoid division by zero

            return marginal_cost_per_unit_energy

    def as_dict(self) -> dict:
        """
        Returns the attributes of the unit as a dictionary, including specific attributes.

        Returns:
            dict: The attributes of the unit as a dictionary.
        """
        # Assuming unit_dict is a dictionary that you want to save to the database
        components_list = [component for component in self.model.dsm_blocks.keys()]

        # Convert the list to a delimited string
        components_string = ",".join(components_list)

        unit_dict = super().as_dict()
        unit_dict.update(
            {
                "unit_type": "demand",
                "components": components_string,
            }
        )

        return unit_dict
