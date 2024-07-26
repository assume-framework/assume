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
    create_boiler,
    create_ev,
    create_heatpump,
    create_thermal_storage,
)

SOLVERS = ["gurobi", "glpk", "cbc", "cplex"]

logger = logging.getLogger(__name__)

# Mapping of component type identifiers to their respective classes
building_components = {
    "heatpump": create_heatpump,
    "boiler": create_boiler,
    "thermal_storage": create_thermal_storage,
    "ev": create_ev,
}


class Building(SupportsMinMax, DSMFlex):
    """
    The Building class represents a building unit in the energy system.

    Args:
        id (str): The unique identifier of the unit.
        unit_operator (str): The operator of the unit.
        bidding_strategies (dict): The bidding strategies of the unit.
        technology (str): The technology of the unit.
        node (str): The node of the unit.
        index (pd.DatetimeIndex): The index for the data of the unit.
        location (tuple[float, float]): The location of the unit.
        components (dict[str, dict]): The components of the unit such as heat pump, electric boiler, and thermal storage.
        objective (str): The objective of the unit, e.g. minimize expenses ("minimize_expenses").
    """

    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        technology: str = "building",
        node: str = "node0",
        index: pd.DatetimeIndex = None,
        location: tuple[float, float] = (0.0, 0.0),
        components: dict[str, dict] = None,
        flexibility_measure: str = "max_load_shift",
        demand: float = 0,
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
            location=location,
            **kwargs,
        )

        self.electricity_price = self.forecaster["price_EOM"]
        self.natural_gas_price = self.forecaster["fuel_price_natural_gas"]
        self.heat_demand = self.forecaster["heat_demand"]
        self.ev_load_profile = self.forecaster["ev_load_profile"]
        self.additional_electricity_load = self.forecaster[
            "360_residential_load_profile"
        ]
        self.demand = demand
        self.flexibility_measure = flexibility_measure
        self.objective = objective

        # Main Model part
        self.model = pyo.ConcreteModel()
        self.define_sets()
        self.define_parameters()
        self.initialize_components(components)
        self.initialize_process_sequence()

        self.define_variables()
        self.define_constraints()
        self.define_objective()

        solvers = check_available_solvers(*SOLVERS)
        if len(solvers) < 1:
            raise Exception(f"None of {SOLVERS} are available")
        self.solver = SolverFactory(solvers[0])

        self.opt_power_requirement = None

    def initialize_components(self, components: dict[str, dict]):
        """
        Initializes the components of the building.

        Args:
            components (dict[str, dict]): The components of the building.
        """
        self.model.dsm_blocks = pyo.Block(list(components.keys()))
        for technology, component_data in components.items():
            if technology in building_components:
                factory_method = building_components[technology]
                self.model.dsm_blocks[technology].transfer_attributes_from(
                    factory_method(
                        self.model, time_steps=self.model.time_steps, **component_data
                    )
                )

    def initialize_process_sequence(self):
        """
        Initializes the process sequence and constraints for the building.
        """
        has_thermal_storage = "thermal_storage" in self.model.dsm_blocks.keys()

        @self.model.Constraint(self.model.time_steps)
        def heat_flow_constraint(m, t):
            """
            Ensures the heat flow from the heat pump or electric boiler to the thermal storage or directly to the demand.
            """
            if has_thermal_storage:
                return (
                    self.model.dsm_blocks["heatpump"].heat_out[t]
                    + self.model.dsm_blocks["boiler"].heat_out[t]
                    + self.model.dsm_blocks["thermal_storage"].discharge_thermal[t]
                    == self.model.heat_demand[t]
                    + self.model.dsm_blocks["thermal_storage"].charge_thermal[t]
                )
            else:
                return (
                    self.model.dsm_blocks["heatpump"].heat_out[t]
                    + self.model.dsm_blocks["boiler"].heat_out[t]
                    == self.model.heat_demand[t]
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
        self.model.heat_demand = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.heat_demand)},
        )
        self.model.ev_load_profile = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.ev_load_profile)},
        )
        self.model.additional_electricity_load = pyo.Param(
            self.model.time_steps,
            initialize={
                t: value for t, value in enumerate(self.additional_electricity_load)
            },
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
        def total_power_input_constraint(m, t):
            """
            Ensures the total power input is the sum of power inputs of all components.
            """
            return (
                m.total_power_input[t]
                == self.model.dsm_blocks["heatpump"].power_in[t]
                + self.model.dsm_blocks["boiler"].power_in[t]
                + self.model.dsm_blocks["ev"].charge_ev[t]
                + self.model.additional_electricity_load[t]
            )

        @self.model.Constraint(self.model.time_steps)
        def variable_cost_constraint(m, t):
            """
            Calculates the variable cost per time step.
            """
            return (
                self.model.variable_cost[t]
                == self.model.dsm_blocks["heatpump"].operating_cost_hp[t]
                + self.model.dsm_blocks["boiler"].operating_cost_boiler[t]
                + self.model.dsm_blocks["ev"].operating_cost_ev[t]
                + self.model.additional_electricity_load
                * self.model.electricity_price[t]
            )

    def define_objective(self):
        """
        Defines the objective for the optimization model.
        """
        if self.objective == "minimize_expenses":

            @self.model.Objective(sense=pyo.minimize)
            def obj_rule(m):
                """
                Minimizes the total variable cost over all time steps.
                """
                total_variable_cost = sum(
                    self.model.variable_cost[t] for t in self.model.time_steps
                )
                return total_variable_cost
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    def calculate_optimal_operation_if_needed(self):
        if self.opt_power_requirement is None and self.objective == "minimize_expenses":
            self.calculate_optimal_operation()

    def calculate_optimal_operation(self):
        """
        Determines the optimal operation of the building.
        """
        # create an instance of the model
        instance = self.model.create_instance()
        # solve the instance
        results = self.solver.solve(instance, tee=False)

        # Check solver status and termination condition
        if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal
        ):
            logger.debug("The model was solved optimally.")

            # Display the Objective Function Value
            objective_value = instance.obj_rule()
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

        for order in orderbook:
            start = order["start_time"]
            end = order["end_time"]
            end_excl = end - self.index.freq
            if isinstance(order["accepted_volume"], dict):
                self.outputs["energy"].loc[start:end_excl] += [
                    order["accepted_volume"][key]
                    for key in order["accepted_volume"].keys()
                ]
            else:
                self.outputs["energy"].loc[start:end_excl] += order["accepted_volume"]

        self.calculate_cashflow("energy", orderbook)

        for start in products_index:
            current_power = self.outputs["energy"][start]
            self.outputs["energy"][start] = current_power

        self.bidding_strategies[marketconfig.market_id].calculate_reward(
            unit=self,
            marketconfig=marketconfig,
            orderbook=orderbook,
        )

    def calculate_marginal_cost(self, start: pd.Timestamp, power: float) -> float:
        """
        Calculate the marginal cost of the unit based on the provided time and power.

        Args:
            start (pandas.Timestamp): The start time of the dispatch.
            power (float): The power output of the unit.

        Returns:
            float: the marginal cost of the unit for the given power.
        """
        for t in self.model.time_steps:
            total_variable_costs = +value(
                self.model.dsm_blocks["heatpump"].operating_cost[t]
            ) + value(self.model.dsm_blocks["electric_boiler"].operating_cost[t])
            total_energy_consumption = value(
                self.model.dsm_blocks["heatpump"].power_in[t]
            ) + value(self.model.dsm_blocks["electric_boiler"].power_in[t])

            if total_energy_consumption > 0:
                marginal_cost_per_unit_energy = (
                    total_variable_costs / total_energy_consumption
                )
            else:
                marginal_cost_per_unit_energy = 0

            return marginal_cost_per_unit_energy

    def as_dict(self) -> dict:
        """
        Returns the attributes of the unit as a dictionary, including specific attributes.

        Returns:
            dict: The attributes of the unit as a dictionary.
        """
        components_list = [component for component in self.model.dsm_blocks.keys()]
        components_string = ",".join(components_list)

        unit_dict = super().as_dict()
        unit_dict.update(
            {
                "unit_type": "demand",
                "components": components_string,
            }
        )

        return unit_dict
