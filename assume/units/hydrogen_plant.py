# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import (
    SolverFactory,
    check_available_solvers,
)

from assume.common.base import SupportsMinMax
from assume.units.dsm_load_shift import DSMFlex

SOLVERS = ["appsi_highs", "gurobi", "glpk", "cbc", "cplex"]

logger = logging.getLogger(__name__)

# Set the log level to ERROR
logging.getLogger("pyomo").setLevel(logging.WARNING)


class HydrogenPlant(DSMFlex, SupportsMinMax):
    """
    Represents a hydrogen plant in the energy system, including electrolyser and optional seasonal hydrogen storage.

    Args:
        id (str): Unique identifier of the plant.
        unit_operator (str): Operator of the plant.
        bidding_strategies (dict): Bidding strategies.
        node (str): Node location of the plant.
        index (pd.DatetimeIndex): Time index for plant data.
        location (tuple): Plant's geographical location.
        components (dict): Components including electrolyser and hydrogen storage.
        objective (str): Optimization objective.
        flexibility_measure (str): Flexibility measure for load shifting.
        demand (float): Hydrogen demand.
        cost_tolerance (float): Maximum allowable cost increase.
    """

    required_technologies = ["electrolyser"]
    optional_technologies = ["hydrogen_seasonal_storage"]

    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        technology: str = "hydrogen_plant",
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
            components=components,
            bidding_strategies=bidding_strategies,
            index=index,
            node=node,
            location=location,
            **kwargs,
        )

        # check if the required components are present in the components dictionary
        for component in self.required_technologies:
            if component not in components.keys():
                raise ValueError(
                    f"Component {component} is required for the hydrogen plant unit."
                )

        # check if the provided components are valid and do not contain any unknown components
        for component in components.keys():
            if (
                component not in self.required_technologies
                and component not in self.optional_technologies
            ):
                raise ValueError(
                    f"Components {component} is not a valid component for the hydrogen plant unit."
                )

        # Initialize parameters
        self.electricity_price = self.forecaster["price_EOM"]
        self.demand = demand

        self.objective = objective
        self.flexibility_measure = flexibility_measure
        self.cost_tolerance = cost_tolerance

        # Check for the presence of components
        self.has_h2seasonal_storage = (
            "hydrogen_seasonal_storage" in self.components.keys()
        )
        self.has_electrolyser = "electrolyser" in self.components.keys()

        # Define the Pyomo model
        self.model = pyo.ConcreteModel()
        self.define_sets()
        self.define_parameters()
        self.define_variables()

        self.initialize_components()
        self.initialize_process_sequence()

        self.define_constraints()
        self.define_objective_opt()

        if self.flexibility_measure == "max_load_shift":
            self.flexibility_cost_tolerance(self.model)
        self.define_objective_flex()

        solvers = check_available_solvers(*SOLVERS)
        if len(solvers) < 1:
            raise Exception(f"None of {SOLVERS} are available")

        self.solver = SolverFactory(solvers[0])
        self.solver_options = {
            "output_flag": False,
            "log_to_console": False,
            "LogToConsole": 0,
        }

        self.opt_power_requirement = None
        self.flex_power_requirement = None

        self.variable_cost_series = None

    def define_sets(self):
        self.model.time_steps = pyo.Set(
            initialize=[idx for idx, _ in enumerate(self.index)]
        )

    def define_parameters(self):
        self.model.electricity_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.electricity_price)},
        )
        self.model.absolute_hydrogen_demand = pyo.Param(initialize=self.demand)

    def define_variables(self):
        self.model.total_power_input = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.variable_cost = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.hydrogen_demand = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )

    def initialize_process_sequence(self):
        """
        Initializes the process sequence and constraints for the hydrogen plant.
        Distributes hydrogen produced by the electrolyser between hydrogen demand
        and optional hydrogen storage.
        """

        @self.model.Constraint(self.model.time_steps)
        def hydrogen_production_distribution(m, t):
            """
            Balances hydrogen produced by the electrolyser to either satisfy the hydrogen demand
            directly, be stored in hydrogen storage, or both if storage is available.
            """
            electrolyser_output = self.model.dsm_blocks["electrolyser"].hydrogen_out[t]

            if self.has_h2seasonal_storage:
                # With storage: demand can be fulfilled by electrolyser, storage discharge, or both
                storage_discharge = self.model.dsm_blocks[
                    "hydrogen_seasonal_storage"
                ].discharge[t]
                storage_charge = self.model.dsm_blocks[
                    "hydrogen_seasonal_storage"
                ].charge[t]

                # Hydrogen can be supplied to demand and/or storage, and storage can also discharge to meet demand
                return (
                    electrolyser_output + storage_discharge
                    == self.model.hydrogen_demand[t] + storage_charge
                )
            else:
                # Without storage: demand is met solely by electrolyser output
                return electrolyser_output == self.model.hydrogen_demand[t]

    def define_constraints(self):
        """
        Defines the constraints for the hydrogen plant model, ensuring that the total hydrogen output
        over all time steps meets the absolute hydrogen demand. Hydrogen can be sourced from the
        electrolyser alone or combined with storage discharge if storage is available.
        """

        @self.model.Constraint()
        def total_hydrogen_demand_constraint(m):
            """
            Ensures that the total hydrogen output over all time steps meets the absolute hydrogen demand.
            If storage is available, the total demand can be fulfilled by both electrolyser output and storage discharge.
            If storage is unavailable, the electrolyser output alone must meet the demand.
            """
            if self.has_h2seasonal_storage:
                # With storage: sum of electrolyser output and storage discharge must meet the total hydrogen demand
                return (
                    sum(
                        self.model.dsm_blocks["electrolyser"].hydrogen_out[t]
                        + self.model.dsm_blocks["hydrogen_seasonal_storage"].discharge[
                            t
                        ]
                        for t in self.model.time_steps
                    )
                    == self.model.absolute_hydrogen_demand
                )
            else:
                # Without storage: sum of electrolyser output alone must meet the total hydrogen demand
                return (
                    sum(
                        self.model.dsm_blocks["electrolyser"].hydrogen_out[t]
                        for t in self.model.time_steps
                    )
                    == self.model.absolute_hydrogen_demand
                )

        # Constraint for total power input
        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
            """
            Ensures the total power input is the sum of power inputs of all components.
            """
            return (
                m.total_power_input[t]
                == self.model.dsm_blocks["electrolyser"].power_in[t]
            )

        # Constraint for variable cost per time step
        @self.model.Constraint(self.model.time_steps)
        def cost_per_time_step(m, t):
            """
            Calculates the variable cost per time step.
            """

            return (
                self.model.variable_cost[t]
                == self.model.dsm_blocks["electrolyser"].operating_cost[t]
            )

    def define_objective_opt(self):
        if self.objective == "min_variable_cost":

            @self.model.Objective(sense=pyo.minimize)
            def obj_rule_opt(m):
                return sum(self.model.variable_cost[t] for t in self.model.time_steps)

    def define_objective_flex(self):
        if self.flexibility_measure == "max_load_shift":

            @self.model.Objective(sense=pyo.maximize)
            def obj_rule_flex(m):
                return sum(m.load_shift[t] for t in self.model.time_steps)

    def calculate_optimal_operation_if_needed(self):
        if (
            self.opt_power_requirement is not None
            and self.flex_power_requirement is None
            and self.flexibility_measure == "max_load_shift"
        ):
            self.determine_optimal_operation_with_flex()

        if self.opt_power_requirement is None and self.objective == "min_variable_cost":
            self.determine_optimal_operation_without_flex()

    def calculate_marginal_cost(self, start: pd.Timestamp, power: float) -> float:
        """
        Calculate the marginal cost of the unit based on the provided time and power.

        Args:
            start (pandas.Timestamp): The start time of the dispatch.
            power (float): The power output of the unit.

        Returns:
            float: the marginal cost of the unit for the given power.
        """
        # Initialize marginal cost
        marginal_cost = 0

        if self.opt_power_requirement[start] > 0:
            marginal_cost = (
                self.variable_cost_series[start] / self.opt_power_requirement[start]
            )

        return marginal_cost

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
