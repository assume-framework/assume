# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime

import pyomo.environ as pyo
from pyomo.opt import (
    SolverFactory,
    check_available_solvers,
)

from assume.common.base import SupportsMinMax
from assume.common.forecasts import Forecaster
from assume.units.dsm_load_shift import DSMFlex

SOLVERS = ["appsi_highs", "gurobi", "glpk", "cbc", "cplex"]

logger = logging.getLogger(__name__)

# Set the log level to ERROR
logging.getLogger("pyomo").setLevel(logging.WARNING)


class SteelPlant(DSMFlex, SupportsMinMax):
    """
    The SteelPlant class represents a steel plant unit in the energy system.

    Args:
        id (str): The unique identifier of the unit.
        unit_operator (str): The operator of the unit.
        bidding_strategies (dict): The bidding strategies of the unit.
        technology (str): The technology of the unit.
        node (str): The node of the unit.
        location (tuple[float, float]): The location of the unit.
        components (dict[str, dict]): The components of the unit such as Electrolyser, DRI Plant, DRI Storage, and Electric Arc Furnace.
        objective (str): The objective of the unit, e.g. minimize variable cost ("min_variable_cost").
        flexibility_measure (str): The flexibility measure of the unit, e.g. maximum load shift ("max_load_shift").
        demand (float): The demand of the unit - the amount of steel to be produced.
        cost_tolerance (float): The cost tolerance of the unit - the maximum cost that can be tolerated when shifting the load.
    """

    required_technologies = ["dri_plant", "eaf"]
    optional_technologies = ["electrolyser", "hydrogen_buffer_storage", "dri_storage"]

    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        forecaster: Forecaster,
        technology: str = "steel_plant",
        node: str = "node0",
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
            forecaster=forecaster,
            node=node,
            location=location,
            **kwargs,
        )

        # check if the required components are present in the components dictionary
        for component in self.required_technologies:
            if component not in components.keys():
                raise ValueError(
                    f"Component {component} is required for the steel plant unit."
                )

        # check if the provided components are valid and do not contain any unknown components
        for component in components.keys():
            if (
                component not in self.required_technologies
                and component not in self.optional_technologies
            ):
                raise ValueError(
                    f"Components {component} is not a valid component for the steel plant unit."
                )

        self.natural_gas_price = self.forecaster["fuel_price_natural_gas"]
        self.hydrogen_price = self.forecaster["price_hydrogen"]
        self.electricity_price = self.forecaster["price_EOM"]
        self.iron_ore_price = self.forecaster.get_price("iron_ore")
        self.steel_demand = demand
        self.steel_price = self.forecaster.get_price("steel")
        self.lime_price = self.forecaster.get_price("lime")
        self.co2_price = self.forecaster.get_price("co2")

        self.objective = objective
        self.flexibility_measure = flexibility_measure
        self.cost_tolerance = cost_tolerance

        # Check for the presence of components
        self.has_h2storage = "hydrogen_buffer_storage" in self.components.keys()
        self.has_dristorage = "dri_storage" in self.components.keys()
        self.has_electrolyser = "electrolyser" in self.components.keys()

        # Main Model part
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

        if self.components["dri_plant"]["fuel_type"] in ["natural_gas", "both"]:
            if self.has_electrolyser:
                self.model.hydrogen_price = pyo.Param(
                    self.model.time_steps,
                    initialize={t: 0 for t in self.model.time_steps},
                )

            else:
                self.model.hydrogen_price = pyo.Param(
                    self.model.time_steps,
                    initialize={
                        t: value for t, value in enumerate(self.hydrogen_price)
                    },
                )

        elif self.components["dri_plant"]["fuel_type"] in ["hydrogen", "both"]:
            if self.has_electrolyser:
                self.model.hydrogen_price = pyo.Param(
                    self.model.time_steps,
                    initialize={t: 0 for t in self.model.time_steps},
                )
            else:
                self.model.hydrogen_price = pyo.Param(
                    self.model.time_steps,
                    initialize={
                        t: value for t, value in enumerate(self.hydrogen_price)
                    },
                )

        self.model.natural_gas_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.natural_gas_price)},
        )
        self.model.steel_demand = pyo.Param(initialize=self.steel_demand)
        self.model.steel_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.steel_price)},
            within=pyo.NonNegativeReals,
        )
        self.model.co2_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.co2_price)},
        )
        self.model.lime_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.lime_price)},
            within=pyo.NonNegativeReals,
        )
        self.model.iron_ore_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.iron_ore_price)},
            within=pyo.NonNegativeReals,
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

    def initialize_process_sequence(self):
        """
        Initializes the process sequence and constraints for the steel plant. Here, the components/ technologies are connected to establish a process for steel production
        """

        # Constraint for direct hydrogen flow from Electrolyser to DRI plant
        @self.model.Constraint(self.model.time_steps)
        def direct_hydrogen_flow_constraint(m, t):
            """
            Ensures the direct hydrogen flow from the electrolyser to the DRI plant or storage.
            """
            if self.has_electrolyser:
                if self.has_h2storage:
                    return (
                        self.model.dsm_blocks["electrolyser"].hydrogen_out[t]
                        + self.model.dsm_blocks["hydrogen_buffer_storage"].discharge[t]
                        == self.model.dsm_blocks["dri_plant"].hydrogen_in[t]
                        + self.model.dsm_blocks["hydrogen_buffer_storage"].charge[t]
                    )
                else:
                    return (
                        self.model.dsm_blocks["electrolyser"].hydrogen_out[t]
                        == self.model.dsm_blocks["dri_plant"].hydrogen_in[t]
                    )
            else:
                # If no electrolyser, ensure DRI plant hydrogen input is as expected
                return self.model.dsm_blocks["dri_plant"].hydrogen_in[t] >= 0

        # Constraint for direct hydrogen flow from Electrolyser to dri plant
        @self.model.Constraint(self.model.time_steps)
        def direct_dri_flow_constraint(m, t):
            """
            Ensures the direct DRI flow from the DRI plant to the EAF or DRI storage.
            """
            # This constraint allows part of the dri produced by the dri plant to go directly to the dri storage
            # The actual amount should ensure that it does not exceed the capacity or demand of the EAF
            if self.has_dristorage:
                return (
                    self.model.dsm_blocks["dri_plant"].dri_output[t]
                    + self.model.dsm_blocks["dri_storage"].discharge[t]
                    == self.model.dsm_blocks["eaf"].dri_input[t]
                    + self.model.dsm_blocks["dri_storage"].charge[t]
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

        # Constraint for total power input
        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
            """
            Ensures the total power input is the sum of power inputs of all components.
            """
            power_input = (
                self.model.dsm_blocks["eaf"].power_in[t]
                + self.model.dsm_blocks["dri_plant"].power_in[t]
            )
            if self.has_electrolyser:
                power_input += self.model.dsm_blocks["electrolyser"].power_in[t]
            return m.total_power_input[t] == power_input

        # Constraint for variable cost per time step
        @self.model.Constraint(self.model.time_steps)
        def cost_per_time_step(m, t):
            """
            Calculates the variable cost per time step.
            """

            variable_cost = (
                self.model.dsm_blocks["eaf"].operating_cost[t]
                + self.model.dsm_blocks["dri_plant"].operating_cost[t]
            )
            if self.has_electrolyser:
                variable_cost += self.model.dsm_blocks["electrolyser"].operating_cost[t]

            return self.model.variable_cost[t] == variable_cost

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

    def calculate_marginal_cost(self, start: datetime, power: float) -> float:
        """
        Calculate the marginal cost of the unit based on the provided time and power.

        Args:
            start (datetime.datetime): The start time of the dispatch.
            power (float): The power output of the unit.

        Returns:
            float: the marginal cost of the unit for the given power.
        """
        # Initialize marginal cost
        marginal_cost = 0

        if self.opt_power_requirement.at[start] > 0:
            marginal_cost = (
                self.variable_cost_series.at[start]
                / self.opt_power_requirement.at[start]
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
