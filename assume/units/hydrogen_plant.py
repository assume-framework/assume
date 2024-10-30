# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import matplotlib.pyplot as plt
import pandas as pd
import pyomo.environ as pyo
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
    optional_technologies = ["h2_seasonal_storage"]

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
        self.hydrogen_demand = demand

        self.objective = objective
        self.flexibility_measure = flexibility_measure
        self.cost_tolerance = cost_tolerance
        self.demand = demand

        # Check for the presence of components
        self.has_has_h2seasonal_storage = (
            "h2_seasonal_storage" in self.components.keys()
        )
        self.has_electrolyser = "electrolyser" in self.components.keys()

        # Define the Pyomo model
        self.model = pyo.ConcreteModel()
        self.define_sets()
        self.define_parameters()
        self.define_variables()

        self.initialize_components()
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

        self.model.hydrogen_demand = pyo.Param(initialize=self.hydrogen_demand)

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
        Here, the components are connected to establish a process for hydrogen production and storage.
        """

        @self.model.Constraint(self.model.time_steps)
        def electrolyser_to_storage_or_demand(m, t):
            """
            Ensures the hydrogen flow from the Electrolyser to either storage or meets immediate demand.
            """
            if self.has_electrolyser:
                if self.has_h2seasonal_storage:
                    return (
                        m.dsm_blocks["electrolyser"].hydrogen_out[t]
                        + m.dsm_blocks["h2_seasonal_storage"].discharge[t]
                        == m.dsm_blocks["h2_seasonal_storage"].charge[t]
                        + self.model.hydrogen_demand[t]
                    )
                else:
                    # If no hydrogen storage, all produced hydrogen should meet demand directly.
                    return (
                        m.dsm_blocks["electrolyser"].hydrogen_out[t]
                        == self.model.hydrogen_demand[t]
                    )
            else:
                # If no electrolyser is present, hydrogen demand must be zero.
                return self.model.hydrogen_demand[t] == 0

        if self.has_storage:

            @self.model.Constraint(self.model.time_steps)
            def storage_balance_constraint(m, t):
                """
                Ensures the state of charge (SOC) for the hydrogen storage matches inputs and outputs.
                """
                soc_previous = (
                    self.model.dsm_blocks["h2_seasonal_storage"].initial_soc
                    if t == self.model.time_steps.first()
                    else self.model.dsm_blocks["h2_seasonal_storage"].soc[t - 1]
                )
                soc_current = self.model.dsm_blocks["h2_seasonal_storage"].soc[t]

                return soc_current == (
                    soc_previous
                    + m.dsm_blocks["h2_seasonal_storage"].charge[t]
                    * m.dsm_blocks["h2_seasonal_storage"].efficiency_charge
                    - m.dsm_blocks["h2_seasonal_storage"].discharge[t]
                    / m.dsm_blocks["h2_seasonal_storage"].efficiency_discharge
                )

    def define_constraints(self):
        @self.model.Constraint(self.model.time_steps)
        def hydrogen_output_constraint(m, t):
            return (
                sum(self.model.hydrogen_demand[t] for t in self.model.time_steps)
                == self.demand
            )

        # Constraint for total power input
        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
            """
            Ensures the total power input is the sum of power inputs of all components.
            """
            power_input = self.model.dsm_blocks["electrolyser"].power_in[t]
            return m.total_power_input[t] == power_input

        # Constraint for variable cost per time step
        @self.model.Constraint(self.model.time_steps)
        def cost_per_time_step(m, t):
            """
            Calculates the variable cost per time step.
            """

            variable_cost = self.model.dsm_blocks["electrolyser"].operating_cost[t]

            return self.model.variable_cost[t] == variable_cost

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

    def determine_optimal_operation_without_flex(self):
        """
        Determines the optimal operation of the steel plant without considering flexibility.
        """
        # create an instance of the model
        instance = self.model.create_instance()
        # switch the instance to the optimal mode by deactivating the flexibility constraints and objective
        instance = self.switch_to_opt(instance)
        # solve the instance
        results = self.solver.solve(instance, options=self.solver_options)

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

        self.opt_power_requirement = pd.Series(
            data=instance.total_power_input.get_values()
        ).set_axis(self.index)

        self.total_cost = sum(
            instance.variable_cost[t].value for t in instance.time_steps
        )

        # Variable cost series
        self.variable_cost_series = pd.Series(
            data=instance.variable_cost.get_values()
        ).set_axis(self.index)

        #########
        # Extract time series data for power, natural gas, and hydrogen
        power_series = pd.Series(data=instance.total_power_input.get_values())
        hydrogen_series = pd.Series(
            data=[
                instance.dsm_blocks["electrolyser"].hydrogen_out[t].value
                for t in instance.time_steps
            ]
        )

        # Set the index based on the steel plant's index (time)
        power_series.index = self.index
        hydrogen_series.index = self.index

        # Print the time series data
        print("Power Input Time Series:\n", power_series)
        print("\nHydrogen Time Series:\n", hydrogen_series)

        # Plotting the time series
        plt.figure(figsize=(12, 8))

        # Power Input Time Series
        plt.subplot(3, 1, 1)
        plt.plot(power_series, label="Power Input", color="blue")
        plt.xlabel("Time")
        plt.ylabel("Power (MW)")
        plt.title("Hydrogen Plant Power Input Time Series")
        plt.legend()

        # Hydrogen Time Series
        plt.subplot(3, 1, 3)
        plt.plot(hydrogen_series, label="Hydrogen Output", color="red")
        plt.xlabel("Time")
        plt.ylabel("Hydrogen (MW)")
        plt.title("Hydrogen Plant Hydrogen Time Series")
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Store results for possible further analysis
        self.opt_power_requirement = power_series

        self.hydrogen_series = hydrogen_series

    def determine_optimal_operation_with_flex(self):
        """
        Determines the optimal operation of the steel plant without considering flexibility.
        """
        # create an instance of the model
        instance = self.model.create_instance()
        # switch the instance to the flexibility mode by deactivating the optimal constraints and objective
        instance = self.switch_to_flex(instance)
        # solve the instance
        results = self.solver.solve(instance, options=self.solver_options)

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

        # Variable cost series
        temp_1 = instance.variable_cost.get_values()
        self.variable_cost_series = pd.Series(data=temp_1)
        self.variable_cost_series.index = self.index

        #########
        # Extract time series data for power, natural gas, and hydrogen
        power_series = pd.Series(data=instance.total_power_input.get_values())
        hydrogen_series = pd.Series(
            data=[
                instance.dsm_blocks["electrolyser"].hydrogen_out[t].value
                for t in instance.time_steps
            ]
        )

        # Set the index based on the steel plant's index (time)
        power_series.index = self.index
        hydrogen_series.index = self.index

        # Print the time series data
        print("Power Input Time Series:\n", power_series)
        print("\nHydrogen Time Series:\n", hydrogen_series)

        # Plotting the time series
        plt.figure(figsize=(12, 8))

        # Power Input Time Series
        plt.subplot(3, 1, 1)
        plt.plot(power_series, label="Power Input", color="blue")
        plt.xlabel("Time")
        plt.ylabel("Power (MW)")
        plt.title("Hydrogen Plant Power Input Time Series")
        plt.legend()

        # Hydrogen Time Series
        plt.subplot(3, 1, 3)
        plt.plot(hydrogen_series, label="Hydrogen Output", color="red")
        plt.xlabel("Time")
        plt.ylabel("Hydrogen (MW)")
        plt.title("Hydrogen Plant Hydrogen Time Series")
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Store results for possible further analysis
        self.flex_power_requirement = power_series

        self.hydrogen_series = hydrogen_series

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
