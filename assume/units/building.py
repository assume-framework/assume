# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import (
    SolverFactory,
    SolverStatus,
    TerminationCondition,
    check_available_solvers,
)

from assume.common.base import SupportsMinMax
from assume.common.utils import str_to_bool
from assume.units.dsm_load_shift import DSMFlex

SOLVERS = ["appsi_highs", "gurobi", "glpk", "cbc", "cplex"]

logger = logging.getLogger(__name__)

# Set the log level to ERROR for Pyomo to reduce verbosity
logging.getLogger("pyomo").setLevel(logging.WARNING)


class Building(DSMFlex, SupportsMinMax):
    """
    Represents a building unit within an energy system, modeling its energy consumption,
    production, and flexibility components. This class integrates various technologies
    such as heat pumps, boilers, thermal storage, electric vehicles, generic storage, and
    photovoltaic (PV) plants to optimize the building's energy usage based on defined
    objectives.

    The `Building` class utilizes the Pyomo optimization library to determine the optimal
    operation strategy that minimizes costs or meets other specified objectives. It handles
    the interactions between different energy components, ensuring that energy demands are
    met while adhering to operational constraints.

    Attributes:
        id (str): Unique identifier for the building unit.
        unit_operator (str): Operator managing the building unit.
        bidding_strategies (dict): Strategies used for energy bidding in the market.
        technology (str): Type of technology the building unit employs.
        node (str): Network node where the building unit is connected.
        index (pd.DatetimeIndex): Time index representing the granularity of the data.
        location (tuple[float, float]): Geographic coordinates (latitude, longitude) of the building.
        components (dict[str, dict]): Sub-components of the building, such as heat pumps or storage systems.
        objective (str): Optimization objective, e.g., "min_variable_cost" to minimize operational expenses.
        flexibility_measure (str): Metric used to assess the building's flexibility, e.g., "max_load_shift".
    """

    # List of optional technologies that a building unit can incorporate
    optional_technologies = [
        "heat_pump",
        "boiler",
        "thermal_storage",
        "electric_vehicle",
        "generic_storage",
        "pv_plant",
    ]

    def __init__(
        self,
        id: str,
        unit_operator: str,
        index: pd.DatetimeIndex,
        bidding_strategies: dict,
        components: dict[str, dict],
        technology: str = "building",
        objective: str = "min_variable_cost",
        node: str = "node0",
        location: tuple[float, float] = (0.0, 0.0),
        flexibility_measure: str = "max_load_shift",
        **kwargs,
    ):
        """
        Initializes a Building instance with specified parameters and sets up the optimization model.

        Args:
            id (str): Unique identifier for the building unit.
            unit_operator (str): Operator managing the building unit.
            index (pd.DatetimeIndex): Time index representing the granularity of the data.
            bidding_strategies (dict): Strategies used for energy bidding in the market.
            components (dict[str, dict]): Sub-components of the building, such as heat pumps or storage systems.
            technology (str, optional): Type of technology the building unit employs. Defaults to "building".
            objective (str, optional): Optimization objective, e.g., "min_variable_cost". Defaults to "min_variable_cost".
            node (str, optional): Network node where the building unit is connected. Defaults to "node0".
            location (tuple[float, float], optional): Geographic coordinates (latitude, longitude) of the building. Defaults to (0.0, 0.0).
            flexibility_measure (str, optional): Metric used to assess the building's flexibility. Defaults to "max_load_shift".
            **kwargs: Additional keyword arguments for parent classes.

        Raises:
            ValueError: If any of the provided components are not recognized as valid technologies.
            Exception: If none of the specified solvers are available.
        """
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

        # Validate provided components against the list of optional technologies
        for component in self.components.keys():
            if component not in self.optional_technologies:
                raise ValueError(
                    f"Component {component} is not a valid component for the building unit."
                )

        # Initialize forecasting data for various energy prices and demands
        self.electricity_price = self.forecaster["price_EOM"]
        self.natural_gas_price = self.forecaster["fuel_price_natural gas"]
        self.heat_demand = self.forecaster["heat_demand"]
        self.ev_load_profile = self.forecaster["ev_load_profile"]
        self.battery_load_profile = self.forecaster["battery_load_profile"]
        self.inflex_demand = self.forecaster[f"{self.id}_load_profile"]

        self.objective = objective
        self.flexibility_measure = flexibility_measure

        # Flags indicating the presence of specific components
        self.has_heatpump = "heat_pump" in self.components
        self.has_boiler = "boiler" in self.components
        self.has_thermal_storage = "thermal_storage" in self.components
        self.has_ev = "electric_vehicle" in self.components
        self.has_battery_storage = "generic_storage" in self.components
        self.has_pv = "pv_plant" in self.components

        # Configuration for electric vehicles selling energy to the market
        if self.has_ev:
            self.ev_sells_energy_to_market = str_to_bool(
                self.components["electric_vehicle"].get(
                    "sells_energy_to_market", "true"
                )
            )
        # Configuration for battery storage selling energy to the market
        if self.has_battery_storage:
            self.battery_sells_energy_to_market = str_to_bool(
                self.components["generic_storage"].get("sells_energy_to_market", "true")
            )

        # Configure PV plant power profile based on availability
        if self.has_pv:
            uses_power_profile = str_to_bool(
                self.components["pv_plant"].get("uses_power_profile", "false")
            )
            profile_key = (
                f"{self.id}_pv_power_profile"
                if not uses_power_profile
                else "availability_solar"
            )
            self.components["pv_plant"][
                "power_profile" if not uses_power_profile else "availability_profile"
            ] = self.forecaster[profile_key]

        # Initialize the Pyomo optimization model
        self.model = pyo.ConcreteModel()
        self.define_sets()
        self.define_parameters()
        self.define_variables()

        self.initialize_components()
        self.initialize_process_sequence()

        self.define_constraints()
        self.define_objective()

        # Select the first available solver from the predefined list
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
        self.variable_cost_series = None

    def define_sets(self) -> None:
        """
        Defines the sets used in the Pyomo optimization model.

        Specifically, this method initializes the `time_steps` set, which represents each
        discrete time interval in the model based on the provided time index.
        """
        self.model.time_steps = pyo.Set(
            initialize=[idx for idx, _ in enumerate(self.index)]
        )

    def define_parameters(self):
        """
        Defines the parameters for the Pyomo optimization model.

        This includes prices for electricity and natural gas, as well as heat and inflexible
        demands. Each parameter is indexed by the defined time steps to allow for time-dependent
        optimization.
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
        self.model.inflex_demand = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.inflex_demand)},
        )

    def define_variables(self):
        """
        Defines the decision variables for the Pyomo optimization model.

        - `variable_power`: Represents the total power input required at each time step.
        - `variable_cost`: Represents the variable cost associated with power usage at each time step.

        Both variables are defined over the `time_steps` set and are continuous real numbers.
        """
        self.model.variable_power = pyo.Var(self.model.time_steps, within=pyo.Reals)
        self.model.variable_cost = pyo.Var(self.model.time_steps, within=pyo.Reals)

    def initialize_process_sequence(self):
        """
        Initializes the process sequence and associated constraints for the building's energy
        components.

        This method sets up constraints to ensure that the heat output from components like heat
        pumps, boilers, and thermal storage meets the building's heat demand while accounting for
        any charging or discharging from thermal storage.
        """
        if self.has_heatpump or self.has_boiler or self.has_thermal_storage:

            @self.model.Constraint(self.model.time_steps)
            def heat_flow_constraint(m, t):
                """
                Ensures that the total heat output from all relevant components matches the heat demand.

                This constraint accounts for the heat produced by the heat pump and boiler, the
                discharge from thermal storage, and ensures that it meets the building's heat demand
                plus any heat being charged into the thermal storage.

                Args:
                    m: Pyomo model reference.
                    t: Current time step.

                Returns:
                    Equality condition balancing heat production and demand.
                """
                return (
                    self.model.dsm_blocks["heat_pump"].heat_out[t]
                    if self.has_heatpump
                    else 0
                ) + (
                    self.model.dsm_blocks["boiler"].heat_out[t]
                    if self.has_boiler
                    else 0
                ) + (
                    self.model.dsm_blocks["thermal_storage"].discharge[t]
                    if self.has_thermal_storage
                    else 0
                ) == self.model.heat_demand[t] + (
                    self.model.dsm_blocks["thermal_storage"].charge[t]
                    if self.has_thermal_storage
                    else 0
                )

    def define_constraints(self):
        """
        Defines the constraints for the Pyomo optimization model.

        Constraints ensure that the total power input aligns with the building's demands and
        the behavior of its components. Additional constraints restrict energy discharge from
        electric vehicles and battery storage based on market participation settings.
        """

        @self.model.Constraint(self.model.time_steps)
        def variable_power_constraint(m, t):
            """
            Ensures that the total power input is the sum of all component inputs minus any self-produced
            or stored energy at each time step.

            This constraint aggregates power from heat pumps, boilers, electric vehicles, generic storage,
            and PV plants, balancing it against the inflexible demand and any energy being discharged
            by storage or PV systems.

            Args:
                m: Pyomo model reference.
                t: Current time step.

            Returns:
                Equality condition balancing total power input.
            """
            variable_power = self.model.inflex_demand[t]

            # Add power inputs from available components
            if self.has_heatpump:
                variable_power += self.model.dsm_blocks["heat_pump"].power_in[t]
            if self.has_boiler:
                variable_power += self.model.dsm_blocks["boiler"].power_in[t]

            # Add and subtract EV and storage power if they exist
            if self.has_ev:
                variable_power += self.model.dsm_blocks["electric_vehicle"].charge[t]
                variable_power -= self.model.dsm_blocks["electric_vehicle"].discharge[t]
            if self.has_battery_storage:
                variable_power += self.model.dsm_blocks["generic_storage"].charge[t]
                variable_power -= self.model.dsm_blocks["generic_storage"].discharge[t]

            # Subtract power from PV plant if it exists
            if self.has_pv:
                variable_power -= self.model.dsm_blocks["pv_plant"].power[t]

            # Assign the calculated total to the model's variable_power for each time step
            return self.model.variable_power[t] == variable_power

        if self.has_ev and not self.ev_sells_energy_to_market:

            @self.model.Constraint(self.model.time_steps)
            def discharge_ev_to_market_constraint(m, t):
                """
                Restricts the discharging rate of the electric vehicle to self-use only.

                If the electric vehicle is not allowed to sell energy to the market, this constraint
                ensures that any energy discharged is solely used to meet the building's demands or
                stored within other storage systems.

                Args:
                    m: Pyomo model reference.
                    t: Current time step.

                Returns:
                    Inequality condition limiting EV discharge.
                """
                return self.model.dsm_blocks["electric_vehicle"].discharge[
                    t
                ] <= self.model.inflex_demand[t] + (
                    self.model.dsm_blocks["heat_pump"].power_in[t]
                    if self.has_heatpump
                    else 0
                ) + (
                    self.model.dsm_blocks["boiler"].power_in[t]
                    if self.has_boiler
                    else 0
                ) + (
                    self.model.dsm_blocks["generic_storage"].charge[t]
                    if self.has_battery_storage
                    else 0
                ) - (
                    self.model.dsm_blocks["generic_storage"].discharge[t]
                    if self.has_battery_storage
                    else 0
                ) - (self.model.dsm_blocks["pv_plant"].power[t] if self.has_pv else 0)

        if self.has_battery_storage and not self.battery_sells_energy_to_market:

            @self.model.Constraint(self.model.time_steps)
            def discharge_battery_to_market_constraint(m, t):
                """
                Restricts the discharging rate of the battery storage to self-use only.

                If the battery storage is not permitted to sell energy to the market, this constraint
                ensures that any energy discharged is used to satisfy the building's demands or
                stored within other storage systems.

                Args:
                    m: Pyomo model reference.
                    t: Current time step.

                Returns:
                    Inequality condition limiting battery discharge.
                """
                return self.model.dsm_blocks["generic_storage"].discharge[
                    t
                ] <= self.model.inflex_demand[t] + (
                    self.model.dsm_blocks["heat_pump"].power_in[t]
                    if self.has_heatpump
                    else 0
                ) + (
                    self.model.dsm_blocks["boiler"].power_in[t]
                    if self.has_boiler
                    else 0
                ) + (
                    self.model.dsm_blocks["electric_vehicle"].charge[t]
                    if self.has_ev
                    else 0
                ) - (
                    self.model.dsm_blocks["electric_vehicle"].discharge[t]
                    if self.has_ev
                    else 0
                ) - (self.model.dsm_blocks["pv_plant"].power[t] if self.has_pv else 0)

        @self.model.Constraint(self.model.time_steps)
        def variable_cost_constraint(m, t):
            """
            Calculates the variable cost associated with power usage at each time step.

            This constraint multiplies the total variable power by the corresponding electricity price
            to determine the variable cost incurred.

            Args:
                m: Pyomo model reference.
                t: Current time step.

            Returns:
                Equality condition defining the variable cost.
            """
            return (
                self.model.variable_cost[t]
                == self.model.variable_power[t] * self.model.electricity_price[t]
            )

    def define_objective(self):
        """
        Defines the objective function for the optimization model.

        Currently supports minimizing the total variable cost over all time steps. If an unknown
        objective is specified, raises a ValueError.

        Raises:
            ValueError: If the specified objective is not recognized.
        """
        if self.objective == "min_variable_cost":

            @self.model.Objective(sense=pyo.minimize)
            def obj_rule(m):
                """
                Objective function to minimize the total variable cost across all time steps.

                Aggregates the variable costs from each time step to form the total cost to be minimized.

                Args:
                    m: Pyomo model reference.

                Returns:
                    Expression representing the total variable cost.
                """
                total_variable_cost = pyo.quicksum(
                    self.model.variable_cost[t] for t in self.model.time_steps
                )
                return total_variable_cost
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    def calculate_optimal_operation_if_needed(self):
        """
        Calculates the optimal operation of the building if it hasn't been computed yet and
        the objective is to minimize variable costs.
        """
        if self.opt_power_requirement is None and self.objective == "min_variable_cost":
            self.calculate_optimal_operation()

    def calculate_optimal_operation(self):
        """
        Solves the optimization model to determine the building's optimal energy operation strategy.

        This method creates an instance of the Pyomo model, solves it using the selected solver,
        and processes the results. It handles solver status checks, logs relevant information,
        and extracts the optimal power requirements and variable costs.

        Additionally, it invokes `write_additional_outputs` to process and store results from
        specific components like battery storage and electric vehicles.
        """
        # Create an instance of the model
        instance = self.model.create_instance()
        # Solve the instance using the configured solver
        results = self.solver.solve(instance, tee=False)

        # Check solver status and termination condition
        if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal
        ):
            logger.debug("The optimization model was solved optimally.")

            # Retrieve and log the objective function value
            objective_value = instance.obj_rule()
            logger.debug(f"The value of the objective function is {objective_value}.")

        elif results.solver.termination_condition == TerminationCondition.infeasible:
            logger.debug("The optimization model is infeasible.")

        else:
            logger.debug(f"Solver Status: {results.solver.status}")
            logger.debug(
                f"Termination Condition: {results.solver.termination_condition}"
            )

        # Extract and store the total power requirement as a Pandas Series
        self.opt_power_requirement = pd.Series(
            data=instance.variable_power.get_values()
        ).set_axis(self.index)

        # Extract and store the variable cost series as a Pandas Series
        self.variable_cost_series = pd.Series(
            data=instance.variable_cost.get_values()
        ).set_axis(self.index)

        # Process additional outputs from specific components
        self.write_additional_outputs(instance)

    def write_additional_outputs(self, instance):
        """
        Extracts and stores additional outputs from the optimization instance for specific components.

        This includes the state of charge (SoC) for battery storage and electric vehicles, normalized
        by their maximum capacities.

        Args:
            instance: The solved Pyomo model instance.
        """
        if self.has_battery_storage:
            model_block = instance.dsm_blocks["generic_storage"]
            soc = pd.Series(data=model_block.soc.get_values(), dtype=float) / pyo.value(
                model_block.max_capacity
            )
            soc.index = self.index
            self.outputs["soc"] = soc

        if self.has_ev:
            model_block = instance.dsm_blocks["electric_vehicle"]
            ev_soc = pd.Series(
                data=model_block.soc.get_values(), dtype=float
            ) / pyo.value(model_block.max_capacity)
            ev_soc.index = self.index
            self.outputs["ev_soc"] = ev_soc

    def calculate_marginal_cost(self, start: pd.Timestamp, power: float) -> float:
        """
        Calculates the marginal cost of operating the building unit at a specific time and power level.

        The marginal cost represents the additional cost incurred by increasing the power output by one unit.

        Args:
            start (pd.Timestamp): The start time of the dispatch period.
            power (float): The power output level of the unit during the dispatch.

        Returns:
            float: The marginal cost of the unit for the given power level. Returns 0 if there is no power requirement.
        """
        # Initialize marginal cost
        marginal_cost = 0

        if self.opt_power_requirement[start] != 0:
            marginal_cost = abs(
                self.variable_cost_series[start] / self.opt_power_requirement[start]
            )
        return marginal_cost

    def as_dict(self) -> dict:
        """
        Serializes the building unit's attributes and components into a dictionary.

        This includes all inherited attributes as well as specific details about the unit type
        and its constituent components.

        Returns:
            dict: A dictionary representation of the building unit's attributes and components.
        """
        # List all component names
        components_list = [component for component in self.model.dsm_blocks.keys()]
        components_string = ",".join(components_list)

        # Retrieve base class attributes
        unit_dict = super().as_dict()
        # Update with Building-specific attributes
        unit_dict.update(
            {
                "unit_type": "demand",
                "components": components_string,
            }
        )

        return unit_dict