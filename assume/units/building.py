# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import pandas as pd
import pyomo.environ as pyo

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
    required_technologies = []
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
        bidding_strategies: dict,
        components: dict[str, dict] = None,
        technology: str = "building",
        objective: str = None,
        node: str = "node0",
        location: tuple[float, float] = (0.0, 0.0),
        index: pd.DatetimeIndex = None,
        flexibility_measure: str = "",
        is_prosumer: str = "No",
        cost_tolerance: float = 10,
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
            is_prosumer (str): Indicates whether the building participates as a prosumer in the market.
                               Accepts "Yes" or "No" (case-insensitive).
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

        # check if the required components are present in the components dictionary
        for component in self.required_technologies:
            if component not in components.keys():
                raise ValueError(
                    f"Component {component} is required for the building plant unit."
                )

        # check if the provided components are valid and do not contain any unknown components
        for component in components.keys():
            if (
                component not in self.required_technologies
                and component not in self.optional_technologies
            ):
                raise ValueError(
                    f"Components {component} is not a valid component for the building unit."
                )

        # Initialize forecasting data for various energy prices and demands
        self.electricity_price = self.forecaster["price_EOM"]
        self.natural_gas_price = self.forecaster["fuel_price_natural gas"]
        self.heat_demand = self.forecaster[f"{self.id}_heat_demand"]
        self.ev_load_profile = self.forecaster["ev_load_profile"]
        self.battery_load_profile = self.forecaster["battery_load_profile"]
        self.inflex_demand = self.forecaster[f"{self.id}_load_profile"]

        self.objective = objective
        self.flexibility_measure = flexibility_measure
        self.cost_tolerance = cost_tolerance
        self.is_prosumer = str_to_bool(is_prosumer)

        # Check for the presence of components
        self.has_heatpump = "heat_pump" in self.components.keys()
        self.has_boiler = "boiler" in self.components.keys()
        self.has_thermal_storage = "thermal_storage" in self.components.keys()
        self.has_ev = "electric_vehicle" in self.components.keys()
        self.has_battery_storage = "generic_storage" in self.components.keys()
        self.has_pv = "pv_plant" in self.components.keys()

        self.opt_power_requirement = None
        self.flex_power_requirement = None
        self.variable_cost_series = None

        # Initialize the Pyomo optimization model
        self.model = pyo.ConcreteModel()
        self.define_sets()

        # Configure PV plant power profile based on availability
        if self.has_pv:
            profile_key = (
                f"{self.id}_pv_power_profile"
                if not str_to_bool(
                    self.components["pv_plant"].get("uses_power_profile", "false")
                )
                else "availability_solar"
            )
            pv_profile = self.forecaster[profile_key]

            # Convert and align pv_profile with Pyomo time steps
            pv_profile = pv_profile.as_pd_series()
            pv_profile.index = list(self.model.time_steps)

            # Assign the aligned profile
            self.components["pv_plant"][
                "power_profile"
                if profile_key.endswith("power_profile")
                else "availability_profile"
            ] = pv_profile

        self.define_parameters()
        self.define_variables()

        self.initialize_components()
        self.initialize_process_sequence()

        self.define_constraints()
        self.define_objective_opt()

        self.determine_optimal_operation_without_flex(switch_flex_off=False)

        # Apply the flexibility function based on flexibility measure
        if self.flexibility_measure in DSMFlex.flexibility_map:
            DSMFlex.flexibility_map[self.flexibility_measure](self, self.model)
        else:
            raise ValueError(f"Unknown flexibility measure: {self.flexibility_measure}")

        self.define_objective_flex()

    def get_prosumer_components(self):
        """
        Identifies and returns a list of components capable of selling energy when the building is a prosumer.

        Returns:
            list[str]: Names of components that can participate in the market.
        """
        prosumer_components = []
        if self.is_prosumer:
            if self.has_ev:
                prosumer_components.append("electric_vehicle")
            if self.has_battery_storage:
                prosumer_components.append("generic_storage")
        return prosumer_components

    def define_sets(self) -> None:
        """
        Defines the sets used in the Pyomo optimization model.

        Specifically, this method initializes the `time_steps` set, which represents each
        discrete time interval in the model based on the provided time index.
        """

        # Create a pyo.Set from FastIndex
        self.model.time_steps = pyo.Set(initialize=list(range(len(self.index))))

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

        - `total_power_input`: Represents the total power input required at each time step.
        - `variable_cost`: Represents the variable cost associated with power usage at each time step.

        Both variables are defined over the `time_steps` set and are continuous real numbers.
        """
        self.model.total_power_input = pyo.Var(self.model.time_steps, within=pyo.Reals)
        self.model.variable_cost = pyo.Var(self.model.time_steps, within=pyo.Reals)

    def initialize_process_sequence(self):
        """
        Connects the energy components in the building system, establishing constraints for energy flow
        between technologies like heat pumps, boilers, EVs, batteries, and PVs.
        """

        # Heat flow constraint for heating components
        if self.has_heatpump or self.has_boiler or self.has_thermal_storage:

            @self.model.Constraint(self.model.time_steps)
            def heating_demand_balance_constraint(m, t):
                """
                Ensures the total heat output matches demand plus storage dynamics.
                """
                heat_pump_output = (
                    self.model.dsm_blocks["heat_pump"].heat_out[t]
                    if self.has_heatpump
                    else 0
                )
                boiler_output = (
                    self.model.dsm_blocks["boiler"].heat_out[t]
                    if self.has_boiler
                    else 0
                )
                thermal_storage_discharge = (
                    self.model.dsm_blocks["thermal_storage"].discharge[t]
                    if self.has_thermal_storage
                    else 0
                )
                thermal_storage_charge = (
                    self.model.dsm_blocks["thermal_storage"].charge[t]
                    if self.has_thermal_storage
                    else 0
                )
                return (
                    heat_pump_output + boiler_output + thermal_storage_discharge
                    == self.model.heat_demand[t] + thermal_storage_charge
                )

        # Electric flow and battery/EV constraints
        if self.has_ev:

            @self.model.Constraint(self.model.time_steps)
            def ev_energy_flow_constraint(m, t):
                """
                Ensures that EV energy flows are connected appropriately.
                """
                ev_discharge = self.model.dsm_blocks["electric_vehicle"].discharge[t]
                # ev_charge = self.model.dsm_blocks["electric_vehicle"].charge[t]
                inflex_demand = self.model.inflex_demand[t]

                pv_output = (
                    self.model.dsm_blocks["pv_plant"].power[t] if self.has_pv else 0
                )
                battery_discharge = (
                    self.model.dsm_blocks["generic_storage"].discharge[t]
                    if self.has_battery_storage
                    else 0
                )
                battery_charge = (
                    self.model.dsm_blocks["generic_storage"].charge[t]
                    if self.has_battery_storage
                    else 0
                )
                return (
                    ev_discharge
                    <= inflex_demand + battery_charge - battery_discharge - pv_output
                )

        if self.has_battery_storage:

            @self.model.Constraint(self.model.time_steps)
            def battery_energy_flow_constraint(m, t):
                """
                Ensures battery storage discharges are appropriately aligned with system demands.
                """
                battery_discharge = self.model.dsm_blocks["generic_storage"].discharge[
                    t
                ]
                # battery_charge = self.model.dsm_blocks["generic_storage"].charge[t]
                heat_pump_power = (
                    self.model.dsm_blocks["heat_pump"].power_in[t]
                    if self.has_heatpump
                    else 0
                )
                boiler_power = (
                    self.model.dsm_blocks["boiler"].power_in[t]
                    if self.has_boiler
                    else 0
                )
                ev_charge = (
                    self.model.dsm_blocks["electric_vehicle"].charge[t]
                    if self.has_ev
                    else 0
                )
                pv_output = (
                    self.model.dsm_blocks["pv_plant"].power[t] if self.has_pv else 0
                )
                return (
                    battery_discharge
                    <= self.model.inflex_demand[t]
                    + heat_pump_power
                    + boiler_power
                    + ev_charge
                    - pv_output
                )

    def define_constraints(self):
        """
        Defines the constraints for the Pyomo optimization model.

        Constraints ensure that the total power input aligns with the building's demands and
        the behavior of its components. Additional constraints restrict energy discharge from
        electric vehicles and battery storage based on market participation settings.
        """

        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
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
            total_power_input = self.model.inflex_demand[t]

            # Add power inputs from available components
            if self.has_heatpump:
                total_power_input += self.model.dsm_blocks["heat_pump"].power_in[t]
            if self.has_boiler:
                total_power_input += self.model.dsm_blocks["boiler"].power_in[t]

            # Add and subtract EV and storage power if they exist
            if self.has_ev:
                total_power_input += self.model.dsm_blocks["electric_vehicle"].charge[t]
                total_power_input -= self.model.dsm_blocks[
                    "electric_vehicle"
                ].discharge[t]
            if self.has_battery_storage:
                total_power_input += self.model.dsm_blocks["generic_storage"].charge[t]
                total_power_input -= self.model.dsm_blocks["generic_storage"].discharge[
                    t
                ]

            # Subtract power from PV plant if it exists
            if self.has_pv:
                total_power_input -= self.model.dsm_blocks["pv_plant"].power[t]

            # Assign the calculated total to the model's total_power_input for each time step
            return self.model.total_power_input[t] == total_power_input

        # Restrict discharge for non-prosumer components
        for component in self.get_prosumer_components():

            @self.model.Constraint(self.model.time_steps)
            def restrict_discharge_to_market(m, t, component=component):
                """
                Restricts discharges to self-use only for non-prosumer components.
                """
                discharge = self.model.dsm_blocks[component].discharge[t]
                return discharge <= self.model.inflex_demand[t]

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
                == self.model.total_power_input[t] * self.model.electricity_price[t]
            )

    def define_objective_opt(self):
        """
        Defines the objective function for the optimization model.

        Currently supports minimizing the total variable cost over all time steps. If an unknown
        objective is specified, raises a ValueError.

        Raises:
            ValueError: If the specified objective is not recognized.
        """
        if self.objective == "min_variable_cost":

            @self.model.Objective(sense=pyo.minimize)
            def obj_rule_opt(m):
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

    def define_objective_flex(self):
        """
        Defines the flexibility objective for the optimization model.

        Args:
            model (pyomo.ConcreteModel): The Pyomo model.
        """
        if self.flexibility_measure == "cost_based_load_shift":

            @self.model.Objective(sense=pyo.maximize)
            def obj_rule_flex(m):
                """
                Maximizes the load shift over all time steps.
                """

                maximise_load_shift = pyo.quicksum(
                    m.load_shift_pos[t] for t in m.time_steps
                )

                return maximise_load_shift

    # def calculate_optimal_operation(self):
    #     """
    #     Solves the optimization model to determine the building's optimal energy operation strategy.

    #     This method creates an instance of the Pyomo model, solves it using the selected solver,
    #     and processes the results. It handles solver status checks, logs relevant information,
    #     and extracts the optimal power requirements and variable costs.

    #     Additionally, it invokes `write_additional_outputs` to process and store results from
    #     specific components like battery storage and electric vehicles.
    #     """
    #     # Create an instance of the model
    #     instance = self.model.create_instance()
    #     # Solve the instance using the configured solver
    #     results = self.solver.solve(instance, tee=False)

    #     # Check solver status and termination condition
    #     if (results.solver.status == SolverStatus.ok) and (
    #         results.solver.termination_condition == TerminationCondition.optimal
    #     ):
    #         logger.debug("The optimization model was solved optimally.")

    #         # Retrieve and log the objective function value
    #         objective_value = instance.obj_rule()
    #         logger.debug(f"The value of the objective function is {objective_value}.")

    #     elif results.solver.termination_condition == TerminationCondition.infeasible:
    #         logger.debug("The optimization model is infeasible.")

    #     else:
    #         logger.debug(f"Solver Status: {results.solver.status}")
    #         logger.debug(
    #             f"Termination Condition: {results.solver.termination_condition}"
    #         )

    #     # Extract and store the total power requirement as a Pandas Series
    #     self.opt_power_requirement = pd.Series(
    #         data=instance.totalvariable_power.get_values()
    #     ).set_axis(self.index)

    #     # Extract and store the variable cost series as a Pandas Series
    #     self.variable_cost_series = pd.Series(
    #         data=instance.variable_cost.get_values()
    #     ).set_axis(self.index)

    #     # Process additional outputs from specific components
    #     self.write_additional_outputs(instance)

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
