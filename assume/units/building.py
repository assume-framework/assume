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
from assume.units.dsm_load_shift import DSMFlex

SOLVERS = ["appsi_highs", "gurobi", "glpk", "cbc", "cplex"]

logger = logging.getLogger(__name__)

# Set the log level to ERROR
logging.getLogger("pyomo").setLevel(logging.WARNING)


class Building(DSMFlex, SupportsMinMax):
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
        objective (str): The objective of the unit, e.g. minimize expenses ("min_variable_cost").
    """

    # There are no mandatory components for the building unit since it can also be a pure demand unit.
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

        # check if the provided components are valid and do not contain any unknown components
        for component in self.components.keys():
            if component not in self.optional_technologies:
                raise ValueError(
                    f"Component {component} is not a valid component for the building unit."
                )

        self.electricity_price = self.forecaster["price_EOM"]
        self.natural_gas_price = self.forecaster["fuel_price_natural gas"]
        self.heat_demand = self.forecaster["heat_demand"]
        self.ev_load_profile = self.forecaster["ev_load_profile"]
        self.battery_load_profile = self.forecaster["battery_load_profile"]
        self.inflex_demand = self.forecaster[f"{self.id}_load_profile"]

        self.objective = objective
        self.flexibility_measure = flexibility_measure

        # Check for the presence of components
        self.has_heatpump = "heat_pump" in self.components
        self.has_boiler = "boiler" in self.components
        self.has_thermal_storage = "thermal_storage" in self.components
        self.has_ev = "electric_vehicle" in self.components
        self.has_battery_storage = "generic_storage" in self.components
        self.has_pv = "pv_plant" in self.components

        if self.has_ev:
            self.ev_sells_energy_to_market = self.components["electric_vehicle"].get(
                "sells_energy_to_market", "true"
            ).lower() in {"y", "yes", "t", "true", "on", "1"}

        if self.has_battery_storage:
            self.battery_sells_energy_to_market = self.components[
                "generic_storage"
            ].get("sells_energy_to_market", "true").lower() in {
                "y",
                "yes",
                "t",
                "true",
                "on",
                "1",
            }

        # Parse the availability of the PV plant
        if self.has_pv:
            uses_power_profile = self.components["pv_plant"].get(
                "uses_power_profile", "false"
            ).lower() in {"y", "yes", "t", "true", "on", "1"}
            profile_key = (
                f"{self.id}_pv_power_profile"
                if not uses_power_profile
                else "availability_solar"
            )
            self.components["pv_plant"][
                "power_profile" if not uses_power_profile else "availability_profile"
            ] = self.forecaster[profile_key]

        # Main Model part
        self.model = pyo.ConcreteModel()
        self.define_sets()
        self.define_parameters()
        self.define_variables()

        self.initialize_components()
        self.initialize_process_sequence()

        self.define_constraints()
        self.define_objective()

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
        self.model.inflex_demand = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.inflex_demand)},
        )

    def define_variables(self):
        """
        Defines the variables for the Pyomo model.
        """
        self.model.variable_power = pyo.Var(self.model.time_steps, within=pyo.Reals)
        self.model.variable_cost = pyo.Var(self.model.time_steps, within=pyo.Reals)

    def initialize_process_sequence(self):
        """
        Initializes the process sequence and constraints for the building.
        """
        if self.has_heatpump or self.has_boiler or self.has_thermal_storage:

            @self.model.Constraint(self.model.time_steps)
            def heat_flow_constraint(m, t):
                """
                Ensures the heat flow from the heat pump or electric boiler to the thermal storage or directly to the demand.
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
        @self.model.Constraint(self.model.time_steps)
        def variable_power_constraint(m, t):
            """
            Ensures the total power input is the sum of power inputs of all components subtracted by the self
            produced/stored energy.
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
                Restricts the discharging rate of the electric vehicle for self usage only.
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
                Restricts the discharging rate of the battery storage for self usage only.
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
            Calculates the variable expense per time step.
            """
            return (
                self.model.variable_cost[t]
                == self.model.variable_power[t] * self.model.electricity_price[t]
            )

    def define_objective(self):
        """
        Defines the objective for the optimization model.
        """
        if self.objective == "min_variable_cost":

            @self.model.Objective(sense=pyo.minimize)
            def obj_rule(m):
                """
                Minimizes the total variable cost over all time steps.
                """
                total_variable_cost = pyo.quicksum(
                    self.model.variable_cost[t] for t in self.model.time_steps
                )
                return total_variable_cost
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    def calculate_optimal_operation_if_needed(self):
        if self.opt_power_requirement is None and self.objective == "min_variable_cost":
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

        # Total power series
        self.opt_power_requirement = pd.Series(
            data=instance.variable_power.get_values()
        ).set_axis(self.index)

        # Variable expense series
        self.variable_cost_series = pd.Series(
            data=instance.variable_cost.get_values()
        ).set_axis(self.index)

        self.write_additional_outputs(instance)

    def write_additional_outputs(self, instance):
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
        Calculate the marginal cost of the unit based on the provided time and power.

        Args:
            start (pandas.Timestamp): The start time of the dispatch.
            power (float): The power output of the unit.

        Returns:
            float: the marginal cost of the unit for the given power.
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
