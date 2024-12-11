# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import ast
import logging
from distutils.util import strtobool

import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import (
    SolverFactory,
    SolverStatus,
    TerminationCondition,
    check_available_solvers,
)

from assume.common.base import SupportsMinMax
from assume.common.fast_pandas import FastSeries
from assume.common.market_objects import MarketConfig, Orderbook
from assume.common.utils import get_products_index
from assume.strategies import NaiveDADSMStrategy
from assume.units.dsm_load_shift import DSMFlex

SOLVERS = ["gurobi", "glpk", "cbc", "cplex"]

logger = logging.getLogger(__name__)


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
        objective (str): The objective of the unit, e.g. minimize expenses ("minimize_expenses").
    """

    # There are no mandatory components for the building unit since it can also be a pure demand unit.
    optional_technologies = ["heatpump", "boiler", "thermal_storage", "electric_vehicle", "generic_storage", "pv_plant"]

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
            components=components,
            **kwargs,
        )

        # check if the provided components are valid and do not contain any unknown components
        for component in self.components.keys():
            if (
                    component not in self.optional_technologies
            ):
                raise ValueError(
                    f"Component {component} is not a valid component for the building unit."
                )

        #JUST FOR MY SIMULATIONS/EXPERIMENTS!!
        #self.electricity_price = self.create_price_given_solar_forecast()
        self.electricity_price = self.create_price_given_grid_load_forecast()
        self.natural_gas_price = self.forecaster["fuel_price_natural gas"]
        self.heat_demand = self.forecaster["heat_demand"]
        self.ev_load_profile = self.forecaster["ev_load_profile"]
        self.battery_load_profile = self.forecaster["battery_load_profile"]
        self.inflex_demand = self.forecaster[
            f"{self.id}_load_profile"
        ]

        self.demand = demand
        self.flexibility_measure = flexibility_measure
        self.objective = objective

        # Main Model part
        self.model = pyo.ConcreteModel()
        self.define_sets()

        self.has_heatpump = "heatpump" in self.components
        self.has_boiler = "boiler" in self.components
        self.has_thermal_storage = "thermal_storage" in self.components
        self.has_ev = "electric_vehicle" in self.components
        self.has_battery_storage = "generic_storage" in self.components
        self.has_pv = "pv_plant" in self.components

        self.define_parameters()

        # Create availability DataFrame for EVs
        # Parse the availability periods
        if self.has_ev:
            if "availability_periods" in self.components["electric_vehicle"]:
                try:
                    # Convert the string to a list of tuples
                    self.components["electric_vehicle"]["availability_periods"] = ast.literal_eval(
                        self.components["electric_vehicle"]["availability_periods"]
                    )
                    self.components["electric_vehicle"]["availability_df"] = self.create_availability_df(
                        self.components["electric_vehicle"]["availability_periods"]
                    )
                except Exception as e:
                    raise ValueError(
                        f"Error processing availability periods for EV: {e}"
                    )
            else:
                raise KeyError(
                    "Missing 'availability_periods' in EV components configuration."
                )

        # Parse the availability of the PV plant
        self.pv_max_power = 0
        if self.has_pv:
            self.pv_uses_power_profile = False
            if not strtobool(self.components["pv_plant"].get("uses_power_profile", "no")):
                pv_availability = pd.Series(self.forecaster["availability_Solar"])
                pv_availability.index = self.model.time_steps
                self.components["pv_plant"]["availability_profile"] = pv_availability
                self.pv_max_power = self.components.get("pv_plant", {}).get("max_power", 0)
            else:
                pv_power = pd.Series(self.forecaster[f"{self.id}_pv_power_profile"])
                pv_power.index = self.model.time_steps
                self.components["pv_plant"]["power_profile"] = pv_power
                self.pv_uses_power_profile = True
                self.pv_max_power = max(pv_power)

        self.define_variables()

        #############################
        # Section for storage units #
        #############################
        if not isinstance(self.bidding_strategies.get("EOM", ""), NaiveDADSMStrategy):
            self.max_power_discharge = abs(self.components.get("generic_storage", {}).get("max_power_discharge", 0))
            self.max_power_charge = -abs(self.components.get("generic_storage", {}).get("max_power_charge", 0))
            self.max_capacity = self.components.get("generic_storage", {}).get("max_capacity", 0)
            self.min_capacity = self.components.get("generic_storage", {}).get("min_capacity", 0)
            self.efficiency_charge = self.components.get("generic_storage", {}).get("efficiency_charge", 1)
            self.efficiency_discharge = self.components.get("generic_storage", {}).get("efficiency_discharge", 1)
            self.initial_soc = self.components.get("generic_storage", {}).get("initial_soc", 1e-6) * self.max_capacity
            self.outputs["soc"] = FastSeries(value=self.initial_soc, index=self.index)
            self.outputs["energy_cost"] = FastSeries(value=0.0, index=self.index)
            self.pv_production = FastSeries(value=0.0, index=self.index)
            self.battery_charge = FastSeries(value=0.0, index=self.index)
            self.pv_availability = FastSeries(
                value=0.0 if not self.has_pv or self.pv_max_power == 0 else self.components["pv_plant"][
                    "power_profile"].values, index=self.index)
        # End section for storage units #
        else:
            self.initialize_components()
            self.define_constraints()
            self.define_objective()
            self.initialize_process_sequence()

        solvers = check_available_solvers(*SOLVERS)
        if len(solvers) < 1:
            raise Exception(f"None of {SOLVERS} are available")
        self.solver = SolverFactory(solvers[0])

        self.opt_power_requirement = None
        self.variable_expenses_series = None

    def create_price_given_solar_forecast(self):
        buy_forecast = self.forecaster["price_EOM"]
        sell_forecast = self.forecaster["price_EOM_sell"]

        price_delta = (buy_forecast - sell_forecast) * (self.forecaster["availability_Solar"])
        return FastSeries(value=round(buy_forecast - price_delta, 2), index=sell_forecast.index)

    def create_price_given_grid_load_forecast(self):
        buy_forecast = self.forecaster["price_EOM"]
        sell_forecast = self.forecaster["price_EOM_sell"]
        community_load = self.forecaster["total_community_load"]
        scaling_factor = max(abs(min(community_load)), max(community_load))

        # Normalization
        community_load_scaled = community_load / scaling_factor
        price_delta = (buy_forecast - sell_forecast) / 2
        mid_price = buy_forecast - price_delta
        return FastSeries(value=np.round(mid_price + community_load_scaled * price_delta, 5), index=sell_forecast.index)

    def create_availability_df(self, availability_periods):
        """
        Create an availability DataFrame based on the provided availability periods.

        Args:
            availability_periods (list of tuples): List of (start, end) tuples for availability periods.

        Returns:
            pd.Series: A series with 1 for available time steps and 0 otherwise.
        """
        availability_series = pd.Series(0, index=self.index)

        for start, end in availability_periods:
            availability_series[start:end] = 1

        return availability_series

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
                        (self.model.dsm_blocks["heatpump"].heat_out[t] if self.has_heatpump else 0)
                        + (self.model.dsm_blocks["boiler"].heat_out[t] if self.has_boiler else 0)
                        + (self.model.dsm_blocks["thermal_storage"].discharge_thermal[
                               t] if self.has_thermal_storage else 0)
                        == self.model.heat_demand[t]
                        + (self.model.dsm_blocks["thermal_storage"].charge_thermal[
                               t] if self.has_thermal_storage else 0)
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
        self.model.inflex_demand = pyo.Param(
            self.model.time_steps,
            initialize={
                t: value for t, value in enumerate(self.inflex_demand)
            },
        )

    def define_variables(self):
        """
        Defines the variables for the Pyomo model.
        """
        self.model.variable_power = pyo.Var(
            self.model.time_steps, within=pyo.Reals
        )
        self.model.variable_expenses = pyo.Var(
            self.model.time_steps, within=pyo.Reals
        )

    def define_constraints(self):

        @self.model.Constraint(self.model.time_steps)
        def total_power_constraint(m, t):
            """
            Ensures the total power input is the sum of power inputs of all components subtracted by the self
            produced/stored energy.
            """
            return (
                    self.model.variable_power[t]
                    ==
                    + self.model.inflex_demand[t]
                    + (self.model.dsm_blocks["heatpump"].power_in[t] if self.has_heatpump else 0)
                    + (self.model.dsm_blocks["boiler"].power_in[t] if self.has_boiler else 0)
                    + (self.model.dsm_blocks["electric_vehicle"].charge[t] if self.has_ev else 0)
                    + (self.model.dsm_blocks["generic_storage"].charge[t] if self.has_battery_storage else 0)
                    - (self.model.dsm_blocks["pv_plant"].power[t] if self.has_pv else 0)
                    - (self.model.dsm_blocks["generic_storage"].discharge[t] if self.has_battery_storage else 0)
                    - (self.model.dsm_blocks["electric_vehicle"].discharge[t] if self.has_ev else 0)
            )

        @self.model.Constraint(self.model.time_steps)
        def variable_expenses_constraint(m, t):
            """
            Calculates the variable expense per time step.
            """
            return (
                    self.model.variable_expenses[t]
                    == self.model.variable_power[t]
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
                    self.model.variable_expenses[t] for t in self.model.time_steps
                )
                return total_variable_cost
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    def calculate_optimal_operation_if_needed(self):
        if self.opt_power_requirement is None and self.variable_expenses_series is None and \
                self.objective in ["minimize_expenses"]:
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
        power = list(instance.variable_power.get_values().values())
        self.opt_power_requirement = FastSeries(value=power, index=self.index)

        # Variable expense series
        expenses = list(instance.variable_expenses.get_values().values())
        self.variable_expenses_series = FastSeries(value=expenses, index=self.index)

        self.write_additional_outputs(instance)

    def write_additional_outputs(self, instance):
        if not isinstance(self.bidding_strategies.get("EOM", ""), NaiveDADSMStrategy):
            return
        if self.has_battery_storage:
            model_block = instance.dsm_blocks["generic_storage"]
            soc = FastSeries(
                value=list(model_block.soc.get_values()), index=self.index
            ) / pyo.value(model_block.max_capacity)
            self.outputs["soc"] = soc
        if self.has_ev:
            model_block = instance.dsm_blocks["electric_vehicle"]
            ev_soc = FastSeries(
                value=list(model_block.ev_battery_soc.get_values()), index=self.index
            ) / pyo.value(model_block.max_capacity)
            self.outputs["ev_soc"] = ev_soc

    def execute_current_dispatch(self, start: pd.Timestamp, end: pd.Timestamp):
        """
        Executes the current dispatch of the unit based on the provided timestamps.

        The dispatch is only executed, if it is in the constraints given by the unit.
        Returns the volume of the unit within the given time range.

        Args:
            start (pandas.Timestamp): The start time of the dispatch.
            end (pandas.Timestamp): The end time of the dispatch.

        Returns:
            pd.Series: The volume of the unit within the given time range.
        """
        if isinstance(self.bidding_strategies.get("EOM", ""), NaiveDADSMStrategy):
            return super().execute_current_dispatch(start, end)

        t = end - self.index.freq
        inflex_demand = self.inflex_demand[t]
        pv_power = self.pv_production[t]
        delta_soc = 0
        soc = self.outputs["soc"][t]
        if self.battery_charge[t] > self.max_power_discharge:
            self.battery_charge.at[t] = self.max_power_discharge
        elif self.battery_charge[t] < self.max_power_charge:
            self.battery_charge.at[t] = self.max_power_charge

        # discharging
        if self.battery_charge[t] > 0:
            max_soc_discharge = self.calculate_soc_max_discharge(soc)

            if self.battery_charge[t] > max_soc_discharge:
                self.battery_charge.at[t] = max_soc_discharge

            delta_soc = (
                    -self.battery_charge[t] / self.efficiency_discharge
            )

        # charging
        elif self.battery_charge[t] < 0:
            max_soc_charge = self.calculate_soc_max_charge(soc)

            if self.battery_charge[t] < max_soc_charge:
                self.battery_charge.at[t] = max_soc_charge

            delta_soc = (
                    -self.battery_charge[t] * self.efficiency_charge
            )
        # Update the energy with the new values from battery_power
        self.outputs["energy"].at[t] = self.battery_charge[t] + pv_power - inflex_demand

        self.outputs["soc"].at[t + self.index.freq] = round(soc + delta_soc, 4)

        return self.outputs["energy"][t]

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
                self.outputs["energy"].loc[start] += order["accepted_volume"]
        self.calculate_cashflow("energy", orderbook)

        if not isinstance(self.bidding_strategies.get("EOM", ""), NaiveDADSMStrategy):
            for start in products_index:
                delta_soc = 0
                soc = self.outputs["soc"][start]
                inflex_demand = self.inflex_demand[start]
                pv_power = self.pv_production[start]

                # discharging
                if self.battery_charge[start] > 0:
                    max_soc_discharge = self.calculate_soc_max_discharge(soc)

                    if self.battery_charge[start] > max_soc_discharge:
                        self.battery_charge.at[start] = max_soc_discharge

                    delta_soc = (
                            -self.battery_charge[start]
                            / self.efficiency_discharge
                    )

                # charging
                elif self.battery_charge[start] < 0:
                    max_soc_charge = self.calculate_soc_max_charge(soc)

                    if self.battery_charge[start] < max_soc_charge:
                        self.battery_charge.at[start] = max_soc_charge

                    delta_soc = (
                            -self.battery_charge[start] * self.efficiency_charge
                    )

                self.outputs["soc"].at[start + self.index.freq:] = round(soc + delta_soc, 4)
                # Update the energy with the new values from battery_power
                self.outputs["energy"].at[start] = self.battery_charge[start] + pv_power - inflex_demand
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

        if self.opt_power_requirement is not None and self.opt_power_requirement[start] != 0:
            marginal_cost = round(abs(
                self.variable_expenses_series[start] / self.opt_power_requirement[start]
            ), 5)
        return marginal_cost

    def calculate_max_discharge(self, start: pd.Timestamp) -> float:
        """
        Calculates the maximum discharging power for the given time period.

        Args:
            start:
            end:

        Returns:

        """
        if not self.has_battery_storage:
            return 0

        max_power_discharge = self.max_power_discharge
        # restrict according to min_soc
        max_soc_discharge = self.calculate_soc_max_discharge(self.outputs["soc"][start])
        max_power_discharge = min(max_power_discharge, max_soc_discharge)
        return max_power_discharge

    def calculate_max_charge(self, start: pd.Timestamp) -> float:
        """
        Calculates the maximum discharging power for the given time period.

        Args:
            start:
            end:

        Returns:

        """
        if not self.has_battery_storage:
            return 0

        max_power_charge = self.max_power_charge
        # restrict charging according to max_soc
        max_soc_charge = self.calculate_soc_max_charge(self.outputs["soc"][start])
        max_power_charge = max(max_power_charge, max_soc_charge)
        return max_power_charge

    def calculate_pv_power(self, start: pd.Timestamp) -> float:
        current_power = 0

        if self.has_pv and self.pv_uses_power_profile:
            power = self.components["pv_plant"].get("power_profile", 0)
            if isinstance(power, pd.Series):
                power.index = self.index[self.index.start:self.index.end]
                current_power = power[start]
        elif self.has_pv:
            av_solar = self.components["pv_plant"].get("availability_profile", 0)
            if isinstance(av_solar, pd.Series):
                av_solar.index = self.index[self.index.start:self.index.end]
                current_power = av_solar[start] * self.pv_max_power

        self.pv_production.at[start] = current_power
        return current_power

    def calculate_soc_max_charge(
            self,
            soc,
    ) -> float:
        """
        Calculates the maximum charge power depending on the current state of charge.

        Args:
            soc (float): The current state of charge.

        Returns:
            float: The maximum charge power.
        """
        power = min(
            0,
            round((soc - self.max_capacity) / self.efficiency_charge, 6),
        )
        return power

    def calculate_soc_max_discharge(self, soc) -> float:
        """
        Calculates the maximum discharge power depending on the current state of charge.

        Args:
            soc (float): The current state of charge.

        Returns:
            float: The maximum discharge power.
        """
        power = max(
            0,
            round((soc - self.min_capacity) * self.efficiency_discharge, 6),
        )
        return power

    def as_dict(self) -> dict:
        """
        Returns the attributes of the unit as a dictionary, including specific attributes.

        Returns:
            dict: The attributes of the unit as a dictionary.
        """
        components_list = [component for component in self.components.keys()]
        components_string = ",".join(components_list)

        unit_dict = super().as_dict()
        unit_dict.update(
            {
                "unit_type": "demand",
                "components": components_string,
            }
        )

        return unit_dict
