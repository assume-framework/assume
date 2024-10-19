# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import ast
import logging

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import (
    SolverFactory,
    SolverStatus,
    TerminationCondition,
    check_available_solvers,
)
from distutils.util import strtobool

from assume.common.base import SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook
from assume.common.utils import get_products_index
from assume.units.dsm_load_shift import DSMFlex
from assume.units.dst_components import (
    create_boiler,
    create_ev,
    create_heatpump,
    create_thermal_storage,
    create_battery_storage,
    create_pv_plant
)

SOLVERS = ["gurobi", "glpk", "cbc", "cplex"]

logger = logging.getLogger(__name__)

# Mapping of component type identifiers to their respective classes
building_components = {
    "heatpump": create_heatpump,
    "boiler": create_boiler,
    "thermal_storage": create_thermal_storage,
    "ev": create_ev,
    "battery_storage": create_battery_storage,
    "pv_plant": create_pv_plant,
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
        self.electricity_price_sell = self.forecaster["price_EOM_sell"]
        self.natural_gas_price = self.forecaster["fuel_price_natural gas"]
        self.heat_demand = self.forecaster["heat_demand"]
        self.ev_load_profile = self.forecaster["ev_load_profile"]
        self.battery_load_profile = self.forecaster["battery_load_profile"]
        self.inflex_demand = self.forecaster[
            f"{self.id}_load_profile"
        ]
        self.pv_power_profile = self.forecaster[
            f"{self.id}_pv_power_profile"
        ]
        self.demand = demand
        self.flexibility_measure = flexibility_measure
        self.objective = objective

        self.has_heatpump = "heatpump" in components
        self.has_boiler = "boiler" in components
        self.has_thermal_storage = "thermal_storage" in components
        self.has_ev = "ev" in components
        self.has_battery_storage = "battery_storage" in components
        self.has_pv = "pv_plant" in components

        # Main Model part
        self.model = pyo.ConcreteModel()
        self.define_sets()
        self.define_parameters()

        # Check fuel type of boiler
        if self.has_boiler:
            if "fuel_type" in components["boiler"]:
                self.is_boiler_electric = True if components["boiler"]["fuel_type"] == "electric" else False

        # Create availability DataFrame for EVs
        # Parse the availability periods
        if self.has_ev:
            if "availability_periods" in components["ev"]:
                try:
                    # Convert the string to a list of tuples
                    components["ev"]["availability_periods"] = ast.literal_eval(
                        components["ev"]["availability_periods"]
                    )
                    components["ev"]["availability_df"] = self.create_availability_df(
                        components["ev"]["availability_periods"]
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
        if self.has_pv:
            pv_availability = self.forecaster["availability_Solar"]
            if pv_availability is not None:
                components["pv_plant"]["availability"] = pv_availability
            else:
                raise KeyError(
                    "Missing 'availability' of PV plants in availability file."
                )

        if self.has_battery_storage:
            sells_energy_input = components["battery_storage"].get("sells_energy_to_market")
            self.sells_battery_energy_to_market = bool(strtobool(sells_energy_input))

        self.define_variables()
        self.initialize_components(components)

        self.define_constraints()
        self.define_objective()

        self.initialize_process_sequence()

        solvers = check_available_solvers(*SOLVERS)
        if len(solvers) < 1:
            raise Exception(f"None of {SOLVERS} are available")
        self.solver = SolverFactory(solvers[0])

        self.opt_power_requirement = None
        self.variable_expenses_series = None

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
        if self.has_heatpump or self.has_boiler or self.has_thermal_storage:
            @self.model.Constraint(self.model.time_steps)
            def heat_flow_constraint(m, t):
                """
                Ensures the heat flow from the heat pump or electric boiler to the thermal storage or directly to the demand.
                """
                return (
                        (self.model.dsm_blocks["heatpump"].heat_out[t] if self.has_heatpump else 0)
                        + (self.model.dsm_blocks["boiler"].heat_out[t] if self.has_boiler else 0)
                        + (self.model.dsm_blocks["thermal_storage"].discharge_thermal[t] if self.has_thermal_storage else 0)
                        == self.model.heat_demand[t]
                        + (self.model.dsm_blocks["thermal_storage"].charge_thermal[t] if self.has_thermal_storage else 0)
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
        self.model.electricity_price_buy = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.electricity_price)},
        )
        self.model.electricity_price_sell = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.electricity_price_sell)},
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
        if self.has_ev:
            self.model.ev_load_profile = pyo.Param(
                self.model.time_steps,
                initialize={t: value for t, value in enumerate(self.ev_load_profile)},
            )
        if self.has_battery_storage:
            self.model.battery_load_profile = pyo.Param(
                self.model.time_steps,
                initialize={t: value for t, value in enumerate(self.battery_load_profile)},
            )
        if self.has_pv:
            self.model.pv_power_profile = pyo.Param(
                self.model.time_steps,
                initialize={t: value for t, value in enumerate(self.pv_power_profile)},
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
                    + (self.model.dsm_blocks["ev"].charge_ev[t] if self.has_ev else 0)
                    + (self.model.dsm_blocks["battery_storage"].charge[t] if self.has_battery_storage else 0)
                    - (self.model.dsm_blocks["pv_plant"].power_out[t] if self.has_pv else 0)
                    # TODO: My idea with sells_battery_energy_to_market is not working since there is no split up anymore!
                    - (self.model.dsm_blocks["battery_storage"].discharge[t] if self.has_battery_storage else 0)
            )

        @self.model.Constraint(self.model.time_steps)
        def variable_expenses_constraint(m, t):
            """
            Calculates the variable expense per time step.
            """
            return (
                # TODO: Also Idea not really possible to have two different prices for buying and selling (variable_power negtive or positive)
                    self.model.variable_expenses[t]
                    == self.model.variable_power[t]
                    * self.model.electricity_price_buy[t]
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

        elif self.objective == "maximize_revenue":

            @self.model.Objective(sense=pyo.minimize)
            def obj_rule(m):
                """
                Maximizes the total variable revenue over all time steps by minimizing the variable power.
                The lower the power, the more power got used for self-consumption and therefore the revenue gets maximized.
                """
                total_variable_revenue = sum(
                    self.model.variable_power[t] for t in self.model.time_steps
                )
                return total_variable_revenue
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    def calculate_optimal_operation_if_needed(self):
        if self.opt_power_requirement is None and self.variable_expenses_series is None and \
                self.objective in ["minimize_expenses", "maximize_revenue"]:
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
        power = instance.variable_power.get_values()
        self.opt_power_requirement = pd.Series(data=power)
        self.opt_power_requirement.index = self.index

        # Variable expense series
        expenses = instance.variable_expenses.get_values()
        self.variable_expenses_series = pd.Series(data=expenses)
        self.variable_expenses_series.index = self.index

        self.write_additional_outputs(instance)

    def write_additional_outputs(self, instance):
        if self.has_battery_storage:
            model_block = instance.dsm_blocks["battery_storage"]
            soc = pd.Series(
                data=model_block.soc.get_values()/model_block.max_capacity, dtype=float
            )
            soc.index = self.index
            self.outputs["soc"] = soc
        if self.has_ev:
            model_block = instance.dsm_blocks["ev"]
            ev_soc = pd.Series(
                data=model_block.ev_battery_soc.get_values()/model_block.max_capacity, index=self.index, dtype=object
            )
            ev_soc.index = self.index
            self.outputs["ev_soc"] = ev_soc
        if self.has_pv:
            pv_power = pd.Series(
                data=instance.dsm_blocks["pv_plant"].power_out.get_values(), index=self.index, dtype=object
            )
            pv_power.index = self.index
            self.outputs["pv_power"] = pv_power

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
            # TODO: For what is this needed??
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
        # Initialize marginal cost
        marginal_cost = 0

        if self.opt_power_requirement[start] != 0:
            marginal_cost = round(abs(
                self.variable_expenses_series[start] / self.opt_power_requirement[start]
            ),2)
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
