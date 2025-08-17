# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime

import pyomo.environ as pyo

from assume.common.base import SupportsMinMax
from assume.common.forecasts import Forecaster
from assume.units.dsm_load_shift import DSMFlex

logger = logging.getLogger(__name__)


class CementPlant(DSMFlex, SupportsMinMax):
    """
    The CementPlant class represents a cement plant unit in the energy system.

    Args:
        id (str): The unique identifier of the unit.
        unit_operator (str): The operator of the unit.
        bidding_strategies (dict): The bidding strategies of the unit.
        technology (str): The technology of the unit.
        node (str): The node of the unit.
        location (tuple[float, float]): The location of the unit.
        components (dict[str, dict]): The components of the unit such as raw_material_mill, clinker_system, cement_mill, and electrolyser.
        objective (str): The objective of the unit, e.g. minimize variable cost ("min_variable_cost").
        flexibility_measure (str): The flexibility measure of the unit, e.g. maximum load shift ("max_load_shift").
        demand (float): The demand of the unit - the amount of cement to be produced.
        cost_tolerance (float): The cost tolerance of the unit - the maximum cost that can be tolerated when shifting the load.
    """

    required_technologies = []
    optional_technologies = [
        "preheater",
        "cement_mill",
        "electrolyser",
    ]

    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        forecaster: Forecaster,
        technology: str = "cement_plant",
        node: str = "node0",
        location: tuple[float, float] = (0.0, 0.0),
        components: dict[str, dict] = None,
        objective: str = None,
        flexibility_measure: str = "",
        peak_load_cap: float = 0,
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
                    f"Component {component} is required for the cement plant unit."
                )

        # check if the provided components are valid and do not contain any unknown components
        for component in components.keys():
            if (
                component not in self.required_technologies
                and component not in self.optional_technologies
            ):
                raise ValueError(
                    f"Components {component} is not a valid component for the cement plant unit."
                )
        # Check the presence of components first
        self.has_preheater = "preheater" in self.components.keys()
        self.has_cement_mill = "cement_mill" in self.components.keys()
        self.has_electrolyser = "electrolyser" in self.components.keys()
        self.has_thermal_storage = "thermal_storage" in self.components.keys()

        # Inject schedule into long-term thermal storage if applicable
        if "thermal_storage" in self.components:
            storage_cfg = self.components["thermal_storage"]
            storage_type = storage_cfg.get("storage_type", "short-term")
            if storage_type == "long-term":
                schedule_key = f"{self.id}_thermal_storage_schedule"
                schedule_series = self.forecaster[schedule_key]
                storage_cfg["storage_schedule_profile"] = schedule_series

        # Add forecasts
        self.natural_gas_price = self.forecaster["natural_gas_price"]
        self.cement_demand_per_time_step = self.forecaster[f"{self.id}_cement_demand"]
        self.hydrogen_price = self.forecaster["price_hydrogen"]
        self.coal_price = self.forecaster["coal_price"]
        self.electricity_price = self.forecaster["electricity_price"]
        self.electricity_price_flex = self.forecaster["electricity_price_flex"]
        self.lime_price = self.forecaster.get_price("lime")
        self.co2_price = self.forecaster.get_price("co2")
        self.demand = demand
        self.objective = objective
        self.flexibility_measure = flexibility_measure
        self.cost_tolerance = cost_tolerance
        self.peak_load_cap = peak_load_cap

        self.optimisation_counter = 0

       # Initialize the model
        self.setup_model()

    def define_parameters(self):
        self.model.electricity_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.electricity_price)},
        )
        self.model.natural_gas_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.natural_gas_price)},
        )
        self.model.coal_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.coal_price)},
        )
        self.model.hydrogen_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.hydrogen_price)},
        )
        # self.model.lime_price = pyo.Param(
        #     self.model.time_steps,
        #     initialize={t: value for t, value in enumerate(self.lime_price)},
        #     within=pyo.NonNegativeReals,
        # )
        self.model.co2_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.co2_price)},
        )
        self.model.absolute_cement_demand = pyo.Param(initialize=self.demand)
        self.model.cement_demand_per_time_step = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.cement_demand_per_time_step)},
        )

    def define_variables(self):
        self.model.total_power_input = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.cumulative_thermal_output = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.variable_cost = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )

    def initialize_process_sequence(self): # need to be developed
        """
        Initializes the process sequence for the cement plant based on available components.
        """
        if not self.demand or self.demand == 0:
            def heat_balance(m, t):
                heat_production = 0
                heat_production += m.dsm_blocks["preheater"].heat_need[t]
                return heat_production == m.cumulative_thermal_output[t]

    def define_constraints(self):
        """
        Defines the constraints for the paper and pulp plant model.
        """

        @self.model.Constraint(self.model.time_steps)
        def absolute_demand_association_constraint(m, t):
            """
            Ensures the thermal output meets the absolute demand.
            """
            if not self.demand or self.demand == 0:
                    return m.cumulative_thermal_output[t] >= m.cement_demand_per_time_step[t]
            else:
                return (
                    sum((m.cumulative_thermal_output[t]) for t in m.time_steps)
                    >= m.absolute_cement_demand
                )

        # Power input constraint
        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
            """
            Calculates total electrical power input from components.
            """
            total_power = 0

            if self.has_preheater:
                total_power += self.model.dsm_blocks["preheater"].power_in[t]

            return m.total_power_input[t] == total_power

        # Operating cost constraint
        @self.model.Constraint(self.model.time_steps)
        def cost_per_time_step(m, t):
            """
            Calculates the variable cost per time step.
            """
            variable_cost = 0
            if self.has_cement_mill:
                variable_cost += self.model.dsm_blocks["cement_mill"].operating_cost[t]
            if self.has_preheater:
                variable_cost += self.model.dsm_blocks["preheater"].operating_cost[
                    t
                ]
            if self.has_electrolyser:
                variable_cost += self.model.dsm_blocks["electrolyser"].operating_cost[t]
            return self.model.variable_cost[t] == variable_cost

    def calculate_marginal_cost(self, start: datetime, power: float) -> float:
        """
        Calculate the marginal cost of the unit based on the provided time and power.

        Args:
            start (datetime): The start time of the dispatch
            power (float): The power output of the unit

        Returns:
            float: The marginal cost of the unit
        """
        marginal_cost = 0

        if self.opt_power_requirement.at[start] > 0:
            marginal_cost = (
                self.variable_cost_series.at[start]
                / self.opt_power_requirement.at[start]
            )

        return marginal_cost