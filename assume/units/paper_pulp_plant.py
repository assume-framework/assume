# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime

import pyomo.environ as pyo

import logging
from assume.common.base import SupportsMinMax
from assume.common.forecasts import Forecaster
from assume.units.dsm_load_shift import DSMFlex
from assume.units.dst_components import HeatPump, Boiler, GenericStorage

logger = logging.getLogger(__name__)

class PaperPulpPlant(DSMFlex, SupportsMinMax):
    """
    Represents a paper and pulp plant in an energy system. This includes components like heat pumps,
    boilers, and storage units for operational optimization.

    Args:
        id (str): Unique identifier for the paper and pulp plant.
        unit_operator (str): The operator responsible for the plant.
        bidding_strategies (dict): A dictionary of bidding strategies that define how the plant participates in energy markets.
        forecaster (Forecaster): A forecaster used to get key variables such as fuel or electricity prices.
        components (dict, optional): A dictionary describing the components of the plant, such as heat pumps and boilers.
        objective (str, optional): The objective function of the plant, typically to minimize variable costs. Default is "min_variable_cost".
        flexibility_measure (str, optional): The flexibility measure used for the plant, such as "max_load_shift". Default is "max_load_shift".
        demand (float, optional): The production demand, representing how much product needs to be produced. Default is 0.
        cost_tolerance (float, optional): The maximum allowable increase in cost when shifting load. Default is 10.
        node (str, optional): The node location where the plant is connected within the energy network. Default is "node0".
        location (tuple[float, float], optional): The geographical coordinates (latitude, longitude) of the paper and pulp plant. Default is (0.0, 0.0).
        **kwargs: Additional keyword arguments to support more specific configurations or parameters.
    """

    required_technologies = ["heat_pump", "boiler"]
    optional_technologies = ["thermal_storage"]

    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        forecaster: Forecaster,
        components: dict[str, dict] = None,
        technology: str = "paper_pulp_plant",
        objective: str = "min_variable_cost",
        flexibility_measure: str = "max_load_shift",
        demand: float = 0,
        cost_tolerance: float = 10,
        node: str = "node0",
        location: tuple[float, float] = (0.0, 0.0),
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

        # Check if the required components are present in the components dictionary
        for component in self.required_technologies:
            if component not in components.keys():
                raise ValueError(
                    f"Component {component} is required for the paper and pulp plant unit."
                )

        # Check if the provided components are valid and do not contain any unknown components
        for component in components.keys():
            if (
                component not in self.required_technologies
                and component not in self.optional_technologies
            ):
                raise ValueError(
                    f"Component {component} is not a valid component for the paper and pulp plant unit."
                )
            
        # Inject schedule into long-term thermal storage if applicable
        if "thermal_storage" in self.components:
            storage_cfg = self.components["thermal_storage"]
            storage_type = storage_cfg.get("storage_type", "short-term")
            if storage_type == "long-term":
                schedule_key = f"{self.id}_thermal_storage_schedule"
                schedule_series = self.forecaster[schedule_key]
                storage_cfg["storage_schedule_profile"] = schedule_series

        # Add price forecasts
        self.electricity_price = self.forecaster["electricity_price"]
        if self.has_boiler and self.components["boiler"]["fuel_type"] == "natural_gas":
            self.natural_gas_price = self.forecaster["natural_gas_price"]
        self.thermal_demand = self.forecaster[f"{self.id}_thermal_demand"]
        self.objective = objective
        self.flexibility_measure = flexibility_measure
        self.cost_tolerance = cost_tolerance

        # Check for the presence of components
        self.has_thermal_storage = (
            "thermal_storage" in self.components.keys()
        )
        self.has_boiler = "boiler" in self.components.keys()
        self.has_heat_pump = "heat_pump" in self.components.keys()


        # Initialize the model
        self.setup_model()

    def define_parameters(self):
        """
        Defines the parameters for the Pyomo model.
        """
        self.model.electricity_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.electricity_price)},
        )
        
        if self.has_boiler and self.components["boiler"]["fuel_type"] == "natural_gas":
            self.model.natural_gas_price = pyo.Param(
                self.model.time_steps,
                initialize={t: value for t, value in enumerate(self.natural_gas_price)},
            )
        
        self.model.thermal_demand = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.thermal_demand)},
        )

    def define_variables(self):
        """
        Defines the decision variables for the optimization model.
        """
        self.model.power_input = pyo.Var(self.model.time_steps, within=pyo.NonNegativeReals)
        self.model.heat_output = pyo.Var(self.model.time_steps, within=pyo.NonNegativeReals)
        self.model.variable_cost = pyo.Var(self.model.time_steps, within=pyo.NonNegativeReals)

    def initialize_process_sequence(self):
        """
        Initializes the process sequence and constraints for the paper and pulp plant.
        Connects heat sources (heat pump and/or boiler) with thermal storage if present.
        """
        @self.model.Constraint(self.model.time_steps)
        def heat_production_distribution(m, t):
            """
            Balances heat produced by available sources with storage operations.
            Heat flows from sources to storage or demand.
            """
            total_heat_production = 0
            
            # Add heat production from available sources
            if self.has_heat_pump:
                total_heat_production += self.model.dsm_blocks["heat_pump"].heat_out[t]
                
            if self.has_boiler:
                total_heat_production += self.model.dsm_blocks["boiler"].heat_out[t]
            
            # Connect with storage if present
            if self.has_thermal_storage:
                storage_discharge = self.model.dsm_blocks["thermal_storage"].discharge[t]
                storage_charge = self.model.dsm_blocks["thermal_storage"].charge[t]
                return total_heat_production + storage_discharge == m.heat_output[t] + storage_charge
            else:
                return total_heat_production == m.heat_output[t]

    def define_constraints(self):
        """
        Defines the constraints for the paper and pulp plant model.
        """
        # Heat demand constraint
        @self.model.Constraint(self.model.time_steps)
        def total_heat_demand_constraint(m, t):
            """
            Ensures total heat production meets demand over optimization horizon.
            """
            return m.heat_output[t] == m.thermal_demand[t]

        # Power input constraint
        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
            """
            Calculates total electrical power input from components.
            """
            total_power = 0
            
            if self.has_heat_pump:
                total_power += self.model.dsm_blocks["heat_pump"].power_in[t]
            
            if self.has_boiler and self.components["boiler"]["fuel_type"] == "electricity":
                total_power += self.model.dsm_blocks["boiler"].power_in[t]
                
            return m.power_input[t] == total_power
        
        # Operating cost constraint
        @self.model.Constraint(self.model.time_steps)
        def cost_per_time_step(m, t):
            """
            Calculates total operating cost from all components.
            """
            total_cost = 0
            
            if self.has_heat_pump:
                total_cost += self.model.dsm_blocks["heat_pump"].operating_cost[t]
                
            if self.has_boiler:
                total_cost += self.model.dsm_blocks["boiler"].operating_cost[t]

                
            return m.variable_cost[t] == total_cost

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
