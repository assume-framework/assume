# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import pandas as pd
import pyomo.environ as pyo

from assume.common.base import SupportsMinMax
from assume.common.forecasts import Forecaster
from assume.units.dsm_load_shift import DSMFlex

SOLVERS = ["appsi_highs", "gurobi", "glpk", "cbc", "cplex"]

logger = logging.getLogger(__name__)

# Set the log level to ERROR
logging.getLogger("pyomo").setLevel(logging.WARNING)


class HydrogenPlant(DSMFlex, SupportsMinMax):
    """
    Represents a hydrogen plant in an energy system. This includes an electrolyser for hydrogen production and optional seasonal hydrogen storage.

    Args:
        id (str): Unique identifier for the hydrogen plant.
        unit_operator (str): The operator responsible for the plant.
        bidding_strategies (dict): A dictionary of bidding strategies that define how the plant participates in energy markets.
        forecaster (Forecaster): A forecaster used to get key variables such as fuel or electricity prices.
        technology (str, optional): The technology used by the plant. Default is "hydrogen_plant".
        components (dict, optional): A dictionary describing the components of the plant, such as electrolyser and hydrogen seasonal storage. Default is an empty dictionary.
        objective (str, optional): The objective function of the plant, typically to minimize variable costs. Default is "min_variable_cost".
        flexibility_measure (str, optional): The flexibility measure used for the plant, such as "max_load_shift". Default is "max_load_shift".
        demand (float, optional): The hydrogen production demand, representing how much hydrogen needs to be produced. Default is 0.
        cost_tolerance (float, optional): The maximum allowable increase in cost when shifting load. Default is 10.
        node (str, optional): The node location where the plant is connected within the energy network. Default is "node0".
        location (tuple[float, float], optional): The geographical coordinates (latitude, longitude) of the hydrogen plant. Default is (0.0, 0.0).
        **kwargs: Additional keyword arguments to support more specific configurations or parameters.

    Attributes:
        required_technologies (list): A list of required technologies for the plant, such as electrolyser.
        optional_technologies (list): A list of optional technologies, such as hydrogen seasonal storage.
    """

    # Required and optional technologies for the hydrogen plant
    required_technologies = ["electrolyser"]
    optional_technologies = ["hydrogen_seasonal_storage"]

    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        forecaster: Forecaster,
        components: dict[str, dict] = None,
        technology: str = "hydrogen_plant",
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

        # Initialize the model
        self.setup_model()

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

        if self.opt_power_requirement.at[start] > 0:
            marginal_cost = (
                self.variable_cost_series.at[start]
                / self.opt_power_requirement.at[start]
            )

        return marginal_cost
