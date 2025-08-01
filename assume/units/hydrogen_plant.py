# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import pyomo.environ as pyo

from assume.common.base import SupportsMinMax
from assume.common.forecasts import Forecaster
from assume.units.dsm_load_shift import DSMFlex

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
        flexibility_measure (str, optional): The flexibility measure used for the plant, such as "cost_based_load_shift". Default is "cost_based_load_shift".
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
        flexibility_measure: str = "cost_based_load_shift",
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
        self.hydrogen_demand = self.forecaster["hydrogen_demand"]
        self.demand = demand

        self.objective = objective
        self.flexibility_measure = flexibility_measure
        self.cost_tolerance = cost_tolerance

        # Check for the presence of components
        self.has_h2seasonal_storage = (
            "hydrogen_seasonal_storage" in self.components.keys()
        )
        self.has_electrolyser = "electrolyser" in self.components.keys()

        # Inject schedule into long-term seasonal storage if applicable
        if "hydrogen_seasonal_storage" in self.components:
            storage_cfg = self.components["hydrogen_seasonal_storage"]
            storage_type = storage_cfg.get("storage_type", "short-term")
            if storage_type == "long-term":
                schedule_key = f"{self.id}_hydrogen_seasonal_storage_schedule"
                schedule_series = self.forecaster[schedule_key]
                storage_cfg["storage_schedule_profile"] = schedule_series

        # Initialize the model
        self.setup_model()

    def define_parameters(self):
        self.model.electricity_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.electricity_price)},
        )
        self.model.absolute_hydrogen_demand = pyo.Param(initialize=self.demand)
        self.model.hydrogen_demand_per_timestep = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.hydrogen_demand)},
        )

    def define_variables(self):
        self.model.total_power_input = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.variable_cost = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.cumulative_hydrogen_output = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )

    def initialize_process_sequence(self):
        # Per-time-step constraint (default)
        if not self.demand or self.demand == 0:

            @self.model.Constraint(self.model.time_steps)
            def direct_hydrogen_balance(m, t):
                total_hydrogen_production = 0
                if self.has_electrolyser:
                    total_hydrogen_production += m.dsm_blocks[
                        "electrolyser"
                    ].hydrogen_out[t]
                if self.has_h2seasonal_storage:
                    storage_discharge = m.dsm_blocks[
                        "hydrogen_seasonal_storage"
                    ].discharge[t]
                    storage_charge = m.dsm_blocks["hydrogen_seasonal_storage"].charge[t]
                    return (
                        total_hydrogen_production + storage_discharge - storage_charge
                        >= m.hydrogen_demand_per_timestep[t]
                    )
                else:
                    return (
                        total_hydrogen_production >= m.hydrogen_demand_per_timestep[t]
                    )
        else:

            @self.model.Constraint(self.model.time_steps)
            def direct_hydrogen_balance(m, t):
                total_hydrogen_production = 0
                if self.has_electrolyser:
                    total_hydrogen_production += m.dsm_blocks[
                        "electrolyser"
                    ].hydrogen_out[t]
                if self.has_h2seasonal_storage:
                    storage_discharge = m.dsm_blocks[
                        "hydrogen_seasonal_storage"
                    ].discharge[t]
                    storage_charge = m.dsm_blocks["hydrogen_seasonal_storage"].charge[t]
                    return (
                        total_hydrogen_production + storage_discharge - storage_charge
                        == m.cumulative_hydrogen_output[t]
                    )
                else:
                    return total_hydrogen_production == m.cumulative_hydrogen_output[t]

    def define_constraints(self):
        """
        Defines key constraints for the hydrogen plant, ensuring that hydrogen production,
        storage, and energy costs are accurately modeled.

        This function includes the following constraints:

        1. **Total Power Input Constraint**:
        - Ensures that the power input required by the electrolyser is correctly accounted for
            at each time step.
        - This constraint ensures that energy demand is properly modeled for optimization
            purposes.

        2. **Variable Cost per Time Step Constraint**:
        - Calculates the operating cost per time step, ensuring that total variable
            costs reflect the electrolyser's energy consumption.
        - This constraint is essential for cost-based optimization of hydrogen production.

        These constraints collectively ensure that hydrogen production aligns with energy
        availability, demand fulfillment, and cost efficiency.
        """

        @self.model.Constraint(self.model.time_steps)
        def absolute_demand_association_constraint(m, t):
            """
            Ensures the thermal output meets the absolute demand.
            """
            if not self.demand or self.demand == 0:
                return pyo.Constraint.Skip
            else:
                return (
                    sum((m.cumulative_hydrogen_output[t]) for t in m.time_steps)
                    >= m.absolute_hydrogen_demand
                )

        # Constraint for total power input
        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
            """
            Ensures the total power input is the sum of power inputs of all components.
            """
            return m.total_power_input[t] == m.dsm_blocks["electrolyser"].power_in[t]

        # Constraint for variable cost per time step
        @self.model.Constraint(self.model.time_steps)
        def cost_per_time_step(m, t):
            """
            Calculates the variable cost per time step.
            """

            return m.variable_cost[t] == m.dsm_blocks["electrolyser"].operating_cost[t]
