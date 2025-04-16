# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import pyomo.environ as pyo

from assume.common.base import SupportsMinMax
from assume.common.forecasts import Forecaster
from assume.common.utils import str_to_bool
from assume.units.dsm_load_shift import DSMFlex

logger = logging.getLogger(__name__)


class BusDepot(DSMFlex, SupportsMinMax):
    """
    Represents a bus depot managing multiple electric vehicle (EV) buses and charging stations.
    This agent optimizes bus charging schedules based on electricity price, grid demand, and available flexibility.

    The `BusDepot` class utilizes the Pyomo optimization library to determine the optimal
    charging strategy that minimizes costs, grid impact, or maximizes renewable utilization.

    Args:
        id (str): Unique identifier for the bus depot.
        unit_operator (str): Operator managing the bus depot.
        bidding_strategies (dict): Strategies used for energy bidding in the market.
        forecaster (Forecaster): A forecaster used to get key variables such as electricity prices.
        charging_stations (dict[str, dict]): Charging stations available in the depot.
        electric_vehicles (dict[str, dict]): Electric buses assigned to the depot.
        objective (str, optional): Optimization objective ("min_cost", "min_grid_load", "max_RE_util"). Default is "min_cost".
        flexibility_measure (str, optional): Metric used to assess the depot's flexibility. Default is "cost_based_load_shift".
        max_power_capacity (float, optional): Maximum allowable power capacity. Default is 1000 kW.
        node (str, optional): Network node where the depot is connected. Default is "bus_node".
        location (tuple[float, float], optional): Geographic coordinates. Default is (0.0, 0.0).
        **kwargs: Additional keyword arguments for custom configurations.

    Attributes:
        required_technologies (list): Required technologies for the depot.
        optional_technologies (list): Optional technologies available.
    """

    required_technologies = ["electric_vehicle"]
    optional_technologies = ["charging_station"]  # "charging_station",

    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        forecaster: Forecaster,
        components: dict[str, dict],
        technology: str = "bus_depot",
        is_prosumer: str = "No",
        objective: str = "min_cost",
        flexibility_measure: str = "cost_based_load_shift",
        cost_tolerance: float = 10,
        node: str = "bus_node",
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
        for required_component in self.required_technologies:
            # Check if any component matches the required base technology
            if not any(
                component.startswith(required_component)
                for component in components.keys()
            ):
                raise ValueError(
                    f"Component {required_component} is required for the Bus Depot unit."
                )

        # Check if the provided components are valid and do not contain any unknown components
        for component in components.keys():
            base_component = next(
                (
                    req
                    for req in self.required_technologies + self.optional_technologies
                    if component.startswith(req)
                ),
                None,
            )
            if base_component is None:
                raise ValueError(
                    f"Component {component} is not a valid component for the Bus Depot unit."
                )

        self.objective = objective
        self.is_prosumer = str_to_bool(is_prosumer)
        self.cost_tolerance = cost_tolerance
        self.flexibility_measure = flexibility_measure
        self.is_prosumer = str_to_bool(is_prosumer)

        self.electricity_price = self.forecaster["electricity_price"]
        for ev_key in self.components:
            if ev_key.startswith("electric_vehicle"):
                ev_config = self.components[ev_key]
                use_charging_profile = str_to_bool(
                    ev_config.get("charging_profile", "false")
                )
                ev_config.pop("charging_profile", None)

                profile_key = (
                    f"{self.id}_{ev_key}_charging_profile"
                    if use_charging_profile
                    else f"{self.id}_{ev_key}_availability_profile"
                )
                ev_profile = self.forecaster[profile_key]
                ev_config[
                    "charging_profile"
                    if use_charging_profile
                    else "availability_profile"
                ] = ev_profile

                # ✅ Direct assignment of range series
                ev_config["range"] = self.forecaster[f"{self.id}_{ev_key}_range"]

        # Load profiles for all EVs dynamically
        for ev_key in self.components:
            if ev_key.startswith("electric_vehicle"):
                ev_config = self.components[ev_key]
                use_charging_profile = str_to_bool(
                    ev_config.get("charging_profile", "false")
                )
                ev_config.pop("charging_profile", None)

                profile_key = (
                    f"{self.id}_{ev_key}_charging_profile"
                    if use_charging_profile
                    else f"{self.id}_{ev_key}_availability_profile"
                )

                ev_profile = self.forecaster[profile_key]

                ev_config[
                    "charging_profile"
                    if use_charging_profile
                    else "availability_profile"
                ] = ev_profile

            # # ✅ Add this to preview the time series you're passing into EV
            # print(f"[DEBUG] Loaded EV profile from '{profile_key_ev}':")
            # print(ev_profile.as_pd_series().head(10))  # ✅ If you want a timestamped view

        # Initialize the model
        self.setup_model(presolve=True)

    def define_parameters(self):
        """Defines parameters including electricity price, power capacity, and bus availability."""
        self.model.electricity_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.electricity_price)},
        )
        for ev_key in self.components:
            if ev_key.startswith("electric_vehicle"):
                ev_range = self.components[ev_key]["range"]
                self.model.add_component(
                    f"{ev_key}_range",
                    pyo.Param(
                        self.model.time_steps,
                        initialize={t: ev_range.iloc[t] for t in range(len(ev_range))},
                    ),
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
        for ev in self.model.dsm_blocks:
            if ev.startswith("electric_vehicle"):
                constraint_name = f"charge_flow_constraint_{ev}"
                self.model.add_component(
                    constraint_name,
                    pyo.Constraint(
                        self.model.time_steps,
                        rule=lambda m, t, ev=ev: m.dsm_blocks[ev].charge[t] >= 0,
                    ),
                )

    def define_constraints(self):
        """
        Defines the optimization constraints for the Pyomo model, ensuring that the Bus Depot's
        energy consumption, production, and cost calculations adhere to operational rules.

        This function establishes the following constraints:

        1. **Grid Export Constraint (for Non-Prosumers)**:
        - Ensures that Bus Depots classified as non-prosumers cannot export power to the grid.
        - This restriction is applied by enforcing a lower bound of zero on the total power
            input, meaning that the Bus Depot can only draw energy from external sources,
            but cannot inject excess energy back into the grid.

        2. **Variable Cost Calculation Constraint**:
        - Computes the variable cost incurred at each time step based on total power input
            and electricity price.
        - This constraint ensures that the total variable cost is directly proportional to
            energy consumption, allowing for accurate cost minimization in the optimization model.

        These constraints help enforce realistic energy system behavior while aligning with
        market regulations and operational objectives.
        """
        if not self.is_prosumer:

            @self.model.Constraint(self.model.time_steps)
            def grid_export_constraint(m, t):
                """Restricts non-prosumers from exporting to the grid."""
                return m.total_power_input[t] >= 0

        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
            """
            Ensures that the total power input is the sum of all component inputs minus any self-produced
            or stored energy at each time step.

            This constraint aggregates power from buses.

            Args:
                m: Pyomo model reference.
                t: Current time step.

            Returns:
                Equality condition balancing total power input.
            """
            return m.total_power_input[t] == sum(
                m.dsm_blocks[ev].charge[t]  # - m.dsm_blocks[ev].discharge[t]
                for ev in m.dsm_blocks
                if ev.startswith("electric_vehicle")
            )

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
            return m.variable_cost[t] == m.total_power_input[t] * m.electricity_price[t]
