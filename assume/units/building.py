# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import pyomo.environ as pyo

from assume.common.base import SupportsMinMax
from assume.common.forecasts import Forecaster
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

    Args:
        id (str): Unique identifier for the building unit.
        unit_operator (str): Operator managing the building unit.
        bidding_strategies (dict): Strategies used for energy bidding in the market.
        forecaster (Forecaster): A forecaster used to get key variables such as fuel or electricity prices.
        components (dict[str, dict]): Sub-components of the building, such as heat pumps or storage systems.
        technology (str, optional): Type of technology the building unit employs. Default is "building".
        objective (str, optional): Optimization objective, e.g., "min_variable_cost" to minimize operational expenses. Default is "min_variable_cost".
        flexibility_measure (str, optional): Metric used to assess the building's flexibility, e.g., "cost_based_load_shift". Default is "cost_based_load_shift".
        is_prosumer (str, optional): Indicates whether the building acts as a prosumer (producing and consuming energy). Default is "No".
        cost_tolerance (float, optional): Maximum allowable cost variation for flexibility measures. Default is 10.
        node (str, optional): Network node where the building unit is connected. Default is "node0".
        location (tuple[float, float], optional): Geographic coordinates (latitude, longitude) of the building. Default is (0.0, 0.0).
        **kwargs: Additional keyword arguments for custom configurations.

    Attributes:
        required_technologies (list): A list of required technologies for the building unit (empty by default).
        optional_technologies (list): A list of optional technologies the building unit can incorporate, such as heat pumps and PV plants.
    """

    # List of required and optional technologies for the building unit
    required_technologies = []
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
        forecaster: Forecaster,
        components: dict[str, dict],
        technology: str = "building",
        objective: str = "min_variable_cost",
        flexibility_measure: str = "cost_based_load_shift",
        is_prosumer: str = "No",
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
            # Assign the aligned profile
            self.components["pv_plant"][
                "power_profile"
                if profile_key.endswith("power_profile")
                else "availability_profile"
            ] = pv_profile

        # Initialize the model
        self.setup_model(presolve=True)

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
        Establishes energy flow constraints for the building system, ensuring proper integration
        of energy components such as heat pumps, boilers, thermal storage, electric vehicles,
        battery storage, and photovoltaic (PV) plants.

        This function defines two key constraints:

        1. **Heating Demand Balance Constraint**:
        - Ensures that the total heat output from heat pumps, boilers, and thermal storage
            discharges meets the building's heat demand while accounting for thermal storage charging.
        - This constraint guarantees that heating components operate efficiently to satisfy demand
            while adhering to storage dynamics.

        2. **Total Power Input Constraint**:
        - Ensures that the total power drawn from the grid (or external sources) balances the
            inflexible demand, power consumption from components, and contributions from self-produced
            or stored energy (e.g., PV generation, battery storage, or electric vehicles).
        - This constraint aggregates power inputs from various components, subtracting any
            self-generated energy or stored power discharge, to maintain a net-zero mismatch.

        These constraints collectively ensure energy conservation and proper interaction between
        different energy assets within the building.

        """

        # Heat flow constraint for heating components
        if self.has_heatpump or self.has_boiler or self.has_thermal_storage:

            @self.model.Constraint(self.model.time_steps)
            def heating_demand_balance_constraint(m, t):
                """
                Ensures the total heat output matches demand plus storage dynamics.
                """
                heat_pump_output = (
                    m.dsm_blocks["heat_pump"].heat_out[t] if self.has_heatpump else 0
                )
                boiler_output = (
                    m.dsm_blocks["boiler"].heat_out[t] if self.has_boiler else 0
                )
                thermal_storage_discharge = (
                    m.dsm_blocks["thermal_storage"].discharge[t]
                    if self.has_thermal_storage
                    else 0
                )
                thermal_storage_charge = (
                    m.dsm_blocks["thermal_storage"].charge[t]
                    if self.has_thermal_storage
                    else 0
                )
                return (
                    heat_pump_output + boiler_output + thermal_storage_discharge
                    == m.heat_demand[t] + thermal_storage_charge
                )

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
            total_power = m.inflex_demand[t]

            # Add power inputs from available components
            if self.has_heatpump:
                total_power += m.dsm_blocks["heat_pump"].power_in[t]
            if self.has_boiler:
                total_power += m.dsm_blocks["boiler"].power_in[t]

            # Add and subtract EV and storage power if they exist
            if self.has_ev:
                total_power += m.dsm_blocks["electric_vehicle"].charge[t]
                total_power -= m.dsm_blocks["electric_vehicle"].discharge[t]
            if self.has_battery_storage:
                total_power += m.dsm_blocks["generic_storage"].charge[t]
                total_power -= m.dsm_blocks["generic_storage"].discharge[t]

            # Subtract power from PV plant if it exists
            if self.has_pv:
                total_power -= m.dsm_blocks["pv_plant"].power[t]

            # Assign the calculated total to the model's total_power_input for each time step
            return m.total_power_input[t] == total_power

    def define_constraints(self):
        """
        Defines the optimization constraints for the Pyomo model, ensuring that the building's
        energy consumption, production, and cost calculations adhere to operational rules.

        This function establishes the following constraints:

        1. **Grid Export Constraint (for Non-Prosumers)**:
        - Ensures that buildings classified as non-prosumers cannot export power to the grid.
        - This restriction is applied by enforcing a lower bound of zero on the total power
            input, meaning that the building can only draw energy from external sources,
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
