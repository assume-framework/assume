# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import pyomo.environ as pyo

from assume.common.base import SupportsMinMax
from assume.common.forecaster import BuildingForecaster
from assume.common.utils import str_to_bool
from assume.units.dsm_load_shift import DSMFlex

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
        "generic_storage",
        "pv_plant",
        "electric_vehicle",
        "charging_station",
    ]

    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        forecaster: BuildingForecaster,
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

        if not isinstance(forecaster, BuildingForecaster):
            raise ValueError(
                f"forecaster must be of type {BuildingForecaster.__name__}"
            )

        allowed_technologies = self.required_technologies + self.optional_technologies

        # check if the required components are present in the components dictionary
        for technology in self.required_technologies:
            if not any(component.startswith(technology) for component in components):
                raise ValueError(
                    f"Component {technology} is required for the building plant unit."
                )

        # check if the provided components are valid and do not contain any unknown components
        for component in components:
            if not any(
                component.startswith(technology) for technology in allowed_technologies
            ):
                raise ValueError(
                    f"Component {component} is not a valid component for the building unit."
                )

        self.market_id = "EOM"  # FIXME enable other markets

        self.electricity_price = self.forecaster.electricity_price
        self.natural_gas_price = self.forecaster.get_price("natural_gas")
        self.heat_demand = self.forecaster.heat_demand
        self.ev_load_profile = self.forecaster.ev_load_profile
        self.battery_load_profile = self.forecaster.battery_load_profile
        self.inflex_demand = self.forecaster.load_profile

        self.objective = objective
        self.flexibility_measure = flexibility_measure
        self.cost_tolerance = cost_tolerance
        self.is_prosumer = str_to_bool(is_prosumer)

        # Check for the presence of components
        self.heat_pumps = [k for k in self.components if k.startswith("heat_pump")]
        self.boilers = [k for k in self.components if k.startswith("boiler")]
        self.thermal_storages = [
            k for k in self.components if k.startswith("thermal_storage")
        ]
        self.evs = [k for k in self.components if k.startswith("electric_vehicle")]
        self.charging_stations = [
            k for k in self.components if k.startswith("charging_station")
        ]
        self.battery_storages = [
            k for k in self.components if k.startswith("generic_storage")
        ]
        self.pv_plants = [k for k in self.components if k.startswith("pv_plant")]

        self.has_heatpump = len(self.heat_pumps) > 0
        self.has_boiler = len(self.boilers) > 0
        self.has_thermal_storage = len(self.thermal_storages) > 0
        self.has_ev = len(self.evs) > 0
        self.has_charging_station = len(self.charging_stations) > 0
        self.has_battery_storage = len(self.battery_storages) > 0
        self.has_pv = len(self.pv_plants) > 0

        if self.has_ev and not self.has_charging_station:
            logger.warning(
                "No charging station is included in the building. "
                "Electric vehicles will be connected directly to the grid."
            )

        # prepare EV-specific profiles
        for ev_key in self.evs:
            ev_config = self.components[ev_key]

            use_charging_profile = str_to_bool(
                ev_config.get("charging_profile", "false")
            )

            profile_key_suffix = (
                "charging_profile" if use_charging_profile else "availability_profile"
            )

            # Remove input config keys; we will fetch the actual profile from forecaster
            ev_config.pop("charging_profile", None)
            ev_config.pop("availability_profile", None)

            profile_key = f"{self.id}_{ev_key}_{profile_key_suffix}"

            try:
                ev_profile = self.forecaster[profile_key]
            except KeyError:
                # fallback to building forecaster EV profile if available
                if (
                    hasattr(self.forecaster, "ev_load_profile")
                    and self.forecaster.ev_load_profile is not None
                ):
                    ev_profile = self.forecaster.ev_load_profile
                else:
                    raise ValueError(
                        f"Required profile '{profile_key}' not found in forecaster for EV '{ev_key}'."
                    ) from None

            ev_config[profile_key_suffix] = ev_profile

            trip_distance_key = f"{self.id}_{ev_key}_trip_distance"
            try:
                ev_config["trip_distance"] = self.forecaster[trip_distance_key]
            except KeyError:
                # optional: if no trip distance is provided, EV will behave like stationary EV
                ev_config["trip_distance"] = None

        for cs_key in self.charging_stations:
            cs_config = self.components[cs_key]
            cs_config.pop("availability_profile", None)
            profile_key = f"{self.id}_{cs_key}_availability_profile"
            try:
                cs_config["availability_profile"] = self.forecaster[profile_key]
            except KeyError:
                # optional; charging station can remain always available
                pass

        # Configure PV plant power profile based on availability
        if self.has_pv:
            for pv_key in self.pv_plants:
                uses_power_profile = str_to_bool(
                    self.components[pv_key].get("uses_power_profile", "false")
                )
                key = "availability_profile" if uses_power_profile else "power_profile"
                self.components[pv_key][key] = self.forecaster.pv_profile

        # Initialize the model
        # self.setup_model(presolve=True)  # NOTE: called in forecaster initialization again!!!

    def define_parameters(self):
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

        # EV trip distance parameters
        # Trip distance is optional and only used for trip-based EV modeling.
        # If provided, it represents the distance traveled by the EV in each time step.
        # This allows the model to estimate energy consumption based on distance and vehicle efficiency.
        for ev_key in self.evs:
            ev_trip_distance = self.components[ev_key].get("trip_distance", None)
            if ev_trip_distance is None:
                continue

            # Normalize trip distance data to a dictionary keyed by time step
            # Handle both pandas Series (with .iloc) and array-like objects (list, numpy array)
            if hasattr(ev_trip_distance, "iloc"):
                trip_distance_init = {
                    t: ev_trip_distance.iloc[t] if t < len(ev_trip_distance) else 0
                    for t in self.model.time_steps
                }
            else:
                trip_distance_init = {
                    t: ev_trip_distance[t] if t < len(ev_trip_distance) else 0
                    for t in self.model.time_steps
                }

            self.model.add_component(
                f"{ev_key}_trip_distance",
                pyo.Param(
                    self.model.time_steps,
                    initialize=trip_distance_init,
                    doc=f"Trip distance profile for {ev_key} (in km or user-defined units)",
                ),
            )

        # EV availability parameters
        # An EV can be represented as either "available to charge" or "willing to charge (charging_profile)".
        # Both are binary profiles: availability_profile indicates when the EV is connected,
        # while charging_profile (from direct load profile) indicates when it wishes to charge.
        # We normalize both to a single "availability" parameter for the optimization model.
        for ev_key in self.evs:
            # Determine which profile type this EV uses
            if "availability_profile" in self.components[ev_key]:
                # Explicit availability profile: when EV is connected to the grid
                profile_data = self.components[ev_key]["availability_profile"]
            elif "charging_profile" in self.components[ev_key]:
                # Charging profile (from direct charging mode): use as availability proxy
                profile_data = self.components[ev_key]["charging_profile"]
            else:
                profile_data = None

            if profile_data is None:
                continue

            # Normalize profile data to a dictionary keyed by time step
            # Handle both pandas Series (with .iloc) and array-like objects (list, numpy array)
            if hasattr(profile_data, "iloc"):
                profile_init = {
                    t: int(profile_data.iloc[t]) if t < len(profile_data) else 0
                    for t in self.model.time_steps
                }
            else:
                profile_init = {
                    t: int(profile_data[t]) if t < len(profile_data) else 0
                    for t in self.model.time_steps
                }

            self.model.add_component(
                f"{ev_key}_availability",
                pyo.Param(
                    self.model.time_steps,
                    initialize=profile_init,
                    within=pyo.Binary,
                    doc=f"Availability profile for {ev_key} (binary: 1 if available, 0 otherwise)",
                ),
            )

    def define_variables(self):
        self.model.total_power_input = pyo.Var(self.model.time_steps, within=pyo.Reals)
        self.model.variable_cost = pyo.Var(self.model.time_steps, within=pyo.Reals)

        # The following variables are only defined when both EVs and charging stations are present.
        # They model the assignment of EVs to specific charging stations and the power flow.
        # Without charging stations, EVs connect directly to the grid and no assignment tracking is needed.
        if self.has_ev and self.has_charging_station:
            self.model.evs = pyo.Set(initialize=self.evs)
            self.model.charging_stations = pyo.Set(initialize=self.charging_stations)

            self.model.is_assigned = pyo.Var(
                self.model.evs,
                self.model.charging_stations,
                self.model.time_steps,
                domain=pyo.Binary,
                doc="1 if EV is assigned to charging station at time t",
            )

            self.model.charge_assignment = pyo.Var(
                self.model.evs,
                self.model.charging_stations,
                self.model.time_steps,
                domain=pyo.NonNegativeReals,
                doc="Charging power from charging station to EV",
            )

    def initialize_process_sequence(self):
        # ------------------------------------------------------------------
        # 1. Heat balance
        # ------------------------------------------------------------------
        if self.has_heatpump or self.has_boiler or self.has_thermal_storage:

            @self.model.Constraint(self.model.time_steps)
            def heating_demand_balance_constraint(m, t):
                heat_supply = 0
                heat_storage_charge = 0

                for hp in self.heat_pumps:
                    heat_supply += m.dsm_blocks[hp].heat_out[t]

                for boiler in self.boilers:
                    heat_supply += m.dsm_blocks[boiler].heat_out[t]

                for ts in self.thermal_storages:
                    heat_supply += m.dsm_blocks[ts].discharge[t]
                    heat_storage_charge += m.dsm_blocks[ts].charge[t]

                return heat_supply == m.heat_demand[t] + heat_storage_charge

        # ------------------------------------------------------------------
        # 2. EV <-> charging station linking
        # ------------------------------------------------------------------
        if self.has_ev and self.has_charging_station:

            @self.model.Constraint(
                self.model.evs,
                self.model.time_steps,
                doc="EV total charge equals sum of charging-station assignments",
            )
            def ev_total_charge_constraint(m, ev, t):
                return m.dsm_blocks[ev].charge[t] == sum(
                    m.charge_assignment[ev, cs, t] for cs in m.charging_stations
                )

            @self.model.Constraint(
                self.model.charging_stations,
                self.model.time_steps,
                doc="Charging station output equals sum of assigned EV charging",
            )
            def station_total_output_constraint(m, cs, t):
                return m.dsm_blocks[cs].charge[t] == sum(
                    m.charge_assignment[ev, cs, t] for ev in m.evs
                )

            @self.model.Constraint(
                self.model.evs,
                self.model.charging_stations,
                self.model.time_steps,
                doc="Charge assignment only if EV assigned to charging station",
            )
            def assignment_indicator_constraint(m, ev, cs, t):
                return (
                    m.charge_assignment[ev, cs, t]
                    <= m.dsm_blocks[cs].max_power * m.is_assigned[ev, cs, t]
                )

            @self.model.Constraint(
                self.model.charging_stations,
                self.model.time_steps,
                doc="At most one EV per charging station at a time",
            )
            def max_one_ev_per_station(m, cs, t):
                return sum(m.is_assigned[ev, cs, t] for ev in m.evs) <= 1

            @self.model.Constraint(
                self.model.evs,
                self.model.time_steps,
                doc="Each EV can be assigned to at most one charging station",
            )
            def max_one_station_per_ev(m, ev, t):
                return sum(m.is_assigned[ev, cs, t] for cs in m.charging_stations) <= 1

            @self.model.Constraint(
                self.model.evs,
                self.model.charging_stations,
                self.model.time_steps,
                doc="Assignment only if EV is available",
            )
            def assignment_only_if_available(m, ev, cs, t):
                ev_availability = getattr(m, f"{ev}_availability", None)
                if ev_availability is not None:
                    return m.is_assigned[ev, cs, t] <= ev_availability[t]
                return pyo.Constraint.Skip

        # ------------------------------------------------------------------
        # 3. Building electric power balance
        # ------------------------------------------------------------------
        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
            total_power = m.inflex_demand[t]

            # electric loads from heat technologies
            for hp in self.heat_pumps:
                total_power += m.dsm_blocks[hp].power_in[t]

            for boiler in self.boilers:
                if hasattr(m.dsm_blocks[boiler], "power_in"):
                    total_power += m.dsm_blocks[boiler].power_in[t]

            # EV / charging station topology
            if self.has_ev and self.has_charging_station:
                # charging stations connect to grid/building
                for cs in self.charging_stations:
                    total_power += m.dsm_blocks[cs].charge[t]
                    if hasattr(m.dsm_blocks[cs], "discharge"):
                        total_power -= m.dsm_blocks[cs].discharge[t]
            elif self.has_ev:
                # no charging station -> EVs directly connect to grid
                for ev in self.evs:
                    total_power += m.dsm_blocks[ev].charge[t]
                    total_power -= m.dsm_blocks[ev].discharge[t]

            # stationary batteries
            for bat in self.battery_storages:
                total_power += m.dsm_blocks[bat].charge[t]
                total_power -= m.dsm_blocks[bat].discharge[t]

            # PV
            for pv in self.pv_plants:
                total_power -= m.dsm_blocks[pv].power[t]

            return m.total_power_input[t] == total_power

    def define_constraints(self):
        if not self.is_prosumer:

            @self.model.Constraint(self.model.time_steps)
            def grid_export_constraint(m, t):
                return m.total_power_input[t] >= 0

        @self.model.Constraint(self.model.time_steps)
        def variable_cost_constraint(m, t):
            return m.variable_cost[t] == m.total_power_input[t] * m.electricity_price[t]
