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
        components (dict[str, dict]): Components like electric vehicles and charging stations available in the depot.
        technology (str, optional): Technology type. Default is "bus_depot".
        is_prosumer (str, optional): Whether the unit can export power ("Yes" or "No"). Default is "No".
        objective (str, optional): Optimization objective ("min_cost", "min_grid_load", "max_RE_util"). Default is "min_cost".
        flexibility_measure (str, optional): Metric used to assess flexibility. Default is "cost_based_load_shift".
        cost_tolerance (float, optional): Tolerance for flexibility measure. Default is 10.
        node (str, optional): Network node where the depot is connected. Default is "bus_node".
        location (tuple[float, float], optional): Geographic coordinates. Default is (0.0, 0.0).
        **kwargs: Additional keyword arguments for custom configurations.

    Attributes:
        required_technologies (list): Required technologies for the depot.
        optional_technologies (list): Optional technologies available.
    """

    required_technologies = ["electric_vehicle"]
    optional_technologies = ["charging_station"]

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
        congestion_threshold: float = 0,
        cost_tolerance: float = 0,
        peak_load_cap: float = 0,
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

        # Component validation for required technologies
        for required_component in self.required_technologies:
            if not any(
                component.startswith(required_component)
                for component in components.keys()
            ):
                raise ValueError(
                    f"Component {required_component} is required for the Bus Depot unit."
                )

        # Validate all provided components against required and optional lists
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

        # Set attributes
        self.objective = objective
        self.is_prosumer = str_to_bool(is_prosumer)
        self.cost_tolerance = cost_tolerance
        self.flexibility_measure = flexibility_measure

        # Fetch electricity price forecast
        try:
            self.electricity_price = self.forecaster["electricity_price"]
        except KeyError:
            raise ValueError("Forecaster must provide 'electricity_price'.")

        self.electricity_price_flex = self.forecaster["electricity_price_flex"]
        self.congestion_signal = self.forecaster["congestion_signal"]
        self.renewable_utilisation_signal = self.forecaster["RE_availability"]
        self.unidirectional_load = self.forecaster["unidirectional_load"]
        self.congestion_threshold = congestion_threshold
        self.peak_load_cap = peak_load_cap

        for ev_key in self.components:
            if ev_key.startswith("electric_vehicle"):
                ev_config = self.components[ev_key]

                # 1. Check if 'charging_profile' flag is set in the input config BEFORE modifying ev_config
                use_charging_profile = str_to_bool(
                    ev_config.get("charging_profile", "false")
                )

                ev_config.pop("charging_profile", None)
                ev_config.pop("availability_profile", None)

                # 3. Determine the correct profile key suffix and the full key to look up in the forecaster
                profile_key_suffix = (
                    "charging_profile"
                    if use_charging_profile
                    else "availability_profile"
                )
                profile_key = f"{self.id}_{ev_key}_{profile_key_suffix}"

                # 4. Fetch the required profile data from the forecaster
                try:
                    ev_profile = self.forecaster[profile_key]
                    logger.debug(f"Loaded profile '{profile_key}' for {ev_key}.")
                except KeyError:
                    # Raise a specific error if the expected profile isn't found in the forecaster
                    raise ValueError(
                        f"Required profile '{profile_key}' not found in forecaster for Electric Vehicle: {ev_key}. "
                        f"Ensure the profile exists. If using 'charging_profile', check the flag in the configuration."
                    ) from None  # Suppress the original KeyError context for clarity

                # 5. Assign the fetched profile data to the correct key in ev_config
                ev_config[profile_key_suffix] = ev_profile

                # 6. Assign range parameter, fetching it from the forecaster
                range_key = f"{self.id}_{ev_key}_range"
                try:
                    ev_config["range"] = self.forecaster[range_key]
                    logger.debug(f"Loaded range profile '{range_key}' for {ev_key}.")
                except KeyError:
                    raise ValueError(
                        f"Required range profile '{range_key}' not found in forecaster for Electric Vehicle: {ev_key}."
                    ) from None

        for cs_key in self.components:
            if cs_key.startswith("charging_station"):
                cs_config = self.components[cs_key]
                # Remove potential profile keys if they are not loaded/managed here
                cs_config.pop("availability_profile", None)
                cs_config.pop("charging_profile", None)  # Just in case

        # Setup the Pyomo model after configuration is complete
        self.setup_model(presolve=True)

    def define_variables(self):
        """
        Defines the decision variables for the Pyomo optimization model.

        - `total_power_input`: Represents the total power input required at each time step.
        - `variable_cost`: Represents the variable cost associated with power usage at each time step.

        Both variables are defined over the `time_steps` set and are continuous real numbers.
        """
        self.model.total_power_input = pyo.Var(self.model.time_steps, within=pyo.Reals)
        self.model.variable_cost = pyo.Var(self.model.time_steps, within=pyo.Reals)
        
        #  1: EV-CS assignment variables
        # Represents whether an electric vehicle (EV) is assigned to a charging station (CS)
        self.model.is_assigned = pyo.Var(
            self.model.evs,
            self.model.charging_stations,
            self.model.time_steps,
            domain=pyo.Binary,
            doc="1 if EV is assigned to CS at time t",
        )

        #  2: Queue variables
        # Represents whether an EV is waiting in the queue for a charging station at each time step
        self.model.in_queue = pyo.Var(
            self.model.evs,
            self.model.time_steps,
            domain=pyo.Binary,
            bounds=(0, 1),
            doc="EV is waiting in queue",
        )

        #  3: Charge assignment variables
        # Represents the power transferred from charging stations to electric vehicles at each time step.
        self.model.charge_assignment = pyo.Var(
            self.model.evs,
            self.model.charging_stations,
            self.model.time_steps,
            domain=pyo.NonNegativeReals,
            doc="Power transferred from CS to EV at time t",
        )

        #  4: Cumulative charged 
        self.model.cumulative_charged = pyo.Var(
            self.model.evs, 
            self.model.time_steps, 
            domain=pyo.NonNegativeReals,
            doc="Cumulative energy charged for each EV"
        )

        #  5: Fully charged indicator
        self.model.is_fully_charged = pyo.Var(
            self.model.evs,
            self.model.time_steps,
            domain=pyo.Binary,
            doc="Binary indicator if EV has reached full charge",
        )

        #  6: Has sufficient charge for travel indicator
        self.model.has_sufficient_charge_for_travel = pyo.Var(
            self.model.evs,
            self.model.time_steps,
            domain=pyo.Binary,
            doc="Binary indicator if EV has enough charge to satisfy travel requirements",
        )

    def define_parameters(self):
        """Defines parameters including electricity price, power capacity, and bus availability."""
        self.model.electricity_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.electricity_price)},
        )

        self.model.electricity_price_flex = pyo.Param(
            self.model.time_steps,
            initialize={
                t: value for t, value in enumerate(self.electricity_price_flex)
            },
        )
        
        #  EV range and availability parameters
        for ev_key in self.components:
            if ev_key.startswith("electric_vehicle"):
                # Range parameter
                ev_range = self.components[ev_key]["range"]
                
                
                if hasattr(ev_range, 'iloc'):  # pandas Series
                    range_init = {t: ev_range.iloc[t] for t in range(len(ev_range))}
                elif hasattr(ev_range, '__getitem__'):  # list veya array
                    range_init = {t: ev_range[t] for t in range(len(ev_range))}
                else:
                    raise TypeError(f"Unexpected range data type for {ev_key}: {type(ev_range)}")
                
                self.model.add_component(
                    f"{ev_key}_range",
                    pyo.Param(
                        self.model.time_steps,
                        initialize=range_init,
                        doc=f"Range forecast for {ev_key}",
                    ),
                )
                
                # Availability parameter
                profile_key = "availability_profile" if "availability_profile" in self.components[ev_key] else "charging_profile"
                profile_data = self.components[ev_key].get(profile_key, [])
                
                
                if hasattr(profile_data, 'iloc'):  # pandas Series
                    profile_init = {t: int(profile_data.iloc[t]) if t < len(profile_data) else 0 
                                for t in self.model.time_steps}
                elif hasattr(profile_data, '__getitem__'):  
                    profile_init = {t: int(profile_data[t]) if t < len(profile_data) else 0 
                                for t in self.model.time_steps}
                else:
                    # Default  1 (AVAILABLE) 
                    profile_init = {t: 1 for t in self.model.time_steps}
                
                self.model.add_component(
                    f"{ev_key}_availability",
                    pyo.Param(
                        self.model.time_steps,
                        initialize=profile_init,
                        within=pyo.Binary,
                        doc=f"Availability profile for {ev_key}: 1 if available at depot, 0 if driving",
                    )
                )
        
        # Compatibility matrix parametresi(no needed for this, but added for completeness)
        self.model.compatible = pyo.Param(
            self.model.evs,
            self.model.charging_stations,
            initialize=1,
            within=pyo.Binary,
            mutable=True,
            doc="Compatibility matrix: 1 if EV can use CS, 0 otherwise",
        )
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

        # 8- EV charge - CS assignment connection
        @self.model.Constraint(
            self.model.evs,
            self.model.time_steps,
            doc="Links EV total charge to assignments from CSs",
        )
        def ev_total_charge(m, ev, t):
            if ev in m.dsm_blocks and hasattr(m.dsm_blocks[ev], "charge"):
                return m.dsm_blocks[ev].charge[t] == sum(
                    m.charge_assignment[ev, cs, t] for cs in m.charging_stations
                )
            else:
                return pyo.Constraint.Skip
        
        # 9: CS discharge - EV assignment connection
        @self.model.Constraint(
            self.model.charging_stations,
            self.model.time_steps,
            doc="Links CS discharge to assignments to EVs"
        )
        def station_discharge_balance(m, cs, t):
            if cs in m.dsm_blocks and hasattr(m.dsm_blocks[cs], "discharge"):
                return m.dsm_blocks[cs].discharge[t] == sum(
                    m.charge_assignment[ev, cs, t] for ev in m.evs
                )
            else:
                return pyo.Constraint.Skip

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
            cs_discharge = sum(
                m.dsm_blocks[cs].discharge[t]
                for cs in m.dsm_blocks
                if cs.startswith("charging_station")
            )
            return m.total_power_input[t] == cs_discharge

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
        
        #  10: Assignment just if EV is available
        # This constraint ensures that an EV can only be assigned to a charging station if it is
        @self.model.Constraint(self.model.evs, self.model.charging_stations, self.model.time_steps)
        def assignment_only_if_available(m, ev, cs, t):
            """EV can only be assigned to a CS if it's available at the depot"""
            ev_availability = getattr(m, f"{ev}_availability", None)
            if ev_availability is not None:
                return m.is_assigned[ev, cs, t] <= ev_availability[t]
            return pyo.Constraint.Skip
        
        #  11: one cs per EV at each timestep
        # This constraint ensures that at most one EV can be assigned to a charging station at each
        @self.model.Constraint(
            self.model.charging_stations,
            self.model.time_steps,
            doc="Max 1 EV per CS at each timestep",
        )
        def max_one_ev_per_station(m, cs, t):
            return sum(m.is_assigned[ev, cs, t] for ev in m.evs) <= 1
        
        #  12: EV can be assigned to at most one CS at a time
        # This constraint ensures that each EV can only be assigned to one charging station at a time
        @self.model.Constraint(
            self.model.evs,
            self.model.time_steps,
            doc="EV can be assigned to at most one CS at a time"
        )
        def one_assignment_per_ev(m, ev, t):
            return sum(m.is_assigned[ev, cs, t] for cs in m.charging_stations) <= 1
        
        # FIXED 13: Improved queue logic for multiple charging stations
        @self.model.Constraint(
            self.model.evs,
            self.model.time_steps,
            doc="Queue logic: EV is in queue if available and not assigned to any CS",
        )
        def enforce_queue_logic(m, ev, t):
            ev_availability = getattr(m, f"{ev}_availability", None)
            if ev_availability is not None:
                # EV is in queue if available but not assigned to any charging station
                total_ev_assignments = sum(m.is_assigned[ev, cs, t] for cs in m.charging_stations)
                return m.in_queue[ev, t] == ev_availability[t] - total_ev_assignments
            return pyo.Constraint.Skip
        
        # NEW: Global queue capacity constraint  
        @self.model.Constraint(
            self.model.time_steps,
            doc="Total queue size cannot exceed (available_evs - charging_stations)"
        )
        def global_queue_capacity_constraint(m, t):
            """
            Mathematical constraint: If we have N charging stations and M available EVs,
            then maximum queue size = max(0, M - N)
            """
            # Count total available EVs
            total_available_evs = 0
            evs_list = sorted(m.evs)
            for ev in evs_list:
                ev_availability = getattr(m, f"{ev}_availability", None)
                if ev_availability is not None:
                    total_available_evs += ev_availability[t]
            
            # Count total charging stations
            total_charging_stations = len(m.charging_stations)
            
            # Calculate maximum possible queue size
            max_queue_size = max(0, total_available_evs - total_charging_stations)
            
            # Total EVs in queue at time t
            total_in_queue = sum(m.in_queue[ev, t] for ev in evs_list)
            
            # Constraint: queue size cannot exceed theoretical maximum
            return total_in_queue <= max_queue_size
        
        #  14: Charge assignment - is_assigned connection
        # This constraint links the charge assignment variable with the is_assigned variable.
        @self.model.Constraint(
            self.model.evs,
            self.model.charging_stations,
            self.model.time_steps,
            doc="Link is_assigned with actual charge flow",
        )
        def assignment_indicator_constraint(m, ev, cs, t):
            M = 1e5  # Big-M value
            return m.charge_assignment[ev, cs, t] <= M * m.is_assigned[ev, cs, t]
        
        #  15: EV only charges when available
        # This constraint ensures that an EV can only charge if it is available (not driving).
        @self.model.Constraint(
            self.model.evs,
            self.model.time_steps,
            doc="Ensures EV cannot charge if it's not available (driving)",
        )
        def availability_based_on_charging(m, ev, t):
            ev_availability = getattr(m, f"{ev}_availability", None)
            if ev_availability is not None:
                M = 1e5  # Big-M value
                charge_sum = sum(
                    m.charge_assignment[ev, cs, t] for cs in m.charging_stations
                )
                return charge_sum <= M * ev_availability[t]
            return pyo.Constraint.Skip
        
        #  16: Compatibility constraint
        @self.model.Constraint(
            self.model.evs,
            self.model.charging_stations,
            self.model.time_steps,
            doc="EV can only be assigned to compatible CS"
        )
        def compatibility_constraint(m, ev, cs, t):
            return m.is_assigned[ev, cs, t] <= m.compatible[ev, cs]
        
        #  17: Cumulative charging tracking
        @self.model.Constraint(self.model.evs, self.model.time_steps)
        def cumulative_charging_tracking(m, ev, t):
            if t == m.time_steps.first():
                return m.cumulative_charged[ev, t] == m.dsm_blocks[ev].charge[t]
            else:
                return (
                    m.cumulative_charged[ev, t]
                    == m.cumulative_charged[ev, t - 1] + m.dsm_blocks[ev].charge[t]
                )
        
        #  18: Fully charged flag
        @self.model.Constraint(self.model.evs, self.model.time_steps)
        def enforce_fully_charged_flag(m, ev, t):
            M = 1e5
            ε = 1e-3
            if ev in m.dsm_blocks and hasattr(m.dsm_blocks[ev], "max_capacity"):
                # If cumulative charge reaches max capacity, set fully charged flag
                return (
                    m.cumulative_charged[ev, t]
                    >= m.dsm_blocks[ev].max_capacity - ε
                    - (1 - m.is_fully_charged[ev, t]) * M
                )
            return pyo.Constraint.Skip
        
        #  19: Fully charged flag - reverse constraint
        @self.model.Constraint(self.model.evs, self.model.time_steps)
        def enforce_fully_charged_flag_reverse(m, ev, t):
            M = 1e5
            ε = 1e-3
            if ev in m.dsm_blocks and hasattr(m.dsm_blocks[ev], "max_capacity"):
                # If cumulative charge is less than max capacity, fully charged flag should be 0
                return (
                    m.cumulative_charged[ev, t]
                    <= m.dsm_blocks[ev].max_capacity - ε
                    + m.is_fully_charged[ev, t] * M
                )
            return pyo.Constraint.Skip
        
        # CONSTRAINT 20: Sufficient charge flag logic (CORRECTED VERSION)
        # PURPOSE: Determines when an EV has enough charge for the next unavailable period
        # WHY NEEDED: Prevents EVs from leaving charging station before they have enough energy
        # for the NEXT usage period (when is_available=0)
        @self.model.Constraint(self.model.evs, self.model.time_steps)
        def enforce_sufficient_charge_for_travel_flag(m, ev, t):
            """
            Sets the has_sufficient_charge_for_travel binary flag based on current SOC vs 
            NEXT unavailable period's usage (not all future usage).
            """
            M = 500  # Conservative Big-M value
            ε = 0.1  # Epsilon for numerical stability
            
            if ev in m.dsm_blocks and hasattr(m.dsm_blocks[ev], "usage"):
                time_steps_list = list(m.time_steps)  # Already in ascending order
                
                # Calculate energy needed for the NEXT unavailable period only
                ev_availability = getattr(m, f"{ev}_availability", None)
                next_usage_total = 0
                
                if ev_availability is not None:
                    # Find the next period where is_available=0
                    in_unavailable_period = False
                    
                    for t_future in time_steps_list:
                        if t_future > t:
                            if ev_availability[t_future] == 0:
                                # Start of unavailable period - collect usage
                                if not in_unavailable_period:
                                    in_unavailable_period = True
                                next_usage_total += m.dsm_blocks[ev].usage[t_future]
                            else:
                                # Available again - if we were in unavailable period, stop
                                if in_unavailable_period:
                                    break
                
                # Current state of charge
                current_soc = m.dsm_blocks[ev].soc[t] if hasattr(m.dsm_blocks[ev], "soc") else 0
                
                # Big-M constraint: current_soc >= next_usage_total - (1-flag)*M
                return (
                    current_soc >= next_usage_total - ε - (1 - m.has_sufficient_charge_for_travel[ev, t]) * M
                )
            return pyo.Constraint.Skip
        
        # CONSTRAINT 22: Charging continuity based on travel requirements
        # PURPOSE: Ensures EVs don't disconnect from charging before meeting travel needs
        # WHY NEEDED: Prevents inefficient charging patterns and ensures EVs have enough energy
        @self.model.Constraint(
            self.model.evs,
            self.model.charging_stations,
            self.model.time_steps,
            doc="Charging continuity: continue charging until travel requirements met"
        )
        def charging_continuity_for_travel(m, ev, cs, t):
            """
            If an EV was charging at t-1 and is still available at t,
            it must continue charging at the same station unless it has sufficient charge.
            """
            if t == m.time_steps.first():
                return pyo.Constraint.Skip
            
            ev_availability = getattr(m, f"{ev}_availability", None)
            if ev_availability is None:
                return pyo.Constraint.Skip
            
            # Only apply when EV is available in both time periods
            if ev_availability[t] == 0 or ev_availability[t-1] == 0:
                return pyo.Constraint.Skip
            
            # If was assigned to this station at t-1, must stay unless sufficient charge
            return m.is_assigned[ev, cs, t] >= m.is_assigned[ev, cs, t-1] - m.has_sufficient_charge_for_travel[ev, t-1]
        
        # CONSTRAINT 23: Optimal charging station utilization
        # PURPOSE: Ensures efficient use of available charging infrastructure
        # WHY NEEDED: Maximizes charging capacity utilization and prevents idle stations
        @self.model.Constraint(
            self.model.time_steps,
            doc="Ensure optimal utilization of charging stations"
        )
        def optimal_station_utilization(m, t):
            """
            Optimal charging station utilization rule:
            - If available_evs <= charging_stations: all EVs should be assigned
            - If available_evs > charging_stations: all stations should be used
            """
            evs_list = list(m.evs)
            
            # Count available EVs at time t
            available_evs = 0
            for ev in evs_list:
                ev_availability = getattr(m, f"{ev}_availability", None)
                if ev_availability is not None:
                    available_evs += ev_availability[t]
            
            if available_evs == 0:
                return pyo.Constraint.Skip
            
            # Total assignments across all EV-station combinations
            total_assignments = sum(
                m.is_assigned[ev, cs, t] 
                for ev in evs_list 
                for cs in m.charging_stations
            )
            
            total_charging_stations = len(m.charging_stations)
            
            if available_evs <= total_charging_stations:
                # Enough stations for all EVs - assign all available EVs
                return total_assignments >= available_evs
            else:
                # More EVs than stations - utilize all stations
                return total_assignments >= total_charging_stations
        
        
        # CONSTRAINT 24: Prevent premature EV switching
        # PURPOSE: Prevents EVs from leaving charging before completing their charge cycle
        # WHY NEEDED: Ensures charging efficiency and prevents incomplete charging sessions
        @self.model.Constraint(
            self.model.evs,
            self.model.time_steps,
            doc="Prevent EV from leaving charging before sufficient charge"
        )
        def prevent_premature_ev_switching(m, ev, t):
            """
            If an EV was charging at t-1 and is still available at t,
            it must continue charging unless it has sufficient charge for travel.
            """
            if t == m.time_steps.first():
                return pyo.Constraint.Skip
            
            ev_availability = getattr(m, f"{ev}_availability", None)
            if ev_availability is None:
                return pyo.Constraint.Skip
            
            # Only apply when EV is available in both periods
            if ev_availability[t] == 0 or ev_availability[t-1] == 0:
                return pyo.Constraint.Skip
            
            # Check if this EV was assigned to any station at t-1
            was_assigned_anywhere = sum(m.is_assigned[ev, cs, t-1] for cs in m.charging_stations)
            
            # If was assigned and doesn't have sufficient charge, must remain assigned
            return (sum(m.is_assigned[ev, cs, t] for cs in m.charging_stations) >= 
                    was_assigned_anywhere - m.has_sufficient_charge_for_travel[ev, t-1])

        # CONSTRAINT 25: Future improvement placeholder
        # PURPOSE: Advanced queue priority logic to minimize switching
        # CHALLENGE: HiGHS solver requires linear constraints only
        # CURRENT STATUS: Basic switching prevention is handled by constraints 20, 22, and 24
        # 
        # FOR FUTURE IMPLEMENTATION: Would require additional binary variables and 
        # linearization techniques to make it compatible with linear solvers
        # 
        # The current constraints already provide:
        # - Correct future usage calculation (Constraint 20)
        # - Charging continuity until sufficient charge (Constraint 22) 
        # - Prevention of premature switching (Constraint 24)
        # This combination significantly reduces unnecessary switching
        
