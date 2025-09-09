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
    # Unified Big-M and epsilon values for consistency
    BIG_M = 1e6
    EPSILON = 1e-3
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

        if self.is_prosumer:
            # ---- Built-in FCR configuration (hardcoded German FCR) ----
            # Active only if self.is_prosumer is True
            self.fcr_enabled = bool(self.is_prosumer)
            self.fcr_symmetric = False          # set True if you want symmetric product by default
            self._FCR_BLOCK_LENGTH = 1         # hours, fixed TSO blocks
            self._FCR_MIN_BID_MW   = 0.01        # minimum capacity per block
            self._FCR_STEP_MW      = 0.01        # increment: bids in integer MW
            # expected FCR value (€/MW/h) as hourly series; if not provided, assume zeros
            try:
                self.fcr_price_eur_per_mw_h = self.forecaster["fcr_price"]
            except KeyError:
                self.fcr_price_eur_per_mw_h = None
        
        # Helper method for safe EV property access
        self._ev_cache = {}

        # Fetch electricity price forecast
        try:
            self.electricity_price = self.forecaster["electricity_price"]
        except KeyError:
            raise ValueError("Forecaster must provide 'electricity_price'.")

        self.electricity_price_flex = self.forecaster["electricity_price_flex"]
        self.congestion_signal = self.forecaster["congestion_signal"]
        self.renewable_utilisation_signal = self.forecaster["RE_availability"]
        self.unidirectional_load = self.forecaster["unidirectional_load"]
        
        # Emergency signal and incentive for objective function calculation
        try:
            self.emergency_signal = self.forecaster["emergency_signal"]
            self.incentive = self.forecaster["incentive"]
        except KeyError:
            # Default values if not provided in forecaster
            self.emergency_signal = [0] * len(self.electricity_price)
            self.incentive = [50] * len(self.electricity_price)
        
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

        # Build EV property cache for efficient access
        self._build_ev_cache()
        self.is_prosumer = str_to_bool(is_prosumer)

        if self.is_prosumer:
            # ---- Built-in FCR configuration (hardcoded German FCR) ----
            # Active only if self.is_prosumer is True
            self.fcr_enabled = bool(self.is_prosumer)
            self.fcr_symmetric = False          # set True if you want symmetric product by default
            self._FCR_BLOCK_LENGTH = 1          # hours, fixed TSO blocks
            self._FCR_MIN_BID_MW   = 0.01        # minimum capacity per block
            self._FCR_STEP_MW      = 0.01        # increment: bids in integer MW
            # expected FCR value (€/MW/h) as hourly series; if not provided, assume zeros
            self.fcr_price_eur_per_mw_h = self.forecaster["fcr_price"]
        
        # Setup the Pyomo model after configuration is complete
        self.setup_model(presolve=True)
    
    def _build_ev_cache(self):
        """Build cache of EV properties for efficient access"""
        for ev_key in self.components:
            if ev_key.startswith("electric_vehicle"):
                ev_config = self.components[ev_key]
                self._ev_cache[ev_key] = {
                    'power_flow_directionality': ev_config.get('power_flow_directionality', 'unidirectional'),
                    'is_prosumer_capable': self.is_prosumer  # Use depot-level prosumer status
                }
    
    def get_ev_property(self, ev_key: str, property_name: str, default_value=None):
        """Helper function to safely get EV properties from cache"""
        if ev_key in self._ev_cache:
            return self._ev_cache[ev_key].get(property_name, default_value)
        return default_value
    
    def _build_compatibility_matrix(self):
        """
        Build compatibility matrix based on power flow directionality.
        Rules:
        - Bidirectional EV can ONLY connect to bidirectional CS
        - Unidirectional EV can ONLY connect to unidirectional CS
        """
        compatibility_init = {}
        
        for ev in self.model.evs:
            ev_directionality = self.get_ev_property(ev, 'power_flow_directionality', 'unidirectional')
            
            for cs in self.model.charging_stations:
                cs_component = self.components.get(cs, {})
                
                # Handle both dict and object cases
                if isinstance(cs_component, dict):
                    cs_directionality = cs_component.get('power_flow_directionality', 'unidirectional')
                elif hasattr(cs_component, 'power_flow_directionality'):
                    cs_directionality = cs_component.power_flow_directionality
                else:
                    cs_directionality = 'unidirectional'
                
                # Compatibility rule: EV and CS must have same directionality
                is_compatible = 1 if ev_directionality == cs_directionality else 0
                compatibility_init[(ev, cs)] = is_compatible
        
        # Add compatibility parameter to model
        self.model.compatible = pyo.Param(
            self.model.evs,
            self.model.charging_stations,
            initialize=compatibility_init,
            within=pyo.Binary,
            doc="Power flow directionality compatibility: 1 if EV-CS compatible, 0 otherwise",
        )

    def define_variables(self):
        """
        Defines the decision variables for the Pyomo optimization model.

        - `total_power_input`: Represents the total power input required at each time step.
        - `variable_cost`: Represents the variable cost associated with power usage at each time step.

        Both variables are defined over the `time_steps` set and are continuous real numbers.
        """
        self.model.total_power_input = pyo.Var(self.model.time_steps, within=pyo.Reals)
        self.model.total_power_output = pyo.Var(self.model.time_steps, within=pyo.Reals)
        self.model.variable_cost = pyo.Var(self.model.time_steps, within=pyo.Reals)
        self.model.variable_rev = pyo.Var(self.model.time_steps, within=pyo.Reals)
        self.model.net_income = pyo.Var(self.model.time_steps, within=pyo.Reals)
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

        # Emergency signal and incentive parameters for objective function calculation
        self.model.emergency_signal = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.emergency_signal)},
            doc="Emergency signal for demand response"
        )
        
        self.model.incentive = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.incentive)},
            doc="Incentive parameter for objective function calculations"
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
        
        # EV-CS Compatibility based on power flow directionality
        self._build_compatibility_matrix()
        
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
        
        # NEW: CS charge balance for bidirectional charging stations
        @self.model.Constraint(
            self.model.charging_stations,
            self.model.time_steps,
            doc="Charging station charge balance for bidirectional stations"
        )
        def station_charge_balance(m, cs, t):
            if cs in m.dsm_blocks and hasattr(m.dsm_blocks[cs], "charge"):
                # For bidirectional charging stations, CS charge should equal sum of EV discharges
                cs_component = self.components.get(cs, {})
                
                # Handle both dict and object cases
                if isinstance(cs_component, dict):
                    is_bidirectional_cs = cs_component.get('power_flow_directionality', 'unidirectional') == 'bidirectional'
                elif hasattr(cs_component, 'power_flow_directionality'):
                    is_bidirectional_cs = cs_component.power_flow_directionality == 'bidirectional'
                else:
                    is_bidirectional_cs = False
                
                if is_bidirectional_cs:
                    # For bidirectional charging stations, charge equals total discharge from bidirectional prosumer EVs
                    total_ev_discharge = sum(
                        m.dsm_blocks[ev].discharge[t]
                        for ev in m.evs
                        if self.get_ev_property(ev, 'power_flow_directionality', 'unidirectional') == 'bidirectional'
                        and self.get_ev_property(ev, 'is_prosumer_capable', False)
                    )
                    return m.dsm_blocks[cs].charge[t] == total_ev_discharge
                else:
                    # Unidirectional charging stations cannot charge from EVs
                    return m.dsm_blocks[cs].charge[t] == 0
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
        def total_power_output_constraint(m, t):
            cs_charge = sum(
                m.dsm_blocks[cs].charge[t]
                for cs in m.dsm_blocks
                if cs.startswith("charging_station")
            )
            return m.total_power_output[t] == cs_charge        

        @self.model.Constraint(self.model.time_steps)
        def variable_cost_constraint(m, t):
            """
            Calculates the variable cost associated with power usage at each time step.

            This constraint multiplies the total variable power by the corresponding electricity price.

            Args:
                m: Pyomo model reference.
                t: Current time step.

            Returns:
                Equality condition defining the variable cost.
            """
            return m.variable_cost[t] == m.total_power_input[t] * m.electricity_price[t]
        #  9.1: Revenue calculation constraint
        # This constraint calculates the revenue generated from the total power input at each time step.
        # It is similar to the variable cost constraint but uses the flexible electricity price.  
        

        @self.model.Constraint(self.model.time_steps)
        def rev_constraint(m, t):
            """
            Calculates the total variable revenue from all EVs at each time step.
            
            Revenue is generated when bidirectional prosumer-capable EVs discharge
            energy back to the grid through charging stations.

            Args:
                m: Pyomo model reference.
                t: Current time step.

            Returns:
                Equality condition defining the total variable revenue.
            """
            if not self.is_prosumer:
                return m.variable_rev[t] == 0
            
            total_revenue = 0
            
            for ev in m.evs:
                # Use helper method for safe property access
                is_bidirectional = self.get_ev_property(ev, 'power_flow_directionality', 'unidirectional') == 'bidirectional'
                is_prosumer_capable = self.get_ev_property(ev, 'is_prosumer_capable', False)
                
                # Only generate revenue for bidirectional prosumer-capable EVs
                if is_bidirectional and is_prosumer_capable:
                    # Revenue = EV discharge * flexible electricity price
                    # No need for availability factor - if EV is not available, it cannot discharge
                    total_revenue += (m.dsm_blocks[ev].discharge[t] * m.electricity_price_flex[t])
            
            return m.variable_rev[t] == total_revenue        
              
        
        @self.model.Constraint(self.model.time_steps)
        def net_income_constraint(m, t):
            """
            Calculates net income as the difference between revenue and variable cost.
            
            Args:
                m: Pyomo model reference.
                t: Current time step.
                
            Returns:
                Equality condition defining net income = revenue - cost.
            """
            # Simplified logic: always use revenue - cost
            # Revenue will be 0 if no prosumer capability exists
            return m.net_income[t] == m.variable_rev[t] - m.variable_cost[t]
        
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
            return m.charge_assignment[ev, cs, t] <= self.BIG_M * m.is_assigned[ev, cs, t]
        
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
                charge_sum = sum(
                    m.charge_assignment[ev, cs, t] for cs in m.charging_stations
                )
                return charge_sum <= self.BIG_M * ev_availability[t]
            return pyo.Constraint.Skip
        
        #  16: Power flow directionality compatibility constraint
        @self.model.Constraint(
            self.model.evs,
            self.model.charging_stations,
            self.model.time_steps,
            doc="EV can only be assigned to compatible CS based on power flow directionality"
        )
        def directionality_compatibility_constraint(m, ev, cs, t):
            """
            Ensures that:
            - Bidirectional EVs can ONLY use bidirectional charging stations
            - Unidirectional EVs can ONLY use unidirectional charging stations
            """
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
            if ev in m.dsm_blocks and hasattr(m.dsm_blocks[ev], "max_capacity"):
                # If cumulative charge reaches max capacity, set fully charged flag
                return (
                    m.cumulative_charged[ev, t]
                    >= m.dsm_blocks[ev].max_capacity - self.EPSILON
                    - (1 - m.is_fully_charged[ev, t]) * self.BIG_M
                )
            return pyo.Constraint.Skip
        
        #  19: Fully charged flag - reverse constraint
        @self.model.Constraint(self.model.evs, self.model.time_steps)
        def enforce_fully_charged_flag_reverse(m, ev, t):
            if ev in m.dsm_blocks and hasattr(m.dsm_blocks[ev], "max_capacity"):
                # If cumulative charge is less than max capacity, fully charged flag should be 0
                return (
                    m.cumulative_charged[ev, t]
                    <= m.dsm_blocks[ev].max_capacity - self.EPSILON
                    + m.is_fully_charged[ev, t] * self.BIG_M
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
                    current_soc >= next_usage_total - self.EPSILON - (1 - m.has_sufficient_charge_for_travel[ev, t]) * self.BIG_M
                )
            return pyo.Constraint.Skip
        
        # SIMPLIFIED CHARGING CONTINUITY CONSTRAINT
        # Consolidates previous constraints 22 and 24 into one unified constraint
        @self.model.Constraint(
            self.model.evs,
            self.model.time_steps,
            doc="Unified charging continuity: EVs continue charging until sufficient charge or unavailable"
        )
        def unified_charging_continuity(m, ev, t):
            """
            Simplified charging continuity logic:
            - If EV was charging at t-1 and is still available at t
            - It must continue to be assigned (somewhere) unless it has sufficient charge for travel
            """
            if t == m.time_steps.first():
                return pyo.Constraint.Skip
            
            ev_availability = getattr(m, f"{ev}_availability", None)
            if ev_availability is None:
                return pyo.Constraint.Skip
            
            # Only apply when EV is available in both time periods
            if ev_availability[t] == 0 or ev_availability[t-1] == 0:
                return pyo.Constraint.Skip
            
            # Check if EV was assigned to any station at t-1
            was_assigned_prev = sum(m.is_assigned[ev, cs, t-1] for cs in m.charging_stations)
            # Check if EV is assigned to any station at t
            is_assigned_current = sum(m.is_assigned[ev, cs, t] for cs in m.charging_stations)
            
            # If was charging and doesn't have sufficient charge, must remain assigned
            return is_assigned_current >= was_assigned_prev - m.has_sufficient_charge_for_travel[ev, t-1]
        
        # CONSTRAINT: Optimal charging station utilization
        @self.model.Constraint(
            self.model.time_steps,
            doc="Ensure efficient utilization of charging stations"
        )
        def optimal_station_utilization(m, t):
            """
            Optimal utilization: use available capacity efficiently
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
            
            # Total assignments
            total_assignments = sum(
                m.is_assigned[ev, cs, t] 
                for ev in evs_list 
                for cs in m.charging_stations
            )
            
            total_charging_stations = len(m.charging_stations)
            
            # Utilize stations efficiently
            if available_evs <= total_charging_stations:
                return total_assignments >= available_evs
            else:
                return total_assignments >= total_charging_stations

        # CONSTRAINT 25: Prosumer discharge restriction for bidirectional EVs
        # PURPOSE: Bidirectional EVs can only discharge if depot is_prosumer=Yes
        # LOGIC: bidirectional + is_prosumer=No -> discharge = 0
        if not self.is_prosumer:
            @self.model.Constraint(self.model.evs, self.model.time_steps)
            def non_prosumer_discharge_restriction(m, ev, t):
                """
                Non-prosumer bidirectional EVs cannot discharge energy.
                Only prosumer-enabled depots allow bidirectional EVs to discharge.
                """
                # Check if EV is bidirectional
                is_bidirectional = self.get_ev_property(ev, 'power_flow_directionality', 'unidirectional') == 'bidirectional'
                
                if is_bidirectional and ev in m.dsm_blocks:
                    # Force discharge to 0 for bidirectional EVs in non-prosumer depots
                    return m.dsm_blocks[ev].discharge[t] == 0
                else:
                    # Skip constraint for unidirectional EVs (already handled in dst_components.py)
                    return pyo.Constraint.Skip
        
        # CONSTRAINT 26: Future improvement placeholder
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

        # Add FCR capacity market if prosumer is enabled
        if self.is_prosumer:
            self._add_fcr_capacity_market(self.model)
        
    def _add_fcr_capacity_market(self, model):
        """
        Hardcoded German FCR: fixed 4h blocks at 00,04,08,12,16,20.
        Bids in integer MW with 1 MW minimum. Profit objective replaces cost objective.
        Activates only when self.is_prosumer is True.
        """
        if not self.is_prosumer:
            return

        import math
        m = model
        L = self._FCR_BLOCK_LENGTH
        step = self._FCR_STEP_MW
        min_bid = self._FCR_MIN_BID_MW

        # ---- Build fixed 4h blocks aligned to wall-clock hours ----
        # Use actual timestamps to pick starts at hours % 4 == 0 (TSO fixed windows)
        ts = self.index  # DatetimeIndex aligned with time_steps (0..T-1)
        starts_idx = [i for i, t in enumerate(ts[:-L+1]) if (t.hour % 4 == 0 and t.minute == 0)]
        # keep only starts that have all L hours inside horizon
        starts_idx = [i for i in starts_idx if i + L <= len(ts)]

        # map to model's time step ids
        m.fcr_blocks = pyo.Set(initialize=starts_idx, ordered=True)

        # ---- Block prices: sum of hourly €/MW/h inside each 4h block ----
        if self.fcr_price_eur_per_mw_h is None:
            price_hourly = [0.0] * len(ts)
        else:
            # ensure list of floats length == horizon
            price_hourly = [float(x) for x in self.fcr_price_eur_per_mw_h]

        block_price = {b: sum(price_hourly[b + k] for k in range(L)) for b in starts_idx}
        m.fcr_block_price = pyo.Param(m.fcr_blocks, initialize=block_price, mutable=False)

        # ---- Static plant-wide min/max electric power envelope (headroom/footroom) ----
        # For bus depot: use charging station capacities from DSM blocks (like heat pump example)
        max_cap = 0.0
        min_cap = 0.0
        
        # Sum all charging station capacities from DSM blocks (following orjinal pattern)
        for tech_key in self.components:
            if tech_key.startswith("charging_station"):
                if hasattr(m.dsm_blocks[tech_key], 'max_power'):
                    max_cap += float(m.dsm_blocks[tech_key].max_power)
                if hasattr(m.dsm_blocks[tech_key], 'min_power'):
                    min_cap += float(m.dsm_blocks[tech_key].min_power)

        self.max_plant_capacity = max_cap
        self.min_plant_capacity = min_cap

        # ---- Variables: integerized capacities with 1 MW steps ----
        M_blocks = int(math.ceil(max(1.0, max_cap) / step))  # conservative big-M
        m.k_up = pyo.Var(m.fcr_blocks, within=pyo.NonNegativeIntegers)
        m.k_dn = pyo.Var(m.fcr_blocks, within=pyo.NonNegativeIntegers)
        m.cap_up = pyo.Var(m.fcr_blocks, within=pyo.NonNegativeReals)
        m.cap_dn = pyo.Var(m.fcr_blocks, within=pyo.NonNegativeReals)
        m.energy_cost = pyo.Var(m.fcr_blocks, within=pyo.NonNegativeReals)
        m.bid_up = pyo.Var(m.fcr_blocks, within=pyo.Binary)
        m.bid_dn = pyo.Var(m.fcr_blocks, within=pyo.Binary)

        @m.Constraint(m.fcr_blocks)
        def cap_up_is_steps(mm, b):  return mm.cap_up[b] == step * mm.k_up[b]
        @m.Constraint(m.fcr_blocks)
        def cap_dn_is_steps(mm, b):  return mm.cap_dn[b] == step * mm.k_dn[b]

        # activate counts only if bid is active; enforce min bid if active
        @m.Constraint(m.fcr_blocks)
        def up_count_active(mm, b):  return mm.k_up[b] <= M_blocks * mm.bid_up[b]
        @m.Constraint(m.fcr_blocks)
        def dn_count_active(mm, b):  return mm.k_dn[b] <= M_blocks * mm.bid_dn[b]
        @m.Constraint(m.fcr_blocks)
        def up_min_bid(mm, b):       return mm.cap_up[b] >= min_bid * mm.bid_up[b]
        @m.Constraint(m.fcr_blocks)
        def dn_min_bid(mm, b):       return mm.cap_dn[b] >= min_bid * mm.bid_dn[b]

        # symmetric option (both directions equal & jointly on/off)
        if self.fcr_symmetric:
            @m.Constraint(m.fcr_blocks)
            def sym_cap(mm, b):   return mm.cap_up[b] == mm.cap_dn[b]
            @m.Constraint(m.fcr_blocks)
            def sym_onoff(mm, b): return mm.bid_up[b] == mm.bid_dn[b]

        # at most one direction per block for asymmetric product (relax/remove if market allows both)
        if not self.fcr_symmetric:
            @m.Constraint(m.fcr_blocks)
            def one_dir(mm, b): return mm.bid_up[b] + mm.bid_dn[b] <= 1

        # ---- Feasibility across all hours inside each 4h block ----
        m.fcr_up_feas   = pyo.ConstraintList()
        m.fcr_down_feas = pyo.ConstraintList()
        for b in starts_idx:
            for k in range(L):
                t = b + k
                # Calculate charging station power at time t (like orjinal total_power_input)
                
                # up: need headroom; down: need footroom (orjinal pattern)
                m.fcr_up_feas.add(   m.cap_up[b]   <= self.max_plant_capacity - m.total_power_output[t] )
                m.fcr_down_feas.add( m.cap_dn[b]   <=  m.total_power_output[t] - self.min_plant_capacity )

        @ m.Expression()
        def fcr_revenue(mm):
            if self.fcr_symmetric:
                # up==down enforced elsewhere; pay once per MW of symmetric capacity
                return sum(mm.fcr_block_price[b] * mm.cap_up[b] for b in mm.fcr_blocks)
            else:
                # asymmetric: sum the two capacities, constraints will force one to zero if needed
                return (
                    sum(mm.fcr_block_price[b] * mm.cap_up[b]   for b in mm.fcr_blocks) +
                    sum(mm.fcr_block_price[b] * mm.cap_dn[b]   for b in mm.fcr_blocks)
                )

    def plot_bus_depot_fcr(self, instance, save_path=None, show=True):
        """
        Three-panel figure for bus depot with FCR integration:
        1) EV charging/discharging + electricity prices (twin y-axis) 
        2) Charging station power output + EV status
        3) FCR market view (per 4h block): capacities (up/down), block price,
           and operational baseline for feasibility context

        Designed for bus depot with electric_vehicle and charging_station technologies.
        """
        
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import pandas as pd

        # ------- style -------
        mpl.rcParams.update({
            "font.size": 13, "font.family": "serif",
            "axes.titlesize": 15, "axes.labelsize": 13,
            "legend.fontsize": 12, "lines.linewidth": 2,
            "axes.grid": True, "grid.linestyle": "--",
            "grid.alpha": 0.7, "figure.dpi": 120
        })

        T = list(instance.time_steps)

        # ------- helpers -------
        def safe_value(v):
            try:
                return float(pyo.value(v))
            except Exception:
                return float(v) if v is not None else 0.0

        def series_or_none(block, name):
            if block is None or not hasattr(block, name):
                return None
            return [safe_value(getattr(block, name)[t]) for t in instance.time_steps]

        def plot_if_nonzero(ax, x, y, label, color, style="-", eps=1e-9):
            if y is None:
                return None
            if any(abs(v) > eps for v in y):
                return ax.plot(x, y, label=label, color=color, linestyle=style)[0]
            return None

        def stairs_from_blocks(block_starts, per_block_values, block_len, horizon_len):
            """
            Expand block-level value (constant over 4h) to hourly stair array.
            """
            y = [0.0] * horizon_len
            for b in block_starts:
                val = safe_value(per_block_values.get(b, 0.0))
                for k in range(block_len):
                    idx = b + k
                    if 0 <= idx < horizon_len:
                        y[idx] = val
            return y

        # ------- Bus Depot Components -------
        B = instance.dsm_blocks
        
        # Electric vehicles
        evs = [key for key in B.keys() if key.startswith("electric_vehicle")]
        ev_charge_data = {}
        ev_discharge_data = {}
        ev_usage_data = {}
        
        for ev in evs:
            ev_block = B[ev]
            ev_charge_data[ev] = series_or_none(ev_block, "charge")
            ev_discharge_data[ev] = series_or_none(ev_block, "discharge") 
            ev_usage_data[ev] = series_or_none(ev_block, "usage")
        
        # Charging stations
        charging_stations = [key for key in B.keys() if key.startswith("charging_station")]
        cs_discharge_data = {}
        
        for cs in charging_stations:
            cs_block = B[cs]
            cs_discharge_data[cs] = series_or_none(cs_block, "discharge")

        # prices
        elec_price = [safe_value(instance.electricity_price[t]) for t in T] if hasattr(instance, "electricity_price") else None
        
        # baseline load and envelope (for FCR feasibility visualization)
        total_elec = [safe_value(instance.total_power_input[t]) for t in T] if hasattr(instance, "total_power_output") else None
        max_cap = getattr(self, "max_plant_capacity", None)
        min_cap = getattr(self, "min_plant_capacity", 0.0)

        # ------- figure -------
        nrows = 3
        fig, axs = plt.subplots(nrows, 1, figsize=(14, 12), sharex=True, constrained_layout=True)

        # ---------- TOP: EV operations + prices ----------
        colors = plt.cm.Set1(range(len(evs)))
        handles = []
        
        for i, ev in enumerate(evs):
            color = colors[i]
            h1 = plot_if_nonzero(axs[0], T, ev_charge_data[ev], f"{ev} Charge [kW]", color, "-")
            h2 = plot_if_nonzero(axs[0], T, ev_discharge_data[ev], f"{ev} Discharge [kW]", color, "--")
            h3 = plot_if_nonzero(axs[0], T, ev_usage_data[ev], f"{ev} Usage [kW]", color, ":")
            
            for h in [h1, h2, h3]:
                if h: handles.append(h)

        axs[0].set_ylabel("EV Power [kW]")
        axs[0].set_title("Electric Vehicle Operations & Electricity Price")
        axs[0].grid(True, which="both", axis="both")

        # Price on twin axis
        if elec_price:
            axp = axs[0].twinx()
            price_line = axp.plot(T, elec_price, label="Electricity Price [€/MWh]", color="red", linestyle="--", alpha=0.7)
            axp.set_ylabel("Price [€/MWh]", color="red")
            axp.tick_params(axis="y", labelcolor="red")
            handles.extend(price_line)

        if handles:
            axs[0].legend(handles, [h.get_label() for h in handles], loc="upper left", frameon=True)

        # ---------- MIDDLE: Charging Station Power ----------
        cs_colors = plt.cm.Set2(range(len(charging_stations)))
        cs_handles = []
        
        for i, cs in enumerate(charging_stations):
            color = cs_colors[i]
            h = plot_if_nonzero(axs[1], T, cs_discharge_data[cs], f"{cs} Power [kW]", color, "-")
            if h: cs_handles.append(h)

        axs[1].set_ylabel("Charging Station Power [kW]")
        axs[1].set_title("Charging Station Power Output")
        axs[1].grid(True, which="both", axis="both")
        
        if cs_handles:
            axs[1].legend(cs_handles, [h.get_label() for h in cs_handles], loc="upper right", frameon=True)

        # ---------- BOTTOM: FCR bids & impact ----------
        fcr_present = hasattr(instance, "fcr_blocks")
        fcr_len = getattr(self, "_FCR_BLOCK_LENGTH", 1)

        def zero_or_list(x, n):
            return [0.0]*n if x is None else x

        fcr_blocks_list = list(getattr(instance, "fcr_blocks", [])) if fcr_present else []
        cap_up_map, cap_dn_map, block_price_map = {}, {}, {}

        if fcr_present and fcr_blocks_list:
            for b in fcr_blocks_list:
                if hasattr(instance, "cap_up"): cap_up_map[b] = safe_value(instance.cap_up[b])
                if hasattr(instance, "cap_dn"): cap_dn_map[b] = safe_value(instance.cap_dn[b])
                if hasattr(instance, "fcr_block_price"): block_price_map[b] = safe_value(instance.fcr_block_price[b])

            H = len(T)
            cap_up_stairs = stairs_from_blocks(fcr_blocks_list, cap_up_map, fcr_len, H) if cap_up_map else [0.0]*H
            cap_dn_stairs = stairs_from_blocks(fcr_blocks_list, cap_dn_map, fcr_len, H) if cap_dn_map else [0.0]*H
            price_stairs = stairs_from_blocks(fcr_blocks_list, block_price_map, fcr_len, H) if block_price_map else [0.0]*H

            # Baseline & capability envelope
            base = zero_or_list(total_elec, len(T))
            axs[2].plot(T, base, color="0.3", lw=2.0, label="Baseline Power Demand [kW]")
            if max_cap is not None:
                axs[2].plot(T, [max_cap]*len(T), color="0.7", ls=":", label="Max Charging Capacity [kW]")
            if min_cap is not None:
                axs[2].plot(T, [min_cap]*len(T), color="0.7", ls="--", label="Min Capacity [kW]")

            # CAPACITY BANDS around the baseline
            # UP = can reduce load: red band between (base - up) and base
            lower_up = [max(min_cap if min_cap is not None else -1e9, base[i] - cap_up_stairs[i]) for i in range(len(T))]
            axs[2].fill_between(T, lower_up, base, color="#d62728", alpha=0.25, step="pre", label="FCR Up Capacity [kW]")

            # DOWN = can increase load: green band between base and (base + down) 
            upper_dn = [min(max_cap if max_cap is not None else 1e9, base[i] + cap_dn_stairs[i]) for i in range(len(T))]
            axs[2].fill_between(T, base, upper_dn, color="#2ca02c", alpha=0.25, step="pre", label="FCR Down Capacity [kW]")

            # FCR block price on twin axis
            axp2 = axs[2].twinx()
            axp2.plot(T, price_stairs, color="#9467bd", ls="--", lw=2, label="FCR Block Price [€/MW per 4h]")
            axp2.set_ylabel("FCR Price [€/MW per 4h]", color="purple")
            axp2.tick_params(axis="y", labelcolor="purple")

            # Shade active FCR blocks
            for b in fcr_blocks_list:
                active = (cap_up_map.get(b, 0.0) > 0.0) or (cap_dn_map.get(b, 0.0) > 0.0)
                if active:
                    axs[2].axvspan(b, min(b + fcr_len, len(T)), color="purple", alpha=0.1)

            # Legend
            h1, l1 = axs[2].get_legend_handles_labels()
            h2, l2 = axp2.get_legend_handles_labels()
            axs[2].legend(h1 + h2, l1 + l2, loc="upper left", frameon=True)

            axs[2].set_ylabel("Power [kW]")
            axs[2].set_title("FCR Capacity Bids (4h blocks) — Red = Up regulation, Green = Down regulation")
            axs[2].grid(True, which="both", axis="both")

        else:
            # No FCR - just show baseline
            _ = plot_if_nonzero(axs[2], T, total_elec, "Baseline Power Demand [kW]", "C1", "-")
            if max_cap is not None:
                axs[2].plot(T, [max_cap]*len(T), color="0.7", ls=":", label="Max Charging Capacity [kW]")
            if min_cap is not None:
                axs[2].plot(T, [min_cap]*len(T), color="0.7", ls="--", label="Min Capacity [kW]")
            axs[2].set_ylabel("Power [kW]")
            axs[2].set_title("Operational Baseline (FCR not active)")
            axs[2].grid(True, which="both", axis="both")
            axs[2].legend(loc="upper left", frameon=True)

        axs[-1].set_xlabel("Time Step")
        fig.autofmt_xdate()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

        # -------- CSV export (bus depot specific) --------
        df_dict = {"time_step": T}
        
        # EV data
        for ev in evs:
            if ev_charge_data[ev]: df_dict[f"{ev}_charge_kW"] = ev_charge_data[ev]
            if ev_discharge_data[ev]: df_dict[f"{ev}_discharge_kW"] = ev_discharge_data[ev]
            if ev_usage_data[ev]: df_dict[f"{ev}_usage_kW"] = ev_usage_data[ev]
        
        # Charging station data
        for cs in charging_stations:
            if cs_discharge_data[cs]: df_dict[f"{cs}_power_kW"] = cs_discharge_data[cs]
        
        # General data
        if elec_price: df_dict["electricity_price_EUR_MWh"] = elec_price
        if total_elec: df_dict["total_power_demand_kW"] = total_elec
        
        # FCR data
        if fcr_present and fcr_blocks_list:
            if cap_up_stairs: df_dict["FCR_up_capacity_kW"] = [x for x in cap_up_stairs]
            if cap_dn_stairs: df_dict["FCR_down_capacity_kW"] = [x for x in cap_dn_stairs]
            if price_stairs: df_dict["FCR_block_price_EUR_MW_4h"] = price_stairs

        df = pd.DataFrame(df_dict)

        if save_path:
            csv_path = save_path.rsplit(".", 1)[0] + "_bus_depot_fcr.csv"
            df.to_csv(csv_path, index=False)
            print(f"Bus depot FCR data exported to: {csv_path}")

        return df

