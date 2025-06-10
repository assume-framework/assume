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

    def define_parameters(self):
        """Defines Pyomo parameters based on fetched forecasts and configurations."""
        self.model.electricity_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.electricity_price)},
            doc="Electricity price forecast",
        )

        # Define range parameter for each EV
        for (
            ev
        ) in self.model.evs:  # Assuming self.model.evs is populated during setup_model
            if ev in self.components and "range" in self.components[ev]:
                ev_range_data = self.components[ev]["range"]
                # Ensure ev_range_data is indexable (like a pandas Series or list)
                try:
                    init_dict = {
                        t: ev_range_data.iloc[t] for t in range(len(ev_range_data))
                    }
                except AttributeError:  # Handle if it's a list/tuple
                    init_dict = {t: ev_range_data[t] for t in range(len(ev_range_data))}
                except Exception as e:
                    raise TypeError(f"Could not process range data for {ev}: {e}")

                # Use add_component to avoid potential conflicts if param already exists
                param_name = f"{ev}_range"
                if hasattr(self.model, param_name):
                    logger.warning(
                        f"Parameter {param_name} already exists. Re-defining."
                    )
                    delattr(self.model, param_name)

                setattr(
                    self.model,
                    param_name,
                    pyo.Param(
                        self.model.time_steps,
                        initialize=init_dict,
                        doc=f"Range forecast for {ev}",
                    ),
                )
            else:
                logger.error(
                    f"Range data missing for EV component {ev} during parameter definition."
                )
                # Depending on requirements, you might raise an error here:
                # raise ValueError(f"Range data missing for EV component {ev}")

        # Default: all compatible
        self.model.compatible = pyo.Param(
            self.model.evs,
            self.model.charging_stations,
            initialize=1,
            within=pyo.Binary,
            mutable=True,
            doc="Default compatibility: all EVs are compatible with all stations",
        )  ##comptatible should go to variable

    def define_variables(self):
        """Defines Pyomo variables for optimization."""
        self.model.total_power_input = pyo.Var(
            self.model.time_steps,
            within=pyo.Reals,
            doc="Total power drawn from the grid at each time step",
        )
        self.model.variable_cost = pyo.Var(
            self.model.time_steps,
            within=pyo.Reals,
            doc="Variable cost incurred at each time step",
        )
        self.model.is_assigned = pyo.Var(
            self.model.evs,
            self.model.charging_stations,
            self.model.time_steps,
            domain=pyo.Binary,
            doc="1 if EV is assigned to CS at time t",
        )

        # Binary variable: 1 if EV is waiting (available but not assigned), 0 otherwise
        self.model.in_queue = pyo.Var(
            self.model.evs,
            self.model.time_steps,
            domain=pyo.Binary,
            doc="EV is waiting in queue",
        )

        # Variable representing charge amount assigned from a specific CS to a specific EV at time t
        # This assumes self.model.evs and self.model.charging_stations are defined in setup_model
        self.model.charge_assignment = pyo.Var(
            self.model.evs,
            self.model.charging_stations,
            self.model.time_steps,
            domain=pyo.NonNegativeReals,  # Charge power must be non-negative
            doc="Power transferred from CS to EV at time t",
        )

        self.model.cumulative_charged = pyo.Var(
            self.model.evs, self.model.time_steps, domain=pyo.NonNegativeReals
        )

        self.model.is_available = pyo.Var(
            self.model.evs,
            self.model.time_steps,
            domain=pyo.Binary,
            doc="Binary indicator: 1 if EV is available (at depot), 0 otherwise",
        )
        self.model.is_fully_charged = pyo.Var(
            self.model.evs,
            self.model.time_steps,
            domain=pyo.Binary,
            doc="Binary indicator if EV has reached full charge",
        )

    def initialize_process_sequence(self):
        """Defines constraints linking components, e.g., EV charge and CS discharge."""

        for ev in self.model.dsm_blocks:
            if ev.startswith("electric_vehicle"):
                # Check if dsm_blocks[ev] has a 'charge' attribute
                if hasattr(self.model.dsm_blocks[ev], "charge"):
                    constraint_name = f"charge_flow_constraint_{ev}"
                    # Remove constraint if it exists before adding
                    if hasattr(self.model, constraint_name):
                        delattr(self.model, constraint_name)
                    setattr(
                        self.model,
                        constraint_name,
                        pyo.Constraint(
                            self.model.time_steps,
                            rule=lambda m, t, ev=ev: m.dsm_blocks[ev].charge[t] >= 0,
                            doc=f"Ensures non-negative charge for {ev}",
                        ),
                    )
                else:
                    logger.warning(
                        f"DSM block {ev} does not have 'charge' attribute for constraint."
                    )

        # Constraint: Total charge received by an EV at time t must equal the sum of charge
        # assigned to it from all charging stations.
        @self.model.Constraint(
            self.model.evs,
            self.model.time_steps,
            doc="Links EV total charge to assignments from CSs",
        )
        def ev_total_charge(m, ev, t):
            # Ensure the EV exists in dsm_blocks and has 'charge' variable
            if ev in m.dsm_blocks and hasattr(m.dsm_blocks[ev], "charge"):
                return m.dsm_blocks[ev].charge[t] == sum(
                    m.charge_assignment[ev, cs, t] for cs in m.charging_stations
                )
            else:
                logger.warning(
                    f"Skipping ev_total_charge constraint for {ev} at time {t} due to missing block/variable."
                )
                return pyo.Constraint.Skip  # Skip constraint if EV block/var is missing
            
        @self.model.Constraint(self.model.charging_stations, self.model.time_steps)
        def station_capacity_limit(m, cs, t):
            return (
                sum(m.charge_assignment[ev, cs, t] for ev in m.evs)
                <= m.dsm_blocks[cs].discharge[t]
            )

        # # Constraint: Total discharge from a charging station at time t must equal the sum of charge
        # # assigned from it to all EVs.
        # @self.model.Constraint(
        #     self.model.charging_stations,
        #     self.model.time_steps,
        #     doc="Links CS total discharge to assignments to EVs",
        # )
        # def station_discharge_balance(m, cs, t):
        #     # Ensure the CS exists in dsm_blocks and has 'discharge' variable
        #     if cs in m.dsm_blocks and hasattr(m.dsm_blocks[cs], "discharge"):
        #         return m.dsm_blocks[cs].discharge[t] == sum(
        #             m.charge_assignment[ev, cs, t] for ev in m.evs
        #         )
        #     else:
        #         logger.warning(
        #             f"Skipping station_discharge_balance constraint for {cs} at time {t} due to missing block/variable."
        #         )
        #         return pyo.Constraint.Skip  # Skip constraint if CS block/var is missing

    def define_constraints(self):
        """Defines core operational constraints for the bus depot model."""
        # Constraint: If not a prosumer, total power input must be non-negative (no grid export).
        if not self.is_prosumer:

            @self.model.Constraint(
                self.model.time_steps, doc="Prevents grid export for non-prosumers"
            )
            def grid_export_constraint(m, t):
                return m.total_power_input[t] >= 0

        @self.model.Constraint(
            self.model.charging_stations,
            self.model.time_steps,
            doc="Max 1 EV per CS at each timestep",
        )
        def max_one_ev_per_station(m, cs, t):
            return sum(m.is_assigned[ev, cs, t] for ev in m.evs) <= 1

        @self.model.Constraint(
            self.model.evs,
            self.model.time_steps,
            doc="Queue logic: EV is in queue if available and not assigned",
        )
        def enforce_queue_logic(m, ev, t):
            return m.in_queue[ev, t] == m.is_available[ev, t] - sum(
                m.is_assigned[ev, cs, t] for cs in m.charging_stations
            )

        @self.model.Constraint(
            self.model.evs,
            self.model.charging_stations,
            self.model.time_steps,
            doc="Link is_assigned with actual charge flow",
        )
        def assignment_indicator_constraint(m, ev, cs, t):
            return m.charge_assignment[ev, cs, t] <= 1e5 * m.is_assigned[ev, cs, t]

        @self.model.Constraint(
            self.model.evs,
            self.model.time_steps,
            doc="Ensures EV cannot charge if it's not available (driving)",
        )
        def availability_based_on_charging(m, ev, t):
            M = 1e5  # A sufficiently large number (e.g., max possible charge rate * number of CS)
            charge_sum = sum(
                m.charge_assignment[ev, cs, t] for cs in m.charging_stations
            )

            # Assuming is_available = 1 means at depot/can charge
            return charge_sum <= M * m.is_available[ev, t]
        
        # # Need to be decided
        # @self.model.Constraint(self.model.evs, self.model.time_steps)
        # def link_ev_charge_from_assignment(m, ev, t):
        #     return m.dsm_blocks[ev].charge[t] == sum(
        #         m.charge_assignment[ev, cs, t] for cs in m.charging_stations
        #     )
        

        @self.model.Constraint(self.model.evs, self.model.time_steps)
        def cumulative_charging_tracking(m, ev, t):
            if t == m.time_steps.first():
                return m.cumulative_charged[ev, t] == m.dsm_blocks[ev].charge[t]
            else:
                return (
                    m.cumulative_charged[ev, t]
                    == m.cumulative_charged[ev, t - 1] + m.dsm_blocks[ev].charge[t]
                )

        @self.model.Constraint(
            self.model.evs, self.model.charging_stations, self.model.time_steps
        )
        def release_station_if_queue_waiting(m, ev, cs, t):
            return (
                m.is_assigned[ev, cs, t]
                <= 1 - m.in_queue[ev, t] * m.is_fully_charged[ev, t]
            )

        @self.model.Constraint(self.model.evs, self.model.time_steps)
        def enforce_fully_charged_flag(m, ev, t):
            M = 1e5
            ε = 1e-3
            return (
                m.cumulative_charged[ev, t]
                >= m.dsm_blocks[ev].max_capacity
                - ε
                - (1 - m.is_fully_charged[ev, t]) * M
            )

        @self.model.Constraint(self.model.evs, self.model.time_steps)
        def track_cumulative_charge(m, ev, t):
            if t == m.time_steps.first():
                return m.cumulative_charged[ev, t] == sum(
                    m.charge_assignment[ev, cs, t] for cs in m.charging_stations
                )
            return m.cumulative_charged[ev, t] == m.cumulative_charged[ev, t - 1] + sum(
                m.charge_assignment[ev, cs, t] for cs in m.charging_stations
            )

        @self.model.Constraint(
            self.model.time_steps,
            doc="Defines total power input based on component flows",
        )
        def total_power_input_constraint(m, t):
            ev_charge_sum = sum(
                m.dsm_blocks[ev].charge[t]
                for ev in m.evs
                if ev in m.dsm_blocks and hasattr(m.dsm_blocks[ev], "charge")
            )

            cs_charge_sum = sum(
                m.dsm_blocks[cs].charge[t]
                for cs in m.charging_stations
                if cs in m.dsm_blocks and hasattr(m.dsm_blocks[cs], "charge")
            )

            cs_discharge_sum = sum(
                m.dsm_blocks[cs].discharge[t]
                for cs in m.charging_stations
                if cs in m.dsm_blocks and hasattr(m.dsm_blocks[cs], "discharge")
            )
            # return m.total_power_input[t] == ev_charge_sum # Assumes CSs are lossless passthrough
            return (
                m.total_power_input[t] == cs_discharge_sum
            )  # Assumes CS discharge is what's drawn from grid perspective

        # Constraint: Defines the variable cost based on total power input and electricity price.
        @self.model.Constraint(
            self.model.time_steps,
            doc="Calculates variable cost based on power input and price",
        )
        def variable_cost_constraint(m, t):
            # Ensure price parameter exists for the time step
            if t in m.electricity_price:
                return (
                    m.variable_cost[t]
                    == m.total_power_input[t] * m.electricity_price[t]
                )
            else:
                logger.warning(
                    f"Skipping variable_cost_constraint at time {t} due to missing electricity price."
                )
                return pyo.Constraint.Skip
