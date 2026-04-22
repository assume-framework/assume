# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import pyomo.environ as pyo

from assume.common.base import SupportsMinMax
from assume.common.forecaster import SteelplantForecaster
from assume.units.dsm_load_shift import DSMFlex

logger = logging.getLogger(__name__)

# Set the log level to ERROR
logging.getLogger("pyomo").setLevel(logging.WARNING)


class SteelPlant(DSMFlex, SupportsMinMax):
    """
    The SteelPlant class represents a steel plant unit within an energy system, which can
    include various components like Direct Reduced Iron (DRI) plants,
    Electric Arc Furnaces (EAF), and other supporting technologies. The class models a unit
    that consumes energy for steel production and may also implement flexibility strategies
    like cost-based load shifting.

    Args:
        id (str): A unique identifier for the steel plant unit.
        unit_operator (str): The operator responsible for the steel plant.
        bidding_strategies (dict): A dictionary of bidding strategies, which define how the unit participates in energy markets.
        forecaster (Forecaster): A forecaster used to get key variables such as fuel or electricity prices.
        technology (str, optional): The technology of the steel plant. Default is "steel_plant".
        components (dict, optional): A dictionary describing the components of the steel plant, such as Electrolyser, DRI Plant, DRI Storage, and Electric Arc Furnace. Default is an empty dictionary.
        objective (str, optional): The objective function for the steel plant, typically focused on minimizing variable costs. Default is "min_variable_cost".
        flexibility_measure (str, optional): The flexibility measure for the steel plant, such as "cost_based_load_shift". Default is "cost_based_load_shift".
        demand (float, optional): The steel production demand, representing the amount of steel that needs to be produced. Default is 0.
        cost_tolerance (float, optional): The maximum allowable cost variation when shifting the load, used in flexibility measures. Default is 10.
        congestion_threshold (float, optional): The threshold for congestion management in the plant’s energy system. Default is 0.
        peak_load_cap (float, optional): The peak load capacity of the steel plant. Default is 0.
        load_profile_deviation (float, optional): Allowed deviation (±) from the normalized load profile, as a fraction (e.g., 0.05 for ±5%). Default is 1.0 (no constraint). Requires normalized_load_profile in forecasts_df.csv.
        node (str, optional): The network node where the steel plant is located in the energy system. Default is "node0".
        location (tuple[float, float], optional): A tuple representing the geographical coordinates (latitude, longitude) of the steel plant. Default is (0.0, 0.0).
        **kwargs: Additional keyword arguments that may be passed to support more specific configurations.

    Attributes:
        required_technologies (list): A list of required technologies for the plant to function, such as DRI plant and EAF.
        optional_technologies (list): A list of optional technologies that could enhance the plant, such as electrolyser or storage systems.
    """

    # Required and optional technologies for the steel plant
    required_technologies = ["dri_plant", "eaf"]
    optional_technologies = ["electrolyser", "hydrogen_buffer_storage", "dri_storage"]

    # Rolling-horizon extensibility hooks (DSMFlex)
    _demand_attr_suffix = "steel_demand"
    _extra_price_attrs = [
        "electricity_price",
        "hydrogen_price",
        "natural_gas_price",
        "steel_price",
        "iron_ore_price",
        "lime_price",
        "co2_price",
    ]
    _component_schema = {
        "eaf": ("power_in", "steel_output", "eaf_power_input", "eaf_steel_output"),
        "dri_plant": ("power_in", "dri_output", "dri_power_input", "dri_output"),
        "electrolyser": (
            "power_in",
            "hydrogen_out",
            "electrolyser_power",
            "hydrogen_prod",
        ),
    }

    def _primary_output_expr(self, m, t):
        """Steel production (tonnes) from the EAF is the tracked output."""
        return m.dsm_blocks["eaf"].steel_output[t]

    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        forecaster: SteelplantForecaster,
        components: dict[str, dict] = None,
        technology: str = "steel_plant",
        objective: str = "min_variable_cost",
        flexibility_measure: str = "cost_based_load_shift",
        demand: float = 0,
        cost_tolerance: float = 10,
        congestion_threshold: float = 0,
        peak_load_cap: float = 0,
        load_profile_deviation: float = 1.0,
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
        if not isinstance(forecaster, SteelplantForecaster):
            raise TypeError(
                f"forecaster must be of type {SteelplantForecaster.__name__}"
            )

        # check if the required components are present in the components dictionary
        for component in self.required_technologies:
            if component not in components.keys():
                raise ValueError(
                    f"Component {component} is required for the steel plant unit."
                )

        # check if the provided components are valid and do not contain any unknown components
        for component in components.keys():
            if (
                component not in self.required_technologies
                and component not in self.optional_technologies
            ):
                raise ValueError(
                    f"Components {component} is not a valid component for the steel plant unit."
                )

        # FIXME assuming only one market
        self.market_id = list(bidding_strategies.keys())[0]

        self.electricity_price = forecaster.electricity_price
        self.steel_demand = demand  # Global demand (total production by end of horizon)
        self.steel_demand_rolling = (
            None  # Will be used in rolling-horizon to track cumulative remaining demand
        )

        # Try to get per-timestep steel demand from forecaster (optional)
        try:
            self.steel_demand_per_timestep = forecaster.steel_demand
        except (AttributeError, KeyError):
            self.steel_demand_per_timestep = None

        self.natural_gas_price = self.forecaster.get_price("natural_gas")
        self.hydrogen_price = self.forecaster.get_price("hydrogen")
        self.iron_ore_price = self.forecaster.get_price("iron_ore")
        self.steel_price = self.forecaster.get_price("steel")
        self.lime_price = self.forecaster.get_price("lime")
        self.co2_price = self.forecaster.get_price("co2")

        # Calculate congestion forecast and set it as a forecast column in the forecaster
        self.congestion_signal = forecaster.congestion_signal
        self.renewable_utilisation_signal = forecaster.renewable_utilisation_signal

        self.objective = objective
        self.flexibility_measure = flexibility_measure
        self.cost_tolerance = cost_tolerance
        self.congestion_threshold = congestion_threshold
        self.peak_load_cap = peak_load_cap
        self.load_profile_deviation = load_profile_deviation

        # Try to get normalized load profile from forecaster (optional)
        self.normalized_load_profile = None
        try:
            self.normalized_load_profile = forecaster.normalized_load_profile
            if self.normalized_load_profile is not None:
                logger.info(
                    f"[LOAD-PROFILE] Loaded normalized_load_profile with {len(self.normalized_load_profile)} timesteps"
                )
            else:
                logger.info(
                    "[LOAD-PROFILE] No normalized_load_profile provided in forecaster"
                )
        except (AttributeError, KeyError):
            logger.info(
                "[LOAD-PROFILE] No normalized_load_profile provided in forecaster"
            )
        self.has_h2storage = "hydrogen_buffer_storage" in self.components.keys()
        self.has_dristorage = "dri_storage" in self.components.keys()
        self.has_electrolyser = "electrolyser" in self.components.keys()

        # Initialize the model
        # self.setup_model()  # NOTE: called in forecaster initialization again!!!

    def define_parameters(self):
        """
        Defines the parameters for the Pyomo model.
        """
        self.model.electricity_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.electricity_price)},
        )

        if self.components["dri_plant"]["fuel_type"] in ["natural_gas", "both"]:
            if self.has_electrolyser:
                self.model.hydrogen_price = pyo.Param(
                    self.model.time_steps,
                    initialize={t: 0 for t in self.model.time_steps},
                )

            else:
                self.model.hydrogen_price = pyo.Param(
                    self.model.time_steps,
                    initialize={
                        t: value for t, value in enumerate(self.hydrogen_price)
                    },
                )

        elif self.components["dri_plant"]["fuel_type"] in ["hydrogen", "both"]:
            if self.has_electrolyser:
                self.model.hydrogen_price = pyo.Param(
                    self.model.time_steps,
                    initialize={t: 0 for t in self.model.time_steps},
                )
            else:
                self.model.hydrogen_price = pyo.Param(
                    self.model.time_steps,
                    initialize={
                        t: value for t, value in enumerate(self.hydrogen_price)
                    },
                )

        self.model.natural_gas_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.natural_gas_price)},
        )

        # Use remaining_demand if available (rolling-horizon mode), otherwise use global steel_demand
        demand_value = self.steel_demand
        if "_remaining_demand" in self.components:
            demand_value = self.components["_remaining_demand"]
            logger.info(f"Using rolling-horizon remaining_demand: {demand_value}")

        self.model.steel_demand = pyo.Param(initialize=demand_value)

        # Optional: per-timestep steel demand from forecaster
        if self.steel_demand_per_timestep is not None:
            self.model.steel_demand_per_timestep = pyo.Param(
                self.model.time_steps,
                initialize={
                    t: value for t, value in enumerate(self.steel_demand_per_timestep)
                },
            )

        # Optional: normalized load profile for tracking
        if self.normalized_load_profile is not None:
            # Convert normalized_load_profile to list form for indexing by integer
            if hasattr(self.normalized_load_profile, "data"):
                profile_values = self.normalized_load_profile.data
            elif hasattr(self.normalized_load_profile, "_data"):
                profile_values = self.normalized_load_profile._data
            elif isinstance(self.normalized_load_profile, dict):
                # If it's a dict, try to get values in order
                profile_values = list(self.normalized_load_profile.values())
            else:
                # Try to convert to list
                profile_values = list(self.normalized_load_profile)

            self.model.normalized_load_profile = pyo.Param(
                self.model.time_steps,
                initialize={
                    t: profile_values[t] if t < len(profile_values) else 0
                    for t in self.model.time_steps
                },
            )
            self.model.load_profile_deviation = pyo.Param(
                initialize=self.load_profile_deviation
            )
            logger.debug(
                f"[LOAD-PROFILE] Load profile parameter set with {len(profile_values)} values, deviation: {self.load_profile_deviation:.2%}"
            )

        self.model.steel_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.steel_price)},
            within=pyo.NonNegativeReals,
        )
        self.model.co2_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.co2_price)},
        )
        self.model.lime_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.lime_price)},
            within=pyo.NonNegativeReals,
        )
        self.model.iron_ore_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.iron_ore_price)},
            within=pyo.NonNegativeReals,
        )

    def define_variables(self):
        """
        Defines the variables for the Pyomo model.
        """
        self.model.total_power_input = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.variable_cost = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )

    def initialize_process_sequence(self):
        """
        Establishes the process sequence and constraints for the steel plant, ensuring that
        different components and technologies interact correctly to support steel production.

        This function defines three key constraints:

        1. **Direct Hydrogen Flow Constraint**:
        - Ensures that hydrogen produced by the electrolyzer is either supplied directly
            to the DRI (Direct Reduced Iron) plant or stored in hydrogen buffer storage.
        - If storage is available, the constraint balances hydrogen inflow and outflow
            between the electrolyzer, hydrogen storage, and the DRI plant.
        - If no electrolyzer exists, it ensures that the DRI plant has an alternative hydrogen
            source (i.e., imported hydrogen).

        2. **Direct DRI Flow Constraint**:
        - Regulates the flow of directly reduced iron (DRI) from the DRI plant to the Electric Arc
            Furnace (EAF) or DRI storage.
        - If DRI storage is present, it ensures that part of the produced DRI can be stored
            for later use, maintaining balance between production, storage, and consumption in the EAF.
        - If no storage is available, all produced DRI must go directly to the EAF.

        3. **Material Flow Constraint from DRI Plant to Electric Arc Furnace**:
        - Ensures that all DRI produced by the DRI plant is consumed by the Electric Arc Furnace.
        - This constraint enforces that the total material output from the DRI plant must match
            the required DRI input for the EAF, preventing material imbalances.

        These constraints collectively ensure proper material and energy flow within the steel
        production process, maintaining energy efficiency and operational feasibility.
        """

        # Constraint for direct hydrogen flow from Electrolyser to DRI plant
        @self.model.Constraint(self.model.time_steps)
        def direct_hydrogen_flow_constraint(m, t):
            """
            Ensures the direct hydrogen flow from the electrolyser to the DRI plant or storage.
            """
            if self.has_electrolyser:
                if self.has_h2storage:
                    return (
                        m.dsm_blocks["electrolyser"].hydrogen_out[t]
                        + m.dsm_blocks["hydrogen_buffer_storage"].discharge[t]
                        == m.dsm_blocks["dri_plant"].hydrogen_in[t]
                        + m.dsm_blocks["hydrogen_buffer_storage"].charge[t]
                    )
                else:
                    return (
                        m.dsm_blocks["electrolyser"].hydrogen_out[t]
                        == m.dsm_blocks["dri_plant"].hydrogen_in[t]
                    )
            else:
                # If no electrolyser, ensure DRI plant hydrogen input is as expected
                return m.dsm_blocks["dri_plant"].hydrogen_in[t] >= 0

        # Constraint for direct hydrogen flow from Electrolyser to dri plant
        @self.model.Constraint(self.model.time_steps)
        def direct_dri_flow_constraint(m, t):
            """
            Ensures the direct DRI flow from the DRI plant to the EAF or DRI storage.
            """
            # This constraint allows part of the dri produced by the dri plant to go directly to the dri storage
            # The actual amount should ensure that it does not exceed the capacity or demand of the EAF
            if self.has_dristorage:
                return (
                    m.dsm_blocks["dri_plant"].dri_output[t]
                    + m.dsm_blocks["dri_storage"].discharge[t]
                    == m.dsm_blocks["eaf"].dri_input[t]
                    + m.dsm_blocks["dri_storage"].charge[t]
                )
            else:
                return (
                    m.dsm_blocks["dri_plant"].dri_output[t]
                    == m.dsm_blocks["eaf"].dri_input[t]
                )

        # Constraint for material flow from dri plant to Electric Arc Furnace
        @self.model.Constraint(self.model.time_steps)
        def shaft_to_arc_furnace_material_flow_constraint(m, t):
            """
            Ensures the material flow from the DRI plant to the Electric Arc Furnace.
            """
            return (
                m.dsm_blocks["dri_plant"].dri_output[t]
                == m.dsm_blocks["eaf"].dri_input[t]
            )

    def define_constraints(self):
        """
        Defines key optimization constraints for the steel plant model, ensuring the proper
        operation of the production process and energy consumption.

        This function establishes the following constraints:

        1. **Steel Output Association Constraint**:
        - Ensures that the total steel output from the Electric Arc Furnace (EAF) across all
            time steps meets the required steel demand.
        - This constraint enforces a global balance between steel production and demand over
            the entire time horizon, rather than enforcing it at each individual time step.

        2. **Total Power Input Constraint**:
        - Ensures that the total power input to the steel plant equals the sum of the power
            consumption of all energy-intensive components, including the EAF, DRI plant,
            and electrolyzer (if present).
        - This constraint ensures that energy demand is correctly accounted for and used
            in the optimization process.

        3. **Variable Cost per Time Step Constraint**:
        - Calculates the total variable operating cost per time step based on the power
            consumption and operating costs of the EAF, DRI plant, and electrolyzer (if available).
        - This constraint is useful for cost optimization, as it ensures that the total
            variable cost is accurately computed for each time step.

        These constraints collectively ensure proper energy and material flow within the steel
        production process, enforcing both production targets and cost minimization strategies.
        """

        # For min_demand strategy, the per-hour minimums are the sole demand target.
        # Skip the global equality constraint entirely — it will conflict since the
        # optimizer is free to produce more than the per-hour minimums.
        _min_demand_strategy = (
            hasattr(self, "normalized_load_profile")
            and self.normalized_load_profile is None
            and hasattr(self, "steel_demand_per_timestep")
            and self.steel_demand_per_timestep is not None
        )
        if not _min_demand_strategy:

            @self.model.Constraint()
            def steel_output_association_constraint(m):
                """
                Global constraint: Ensures total steel output meets the global steel demand.

                This enforces that the total steel output equals the demand (or nearly equal
                in rolling horizon windows where it's updated per window).
                """
                total_output = sum(
                    m.dsm_blocks["eaf"].steel_output[t] for t in m.time_steps
                )
                return total_output == m.steel_demand

        # NOTE: Per-timestep minimum steel output constraint is NOT added here.
        # It is added dynamically during rolling-horizon optimization when values are properly
        # initialized for each window. Adding it here during setup could cause infeasibility
        # with the full-horizon initial solve.
        # if self.steel_demand_per_timestep is not None:
        #     @self.model.Constraint(self.model.time_steps)
        #     def steel_demand_per_timestep_constraint(m, t):
        #         """
        #         Per-timestep constraint: Ensures minimum steel output each hour.
        #         If per-timestep demand is provided (from forecasts_df), enforce that the EAF produces
        #         at least the specified amount each hour.
        #         """
        #         return m.dsm_blocks["eaf"].steel_output[t] >= m.steel_demand_per_timestep[t]

        # Constraint for total power input
        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
            """steel_demand given
            Ensures the total power input is the sum of power inputs of all components.
            """
            power_input = (
                m.dsm_blocks["eaf"].power_in[t] + m.dsm_blocks["dri_plant"].power_in[t]
            )
            if self.has_electrolyser:
                power_input += m.dsm_blocks["electrolyser"].power_in[t]

            return m.total_power_input[t] == power_input

        # Constraint for variable cost per time step
        @self.model.Constraint(self.model.time_steps)
        def cost_per_time_step(m, t):
            """
            Calculates the variable cost per time step.
            """

            variable_cost = (
                m.dsm_blocks["eaf"].operating_cost[t]
                + m.dsm_blocks["dri_plant"].operating_cost[t]
            )
            if self.has_electrolyser:
                variable_cost += m.dsm_blocks["electrolyser"].operating_cost[t]

            return m.variable_cost[t] == variable_cost

        # Constraint for normalized load profile tracking (if profile is provided)
        # NOTE: This constraint is added during rolling horizon optimization, not during initial setup
        # to avoid conflicts with the initial cost-minimization objective
        if self.normalized_load_profile is not None:
            logger.info(
                "[LOAD-PROFILE] Normalized load profile is available - will be used in rolling horizon optimization"
            )
