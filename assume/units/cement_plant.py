# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import pyomo.environ as pyo

from assume.common.base import SupportsMinMax
from assume.common.forecaster import CementForecaster
from assume.units.dsm_load_shift import DSMFlex

logger = logging.getLogger(__name__)

# Set the log level to ERROR
logging.getLogger("pyomo").setLevel(logging.WARNING)

# Components whose availability may be shaped by a forecast profile
AVAILABILITY_COMPONENTS = ("preheater", "calciner", "kiln")


class CementPlant(DSMFlex, SupportsMinMax):
    """
    The CementPlant class represents a cement plant unit within an energy system.

    The plant is modelled as the kiln line of a cement works: a **preheater** heats the
    raw meal, a **calciner** drives the calcination reaction that releases the process
    CO2, and a **kiln** finishes the clinker at high temperature. Each stage is
    fuel-switchable between electricity, a natural gas / coal mix and hydrogen, so the
    same plant description covers a conventional and an electrified configuration.

    Optional components extend the line:

    - an **electrolyser** producing hydrogen on site for the calciner and kiln burners,
    - a **thermal storage** which, in its electric-heater mode, buys power when it is
      cheap and later displaces calciner burner heat (an E-TES),
    - a **raw material mill** and a **cement mill** - electric grinding steps that
      convert raw materials to raw meal and clinker to cement respectively.

    Waste heat from the kiln is recovered into the preheater, which reduces the fuel the
    preheater needs for a given raw meal throughput.

    The two mills are not connected into the kiln line automatically. A cement mill
    alongside a kiln line still grinds that line's clinker output (as it always has),
    but a plant made of only a raw material mill and/or only a cement mill - with no
    preheater/calciner/kiln at all - runs as a **standalone grinding operation**: the
    declared demand then targets that mill's own ground tonnage directly (cement for a
    lone cement mill, raw meal for a lone raw material mill) instead of clinker.

    Both full-horizon and rolling-horizon optimisation are supported. In rolling-horizon
    mode the plant re-optimises a look-ahead window at each market round, carrying
    thermal storage fill levels and on/off states from the previously committed period
    and tracking the clinker (or, for a standalone mill, the ground tonnage) still to be
    produced.

    Args:
        id (str): A unique identifier for the cement plant unit.
        unit_operator (str): The operator responsible for the cement plant.
        bidding_strategies (dict): A dictionary of bidding strategies, which define how the unit participates in energy markets.
        forecaster (CementForecaster): A forecaster used to get key variables such as fuel or electricity prices.
        components (dict, optional): A dictionary describing the components of the cement plant. Default is an empty dictionary.
        technology (str, optional): The technology of the cement plant. Default is "cement_plant".
        objective (str, optional): The objective function of the plant. Default is "min_variable_cost".
        flexibility_measure (str, optional): The flexibility measure for the cement plant, such as "cost_based_load_shift". Default is "cost_based_load_shift".
        demand (float, optional): The production demand in tonnes over the simulation horizon - clinker when a preheater/calciner/kiln is configured, otherwise the ground tonnage of a standalone mill. Default is 0.
        cost_tolerance (float, optional): The maximum allowable cost increase (in %) when shifting the load. Default is 10.
        congestion_threshold (float, optional): The threshold above which the congestion signal counts as congestion. Default is 0.
        peak_load_cap (float, optional): The peak load cap (in % of the peak load) used by the peak load shifting measure. Default is 0.
        load_profile_deviation (float, optional): Allowed deviation (±) from the normalized load profile, as a fraction (e.g. 0.05 for ±5%). Default is 1.0 (no constraint).
        raw_meal_to_clinker_ratio (float, optional): Tonnes of raw meal per tonne of clinker. Default is 1.55.
        waste_heat_per_t_clinker (float, optional): Kiln waste heat available per tonne of clinker (MWh_th/t). Default is 0.22.
        waste_heat_utilisation (float, optional): Share of the available waste heat the preheater can use. Default is 0.90.
        node (str, optional): The network node where the cement plant is located. Default is "node0".
        location (tuple[float, float], optional): The geographical coordinates (latitude, longitude) of the cement plant. Default is (0.0, 0.0).
        **kwargs: Additional keyword arguments that may be passed to support more specific configurations.

    Attributes:
        required_technologies (list): Technologies the plant cannot operate without (none: any subset of the kiln line may be modelled).
        optional_technologies (list): Technologies the plant may consist of.
    """

    # Required and optional technologies for the cement plant
    required_technologies = []
    optional_technologies = [
        "preheater",
        "calciner",
        "kiln",
        "raw_material_mill",
        "cement_mill",
        "electrolyser",
        "thermal_storage",
    ]

    # Stages that produce clinker, in the order in which they may terminate the line
    clinker_stages = ("kiln", "calciner")

    # Rolling-horizon extensibility hooks (DSMFlex)
    _demand_attr_suffix = "clinker_demand"
    _extra_price_attrs = [
        "natural_gas_price",
        "coal_price",
        "hydrogen_price",
        "co2_price",
    ]
    _component_schema = {
        "preheater": (
            "power_in",
            "raw_meal_out",
            "preheater_power_input",
            "preheater_raw_meal_output",
        ),
        "calciner": (
            "power_in",
            "clinker_out",
            "calciner_power_input",
            "calciner_clinker_output",
        ),
        "kiln": ("power_in", "clinker_out", "kiln_power_input", "kiln_clinker_output"),
        "raw_material_mill": (
            "power_in",
            "material_output",
            "raw_material_mill_power_input",
            "raw_material_mill_output",
        ),
        "cement_mill": (
            "power_in",
            "material_output",
            "cement_mill_power_input",
            "cement_output",
        ),
        "electrolyser": (
            "power_in",
            "hydrogen_out",
            "electrolyser_power",
            "hydrogen_prod",
        ),
        "thermal_storage": (
            "power_in",
            "discharge",
            "thermal_storage_power_input",
            "thermal_storage_discharge",
        ),
    }

    def _primary_output_expr(self, m, t):
        """The plant's tracked output (tonnes): clinker, or a standalone mill's own
        ground tonnage when there is no kiln line at all."""
        return m.clinker_rate[t]

    def _component_power_expr(self, m, t):
        """Plant load: electric heating, auxiliaries and the electric components."""
        total_power = pyo.quicksum(
            m.dsm_blocks[block].power_in[t] for block in m.dsm_blocks
        )
        for stage in ("preheater", "calciner", "kiln"):
            if stage in m.dsm_blocks:
                total_power += m.dsm_blocks[stage].aux_power[t]
        return total_power

    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        forecaster: CementForecaster,
        components: dict[str, dict] | None = None,
        technology: str = "cement_plant",
        objective: str = "min_variable_cost",
        flexibility_measure: str = "cost_based_load_shift",
        demand: float = 0,
        cost_tolerance: float = 10,
        congestion_threshold: float = 0,
        peak_load_cap: float = 0,
        load_profile_deviation: float = 1.0,
        raw_meal_to_clinker_ratio: float = 1.55,
        waste_heat_per_t_clinker: float = 0.22,
        waste_heat_utilisation: float = 0.90,
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
        if not isinstance(forecaster, CementForecaster):
            raise TypeError(f"forecaster must be of type {CementForecaster.__name__}")

        if self.components is None:
            self.components = {}

        # check if the required components are present in the components dictionary
        for component in self.required_technologies:
            if component not in self.components.keys():
                raise ValueError(
                    f"Component {component} is required for the cement plant unit."
                )

        # check if the provided components are valid and do not contain any unknown components
        for component in self.components.keys():
            if (
                component not in self.required_technologies
                and component not in self.optional_technologies
            ):
                raise ValueError(
                    f"Component '{component}' is not a valid component for the cement plant unit."
                )

        # FIXME assuming only one market
        self.market_id = list(bidding_strategies.keys())[0]

        # Global demand (total clinker production by the end of the horizon)
        self.clinker_demand = demand
        # Per-timestep minimum clinker production (optional, from the forecaster)
        self.clinker_demand_per_timestep = getattr(forecaster, "clinker_demand", None)

        self.natural_gas_price = self.forecaster.get_price("natural_gas")
        self.coal_price = self.forecaster.get_price("coal")
        self.hydrogen_price = self.forecaster.get_price("hydrogen")
        self.co2_price = self.forecaster.get_price("co2")

        # DSM signals used by the flexibility measures
        self.congestion_signal = forecaster.congestion_signal
        self.renewable_utilisation_signal = forecaster.renewable_utilisation_signal

        self.objective = objective
        self.flexibility_measure = flexibility_measure
        self.cost_tolerance = cost_tolerance
        self.congestion_threshold = congestion_threshold
        self.peak_load_cap = peak_load_cap
        self.load_profile_deviation = load_profile_deviation

        self.raw_meal_to_clinker_ratio = raw_meal_to_clinker_ratio
        self.waste_heat_per_t_clinker = waste_heat_per_t_clinker
        self.waste_heat_utilisation = waste_heat_utilisation

        # Optional profile guiding production in rolling-horizon mode
        self.normalized_load_profile = getattr(
            forecaster, "normalized_load_profile", None
        )

        # Component flags
        self.has_preheater = "preheater" in self.components.keys()
        self.has_calciner = "calciner" in self.components.keys()
        self.has_kiln = "kiln" in self.components.keys()
        self.has_raw_material_mill = "raw_material_mill" in self.components.keys()
        self.has_cement_mill = "cement_mill" in self.components.keys()
        self.has_electrolyser = "electrolyser" in self.components.keys()
        self.has_thermal_storage = "thermal_storage" in self.components.keys()

        # The stage whose clinker output is the plant output
        self.final_clinker_stage = next(
            (stage for stage in self.clinker_stages if stage in self.components), None
        )
        # The stage that receives the preheated raw meal: the calciner if present,
        # otherwise the kiln directly (a single-stage line without a separate
        # precalciner, where the kiln itself performs the calcination reaction).
        self.raw_meal_receiving_stage = (
            "calciner" if self.has_calciner else ("kiln" if self.has_kiln else None)
        )
        # When there is no kiln line at all, a lone mill becomes the plant's own
        # standalone output (cement mill takes priority over the raw material mill,
        # mirroring how final_clinker_stage prefers the kiln over the calciner).
        self._standalone_output_stage = None
        if self.final_clinker_stage is None:
            if self.has_cement_mill:
                self._standalone_output_stage = "cement_mill"
            elif self.has_raw_material_mill:
                self._standalone_output_stage = "raw_material_mill"

        if self.has_electrolyser and not (self.has_calciner or self.has_kiln):
            raise ValueError(
                "An electrolyser needs a calciner or a kiln to supply with hydrogen."
            )
        if self.has_thermal_storage and not self.has_calciner:
            raise ValueError(
                "The thermal storage of a cement plant buffers the calciner, so a "
                "calciner is required."
            )
        if (
            self.final_clinker_stage is None
            and self._standalone_output_stage is None
            and (self.clinker_demand or self.clinker_demand_per_timestep is not None)
        ):
            raise ValueError(
                "A production demand requires a calciner, a kiln, or a standalone "
                "raw material mill / cement mill in the cement plant."
            )

        # Inject the schedule of a long-term thermal storage, and hand per-component
        # availability profiles down to the component configs, so both are sliced along
        # with the components in rolling-horizon windows.
        storage_cfg = self.components.get("thermal_storage")
        if (
            storage_cfg is not None
            and storage_cfg.get("storage_type", "short-term") == "long-term"
        ):
            storage_cfg["storage_schedule_profile"] = (
                forecaster.thermal_storage_schedule
            )

        availability_profiles = getattr(forecaster, "availability_profiles", None) or {}
        for tech in AVAILABILITY_COMPONENTS:
            profile = availability_profiles.get(tech)
            if tech in self.components and profile is not None:
                self.components[tech]["availability_profile"] = profile

        # Initialize the model
        # self.setup_model()  # NOTE: called in forecaster initialization again!!!

    def define_parameters(self):
        """
        Defines the parameters for the Pyomo model.

        All price and profile series are aligned to the current model index, so the same
        code path serves the full-horizon model and the shorter rolling-horizon windows.
        """
        self.model.electricity_price = pyo.Param(
            self.model.time_steps,
            initialize={
                t: value
                for t, value in enumerate(
                    self._values_for_model(self.forecaster.electricity_price)
                )
            },
        )
        self.model.natural_gas_price = pyo.Param(
            self.model.time_steps,
            initialize={
                t: value
                for t, value in enumerate(
                    self._values_for_model(self.natural_gas_price)
                )
            },
        )
        self.model.coal_price = pyo.Param(
            self.model.time_steps,
            initialize={
                t: value
                for t, value in enumerate(self._values_for_model(self.coal_price))
            },
        )
        # Hydrogen produced on site is paid for through the electrolyser's electricity
        # bill, so charging the hydrogen price on top would double count it.
        if self.has_electrolyser:
            self.model.hydrogen_price = pyo.Param(
                self.model.time_steps,
                initialize={t: 0 for t in self.model.time_steps},
            )
        else:
            self.model.hydrogen_price = pyo.Param(
                self.model.time_steps,
                initialize={
                    t: value
                    for t, value in enumerate(
                        self._values_for_model(self.hydrogen_price)
                    )
                },
            )
        self.model.co2_price = pyo.Param(
            self.model.time_steps,
            initialize={
                t: value
                for t, value in enumerate(self._values_for_model(self.co2_price))
            },
            within=pyo.NonNegativeReals,
        )

        self.model.raw_meal_to_clinker_ratio = pyo.Param(
            initialize=self.raw_meal_to_clinker_ratio
        )
        self.model.waste_heat_per_t_clinker = pyo.Param(
            initialize=self.waste_heat_per_t_clinker
        )
        self.model.waste_heat_utilisation = pyo.Param(
            initialize=self.waste_heat_utilisation
        )

        # Use the remaining demand in rolling-horizon mode, otherwise the global demand
        demand_value = self.clinker_demand
        if self._rh_window_remaining_demand is not None:
            demand_value = self._rh_window_remaining_demand
            logger.info(f"Using rolling-horizon remaining_demand: {demand_value}")

        self.model.clinker_demand = pyo.Param(initialize=demand_value)

        # Optional: per-timestep minimum clinker production
        if self.clinker_demand_per_timestep is not None:
            self.model.clinker_demand_per_timestep = pyo.Param(
                self.model.time_steps,
                initialize={
                    t: value
                    for t, value in enumerate(
                        self._values_for_model(self.clinker_demand_per_timestep)
                    )
                },
            )

        # Optional: normalized load profile used to shape production
        if self.normalized_load_profile is not None:
            self.model.normalized_load_profile = pyo.Param(
                self.model.time_steps,
                initialize={
                    t: value
                    for t, value in enumerate(
                        self._values_for_model(self.normalized_load_profile)
                    )
                },
            )
            self.model.load_profile_deviation = pyo.Param(
                initialize=self.load_profile_deviation
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
        self.model.clinker_rate = pyo.Var(
            self.model.time_steps,
            within=pyo.NonNegativeReals,
            doc="The plant's tracked production at each time step (tonnes): clinker "
            "leaving the kiln line, or a standalone mill's own ground tonnage",
        )
        self.model.cumulative_thermal_output = pyo.Var(
            self.model.time_steps,
            within=pyo.NonNegativeReals,
            doc="Process heat generated by the last thermal stage (MWh_th)",
        )

        if self.has_electrolyser:
            self.model.h2_to_calciner = pyo.Var(
                self.model.time_steps, within=pyo.NonNegativeReals
            )
            self.model.h2_to_kiln = pyo.Var(
                self.model.time_steps, within=pyo.NonNegativeReals
            )
            self.model.h2_unutilised = pyo.Var(
                self.model.time_steps, within=pyo.NonNegativeReals
            )

    def initialize_process_sequence(self):
        """
        Establishes the material and energy flows between the components of the cement plant.

        The following links are added, each only if the components involved are configured:

        1. **Raw meal flow**: the preheated raw meal covers what the first clinker-forming
           stage needs - the calciner if present, otherwise the kiln directly - scaled by
           ``raw_meal_to_clinker_ratio``.
        2. **Clinker flow**: the kiln finishes the calcined material of the calciner, and
           the clinker of the last stage of the line becomes the plant's tracked output.
           If there is no kiln line at all, a lone raw material mill or cement mill
           becomes the plant's tracked output instead (see 6.).
        3. **Waste heat recovery**: heat recovered from the kiln reduces the fuel the
           preheater needs. Without a kiln the preheater's waste heat port is closed.
        4. **Calciner heat**: the heat driving calcination is the burner output plus the
           thermal storage discharge, minus what a heat-charged storage takes away.
        5. **Hydrogen flow**: hydrogen from the electrolyser is routed to the calciner and
           kiln burners.
        6. **Cement grinding**: when a kiln line is configured, the cement mill grinds
           the clinker it produces. A cement mill with no kiln line is not connected to
           anything and instead becomes the plant's own standalone output (see 2.).
        """

        if self.has_preheater and self.raw_meal_receiving_stage is not None:

            @self.model.Constraint(self.model.time_steps)
            def raw_meal_flow_constraint(m, t):
                """
                Links the preheated raw meal to the material calcined downstream.
                """
                return (
                    m.dsm_blocks["preheater"].raw_meal_out[t]
                    == m.dsm_blocks[self.raw_meal_receiving_stage].clinker_out[t]
                    * m.raw_meal_to_clinker_ratio
                )

        if self.has_calciner and self.has_kiln:

            @self.model.Constraint(self.model.time_steps)
            def calciner_to_kiln_flow_constraint(m, t):
                """
                Links the calcined material of the calciner to the kiln throughput.
                """
                return (
                    m.dsm_blocks["kiln"].clinker_out[t]
                    == m.dsm_blocks["calciner"].clinker_out[t]
                )

        @self.model.Constraint(self.model.time_steps)
        def clinker_rate_constraint(m, t):
            """
            Sets the plant's tracked output to the last stage of the kiln line, or, for
            a standalone mill with no kiln line at all, to that mill's own ground
            tonnage.
            """
            if self.final_clinker_stage is not None:
                return (
                    m.clinker_rate[t]
                    == m.dsm_blocks[self.final_clinker_stage].clinker_out[t]
                )
            if self._standalone_output_stage is not None:
                return (
                    m.clinker_rate[t]
                    == m.dsm_blocks[self._standalone_output_stage].material_output[t]
                )
            return m.clinker_rate[t] == 0

        if self.has_preheater:

            @self.model.Constraint(self.model.time_steps)
            def waste_heat_recovery_constraint(m, t):
                """
                Recovers kiln waste heat into the preheater.
                """
                if not self.has_kiln:
                    return m.dsm_blocks["preheater"].external_heat_in[t] == 0
                return (
                    m.dsm_blocks["preheater"].external_heat_in[t]
                    == m.dsm_blocks["kiln"].clinker_out[t]
                    * m.waste_heat_per_t_clinker
                    * m.waste_heat_utilisation
                )

        if self.has_calciner:

            @self.model.Constraint(self.model.time_steps)
            def calciner_heat_constraint(m, t):
                """
                Determines the heat available for calcination, buffered by the thermal
                storage where present.
                """
                effective_heat = m.dsm_blocks["calciner"].heat_out[t]
                if self.has_thermal_storage:
                    storage = m.dsm_blocks["thermal_storage"]
                    effective_heat += storage.discharge[t]
                    # A storage without an electric heater is charged with burner heat,
                    # which is therefore not available to the calciner.
                    if self.components["thermal_storage"].storage_type != (
                        "short-term_with_generator"
                    ):
                        effective_heat -= storage.charge[t]
                return m.dsm_blocks["calciner"].effective_heat_in[t] == effective_heat

        if self.has_electrolyser:

            @self.model.Constraint(self.model.time_steps)
            def electrolyser_supply_constraint(m, t):
                """
                Splits the hydrogen production between the burners it can supply.
                """
                return m.dsm_blocks["electrolyser"].hydrogen_out[t] == (
                    m.h2_to_calciner[t] + m.h2_to_kiln[t] + m.h2_unutilised[t]
                )

            @self.model.Constraint(self.model.time_steps)
            def calciner_hydrogen_constraint(m, t):
                """
                Routes hydrogen to the calciner burner.
                """
                if not self.has_calciner:
                    return m.h2_to_calciner[t] == 0
                return m.dsm_blocks["calciner"].hydrogen_in[t] == m.h2_to_calciner[t]

            @self.model.Constraint(self.model.time_steps)
            def kiln_hydrogen_constraint(m, t):
                """
                Routes hydrogen to the kiln burner.
                """
                if not self.has_kiln:
                    return m.h2_to_kiln[t] == 0
                return m.dsm_blocks["kiln"].hydrogen_in[t] == m.h2_to_kiln[t]

        if self.has_cement_mill and self.final_clinker_stage is not None:

            @self.model.Constraint(self.model.time_steps)
            def cement_grinding_constraint(m, t):
                """
                Grinds the clinker the kiln line produces into cement.
                """
                return (
                    m.dsm_blocks["cement_mill"].material_input[t] == m.clinker_rate[t]
                )

        @self.model.Constraint(self.model.time_steps)
        def cumulative_thermal_output_constraint(m, t):
            """
            Reports the process heat of the last thermal stage of the line.
            """
            for stage in ("kiln", "calciner", "preheater"):
                if stage in self.components:
                    return (
                        m.cumulative_thermal_output[t]
                        == m.dsm_blocks[stage].heat_out[t]
                    )
            return m.cumulative_thermal_output[t] == 0

    def define_constraints(self):
        """
        Defines the plant-level optimization constraints of the cement plant.

        1. **Demand constraint**: the plant's tracked production (clinker, or a standalone
           mill's own ground tonnage) over the horizon meets the declared demand. It is
           skipped when per-timestep minimums are supplied instead, since the optimizer
           is then free to produce more than the sum of the minimums.
        2. **Per-timestep demand constraint**: production per time step covers the
           declared hourly minimum (full-horizon mode only; in rolling-horizon mode the
           per-window equivalent is added by ``_add_window_demand_constraints``).
        3. **Total power input constraint**: the plant load is the sum of the electric
           heating, auxiliary and electric-component loads.
        4. **Variable cost constraint**: the cost per time step is the sum of the component
           operating costs, which already include fuel, auxiliary electricity and CO2.
        """

        # With per-hour minimums the global equality would conflict with them, since
        # producing more than the sum of the minimums must stay feasible.
        _min_demand_strategy = (
            self.normalized_load_profile is None
            and self.clinker_demand_per_timestep is not None
        )
        if not _min_demand_strategy:

            @self.model.Constraint()
            def clinker_output_association_constraint(m):
                """
                Global constraint: total tracked production meets the declared demand.
                """
                return (
                    pyo.quicksum(m.clinker_rate[t] for t in m.time_steps)
                    == m.clinker_demand
                )

        if (
            self.clinker_demand_per_timestep is not None
            and self.horizon_mode != "rolling_horizon"
        ):

            @self.model.Constraint(self.model.time_steps)
            def clinker_demand_per_timestep_constraint(m, t):
                return m.clinker_rate[t] >= m.clinker_demand_per_timestep[t]

        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
            """
            Ensures the total power input is the sum of power inputs of all components.
            """
            return m.total_power_input[t] == self._component_power_expr(m, t)

        @self.model.Constraint(self.model.time_steps)
        def cost_per_time_step(m, t):
            """
            Calculates the variable cost per time step.
            """
            return m.variable_cost[t] == pyo.quicksum(
                m.dsm_blocks[block].operating_cost[t] for block in m.dsm_blocks
            )
