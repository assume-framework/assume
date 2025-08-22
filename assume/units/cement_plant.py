# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime

import pyomo.environ as pyo

from assume.common.base import SupportsMinMax
from assume.common.forecasts import Forecaster
from assume.units.dsm_load_shift import DSMFlex

logger = logging.getLogger(__name__)


class CementPlant(DSMFlex, SupportsMinMax):
    """
    The CementPlant class represents a cement plant unit in the energy system.

    Args:
        id (str): The unique identifier of the unit.
        unit_operator (str): The operator of the unit.
        bidding_strategies (dict): The bidding strategies of the unit.
        technology (str): The technology of the unit.
        node (str): The node of the unit.
        location (tuple[float, float]): The location of the unit.
        components (dict[str, dict]): The components of the unit such as raw_material_mill, clinker_system, cement_mill, and electrolyser.
        objective (str): The objective of the unit, e.g. minimize variable cost ("min_variable_cost").
        flexibility_measure (str): The flexibility measure of the unit, e.g. maximum load shift ("max_load_shift").
        demand (float): The demand of the unit - the amount of cement to be produced.
        cost_tolerance (float): The cost tolerance of the unit - the maximum cost that can be tolerated when shifting the load.
    """

    required_technologies = []
    optional_technologies = [
        "preheater",
        "calciner",
        "kiln",
        "cement_mill",
        "electrolyser",
        "thermal_storage",
    ]

    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        forecaster: Forecaster,
        technology: str = "cement_plant",
        node: str = "node0",
        location: tuple[float, float] = (0.0, 0.0),
        components: dict[str, dict] = None,
        objective: str = None,
        flexibility_measure: str = "",
        peak_load_cap: float = 0,
        demand: float = 0,
        cost_tolerance: float = 10,
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
                    f"Component {component} is required for the cement plant unit."
                )

        # check if the provided components are valid and do not contain any unknown components
        for component in components.keys():
            if (
                component not in self.required_technologies
                and component not in self.optional_technologies
            ):
                raise ValueError(
                    f"Components {component} is not a valid component for the cement plant unit."
                )
        # Check the presence of components first
        self.has_preheater = "preheater" in self.components.keys()
        self.has_calciner = "calciner" in self.components.keys()
        self.has_kiln = "kiln" in self.components.keys()
        self.has_cement_mill = "cement_mill" in self.components.keys()
        self.has_electrolyser = "electrolyser" in self.components.keys()
        self.has_thermal_storage = "thermal_storage" in self.components.keys()

        # Inject schedule into long-term thermal storage if applicable
        if "thermal_storage" in self.components:
            storage_cfg = self.components["thermal_storage"]
            storage_type = storage_cfg.get("storage_type", "short-term")
            if storage_type == "long-term":
                schedule_key = f"{self.id}_thermal_storage_schedule"
                schedule_series = self.forecaster[schedule_key]
                storage_cfg["storage_schedule_profile"] = schedule_series

        # Add forecasts
        self.natural_gas_price = self.forecaster["natural_gas_price"]
        self.clinker_demand_per_time_step = self.forecaster[f"{self.id}_clinker_demand"]
        self.hydrogen_price = self.forecaster["hydrogen_gas_price"]
        self.coal_price = self.forecaster["coal_price"]
        self.electricity_price = self.forecaster["electricity_price"]
        self.electricity_price_flex = self.forecaster["electricity_price_flex"]
        self.co2_price = self.forecaster.get_price("co2")
        self.demand = demand
        self.objective = objective
        self.flexibility_measure = flexibility_measure
        self.cost_tolerance = cost_tolerance
        self.peak_load_cap = peak_load_cap

        self.optimisation_counter = 0

        # Initialize the model
        self.setup_model()

    def define_parameters(self):
        self.model.electricity_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.electricity_price)},
        )
        self.model.natural_gas_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.natural_gas_price)},
        )
        self.model.coal_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.coal_price)},
        )
        if self.has_electrolyser:
            self.model.hydrogen_price = pyo.Param(
                self.model.time_steps,
                initialize={t: 0 for t in self.model.time_steps},
            )
        else:
            self.model.hydrogen_price = pyo.Param(
                self.model.time_steps,
                initialize={t: value for t, value in enumerate(self.hydrogen_price)},
            )
        self.model.raw_meal_to_clinker_ratio = pyo.Param(initialize=1.55)
        self.model.co2_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.co2_price)},
        )
        self.model.absolute_cement_demand = pyo.Param(initialize=self.demand)
        self.model.clinker_demand_per_time_step = pyo.Param(
            self.model.time_steps,
            initialize={
                t: value for t, value in enumerate(self.clinker_demand_per_time_step)
            },
        )

    def define_variables(self):
        self.model.total_power_input = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.cumulative_thermal_output = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.variable_cost = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.preheated_raw_meal = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.clinker_rate = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )

        # ---- Hydrogen routing vars (active only if electrolyser exists) ----
        self.model.h2_to_calciner = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.h2_to_kiln = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.h2_unutilised = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )  # optional slack

    def initialize_process_sequence(self):
        """
        Initializes the process sequence for the cement plant based on available components.
        """
        if not self.demand or self.demand == 0:
            # m.preheated_raw_meal[t]  (t/h equivalent proxy)
            # m.clinker_rate[t]        (t/h final clinker)

            # ----- Mass links -----
            if self.has_preheater and self.has_calciner:

                @self.model.Constraint(self.model.time_steps)
                def ph_to_cc_mass_link(m, t):
                    # clinker rate leaving calciner equals preheater raw meal divided by ratio
                    return (
                        m.dsm_blocks["calciner"].clinker_out[t]
                        == m.dsm_blocks["preheater"].raw_meal_out[t]
                        / m.raw_meal_to_clinker_ratio
                    )

                @self.model.Constraint(self.model.time_steps)
                def raw_meal_alias(m, t):
                    return (
                        m.preheated_raw_meal[t]
                        == m.dsm_blocks["preheater"].raw_meal_out[t]
                    )

            if self.has_calciner and self.has_kiln:

                @self.model.Constraint(self.model.time_steps)
                def cc_to_rk_mass_link(m, t):
                    # kiln inlet equals calciner outlet (simple 1:1; add losses later if needed)
                    return (
                        m.dsm_blocks["kiln"].clinker_out[t]
                        == m.dsm_blocks["calciner"].clinker_out[t]
                    )

                @self.model.Constraint(self.model.time_steps)
                def clinker_alias_from_kiln(m, t):
                    return m.clinker_rate[t] == m.dsm_blocks["kiln"].clinker_out[t]

            if self.has_calciner and self.has_kiln:

                @self.model.Constraint(self.model.time_steps)
                def clinker_alias_from_calciner(m, t):
                    return m.clinker_rate[t] == m.dsm_blocks["calciner"].clinker_out[t]

            elif self.has_preheater and not (self.has_calciner or self.has_kiln):

                @self.model.Constraint(self.model.time_steps)
                def raw_meal_only_alias(m, t):
                    return (
                        m.preheated_raw_meal[t]
                        == m.dsm_blocks["preheater"].raw_meal_out[t]
                    )

            # ----- Waste Heat Recovery (kiln → preheater) -----
            if self.has_kiln and self.has_preheater:
                # default WH params if not set externally
                if not hasattr(self.model, "wh_per_t_clinker"):
                    self.model.wh_per_t_clinker = pyo.Param(
                        initialize=0.22
                    )  # [MWh_th/t]
                if not hasattr(self.model, "wh_util_eff"):
                    self.model.wh_util_eff = pyo.Param(initialize=0.90)

                @self.model.Constraint(self.model.time_steps)
                def whr_kiln_to_preheater(m, t):
                    # recovered waste heat = clinker_out * yield * efficiency
                    return (
                        m.dsm_blocks["preheater"].external_heat_in[t]
                        == m.dsm_blocks["kiln"].clinker_out[t]
                        * m.wh_per_t_clinker
                        * m.wh_util_eff
                    )

            # ----- Bind plant thermal output (for legacy thermal-demand & plotting) -----
            if self.has_kiln:

                @self.model.Constraint(self.model.time_steps)
                def plant_heat_from_kiln(m, t):
                    # choose kiln heat as plant "cumulative thermal output"
                    return (
                        m.cumulative_thermal_output[t]
                        == m.dsm_blocks["kiln"].heat_out[t]
                    )

            elif self.has_calciner:

                @self.model.Constraint(self.model.time_steps)
                def plant_heat_from_calciner(m, t):
                    return (
                        m.cumulative_thermal_output[t]
                        == m.dsm_blocks["calciner"].heat_out[t]
                    )

            elif self.has_preheater:

                @self.model.Constraint(self.model.time_steps)
                def plant_heat_from_preheater(m, t):
                    return (
                        m.cumulative_thermal_output[t]
                        == m.dsm_blocks["preheater"].heat_out[t]
                    )

                # ---------------- Electrolyser → H2 routing ----------------
            if self.has_electrolyser:
                el = self.model.dsm_blocks["electrolyser"]

                # Variables for routing (already declared in define_variables)
                has_cc = self.has_calciner
                has_rk = self.has_kiln

                if has_cc:
                    cc = self.model.dsm_blocks["calciner"]

                    @self.model.Constraint(self.model.time_steps)
                    def bind_h2_to_calciner(m, t):
                        return cc.hydrogen_in[t] == m.h2_to_calciner[t]

                if has_rk:
                    rk = self.model.dsm_blocks["kiln"]

                    @self.model.Constraint(self.model.time_steps)
                    def bind_h2_to_kiln(m, t):
                        return rk.hydrogen_in[t] == m.h2_to_kiln[t]

                @self.model.Constraint(self.model.time_steps)
                def electrolyser_supply_balance(m, t):
                    routed = 0
                    if has_cc:
                        routed += m.h2_to_calciner[t]
                    if has_rk:
                        routed += m.h2_to_kiln[t]
                    return el.hydrogen_out[t] >= routed + m.h2_unutilised[t]

            # ---------------- Dynamic E‑TES coupling to Calciner ----------------
            if self.has_calciner:
                cc = self.model.dsm_blocks["calciner"]

                if self.has_thermal_storage:
                    ts = self.model.dsm_blocks["thermal_storage"]

                    @self.model.Constraint(self.model.time_steps)
                    def tes_to_calciner_effective_heat(m, t):
                        # TES buffers calciner: usable calciner heat = burner heat + TES discharge - TES charge
                        return (
                            cc.effective_heat_in[t] == cc.heat_out[t] + ts.discharge[t]
                        )
                else:
                    # No TES present → bind effective heat directly to burner heat (zero-config behavior)
                    @self.model.Constraint(self.model.time_steps)
                    def bind_eff_no_tes(m, t):
                        return cc.effective_heat_in[t] == cc.heat_out[t]

    def define_constraints(self):
        """
        Defines the constraints for the paper and pulp plant model.
        """

        @self.model.Constraint(self.model.time_steps)
        def absolute_demand_association_constraint(m, t):
            """
            Ensures the thermal output meets the absolute demand.
            """
            if not self.demand or self.demand == 0:
                if self.has_kiln:
                    return (
                        m.dsm_blocks["kiln"].clinker_out[t]
                        >= m.clinker_demand_per_time_step[t]
                    )

            else:
                # Absolute totals: prefer clinker if kiln & target provided; else thermal
                if self.has_kiln:
                    return (
                        sum(m.dsm_blocks["kiln"].clinker_out[t] for t in m.time_steps)
                        >= m.absolute_clinker_demand
                    )

        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
            total_power = 0.0

            if self.has_preheater:
                total_power += self.model.dsm_blocks["preheater"].power_in[t]
                if hasattr(self.model.dsm_blocks["preheater"], "aux_power"):
                    total_power += self.model.dsm_blocks["preheater"].aux_power[t]

            if self.has_calciner:
                total_power += self.model.dsm_blocks["calciner"].power_in[t]
                if hasattr(self.model.dsm_blocks["calciner"], "aux_power"):
                    total_power += self.model.dsm_blocks["calciner"].aux_power[t]

            if self.has_kiln:
                total_power += self.model.dsm_blocks["kiln"].power_in[t]
                if hasattr(self.model.dsm_blocks["kiln"], "aux_power"):
                    total_power += self.model.dsm_blocks["kiln"].aux_power[t]

            if self.has_electrolyser:
                total_power += self.model.dsm_blocks["electrolyser"].power_in[t]
            if self.has_thermal_storage:
                total_power += self.model.dsm_blocks["thermal_storage"].power_in[t]

            if self.has_cement_mill:
                total_power += self.model.dsm_blocks["cement_mill"].power_in[t]

            return m.total_power_input[t] == total_power

        # Operating cost constraint
        @self.model.Constraint(self.model.time_steps)
        def cost_per_time_step(m, t):
            """
            Calculates the variable cost per time step.
            """
            variable_cost = 0
            if self.has_cement_mill:
                variable_cost += self.model.dsm_blocks["cement_mill"].operating_cost[t]
            if self.has_preheater:
                variable_cost += self.model.dsm_blocks["preheater"].operating_cost[t]
            if self.has_calciner:
                variable_cost += self.model.dsm_blocks["calciner"].operating_cost[t]
            if self.has_kiln:
                variable_cost += m.dsm_blocks["kiln"].operating_cost[t]
            if self.has_electrolyser:
                variable_cost += self.model.dsm_blocks["electrolyser"].operating_cost[t]
            if self.has_thermal_storage:
                variable_cost += self.model.dsm_blocks[
                    "thermal_storage"
                ].operating_cost[t]
            return self.model.variable_cost[t] == variable_cost

    def calculate_marginal_cost(self, start: datetime, power: float) -> float:
        """
        Calculate the marginal cost of the unit based on the provided time and power.

        Args:
            start (datetime): The start time of the dispatch
            power (float): The power output of the unit

        Returns:
            float: The marginal cost of the unit
        """
        marginal_cost = 0

        if self.opt_power_requirement.at[start] > 0:
            marginal_cost = (
                self.variable_cost_series.at[start]
                / self.opt_power_requirement.at[start]
            )

        return marginal_cost
