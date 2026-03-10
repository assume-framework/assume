# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as ps
from pathlib import Path
from assume.common.utils import str_to_bool

import matplotlib as mpl
import matplotlib.pyplot as plt
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
        is_prosumer: str = "No",
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
        self.co2_price = self.forecaster["co2_price"]
        self.demand = demand
        self.objective = objective
        self.flexibility_measure = flexibility_measure
        self.cost_tolerance = cost_tolerance
        self.peak_load_cap = peak_load_cap
        # Example: inject plant-scoped series (pseudo-code)
        plant_id = str(id)
        f = forecaster
        # f.load_series is whatever your project’s method is to register Series
        self.baseline_power = self.forecaster[f"{self.id}_baseline_power"]
        self.max_up = self.forecaster[f"{self.id}_max_up"]
        self.max_down = self.forecaster[f"{self.id}_max_down"]
        self.price_up = self.forecaster[f"{self.id}_price_up"]
        self.price_down = self.forecaster[f"{self.id}_price_down"]


        self.optimisation_counter = 0
        self.is_prosumer = str_to_bool(is_prosumer)
        if self.is_prosumer:
            self.fcr_price = self.forecaster["DE_fcr_price"]  # symmetric FCR
            self.afrr_price_pos = self.forecaster["DE_pos_price"]  # aFRR up
            self.afrr_price_neg = self.forecaster["DE_neg_price"]  # aFRR down)

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
        self.model.raw_meal_to_clinker_ratio = pyo.Param(initialize=1.73)
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
        self.model.total_co2_emission = pyo.Var(self.model.time_steps, within=pyo.NonNegativeReals)


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
                        initialize= 0.802    # 0.061

                    )  # [MWh_th/t]
                if not hasattr(self.model, "wh_util_eff"):
                    self.model.wh_util_eff = pyo.Param(initialize=0.90)

                @self.model.Constraint(self.model.time_steps)
                def whr_kiln_to_preheater(m, t):
                    # recovered waste heat = clinker_out * yield * efficiency
                    return (
                        m.dsm_blocks["preheater"].external_heat_in[t]
                        <= m.dsm_blocks["kiln"].clinker_out[t]
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

            # ---------------- Dynamic E-TES coupling to Calciner ----------------
            if self.has_calciner:
                cc = self.model.dsm_blocks["calciner"]

                if self.has_thermal_storage:
                    ts = self.model.dsm_blocks["thermal_storage"]

                    # Decide mode once (Param is fixed, so this is safe)
                    is_gen_mode = False
                    if hasattr(ts, "is_generator_mode"):
                        is_gen_mode = bool(pyo.value(ts.is_generator_mode))

                    if is_gen_mode:
                        # Generator mode: TES charge comes from its own electric heater (charge == power_in * eta)
                        @self.model.Constraint(self.model.time_steps)
                        def tes_to_calciner_effective_heat(m, t):
                            return cc.effective_heat_in[t] == cc.heat_out[t] + ts.discharge[t]

                    else:
                        # Non-generator mode: TES can ONLY charge from calciner-produced heat
                        # → charging diverts heat away from calciner process in the same timestep
                        @self.model.Constraint(self.model.time_steps)
                        def tes_charge_limited_by_calciner(m, t):
                            return ts.charge[t] <= cc.heat_out[t]

                        @self.model.Constraint(self.model.time_steps)
                        def tes_to_calciner_effective_heat(m, t):
                            return (
                                cc.effective_heat_in[t]
                                == cc.heat_out[t] - ts.charge[t] + ts.discharge[t]
                            )

                else:
                    @self.model.Constraint(self.model.time_steps)
                    def bind_eff_no_tes(m, t):
                        return cc.effective_heat_in[t] == cc.heat_out[t]


    def define_constraints(self):
        if self.is_prosumer:
            # self._add_fcr_capacity_market(self.model)
            # self._add_reserve_capacity_market_guardrail(self.model)
            self._add_reserve_capacity_market_threshold_guardrail(self.model)
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

        @self.model.Constraint(self.model.time_steps)
        def total_co2_emission_constraint(m, t):
            total = 0.0

            # Preheater (energy CO2)
            if self.has_preheater and ("preheater" in m.dsm_blocks):
                ph = m.dsm_blocks["preheater"]
                if hasattr(ph, "co2_emission"):
                    total += ph.co2_emission[t]

            # Calciner (process + energy CO2; fallback to co2_emission if alternative model)
            if self.has_calciner and ("calciner" in m.dsm_blocks):
                cc = m.dsm_blocks["calciner"]
                if hasattr(cc, "co2_proc"):
                    total += cc.co2_proc[t]
                if hasattr(cc, "co2_energy"):
                    total += cc.co2_energy[t]
                if (not hasattr(cc, "co2_proc")) and (not hasattr(cc, "co2_energy")) and hasattr(cc, "co2_emission"):
                    total += cc.co2_emission[t]

            # Kiln (energy CO2)
            if self.has_kiln and ("kiln" in m.dsm_blocks):
                rk = m.dsm_blocks["kiln"]
                if hasattr(rk, "co2_energy"):
                    total += rk.co2_energy[t]

            return m.total_co2_emission[t] == total


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
    
    def dashboard_cement(
            self,
            instance,
            baseline_instance=None,   # optional comparator (e.g., BAU)
            html_path="./outputs/cement_full_dashboard.html",
            sankey_max_steps=168,     # limit steps for file size (None = all)
            ):
        """
        One-stop interactive dashboard for the cement plant:
        • Dynamic operation (electric balance + pyroline thermal balance)
        • Time-slider Sankey (per-step flows Electricity/Fuels → Calciner/Kiln/Preheater/TES)
        • Storage analytics (duration curve, FIFO delivered-cost vs. displaced cost)
        • Emissions analytics (combustion + scope-2 + calcination; intensity vs t_clinker)
        • Economics analytics (variable-cost stack, KPIs)
        • Reserve analytics (FCR/aFRR) if reserve model is attached
        Works with any subset of: preheater, calciner, kiln, electrolyser, cement_mill, thermal_storage, pv_plant.
        """

        # --------------------- helpers ---------------------
        def safe(v):
            try:
                return float(pyo.value(v))
            except Exception:
                try:
                    return float(v)
                except Exception:
                    return 0.0

        def vec_param(model, name):
            return (
                [safe(getattr(model, name)[t]) for t in instance.time_steps]
                if hasattr(model, name)
                else None
            )

        def series_or_none(block, name):
            if block is None or not hasattr(block, name):
                return None
            return [safe(getattr(block, name)[t]) for t in instance.time_steps]

        def exists(a):
            return (a is not None) and any(abs(x) > 1e-9 for x in a)

        def zero_if_none(a):
            return a if a is not None else [0.0] * len(T)

        T  = list(instance.time_steps)
        dt = 1.0  # h/step

        # --------------------- blocks (IndexedBlock access) ---------------------
        B  = instance.dsm_blocks
        ph = B["preheater"]       if "preheater"       in B else None
        cc = B["calciner"]        if "calciner"        in B else None
        rk = B["kiln"]            if "kiln"            in B else None
        cm = B["cement_mill"]     if "cement_mill"     in B else None
        el = B["electrolyser"]    if "electrolyser"    in B else None
        ts = B["thermal_storage"] if "thermal_storage" in B else None
        pv = B["pv_plant"]        if "pv_plant"        in B else None

        # --------------------- prices & demand proxies ---------------------
        elec_price = vec_param(instance, "electricity_price")
        ng_price   = vec_param(instance, "natural_gas_price")
        h2_price   = vec_param(instance, "hydrogen_price")
        coal_price = vec_param(instance, "coal_price")
        co2_price  = vec_param(instance, "co2_price")

        # clinker series (if present)
        clinker_rate = (
            series_or_none(cc, "clinker_out")
            if cc and hasattr(cc, "clinker_out") else
            (vec_param(instance, "clinker_rate") if hasattr(instance, "clinker_rate") else None)
        )

        # --------------------- electric supply & loads ---------------------
        pv_P  = series_or_none(pv, "power")  # MW_e
        grid  = [safe(instance.grid_power[t]) for t in T] if hasattr(instance, "grid_power") else [0.0]*len(T)
        # Electric loads (MW_e)
        cc_P  = series_or_none(cc, "power_in")
        rk_P  = series_or_none(rk, "power_in")
        el_P  = series_or_none(el, "power_in")
        cm_P  = series_or_none(cm, "power_in")
        ts_P  = series_or_none(ts, "power_in")  # TES heater (only if generator-mode)

        total_power = (
            [safe(instance.total_power_input[t]) for t in T]
            if hasattr(instance, "total_power_input") else
            [zero_if_none(cc_P)[i] + zero_if_none(rk_P)[i] + zero_if_none(el_P)[i]
            + zero_if_none(cm_P)[i] + zero_if_none(ts_P)[i]
            for i in range(len(T))]
        )

        # --------------------- fuels & pyroline thermal flows ---------------------
        # Fuels to calciner/kiln (MW_th)
        cc_NG = series_or_none(cc, "natural_gas_in");  rk_NG = series_or_none(rk, "natural_gas_in")
        cc_H2 = series_or_none(cc, "hydrogen_in");     rk_H2 = series_or_none(rk, "hydrogen_in")
        cc_CO = series_or_none(cc, "coal_in");         rk_CO = series_or_none(rk, "coal_in")

        # Thermal outputs (MW_th)
        cc_Qout = series_or_none(cc, "heat_out")
        rk_Qout = series_or_none(rk, "heat_out")
        # Effective heat into calciner (TES discharge etc.)
        cc_Qeff = series_or_none(cc, "effective_heat_in")
        # Waste heat routed to preheater (MW_th)
        ph_WH   = series_or_none(ph, "external_heat_in")

        # TES (MW_th / MWh_th)
        ts_ch  = series_or_none(ts, "charge")
        ts_ds  = series_or_none(ts, "discharge")
        ts_soc = series_or_none(ts, "soc")
        # charging mode detection
        is_gen_param = getattr(ts, "is_generator_mode", None) if ts else None
        is_gen_mode  = bool(safe(is_gen_param)) if is_gen_param is not None else False

        # --------------------- OPERATION PANELS ---------------------
        # A) Electric balance + prices + clinker
        fig_op = ps.make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
            subplot_titles=("Electric balance (PV + Grid vs. Loads)",
                            "Pyroline thermal (Calciner/Kiln/TES/Waste heat)",
                            "Price signals & Clinker rate")
        )
        # Row 1: supply (bars) and loads (lines)
        if exists(pv_P): fig_op.add_trace(go.Bar(x=T, y=pv_P, name="PV [MWₑ]"), row=1, col=1)
        fig_op.add_trace(go.Bar(x=T, y=grid, name="Grid [MWₑ]"), row=1, col=1)

        fig_op.add_trace(go.Scatter(x=T, y=total_power, name="Elec loads total [MWₑ]", mode="lines"), row=1, col=1)
        for y, lbl in [(cc_P,"Calciner P [MWₑ]"), (rk_P,"Kiln P [MWₑ]"),
                    (el_P,"Electrolyser P [MWₑ]"), (cm_P,"Cement mill P [MWₑ]"),
                    (ts_P,"TES heater P [MWₑ]")]:
            if exists(y): fig_op.add_trace(go.Scatter(x=T, y=y, name=lbl, mode="lines"), row=1, col=1)

        # Row 2: pyroline thermal + TES
        for y, lbl in [(cc_Qout,"Calciner heat out [MWₜₕ]"),
                    (rk_Qout,"Kiln heat out [MWₜₕ]"),
                    (cc_Qeff,"Calciner eff. heat in [MWₜₕ]"),
                    (ph_WH,"Waste heat → Preheater [MWₜₕ]"),
                    (ts_ds,"TES discharge [MWₜₕ]")]:
            if exists(y): fig_op.add_trace(go.Bar(x=T, y=y, name=lbl), row=2, col=1)
        if exists(ts_ch):
            fig_op.add_trace(go.Bar(x=T, y=[-v for v in ts_ch], name="TES charge (−) [MWₜₕ]"), row=2, col=1)

        # Row 3: prices + clinker
        if exists(elec_price): fig_op.add_trace(go.Scatter(x=T, y=elec_price, name="Elec price [€/MWhₑ]"), row=3, col=1)
        if exists(ng_price):   fig_op.add_trace(go.Scatter(x=T, y=ng_price,   name="NG price [€/MWhₜₕ]"), row=3, col=1)
        if exists(h2_price):   fig_op.add_trace(go.Scatter(x=T, y=h2_price,   name="H₂ price [€/MWhₜₕ]"), row=3, col=1)
        if exists(coal_price): fig_op.add_trace(go.Scatter(x=T, y=coal_price, name="Coal price [€/MWhₜₕ]"), row=3, col=1)
        if exists(clinker_rate):
            fig_op.add_trace(go.Scatter(x=T, y=clinker_rate, name="Clinker rate [t/h]", yaxis="y4", mode="lines"), row=3, col=1)
            # add a secondary y on row 3 by updating layout after creation
            fig_op.layout["yaxis3"].title = "€/MWh"
        fig_op.update_layout(barmode="stack", height=930, title="Cement Plant – Dynamic Operation")
        fig_op.update_yaxes(title_text="MWₑ", row=1, col=1)
        fig_op.update_yaxes(title_text="MWₜₕ", row=2, col=1)
        fig_op.update_yaxes(title_text="€/MWh", row=3, col=1)

        # --------------------- TIME-SLIDER SANKEY ---------------------
        maxN = len(T) if sankey_max_steps is None else min(len(T), sankey_max_steps)
        steps = list(range(maxN))

        nodes = [
            "Electricity","PV","Grid","Electrolyser","Cement Mill",
            "Calciner (E)","Kiln (E)","TES (charge)","TES (discharge)",
            "Natural Gas","Hydrogen","Coal","Calciner (fuel)","Kiln (fuel)",
            "Preheater (WH)","Preheater","Calciner","Kiln","Clinker"
        ]
        nid = {n:i for i,n in enumerate(nodes)}
        sankey_traces = []

        for k in steps:
            src, tgt, val, lbl = [], [], [], []

            def add(s,t,v,l):
                if v > 1e-9:
                    src.append(nid[s]); tgt.append(nid[t]); val.append(float(v)); lbl.append(l)

            # Electricity split
            if exists(pv_P): add("PV","Electricity", pv_P[k], "PV→E [MWₑ]")
            add("Grid","Electricity", grid[k], "Grid→E [MWₑ]")
            if exists(el_P): add("Electricity","Electrolyser", el_P[k], "E→EL [MWₑ]")
            if exists(cm_P): add("Electricity","Cement Mill", cm_P[k], "E→CM [MWₑ]")
            if exists(cc_P): add("Electricity","Calciner (E)", cc_P[k], "E→Calciner [MWₑ]")
            if exists(rk_P): add("Electricity","Kiln (E)", rk_P[k], "E→Kiln [MWₑ]")
            if exists(ts_P) and ts_P[k] > 1e-9: add("Electricity","TES (charge)", ts_P[k], "E→TES [MWₑ]")

            # Fuels to calciner/kiln
            if exists(cc_NG): add("Natural Gas","Calciner (fuel)", cc_NG[k], "NG→Calc [MWₜₕ]")
            if exists(cc_H2): add("Hydrogen","Calciner (fuel)", cc_H2[k], "H₂→Calc [MWₜₕ]")
            if exists(cc_CO): add("Coal","Calciner (fuel)", cc_CO[k], "Coal→Calc [MWₜₕ]")

            if exists(rk_NG): add("Natural Gas","Kiln (fuel)", rk_NG[k], "NG→Kiln [MWₜₕ]")
            if exists(rk_H2): add("Hydrogen","Kiln (fuel)", rk_H2[k], "H₂→Kiln [MWₜₕ]")
            if exists(rk_CO): add("Coal","Kiln (fuel)", rk_CO[k], "Coal→Kiln [MWₜₕ]")

            # Thermal flows to units
            if exists(cc_Qout): add("Calciner (fuel)","Calciner", cc_Qout[k], "Fuel→Calciner heat [MWₜₕ]")
            if exists(rk_Qout): add("Kiln (fuel)","Kiln", rk_Qout[k], "Fuel→Kiln heat [MWₜₕ]")
            if exists(ts_ds):   add("TES (discharge)","Calciner", ts_ds[k], "TES→Calc [MWₜₕ]")
            if exists(ph_WH):   add("Preheater (WH)","Preheater", ph_WH[k], "WH→Preheater [MWₜₕ]")

            # Downstream (simplified): Calciner+Kiln → Clinker
            q_ck = (cc_Qout[k] if cc_Qout else 0.0) + (rk_Qout[k] if rk_Qout else 0.0)
            if q_ck > 1e-9: add("Calciner","Clinker", q_ck, "Pyroline → Clinker [MWₜₕ]")

            sankey_traces.append(go.Sankey(
                node=dict(label=nodes, pad=12, thickness=18),
                link=dict(source=src, target=tgt, value=val, label=lbl),
                visible=False, valueformat=".2f"
            ))
        if sankey_traces: sankey_traces[0].visible = True

        sliders = [dict(
            active=0, currentvalue={"prefix":"t = "}, pad={"t": 10},
            steps=[dict(method="update",
                        args=[{"visible":[(i==k) for i in range(len(sankey_traces))]}],
                        label=str(T[k])) for k in range(len(sankey_traces))]
        )]
        fig_sk = go.Figure(data=sankey_traces)
        fig_sk.update_layout(title="Per-step Sankey (cement)", sliders=sliders, height=560)

        # --------------------- STORAGE ANALYTICS (same logic as steam, adapted) ---------------------
        figs_tes = []
        if ts is not None:
            # duration curve
            if exists(ts_ds):
                dc = sorted([v for v in ts_ds if v > 1e-9], reverse=True)
                fig_dc = go.Figure(go.Scatter(y=dc, mode="lines", name="Discharge"))
                fig_dc.update_layout(title="TES Discharge Duration Curve", yaxis_title="MWₜₕ",
                                    xaxis_title="Ranked hours")
                figs_tes.append(fig_dc)

            # SOC & flows
            fig_ts = ps.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                    subplot_titles=("TES charge/discharge", "TES state of charge"))
            if exists(ts_ch): fig_ts.add_trace(go.Scatter(x=T, y=ts_ch, name="Charge [MWₜₕ]"), row=1, col=1)
            if exists(ts_ds): fig_ts.add_trace(go.Scatter(x=T, y=ts_ds, name="Discharge [MWₜₕ]"), row=1, col=1)
            if exists(ts_soc): fig_ts.add_trace(go.Scatter(x=T, y=ts_soc, name="SOC [MWhₜₕ]", fill="tozeroy"), row=2, col=1)
            fig_ts.update_layout(title="TES Operation", height=520)
            figs_tes.append(fig_ts)

            # simple delivered-cost vs displaced calciner/kiln cost (€/MWh_th)
            if elec_price and ts_ds:
                eta_d  = safe(getattr(ts, "eta_discharge", 0.95)) if ts else 0.95
                eta_e2h= safe(getattr(ts, "eta_electric_to_heat", 0.95)) if ts else 0.95
                # delivered €/MWh_th for E-charged storage: p_e / (eta_e2h * eta_d)
                c_delivered = [elec_price[i] / max(eta_e2h * eta_d, 1e-9) for i in range(len(T))]
                # displaced cost proxy: min of fossil marginal costs (simple)
                eta_foss = 0.90
                c_NG   = [ng_price[i]   / eta_foss if ng_price   else np.inf for i in range(len(T))]
                c_COAL = [coal_price[i] / eta_foss if coal_price else np.inf for i in range(len(T))]
                c_H2   = [h2_price[i]   / eta_foss if h2_price   else np.inf for i in range(len(T))]
                displaced = [min(c_NG[i], c_COAL[i], c_H2[i]) for i in range(len(T))]
                x=[]; y=[]; col=[]
                for i in range(len(T)):
                    if ts_ds[i] > 1e-9:
                        x.append(c_delivered[i]); y.append(displaced[i]); col.append(ts_ds[i])
                if x:
                    fig_sc = go.Figure(go.Scatter(x=x, y=y, mode="markers",
                                                marker=dict(size=6, color=col, colorbar=dict(title="Discharge [MWₜₕ]"))))
                    fig_sc.update_layout(title="TES Delivered Cost vs Displaced Pyroline Cost",
                                        xaxis_title="TES delivered [€/MWhₜₕ]", yaxis_title="Cheapest alt. [€/MWhₜₕ]")
                    figs_tes.append(fig_sc)
        
                # --- add the two price–operation scatter plots (like steam dashboard) ---
        if elec_price is not None:
            if ts_ch is not None:
                fig_chg_price = go.Figure(
                    go.Scatter(
                        x=ts_ch,
                        y=elec_price,
                        mode="markers",
                        marker=dict(size=6, opacity=0.6),
                        name="Charge pts",
                    )
                )
                fig_chg_price.update_layout(
                    title="TES Charge vs. Electricity Price",
                    xaxis_title="TES Charge Power [MW_th]",
                    yaxis_title="Electricity Price [€/MWh_e]",
                    height=400,
                )
                figs_tes.append(fig_chg_price)

            if ts_ds is not None:
                fig_dis_price = go.Figure(
                    go.Scatter(
                        x=ts_ds,
                        y=elec_price,
                        mode="markers",
                        marker=dict(size=6, opacity=0.6),
                        name="Discharge pts",
                    )
                )
                fig_dis_price.update_layout(
                    title="TES Discharge vs. Electricity Price",
                    xaxis_title="TES Discharge Power [MW_th]",
                    yaxis_title="Electricity Price [€/MWh_e]",
                    height=400,
                )
                figs_tes.append(fig_dis_price)

        # --------------------- EMISSIONS ANALYTICS (combustion + scope-2 + calcination) ---------------------
        grid_ef  = safe(getattr(instance, "electricity_emission_factor", 0.0)) if hasattr(instance,"electricity_emission_factor") else 0.0
        ng_ef    = 0.202  # tCO2/MWh_th
        coal_ef  = 0.341  # tCO2/MWh_th
        h2_ef    = 0.0
        calc_ef  = safe(getattr(cc, "calcination_emission_factor", 0.0)) if cc else safe(getattr(instance, "calcination_emission_factor", 0.0))  # tCO2/t clinker

        # combustion scope-1
        e_cc = [zero_if_none(cc_NG)[i]*ng_ef + zero_if_none(cc_CO)[i]*coal_ef + zero_if_none(cc_H2)[i]*h2_ef for i in range(len(T))]
        e_rk = [zero_if_none(rk_NG)[i]*ng_ef + zero_if_none(rk_CO)[i]*coal_ef + zero_if_none(rk_H2)[i]*h2_ef for i in range(len(T))]
        # scope-2 electric emissions (allocate to loads)
        el_load = total_power
        e_el = [el_load[i] * grid_ef for i in range(len(T))]
        # calcination (process) emissions
        e_proc = [ (clinker_rate[i] * calc_ef) if clinker_rate else 0.0 for i in range(len(T)) ]

        co2_ts = [e_cc[i] + e_rk[i] + e_el[i] + e_proc[i] for i in range(len(T))]
        E_total = float(np.sum(co2_ts) * dt)  # t
        Q_clink = float(np.sum(clinker_rate) * dt) if clinker_rate else 0.0  # t clinker
        intensity_tCO2_per_tcl = (E_total / Q_clink) if Q_clink > 1e-9 else np.nan

        fig_em = ps.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                                subplot_titles=("Emissions by source (tCO₂/h)", "Emissions intensity [tCO₂/t clinker]"))
        fig_em.add_trace(go.Scatter(x=T, y=e_cc,  name="Calciner combustion"), row=1, col=1)
        fig_em.add_trace(go.Scatter(x=T, y=e_rk,  name="Kiln combustion"),     row=1, col=1)
        fig_em.add_trace(go.Scatter(x=T, y=e_el,  name="Electricity (scope-2)"),row=1, col=1)
        if exists(e_proc): fig_em.add_trace(go.Scatter(x=T, y=e_proc, name="Calcination (process)"), row=1, col=1)
        intens_ts = [ (co2_ts[i]/max(clinker_rate[i],1e-9)) if clinker_rate else np.nan for i in range(len(T)) ]
        fig_em.add_trace(go.Scatter(x=T, y=intens_ts, name="Intensity [tCO₂/t]"), row=2, col=1)
        fig_em.update_layout(title=f"Emissions (Total={E_total:,.1f} t, Intensity={intensity_tCO2_per_tcl:.3f} tCO₂/t)",
                            height=740)
        fig_em.update_yaxes(title_text="tCO₂/h", row=1, col=1)
        fig_em.update_yaxes(title_text="tCO₂/t clinker", row=2, col=1)

        # --------------------- ECONOMICS ANALYTICS ---------------------
        # Hourly total variable cost [€/h]
        cost_ts = (
            [safe(instance.variable_cost[t]) for t in T]
            if hasattr(instance, "variable_cost")
            else None
        )

        # Clinker series for marginal cost denominator [t/h]
        clk_ts = (
            [safe(cc.clinker_out[t]) for t in instance.time_steps]
            if (cc is not None and hasattr(cc, "clinker_out"))
            else (
                [safe(getattr(instance, "clinker_rate")[t]) for t in instance.time_steps]
                if hasattr(instance, "clinker_rate")
                else None
            )
        )

        # Marginal cost [€/t clinker] = variable_cost[t] / clinker_rate[t]
        marg_ts = None
        if cost_ts is not None and clk_ts is not None:
            eps = 1e-9
            marg_ts = [cost_ts[i] / max(clk_ts[i], eps) for i in range(len(T))]

        # Component operating cost time series (if blocks expose .operating_cost[t])
        c_cc = series_or_none(cc, "operating_cost")
        c_rk = series_or_none(rk, "operating_cost")
        c_el = series_or_none(el, "operating_cost")
        c_cm = series_or_none(cm, "operating_cost")
        c_ts = series_or_none(ts, "operating_cost")
        c_pv = series_or_none(pv, "operating_cost")

        # Grid energy cost proxy (€/h) if no explicit operating_cost covers it
        grid_cost = (
            [(elec_price[i] * grid[i]) if elec_price else 0.0 for i in range(len(T))]
        )

        # Figure: stacked component costs (left), total cost line (left), marginal €/t line (right)
        fig_ec = ps.make_subplots(
            rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.07,
            specs=[[{"secondary_y": True}], [{"type": "table"}]],
            subplot_titles=("Variable cost stack (€/h) + Marginal cost (€/t)", "KPIs"),
        )

        # Stacked bars (component breakdown). Only add if there’s data.
        for series, name in [
            (c_cc, "Calciner €"),
            (c_rk, "Kiln €"),
            (c_el, "Electrolyser €"),
            (c_cm, "Cement mill €"),
            (c_ts, "TES €"),
            (c_pv, "PV €"),
            (grid_cost, "Grid €"),
        ]:
            if exists(series):
                fig_ec.add_trace(go.Bar(x=T, y=series, name=name), row=1, col=1)

        # Total variable cost line [€/h] (if provided by the model)
        if exists(cost_ts):
            fig_ec.add_trace(
                go.Scatter(x=T, y=cost_ts, name="Total variable cost [€/h]", mode="lines"),
                row=1, col=1, secondary_y=False,
            )

        # Marginal cost line [€/t clinker] on secondary axis
        if marg_ts is not None and any(abs(v) > 1e-9 for v in marg_ts):
            fig_ec.add_trace(
                go.Scatter(
                    x=T, y=marg_ts, name="Marginal cost [€/t clinker]",
                    mode="lines", line=dict(width=2, dash="dot")
                ),
                row=1, col=1, secondary_y=True,
            )

        # Axis titles
        fig_ec.update_yaxes(title_text="€/h", row=1, col=1, secondary_y=False)
        fig_ec.update_yaxes(title_text="€/t clinker", row=1, col=1, secondary_y=True)
        fig_ec.update_layout(barmode="stack")

        # KPIs
        # Total variable cost over horizon (€/h integrated over hours)
        if cost_ts is not None:
            total_cost = float(np.nansum(cost_ts) * dt)
        else:
            # fallback to grid_cost only if no cost_ts
            total_cost = float(np.nansum(grid_cost) * dt)

        # Q_clink was computed earlier in the Emissions section:
        #   Q_clink = float(np.sum(clinker_rate) * dt) if clinker_rate else 0.0
        LCOC_var = (total_cost / max(Q_clink, 1e-9)) if Q_clink > 1e-9 else np.nan  # €/t clinker

        rows = [
            ("Total variable cost [€]", f"{total_cost:,.0f}"),
            ("Clinker produced [t]", f"{Q_clink:,.0f}"),
            ("Variable cost / t clinker [€/t]", f"{LCOC_var:,.2f}" if np.isfinite(LCOC_var) else "—"),
        ]

        # Marginal-cost statistics (if we have a valid series)
        if marg_ts is not None:
            arr = np.array(marg_ts, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                rows += [
                    ("Average marginal cost [€/t]", f"{float(np.mean(arr)):.2f}"),
                    (
                        "P10 / P50 / P90 [€/t]",
                        f"{float(np.percentile(arr,10)):.2f} / "
                        f"{float(np.percentile(arr,50)):.2f} / "
                        f"{float(np.percentile(arr,90)):.2f}"
                    ),
                    ("Min / Max [€/t]", f"{float(np.min(arr)):.2f} / {float(np.max(arr)):.2f}"),
                ]

        # Table
        fig_ec.add_trace(
            go.Table(
                header=dict(values=["Economic KPI", "Value"], align="left"),
                cells=dict(
                    values=[[r[0] for r in rows], [r[1] for r in rows]],
                    align="left",
                ),
            ),
            row=2, col=1,
        )

        fig_ec.update_layout(title="Economics", height=760)


        # --------------------- BASELINE COMPARISON (optional) ---------------------
        figs_cmp = []
        if baseline_instance is not None and hasattr(baseline_instance, "time_steps"):
            # quick totals for baseline (E_total_base, cost_base, clinker_base)
            def totals(inst):
                BB = inst.dsm_blocks
                def g(b,n):
                    return [safe(getattr(BB[b], n)[t]) for t in inst.time_steps] if (b in BB and hasattr(BB[b], n)) else [0.0]*len(inst.time_steps)
                T0 = list(inst.time_steps)
                # emissions
                ccNG, ccCO, ccH2 = g("calciner","natural_gas_in"), g("calciner","coal_in"), g("calciner","hydrogen_in")
                rkNG, rkCO, rkH2 = g("kiln","natural_gas_in"),     g("kiln","coal_in"),     g("kiln","hydrogen_in")
                elP = [g("calciner","power_in")[i] + g("kiln","power_in")[i] +
                    g("electrolyser","power_in")[i] + g("cement_mill","power_in")[i] +
                    g("thermal_storage","power_in")[i] for i in range(len(T0))]
                gridEF = safe(getattr(inst,"electricity_emission_factor", grid_ef))
                e_el0  = [elP[i]*gridEF for i in range(len(T0))]
                # calcination
                clk = g("calciner","clinker_out")
                calcEF = safe(getattr(inst,"calcination_emission_factor", calc_ef))
                e_proc0= [(clk[i]*calcEF) for i in range(len(T0))]
                E0 = float((np.sum(ccNG)*ng_ef + np.sum(ccCO)*coal_ef + np.sum(ccH2)*h2_ef
                        + np.sum(rkNG)*ng_ef + np.sum(rkCO)*coal_ef + np.sum(rkH2)*h2_ef)*dt
                        + np.sum(e_el0)*dt + np.sum(e_proc0)*dt)
                # cost
                cost0 = [safe(inst.variable_cost[t]) for t in T0] if hasattr(inst,"variable_cost") else [0.0]*len(T0)
                C0 = float(np.sum(cost0)*dt)
                # clinker
                Q0 = float(np.sum(clk)*dt)
                return E0, C0, Q0
            E0, C0, Q0 = totals(baseline_instance)
            dE, dC, dQ = E_total - E0, total_cost - C0, Q_clink - Q0
            mac = (dC / dE) if abs(dE) > 1e-9 else np.nan
            fig_b = go.Figure(go.Table(
                header=dict(values=["Baseline vs Scenario","Value"], align="left"),
                cells=dict(values=list(zip(*[
                    ("Δ Emissions [tCO₂]", f"{dE:,.1f}"),
                    ("Δ Variable cost [€]", f"{dC:,.0f}"),
                    ("Δ Clinker [t]", f"{dQ:,.1f}"),
                    ("Marginal abatement cost [€/tCO₂]", f"{mac:,.1f}" if mac==mac else "—"),
                ])), align="left")
            ))
            fig_b.update_layout(title="Baseline Comparison (Economy & CO₂)")
            figs_cmp.append(fig_b)

        # --------------------- RESERVE ANALYTICS (if built) ---------------------
        figs_res = []
        if bool(getattr(self, "is_prosumer", False)) and hasattr(instance, "fcr_blocks"):
            Lblk = int(getattr(self, "_FCR_BLOCK_LENGTH", 4))
            R = self._reserve_series(instance, T, Lblk)
            # Blocks stacked bars + prices
            x_blk = R.get("blocks", [])
            if x_blk:
                cap_sym = R["cap_sym"]; cap_up = R["cap_up"]; cap_dn = R["cap_dn"]
                chosen  = R["chosen"]
                y_sym = [cap_sym[i] if chosen[x_blk[i]]=="FCR" else 0.0 for i in range(len(x_blk))]
                y_up  = [cap_up[i]  if chosen[x_blk[i]]=="UP"  else 0.0 for i in range(len(x_blk))]
                y_dn  = [cap_dn[i]  if chosen[x_blk[i]]=="DN"  else 0.0 for i in range(len(x_blk))]
                fig_blk = ps.make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
                fig_blk.add_trace(go.Bar(x=x_blk, y=y_sym, name="FCR sym [MW]", marker_color="#f0ad4e"), secondary_y=False)
                fig_blk.add_trace(go.Bar(x=x_blk, y=y_up,  name="aFRR Up [MW]", marker_color="#d62728"), secondary_y=False)
                fig_blk.add_trace(go.Bar(x=x_blk, y=y_dn,  name="aFRR Down [MW]", marker_color="#2ca02c"), secondary_y=False)
                p_sym = R.get("price_sym", []); p_pos = R.get("price_pos", []); p_neg = R.get("price_neg", [])
                if p_sym and any(abs(v)>1e-9 for v in p_sym):
                    fig_blk.add_trace(go.Scatter(x=x_blk, y=p_sym, name="FCR price [€/MW·4h]", line=dict(dash="dash", color="#9467bd")), secondary_y=True)
                if p_pos and any(abs(v)>1e-9 for v in p_pos):
                    fig_blk.add_trace(go.Scatter(x=x_blk, y=p_pos, name="aFRR+ price [€/MW·4h]", line=dict(dash="dot", color="#1f77b4")), secondary_y=True)
                if p_neg and any(abs(v)>1e-9 for v in p_neg):
                    fig_blk.add_trace(go.Scatter(x=x_blk, y=p_neg, name="aFRR− price [€/MW·4h]", line=dict(dash="dot", color="#ff7f0e")), secondary_y=True)
                fig_blk.update_layout(barmode="stack", title="Reserve Blocks: Capacity & Price", height=420)
                fig_blk.update_yaxes(title_text="MW", secondary_y=False); fig_blk.update_yaxes(title_text="€/MW·4h", secondary_y=True)
                figs_res.append(fig_blk)

                    # --- Revenue by block & KPI table (awarded-only) ---
            # revenue per block from awarded product only
            p_sym = R.get("price_sym", [])
            p_pos = R.get("price_pos", [])
            p_neg = R.get("price_neg", [])

            rev_sym = [(p_sym[i] * y_sym[i]) if i < len(p_sym) else 0.0 for i in range(len(x_blk))]
            rev_up  = [(p_pos[i] * y_up[i])  if i < len(p_pos) else 0.0 for i in range(len(x_blk))]
            rev_dn  = [(p_neg[i] * y_dn[i])  if i < len(p_neg) else 0.0 for i in range(len(x_blk))]
            rev_tot = [rev_sym[i] + rev_up[i] + rev_dn[i] for i in range(len(x_blk))]

            # KPIs
            blk_len = int(getattr(self, "_FCR_BLOCK_LENGTH", 4))  # h per block
            tot_sym_MWblk = float(np.sum(y_sym))
            tot_up_MWblk  = float(np.sum(y_up))
            tot_dn_MWblk  = float(np.sum(y_dn))
            tot_sym_MWh   = blk_len * tot_sym_MWblk
            tot_up_MWh    = blk_len * tot_up_MWblk
            tot_dn_MWh    = blk_len * tot_dn_MWblk

            total_revenue = float(np.sum(rev_tot))
            denom_MWblk = max(tot_sym_MWblk + tot_up_MWblk + tot_dn_MWblk, 1e-9)
            mw_w_price_blk  = total_revenue / denom_MWblk              # €/MW·4h
            mw_w_price_hour = mw_w_price_blk / max(blk_len, 1e-9)      # €/MW·h

            blocks_with_bid = sum(1 for i in range(len(x_blk)) if (y_sym[i] + y_up[i] + y_dn[i]) > 1e-9)
            min_bid = float(getattr(self, "_FCR_MIN_BID_MW", 1.0))
            step_mw = float(getattr(self, "_FCR_STEP_MW", 1.0))

            kpi_rows = [
                ("Blocks with any bid",           f"{blocks_with_bid} / {len(x_blk)}"),
                ("Total FCR (MW·blocks)",         f"{tot_sym_MWblk:,.0f}"),
                ("Total aFRR Up (MW·blocks)",     f"{tot_up_MWblk:,.0f}"),
                ("Total aFRR Down (MW·blocks)",   f"{tot_dn_MWblk:,.0f}"),
                ("Total FCR (MW·h)",              f"{tot_sym_MWh:,.0f}"),
                ("Total aFRR Up (MW·h)",          f"{tot_up_MWh:,.0f}"),
                ("Total aFRR Down (MW·h)",        f"{tot_dn_MWh:,.0f}"),
                ("Total reserve revenue [€]",     f"{total_revenue:,.0f}"),
                ("MW-weighted price [€/MW·4h]",   f"{mw_w_price_blk:,.1f}"),
                ("Hourly-normalised [€/MW·h]",    f"{mw_w_price_hour:,.2f}"),
                ("Min bid / Step [MW]",           f"{min_bid:g} / {step_mw:g}"),
            ]

            fig_res_kpi = go.Figure(
                go.Table(
                    header=dict(values=["Reserve KPI", "Value"], align="left"),
                    cells=dict(
                        values=[[r[0] for r in kpi_rows], [r[1] for r in kpi_rows]],
                        align="left",
                    ),
                )
            )
            fig_res_kpi.update_layout(title="Reserve Market KPIs")
            figs_res.append(fig_res_kpi)

        # --------------------- WRITE SINGLE HTML ---------------------
        Path(html_path).parent.mkdir(parents=True, exist_ok=True)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("<html><head><meta charset='utf-8'><title>Cement Plant – Full Dashboard</title></head><body>")
            f.write("<h2>Cement Plant – Full Dashboard</h2>")

            f.write("<h3>Dynamic Operation</h3>")
            f.write(fig_op.to_html(full_html=False, include_plotlyjs="cdn"))

            f.write("<hr><h3>Time-slider Sankey</h3>")
            f.write(fig_sk.to_html(full_html=False, include_plotlyjs=False))

            if figs_tes:
                f.write("<hr><h3>Thermal Storage Analytics</h3>")
                for g in figs_tes:
                    f.write(g.to_html(full_html=False, include_plotlyjs=False))

            f.write("<hr><h3>Emissions Analytics</h3>")
            f.write(fig_em.to_html(full_html=False, include_plotlyjs=False))

            f.write("<hr><h3>Economics</h3>")
            f.write(fig_ec.to_html(full_html=False, include_plotlyjs=False))

            if figs_cmp:
                f.write("<hr><h3>Baseline Comparison</h3>")
                for g in figs_cmp:
                    f.write(g.to_html(full_html=False, include_plotlyjs=False))

            if figs_res:
                f.write("<hr><h3>Reserve Market (FCR / aFRR) Analytics</h3>")
                for g in figs_res:
                    f.write(g.to_html(full_html=False, include_plotlyjs=False))

            f.write("</body></html>")
        print(f"Full dashboard saved to: {html_path}")

    ###################Dashboard functions####################
    def plot_1(self, instance, save_name=None, out_dir="./outputs", show=True):
        """
        Two-panel matplotlib figure (steam-plant style) + CSV export:
        Top: electric inputs (calciner/kiln/electrolyser/cement mill/TES heater) and fossil inputs (NG/H2/Coal) + energy prices (twin y-axis)
        Bottom: pyroline & TES operation (calciner/kiln heat out, calciner effective heat in, WH→preheater, TES charge/discharge/SOC)

        If save_path is given, saves a PNG and a CSV (same columns as plotted, plus prices and variable cost).
        Presence-aware: draws only what exists.
        """

        # ------- style -------
        mpl.rcParams.update(
            {
                "font.size": 13,
                "font.family": "serif",
                "axes.titlesize": 15,
                "axes.labelsize": 13,
                "legend.fontsize": 12,
                "lines.linewidth": 2,
                "axes.grid": True,
                "grid.linestyle": "--",
                "grid.alpha": 0.7,
                "figure.dpi": 120,
            }
        )

        T = list(instance.time_steps)

        # ------- helpers -------
        def safe_value(v):
            try:
                return float(pyo.value(v))
            except Exception:
                return 0.0

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

        # ------- blocks (IndexedBlock-style access) -------
        B = instance.dsm_blocks
        ph = B["preheater"] if "preheater" in B else None
        cc = B["calciner"] if "calciner" in B else None
        rk = B["kiln"] if "kiln" in B else None
        cm = B["cement_mill"] if "cement_mill" in B else None
        el = B["electrolyser"] if "electrolyser" in B else None
        ts = B["thermal_storage"] if "thermal_storage" in B else None

        # ------- series -------
        # Electric inputs (MW_e)
        ph_P = series_or_none(ph, "power_in")                 # preheater
        cc_P = series_or_none(cc, "power_in")
        rk_P = series_or_none(rk, "power_in")
        el_P = series_or_none(el, "power_in")
        cm_P = series_or_none(cm, "power_in")                 # cement mill
        ts_P = series_or_none(ts, "power_in")                 # TES heater (generator mode)

        # Fossil/H2 fuel inputs (MW_th) to calciner/kiln
        cc_NG = series_or_none(cc, "natural_gas_in")
        cc_H2 = series_or_none(cc, "hydrogen_in")
        cc_CO = series_or_none(cc, "coal_in")

        rk_NG = series_or_none(rk, "natural_gas_in")
        rk_H2 = series_or_none(rk, "hydrogen_in")
        rk_CO = series_or_none(rk, "coal_in")

        # Pyroline thermal flows (MW_th)
        cc_Qout = series_or_none(cc, "heat_out")
        rk_Qout = series_or_none(rk, "heat_out")
        cc_Qeff = series_or_none(cc, "effective_heat_in")     # includes TES discharge if present
        ph_WH   = series_or_none(ph, "external_heat_in")      # waste heat routed to preheater

        # TES (MW_th / MWh_th)
        ts_ch  = series_or_none(ts, "charge")
        ts_ds  = series_or_none(ts, "discharge")
        ts_soc = series_or_none(ts, "soc")

        # Prices & plant-level metrics (optional)
        elec_price = (
            [safe_value(instance.electricity_price[t]) for t in T]
            if hasattr(instance, "electricity_price")
            else None
        )
        ng_price = (
            [safe_value(instance.natural_gas_price[t]) for t in T]
            if hasattr(instance, "natural_gas_price")
            else None
        )
        coal_price = (
            [safe_value(instance.coal_price[t]) for t in T]
            if hasattr(instance, "coal_price")
            else None
        )
        h2_price = (
            [safe_value(instance.hydrogen_price[t]) for t in T]
            if hasattr(instance, "hydrogen_price")
            else None
        )
        co2_price = (
            [safe_value(instance.co2_price[t]) for t in T]
            if hasattr(instance, "co2_price")
            else None
        )
        variable_cost = (
            [safe_value(instance.variable_cost[t]) for t in T]
            if hasattr(instance, "variable_cost")
            else None
        )

        # Emissions (tCO2 per time step) – presence-aware
        ph_co2 = series_or_none(ph, "co2_emission")
        rk_co2 = series_or_none(rk, "co2_energy")
        cc_co2_proc = series_or_none(cc, "co2_proc")
        cc_co2_energy = series_or_none(cc, "co2_energy")
        cc_co2_alt = series_or_none(cc, "co2_emission")

        cc_co2_total = None
        if (cc_co2_proc is not None) or (cc_co2_energy is not None):
            cc_co2_total = [
                (cc_co2_proc[i] if cc_co2_proc is not None else 0.0)
                + (cc_co2_energy[i] if cc_co2_energy is not None else 0.0)
                for i in range(len(T))
            ]
        elif cc_co2_alt is not None:
            cc_co2_total = cc_co2_alt

        # Prefer plant-level aggregation if available; else compute from components
        total_co2 = (
            [safe_value(instance.total_co2_emission[t]) for t in T]
            if hasattr(instance, "total_co2_emission")
            else (
                [
                    (ph_co2[i] if ph_co2 is not None else 0.0)
                    + (cc_co2_total[i] if cc_co2_total is not None else 0.0)
                    + (rk_co2[i] if rk_co2 is not None else 0.0)
                    for i in range(len(T))
                ]
                if (ph_co2 is not None) or (cc_co2_total is not None) or (rk_co2 is not None)
                else None
            )
        )

        # Optional plant aggregates
        total_power = (
            [safe_value(instance.total_power_input[t]) for t in T]
            if hasattr(instance, "total_power_input")
            else None
        )
        clinker_rate = (
            [safe_value(instance.clinker_rate[t]) for t in T]
            if hasattr(instance, "clinker_rate")
            else (
                [safe_value(instance.clinker_demand_per_time_step[t]) for t in T]
                if hasattr(instance, "clinker_demand_per_time_step")
                else None
            )
        )

        # ------- figure -------
        fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex=True, constrained_layout=True)

        # ---------- TOP: electric & fossil inputs + prices ----------
        lines_top = []
        # Electric loads
        lines_top += [
            plot_if_nonzero(axs[0], T, cc_P, "Calciner Power [MWₑ]", "C1"),
            plot_if_nonzero(axs[0], T, rk_P, "Kiln Power [MWₑ]", "C5", "-."),
            plot_if_nonzero(axs[0], T, el_P, "Electrolyser Power [MWₑ]", "C9", "--"),
            plot_if_nonzero(axs[0], T, cm_P, "Cement Mill Power [MWₑ]", "C4", ":"),
            plot_if_nonzero(axs[0], T, ts_P, "TES Heater Power [MWₑ]", "C19", "-."),
        ]

        # Fossil/H2 fuels into calciner/kiln (fuel power)
        lines_top += [
            plot_if_nonzero(axs[0], T, cc_NG, "Calciner NG [MWₜₕ]", "C2"),
            plot_if_nonzero(axs[0], T, cc_H2, "Calciner H₂ [MWₜₕ]", "C3", "--"),
            plot_if_nonzero(axs[0], T, cc_CO, "Calciner Coal [MWₜₕ]", "C7", "-."),
            plot_if_nonzero(axs[0], T, rk_NG, "Kiln NG [MWₜₕ]", "C8"),
            plot_if_nonzero(axs[0], T, rk_H2, "Kiln H₂ [MWₜₕ]", "C0", "--"),
            plot_if_nonzero(axs[0], T, rk_CO, "Kiln Coal [MWₜₕ]", "C6", "-."),
        ]

        axs[0].set_ylabel("Inputs (MWₑ / MWₜₕ)")
        axs[0].set_title("Cement Plant – Inputs & Energy Prices")
        axs[0].grid(True, which="both", axis="both")

        # Prices (twin y-axis)
        axp = axs[0].twinx()
        lp = []
        if elec_price:
            lp += axp.plot(T, elec_price, label="Elec Price [€/MWhₑ]", color="C10", linestyle="--")
        if ng_price:
            lp += axp.plot(T, ng_price, label="NG Price [€/MWhₜₕ]", color="C11", linestyle=":")
        if coal_price:
            lp += axp.plot(T, coal_price, label="Coal Price [€/MWhₜₕ]", color="C12", linestyle="-.")
        # if h2_price:
        #     lp += axp.plot(T, h2_price, label="H₂ Price [€/MWhₜₕ]", color="C13", linestyle="-")
        axp.set_ylabel("Energy Price", color="gray")
        axp.tick_params(axis="y", labelcolor="gray")

        # Legend (only existing handles)
        handles = [h for h in lines_top if h is not None] + lp
        labels = [h.get_label() for h in handles]
        if handles:
            axs[0].legend(handles, labels, loc="upper left", frameon=True, ncol=2)

        # ---------- BOTTOM: pyroline & TES ----------
        l1 = plot_if_nonzero(axs[1], T, cc_Qout, "Calciner Heat Out [MWₜₕ]", "C1")
        l2 = plot_if_nonzero(axs[1], T, rk_Qout, "Kiln Heat Out [MWₜₕ]", "C5", "-.")
        l3 = plot_if_nonzero(axs[1], T, cc_Qeff, "Calciner Effective Heat In [MWₜₕ]", "C2", "--")
        l4 = plot_if_nonzero(axs[1], T, ph_WH, "Waste Heat → Preheater [MWₜₕ]", "C9", ":")

        # TES
        p0 = plot_if_nonzero(axs[1], T, ts_ch, "TES Charge [MWₜₕ]", "C3", "-")
        p1 = plot_if_nonzero(axs[1], T, ts_ds, "TES Discharge [MWₜₕ]", "C4", "--")
        if ts_soc and any(abs(v) > 1e-9 for v in ts_soc):
            axs[1].fill_between(T, ts_soc, 0, color="C0", alpha=0.25, label="TES SOC [MWhₜₕ]")

        axs[1].set_ylabel("MWₜₕ / MWhₜₕ")
        axs[1].set_title("Pyroline & Thermal Storage")
        axs[1].grid(True, which="both", axis="both")
        blines = [h for h in [l1, l2, l3, l4, p0, p1] if h is not None]
        if blines:
            axs[1].legend(loc="upper left", frameon=True, ncol=2)
        axs[1].set_xlabel("Time step")

        plt.tight_layout()

        # --- variable & marginal cost helpers ---
        var_cost = (
            [safe_value(instance.variable_cost[t]) for t in T]
            if hasattr(instance, "variable_cost") else None
        )

        # clinker series (prefer calciner.clinker_out; fallback to model param)
        clinker_rate = (
            [safe_value(cc.clinker_out[t]) for t in T] if (cc is not None and hasattr(cc, "clinker_out"))
            else ([safe_value(instance.clinker_rate[t]) for t in T] if hasattr(instance, "clinker_rate") else None)
        )

        # marginal cost €/t clinker (hourly): cost[t] / max(clinker[t], eps)
        marg_cost = None
        if var_cost is not None and clinker_rate is not None:
            eps = 1e-9
            marg_cost = [ (var_cost[i] / max(clinker_rate[i], eps)) for i in range(len(T)) ]

        # -------- CSV export (same columns as plotted, plus prices etc.) --------
        df = pd.DataFrame(
            {
                "t": T,
                # Electric inputs
                "Preheater Power [MW_e]": ph_P if ph_P else None,
                "Calciner Power [MW_e]": cc_P if cc_P else None,
                "Kiln Power [MW_e]": rk_P if rk_P else None,
                "Electrolyser Power [MW_e]": el_P if el_P else None,
                "Cement Mill Power [MW_e]": cm_P if cm_P else None,
                "TES Heater Power [MW_e]": ts_P if ts_P else None,
                # Fuels to calciner/kiln
                "Calciner NG [MW_th]": cc_NG if cc_NG else None,
                "Calciner H2 [MW_th]": cc_H2 if cc_H2 else None,
                "Calciner Coal [MW_th]": cc_CO if cc_CO else None,
                "Kiln NG [MW_th]": rk_NG if rk_NG else None,
                "Kiln H2 [MW_th]": rk_H2 if rk_H2 else None,
                "Kiln Coal [MW_th]": rk_CO if rk_CO else None,
                # Pyroline thermal & WH
                "Calciner Heat Out [MW_th]": cc_Qout if cc_Qout else None,
                "Kiln Heat Out [MW_th]": rk_Qout if rk_Qout else None,
                "Calciner Effective Heat In [MW_th]": cc_Qeff if cc_Qeff else None,
                "Waste Heat to Preheater [MW_th]": ph_WH if ph_WH else None,
                # TES
                "TES Charge [MW_th]": ts_ch if ts_ch else None,
                "TES Discharge [MW_th]": ts_ds if ts_ds else None,
                "TES SOC [MWh_th]": ts_soc if ts_soc else None,

                # Emissions
                "Preheater CO2 [tCO2/step]": ph_co2 if ph_co2 else None,
                "Calciner CO2 process [tCO2/step]": cc_co2_proc if cc_co2_proc else None,
                "Calciner CO2 energy [tCO2/step]": cc_co2_energy if cc_co2_energy else None,
                "Calciner CO2 total [tCO2/step]": cc_co2_total if cc_co2_total else None,
                "Kiln CO2 [tCO2/step]": rk_co2 if rk_co2 else None,
                "Total CO2 [tCO2/step]": total_co2 if total_co2 else None,

                # Prices & plant-level
                "Elec Price [€/MWh_e]": elec_price if elec_price else None,
                "NG Price [€/MWh_th]": ng_price if ng_price else None,
                "Coal Price [€/MWh_th]": coal_price if coal_price else None,
                # "H2 Price [€/MWh_th]": h2_price if h2_price else None,
                "CO2 Price [€/tCO2]": co2_price if co2_price else None,
                "Total Variable Cost [€]": variable_cost if variable_cost else None,
                "Total Power Input [MW_e]": total_power if total_power else None,
                "Clinker Rate [t/h]": clinker_rate if clinker_rate else None,
                "Variable Cost [€/h]": var_cost if var_cost else None,
                "Marginal Cost [€/t clinker]": marg_cost if marg_cost else None,

            }
        )
            # ---------- SAVE: one name -> both .png and .csv ----------
        if save_name:
            out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(save_name).name  # strip any path
            # If user passed an extension, drop it
            if "." in stem:
                stem = Path(stem).stem
            png_path = out_dir / f"{stem}.png"
            csv_path = out_dir / f"{stem}.csv"

            plt.savefig(png_path, dpi=300, bbox_inches="tight")
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"Saved: {png_path}\nSaved: {csv_path}")

        if show:
            plt.show()
        plt.close(fig)

    def plot_capacity_products(self, instance, save_name=None, out_dir="./outputs", show=True):
        """
        Three-panel figure (cement twin of steam plot_2):
        1) Unit inputs (electric & fossil) + energy prices (twin y-axis)
        2) TES operation (charge, discharge, SOC)
        3) Reserve market view (FCR ± and aFRR Up/Down): capacity bands around baseline + AS prices

        If `save_name` is provided, saves:
        <out_dir>/<save_name>.png  and  <out_dir>/<save_name>.csv
        """

        # ------- style -------
        mpl.rcParams.update(
            {
                "font.size": 13,
                "font.family": "serif",
                "axes.titlesize": 15,
                "axes.labelsize": 13,
                "legend.fontsize": 12,
                "lines.linewidth": 2,
                "axes.grid": True,
                "grid.linestyle": "--",
                "grid.alpha": 0.7,
                "figure.dpi": 120,
            }
        )

        T = list(instance.time_steps)
        H = len(T)

        # ------- helpers -------
        def safe_value(v):
            try:
                return float(pyo.value(v))
            except Exception:
                try:
                    return float(v)
                except Exception:
                    return 0.0

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

        # ------- blocks (IndexedBlock) -------
        B  = instance.dsm_blocks
        ph = B["preheater"]      if "preheater"      in B else None
        cc = B["calciner"]       if "calciner"       in B else None
        rk = B["kiln"]           if "kiln"           in B else None
        cm = B["cement_mill"]    if "cement_mill"    in B else None
        el = B["electrolyser"]   if "electrolyser"   in B else None
        ts = B["thermal_storage"]if "thermal_storage" in B else None

        # ------- unit inputs -------
        # Electric loads (MW_e)
        cc_P = series_or_none(cc, "power_in")
        rk_P = series_or_none(rk, "power_in")
        el_P = series_or_none(el, "power_in")
        cm_P = series_or_none(cm, "power_in")
        ts_P = series_or_none(ts, "power_in")  # TES heater

        # Fuels to calciner/kiln (MW_th)
        cc_NG = series_or_none(cc, "natural_gas_in")
        cc_H2 = series_or_none(cc, "hydrogen_in")
        cc_CO = series_or_none(cc, "coal_in")
        rk_NG = series_or_none(rk, "natural_gas_in")
        rk_H2 = series_or_none(rk, "hydrogen_in")
        rk_CO = series_or_none(rk, "coal_in")

        # TES (MW_th / MWh_th)
        ts_ch  = series_or_none(ts, "charge")
        ts_ds  = series_or_none(ts, "discharge")
        ts_soc = series_or_none(ts, "soc")

        # prices
        elec_price = [safe_value(instance.electricity_price[t]) for t in T] if hasattr(instance, "electricity_price") else None
        ng_price   = [safe_value(instance.natural_gas_price[t]) for t in T] if hasattr(instance, "natural_gas_price") else None
        h2_price   = [safe_value(instance.hydrogen_price[t])    for t in T] if hasattr(instance, "hydrogen_price")    else None

        # baseline & envelope
        total_elec = [safe_value(instance.total_power_input[t]) for t in T] if hasattr(instance, "total_power_input") else None
        max_cap = getattr(self, "max_plant_capacity", None)
        min_cap = getattr(self, "min_plant_capacity", 0.0)

        # ------- figure -------
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)

        # ---------- TOP: inputs + prices ----------
        l1 = plot_if_nonzero(axs[0], T, cc_P, "Calciner Power [MWₑ]", "C1")
        l2 = plot_if_nonzero(axs[0], T, rk_P, "Kiln Power [MWₑ]", "C5", "-.")
        l3 = plot_if_nonzero(axs[0], T, el_P, "Electrolyser Power [MWₑ]", "C9", "--")
        l4 = plot_if_nonzero(axs[0], T, cm_P, "Cement Mill Power [MWₑ]", "C4", ":")
        l5 = plot_if_nonzero(axs[0], T, ts_P, "TES Heater Power [MWₑ]", "C19", "-.")

        l6 = plot_if_nonzero(axs[0], T, cc_NG, "Calciner NG [MWₜₕ]", "C2")
        l7 = plot_if_nonzero(axs[0], T, cc_H2, "Calciner H₂ [MWₜₕ]", "C3", "--")
        l8 = plot_if_nonzero(axs[0], T, cc_CO, "Calciner Coal [MWₜₕ]", "C7", "-.")
        l9 = plot_if_nonzero(axs[0], T, rk_NG, "Kiln NG [MWₜₕ]", "C8")
        lA = plot_if_nonzero(axs[0], T, rk_H2, "Kiln H₂ [MWₜₕ]", "C0", "--")
        lB = plot_if_nonzero(axs[0], T, rk_CO, "Kiln Coal [MWₜₕ]", "C6", "-.")

        axs[0].set_ylabel("Inputs")
        axs[0].set_title("Unit Inputs & Energy Prices")
        axp = axs[0].twinx()
        lp = []
        if elec_price: lp += axp.plot(T, elec_price, label="Elec Price [€/MWhₑ]", color="C6", linestyle="--")
        if ng_price:   lp += axp.plot(T, ng_price,   label="NG Price [€/MWhₜₕ]", color="C8", linestyle=":")
        if h2_price:   lp += axp.plot(T, h2_price,   label="H₂ Price [€/MWhₜₕ]", color="C9", linestyle="-.")
        axp.set_ylabel("Energy Price", color="gray"); axp.tick_params(axis="y", labelcolor="gray")
        handles = [h for h in [l1,l2,l3,l4,l5,l6,l7,l8,l9,lA,lB] if h is not None] + lp
        if handles:
            axs[0].legend(handles, [h.get_label() for h in handles], loc="upper left", frameon=True, ncol=2)

        # ---------- MIDDLE: TES operation ----------
        p0 = plot_if_nonzero(axs[1], T, ts_ch, "TES Charge [MWₜₕ]", "C2", "-")
        p1 = plot_if_nonzero(axs[1], T, ts_ds, "TES Discharge [MWₜₕ]", "C3", "--")
        if ts_soc and any(abs(v) > 1e-9 for v in ts_soc):
            axs[1].fill_between(T, ts_soc, 0, color="C0", alpha=0.25, label="TES SOC [MWhₜₕ]")
        axs[1].set_ylabel("MWₜₕ / MWhₜₕ")
        axs[1].set_title("Thermal Storage Operation")
        if any([p0, p1]): axs[1].legend(loc="upper right", frameon=True)

        # ---------- BOTTOM: Reserve products around baseline ----------
        fcr_present = hasattr(instance, "fcr_blocks") and list(getattr(instance, "fcr_blocks"))
        axs[2].set_ylabel("MW / MWₑ")

        base = total_elec if total_elec else [0.0] * H
        axs[2].plot(T, base, color="0.25", lw=2, label="Baseline Elec Load [MWₑ]")
        if max_cap is not None:
            axs[2].plot(T, [max_cap]*H, color="0.7", ls=":", label="Max Capability [MWₑ]")
        if min_cap is not None:
            axs[2].plot(T, [min_cap]*H, color="0.7", ls="--", label="Min Capability [MWₑ]")

        if fcr_present:
            Lblk = int(getattr(self, "_FCR_BLOCK_LENGTH", 4))
            R = self._reserve_series(instance, T, Lblk)

            base_arr = np.array(base)

            # FCR symmetric band: ±cap around baseline
            if R.get("mask_sym") and any(abs(v) > 1e-9 for v in R["mask_sym"]):
                sym = np.array(R["mask_sym"])
                lo = np.maximum(min_cap, base_arr - sym) if min_cap is not None else (base_arr - sym)
                hi = np.minimum(max_cap, base_arr + sym) if max_cap is not None else (base_arr + sym)
                axs[2].fill_between(T, lo, hi, step="pre", color="#f0ad4e", alpha=0.25, label="FCR sym [±MW]")

            # aFRR Up: [base - up, base]
            if R.get("mask_up") and any(abs(v) > 1e-9 for v in R["mask_up"]):
                up = np.array(R["mask_up"])
                lo = np.maximum(min_cap, base_arr - up) if min_cap is not None else (base_arr - up)
                axs[2].fill_between(T, lo, base_arr, step="pre", color="#d62728", alpha=0.25, label="aFRR Up [MW]")

            # aFRR Down: [base, base + down]
            if R.get("mask_dn") and any(abs(v) > 1e-9 for v in R["mask_dn"]):
                dn = np.array(R["mask_dn"])
                hi = np.minimum(max_cap, base_arr + dn) if max_cap is not None else (base_arr + dn)
                axs[2].fill_between(T, base_arr, hi, step="pre", color="#2ca02c", alpha=0.25, label="aFRR Down [MW]")

            # AS prices (stairs) on twin axis
            axp2 = axs[2].twinx()
            if R.get("stairs_price_sym") and any(abs(v) > 1e-9 for v in R["stairs_price_sym"]):
                axp2.plot(T, R["stairs_price_sym"], color="#9467bd", ls="--", label="FCR price [€/MW·4h]")
            if R.get("stairs_price_pos") and any(abs(v) > 1e-9 for v in R["stairs_price_pos"]):
                axp2.plot(T, R["stairs_price_pos"], color="#1f77b4", ls=":",  label="aFRR+ price [€/MW·4h]")
            if R.get("stairs_price_neg") and any(abs(v) > 1e-9 for v in R["stairs_price_neg"]):
                axp2.plot(T, R["stairs_price_neg"], color="#ff7f0e", ls="-.", label="aFRR− price [€/MW·4h]")
            axp2.set_ylabel("AS price", color="gray"); axp2.tick_params(axis="y", labelcolor="gray")

            # merge legends
            h1, l1 = axs[2].get_legend_handles_labels()
            h2, l2 = axp2.get_legend_handles_labels()
            axs[2].legend(h1 + h2, l1 + l2, loc="upper left", frameon=True)

            axs[2].set_title("Ancillary Bids — FCR (±) and aFRR (Up/Down) around Baseline")
        else:
            axs[2].set_title("Operational Baseline (no reserve structures)")
            axs[2].legend(loc="upper left", frameon=True)

        axs[-1].set_xlabel("Time step")
        fig.autofmt_xdate()
        plt.tight_layout()
        # --- variable & marginal cost helpers ---
        var_cost = (
            [safe_value(instance.variable_cost[t]) for t in T]
            if hasattr(instance, "variable_cost") else None
        )

        # clinker series (prefer calciner.clinker_out; fallback to model param)
        clinker_rate = (
            [safe_value(cc.clinker_out[t]) for t in T] if (cc is not None and hasattr(cc, "clinker_out"))
            else ([safe_value(instance.clinker_rate[t]) for t in T] if hasattr(instance, "clinker_rate") else None)
        )

        # marginal cost €/t clinker (hourly): cost[t] / max(clinker[t], eps)
        marg_cost = None
        if var_cost is not None and clinker_rate is not None:
            eps = 1e-9
            marg_cost = [ (var_cost[i] / max(clinker_rate[i], eps)) for i in range(len(T)) ]


        # ---------- SAVE: one name -> both .png and .csv ----------
        df = pd.DataFrame({
            "t": T,
            # Electric inputs
            "Calciner Power [MW_e]": cc_P if cc_P else None,
            "Kiln Power [MW_e]": rk_P if rk_P else None,
            "Electrolyser Power [MW_e]": el_P if el_P else None,
            "Cement Mill Power [MW_e]": cm_P if cm_P else None,
            "TES Heater Power [MW_e]": ts_P if ts_P else None,
            # Fuels
            "Calciner NG [MW_th]": cc_NG if cc_NG else None,
            "Calciner H2 [MW_th]": cc_H2 if cc_H2 else None,
            "Calciner Coal [MW_th]": cc_CO if cc_CO else None,
            "Kiln NG [MW_th]": rk_NG if rk_NG else None,
            "Kiln H2 [MW_th]": rk_H2 if rk_H2 else None,
            "Kiln Coal [MW_th]": rk_CO if rk_CO else None,
            # TES
            "TES Charge [MW_th]": ts_ch if ts_ch else None,
            "TES Discharge [MW_th]": ts_ds if ts_ds else None,
            "TES SOC [MWh_th]": ts_soc if ts_soc else None,
            # Prices & baseline
            "Elec Price [€/MWh_e]": elec_price if elec_price else None,
            "NG Price [€/MWh_th]": ng_price if ng_price else None,
            "H2 Price [€/MWh_th]": h2_price if h2_price else None,
            "Total Elec Load [MW_e]": total_elec if total_elec else None,
            "Variable Cost [€/h]": var_cost if var_cost else None,
            "Marginal Cost [€/t clinker]": marg_cost if marg_cost else None,

        })

        # add reserve (awarded) series if present
        if fcr_present:
            R = self._reserve_series(instance, T, int(getattr(self, "_FCR_BLOCK_LENGTH", 4)))
            def as_list(x, n): return list(x) if hasattr(x, "__len__") else [x]*n
            df["FCR Sym Cap [MW]"]   = as_list(R.get("mask_sym", [0.0]*H), H)
            df["aFRR Up Cap [MW]"]   = as_list(R.get("mask_up",  [0.0]*H), H)
            df["aFRR Down Cap [MW]"] = as_list(R.get("mask_dn",  [0.0]*H), H)
            if R.get("stairs_price_sym") is not None: df["FCR Price [€/MW·4h]"]   = as_list(R["stairs_price_sym"], H)
            if R.get("stairs_price_pos") is not None: df["aFRR Up Price [€/MW·4h]"] = as_list(R["stairs_price_pos"], H)
            if R.get("stairs_price_neg") is not None: df["aFRR Down Price [€/MW·4h]"] = as_list(R["stairs_price_neg"], H)

        if save_name:
            out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(save_name).stem
            png_path = out_dir / f"{stem}.png"
            csv_path = out_dir / f"{stem}.csv"
            plt.savefig(png_path, dpi=300, bbox_inches="tight")
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"Saved: {png_path}\nSaved: {csv_path}")

        if show:
            plt.show()
        plt.close(fig)

    ###### additional fucntions ######

    def _add_reserve_capacity_market_threshold_guardrail(self, model):
        """
        Reserve capacity market with threshold/guardrail strategy (cement variant).

        Key rule for ThermalStorage:
        - storage_type == "short-term": NO electric actuator -> must NOT contribute to max_cap/min_cap or total_power_input
        - storage_type == "short-term_with_generator": electric heater exists -> include ts.max_power (MW_e) and ts.power_in[t]

        Products:
        One 4h-block product per block: FCR (symmetric) OR aFRR (up XOR down).
        Block is ineligible if ANY hour in it has el-price <= threshold.

        Revenue (block-based) is added as Expression: m.reserve_revenue.
        """
        import math
        import pyomo.environ as pyo

        m = model

        # ---------------- helpers ----------------
        def _as_float_list(x, N):
            if x is None:
                return [0.0] * N
            v = [float(xx) for xx in x]
            if len(v) < N:
                v = v + [v[-1]] * (N - len(v))
            elif len(v) > N:
                v = v[:N]
            return v

        def _hourly_to_block_sum(hourly, starts, L):
            ps = [0.0]
            for x in hourly:
                ps.append(ps[-1] + x)
            return {b: ps[b + L] - ps[b] for b in starts}

        def _safe_getattr(obj, name, default=None):
            return getattr(obj, name) if hasattr(obj, name) else default

        def _pval(x, default=0.0):
            try:
                return float(pyo.value(x))
            except Exception:
                return float(default)

        def _get_comp_fuel(name):
            if hasattr(self, "components") and isinstance(self.components, dict) and name in self.components:
                return getattr(self.components[name], "fuel_type", None)
            return None

        def _get_storage_type():
            if hasattr(self, "components") and isinstance(self.components, dict) and "thermal_storage" in self.components:
                st = getattr(self.components["thermal_storage"], "storage_type", None)
                return (st or "").lower()
            return ""

        enable_fcr  = bool(getattr(self, "enable_fcr", False))
        enable_afrr = bool(getattr(self, "enable_afrr", True))

        L       = int(getattr(self, "_FCR_BLOCK_LENGTH", 4))   # hours per block
        step    = float(getattr(self, "_FCR_STEP_MW", 1.0))    # MW granularity
        min_bid = float(getattr(self, "_FCR_MIN_BID_MW", 1.0)) # MW minimum bid

        price_threshold = float(getattr(self, "reserve_price_threshold", 0.0))  # €/MWh_e

        # ---------------- time base & 4h blocks ----------------
        ts = self.index
        Tn = len(ts)
        starts = [i for i, dt in enumerate(ts[: -L + 1]) if (dt.hour % 4 == 0 and dt.minute == 0)]
        starts = [i for i in starts if i + L <= Tn]
        m.fcr_blocks = pyo.Set(initialize=starts, ordered=True)

        # ---------------- price series (hourly) ----------------
        if hasattr(m, "electricity_price"):
            el_price_hour = [float(pyo.value(m.electricity_price[t])) for t in m.time_steps]
        else:
            el_price_hour = _as_float_list(_safe_getattr(self, "electricity_price", None), Tn)

        price_fcr_hour = _as_float_list(_safe_getattr(self, "fcr_price", None), Tn)         # €/MW/h
        price_pos_hour = _as_float_list(_safe_getattr(self, "afrr_price_pos", None), Tn)    # €/MW/h
        price_neg_hour = _as_float_list(_safe_getattr(self, "afrr_price_neg", None), Tn)    # €/MW/h

        fcr_block_price = _hourly_to_block_sum(price_fcr_hour, starts, L) if enable_fcr else {b: 0.0 for b in starts}
        pos_block_price = _hourly_to_block_sum(price_pos_hour, starts, L) if enable_afrr else {b: 0.0 for b in starts}
        neg_block_price = _hourly_to_block_sum(price_neg_hour, starts, L) if enable_afrr else {b: 0.0 for b in starts}

        m.fcr_block_price      = pyo.Param(m.fcr_blocks, initialize=fcr_block_price, mutable=False)
        m.afrr_block_price_pos = pyo.Param(m.fcr_blocks, initialize=pos_block_price, mutable=False)
        m.afrr_block_price_neg = pyo.Param(m.fcr_blocks, initialize=neg_block_price, mutable=False)

        # ---------------- eligibility (guardrail) ----------------
        eligible = {}
        for b in starts:
            ok = all(el_price_hour[t] > price_threshold for t in range(b, b + L))
            eligible[b] = 1 if ok else 0
        m.block_eligible = pyo.Param(m.fcr_blocks, initialize=eligible, within=pyo.Boolean, mutable=False)

        # ---------------- thermal storage mode ----------------
        storage_type = _get_storage_type()
        tes_gen_mode = (storage_type == "short-term_with_generator")

        # ---------------- capacity envelope (electric head/foot room), in MW_e ----------------
        max_cap = float(getattr(self, "max_plant_capacity", 0.0))
        min_cap = float(getattr(self, "min_plant_capacity", 0.0))

        if (max_cap <= 0.0) and hasattr(m, "dsm_blocks"):
            B = m.dsm_blocks

            def _add_block_cap(name):
                nonlocal max_cap, min_cap
                if name not in B:
                    return
                blk = B[name]

                # Calciner/kiln are only "electric" if fuel_type == electricity
                if name in ("calciner", "kiln"):
                    if _get_comp_fuel(name) != "electricity":
                        return

                # Thermal storage: include ONLY if generator mode; use MW_e cap = blk.max_power
                if name == "thermal_storage":
                    if not tes_gen_mode:
                        return
                    if hasattr(blk, "max_power"):
                        max_cap += _pval(blk.max_power)
                    else:
                        # should not happen in generator mode, but keep safe:
                        max_cap += 0.0
                    if hasattr(blk, "min_power"):
                        min_cap += _pval(blk.min_power)
                    return

                # Other electric blocks: include if they expose max_power
                if hasattr(blk, "max_power"):
                    max_cap += _pval(blk.max_power)
                    if hasattr(blk, "min_power"):
                        min_cap += _pval(blk.min_power)

            _add_block_cap("calciner")
            _add_block_cap("kiln")
            _add_block_cap("electrolyser")
            _add_block_cap("cement_mill")
            _add_block_cap("thermal_storage")

        # Persist for transparency
        self.max_plant_capacity = max_cap
        self.min_plant_capacity = min_cap

        # ---------------- total electric load per hour (MW_e) ----------------
        # Reuse if present; else build Expression with correct TES handling.
        if not hasattr(m, "total_power_input"):
            def _total_P_rule(mm, t):
                P = 0.0
                if hasattr(mm, "dsm_blocks"):
                    B = mm.dsm_blocks

                    if "calciner" in B and _get_comp_fuel("calciner") == "electricity" and hasattr(B["calciner"], "power_in"):
                        P += B["calciner"].power_in[t]
                    if "kiln" in B and _get_comp_fuel("kiln") == "electricity" and hasattr(B["kiln"], "power_in"):
                        P += B["kiln"].power_in[t]
                    if "electrolyser" in B and hasattr(B["electrolyser"], "power_in"):
                        P += B["electrolyser"].power_in[t]
                    if "cement_mill" in B and hasattr(B["cement_mill"], "power_in"):
                        P += B["cement_mill"].power_in[t]

                    # TES electric actuator ONLY if generator mode
                    if tes_gen_mode and "thermal_storage" in B and hasattr(B["thermal_storage"], "power_in"):
                        P += B["thermal_storage"].power_in[t]

                if hasattr(mm, "grid_power"):
                    P += mm.grid_power[t]
                return P

            m.total_power_input = pyo.Expression(m.time_steps, rule=_total_P_rule)

        # ---------------- decision variables (integerized steps + binaries) ----------------
        M_blocks = int(math.ceil(max(1.0, max_cap) / step)) if step > 0 else 0

        # FCR symmetric
        m.k_sym   = pyo.Var(m.fcr_blocks, within=pyo.NonNegativeIntegers)
        m.cap_sym = pyo.Var(m.fcr_blocks, within=pyo.NonNegativeReals)
        m.bid_sym = pyo.Var(m.fcr_blocks, within=pyo.Binary)

        # aFRR up/down
        m.k_up    = pyo.Var(m.fcr_blocks, within=pyo.NonNegativeIntegers)
        m.k_dn    = pyo.Var(m.fcr_blocks, within=pyo.NonNegativeIntegers)
        m.cap_up  = pyo.Var(m.fcr_blocks, within=pyo.NonNegativeReals)
        m.cap_dn  = pyo.Var(m.fcr_blocks, within=pyo.NonNegativeReals)
        m.bid_up  = pyo.Var(m.fcr_blocks, within=pyo.Binary)
        m.bid_dn  = pyo.Var(m.fcr_blocks, within=pyo.Binary)

        @m.Constraint(m.fcr_blocks)
        def step_sym(mm, b): return mm.cap_sym[b] == step * mm.k_sym[b]
        @m.Constraint(m.fcr_blocks)
        def step_up(mm, b):  return mm.cap_up[b]  == step * mm.k_up[b]
        @m.Constraint(m.fcr_blocks)
        def step_dn(mm, b):  return mm.cap_dn[b]  == step * mm.k_dn[b]

        @m.Constraint(m.fcr_blocks)
        def min_sym(mm, b): return mm.cap_sym[b] >= min_bid * mm.bid_sym[b]
        @m.Constraint(m.fcr_blocks)
        def min_up(mm, b):  return mm.cap_up[b]  >= min_bid * mm.bid_up[b]
        @m.Constraint(m.fcr_blocks)
        def min_dn(mm, b):  return mm.cap_dn[b]  >= min_bid * mm.bid_dn[b]

        @m.Constraint(m.fcr_blocks)
        def act_sym(mm, b): return mm.k_sym[b] <= M_blocks * mm.bid_sym[b]
        @m.Constraint(m.fcr_blocks)
        def act_up(mm, b):  return mm.k_up[b]  <= M_blocks * mm.bid_up[b]
        @m.Constraint(m.fcr_blocks)
        def act_dn(mm, b):  return mm.k_dn[b]  <= M_blocks * mm.bid_dn[b]

        # mutual exclusivity
        @m.Constraint(m.fcr_blocks)
        def one_product(mm, b): return mm.bid_sym[b] + mm.bid_up[b] + mm.bid_dn[b] <= 1
        @m.Constraint(m.fcr_blocks)
        def afrr_one_side(mm, b): return mm.bid_up[b] + mm.bid_dn[b] <= 1

        # disable by flags
        if not enable_fcr:
            @m.Constraint(m.fcr_blocks)
            def no_fcr(mm, b): return mm.bid_sym[b] == 0
        if not enable_afrr:
            @m.Constraint(m.fcr_blocks)
            def no_afrr_up(mm, b): return mm.bid_up[b] == 0
            @m.Constraint(m.fcr_blocks)
            def no_afrr_dn(mm, b): return mm.bid_dn[b] == 0

        # eligibility via threshold guardrail
        @m.Constraint(m.fcr_blocks)
        def eligible_sym(mm, b): return mm.cap_sym[b] <= max_cap * m.block_eligible[b]
        @m.Constraint(m.fcr_blocks)
        def eligible_up(mm, b):  return mm.cap_up[b]  <= max_cap * m.block_eligible[b]
        @m.Constraint(m.fcr_blocks)
        def eligible_dn(mm, b):  return mm.cap_dn[b]  <= max_cap * m.block_eligible[b]

        # feasibility inside each block hour (headroom/footroom)
        m.fcr_up_feas    = pyo.ConstraintList()
        m.fcr_down_feas  = pyo.ConstraintList()
        m.fcr_sym_feas   = pyo.ConstraintList()

        for b in starts:
            for k in range(L):
                t = b + k
                m.fcr_up_feas.add(   m.cap_up[b]  <= (max_cap - m.total_power_input[t]) )
                m.fcr_down_feas.add( m.cap_dn[b]  <= (m.total_power_input[t] - min_cap) )
                m.fcr_sym_feas.add(  m.cap_sym[b] <= (max_cap - m.total_power_input[t]) )
                m.fcr_sym_feas.add(  m.cap_sym[b] <= (m.total_power_input[t] - min_cap) )

        # ---------------- revenue expression ----------------
        @m.Expression()
        def reserve_revenue(mm):
            return (
                sum(mm.fcr_block_price[b]       * mm.cap_sym[b] for b in mm.fcr_blocks)
                + sum(mm.afrr_block_price_pos[b] * mm.cap_up[b]  for b in mm.fcr_blocks)
                + sum(mm.afrr_block_price_neg[b] * mm.cap_dn[b]  for b in mm.fcr_blocks)
            )

        # ---------------- hourly variable cost (fallback Expression) ----------------
        if not hasattr(m, "variable_cost"):
            def _pE(mm, t):
                return mm.electricity_price[t] if hasattr(mm, "electricity_price") else el_price_hour[int(t)]

            def _vc_rule(mm, t):
                cost = 0.0
                if hasattr(mm, "grid_power"):
                    cost += _pE(mm, t) * mm.grid_power[t]
                if hasattr(mm, "dsm_blocks"):
                    B = mm.dsm_blocks
                    for name in ("calciner", "kiln", "electrolyser", "cement_mill", "thermal_storage"):
                        if name in B and hasattr(B[name], "operating_cost"):
                            cost += B[name].operating_cost[t]
                return cost

            m.variable_cost = pyo.Expression(m.time_steps, rule=_vc_rule)

        # Optional: block-sum variable cost (diagnostics)
        def _blk_cost(mm, b):
            return sum(mm.variable_cost[t] for t in mm.time_steps if (t >= b and t < b + L))
        m.block_var_cost = pyo.Expression(m.fcr_blocks, rule=_blk_cost)


    def _reserve_series(self, instance, T, block_len):
        """
        Build a compact, strategy-agnostic data bundle for reserve analytics.

        Returns a dict with:
        blocks               : list[int] block starts (model indices)
        cap_sym/cap_up/cap_dn: per-block capacities [MW] aligned to `blocks`
        price_sym/pos/neg    : per-block prices [€/MW·block] aligned to `blocks`
        chosen               : dict {block_start -> "FCR" | "UP" | "DN" | "NONE"}
        sym_stairs/up_stairs/dn_stairs : hourly capacity stairs (raw, not masked)
        stairs_price_sym/pos/neg       : hourly price stairs (or None)
        mask_sym/mask_up/mask_dn       : hourly stairs masked by awarded product only
        """

        # ---------- helpers ----------
        def s(v):
            try:
                return float(pyo.value(v))
            except Exception:
                return float(v) if v is not None else 0.0

        def pull_map(pyobj, keys):
            """Read a Pyomo Var/Param into a {key -> float} dict for given keys."""
            if pyobj is None:
                return {}
            d = {}
            for k in keys:
                d[k] = s(pyobj[k])
            return d

        def stairs(blocks, block_map, L, H):
            """Expand per-block constant value into an hourly stair vector of length H."""
            if not block_map:
                return [0.0] * H
            y = [0.0] * H
            for b in blocks:
                val = float(block_map.get(b, 0.0))
                for k in range(L):
                    t = b + k
                    if 0 <= t < H:
                        y[t] = val
            return y

        # ---------- inputs from model / instance ----------
        H = len(T)
        # Block index set
        blocks = list(getattr(instance, "fcr_blocks", []))
        if not blocks:
            # Nothing to show
            zero = [0.0] * H
            return {
                "blocks": [],
                "cap_sym": [],
                "cap_up": [],
                "cap_dn": [],
                "price_sym": [],
                "price_pos": [],
                "price_neg": [],
                "chosen": {},
                "sym_stairs": zero,
                "up_stairs": zero,
                "dn_stairs": zero,
                "stairs_price_sym": None,
                "stairs_price_pos": None,
                "stairs_price_neg": None,
                "mask_sym": zero,
                "mask_up": zero,
                "mask_dn": zero,
            }

        # Per-block capacities (your model uses cap_up, cap_dn; symmetric product is represented by cap_up==cap_dn)
        cap_up_map = pull_map(getattr(instance, "cap_up", None), blocks)
        cap_dn_map = pull_map(getattr(instance, "cap_dn", None), blocks)
        # For convenience, define "sym" per block as min(up, dn) — that’s the feasible symmetric MW
        # If you have a dedicated symmetric var, you could switch to that; this is robust.
        cap_sym_map = {
            b: min(cap_up_map.get(b, 0.0), cap_dn_map.get(b, 0.0)) for b in blocks
        }

        # Per-block prices (€/MW·block = sum of hourly €/MW/h over the block)
        price_sym_map = pull_map(
            getattr(instance, "fcr_block_price", None), blocks
        )  # FCR
        price_pos_map = pull_map(
            getattr(instance, "afrr_block_price_pos", None), blocks
        )  # aFRR+
        price_neg_map = pull_map(
            getattr(instance, "afrr_block_price_neg", None), blocks
        )  # aFRR−

        # Optional: bid-award binaries for clarity of "chosen" product if present
        bid_sym_v = getattr(instance, "bid_sym", None)
        bid_up_v = getattr(instance, "bid_up", None)
        bid_dn_v = getattr(instance, "bid_dn", None)

        # ---------- chosen product per block ----------
        chosen = {}
        for b in blocks:
            # Prefer explicit binaries if available
            y_sym = int(round(s(bid_sym_v[b]))) if bid_sym_v is not None else None
            y_up = int(round(s(bid_up_v[b]))) if bid_up_v is not None else None
            y_dn = int(round(s(bid_dn_v[b]))) if bid_dn_v is not None else None

            if y_sym is not None or y_up is not None or y_dn is not None:
                if y_sym == 1:
                    chosen[b] = "FCR"
                elif y_up == 1:
                    chosen[b] = "UP"
                elif y_dn == 1:
                    chosen[b] = "DN"
                else:
                    chosen[b] = "NONE"
            else:
                # Fallback: infer from capacities/prices (award only one side if multiple > 0)
                cu, cd, cs = (
                    cap_up_map.get(b, 0.0),
                    cap_dn_map.get(b, 0.0),
                    cap_sym_map.get(b, 0.0),
                )
                pu, pn, ps = (
                    price_pos_map.get(b, 0.0),
                    price_neg_map.get(b, 0.0),
                    price_sym_map.get(b, 0.0),
                )

                # Use the revenue contributions to decide the single awarded product
                rev_candidates = [
                    ("FCR", cs * ps),
                    ("UP", cu * pu),
                    ("DN", cd * pn),
                ]
                # Pick the product with largest revenue; break ties toward NONE if all zero
                best = max(rev_candidates, key=lambda x: x[1])
                chosen[b] = best[0] if best[1] > 1e-9 else "NONE"

        # ---------- per-block arrays (aligned with `blocks`) ----------
        cap_sym = [cap_sym_map.get(b, 0.0) for b in blocks]
        cap_up = [cap_up_map.get(b, 0.0) for b in blocks]
        cap_dn = [cap_dn_map.get(b, 0.0) for b in blocks]

        price_sym = [price_sym_map.get(b, 0.0) for b in blocks]
        price_pos = [price_pos_map.get(b, 0.0) for b in blocks]
        price_neg = [price_neg_map.get(b, 0.0) for b in blocks]

        # ---------- hourly stairs (raw totals, not masked) ----------
        sym_stairs = stairs(blocks, cap_sym_map, block_len, H)
        up_stairs = stairs(blocks, cap_up_map, block_len, H)
        dn_stairs = stairs(blocks, cap_dn_map, block_len, H)

        # Hourly price stairs (these are the same price repeated over each hour inside the block)
        stairs_price_sym = (
            stairs(blocks, price_sym_map, block_len, H)
            if any(abs(v) > 1e-12 for v in price_sym)
            else None
        )
        stairs_price_pos = (
            stairs(blocks, price_pos_map, block_len, H)
            if any(abs(v) > 1e-12 for v in price_pos)
            else None
        )
        stairs_price_neg = (
            stairs(blocks, price_neg_map, block_len, H)
            if any(abs(v) > 1e-12 for v in price_neg)
            else None
        )

        # ---------- hourly masks (only the awarded product per hour is kept) ----------
        mask_sym = [0.0] * H
        mask_up = [0.0] * H
        mask_dn = [0.0] * H
        for b in blocks:
            lab = chosen.get(b, "NONE")
            cS = cap_sym_map.get(b, 0.0)
            cU = cap_up_map.get(b, 0.0)
            cD = cap_dn_map.get(b, 0.0)
            for k in range(block_len):
                t = b + k
                if 0 <= t < H:
                    if lab == "FCR":
                        mask_sym[t] = cS
                    elif lab == "UP":
                        mask_up[t] = cU
                    elif lab == "DN":
                        mask_dn[t] = cD
                    # "NONE": leave zeros

        return {
            "blocks": blocks,
            "cap_sym": cap_sym,
            "cap_up": cap_up,
            "cap_dn": cap_dn,
            "price_sym": price_sym,
            "price_pos": price_pos,
            "price_neg": price_neg,
            "chosen": chosen,
            "sym_stairs": sym_stairs,
            "up_stairs": up_stairs,
            "dn_stairs": dn_stairs,
            "stairs_price_sym": stairs_price_sym,
            "stairs_price_pos": stairs_price_pos,
            "stairs_price_neg": stairs_price_neg,
            "mask_sym": mask_sym,
            "mask_up": mask_up,
            "mask_dn": mask_dn,
        }
        
    def compute_reg_capacity_and_price(
        self,
        instance,
        *,
        eta_e2h_default=1.0,
        eta_fossil_default=0.90,
        tes_eta_rt=None,
        tes_value_mode="price",
        lookahead_hours=24,
        include_co2=True,
        co2_price_eur_per_t=None,
        activation_duration_h: float = 1.0,   # <-- add this
        **kwargs,                              # <-- and this for forwards-compat
    ):
        """
        Returns per-hour dict with:
        - max_up_MW, max_down_MW
        - price_up_EUR_per_MWhe, price_down_EUR_per_MWhe
        - best_up_lever, best_down_lever

        Notes:
        • activation_duration_h is currently not used inside this function
            (prices are marginal €/MWhₑ for the first MW). Keep it in the
            signature so callers can pass it; convert to €/MW for bids
            outside using: price_[€/MWh] * activation_duration_h [h].
        """

        import math
        import pyomo.environ as pyo
        import numpy as np

        # ---- helpers ----
        def safe(v):
            try:
                return float(pyo.value(v))
            except Exception:
                try:
                    return float(v)
                except Exception:
                    return 0.0

        def _as_hourly(x, H, default=0.0):
            """Coerce scalar / sequence / FastSeries-like into list[float] length H."""
            if x is None:
                return [default] * H
            # scalar?
            try:
                val = float(x)
                return [val] * H
            except Exception:
                pass
            # sequence-like
            try:
                seq = list(x)
            except Exception:
                try:
                    return [float(x[i]) for i in range(H)]
                except Exception:
                    return [default] * H
            if len(seq) < H:
                seq = seq + [seq[-1]] * (H - len(seq))
            elif len(seq) > H:
                seq = seq[:H]
            out = []
            for v in seq:
                try:
                    out.append(float(v))
                except Exception:
                    out.append(default)
            return out

        def ser(block, name, T):
            if block is None or not hasattr(block, name):
                return None
            return [safe(getattr(block, name)[t]) for t in T]

        # ---- time base first (so H is known) ----
        T = list(instance.time_steps)
        H = len(T)

        # ---- blocks ----
        B  = instance.dsm_blocks
        ph = B["preheater"]      if "preheater"      in B else None
        cc = B["calciner"]       if "calciner"       in B else None
        rk = B["kiln"]           if "kiln"           in B else None
        el = B["electrolyser"]   if "electrolyser"   in B else None
        ts = B["thermal_storage"]if "thermal_storage" in B else None
        cm = B["cement_mill"]    if "cement_mill"    in B else None

        # ---- prices (hourly vectors) ----
        pE  = _as_hourly([safe(instance.electricity_price[t]) for t in T] if hasattr(instance, "electricity_price") else 0.0, H, default=0.0)
        pNG = _as_hourly([safe(instance.natural_gas_price[t]) for t in T] if hasattr(instance, "natural_gas_price") else 0.0, H, default=0.0)
        pCO = _as_hourly([safe(instance.coal_price[t])        for t in T] if hasattr(instance, "coal_price")        else 0.0, H, default=0.0)
        pH2 = _as_hourly([safe(instance.hydrogen_price[t])    for t in T] if hasattr(instance, "hydrogen_price")    else None, H, default=0.0)

        # CO2 price [€/tCO2] hourly
        pCO2 = _as_hourly(co2_price_eur_per_t, H, default=0.0) if include_co2 else [0.0]*H

        # ---- emission factors [tCO2/MWh_th] for fuels (adjust if your model differs) ----
        EF_NG   = 0.202
        EF_COAL = 0.341
        EF_H2   = 0.0   # direct (scope-1) zero

        # ---- efficiencies ----
        eta_fossil_cc = safe(getattr(cc, "eta_fossil", eta_fossil_default)) if cc else eta_fossil_default
        eta_fossil_rk = safe(getattr(rk, "eta_fossil", eta_fossil_default)) if rk else eta_fossil_default
        eta_e2h_cc    = safe(getattr(cc, "eta_electric_to_heat", eta_e2h_default)) if cc else eta_e2h_default
        eta_e2h_rk    = safe(getattr(rk, "eta_electric_to_heat", eta_e2h_default)) if rk else eta_e2h_default
        eta_el        = safe(getattr(el, "efficiency", 0.65)) if el else 0.65  # MWh_H2 per MWh_e

        # TES round-trip (if needed for look-ahead)
        if tes_eta_rt is None:
            eta_c = safe(getattr(ts, "eta_charge", 0.95)) if ts else 0.95
            eta_d = safe(getattr(ts, "eta_discharge", 0.95)) if ts else 0.95
            tes_eta_rt = eta_c * eta_d

        # ---- series & caps ----
        P_cc = ser(cc, "power_in", T); P_rk = ser(rk, "power_in", T)
        P_el = ser(el, "power_in", T); P_ts = ser(ts, "power_in", T)
        P_cm = ser(cm, "power_in", T)

        def cap(block, name):
            return safe(getattr(block, name)) if (block is not None and hasattr(block, name)) else 0.0

        Pcc_max  = cap(cc, "max_power")
        Prk_max  = cap(rk, "max_power")
        Pel_max  = cap(el, "max_power")
        Pts_max  = cap(ts, "max_Pelec")
        Pcm_max  = cap(cm, "max_power")

        # Fossil price seen by each process, incl. CO2
        def fossil_price_for_calciner(i):
            co = ser(cc, "coal_in", T) if cc else None
            ng = ser(cc, "natural_gas_in", T) if cc else None
            if co and co[i] > 1e-9:
                return pCO[i] + pCO2[i] * EF_COAL
            if ng and ng[i] > 1e-9:
                return pNG[i] + pCO2[i] * EF_NG
            # fallback (BAU often coal)
            if any(pCO): return pCO[i] + pCO2[i] * EF_COAL
            return pNG[i] + pCO2[i] * EF_NG

        def fossil_price_for_kiln(i):
            co = ser(rk, "coal_in", T) if rk else None
            ng = ser(rk, "natural_gas_in", T) if rk else None
            h2 = ser(rk, "hydrogen_in", T) if rk else None
            if h2 and h2[i] > 1e-9:
                return pH2[i] + pCO2[i] * EF_H2
            if co and co[i] > 1e-9:
                return pCO[i] + pCO2[i] * EF_COAL
            if ng and ng[i] > 1e-9:
                return pNG[i] + pCO2[i] * EF_NG
            # fallback priority: coal → NG → H2
            if any(pCO): return pCO[i] + pCO2[i] * EF_COAL
            if any(pNG): return pNG[i] + pCO2[i] * EF_NG
            return pH2[i] + pCO2[i] * EF_H2

        # Optional TES “value of charge” via look-ahead
        tes_val = [0.0]*H
        if ts and P_ts and any(p > 1e-9 for p in P_ts) and tes_value_mode == "lookahead":
            for i in range(H):
                j2 = min(H, i + 1 + int(lookahead_hours))
                fut = pE[i+1:j2] if j2 > i+1 else []
                tes_val[i] = max(tes_eta_rt * (max(fut) if fut else pE[i]) - pE[i], 0.0)

        # ---- build stacks per hour ----
        max_up   = [0.0]*H
        max_down = [0.0]*H
        price_up = [math.nan]*H
        price_dn = [math.nan]*H
        best_up  = [None]*H
        best_dn  = [None]*H

        for i in range(H):
            levers_up = []   # (name, room_MW, c_up)
            levers_dn = []   # (name, room_MW, c_dn)

            # TES charger (electrical heater mode)
            if ts and P_ts is not None and Pts_max > 0.0:
                up_room = max(0.0, Pts_max - P_ts[i])
                dn_room = max(0.0, P_ts[i])
                c_up = pE[i]
                c_dn = (tes_val[i] if tes_value_mode == "lookahead" else pE[i])
                if up_room > 1e-6: levers_up.append(("TES_charge", up_room, c_up))
                if dn_room > 1e-6: levers_dn.append(("TES_charge", dn_room, c_dn))

            # Electrolyser
            if el and P_el is not None and Pel_max > 0.0:
                up_room = max(0.0, Pel_max - P_el[i])
                dn_room = max(0.0, P_el[i])
                # value by H2 if available
                valH2 = pH2[i] if pH2 is not None else 0.0
                c_up = pE[i] - valH2 * eta_el
                c_dn = valH2 * eta_el - pE[i]
                if up_room > 1e-6: levers_up.append(("Electrolyser", up_room, c_up))
                if dn_room > 1e-6: levers_dn.append(("Electrolyser", dn_room, c_dn))

            # E-Calciner (with fossil backfill)
            if cc and P_cc is not None and Pcc_max > 0.0:
                up_room = max(0.0, Pcc_max - P_cc[i])
                dn_room = max(0.0, P_cc[i])
                pF = fossil_price_for_calciner(i)  # €/MWh_th including CO2
                c_e2h_th = eta_e2h_cc
                c_up = pE[i] - c_e2h_th * (pF / max(eta_fossil_cc, 1e-9))
                c_dn = c_e2h_th * (pF / max(eta_fossil_cc, 1e-9)) - pE[i]
                if up_room > 1e-6: levers_up.append(("E-Calciner", up_room, c_up))
                if dn_room > 1e-6: levers_dn.append(("E-Calciner", dn_room, c_dn))

            # E-Kiln (with fossil/H2 backfill)
            if rk and P_rk is not None and Prk_max > 0.0:
                up_room = max(0.0, Prk_max - P_rk[i])
                dn_room = max(0.0, P_rk[i])
                pF = fossil_price_for_kiln(i)
                c_e2h_th = eta_e2h_rk
                c_up = pE[i] - c_e2h_th * (pF / max(eta_fossil_rk, 1e-9))
                c_dn = c_e2h_th * (pF / max(eta_fossil_rk, 1e-9)) - pE[i]
                if up_room > 1e-6: levers_up.append(("E-Kiln", up_room, c_up))
                if dn_room > 1e-6: levers_dn.append(("E-Kiln", dn_room, c_dn))

            # Cement mill auxiliaries (down only, usually disfavored)
            if cm and P_cm is not None:
                up_room = max(0.0, (Pcm_max - P_cm[i]) if Pcm_max > 0 else 0.0)
                dn_room = max(0.0, P_cm[i])
                # Make down extremely expensive if you wish to avoid impacting output:
                VOLL_clk = 1e6
                if up_room > 1e-6: levers_up.append(("CementMill", up_room, pE[i]))
                # Comment-out next line if you NEVER want to reduce mill load:
                # if dn_room > 1e-6: levers_dn.append(("CementMill", dn_room, VOLL_clk))

            # aggregate capacities
            max_up[i]   = sum(r for _, r, _ in levers_up) if levers_up else 0.0
            max_down[i] = sum(r for _, r, _ in levers_dn) if levers_dn else 0.0

            # marginal (first MW) = cheapest lever cost
            if levers_up:
                name, room, cost = min(levers_up, key=lambda x: x[2])
                price_up[i] = float(max(cost, 0.0))
                best_up[i]  = name
            if levers_dn:
                name, room, cost = min(levers_dn, key=lambda x: x[2])
                price_dn[i] = float(max(cost, 0.0))
                best_dn[i]  = name
                
            if max_down[i] <= 1e-9:
                price_dn[i] = 0.0
                best_dn[i]  = "—"
            if max_up[i] <= 1e-9:
                price_up[i] = 0.0
                best_up[i]  = "—"

        return {
            "time": T,
            "max_up_MW": max_up,
            "max_down_MW": max_down,
            "price_up_EUR_per_MWhe": price_up,
            "price_down_EUR_per_MWhe": price_dn,
            "best_up_lever": best_up,
            "best_down_lever": best_dn,
        }

    def build_redispatch_bids_from_instance(
        self,
        instance,
        *,
        activation_duration_h: float = 1.0,
        include_co2: bool = True,
        co2_price_eur_per_t: float | None = None,
        csv_path: str = "./outputs/redispatch_bids.csv",
    ):
        """
        Uses the solved `instance` to:
        • read baseline (total_power_input, clinker rate, variable cost, prices),
        • call compute_reg_capacity_and_price(...) for hourly feasible up/down + prices,
        • write a single CSV (TSO-ready).
        """
        import pyomo.environ as pyo
        import pandas as pd
        import numpy as np

        def safe(v):
            try:
                return float(pyo.value(v))
            except Exception:
                try:
                    return float(v)
                except Exception:
                    return 0.0

        T = list(instance.time_steps)

        # Baseline series
        base_elec = [safe(instance.total_power_input[t]) for t in T] if hasattr(instance, "total_power_input") else [0.0]*len(T)

        kiln = instance.dsm_blocks["kiln"] if ("kiln" in instance.dsm_blocks) else None
        if kiln is not None and hasattr(kiln, "clinker_out"):
            clk = [safe(kiln.clinker_out[t]) for t in T]
        elif hasattr(instance, "clinker_rate"):
            clk = [safe(instance.clinker_rate[t]) for t in T]
        else:
            clk = [0.0]*len(T)

        pE = [safe(instance.electricity_price[t]) for t in T] if hasattr(instance, "electricity_price") else None
        vc = [safe(instance.variable_cost[t])    for t in T] if hasattr(instance, "variable_cost")    else None

        # Compute redispatch feasible capacities & activation prices
        out = self.compute_reg_capacity_and_price(
            instance,
            activation_duration_h=activation_duration_h,
            tes_value_mode="price",                 # conservative for redispatch
            include_co2=include_co2,
            co2_price_eur_per_t=co2_price_eur_per_t,
        )

        df = pd.DataFrame({
            "t": T,
            "Baseline Elec Load [MW_e]": base_elec,
            "Baseline Clinker [t/h]":    clk,
            "Elec Price [€/MWh_e]":      pE if pE else np.nan,
            "Variable Cost [€/h]":       vc if vc else np.nan,
            "Max Up [MW_e]":             out["max_up_MW"],
            "Max Down [MW_e]":           out["max_down_MW"],
            "Price Up [€/MWh_e]":        out["price_up_EUR_per_MWhe"],
            "Price Down [€/MWh_e]":      out["price_down_EUR_per_MWhe"],
            "Best Up Lever":             out["best_up_lever"],
            "Best Down Lever":           out["best_down_lever"],
        })

        if csv_path:
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        self.redispatch_last = {"instance": instance, "table": df, "raw": out}
        return out, df
