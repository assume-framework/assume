# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime

import pyomo.environ as pyo

from assume.common.base import SupportsMinMax
from assume.common.forecasts import Forecaster
from assume.units.dsm_load_shift import DSMFlex

import matplotlib as mpl
import pandas as pd
import pyomo.environ as pyo
from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as ps

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

    ###################Dashboard functions####################

    def render_sankey_timeseries(self, instance, html_path="./outputs/sankey_timeseries.html",
                                step_stride=1, # use >1 to downsample frames if you have many timesteps
                                min_flow=1e-6  # hide tiny links for readability
                                ):
        def safe_value(v):
            try: return float(pyo.value(v))
            except Exception: return 0.0

        # ---- blocks (optional)
        B = instance.dsm_blocks
        ph = B["preheater"]     if "preheater"     in B else None
        cc = B["calciner"]      if "calciner"      in B else None
        rk = B["kiln"]          if "kiln"          in B else None
        el = B["electrolyser"]  if "electrolyser"  in B else None
        ts = B["thermal_storage"] if "thermal_storage" in B else None

        T = list(instance.time_steps)
        T = T[::step_stride]  # optional downsample

        def s_or_none(block, name):
            if block is None or not hasattr(block, name): return None
            return [safe_value(getattr(block, name)[t]) for t in T]

        # series (per-timestep)
        # electricity inputs (MWe)
        phP = s_or_none(ph, "power_in")
        ccP = s_or_none(cc, "power_in")
        rkP = s_or_none(rk, "power_in")
        tsP = s_or_none(ts, "power_in")  # TES heater (E→H conversion internal to TES)

        # fuels (MWth)
        phNG, phCoal = s_or_none(ph, "natural_gas_in"), s_or_none(ph, "coal_in")
        ccNG, ccCoal, ccH2 = s_or_none(cc, "natural_gas_in"), s_or_none(cc, "coal_in"), s_or_none(cc, "hydrogen_in")
        rkNG, rkCoal, rkH2 = s_or_none(rk, "natural_gas_in"), s_or_none(rk, "coal_in"), s_or_none(rk, "hydrogen_in")

        # unit outputs / couplings (MWth)
        phOut   = s_or_none(ph, "heat_out")
        phWH    = s_or_none(ph, "external_heat_in")   # WHR → PH
        ccOut   = s_or_none(cc, "heat_out")
        ccEffIn = s_or_none(cc, "effective_heat_in")  # = burner heat + TES discharge (if present)
        rkOut   = s_or_none(rk, "heat_out")

        tsChg, tsDis, tsSOC = s_or_none(ts, "charge"), s_or_none(ts, "discharge"), s_or_none(ts, "soc")

        # ---- node list (fixed order; show node even if zero in some frames)
        nodes = [
            "Electricity", "Natural Gas", "Coal", "Hydrogen", "WHR", "TES (discharge)",
            "Preheater", "Calciner", "Kiln", "Clinker"
        ]
        n = {name: i for i, name in enumerate(nodes)}

        # helper to build links for one timestep index k
        def links_at(k):
            src, tgt, val, lbl = [], [], [], []

            def add(a, b, v, text):
                if v is None: return
                if v <= min_flow: return
                src.append(n[a]); tgt.append(n[b]); val.append(float(v)); lbl.append(text)

            # Electricity to units / TES heater (units as input power)
            add("Electricity", "Preheater", phP[k] if phP else None, "E→PH (MWₑ)")
            add("Electricity", "Calciner",  ccP[k] if ccP else None, "E→CC (MWₑ)")
            add("Electricity", "Kiln",      rkP[k] if rkP else None, "E→Kiln (MWₑ)")
            add("Electricity", "TES (discharge)", tsP[k] if tsP else None, "E→TES heater (MWₑ)")

            # Fuels to units (thermal)
            add("Natural Gas", "Preheater", phNG[k]  if phNG  else None, "NG→PH (MWₜₕ)")
            add("Natural Gas", "Calciner",  ccNG[k]  if ccNG  else None, "NG→CC (MWₜₕ)")
            add("Natural Gas", "Kiln",      rkNG[k]  if rkNG  else None, "NG→Kiln (MWₜₕ)")

            add("Coal", "Preheater", phCoal[k] if phCoal else None, "Coal→PH (MWₜₕ)")
            add("Coal", "Calciner",  ccCoal[k] if ccCoal else None, "Coal→CC (MWₜₕ)")
            add("Coal", "Kiln",      rkCoal[k] if rkCoal else None, "Coal→Kiln (MWₜₕ)")

            add("Hydrogen", "Calciner", ccH2[k] if ccH2 else None, "H₂→CC (MWₜₕ)")
            add("Hydrogen", "Kiln",     rkH2[k] if rkH2 else None, "H₂→Kiln (MWₜₕ)")

            # WHR → PH; TES → CC
            add("WHR", "Preheater", phWH[k] if phWH else None, "WHR→PH (MWₜₕ)")
            add("TES (discharge)", "Calciner", tsDis[k] if tsDis else None, "TES→CC (MWₜₕ)")

            # Internal plant links
            add("Preheater", "Calciner", phOut[k] if phOut else None, "PH→CC (MWₜₕ)")
            add("Calciner", "Kiln", ccEffIn[k] if ccEffIn else (ccOut[k] if ccOut else None), "CC→Kiln (MWₜₕ)")
            add("Kiln", "Clinker", rkOut[k] if rkOut else None, "Kiln→Clinker (MWₜₕ)")

            return src, tgt, val, lbl

        # ---- build frames
        frames = []
        for i, t in enumerate(T):
            src, tgt, val, lbl = links_at(i)
            frame = go.Frame(
                data=[go.Sankey(
                    node=dict(label=nodes, pad=12, thickness=18),
                    link=dict(source=src, target=tgt, value=val, label=lbl, hovertemplate="%{label}<br>Flow: %{value:.2f}<extra></extra>")
                )],
                name=f"{t}"
            )
            frames.append(frame)

        # initial (first timestep)
        init_src, init_tgt, init_val, init_lbl = links_at(0)
        fig = go.Figure(
            data=[go.Sankey(
                node=dict(label=nodes, pad=12, thickness=18),
                link=dict(source=init_src, target=init_tgt, value=init_val, label=init_lbl,
                        hovertemplate="%{label}<br>Flow: %{value:.2f}<extra></extra>")
            )],
            frames=frames
        )

        # slider & buttons
        steps = [dict(method="animate",
                    args=[[f.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                    label=f"t={f.name}") for f in frames]

        fig.update_layout(
            title="Time‑dependent Sankey (per time step)",
            font_size=12,
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play", method="animate",
                        args=[None, {"fromcurrent": True, "frame": {"duration": 400, "redraw": True}}]),
                    dict(label="Pause", method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
                ],
                x=0.02, y=1.12, xanchor="left", yanchor="top"
            )],
            sliders=[dict(active=0, y= -0.06, x=0.05, len=0.9, steps=steps)]
        )

        fig.write_html(html_path)
        print(f"Sankey timeseries saved to: {html_path}")


    ##################################################################################

    def render_emissions_analytics(
        self,
        instance,
        baseline_instance=None,                     # optional: pass a "no-TES" or BAU instance for comparison
        html_path="./outputs/emissions_analytics.html",
        elec_ef_param_name="electricity_emission_factor"  # tCO2/MWh_e on the model (optional)
    ):
        # ---------- helpers ----------
        def safe_val(v):
            try: return float(pyo.value(v))
            except Exception: return 0.0

        def series_or_none(block, name):
            if block is None or not hasattr(block, name): return None
            return [safe_val(getattr(block, name)[t]) for t in instance.time_steps]

        def get_block(name):
            return instance.dsm_blocks[name] if name in instance.dsm_blocks else None

        T = list(instance.time_steps)
        dt = 1.0  # hours per step (adapt if different)

        # ---------- blocks ----------
        ph = get_block("preheater")
        cc = get_block("calciner")
        rk = get_block("kiln")

        # ---------- fuel inputs (MW_th) & power (MW_e) ----------
        ph_ng, ph_coal, ph_pwr = series_or_none(ph, "natural_gas_in"), series_or_none(ph, "coal_in"), series_or_none(ph, "power_in")
        cc_ng, cc_coal, cc_pwr, cc_h2 = series_or_none(cc, "natural_gas_in"), series_or_none(cc, "coal_in"), series_or_none(cc, "power_in"), series_or_none(cc, "hydrogen_in")
        rk_ng, rk_coal, rk_pwr, rk_h2 = series_or_none(rk, "natural_gas_in"), series_or_none(rk, "coal_in"), series_or_none(rk, "power_in"), series_or_none(rk, "hydrogen_in")

        clinker_rate = [safe_val(instance.clinker_rate[t]) for t in T] if hasattr(instance, "clinker_rate") else None

        # ---------- emission factors ----------
        # From blocks if available (your Calciner exposes these; we’ll fall back to typical values otherwise)
        ng_ef    = safe_val(getattr(cc, "ng_co2_factor", 0.202))  if cc else 0.202   # tCO2/MWh_th NG
        coal_ef  = safe_val(getattr(cc, "coal_co2_factor", 0.341)) if cc else 0.341  # tCO2/MWh_th Coal
        # Electricity EF (scope 2): optional param on model, default 0
        grid_ef  = safe_val(getattr(instance, elec_ef_param_name, 0.0)) if hasattr(instance, elec_ef_param_name) else 0.0
        # H2 EF (direct) assumed 0; set here if you want non-zero
        h2_ef    = 0.0

        # Process CO2 from calcination
        cc_proc = series_or_none(cc, "co2_proc")  # [tCO2/h]
        # If not available: process EF × clinker
        if cc_proc is None and clinker_rate is not None:
            cf_proc = safe_val(getattr(cc, "cf_proc", 0.525)) if cc else 0.525  # tCO2/t_clinker
            cc_proc = [cf_proc * c for c in clinker_rate]
        elif cc_proc is None:
            cc_proc = [0.0 for _ in T]

        # ---------- energy CO2 time series (tCO2/h) ----------
        def mul(series, ef):
            return [ (series[i] if series else 0.0) * ef for i in range(len(T)) ]

        ph_e = mul(ph_ng, ng_ef)
        ph_c = mul(ph_coal, coal_ef)
        ph_el= mul(ph_pwr, grid_ef)    # scope 2

        cc_e = mul(cc_ng, ng_ef)
        cc_c = mul(cc_coal, coal_ef)
        cc_el= mul(cc_pwr, grid_ef)
        cc_h = mul(cc_h2, h2_ef)

        rk_e = mul(rk_ng, ng_ef)
        rk_c = mul(rk_coal, coal_ef)
        rk_el= mul(rk_pwr, grid_ef)
        rk_h = mul(rk_h2, h2_ef)

        # ---------- totals per hour ----------
        proc_ts   = cc_proc
        energy_ts = [ph_e[i]+ph_c[i]+ph_el[i] + cc_e[i]+cc_c[i]+cc_el[i]+cc_h[i] + rk_e[i]+rk_c[i]+rk_el[i]+rk_h[i] for i in range(len(T))]
        total_ts  = [proc_ts[i] + energy_ts[i] for i in range(len(T))]

        # ---------- CO2 price & carbon cost ----------
        co2_price = [safe_val(instance.co2_price[t]) for t in T] if hasattr(instance, "co2_price") else [0.0]*len(T)
        carbon_cost_ts = [ total_ts[i] * co2_price[i] for i in range(len(T)) ]  # €/h

        # ---------- KPIs ----------
        E_proc   = float(np.sum(proc_ts)   * dt)  # tCO2
        E_energy = float(np.sum(energy_ts) * dt)  # tCO2
        E_total  = E_proc + E_energy
        tot_carbon_cost = float(np.sum(carbon_cost_ts) * dt)

        if clinker_rate is not None:
            clinker_total = float(np.sum(clinker_rate) * dt)
            intensity_t = E_total / clinker_total if clinker_total > 1e-9 else np.nan
            intensity_e = E_energy / clinker_total if clinker_total > 1e-9 else np.nan
            intensity_p = E_proc  / clinker_total if clinker_total > 1e-9 else np.nan
        else:
            clinker_total, intensity_t, intensity_e, intensity_p = (np.nan,)*4

        # ---------- charts ----------
        figs = []

        # KPI table
        krows = [
            ("Process CO₂ [t]", f"{E_proc:,.1f}"),
            ("Energy CO₂ [t]",  f"{E_energy:,.1f}"),
            ("Total CO₂ [t]",   f"{E_total:,.1f}"),
            ("Carbon cost [€]", f"{tot_carbon_cost:,.0f}"),
            ("Clinker [t]",     f"{clinker_total:,.0f}" if clinker_rate is not None else "—"),
            ("Intensity total [tCO₂/t]",  f"{intensity_t:.3f}" if intensity_t==intensity_t else "—"),
            ("Intensity energy [tCO₂/t]", f"{intensity_e:.3f}" if intensity_e==intensity_e else "—"),
            ("Intensity process [tCO₂/t]",f"{intensity_p:.3f}" if intensity_p==intensity_p else "—"),
            ("Grid EF [tCO₂/MWhₑ]",       f"{grid_ef:.3f}"),
            ("NG EF [tCO₂/MWhₜₕ]",        f"{ng_ef:.3f}"),
            ("Coal EF [tCO₂/MWhₜₕ]",      f"{coal_ef:.3f}"),
        ]
        kfig = go.Figure(data=[go.Table(
            header=dict(values=["KPI","Value"], align="left"),
            cells=dict(values=list(zip(*krows)), align="left")
        )])
        kfig.update_layout(title="Emissions KPIs")
        figs.append(kfig)

        # Stacked area over time (process + energy by fuel & scope)
        area = ps.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                subplot_titles=("CO₂ by source (stacked)", "Carbon cost (€/h)"))
        # Stack components:
        def add_area(y, name):
            area.add_trace(go.Scatter(x=T, y=y, name=name, stackgroup="one", mode="lines"), row=1, col=1)

        add_area(proc_ts, "Process CO₂")
        add_area(ph_e, "PH NG CO₂"); add_area(ph_c, "PH Coal CO₂"); add_area(ph_el, "PH Elec (scope2)")
        add_area(cc_e, "CC NG CO₂"); add_area(cc_c, "CC Coal CO₂"); add_area(cc_el, "CC Elec (scope2)"); add_area(cc_h, "CC H₂ CO₂")
        add_area(rk_e, "Kiln NG CO₂"); add_area(rk_c, "Kiln Coal CO₂"); add_area(rk_el, "Kiln Elec (scope2)"); add_area(rk_h, "Kiln H₂ CO₂")

        area.add_trace(go.Scatter(x=T, y=carbon_cost_ts, name="Carbon cost [€/h]", mode="lines"), row=2, col=1)
        area.update_yaxes(title_text="tCO₂/h", row=1, col=1)
        area.update_yaxes(title_text="€/h", row=2, col=1)
        area.update_layout(title="Time Series: Emissions & Carbon Cost")
        figs.append(area)

        # By-unit bar (sum over horizon)
        unit_totals = {
            "Process (calcination)": E_proc,
            "PH energy": float(np.sum(ph_e)+np.sum(ph_c)+np.sum(ph_el))*dt,
            "CC energy": float(np.sum(cc_e)+np.sum(cc_c)+np.sum(cc_el)+np.sum(cc_h))*dt,
            "Kiln energy": float(np.sum(rk_e)+np.sum(rk_c)+np.sum(rk_el)+np.sum(rk_h))*dt,
        }
        b1 = go.Figure(go.Bar(x=list(unit_totals.keys()), y=list(unit_totals.values())))
        b1.update_layout(title="CO₂ by Unit (sum over horizon)", yaxis_title="tCO₂")
        figs.append(b1)

        # By-fuel bar (energy CO₂ only)
        fuel_totals = {
            "NG":   float((np.sum(ph_e)+np.sum(cc_e)+np.sum(rk_e))*dt),
            "Coal": float((np.sum(ph_c)+np.sum(cc_c)+np.sum(rk_c))*dt),
            "Elec (scope2)": float((np.sum(ph_el)+np.sum(cc_el)+np.sum(rk_el))*dt),
            "H₂":   float((np.sum(cc_h)+np.sum(rk_h))*dt),
        }
        b2 = go.Figure(go.Bar(x=list(fuel_totals.keys()), y=list(fuel_totals.values())))
        b2.update_layout(title="Energy CO₂ by Fuel (sum over horizon)", yaxis_title="tCO₂")
        figs.append(b2)

        # Duration curve of marginal emission intensity (tCO2 per MW_th delivered to hot section)
        # Approximate delivered hot-section power = calciner effective + kiln heat_out
        cc_eff = series_or_none(cc, "effective_heat_in") or series_or_none(cc, "heat_out")
        rk_out= series_or_none(rk, "heat_out")
        delivered = [ (cc_eff[i] if cc_eff else 0.0) + (rk_out[i] if rk_out else 0.0) for i in range(len(T)) ]
        intensity_ts = [ (total_ts[i]/max(delivered[i],1e-9)) if delivered[i]>1e-9 else np.nan for i in range(len(T)) ]
        dur = go.Figure()
        dur.add_trace(go.Scatter(y=sorted([v for v in intensity_ts if v==v], reverse=True), name="Intensity duration"))
        dur.update_layout(title="Duration Curve: Marginal Emission Intensity [tCO₂/MWₜₕ]", yaxis_title="tCO₂/MWₜₕ", xaxis_title="Ranked hours")
        figs.append(dur)

        # TES effectiveness scatter (if TES present): discharge vs reduction in burner CO2 (heuristic)
        ts = instance.dsm_blocks["thermal_storage"] if "thermal_storage" in instance.dsm_blocks else None
        if ts is not None and hasattr(ts, "discharge"):
            ts_dis = [safe_val(ts.discharge[t]) for t in instance.time_steps]
            # heuristic: when TES discharges, assume it offsets the currently dominant burner stream in calciner
            burner_co2 = [cc_e[i]+cc_c[i]+cc_el[i]+cc_h[i] for i in range(len(T))]
            # Use change vs median as crude reference:
            med_burn = float(np.nanmedian(burner_co2)) if len(burner_co2)>0 else 0.0
            avoided = [max(0.0, med_burn - burner_co2[i]) for i in range(len(T))]
            sfig = go.Figure(go.Scatter(x=ts_dis, y=avoided, mode="markers", text=[f"t={t}" for t in T]))
            sfig.update_layout(title="TES Discharge vs. Avoided Burner CO₂ (heuristic)",
                            xaxis_title="TES Discharge [MWₜₕ]", yaxis_title="Avoided CO₂ [t/h]")
            figs.append(sfig)

        # Baseline comparison (ΔCO₂, Δcost, MAC)
        if baseline_instance is not None:
            def totals_for(inst):
                # recurse this function for another instance
                # minimal duplicate logic: only totals
                B = inst.dsm_blocks
                def g(b,n): return [safe_val(getattr(B[b], n)[t]) for t in inst.time_steps] if (b in B and hasattr(B[b], n)) else [0.0]*len(inst.time_steps)
                # fuels
                PHng, PHc, PHe = g("preheater","natural_gas_in"), g("preheater","coal_in"), g("preheater","power_in")
                CCng, CCc, CCe, CCh = g("calciner","natural_gas_in"), g("calciner","coal_in"), g("calciner","power_in"), g("calciner","hydrogen_in")
                RKng, RKc, RKe, RKh = g("kiln","natural_gas_in"), g("kiln","coal_in"), g("kiln","power_in"), g("kiln","hydrogen_in")
                # process
                ccproc = g("calciner","co2_proc")
                # CO2
                PH = (np.sum(PHng)*ng_ef + np.sum(PHc)*coal_ef + np.sum(PHe)*grid_ef)*dt
                CC = (np.sum(CCng)*ng_ef + np.sum(CCc)*coal_ef + np.sum(CCe)*grid_ef + np.sum(CCh)*h2_ef)*dt
                RK = (np.sum(RKng)*ng_ef + np.sum(RKc)*coal_ef + np.sum(RKe)*grid_ef + np.sum(RKh)*h2_ef)*dt
                PROC = np.sum(ccproc)*dt
                total = PH+CC+RK+PROC
                # carbon cost
                CO2p = [safe_val(inst.co2_price[t]) for t in inst.time_steps] if hasattr(inst,"co2_price") else [0.0]*len(inst.time_steps)
                # need hourly totals for cost:
                tot_ts = []
                for i in range(len(inst.time_steps)):
                    e_ph = (PHng[i]*ng_ef + PHc[i]*coal_ef + PHe[i]*grid_ef)
                    e_cc = (CCng[i]*ng_ef + CCc[i]*coal_ef + CCe[i]*grid_ef + CCh[i]*h2_ef)
                    e_rk = (RKng[i]*ng_ef + RKc[i]*coal_ef + RKe[i]*grid_ef + RKh[i]*h2_ef)
                    proc = ccproc[i]
                    tot_ts.append(e_ph+e_cc+e_rk+proc)
                cost = float(np.sum([tot_ts[i]*CO2p[i] for i in range(len(inst.time_steps))]) * dt)
                return total, cost

            base_total, base_cost = totals_for(baseline_instance)
            dCO2  = E_total - base_total
            dCost = tot_carbon_cost - base_cost
            mac = (dCost / dCO2) if abs(dCO2) > 1e-9 else np.nan

            comp = go.Figure(data=[go.Table(
                header=dict(values=["Metric","Value"], align="left"),
                cells=dict(values=list(zip(*[
                    ("Baseline CO₂ [t]", f"{base_total:,.1f}"),
                    ("Scenario CO₂ [t]", f"{E_total:,.1f}"),
                    ("ΔCO₂ [t]", f"{dCO2:,.1f}"),
                    ("Baseline carbon cost [€]", f"{base_cost:,.0f}"),
                    ("Scenario carbon cost [€]", f"{tot_carbon_cost:,.0f}"),
                    ("Δ cost [€]", f"{dCost:,.0f}"),
                    ("MAC [€/tCO₂]", f"{mac:.1f}" if mac==mac else "—"),
                ])), align="left")
            )])
            comp.update_layout(title="Baseline Comparison & MAC")
            figs.append(comp)

        # ---------- write HTML ----------
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("<html><head><meta charset='utf-8'><title>Emissions Analytics</title></head><body>")
            f.write("<h2>Emissions Analytics</h2>")
            for i, fig in enumerate(figs, 1):
                f.write(fig.to_html(full_html=False, include_plotlyjs="cdn" if i==1 else False))
                f.write("<hr>")
            f.write("</body></html>")
        print(f"Emissions analytics saved to: {html_path}")


    def render_sankey_timeseries_1(self, instance, html_path="./outputs/sankey_timeseries_!.html",
                                step_stride=1, # use >1 to downsample frames if you have many timesteps
                                min_flow=1e-6  # hide tiny links for readability
                                ):
        def safe_value(v):
            try: return float(pyo.value(v))
            except Exception: return 0.0

        # ---- blocks (optional)
        B = instance.dsm_blocks
        ph = B["preheater"]     if "preheater"     in B else None
        cc = B["calciner"]      if "calciner"      in B else None
        rk = B["kiln"]          if "kiln"          in B else None
        el = B["electrolyser"]  if "electrolyser"  in B else None
        ts = B["thermal_storage"] if "thermal_storage" in B else None

        T = list(instance.time_steps)
        T = T[::step_stride]  # optional downsample

        def s_or_none(block, name):
            if block is None or not hasattr(block, name): return None
            return [safe_value(getattr(block, name)[t]) for t in T]

        # series (per-timestep)
        # electricity inputs (MWe)
        phP = s_or_none(ph, "power_in")
        ccP = s_or_none(cc, "power_in")
        rkP = s_or_none(rk, "power_in")
        tsP = s_or_none(ts, "power_in")  # TES heater (E→H conversion internal to TES)

        # fuels (MWth)
        phNG, phCoal = s_or_none(ph, "natural_gas_in"), s_or_none(ph, "coal_in")
        ccNG, ccCoal, ccH2 = s_or_none(cc, "natural_gas_in"), s_or_none(cc, "coal_in"), s_or_none(cc, "hydrogen_in")
        rkNG, rkCoal, rkH2 = s_or_none(rk, "natural_gas_in"), s_or_none(rk, "coal_in"), s_or_none(rk, "hydrogen_in")

        # unit outputs / couplings (MWth)
        phOut   = s_or_none(ph, "heat_out")
        phWH    = s_or_none(ph, "external_heat_in")   # WHR → PH
        ccOut   = s_or_none(cc, "heat_out")
        ccEffIn = s_or_none(cc, "effective_heat_in")  # = burner heat + TES discharge (if present)
        rkOut   = s_or_none(rk, "heat_out")

        tsChg, tsDis, tsSOC = s_or_none(ts, "charge"), s_or_none(ts, "discharge"), s_or_none(ts, "soc")

        # ---- node list (fixed order; show node even if zero in some frames)
        nodes = [
            "Electricity", "Natural Gas", "Coal", "Hydrogen", "WHR", "TES (discharge)",
            "Preheater", "Calciner", "Kiln", "Clinker"
        ]
        n = {name: i for i, name in enumerate(nodes)}

        # helper to build links for one timestep index k
        def links_at(k):
            src, tgt, val, lbl = [], [], [], []

            def add(a, b, v, text):
                if v is None: return
                if v <= min_flow: return
                src.append(n[a]); tgt.append(n[b]); val.append(float(v)); lbl.append(text)

            # Electricity to units / TES heater (units as input power)
            add("Electricity", "Preheater", phP[k] if phP else None, "E→PH (MWₑ)")
            add("Electricity", "Calciner",  ccP[k] if ccP else None, "E→CC (MWₑ)")
            add("Electricity", "Kiln",      rkP[k] if rkP else None, "E→Kiln (MWₑ)")
            add("Electricity", "TES (discharge)", tsP[k] if tsP else None, "E→TES heater (MWₑ)")

            # Fuels to units (thermal)
            add("Natural Gas", "Preheater", phNG[k]  if phNG  else None, "NG→PH (MWₜₕ)")
            add("Natural Gas", "Calciner",  ccNG[k]  if ccNG  else None, "NG→CC (MWₜₕ)")
            add("Natural Gas", "Kiln",      rkNG[k]  if rkNG  else None, "NG→Kiln (MWₜₕ)")

            add("Coal", "Preheater", phCoal[k] if phCoal else None, "Coal→PH (MWₜₕ)")
            add("Coal", "Calciner",  ccCoal[k] if ccCoal else None, "Coal→CC (MWₜₕ)")
            add("Coal", "Kiln",      rkCoal[k] if rkCoal else None, "Coal→Kiln (MWₜₕ)")

            add("Hydrogen", "Calciner", ccH2[k] if ccH2 else None, "H₂→CC (MWₜₕ)")
            add("Hydrogen", "Kiln",     rkH2[k] if rkH2 else None, "H₂→Kiln (MWₜₕ)")

            # WHR → PH; TES → CC
            add("WHR", "Preheater", phWH[k] if phWH else None, "WHR→PH (MWₜₕ)")
            add("TES (discharge)", "Calciner", tsDis[k] if tsDis else None, "TES→CC (MWₜₕ)")

            # Internal plant links
            add("Preheater", "Calciner", phOut[k] if phOut else None, "PH→CC (MWₜₕ)")
            add("Calciner", "Kiln", ccEffIn[k] if ccEffIn else (ccOut[k] if ccOut else None), "CC→Kiln (MWₜₕ)")
            add("Kiln", "Clinker", rkOut[k] if rkOut else None, "Kiln→Clinker (MWₜₕ)")

            return src, tgt, val, lbl

        # ---- build frames
        frames = []
        for i, t in enumerate(T):
            src, tgt, val, lbl = links_at(i)
            frame = go.Frame(
                data=[go.Sankey(
                    node=dict(label=nodes, pad=12, thickness=18),
                    link=dict(source=src, target=tgt, value=val, label=lbl, hovertemplate="%{label}<br>Flow: %{value:.2f}<extra></extra>")
                )],
                name=f"{t}"
            )
            frames.append(frame)

        # initial (first timestep)
        init_src, init_tgt, init_val, init_lbl = links_at(0)
        fig = go.Figure(
            data=[go.Sankey(
                node=dict(label=nodes, pad=12, thickness=18),
                link=dict(source=init_src, target=init_tgt, value=init_val, label=init_lbl,
                        hovertemplate="%{label}<br>Flow: %{value:.2f}<extra></extra>")
            )],
            frames=frames
        )

        # slider & buttons
        steps = [dict(method="animate",
                    args=[[f.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                    label=f"t={f.name}") for f in frames]

        fig.update_layout(
            title="Time‑dependent Sankey (per time step)",
            font_size=12,
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play", method="animate",
                        args=[None, {"fromcurrent": True, "frame": {"duration": 400, "redraw": True}}]),
                    dict(label="Pause", method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
                ],
                x=0.02, y=1.12, xanchor="left", yanchor="top"
            )],
            sliders=[dict(active=0, y= -0.06, x=0.05, len=0.9, steps=steps)]
        )

        fig.write_html(html_path)
        print(f"Sankey timeseries saved to: {html_path}")


    def render_cement_dashboard(self, instance, html_path="./outputs/cement_dashboard.html"):
        # ---- helpers
        def safe_value(v):
            try:
                return float(pyo.value(v))
            except Exception:
                return 0.0

        def series_or_none(block, name):
            if block is None or not hasattr(block, name):
                return None
            return [safe_value(getattr(block, name)[t]) for t in instance.time_steps]

        blocks = instance.dsm_blocks
        ph = blocks["preheater"]    if "preheater"    in blocks else None
        cc = blocks["calciner"]     if "calciner"     in blocks else None
        rk = blocks["kiln"]         if "kiln"         in blocks else None
        el = blocks["electrolyser"] if "electrolyser" in blocks else None
        ts = blocks["thermal_storage"] if "thermal_storage" in blocks else None

        T = list(instance.time_steps)

        # ---- gather time series
        # Preheater
        ph_power = series_or_none(ph, "power_in")
        ph_ng    = series_or_none(ph, "natural_gas_in")
        ph_coal  = series_or_none(ph, "coal_in")
        ph_out   = series_or_none(ph, "heat_out")
        ph_whr   = series_or_none(ph, "external_heat_in")

        # Calciner
        cc_power = series_or_none(cc, "power_in")
        cc_ng    = series_or_none(cc, "natural_gas_in")
        cc_coal  = series_or_none(cc, "coal_in")
        cc_h2    = series_or_none(cc, "hydrogen_in")
        cc_out   = series_or_none(cc, "heat_out")
        cc_effin = series_or_none(cc, "effective_heat_in")

        # Kiln
        rk_power = series_or_none(rk, "power_in")
        rk_ng    = series_or_none(rk, "natural_gas_in")
        rk_coal  = series_or_none(rk, "coal_in")
        rk_h2    = series_or_none(rk, "hydrogen_in")
        rk_out   = series_or_none(rk, "heat_out")

        # Electrolyser
        el_power = series_or_none(el, "power_in")
        el_h2    = series_or_none(el, "hydrogen_out")

        # TES
        ts_charge    = series_or_none(ts, "charge")
        ts_discharge = series_or_none(ts, "discharge")
        ts_soc       = series_or_none(ts, "soc")
        ts_power     = series_or_none(ts, "power_in")

        # Prices
        elec_price = [safe_value(instance.electricity_price[t]) for t in T] if hasattr(instance, "electricity_price") else None
        ng_price   = [safe_value(instance.natural_gas_price[t]) for t in T] if hasattr(instance, "natural_gas_price") else None
        coal_price = [safe_value(instance.coal_price[t])        for t in T] if hasattr(instance, "coal_price")        else None
        h2_price   = [safe_value(instance.hydrogen_price[t])    for t in T] if hasattr(instance, "hydrogen_price")    else None

        clinker_rate = [safe_value(instance.clinker_rate[t]) for t in T] if hasattr(instance, "clinker_rate") else None

        # ---- configuration summary (detected by flow > 0)
        def total(x): 
            return float(np.nansum(x)) if x is not None else 0.0

        present = {
            "Preheater": ph is not None,
            "Calciner":  cc is not None,
            "Kiln":      rk is not None,
            "Electrolyser": el is not None,
            "TES":       ts is not None,
        }
        fuels_used = {
            "Electricity→Preheater": total(ph_power) > 1e-6,
            "Electricity→Calciner":  total(cc_power) > 1e-6,
            "Electricity→Kiln":      total(rk_power) > 1e-6,
            "NaturalGas":            total(cc_ng)+total(ph_ng)+total(rk_ng) > 1e-6,
            "Coal":                  total(cc_coal)+total(ph_coal)+total(rk_coal) > 1e-6,
            "Hydrogen":              total(cc_h2)+total(rk_h2) > 1e-6 or total(el_h2) > 1e-6,
            "WHR→Preheater":         total(ph_whr) > 1e-6,
            "TES charge":            total(ts_charge) > 1e-6,
            "TES discharge":         total(ts_discharge) > 1e-6,
        }

        config_lines = []
        for k, v in present.items():
            config_lines.append(f"{k}: {'✓' if v else '—'}")
        for k, v in fuels_used.items():
            config_lines.append(f"{k}: {'✓' if v else '—'}")
        config_text = "<br>".join(config_lines)

        # ---- Sankey data (energy totals over horizon, MW_th·h = MWh_th)
        nodes = []
        def add_node(name): 
            if name not in nodes: nodes.append(name)
            return nodes.index(name)

        # supply nodes
        nElec = add_node("Electricity")
        nNG   = add_node("Natural Gas")
        nCoal = add_node("Coal")
        nH2   = add_node("Hydrogen")
        nWHR  = add_node("WHR")
        nTES  = add_node("TES (discharge)")

        # unit nodes
        nPH   = add_node("Preheater")
        nCC   = add_node("Calciner")
        nRK   = add_node("Kiln")
        nCLK  = add_node("Clinker")

        links_src, links_tgt, links_val, links_lbl = [], [], [], []

        def add_link(src, tgt, value, label):
            if value <= 1e-6: 
                return
            links_src.append(src); links_tgt.append(tgt); links_val.append(value); links_lbl.append(label)

        # Electricity to units / TES heater (thermalized via TES power_in*eta_e2h would require inside TES; here we just show electric as "power in")
        elec_to_ph = total(ph_power)
        elec_to_cc = total(cc_power)
        elec_to_rk = total(rk_power)
        elec_to_ts = total(ts_power)

        add_link(nElec, nPH, elec_to_ph, "E→PH (MWh_e)")
        add_link(nElec, nCC, elec_to_cc, "E→CC (MWh_e)")
        add_link(nElec, nRK, elec_to_rk, "E→Kiln (MWh_e)")
        add_link(nElec, nTES, elec_to_ts, "E→TES heater (MWh_e)")

        # Fossil / H2 to units (thermal MWh_th accounted as inputs vars are MW_th)
        add_link(nNG,   nPH, total(ph_ng),   "NG→PH (MWh_th)")
        add_link(nNG,   nCC, total(cc_ng),   "NG→CC (MWh_th)")
        add_link(nNG,   nRK, total(rk_ng),   "NG→Kiln (MWh_th)")
        add_link(nCoal, nPH, total(ph_coal), "Coal→PH (MWh_th)")
        add_link(nCoal, nCC, total(cc_coal), "Coal→CC (MWh_th)")
        add_link(nCoal, nRK, total(rk_coal), "Coal→Kiln (MWh_th)")

        add_link(nH2, nCC, total(cc_h2), "H2→CC (MWh_th)")
        add_link(nH2, nRK, total(rk_h2), "H2→Kiln (MWh_th)")

        # WHR to preheater; TES discharge to calciner
        add_link(nWHR, nPH, total(ph_whr), "WHR→PH (MWh_th)")
        add_link(nTES, nCC, total(ts_discharge), "TES→CC (MWh_th)")

        # Unit outputs to clinker (use effective calciner heat in for calciner contribution; kiln heat out for kiln)
        add_link(nPH,  nCC, total(ph_out), "PH→CC (MWh_th)")
        add_link(nCC,  nRK, total(cc_effin) if cc_effin is not None else total(cc_out), "CC eff→Kiln (MWh_th)")
        add_link(nRK,  nCLK, total(rk_out), "Kiln→Clinker (MWh_th)")

        sankey = go.Sankey(
            node=dict(label=nodes, pad=12, thickness=18),
            link=dict(source=links_src, target=links_tgt, value=links_val, label=links_lbl),
            valueformat=".1f"
        )

        # ---- time series figure (subplots-like with buttons)
        ts_traces = []
        x = T

        def add_trace(y, name, yaxis="y1", mode="lines"):
            if y is None or max(abs(v) for v in y) < 1e-9:
                return
            ts_traces.append(go.Scatter(x=x, y=y, name=name, mode=mode, yaxis=yaxis))

        # Inputs
        add_trace(ph_ng,   "Preheater NG [MW_th]")
        add_trace(cc_ng,   "Calciner NG [MW_th]")
        add_trace(rk_ng,   "Kiln NG [MW_th]")
        add_trace(ph_coal, "Preheater Coal [MW_th]")
        add_trace(cc_coal, "Calciner Coal [MW_th]")
        add_trace(rk_coal, "Kiln Coal [MW_th]")
        add_trace(cc_h2,   "Calciner H₂ [MW_th]")
        add_trace(rk_h2,   "Kiln H₂ [MW_th]")
        add_trace(ph_power,"Preheater Power [MW_e]")
        add_trace(cc_power,"Calciner Power [MW_e]")
        add_trace(rk_power,"Kiln Power [MW_e]")
        add_trace(ph_whr,  "WH→Preheater [MW_th]")
        add_trace(ts_power,"TES Heater Power [MW_e]")

        # Prices on secondary axis
        add_trace(elec_price, "Elec Price [€/MWh_e]", "y2")
        add_trace(ng_price,   "NG Price [€/MWh_th]",  "y2")
        add_trace(coal_price, "Coal Price [€/MWh_th]","y2")
        add_trace(h2_price,   "H₂ Price [€/MWh_th]",  "y2")

        # TES panel (charge/discharge/SOC)
        tes_traces = []
        if ts_charge is not None:
            tes_traces.append(go.Scatter(x=x, y=ts_charge,    name="TES Charge [MW_th]",    mode="lines"))
        if ts_discharge is not None:
            tes_traces.append(go.Scatter(x=x, y=ts_discharge, name="TES Discharge [MW_th]", mode="lines"))
        if ts_soc is not None:
            tes_traces.append(go.Scatter(x=x, y=ts_soc,       name="TES SOC [MWh_th]",      mode="lines", fill="tozeroy"))

        # Clinker rate
        if clinker_rate is not None:
            add_trace(clinker_rate, "Clinker Rate [t/h]")

        # ---- build figure (Sankey + time series + TES + config)
        fig1 = go.Figure(data=[sankey])
        fig1.update_layout(title="Energy Flow Sankey (sum over horizon)", font_size=12)

        fig2 = go.Figure(data=ts_traces)
        fig2.update_layout(
            title="Unit Inputs & Energy Prices (time series)",
            xaxis_title="Time step",
            yaxis=dict(title="Inputs / Power"),
            yaxis2=dict(title="Price", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )

        fig3 = go.Figure(data=tes_traces)
        fig3.update_layout(
            title="TES Operation",
            xaxis_title="Time step",
            yaxis_title="MW_th / MWh_th",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )

        # Config as an annotation panel (simple)
        fig1.add_annotation(
            x=1.02, y=0.5, xref="paper", yref="paper",
            showarrow=False, align="left",
            text=f"<b>Configuration</b><br>{config_text}"
        )

        # ---- write out a single HTML
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("<html><head><meta charset='utf-8'><title>Cement Dashboard</title></head><body>")
            f.write("<h2>Cement Plant Dashboard</h2>")
            f.write(fig1.to_html(full_html=False, include_plotlyjs="cdn"))
            f.write("<hr>")
            f.write(fig2.to_html(full_html=False, include_plotlyjs=False))
            f.write("<hr>")
            f.write(fig3.to_html(full_html=False, include_plotlyjs=False))
            f.write("</body></html>")

        print(f"Dashboard saved to: {html_path}")

    def render_sankey_timeseries(self, instance, html_path="./outputs/sankey_timeseries.html",
                                step_stride=1, # use >1 to downsample frames if you have many timesteps
                                min_flow=1e-6  # hide tiny links for readability
                                ):
        def safe_value(v):
            try: return float(pyo.value(v))
            except Exception: return 0.0

        # ---- blocks (optional)
        B = instance.dsm_blocks
        ph = B["preheater"]     if "preheater"     in B else None
        cc = B["calciner"]      if "calciner"      in B else None
        rk = B["kiln"]          if "kiln"          in B else None
        el = B["electrolyser"]  if "electrolyser"  in B else None
        ts = B["thermal_storage"] if "thermal_storage" in B else None

        T = list(instance.time_steps)
        T = T[::step_stride]  # optional downsample

        def s_or_none(block, name):
            if block is None or not hasattr(block, name): return None
            return [safe_value(getattr(block, name)[t]) for t in T]

        # series (per-timestep)
        # electricity inputs (MWe)
        phP = s_or_none(ph, "power_in")
        ccP = s_or_none(cc, "power_in")
        rkP = s_or_none(rk, "power_in")
        tsP = s_or_none(ts, "power_in")  # TES heater (E→H conversion internal to TES)

        # fuels (MWth)
        phNG, phCoal = s_or_none(ph, "natural_gas_in"), s_or_none(ph, "coal_in")
        ccNG, ccCoal, ccH2 = s_or_none(cc, "natural_gas_in"), s_or_none(cc, "coal_in"), s_or_none(cc, "hydrogen_in")
        rkNG, rkCoal, rkH2 = s_or_none(rk, "natural_gas_in"), s_or_none(rk, "coal_in"), s_or_none(rk, "hydrogen_in")

        # unit outputs / couplings (MWth)
        phOut   = s_or_none(ph, "heat_out")
        phWH    = s_or_none(ph, "external_heat_in")   # WHR → PH
        ccOut   = s_or_none(cc, "heat_out")
        ccEffIn = s_or_none(cc, "effective_heat_in")  # = burner heat + TES discharge (if present)
        rkOut   = s_or_none(rk, "heat_out")

        tsChg, tsDis, tsSOC = s_or_none(ts, "charge"), s_or_none(ts, "discharge"), s_or_none(ts, "soc")

        # ---- node list (fixed order; show node even if zero in some frames)
        nodes = [
            "Electricity", "Natural Gas", "Coal", "Hydrogen", "WHR", "TES (discharge)",
            "Preheater", "Calciner", "Kiln", "Clinker"
        ]
        n = {name: i for i, name in enumerate(nodes)}

        # helper to build links for one timestep index k
        def links_at(k):
            src, tgt, val, lbl = [], [], [], []

            def add(a, b, v, text):
                if v is None: return
                if v <= min_flow: return
                src.append(n[a]); tgt.append(n[b]); val.append(float(v)); lbl.append(text)

            # Electricity to units / TES heater (units as input power)
            add("Electricity", "Preheater", phP[k] if phP else None, "E→PH (MWₑ)")
            add("Electricity", "Calciner",  ccP[k] if ccP else None, "E→CC (MWₑ)")
            add("Electricity", "Kiln",      rkP[k] if rkP else None, "E→Kiln (MWₑ)")
            add("Electricity", "TES (discharge)", tsP[k] if tsP else None, "E→TES heater (MWₑ)")

            # Fuels to units (thermal)
            add("Natural Gas", "Preheater", phNG[k]  if phNG  else None, "NG→PH (MWₜₕ)")
            add("Natural Gas", "Calciner",  ccNG[k]  if ccNG  else None, "NG→CC (MWₜₕ)")
            add("Natural Gas", "Kiln",      rkNG[k]  if rkNG  else None, "NG→Kiln (MWₜₕ)")

            add("Coal", "Preheater", phCoal[k] if phCoal else None, "Coal→PH (MWₜₕ)")
            add("Coal", "Calciner",  ccCoal[k] if ccCoal else None, "Coal→CC (MWₜₕ)")
            add("Coal", "Kiln",      rkCoal[k] if rkCoal else None, "Coal→Kiln (MWₜₕ)")

            add("Hydrogen", "Calciner", ccH2[k] if ccH2 else None, "H₂→CC (MWₜₕ)")
            add("Hydrogen", "Kiln",     rkH2[k] if rkH2 else None, "H₂→Kiln (MWₜₕ)")

            # WHR → PH; TES → CC
            add("WHR", "Preheater", phWH[k] if phWH else None, "WHR→PH (MWₜₕ)")
            add("TES (discharge)", "Calciner", tsDis[k] if tsDis else None, "TES→CC (MWₜₕ)")

            # Internal plant links
            add("Preheater", "Calciner", phOut[k] if phOut else None, "PH→CC (MWₜₕ)")
            add("Calciner", "Kiln", ccEffIn[k] if ccEffIn else (ccOut[k] if ccOut else None), "CC→Kiln (MWₜₕ)")
            add("Kiln", "Clinker", rkOut[k] if rkOut else None, "Kiln→Clinker (MWₜₕ)")

            return src, tgt, val, lbl

        # ---- build frames
        frames = []
        for i, t in enumerate(T):
            src, tgt, val, lbl = links_at(i)
            frame = go.Frame(
                data=[go.Sankey(
                    node=dict(label=nodes, pad=12, thickness=18),
                    link=dict(source=src, target=tgt, value=val, label=lbl, hovertemplate="%{label}<br>Flow: %{value:.2f}<extra></extra>")
                )],
                name=f"{t}"
            )
            frames.append(frame)

        # initial (first timestep)
        init_src, init_tgt, init_val, init_lbl = links_at(0)
        fig = go.Figure(
            data=[go.Sankey(
                node=dict(label=nodes, pad=12, thickness=18),
                link=dict(source=init_src, target=init_tgt, value=init_val, label=init_lbl,
                        hovertemplate="%{label}<br>Flow: %{value:.2f}<extra></extra>")
            )],
            frames=frames
        )

        # slider & buttons
        steps = [dict(method="animate",
                    args=[[f.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                    label=f"t={f.name}") for f in frames]

        fig.update_layout(
            title="Time‑dependent Sankey (per time step)",
            font_size=12,
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play", method="animate",
                        args=[None, {"fromcurrent": True, "frame": {"duration": 400, "redraw": True}}]),
                    dict(label="Pause", method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
                ],
                x=0.02, y=1.12, xanchor="left", yanchor="top"
            )],
            sliders=[dict(active=0, y= -0.06, x=0.05, len=0.9, steps=steps)]
        )

        fig.write_html(html_path)
        print(f"Sankey timeseries saved to: {html_path}")

    def animated_tes_sankey(self, instance, html_path="./outputs/cement_tes_playback.html"):
        def safe_value(v):
            try:
                return float(pyo.value(v))
            except Exception:
                return 0.0

        T = list(instance.time_steps)
        blocks = instance.dsm_blocks
        ts = blocks["thermal_storage"] if "thermal_storage" in blocks else None
        cc = blocks["calciner"] if "calciner" in blocks else None

        def series_or_none(block, name):
            if block is None or not hasattr(block, name):
                return None
            return [safe_value(getattr(block, name)[t]) for t in T]

        ts_charge    = series_or_none(ts, "charge")
        ts_discharge = series_or_none(ts, "discharge")
        ts_soc       = series_or_none(ts, "soc")

        cc_heat_out  = series_or_none(cc, "heat_out")
        cc_effin     = series_or_none(cc, "effective_heat_in")

        # --- Nodes
        nodes = ["Electricity", "TES (charge)", "TES (discharge)", "Calciner", "Clinker"]
        nElec, nCharge, nDis, nCC, nClk = range(len(nodes))

        frames = []
        for ti, t in enumerate(T):
            links_src, links_tgt, links_val, links_lbl = [], [], [], []

            def add_link(src, tgt, val, lbl):
                if val > 1e-6:
                    links_src.append(src)
                    links_tgt.append(tgt)
                    links_val.append(val)
                    links_lbl.append(lbl)

            add_link(nElec, nCharge, ts_charge[ti], f"E→TES {ts_charge[ti]:.1f}")
            add_link(nDis, nCC, ts_discharge[ti], f"TES→CC {ts_discharge[ti]:.1f}")
            add_link(nCC, nClk, cc_effin[ti] if cc_effin else cc_heat_out[ti], f"CC→Clinker")

            frame = go.Frame(
                data=[go.Sankey(
                    node=dict(label=nodes, pad=12, thickness=18),
                    link=dict(source=links_src, target=links_tgt, value=links_val, label=links_lbl)
                )],
                name=f"t={t}"
            )
            frames.append(frame)

        # Initial Sankey (t=0)
        fig = go.Figure(
            data=[go.Sankey(
                node=dict(label=nodes, pad=12, thickness=18),
                link=dict(source=[], target=[], value=[], label=[])
            )],
            frames=frames
        )

        # Layout with play controls
        fig.update_layout(
            title="TES Dynamics: Time Playback",
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                ]
            )],
            sliders=[dict(
                steps=[dict(method="animate", args=[[f.name], {"mode": "immediate"}], label=f.name) for f in frames],
                active=0
            )]
        )

        fig.write_html(html_path)
        print(f"Animated TES Sankey saved → {html_path}")

    
    def render_storage_analytics(self, instance, html_path="./outputs/storage_analytics.html"):
        """
        Detailed interactive analysis for TES.
        Includes KPIs, time-series, duration curves, cycles, price arbitrage, and energy balance.
        """

        # -------- helpers
        def safe_value(v):
            try:
                return float(pyo.value(v))
            except Exception:
                return 0.0

        def series_or_none(block, name):
            if block is None or not hasattr(block, name):
                return None
            return [safe_value(getattr(block, name)[t]) for t in instance.time_steps]

        # -------- grab blocks
        blocks = instance.dsm_blocks
        ts = blocks["thermal_storage"] if "thermal_storage" in blocks else None
        cc = blocks["calciner"] if "calciner" in blocks else None

        if ts is None:
            print("No TES block found; skipping storage analytics.")
            return

        T = list(instance.time_steps)

        # -------- TES series
        charge    = series_or_none(ts, "charge")      # MW_th
        discharge = series_or_none(ts, "discharge")   # MW_th
        soc       = series_or_none(ts, "soc")         # MWh_th
        p_in      = series_or_none(ts, "power_in")    # MW_e (if generator mode)
        # Efficiencies (if present); if not, use 1.0 safely
        eta_ch = getattr(ts, "eta_charge", None)
        eta_di = getattr(ts, "eta_discharge", None)
        eta_e2h = getattr(ts, "eta_e2h", None)
        eta_charge = safe_value(eta_ch) if eta_ch is not None else 1.0
        eta_discharge = safe_value(eta_di) if eta_di is not None else 1.0
        eta_elec_to_heat = safe_value(eta_e2h) if eta_e2h is not None else 1.0

        # -------- Prices
        elec_price = [safe_value(instance.electricity_price[t]) for t in T] \
            if hasattr(instance, "electricity_price") else None
        ng_price   = [safe_value(instance.natural_gas_price[t]) for t in T] \
            if hasattr(instance, "natural_gas_price") else None
        coal_price = [safe_value(instance.coal_price[t]) for t in T] \
            if hasattr(instance, "coal_price") else None
        h2_price   = [safe_value(instance.hydrogen_price[t]) for t in T] \
            if hasattr(instance, "hydrogen_price") else None
        co2_price  = [safe_value(instance.co2_price[t]) for t in T] \
            if hasattr(instance, "co2_price") else None

        # -------- Calciner for substitution analysis
        cc_heat_out  = series_or_none(cc, "heat_out")
        cc_eff_in    = series_or_none(cc, "effective_heat_in")
        cc_ng        = series_or_none(cc, "natural_gas_in")
        cc_coal      = series_or_none(cc, "coal_in")
        cc_h2        = series_or_none(cc, "hydrogen_in")
        cc_power     = series_or_none(cc, "power_in")
        # fossil split efficiency (for delivered cost calc)
        eta_fossil = safe_value(getattr(cc, "eta_fossil", 1.0)) if cc else 1.0
        eta_electric = safe_value(getattr(cc, "eta_electric", 1.0)) if cc else 1.0
        ng_ef = safe_value(getattr(cc, "ng_co2_factor", 0.0)) if cc else 0.0
        coal_ef = safe_value(getattr(cc, "coal_co2_factor", 0.0)) if cc else 0.0

        # -------- Build a DataFrame for convenience
        df = pd.DataFrame({
            "t": T,
            "charge_MWth": charge,
            "discharge_MWth": discharge,
            "soc_MWh": soc,
            "tes_power_MWe": p_in,
            "elec_price": elec_price,
            "ng_price": ng_price,
            "coal_price": coal_price,
            "h2_price": h2_price,
            "co2_price": co2_price,
            "cc_heat_out": cc_heat_out,
            "cc_eff_in": cc_eff_in,
            "cc_ng": cc_ng,
            "cc_coal": cc_coal,
            "cc_h2": cc_h2,
            "cc_power": cc_power,
        })

        # -------- KPIs
        dt = 1.0  # time step hours (adjust if different)
        E_charge = float(np.nansum(df["charge_MWth"]) * dt)       # MWh_th into TES (thermal bus)
        E_dis    = float(np.nansum(df["discharge_MWth"]) * dt)    # MWh_th from TES
        E_loss   = max(E_charge * eta_charge - E_dis * eta_discharge, 0.0) if (eta_charge and eta_discharge) else max(E_charge - E_dis, 0.0)

        # round-trip efficiency (thermal): discharge / (charge * eta_charge^-1 * eta_discharge?)
        # We report simple E_dis / E_charge for visibility; users can interpret with ηs.
        rte_simple = (E_dis / E_charge) if E_charge > 1e-9 else np.nan

        avg_soc = float(np.nanmean(df["soc_MWh"])) if soc is not None else np.nan
        max_soc = float(np.nanmax(df["soc_MWh"])) if soc is not None else np.nan
        util = (E_dis / max(1e-9, max_soc)) if (soc is not None and max_soc > 1e-9) else np.nan

        # cycles (surrogate): count zero-crossings of net flow and sum down‑up minima as “full cycles”
        net = df["discharge_MWth"].fillna(0.0) - df["charge_MWth"].fillna(0.0)
        # simple half-cycle counting: number of local extrema of SOC / 2
        soc_series = df["soc_MWh"].fillna(method="ffill").fillna(0.0)
        extrema = 0
        for i in range(1, len(soc_series)-1):
            if (soc_series[i] > soc_series[i-1] and soc_series[i] > soc_series[i+1]) or \
            (soc_series[i] < soc_series[i-1] and soc_series[i] < soc_series[i+1]):
                extrema += 1
        approx_cycles = max(extrema // 2, 0)

        # -------- Price spread / implied substitution
        # Implied delivered cost of burner heat (very approximate):
        # NG-delivered = NG/η_fossil + CO2*EF_ng ; Coal similar; H2-delivered = H2/η_fossil ; Elec-delivered = Elec/η_electric
        if ng_price is not None:
            ng_del = np.array(df["ng_price"]) / max(eta_fossil, 1e-9) + (np.array(df["co2_price"]) * ng_ef if co2_price is not None else 0.0)
        else:
            ng_del = None
        coal_del = (np.array(df["coal_price"]) / max(eta_fossil, 1e-9) + (np.array(df["co2_price"]) * coal_ef)) if (coal_price is not None and co2_price is not None) else None
        h2_del = (np.array(df["h2_price"]) / max(eta_fossil, 1e-9)) if h2_price is not None else None
        elec_del = (np.array(df["elec_price"]) / max(eta_elec_to_heat*eta_discharge, 1e-9)) if elec_price is not None else None

        # Pick the **active** burner in calciner per hour to estimate displaced cost
        # (if multiple fuels, pick the one with highest input that hour)
        displaced = []
        for i in range(len(df)):
            cc_inputs = [
                ("NG",   df.loc[i, "cc_ng"]   or 0.0, ng_del[i]   if ng_del   is not None else None),
                ("Coal", df.loc[i, "cc_coal"] or 0.0, coal_del[i] if coal_del is not None else None),
                ("H2",   df.loc[i, "cc_h2"]   or 0.0, h2_del[i]   if h2_del   is not None else None),
                ("Elec", df.loc[i, "cc_power"]or 0.0, (np.array(df["elec_price"])[i]/max(eta_electric,1e-9)) if elec_price is not None else None)
            ]
            fuel, qty, cost = max(cc_inputs, key=lambda x: x[1])
            displaced.append(cost if (cost is not None and df.loc[i, "discharge_MWth"] > 1e-9) else np.nan)
        df["displaced_cost_EUR_per_MWhth"] = displaced
        df["tes_delivered_cost_EUR_per_MWhth"] = (np.array(df["elec_price"])/max(eta_elec_to_heat*eta_discharge,1e-9)) if elec_price is not None else np.nan

        # -------- Figures
        figs = []

        # KPIs table
        kpi_rows = [
            ("Energy charged [MWh_th]", f"{E_charge:,.1f}"),
            ("Energy discharged [MWh_th]", f"{E_dis:,.1f}"),
            ("Simple round-trip eff [–]", f"{rte_simple:.3f}" if rte_simple==rte_simple else "—"),
            ("Approx cycles [#]", f"{approx_cycles}"),
            ("Avg SOC [MWh_th]", f"{avg_soc:,.1f}" if avg_soc==avg_soc else "—"),
            ("Max SOC [MWh_th]", f"{max_soc:,.1f}" if max_soc==max_soc else "—"),
            ("Utilization (E_dis / MaxSOC) [h]", f"{util:.2f}" if util==util else "—"),
        ]
        kfig = go.Figure(data=[go.Table(
            header=dict(values=["KPI", "Value"], align="left"),
            cells=dict(values=list(zip(*kpi_rows)), align="left")
        )])
        kfig.update_layout(title="TES KPIs")
        figs.append(kfig)

        # Time series: charge/discharge/SOC
        ts_fig = ps.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                subplot_titles=("Charge & Discharge [MW_th]", "SOC [MWh_th]"))
        ts_fig.add_trace(go.Scatter(x=T, y=df["charge_MWth"], name="Charge [MW_th]"), row=1, col=1)
        ts_fig.add_trace(go.Scatter(x=T, y=df["discharge_MWth"], name="Discharge [MW_th]"), row=1, col=1)
        ts_fig.add_trace(go.Scatter(x=T, y=df["soc_MWh"], name="SOC [MWh_th]", fill="tozeroy"), row=2, col=1)
        ts_fig.update_layout(title="TES Operation (Time Series)")
        figs.append(ts_fig)

        # Duration curves
        dur_fig = ps.make_subplots(rows=1, cols=3, subplot_titles=("Charge duration", "Discharge duration", "SOC duration"))
        dur_fig.add_trace(go.Scatter(y=sorted(df["charge_MWth"], reverse=True), name="Charge"), row=1, col=1)
        dur_fig.add_trace(go.Scatter(y=sorted(df["discharge_MWth"], reverse=True), name="Discharge"), row=1, col=2)
        dur_fig.add_trace(go.Scatter(y=sorted(df["soc_MWh"], reverse=True), name="SOC"), row=1, col=3)
        dur_fig.update_layout(title="Duration Curves")
        figs.append(dur_fig)

        # Cycle histogram (based on SOC turning points)
        # Build simple amplitude series: absolute SOC differences between consecutive extrema
        turn_idx = []
        for i in range(1, len(soc_series)-1):
            if (soc_series[i] > soc_series[i-1] and soc_series[i] > soc_series[i+1]) or \
            (soc_series[i] < soc_series[i-1] and soc_series[i] < soc_series[i+1]):
                turn_idx.append(i)
        amplitudes = []
        for i in range(1, len(turn_idx)):
            amplitudes.append(abs(soc_series[turn_idx[i]] - soc_series[turn_idx[i-1]]))
        cyc_fig = go.Figure()
        cyc_fig.add_trace(go.Histogram(x=amplitudes, nbinsx=20, name="Cycle ΔSOC [MWh_th]"))
        cyc_fig.update_layout(title="Cycle Amplitude Histogram", xaxis_title="ΔSOC per half-cycle [MWh_th]")
        figs.append(cyc_fig)

        # Price arbitrage scatter: TES delivered cost vs displaced burner cost (only at discharge hours)
        if elec_price is not None:
            mask = (df["discharge_MWth"] > 1e-9) & df["tes_delivered_cost_EUR_per_MWhth"].notna() & pd.Series(displaced).notna()
            pfig = go.Figure()
            pfig.add_trace(go.Scatter(
                x=df.loc[mask, "tes_delivered_cost_EUR_per_MWhth"],
                y=df.loc[mask, "displaced_cost_EUR_per_MWhth"],
                mode="markers",
                name="Hours with discharge",
                text=[f"t={t}" for t in df.loc[mask, "t"]],
            ))
            pfig.add_shape(type="line", x0=0, y0=0, x1=max(df.loc[mask,"tes_delivered_cost_EUR_per_MWhth"].max(), 
                                                        df.loc[mask,"displaced_cost_EUR_per_MWhth"].max()),
                        y1=max(df.loc[mask,"tes_delivered_cost_EUR_per_MWhth"].max(), 
                                df.loc[mask,"displaced_cost_EUR_per_MWhth"].max()),
                        line=dict(dash="dash"))
            pfig.update_layout(
                title="Arbitrage Check: TES delivered cost vs displaced burner cost",
                xaxis_title="TES delivered cost [€/MWh_th]",
                yaxis_title="Displaced burner cost [€/MWh_th]"
            )
            figs.append(pfig)

        # Energy balance waterfall (charge → losses → discharge)
        # Approximate thermal losses as charge*eta_charge - discharge/eta_discharge (if known)
        losses_est = max(E_charge*eta_charge - E_dis*eta_discharge, 0.0) if (eta_charge and eta_discharge) else max(E_charge - E_dis, 0.0)
        wfig = go.Figure(go.Waterfall(
            name="TES",
            orientation="v",
            measure=["relative", "relative", "total"],
            x=["Charge in", "Losses", "Discharge out"],
            text=[f"{E_charge:.1f}", f"-{losses_est:.1f}", f"{E_dis:.1f}"],
            y=[E_charge, -losses_est, E_dis - E_charge + losses_est],
        ))
        wfig.update_layout(title="TES Energy Balance (MWh_th)")
        figs.append(wfig)

        # Optional: marginal substitution bar — average reduction in calciner inputs during discharge hours
        if cc is not None:
            m = (df["discharge_MWth"] > 1e-9)
            red_ng   = float(max(0.0, (df.loc[m, "cc_ng"].diff(-1)*0).sum()))  # placeholder; true marginal requires a “with vs without TES” compare
            # Instead, show average inputs during discharge vs no-discharge hours:
            comp_rows = [
                ("Avg NG input when discharging [MW_th]",   f"{df.loc[m, 'cc_ng'].mean():.2f}"),
                ("Avg H2 input when discharging [MW_th]",   f"{df.loc[m, 'cc_h2'].mean():.2f}"),
                ("Avg Elec input when discharging [MW_e]",  f"{df.loc[m, 'cc_power'].mean():.2f}"),
                ("Avg NG input otherwise [MW_th]",          f"{df.loc[~m, 'cc_ng'].mean():.2f}"),
                ("Avg H2 input otherwise [MW_th]",          f"{df.loc[~m, 'cc_h2'].mean():.2f}"),
                ("Avg Elec input otherwise [MW_e]",         f"{df.loc[~m, 'cc_power'].mean():.2f}"),
            ]
            compfig = go.Figure(data=[go.Table(
                header=dict(values=["Metric", "Value"], align="left"),
                cells=dict(values=list(zip(*comp_rows)), align="left")
            )])
            compfig.update_layout(title="Calciner Inputs vs. TES Discharge (diagnostic)")
            figs.append(compfig)

        # -------- Write a single HTML
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("<html><head><meta charset='utf-8'><title>TES Analytics</title></head><body>")
            f.write("<h2>Thermal Storage Analytics</h2>")
            for i, fig in enumerate(figs, 1):
                f.write(fig.to_html(full_html=False, include_plotlyjs="cdn" if i==1 else False))
                f.write("<hr>")
            f.write("</body></html>")
        print(f"Storage analytics saved to: {html_path}")
