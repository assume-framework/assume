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

# These come from your codebase (same as CementPlant)
# from assume.units.dsm_load_shift import DSMFlex, SupportsMinMax
# from assume.forecast import Forecaster

# And your technology classes in dst_components.py
# from dst_components import DigestateSeparation, Dryer, TCRBooster, PSA, BatteryStorage


class TCRPlant(DSMFlex, SupportsMinMax):
    """
    Master unit for the TCR® bio-waste / sewage-sludge processing plant (black-box only).

    Components expected (black-box chain):
        digestate_separation -> dryer -> tcr_booster -> psa

    Optional:
        battery_storage (only in SEPH35)

    Args mirror CementPlant:
        id, unit_operator, bidding_strategies, forecaster, node, location, components, objective,
        flexibility_measure, is_prosumer, peak_load_cap, demand/cost_tolerance, etc.

    Notes:
      - Plant-level products for reporting:
            hydrogen_out = psa.h2_out
            oil_out      = tcr_booster.oil_out
            char_out     = tcr_booster.char_out
            tail_gas_out = psa.tail_gas_out
      - Waste/export streams:
            liquid_digestate_out = digestate_separation.liquid_out
            dryer_water_removed  = dryer.water_removed
            tcr_water_phase_out  = tcr_booster.water_out
    """

    # all required for the black-box chain
    required_technologies = [
        "digestate_separation",
        "dryer",
        "tcr_booster",
        "psa",
    ]

    optional_technologies = [
        "battery_storage",
    ]

    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        forecaster,  # Forecaster
        technology: str = "tcr_plant",
        node: str = "node0",
        location: tuple[float, float] = (0.0, 0.0),
        components: dict[str, dict] = None,
        objective: str = None,
        flexibility_measure: str = "",
        is_prosumer: str = "No",
        peak_load_cap: float = 0,
        demand: float = 0,
        # production target / feed constraint (optional; choose one concept)
        feed_target: float = 0.0,  # e.g., annual or per-horizon feed in t (if you use it)
        cost_tolerance: float = 10.0,
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
                    f"Component {component} is required for the TCR plant unit."
                )

        # check if the provided components are valid and do not contain any unknown components
        for component in components.keys():
            if (
                component not in self.required_technologies
                and component not in self.optional_technologies
            ):
                raise ValueError(
                    f"Components {component} is not a valid component for the TCR plant unit."
                )

        self.has_battery_storage = "battery_storage" in self.components.keys()
        # Check the presence of components first
        self.has_digester_seperation = "digestate_separation" in self.components.keys()
        self.has_dryer = "dryer" in self.components.keys()
        self.has_tcr_booster = "tcr_booster" in self.components.keys()
        self.has_psa = "psa" in self.components.keys()

        self.oil_demand_per_time_step = self.forecaster[f"{self.id}_oil_demand"]
        self.electricity_price = self.forecaster["electricity_price"]
        self.co2_price = self.forecaster["co2_price"]
        self.natural_gas_price = self.forecaster["natural_gas_price"]

        # flexibility / objective metadata
        self.demand = demand
        self.objective = objective
        self.flexibility_measure = flexibility_measure
        self.cost_tolerance = cost_tolerance
        self.peak_load_cap = peak_load_cap
        # f.load_series is whatever your project’s method is to register Series
        self.oil_demand_per_time_step = self.forecaster[f"{self.id}_oil_demand"]

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
        if self.natural_gas_price is not None:
            self.model.natural_gas_price = pyo.Param(
                self.model.time_steps,
                initialize={t: value for t, value in enumerate(self.natural_gas_price)},
            )
        self.model.co2_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.co2_price)},
        )

        self.model.absolute_oil_demand = pyo.Param(initialize=float(self.demand))  # will be 0
        self.model.oil_demand_per_time_step = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.oil_demand_per_time_step)},
        )

        # self.model.absolute_feed_target = pyo.Param(initialize=float(self.feed_target))
        self.model.peak_load_cap = pyo.Param(initialize=float(self.peak_load_cap))
        self.model.cost_tolerance = pyo.Param(initialize=float(self.cost_tolerance))


    def define_variables(self):
        # --- plant-level electricity rollup (MW) ---
        self.model.total_power_input = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )

        # --- plant-level economics & emissions rollup ---
        self.model.variable_cost = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.total_co2_emission = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )

        # --- plant-level throughput and outputs (t/h equivalents) ---
        # Feed into black-box (typically separator feed_in)
        self.model.feed_in_rate = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )

        # Products
        self.model.hydrogen_rate = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.oil_rate = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.char_rate = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.tail_gas_rate = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )

        # Wastes / exports
        self.model.liquid_digestate_rate = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.dryer_water_removed_rate = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.tcr_water_phase_rate = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )

        # Optional: if you later want a peak-power objective/constraint
        self.model.peak_power = pyo.Var(within=pyo.NonNegativeReals)

    # ---------------------------------------------------------------------
    # Process wiring (black-box chain)
    # ---------------------------------------------------------------------
    def initialize_process_sequence(self) -> None:
        """
        Initializes the process sequence for the TCR plant.

        Central connections:
        - Material chain: digestate_separation -> dryer -> tcr_booster -> psa
        - Tandem constraint: whenever TCR produces gas, PSA must process it and must produce H2.
        - Electricity bus:
                grid_import supplies (i) plant electric loads and (ii) optional battery charging
                battery discharge can supply part of plant electric load
                plant load = sum(power_in of relevant technologies)
        """
        m = self.model
        T = m.time_steps

        # ----------------------------
        # 1) Material links (black-box)
        # ----------------------------
        if self.has_digester_seperation and self.has_dryer:

            @m.Constraint(T)
            def sep_to_dryer_mass_link(m, t):
                return (
                    m.dsm_blocks["dryer"].wet_in[t]
                    == m.dsm_blocks["digestate_separation"].solid_out[t]
                )

            @m.Constraint(T)
            def plant_feed_alias(m, t):
                # plant inlet feed rate (wet) = separator feed_in
                return m.feed_in_rate[t] == m.dsm_blocks["digestate_separation"].feed_in[t]

            @m.Constraint(T)
            def plant_liquid_digestate_alias(m, t):
                return (
                    m.liquid_digestate_rate[t]
                    == m.dsm_blocks["digestate_separation"].liquid_out[t]
                )

        if self.has_dryer and self.has_tcr_booster:

            @m.Constraint(T)
            def dryer_to_tcr_mass_link(m, t):
                return (
                    m.dsm_blocks["tcr_booster"].feed_in[t]
                    == m.dsm_blocks["dryer"].dry_out[t]
                )

            @m.Constraint(T)
            def plant_dryer_water_removed_alias(m, t):
                return (
                    m.dryer_water_removed_rate[t]
                    == m.dsm_blocks["dryer"].water_removed[t]
                )

        # ----------------------------
        # 2) TCR -> PSA tandem coupling
        # ----------------------------
        if self.has_tcr_booster and self.has_psa:
            tb = m.dsm_blocks["tcr_booster"]
            psa = m.dsm_blocks["psa"]

            # (i) Gas must flow into PSA (no bypass)
            @m.Constraint(T)
            def tcr_to_psa_gas_link(m, t):
                return psa.gas_in[t] == tb.gas_out[t]

            # (ii) Tandem: Hydrogen production must follow incoming gas (binding recovery)
            # h2_out = r * x_H2 * gas_in
            # This removes the "h2_out = 0" cost-minimising bypass.
            @m.Constraint(T)
            def psa_h2_production_binding(m, t):
                return (
                    psa.h2_out[t]
                    == psa.h2_recovery * psa.h2_mass_fraction_in_syngas * psa.gas_in[t]
                )

            # (iii) Tail gas as residual of PSA inlet (simple robust balance)
            @m.Constraint(T)
            def psa_tail_gas_residual(m, t):
                return psa.tail_gas_out[t] == psa.gas_in[t] - psa.h2_out[t]

            # ---- Plant-level product aliases from TCR ----
            @m.Constraint(T)
            def plant_oil_alias(m, t):
                return m.oil_rate[t] == tb.oil_out[t]

            @m.Constraint(T)
            def plant_char_alias(m, t):
                return m.char_rate[t] == tb.char_out[t]

            @m.Constraint(T)
            def plant_tcr_water_phase_alias(m, t):
                return m.tcr_water_phase_rate[t] == tb.water_out[t]

        # ---- Plant-level aliases from PSA ----
        if self.has_psa:
            psa = m.dsm_blocks["psa"]

            @m.Constraint(T)
            def plant_h2_alias(m, t):
                return m.hydrogen_rate[t] == psa.h2_out[t]

            @m.Constraint(T)
            def plant_tail_gas_alias(m, t):
                return m.tail_gas_rate[t] == psa.tail_gas_out[t]

        # ----------------------------------------
        # 3) Electricity bus: grid + optional battery
        # ----------------------------------------
        # Ensure grid_import exists (Var). Preferably declare this in define_variables(), but safe here.
        if not hasattr(m, "grid_import"):
            m.grid_import = pyo.Var(T, within=pyo.NonNegativeReals)

        # Build plant electric load as sum of power_in[t] of electricity-using tech blocks.
        # NG-mode blocks should enforce power_in == 0 internally (as in your other components).
        def _tech_power_sum_at_t(m, t):
            total = 0
            for tech_name, blk in m.dsm_blocks.items():
                if tech_name == "battery_storage":
                    continue
                if hasattr(blk, "power_in"):
                    total += blk.power_in[t]
            return total

        @m.Constraint(T)
        def bind_total_power_input(m, t):
            return m.total_power_input[t] == _tech_power_sum_at_t(m, t)

        # Battery coupling if present
        if self.has_battery_storage:
            b = m.dsm_blocks["battery_storage"]

            # Align to your battery variable names (you said you fixed charge/discharge naming earlier)
            if (not hasattr(b, "charge")) or (not hasattr(b, "discharge")):
                raise AttributeError(
                    "battery_storage block must have charge[t] and discharge[t] for plant electricity coupling."
                )

            @m.Constraint(T)
            def electricity_bus_balance(m, t):
                # grid_import + discharge = plant_load + charge
                return (
                    m.grid_import[t] + b.discharge[t]
                    == m.total_power_input[t] + b.charge[t]
                )

        else:

            @m.Constraint(T)
            def electricity_bus_balance(m, t):
                return m.grid_import[t] == m.total_power_input[t]


    # ---------------------------------------------------------------------
    # Constraints: connections + plant-level bookkeeping
    # ---------------------------------------------------------------------
    def define_constraints(self):
        if self.is_prosumer:
            # self._add_fcr_capacity_market(self.model)
            # self._add_reserve_capacity_market_guardrail(self.model)
            self._add_reserve_capacity_market_threshold_guardrail(self.model)

        """
        Defines the constraints for the TCR plant model (black-box chain).
        """
        m = self.model
        T = m.time_steps

        # ---------------------------------------------------------
        # 1) Absolute target constraints (feed target, optional H2 demand)
        # ---------------------------------------------------------
        
        # def absolute_feed_target_constraint(m):
        #     """
        #     Ensures the plant processes at least the absolute feed target over the horizon.
        #     Interpretation: feed_in_rate is t/h, time step is 1h => sum is t.
        #     """
        #     if (not self.feed_target) or self.feed_target == 0:
        #         return pyo.Constraint.Skip
        #     if self.has_digester_seperation:
        #         return (
        #             sum(m.dsm_blocks["digestate_separation"].feed_in[t] for t in T)
        #             >= m.absolute_feed_target
        #         )
        #     return pyo.Constraint.Skip
        @m.Constraint(T)
        def oil_demand_association_constraint(m, t):
            """
            Ensures oil output meets the demand definition.
            If self.demand==0: enforce per-time-step oil demand series.
            Else: enforce absolute oil demand total over horizon.
            """
            if (not self.demand) or self.demand == 0:
                if self.has_tcr_booster:
                    return (
                        m.dsm_blocks["tcr_booster"].oil_out[t]
                        >= m.oil_demand_per_time_step[t]
                    )
                return pyo.Constraint.Skip
            else:
                if self.has_tcr_booster:
                    return (
                        sum(m.dsm_blocks["tcr_booster"].oil_out[t] for t in T)
                        >= m.absolute_oil_demand
                    )
                return pyo.Constraint.Skip


        # ---------------------------------------------------------
        # 2) Total power input roll-up (sum of all tech power_in)
        #    NOTE: battery charge/discharge is handled in electricity bus balance (initialize_process_sequence)
        # ---------------------------------------------------------
        @m.Constraint(T)
        def total_power_input_constraint(m, t):
            total_power = 0.0

            if self.has_digester_seperation:
                total_power += m.dsm_blocks["digestate_separation"].power_in[t]

            if self.has_dryer:
                # Dryer has power_in in electricity/both mode; equals 0 in NG mode (internal constraint)
                total_power += m.dsm_blocks["dryer"].power_in[t]

            if self.has_tcr_booster:
                # same logic as dryer
                total_power += m.dsm_blocks["tcr_booster"].power_in[t]

            if self.has_psa:
                total_power += m.dsm_blocks["psa"].power_in[t]

            return m.total_power_input[t] == total_power

        # ---------------------------------------------------------
        # 3) Variable cost roll-up (sum of component operating_cost)
        # ---------------------------------------------------------
        @m.Constraint(T)
        def cost_per_time_step(m, t):
            variable_cost = 0.0

            # --- add component costs, but remove electricity part for battery case ---
            # easiest: include component operating_cost as-is ONLY if NO battery
            if not self.has_battery_storage:
                if self.has_digester_seperation:
                    variable_cost += m.dsm_blocks["digestate_separation"].operating_cost[t]
                if self.has_dryer:
                    variable_cost += m.dsm_blocks["dryer"].operating_cost[t]
                if self.has_tcr_booster:
                    variable_cost += m.dsm_blocks["tcr_booster"].operating_cost[t]
                if self.has_psa:
                    variable_cost += m.dsm_blocks["psa"].operating_cost[t]
            else:
                # Battery present: charge electricity only on grid_import.
                # So here include ONLY non-electric component costs.
                # (You need to expose non-electric cost vars per component, or compute them here.)
                # Quick pragmatic approach for now:
                if self.has_dryer and hasattr(m.dsm_blocks["dryer"], "natural_gas_in"):
                    dr = m.dsm_blocks["dryer"]
                    variable_cost += dr.natural_gas_in[t] * m.natural_gas_price[t]
                    if hasattr(dr, "co2_energy"):
                        variable_cost += dr.co2_energy[t] * m.co2_price[t]

                if self.has_tcr_booster and hasattr(m.dsm_blocks["tcr_booster"], "natural_gas_in"):
                    tb = m.dsm_blocks["tcr_booster"]
                    variable_cost += tb.natural_gas_in[t] * m.natural_gas_price[t]
                    if hasattr(tb, "co2_energy"):
                        variable_cost += tb.co2_energy[t] * m.co2_price[t]

                # disposal costs if you model them as params on blocks:
                if self.has_digester_seperation and hasattr(m.dsm_blocks["digestate_separation"], "liquid_disposal_cost"):
                    sep = m.dsm_blocks["digestate_separation"]
                    variable_cost += sep.liquid_out[t] * sep.liquid_disposal_cost
                if self.has_dryer and hasattr(m.dsm_blocks["dryer"], "water_disposal_cost"):
                    dr = m.dsm_blocks["dryer"]
                    variable_cost += dr.water_removed[t] * dr.water_disposal_cost

                # now add electricity cost ONCE, based on grid import
                variable_cost += m.grid_import[t] * m.electricity_price[t]

            return m.variable_cost[t] == variable_cost

        # ---------------------------------------------------------
        # 4) Total CO2 emission roll-up
        #    Prefer component CO2 vars if present; fallback to NG input * factor if needed.
        # ---------------------------------------------------------
        @m.Constraint(T)
        def total_co2_emission_constraint(m, t):
            total = 0.0

            # Dryer (NG energy CO2 if modelled)
            if self.has_dryer and ("dryer" in m.dsm_blocks):
                dr = m.dsm_blocks["dryer"]
                if hasattr(dr, "co2_energy"):
                    total += dr.co2_energy[t]
                # fallback: natural_gas_in[t] * ng_co2_factor if co2_energy not present
                elif hasattr(dr, "natural_gas_in") and hasattr(dr, "ng_co2_factor"):
                    total += dr.natural_gas_in[t] * dr.ng_co2_factor

            # TCR Booster (NG energy CO2 if modelled)
            if self.has_tcr_booster and ("tcr_booster" in m.dsm_blocks):
                tb = m.dsm_blocks["tcr_booster"]
                if hasattr(tb, "co2_energy"):
                    total += tb.co2_energy[t]
                elif hasattr(tb, "natural_gas_in") and hasattr(tb, "ng_co2_factor"):
                    total += tb.natural_gas_in[t] * tb.ng_co2_factor

            # Other units typically electricity-only => no direct energy CO2 here
            return m.total_co2_emission[t] == total

        # ---------------------------------------------------------
        # 5) Bind plant-level reporting rates to component streams (optional but useful)
        # ---------------------------------------------------------
        if self.has_digester_seperation:

            @m.Constraint(T)
            def bind_feed_in_rate(m, t):
                return m.feed_in_rate[t] == m.dsm_blocks["digestate_separation"].feed_in[t]

            @m.Constraint(T)
            def bind_liquid_digestate_rate(m, t):
                return m.liquid_digestate_rate[t] == m.dsm_blocks["digestate_separation"].liquid_out[t]

        if self.has_dryer:

            @m.Constraint(T)
            def bind_dryer_water_removed_rate(m, t):
                return m.dryer_water_removed_rate[t] == m.dsm_blocks["dryer"].water_removed[t]

        if self.has_tcr_booster:

            @m.Constraint(T)
            def bind_oil_rate(m, t):
                return m.oil_rate[t] == m.dsm_blocks["tcr_booster"].oil_out[t]

            @m.Constraint(T)
            def bind_char_rate(m, t):
                return m.char_rate[t] == m.dsm_blocks["tcr_booster"].char_out[t]

            @m.Constraint(T)
            def bind_tcr_water_phase_rate(m, t):
                return m.tcr_water_phase_rate[t] == m.dsm_blocks["tcr_booster"].water_out[t]

        if self.has_psa:

            @m.Constraint(T)
            def bind_hydrogen_rate(m, t):
                return m.hydrogen_rate[t] == m.dsm_blocks["psa"].h2_out[t]

            @m.Constraint(T)
            def bind_tail_gas_rate(m, t):
                return m.tail_gas_rate[t] == m.dsm_blocks["psa"].tail_gas_out[t]

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

    def _add_reserve_capacity_market_threshold_guardrail(self, model):
        """
        Reserve capacity market with threshold/guardrail strategy (TCR plant variant).

        Products (per 4h block):
        - FCR symmetric OR aFRR up OR aFRR down (mutually exclusive).
        Eligibility guardrail:
        - Block ineligible if ANY hour within it has el_price <= threshold.

        Feasibility:
        - Reserve modulation is constrained against net grid import:
                P_grid[t] = m.grid_import[t]
            Upward reserve = headroom to increase P_grid (e.g., charge battery more / increase electric use)
            Downward reserve = footroom to reduce P_grid (e.g., discharge battery / reduce electric use)
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

        def _pval(x, default=0.0):
            try:
                return float(pyo.value(x))
            except Exception:
                return float(default)

        enable_fcr  = bool(getattr(self, "enable_fcr", False))
        enable_afrr = bool(getattr(self, "enable_afrr", True))

        L       = int(getattr(self, "_FCR_BLOCK_LENGTH", 4))   # hours per block
        step    = float(getattr(self, "_FCR_STEP_MW", 1.0))    # MW granularity
        min_bid = float(getattr(self, "_FCR_MIN_BID_MW", 1.0)) # MW minimum bid

        price_threshold = float(getattr(self, "reserve_price_threshold", 0.0))  # €/MWh_e

        # ---------------- time base & 4h blocks ----------------
        ts = self.index
        Tn = len(ts)

        # blocks start at clock times 00:00, 04:00, 08:00, ...
        starts = [i for i, dt in enumerate(ts[: -L + 1]) if (dt.hour % 4 == 0 and dt.minute == 0)]
        starts = [i for i in starts if i + L <= Tn]
        m.fcr_blocks = pyo.Set(initialize=starts, ordered=True)

        # ---------------- price series (hourly) ----------------
        if hasattr(m, "electricity_price"):
            el_price_hour = [float(pyo.value(m.electricity_price[t])) for t in m.time_steps]
        else:
            el_price_hour = _as_float_list(getattr(self, "electricity_price", None), Tn)

        price_fcr_hour = _as_float_list(getattr(self, "fcr_price", None), Tn)         # €/MW/h
        price_pos_hour = _as_float_list(getattr(self, "afrr_price_pos", None), Tn)    # €/MW/h
        price_neg_hour = _as_float_list(getattr(self, "afrr_price_neg", None), Tn)    # €/MW/h

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

        # ---------------- capacity envelope on net grid power (MW) ----------------
        # Preferred: use explicit grid connection cap if you have it
        max_cap = float(getattr(self, "peak_load_cap", 0.0))
        min_cap = float(getattr(self, "min_grid_import", 0.0))  # usually 0

        # If max_cap not provided, derive from equipment caps (sum of max_power)
        if max_cap <= 0.0 and hasattr(m, "dsm_blocks"):
            B = m.dsm_blocks
            for name, blk in B.items():
                # include battery max power if present
                if name == "battery_storage":
                    # common names: max_power or max_charge_power; keep safe
                    if hasattr(blk, "max_power"):
                        max_cap += _pval(blk.max_power)
                    elif hasattr(blk, "max_charge"):
                        max_cap += _pval(blk.max_charge)
                    continue

                # include any electric block with a max_power param
                if hasattr(blk, "max_power"):
                    max_cap += _pval(blk.max_power)

        self.max_grid_capacity = max_cap
        self.min_grid_capacity = min_cap

        # ---------------- net grid power (MW) ----------------
        if not hasattr(m, "grid_import"):
            raise AttributeError("Reserve market requires m.grid_import[t] to be defined (Var).")

        # Use grid_import directly (no export in your model)
        m.net_grid_power = pyo.Expression(m.time_steps, rule=lambda mm, t: mm.grid_import[t])

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

        # mutual exclusivity: one product per block
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

        # ---------------- feasibility inside each block hour (headroom/footroom) ----------------
        # Headroom: ability to increase grid import
        # Footroom: ability to decrease grid import (not below min_cap, usually 0)
        m.fcr_up_feas   = pyo.ConstraintList()
        m.fcr_dn_feas   = pyo.ConstraintList()
        m.fcr_sym_feas  = pyo.ConstraintList()

        for b in starts:
            for k in range(L):
                t = b + k
                # aFRR up: increase import <= headroom
                m.fcr_up_feas.add( m.cap_up[b]  <= (max_cap - m.net_grid_power[t]) )
                # aFRR down: decrease import <= footroom
                m.fcr_dn_feas.add( m.cap_dn[b]  <= (m.net_grid_power[t] - min_cap) )
                # FCR symmetric must satisfy both
                m.fcr_sym_feas.add( m.cap_sym[b] <= (max_cap - m.net_grid_power[t]) )
                m.fcr_sym_feas.add( m.cap_sym[b] <= (m.net_grid_power[t] - min_cap) )

        # ---------------- revenue expression ----------------
        @m.Expression()
        def reserve_revenue(mm):
            return (
                sum(mm.fcr_block_price[b]        * mm.cap_sym[b] for b in mm.fcr_blocks)
                + sum(mm.afrr_block_price_pos[b] * mm.cap_up[b]  for b in mm.fcr_blocks)
                + sum(mm.afrr_block_price_neg[b] * mm.cap_dn[b]  for b in mm.fcr_blocks)
            )
    
    def plot_1(self, instance, save_name=None, out_dir="./outputs", show=True):
        """
        Two-panel figure + CSV export for the TCR plant.

        Top:
            - Electric loads per technology (MW_e)
            - Grid import (MW_e)
            - Battery charge/discharge (MW_e)
            - Electricity / NG / CO2 prices (twin y-axis)

        Bottom:
            - Material chain rates (t/h): feed_in, solid_out, liquid_out, dry_out, water_removed
            - Products (t/h): oil_out, gas_out, char_out, h2_out, tail_gas_out
            - Oil demand series (t/h) if available
            - Battery charge/discharge (MW_e) + SOC as area (MWh)
            - Plant totals (variable_cost, total_co2_emission) exported to CSV

        If save_name is given: saves PNG + CSV into out_dir.
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
            obj = getattr(block, name)
            try:
                return [safe_value(obj[t]) for t in instance.time_steps]
            except Exception:
                return None

        def series_model_or_none(name):
            if not hasattr(instance, name):
                return None
            obj = getattr(instance, name)
            try:
                return [safe_value(obj[t]) for t in instance.time_steps]
            except Exception:
                return None

        def plot_if_nonzero(ax, x, y, label, color, style="-", eps=1e-9):
            if y is None:
                return None
            if any(abs(v) > eps for v in y):
                return ax.plot(x, y, label=label, color=color, linestyle=style)[0]
            return None

        # ------- blocks -------
        B = instance.dsm_blocks
        sep = B["digestate_separation"] if "digestate_separation" in B else None
        dr = B["dryer"] if "dryer" in B else None
        tb = B["tcr_booster"] if "tcr_booster" in B else None
        psa = B["psa"] if "psa" in B else None
        bat = B["battery_storage"] if "battery_storage" in B else None

        # ------- series (electricity) -------
        sep_P = series_or_none(sep, "power_in")
        dr_P = series_or_none(dr, "power_in")
        tb_P = series_or_none(tb, "power_in")
        psa_P = series_or_none(psa, "power_in")

        grid_import = series_model_or_none("grid_import")
        total_power = series_model_or_none("total_power_input")

        # battery (support both naming conventions)
        bat_ch = series_or_none(bat, "charge_power")
        if bat_ch is None:
            bat_ch = series_or_none(bat, "charge")

        bat_ds = series_or_none(bat, "discharge_power")
        if bat_ds is None:
            bat_ds = series_or_none(bat, "discharge")

        bat_soc = series_or_none(bat, "soc")

        # ------- series (fuels, if NG mode exists in dryer / booster) -------
        dr_NG = series_or_none(dr, "natural_gas_in")
        tb_NG = series_or_none(tb, "natural_gas_in")

        # ------- series (material / products) -------
        # separator
        sep_feed = series_or_none(sep, "feed_in")
        sep_solid = series_or_none(sep, "solid_out")
        sep_liq = series_or_none(sep, "liquid_out")

        # dryer
        dr_wet_in = series_or_none(dr, "wet_in")
        dr_dry_out = series_or_none(dr, "dry_out")
        dr_water_rm = series_or_none(dr, "water_removed")

        # tcr booster
        tb_feed = series_or_none(tb, "feed_in")
        tb_oil = series_or_none(tb, "oil_out")
        tb_gas = series_or_none(tb, "gas_out")
        tb_char = series_or_none(tb, "char_out")
        tb_water = series_or_none(tb, "water_out")

        # psa
        psa_gas_in = series_or_none(psa, "gas_in")
        psa_h2 = series_or_none(psa, "h2_out")
        psa_tail = series_or_none(psa, "tail_gas_out")

        # oil demand series (Param on model, if you defined it)
        oil_demand = None
        if hasattr(instance, "oil_demand_per_time_step"):
            try:
                oil_demand = [
                    safe_value(instance.oil_demand_per_time_step[t])
                    for t in instance.time_steps
                ]
            except Exception:
                oil_demand = None

        # ------- prices & plant metrics -------
        elec_price = series_model_or_none("electricity_price")
        ng_price = series_model_or_none("natural_gas_price")
        co2_price = series_model_or_none("co2_price")

        variable_cost = series_model_or_none("variable_cost")
        total_co2 = series_model_or_none("total_co2_emission")

        # fallback CO2 aggregation if plant total not present
        if total_co2 is None:
            total = [0.0 for _ in T]
            if dr is not None and hasattr(dr, "co2_energy"):
                y = series_or_none(dr, "co2_energy")
                if y:
                    total = [total[i] + y[i] for i in range(len(T))]
            if tb is not None and hasattr(tb, "co2_energy"):
                y = series_or_none(tb, "co2_energy")
                if y:
                    total = [total[i] + y[i] for i in range(len(T))]
            if any(abs(v) > 1e-12 for v in total):
                total_co2 = total

        # ------- figure -------
        fig, axs = plt.subplots(
            2, 1, figsize=(12, 9), sharex=True, constrained_layout=True
        )

        # =========================
        # TOP: electricity & prices
        # =========================
        lines_top = []
        lines_top += [
            plot_if_nonzero(axs[0], T, sep_P, "Separation Power [MWₑ]", "C0"),
            plot_if_nonzero(axs[0], T, dr_P, "Dryer Power [MWₑ]", "C1", "--"),
            plot_if_nonzero(axs[0], T, tb_P, "TCR Booster Power [MWₑ]", "C2", "-."),
            plot_if_nonzero(axs[0], T, psa_P, "PSA Power [MWₑ]", "C3", ":"),
            plot_if_nonzero(axs[0], T, total_power, "Total Plant Power [MWₑ]", "C4"),
            plot_if_nonzero(axs[0], T, grid_import, "Grid Import [MWₑ]", "C5", "--"),
            plot_if_nonzero(axs[0], T, bat_ch, "Battery Charge [MWₑ]", "C6"),
            plot_if_nonzero(axs[0], T, bat_ds, "Battery Discharge [MWₑ]", "C7", "--"),
        ]

        axs[0].set_ylabel("Power (MWₑ)")
        axs[0].set_title("TCR Plant – Electricity Operation & Prices")
        axs[0].grid(True, which="both", axis="both")

        # prices on twin axis
        axp = axs[0].twinx()
        lp = []
        if elec_price:
            lp += axp.plot(
                T,
                elec_price,
                label="Elec Price [€/MWhₑ]",
                color="C10",
                linestyle="--",
            )
        if ng_price:
            lp += axp.plot(
                T, ng_price, label="NG Price [€/MWh]", color="C11", linestyle=":"
            )
        if co2_price:
            lp += axp.plot(
                T,
                co2_price,
                label="CO₂ Price [€/tCO₂]",
                color="C12",
                linestyle="-.",
            )
        axp.set_ylabel("Price", color="gray")
        axp.tick_params(axis="y", labelcolor="gray")

        handles = [h for h in lines_top if h is not None] + lp
        labels = [h.get_label() for h in handles]
        if handles:
            axs[0].legend(handles, labels, loc="upper left", frameon=True, ncol=2)

        # ==========================================
        # BOTTOM: material chain + products + demand
        #         + battery charge/discharge & SOC area
        # ==========================================
        lines_bot = []
        lines_bot += [
            plot_if_nonzero(axs[1], T, sep_feed, "Separator Feed In [t/h]", "C0"),
            plot_if_nonzero(
                axs[1], T, sep_solid, "Separator Solid Out [t/h]", "C1", "--"
            ),
            plot_if_nonzero(
                axs[1], T, sep_liq, "Liquid Digestate Out [t/h]", "C2", "-."
            ),
            plot_if_nonzero(axs[1], T, dr_dry_out, "Dryer Dry Out [t/h]", "C3", ":"),
            plot_if_nonzero(
                axs[1], T, dr_water_rm, "Dryer Water Removed [t/h]", "C4"
            ),
            plot_if_nonzero(axs[1], T, tb_oil, "Oil Out [t/h]", "C5"),
            plot_if_nonzero(axs[1], T, tb_gas, "Gas Out [t/h]", "C6", "--"),
            plot_if_nonzero(axs[1], T, tb_char, "Char Out [t/h]", "C7", "-."),
            plot_if_nonzero(axs[1], T, psa_h2, "H₂ Out [t/h]", "C8", ":"),
            plot_if_nonzero(axs[1], T, psa_tail, "Tail Gas Out [t/h]", "C9", "--"),
            plot_if_nonzero(axs[1], T, oil_demand, "Oil Demand [t/h]", "C10", ":"),
            # Battery on bottom panel too (as requested)
            plot_if_nonzero(axs[1], T, bat_ch, "Battery Charge [MWₑ]", "C6", "-"),
            plot_if_nonzero(axs[1], T, bat_ds, "Battery Discharge [MWₑ]", "C7", "--"),
        ]

        # SOC as area plot
        if bat_soc is not None and any(abs(v) > 1e-9 for v in bat_soc):
            axs[1].fill_between(
                T,
                bat_soc,
                0,
                alpha=0.25,
                label="Battery SOC [MWh]",
            )

        axs[1].set_ylabel("Flows (t/h) and Battery (MWₑ, MWh)")
        axs[1].set_title("TCR Plant – Material Chain, Products, Oil Demand, and Battery")
        axs[1].grid(True, which="both", axis="both")
        blines = [h for h in lines_bot if h is not None]
        if blines:
            axs[1].legend(loc="upper left", frameon=True, ncol=2)
        axs[1].set_xlabel("Time step")

        plt.tight_layout()

        # ------- CSV export -------
        df = pd.DataFrame(
            {
                "t": T,
                # electricity (MW)
                "Separation Power [MW_e]": sep_P if sep_P else None,
                "Dryer Power [MW_e]": dr_P if dr_P else None,
                "TCR Booster Power [MW_e]": tb_P if tb_P else None,
                "PSA Power [MW_e]": psa_P if psa_P else None,
                "Total Plant Power [MW_e]": total_power if total_power else None,
                "Grid Import [MW_e]": grid_import if grid_import else None,
                "Battery Charge [MW_e]": bat_ch if bat_ch else None,
                "Battery Discharge [MW_e]": bat_ds if bat_ds else None,
                "Battery SOC [MWh]": bat_soc if bat_soc else None,
                # NG inputs (if present)
                "Dryer NG In [MW_th]": dr_NG if dr_NG else None,
                "TCR Booster NG In [MW_th]": tb_NG if tb_NG else None,
                # chain & products (t/h)
                "Separator Feed In [t/h]": sep_feed if sep_feed else None,
                "Separator Solid Out [t/h]": sep_solid if sep_solid else None,
                "Liquid Digestate Out [t/h]": sep_liq if sep_liq else None,
                "Dryer Wet In [t/h]": dr_wet_in if dr_wet_in else None,
                "Dryer Dry Out [t/h]": dr_dry_out if dr_dry_out else None,
                "Dryer Water Removed [t/h]": dr_water_rm if dr_water_rm else None,
                "TCR Feed In [t/h]": tb_feed if tb_feed else None,
                "Oil Out [t/h]": tb_oil if tb_oil else None,
                "Gas Out [t/h]": tb_gas if tb_gas else None,
                "Char Out [t/h]": tb_char if tb_char else None,
                "TCR Water Phase Out [t/h]": tb_water if tb_water else None,
                "PSA Gas In [t/h]": psa_gas_in if psa_gas_in else None,
                "H2 Out [t/h]": psa_h2 if psa_h2 else None,
                "Tail Gas Out [t/h]": psa_tail if psa_tail else None,
                "Oil Demand [t/h]": oil_demand if oil_demand else None,
                # prices
                "Elec Price [€/MWh_e]": elec_price if elec_price else None,
                "NG Price [€/MWh]": ng_price if ng_price else None,
                "CO2 Price [€/tCO2]": co2_price if co2_price else None,
                # plant metrics
                "Variable Cost [€]": variable_cost if variable_cost else None,
                "Total CO2 [tCO2/step]": total_co2 if total_co2 else None,
            }
        )

        # ---------- SAVE ----------
        if save_name:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(save_name).name
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
        Three-panel figure + CSV export for the TCR plant (capacity products view).

        1) Plant electricity (tech power + grid import) + energy prices (twin y-axis)
        2) Battery operation (charge, discharge, SOC as area)
        3) Reserve market view (FCR sym ±, aFRR up/down) as capacity bands around baseline grid import
        + ancillary prices (twin y-axis)

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
            try:
                return [safe_value(getattr(block, name)[t]) for t in instance.time_steps]
            except Exception:
                return None

        def series_model_or_none(name):
            if not hasattr(instance, name):
                return None
            try:
                return [safe_value(getattr(instance, name)[t]) for t in instance.time_steps]
            except Exception:
                return None

        def plot_if_nonzero(ax, x, y, label, color=None, style="-", eps=1e-9):
            if y is None:
                return None
            if any(abs(v) > eps for v in y):
                if color is None:
                    return ax.plot(x, y, label=label, linestyle=style)[0]
                return ax.plot(x, y, label=label, color=color, linestyle=style)[0]
            return None

        def _reserve_series(inst, hours, Lblk):
            """
            Expand block variables to hourly "stairs" arrays.
            Expected vars on instance:
            - cap_sym[b], cap_up[b], cap_dn[b]
            - fcr_block_price[b], afrr_block_price_pos[b], afrr_block_price_neg[b]
            - fcr_blocks set
            """
            out = {
                "mask_sym": [0.0] * len(hours),
                "mask_up": [0.0] * len(hours),
                "mask_dn": [0.0] * len(hours),
                "stairs_price_sym": [0.0] * len(hours),
                "stairs_price_pos": [0.0] * len(hours),
                "stairs_price_neg": [0.0] * len(hours),
            }
            if not hasattr(inst, "fcr_blocks"):
                return out

            blocks = list(getattr(inst, "fcr_blocks"))
            if not blocks:
                return out

            for b in blocks:
                # capacities
                cap_sym = safe_value(getattr(inst, "cap_sym")[b]) if hasattr(inst, "cap_sym") else 0.0
                cap_up  = safe_value(getattr(inst, "cap_up")[b])  if hasattr(inst, "cap_up")  else 0.0
                cap_dn  = safe_value(getattr(inst, "cap_dn")[b])  if hasattr(inst, "cap_dn")  else 0.0

                # block prices (€/MW·4h)
                p_sym = safe_value(getattr(inst, "fcr_block_price")[b]) if hasattr(inst, "fcr_block_price") else 0.0
                p_up  = safe_value(getattr(inst, "afrr_block_price_pos")[b]) if hasattr(inst, "afrr_block_price_pos") else 0.0
                p_dn  = safe_value(getattr(inst, "afrr_block_price_neg")[b]) if hasattr(inst, "afrr_block_price_neg") else 0.0

                for k in range(Lblk):
                    t = b + k
                    if 0 <= t < len(hours):
                        out["mask_sym"][t] = cap_sym
                        out["mask_up"][t] = cap_up
                        out["mask_dn"][t] = cap_dn
                        out["stairs_price_sym"][t] = p_sym
                        out["stairs_price_pos"][t] = p_up
                        out["stairs_price_neg"][t] = p_dn
            return out

        # ------- blocks -------
        B = instance.dsm_blocks
        sep = B["digestate_separation"] if "digestate_separation" in B else None
        dr = B["dryer"] if "dryer" in B else None
        tb = B["tcr_booster"] if "tcr_booster" in B else None
        psa = B["psa"] if "psa" in B else None
        bat = B["battery_storage"] if "battery_storage" in B else None

        # ------- electricity series (MW) -------
        sep_P = series_or_none(sep, "power_in")
        dr_P = series_or_none(dr, "power_in")
        tb_P = series_or_none(tb, "power_in")
        psa_P = series_or_none(psa, "power_in")

        total_power = series_model_or_none("total_power_input")
        grid_import = series_model_or_none("grid_import")

        # battery (support both naming conventions)
        bat_ch = series_or_none(bat, "charge_power") or series_or_none(bat, "charge")
        bat_ds = series_or_none(bat, "discharge_power") or series_or_none(bat, "discharge")
        bat_soc = series_or_none(bat, "soc")

        # ------- prices -------
        elec_price = series_model_or_none("electricity_price")
        ng_price = series_model_or_none("natural_gas_price")
        co2_price = series_model_or_none("co2_price")

        # ------- baseline for reserves (use grid import) -------
        base = grid_import if grid_import is not None else (total_power if total_power is not None else [0.0] * H)

        # capability envelope (MW)
        max_cap = getattr(self, "max_grid_capacity", None)
        if max_cap is None or (isinstance(max_cap, (int, float)) and max_cap <= 0):
            max_cap = getattr(self, "peak_load_cap", None)
        min_cap = getattr(self, "min_grid_capacity", 0.0)

        # reserve presence?
        fcr_present = hasattr(instance, "fcr_blocks") and len(list(getattr(instance, "fcr_blocks"))) > 0
        Lblk = int(getattr(self, "_FCR_BLOCK_LENGTH", 4))
        R = _reserve_series(instance, T, Lblk) if fcr_present else None

        # ------- figure -------
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)

        # ---------- PANEL 1: electricity + prices ----------
        l_top = []
        l_top += [
            plot_if_nonzero(axs[0], T, sep_P, "Separation Power [MWₑ]", "C0"),
            plot_if_nonzero(axs[0], T, dr_P, "Dryer Power [MWₑ]", "C1", "--"),
            plot_if_nonzero(axs[0], T, tb_P, "TCR Booster Power [MWₑ]", "C2", "-."),
            plot_if_nonzero(axs[0], T, psa_P, "PSA Power [MWₑ]", "C3", ":"),
            plot_if_nonzero(axs[0], T, total_power, "Total Plant Load [MWₑ]", "C4"),
            plot_if_nonzero(axs[0], T, grid_import, "Grid Import [MWₑ]", "C5", "--"),
        ]
        axs[0].set_ylabel("Power (MWₑ)")
        axs[0].set_title("TCR Plant – Electricity Operation & Energy Prices")

        axp = axs[0].twinx()
        lp = []
        if elec_price:
            lp += axp.plot(T, elec_price, label="Elec Price [€/MWhₑ]", color="C6", linestyle="--")
        if ng_price:
            lp += axp.plot(T, ng_price, label="NG Price [€/MWh]", color="C7", linestyle=":")
        if co2_price:
            lp += axp.plot(T, co2_price, label="CO₂ Price [€/tCO₂]", color="C8", linestyle="-.")

        axp.set_ylabel("Price", color="gray")
        axp.tick_params(axis="y", labelcolor="gray")

        handles = [h for h in l_top if h is not None] + lp
        if handles:
            axs[0].legend(handles, [h.get_label() for h in handles], loc="upper left", frameon=True, ncol=2)

        # ---------- PANEL 2: battery operation ----------
        p0 = plot_if_nonzero(axs[1], T, bat_ch, "Battery Charge [MWₑ]", "C6", "-")
        p1 = plot_if_nonzero(axs[1], T, bat_ds, "Battery Discharge [MWₑ]", "C7", "--")
        if bat_soc is not None and any(abs(v) > 1e-9 for v in bat_soc):
            axs[1].fill_between(T, bat_soc, 0, alpha=0.25, label="Battery SOC [MWh]")
        axs[1].set_ylabel("MWₑ / MWh")
        axs[1].set_title("Battery Operation")
        if any([p0, p1]) or (bat_soc is not None):
            axs[1].legend(loc="upper left", frameon=True)

        # ---------- PANEL 3: reserve products around baseline ----------
        axs[2].set_ylabel("MW")
        axs[2].plot(T, base, color="0.25", lw=2, label="Baseline (Grid Import) [MW]")

        if max_cap is not None:
            axs[2].plot(T, [max_cap] * H, color="0.7", ls=":", label="Max Capability [MW]")
        if min_cap is not None:
            axs[2].plot(T, [min_cap] * H, color="0.7", ls="--", label="Min Capability [MW]")

        if fcr_present and R is not None:
            base_arr = np.array(base, dtype=float)

            # FCR symmetric: ±cap_sym around baseline
            sym = np.array(R["mask_sym"], dtype=float)
            if any(abs(v) > 1e-9 for v in sym):
                lo = np.maximum(min_cap, base_arr - sym) if min_cap is not None else (base_arr - sym)
                hi = np.minimum(max_cap, base_arr + sym) if max_cap is not None else (base_arr + sym)
                axs[2].fill_between(T, lo, hi, step="pre", alpha=0.25, label="FCR sym [±MW]")

            # aFRR up: reduction band (base - cap_up to base)
            up = np.array(R["mask_up"], dtype=float)
            if any(abs(v) > 1e-9 for v in up):
                lo = np.maximum(min_cap, base_arr - up) if min_cap is not None else (base_arr - up)
                axs[2].fill_between(T, lo, base_arr, step="pre", alpha=0.25, label="aFRR Up [MW]")

            # aFRR down: increase band (base to base + cap_dn)
            dn = np.array(R["mask_dn"], dtype=float)
            if any(abs(v) > 1e-9 for v in dn):
                hi = np.minimum(max_cap, base_arr + dn) if max_cap is not None else (base_arr + dn)
                axs[2].fill_between(T, base_arr, hi, step="pre", alpha=0.25, label="aFRR Down [MW]")

            # ancillary prices on twin axis
            axp2 = axs[2].twinx()
            hp = []
            if any(abs(v) > 1e-9 for v in R["stairs_price_sym"]):
                hp += axp2.plot(T, R["stairs_price_sym"], ls="--", label="FCR price [€/MW·4h]")
            if any(abs(v) > 1e-9 for v in R["stairs_price_pos"]):
                hp += axp2.plot(T, R["stairs_price_pos"], ls=":", label="aFRR+ price [€/MW·4h]")
            if any(abs(v) > 1e-9 for v in R["stairs_price_neg"]):
                hp += axp2.plot(T, R["stairs_price_neg"], ls="-.", label="aFRR− price [€/MW·4h]")

            axp2.set_ylabel("AS price", color="gray")
            axp2.tick_params(axis="y", labelcolor="gray")

            h1, l1 = axs[2].get_legend_handles_labels()
            h2, l2 = axp2.get_legend_handles_labels()
            axs[2].legend(h1 + h2, l1 + l2, loc="upper left", frameon=True, ncol=2)

            axs[2].set_title("Ancillary Bids — FCR (±) and aFRR (Up/Down) around Baseline")
        else:
            axs[2].set_title("Operational Baseline (no reserve structures)")
            axs[2].legend(loc="upper left", frameon=True)

        axs[-1].set_xlabel("Time step")
        fig.autofmt_xdate()
        plt.tight_layout()

        # ---------- CSV export ----------
        df = pd.DataFrame(
            {
                "t": T,
                # electricity
                "Separation Power [MW_e]": sep_P if sep_P else None,
                "Dryer Power [MW_e]": dr_P if dr_P else None,
                "TCR Booster Power [MW_e]": tb_P if tb_P else None,
                "PSA Power [MW_e]": psa_P if psa_P else None,
                "Total Plant Load [MW_e]": total_power if total_power else None,
                "Grid Import [MW_e]": grid_import if grid_import else None,
                # battery
                "Battery Charge [MW_e]": bat_ch if bat_ch else None,
                "Battery Discharge [MW_e]": bat_ds if bat_ds else None,
                "Battery SOC [MWh]": bat_soc if bat_soc else None,
                # prices
                "Elec Price [€/MWh_e]": elec_price if elec_price else None,
                "NG Price [€/MWh]": ng_price if ng_price else None,
                "CO2 Price [€/tCO2]": co2_price if co2_price else None,
                # reserve (expanded to hourly)
                "Baseline Grid Import [MW]": base,
            }
        )

        if fcr_present and R is not None:
            df["FCR Sym Cap [MW]"] = R["mask_sym"]
            df["aFRR Up Cap [MW]"] = R["mask_up"]
            df["aFRR Down Cap [MW]"] = R["mask_dn"]
            df["FCR Price [€/MW·4h]"] = R["stairs_price_sym"]
            df["aFRR+ Price [€/MW·4h]"] = R["stairs_price_pos"]
            df["aFRR− Price [€/MW·4h]"] = R["stairs_price_neg"]

        # ---------- SAVE ----------
        if save_name:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(save_name).stem
            png_path = out_dir / f"{stem}.png"
            csv_path = out_dir / f"{stem}.csv"
            plt.savefig(png_path, dpi=300, bbox_inches="tight")
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"Saved: {png_path}\nSaved: {csv_path}")

        if show:
            plt.show()
        plt.close(fig)