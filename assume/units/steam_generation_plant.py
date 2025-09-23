# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as ps
import pyomo.environ as pyo

from assume.common.base import SupportsMinMax
from assume.common.forecasts import Forecaster
from assume.common.utils import str_to_bool
from assume.units.dsm_load_shift import DSMFlex

logger = logging.getLogger(__name__)


class SteamPlant(DSMFlex, SupportsMinMax):
    """
    Represents a paper and pulp plant in an energy system. This includes components like heat pumps,
    boilers, and storage units for operational optimization.

    Args:
        id (str): Unique identifier for the paper and pulp plant.
        unit_operator (str): The operator responsible for the plant.
        bidding_strategies (dict): A dictionary of bidding strategies that define how the plant participates in energy markets.
        forecaster (Forecaster): A forecaster used to get key variables such as fuel or electricity prices.
        components (dict, optional): A dictionary describing the components of the plant, such as heat pumps and boilers.
        objective (str, optional): The objective function of the plant, typically to minimize variable costs. Default is "min_variable_cost".
        flexibility_measure (str, optional): The flexibility measure used for the plant, such as "max_load_shift". Default is "max_load_shift".
        demand (float, optional): The production demand, representing how much product needs to be produced. Default is 0.
        cost_tolerance (float, optional): The maximum allowable increase in cost when shifting load. Default is 10.
        node (str, optional): The node location where the plant is connected within the energy network. Default is "node0".
        location (tuple[float, float], optional): The geographical coordinates (latitude, longitude) of the paper and pulp plant. Default is (0.0, 0.0).
        **kwargs: Additional keyword arguments to support more specific configurations or parameters.
    """

    required_technologies = []
    optional_technologies = [
        "heat_pump",
        "boiler",
        "thermal_storage",
        "heat_resistor",
        "pv_plant",
    ]

    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        forecaster: Forecaster,
        components: dict[str, dict] = None,
        technology: str = "steam_generator_plant",
        objective: str = "min_variable_cost",
        flexibility_measure: str = "max_load_shift",
        is_prosumer: str = "No",
        demand: float = 0,
        cost_tolerance: float = 10,
        congestion_threshold: float = 0,
        peak_load_cap: float = 0,
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

        # Check if the required components are present in the components dictionary
        for component in self.required_technologies:
            if component not in components.keys():
                raise ValueError(
                    f"Component {component} is required for the steam generator plant unit."
                )

        # Check if the provided components are valid and do not contain any unknown components
        for component in components.keys():
            if (
                component not in self.required_technologies
                and component not in self.optional_technologies
            ):
                raise ValueError(
                    f"Component {component} is not a valid component for the steam generator plant unit."
                )
        # Check for the presence of components first
        self.has_thermal_storage = "thermal_storage" in self.components.keys()
        self.has_heat_resistor = "heat_resistor" in self.components.keys()
        self.has_boiler = "boiler" in self.components.keys()
        self.has_heatpump = "heat_pump" in self.components.keys()
        self.has_pv = "pv_plant" in self.components.keys()

        # Configure PV plant power profile based on availability
        if self.has_pv:
            profile_key = (
                f"{self.id}_pv_power_profile"
                if not str_to_bool(
                    self.components["pv_plant"].get("uses_power_profile", "false")
                )
                else "availability_solar"
            )
            pv_profile = self.forecaster[profile_key]
            # Assign the aligned profile
            self.components["pv_plant"][
                "power_profile"
                if profile_key.endswith("power_profile")
                else "availability_profile"
            ] = pv_profile

        # Inject schedule into long-term thermal storage if applicable
        if "thermal_storage" in self.components:
            storage_cfg = self.components["thermal_storage"]
            storage_type = storage_cfg.get("storage_type", "short-term")
            if storage_type == "long-term":
                schedule_key = f"{self.id}_thermal_storage_schedule"
                schedule_series = self.forecaster[schedule_key]
                storage_cfg["storage_schedule_profile"] = schedule_series

        # Add price forecasts
        self.electricity_price = self.forecaster["PL_B"]
        self.electricity_price_flex = self.forecaster["PL_B"]
        if self.has_boiler and self.components["boiler"]["fuel_type"] == "natural_gas":
            self.natural_gas_price = self.forecaster["natural_gas_price"]
        if self.has_boiler and self.components["boiler"]["fuel_type"] == "hydrogen_gas":
            self.hydrogen_gas_price = self.forecaster["hydrogen_gas_price"]
        self.demand = demand
        self.thermal_demand = self.forecaster[f"{self.id}_thermal_demand"]
        self.congestion_signal = self.forecaster[f"{self.id}_congestion_signal"]
        self.renewable_utilisation_signal = self.forecaster["availability_solar"]
        self.congestion_threshold = congestion_threshold
        self.peak_load_cap = peak_load_cap
        self.objective = objective
        self.flexibility_measure = flexibility_measure
        self.cost_tolerance = cost_tolerance
        self.is_prosumer = str_to_bool(is_prosumer)

        if self.is_prosumer:
            # ---- Built-in FCR configuration (hardcoded German FCR) ----
            # Active only if self.is_prosumer is True
            self.fcr_enabled = bool(self.is_prosumer)
            self.fcr_symmetric = False          # set True if you want symmetric product by default
            self._FCR_BLOCK_LENGTH = 4          # hours, fixed TSO blocks
            self._FCR_MIN_BID_MW   = 1.0        # minimum capacity per block
            self._FCR_STEP_MW      = 1.0        # increment: bids in integer MW
            # expected FCR/aFRR value (€/MW/h) as hourly series; if not provided, assume zeros
            self.fcr_price = self.forecaster["DE_fcr_price"]
            self.afrr_price_pos = self.forecaster["DE_pos_price"]
            self.afrr_price_neg = self.forecaster["DE_neg_price"]

        # Initialize the model
        self.setup_model()

    def define_parameters(self):
        """
        Defines the parameters for the Pyomo model.
        """
        self.model.electricity_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.electricity_price)},
        )

        if self.has_boiler and self.components["boiler"]["fuel_type"] == "natural_gas":
            self.model.natural_gas_price = pyo.Param(
                self.model.time_steps,
                initialize={t: value for t, value in enumerate(self.natural_gas_price)},
            )

        if self.has_boiler and self.components["boiler"]["fuel_type"] == "hydrogen_gas":
            self.model.hydrogen_gas_price = pyo.Param(
                self.model.time_steps,
                initialize={
                    t: value for t, value in enumerate(self.hydrogen_gas_price)
                },
            )

        self.model.absolute_demand = pyo.Param(initialize=float(self.demand or 0.0))
        self.model.thermal_demand = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.thermal_demand)},
        )

    def define_variables(self):
        """
        Defines the decision variables for the optimization model.
        """
        self.model.cumulative_thermal_output = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.total_power_input = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.variable_cost = pyo.Var(self.model.time_steps, within=pyo.Reals)

        # Electric supply split
        if self.has_pv:
            # PV energy actually used on-site for electric loads [MW_e]
            self.model.pv_used = pyo.Var(
                self.model.time_steps, within=pyo.NonNegativeReals
            )
            self.model.pv_curtail = pyo.Var(
                self.model.time_steps, within=pyo.NonNegativeReals
            )

        self.model.grid_power = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )

    def initialize_process_sequence(self):
        # --- PV availability cap: pv_used <= PV generation ---
        if self.has_pv:

            @self.model.Constraint(self.model.time_steps)
            def pv_use_cap(m, t):
                return m.pv_used[t] == m.dsm_blocks["pv_plant"].power[t]

            # @self.model.Constraint(self.model.time_steps)
            # def pv_power_partition(m, t):
            #     # PV generation = pv_used + curtailment   (no export case)
            #     return m.dsm_blocks["pv_plant"].power[t] == m.pv_used[t] + m.pv_curtail[t]

        # --- Electricity balance: PV_used + grid == electric loads ---
        @self.model.Constraint(self.model.time_steps)
        def electricity_balance(m, t):
            # supply
            supply = self.model.grid_power[t]
            if self.has_pv:
                supply += m.pv_used[t]

            # loads
            loads = 0
            if self.has_heatpump:
                loads += m.dsm_blocks["heat_pump"].power_in[t]
            if self.has_heat_resistor:
                loads += m.dsm_blocks["heat_resistor"].power_in[t]
            if self.has_boiler and self.components["boiler"].fuel_type == "electricity":
                loads += m.dsm_blocks["boiler"].power_in[t]
            if self.has_thermal_storage:
                # works for both modes: zero when not generator
                loads += m.dsm_blocks["thermal_storage"].power_in[t]

            return supply == loads

        @self.model.Constraint(self.model.time_steps)
        def thermal_bus_balance_and_demand(m, t):
            """
            Thermal bus balance:
            supply_to_load(t) = sum(heat_out of all producers)
                                + TES_discharge(t)
                                - TES_charge(t)  [only if TES draws heat from bus]

            Then enforce demand satisfaction against this supply.
            """
            # ---- Sum all producers dynamically
            supply = 0

            if self.has_boiler:
                supply += m.dsm_blocks["boiler"].heat_out[t]

            if self.has_heatpump:
                # assume HP 'heat_out' is on the same thermal bus
                supply += m.dsm_blocks["heat_pump"].heat_out[t]

            if self.has_heat_resistor:
                supply += m.dsm_blocks["heat_resistor"].heat_out[t]

            # if you have any other direct thermal producers (e.g., solar-thermal), add here:
            # if self.has_solar_thermal:
            #     supply += m.dsm_blocks["solar_thermal"].heat_out[t]

            # ---- TES coupling
            if self.has_thermal_storage:
                ts = m.dsm_blocks["thermal_storage"]
                # Always add discharge as usable heat to the bus
                supply += ts.discharge[t]

                # Subtract charge ONLY if TES is charged from the thermal bus.
                # In your new design, generator-mode TES is electric-charged internally,
                # so we should NOT subtract charge in that case.
                is_gen = getattr(
                    self.components["thermal_storage"], "is_generator_mode", None
                )
                # If the storage class sets a flag/param on the model block, prefer that:
                if hasattr(ts, "is_generator_mode"):
                    is_gen = bool(pyo.value(ts.is_generator_mode))
                # Fallback: infer from storage_type string stored on the Python-side component
                if isinstance(is_gen, (type(None),)):
                    stype = getattr(
                        self.components["thermal_storage"], "storage_type", ""
                    )
                    is_gen = stype == "short-term_with_generator"

                if not is_gen:
                    # heat-charged TES draws from the same bus → subtract
                    supply -= ts.charge[t]

            # ---- Bind the bus total to cumulative_thermal_output for plotting/consistency
            #     (If you already bind this elsewhere, replace '==' with pyo.Constraint.Skip)
            return m.cumulative_thermal_output[t] == supply

    def define_constraints(self):
        """
        Defines the constraints for the steam generation plant model.
        """
        # Only attach FCR if this plant is a prosumer
        if self.is_prosumer:
            self._add_fcr_capacity_market(self.model)

        @self.model.Constraint(self.model.time_steps)
        def meet_demand(m, t):
            """
            Enforce demand either per-step (default) or as absolute sum if self.demand is set.
            """
            if not self.demand or self.demand == 0:
                # per-step demand vector
                return m.cumulative_thermal_output[t] >= m.thermal_demand[t]
            else:
                # absolute demand over horizon
                return (
                    sum(m.cumulative_thermal_output[τ] for τ in m.time_steps)
                    >= m.absolute_demand
                )

        # Power input constraint
        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
            total_power = 0
            if self.has_heatpump:
                total_power += m.dsm_blocks["heat_pump"].power_in[t]
            if self.has_heat_resistor:
                total_power += m.dsm_blocks["heat_resistor"].power_in[t]
            if self.has_boiler and self.components["boiler"].fuel_type == "electricity":
                total_power += m.dsm_blocks["boiler"].power_in[t]
            if self.has_thermal_storage:
                total_power += m.dsm_blocks["thermal_storage"].power_in[
                    t
                ]  # zero when not generator
            return m.total_power_input[t] == total_power

        # Operating cost constraint
        @self.model.Constraint(self.model.time_steps)
        def cost_per_time_step(m, t):
            total_cost = 0
            # if self.has_heatpump:
            #     total_cost += m.dsm_blocks["heat_pump"].operating_cost[t]
            # if self.has_heat_resistor:
            #     total_cost += m.dsm_blocks["heat_resistor"].operating_cost[t]
            # if self.has_boiler:
            #     total_cost += m.dsm_blocks["boiler"].operating_cost[t]
            # if self.has_thermal_storage:
            #     total_cost += m.dsm_blocks["thermal_storage"].operating_cost[t]
            # if self.has_pv and hasattr(m.dsm_blocks["pv_plant"], "operating_cost"):
            #     total_cost += m.dsm_blocks["pv_plant"].operating_cost[
            #         t
            #     ]  # typically zero
            if self.has_boiler:
                boiler = self.components.get("boiler", None)
                fuel = getattr(boiler, "fuel_type", None) if boiler else None

                if fuel in {"natural_gas", "hydrogen"}:
                    # Only add if the block/var exists (guards against missing attrs)
                        total_cost += m.dsm_blocks["boiler"].operating_cost[t]

            # pay for grid imports
            total_cost += m.grid_power[t] * m.electricity_price[t]
            return m.variable_cost[t] == total_cost

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
        N = len(ts)  # number of time steps in the horizon

        # ---- Block prices: sum of hourly €/MW/h inside each 4h block ----
        price_hourly_fcr      = [0.0]*N if self.fcr_price is None      else [float(x) for x in self.fcr_price]
        price_hourly_afrr_pos = [0.0]*N if self.afrr_price_pos is None else [float(x) for x in self.afrr_price_pos]
        price_hourly_afrr_neg = [0.0]*N if self.afrr_price_neg is None else [float(x) for x in self.afrr_price_neg]

        def _block(v):
            ps = [0.0]
            for x in v: ps.append(ps[-1] + x)
            return {b: ps[b + L] - ps[b] for b in starts_idx}

        m.fcr_block_price       = pyo.Param(m.fcr_blocks, initialize=_block(price_hourly_fcr),      mutable=False)
        m.afrr_block_price_pos  = pyo.Param(m.fcr_blocks, initialize=_block(price_hourly_afrr_pos), mutable=False)
        m.afrr_block_price_neg  = pyo.Param(m.fcr_blocks, initialize=_block(price_hourly_afrr_neg), mutable=False)


        # ---- Static plant-wide min/max electric power envelope (headroom/footroom) ----
        max_cap = 0.0
        min_cap = 0.0
        if self.has_heatpump:
            max_cap += float(m.dsm_blocks["heat_pump"].max_power)
            min_cap += float(m.dsm_blocks["heat_pump"].min_power)
        if self.has_heat_resistor:
            max_cap += float(m.dsm_blocks["heat_resistor"].max_power)
            min_cap += float(m.dsm_blocks["heat_resistor"].min_power)
        if self.has_boiler:
            boiler = self.components["boiler"]
            if getattr(boiler, "fuel_type", None) == "electricity":
                max_cap += float(m.dsm_blocks["boiler"].max_power)
                min_cap += float(m.dsm_blocks["boiler"].min_power)
        if self.has_thermal_storage:
            max_cap += float(m.dsm_blocks["thermal_storage"].max_Pelec)
            min_cap += 0.0

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
                # up: need headroom; down: need footroom
                m.fcr_up_feas.add(   m.cap_up[b]   <= self.max_plant_capacity - m.total_power_input[t] )
                m.fcr_down_feas.add( m.cap_dn[b]   <= m.total_power_input[t] - self.min_plant_capacity )


        @ m.Expression()
        def fcr_revenue(mm):
            if self.fcr_symmetric:
                # up==down enforced elsewhere; pay once per MW of symmetric capacity
                return sum(mm.fcr_block_price[b] * mm.cap_up[b] for b in mm.fcr_blocks)
            else:
                # asymmetric: sum the two capacities, constraints will force one to zero if needed
                return (
                    sum(mm.afrr_block_price_pos[b] * mm.cap_up[b]   for b in mm.fcr_blocks) +
                    sum(mm.afrr_block_price_neg[b] * mm.cap_dn[b]   for b in mm.fcr_blocks)
                )


    ################### PLOT #######################
    def plot_1(self, instance, save_path=None, show=True):
        """
        Two-panel matplotlib figure:
        Top: unit inputs (electric & fossil) + energy prices (twin y-axis)
        Bottom: TES operation (charge, discharge, SOC)
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

        # ------- blocks -------
        B = instance.dsm_blocks
        hp = B["heat_pump"] if "heat_pump" in B else None
        hr = B["heat_resistor"] if "heat_resistor" in B else None
        br = B["boiler"] if "boiler" in B else None
        ts = B["thermal_storage"] if "thermal_storage" in B else None
        pv = B["pv_plant"] if "pv_plant" in B else None

        # ------- unit inputs -------
        hp_P = series_or_none(hp, "power_in")  # MW_e
        hp_Q = series_or_none(hp, "heat_out")  # MW_th

        hr_P = series_or_none(hr, "power_in")  # MW_e
        hr_Q = series_or_none(hr, "heat_out")  # MW_th

        # boiler can be fossil or electric — guard all
        eb_P = series_or_none(br, "power_in")  # MW_e (e-boiler mode)
        br_Q = series_or_none(br, "heat_out")  # MW_th
        br_NG = series_or_none(br, "natural_gas_in")  # MW_th fuel
        br_H2 = series_or_none(br, "hydrogen_in")  # MW_th fuel
        br_CO = series_or_none(br, "coal_in")  # MW_th fuel (if applicable)

        # PV & grid (electric supply)
        pv_P = series_or_none(pv, "power")  # MW_e
        grid = (
            [safe_value(instance.grid_power[t]) for t in T]
            if hasattr(instance, "grid_power")
            else None
        )

        # TES
        ts_ch = series_or_none(ts, "charge")  # MW_th
        ts_ds = series_or_none(ts, "discharge")  # MW_th
        ts_soc = series_or_none(ts, "soc")  # MWh_th
        ts_P = series_or_none(ts, "power_in")  # MW_e (0 if not generator mode)

        # prices
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
        h2_price = (
            [safe_value(instance.hydrogen_price[t]) for t in T]
            if hasattr(instance, "hydrogen_price")
            else None
        )
        variable_cost = (
            [safe_value(instance.variable_cost[t]) for t in T]
            if hasattr(instance, "variable_cost")
            else None
        )

        # ------- figure -------
        fig, axs = plt.subplots(
            2, 1, figsize=(10, 8), sharex=True, constrained_layout=True
        )

        # ---------- TOP: inputs + prices ----------
        # Electric loads
        l_hpP = plot_if_nonzero(axs[0], T, hp_P, "Heat Pump Power [MWₑ]", "C1")
        l_hrP = plot_if_nonzero(axs[0], T, hr_P, "Heat Resistor Power [MWₑ]", "C5")
        l_ebP = plot_if_nonzero(axs[0], T, eb_P, "E-Boiler Power [MWₑ]", "C11")
        l_tsP = plot_if_nonzero(axs[0], T, ts_P, "TES Heater Power [MWₑ]", "C19", "-.")

        # Fossil inputs to boiler (fuel power)
        l_ng = plot_if_nonzero(axs[0], T, br_NG, "Boiler NG [MWₜₕ]", "C2")
        l_h2 = plot_if_nonzero(axs[0], T, br_H2, "Boiler H₂ [MWₜₕ]", "C3", "--")
        l_co = plot_if_nonzero(axs[0], T, br_CO, "Boiler Coal [MWₜₕ]", "C7", "-.")

        # Supply lines (optional)
        l_pv = plot_if_nonzero(axs[0], T, pv_P, "PV [MWₑ]", "C4")
        l_gr = plot_if_nonzero(axs[0], T, grid, "Grid Import [MWₑ]", "C0", ":")

        axs[0].set_ylabel("Inputs")
        axs[0].set_title("Unit Inputs & Energy Prices")
        axs[0].grid(True, which="both", axis="both")

        # Prices (twin y-axis)
        axp = axs[0].twinx()
        lp = []
        if elec_price:
            lp += axp.plot(
                T, elec_price, label="Elec Price [€/MWhₑ]", color="C6", linestyle="--"
            )
        if ng_price:
            lp += axp.plot(
                T, ng_price, label="NG Price [€/MWhₜₕ]", color="C8", linestyle=":"
            )
        if h2_price:
            lp += axp.plot(
                T, h2_price, label="H₂ Price [€/MWhₜₕ]", color="C9", linestyle="-."
            )

        axp.set_ylabel("Energy Price", color="gray")
        axp.tick_params(axis="y", labelcolor="gray")

        # Legend (only existing handles)
        handles = [
            h
            for h in [l_hpP, l_hrP, l_ebP, l_tsP, l_ng, l_h2, l_co, l_pv, l_gr]
            if h is not None
        ] + lp
        labels = [h.get_label() for h in handles]
        if handles:
            axs[0].legend(handles, labels, loc="upper left", frameon=True)

        # ---------- BOTTOM: TES operation ----------
        p0 = plot_if_nonzero(axs[1], T, ts_ch, "TES Charge [MWₜₕ]", "C2", "-")
        p1 = plot_if_nonzero(axs[1], T, ts_ds, "TES Discharge [MWₜₕ]", "C3", "--")
        if ts_soc and any(abs(v) > 1e-9 for v in ts_soc):
            axs[1].fill_between(
                T, ts_soc, 0, color="C0", alpha=0.25, label="TES SOC [MWhₜₕ]"
            )

        axs[1].set_ylabel("MWₜₕ / MWhₜₕ")
        axs[1].set_title("Thermal Storage Operation")
        axs[1].grid(True, which="both", axis="both")
        blines = [h for h in [p0, p1] if h is not None]
        if blines:
            axs[1].legend(loc="upper right", frameon=True)
        axs[1].set_xlabel("Time step")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


        # -------- optional CSV export (same columns as plotted) --------
        df = pd.DataFrame(
            {
                "t": T,
                "HP Power [MW_e]": hp_P if hp_P else None,
                "Resistor Power [MW_e]": hr_P if hr_P else None,
                "E-Boiler Power [MW_e]": eb_P if eb_P else None,
                "TES Heater Power [MW_e]": ts_P if ts_P else None,
                "Boiler NG [MW_th]": br_NG if br_NG else None,
                "Boiler H2 [MW_th]": br_H2 if br_H2 else None,
                "Boiler Coal [MW_th]": br_CO if br_CO else None,
                "PV [MW_e]": pv_P if pv_P else None,
                "Grid Import [MW_e]": grid if grid else None,
                "TES Charge [MW_th]": ts_ch if ts_ch else None,
                "TES Discharge [MW_th]": ts_ds if ts_ds else None,
                "TES SOC [MWh_th]": ts_soc if ts_soc else None,
                "Elec Price [€/MWh_e]": elec_price if elec_price else None,
                "NG Price [€/MWh_th]": ng_price if ng_price else None,
                "H2 Price [€/MWh_th]": h2_price if h2_price else None,
                "Total Variable Cost [€]": variable_cost if variable_cost else None,
            }
        )
        # example: write next to figure if save_path is given
        if save_path:
            csv_path = save_path.rsplit(".", 1)[0] + ".csv"
            df.to_csv(csv_path, index=False)


    def plot_2(self, instance, save_path=None, show=True):
        """
        Three-panel figure:
        1) Unit inputs (electric & fossil) + energy prices (twin y-axis)
        2) TES operation (charge, discharge, SOC)
        3) FCR market view (per 4h block): capacities (up/down or symmetric), block price,
            and operational baseline & headroom/footroom for feasibility context

        Presence-aware: draws only what exists in `instance`.
        Also writes a CSV next to the figure if `save_path` is given.
        """

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
            Expand block-level value (constant over 4h) to hourly stair array (len=horizon_len).
            block_starts: iterable of integer start indices
            per_block_values: dict {b -> value}
            Returns: list[float] of length horizon_len
            """
            y = [0.0] * horizon_len
            for b in block_starts:
                val = safe_value(per_block_values.get(b, 0.0))
                for k in range(block_len):
                    idx = b + k
                    if 0 <= idx < horizon_len:
                        y[idx] = val
            return y

        # ------- blocks -------
        B = instance.dsm_blocks
        hp = B["heat_pump"] if "heat_pump" in B else None
        hr = B["heat_resistor"] if "heat_resistor" in B else None
        br = B["boiler"] if "boiler" in B else None
        ts = B["thermal_storage"] if "thermal_storage" in B else None
        pv = B["pv_plant"] if "pv_plant" in B else None

        # ------- unit inputs -------
        hp_P = series_or_none(hp, "power_in")    # MW_e
        hr_P = series_or_none(hr, "power_in")    # MW_e
        eb_P = series_or_none(br, "power_in")    # MW_e (if e-boiler)
        br_Q = series_or_none(br, "heat_out")    # MW_th (boiler heat)
        br_NG = series_or_none(br, "natural_gas_in")
        br_H2 = series_or_none(br, "hydrogen_in")
        br_CO = series_or_none(br, "coal_in")

        pv_P = series_or_none(pv, "power")       # MW_e
        grid = [safe_value(instance.grid_power[t]) for t in T] if hasattr(instance, "grid_power") else None

        # TES
        ts_ch  = series_or_none(ts, "charge")    # MW_th
        ts_ds  = series_or_none(ts, "discharge") # MW_th
        ts_soc = series_or_none(ts, "soc")       # MWh_th
        ts_P   = series_or_none(ts, "power_in")  # MW_e (if generator mode)

        # prices
        elec_price = [safe_value(instance.electricity_price[t]) for t in T] if hasattr(instance, "electricity_price") else None
        ng_price   = [safe_value(instance.natural_gas_price[t]) for t in T] if hasattr(instance, "natural_gas_price") else None
        h2_price   = [safe_value(instance.hydrogen_price[t]) for t in T] if hasattr(instance, "hydrogen_price") else None

        # baseline load and envelope (for FCR feasibility/impact visualization)
        total_elec = [safe_value(instance.total_power_input[t]) for t in T] if hasattr(instance, "total_power_input") else None
        max_cap = getattr(self, "max_plant_capacity", None)  # float or None
        min_cap = getattr(self, "min_plant_capacity", 0.0)

        # ------- figure -------
        nrows = 3
        fig, axs = plt.subplots(nrows, 1, figsize=(12, 10), sharex=True, constrained_layout=True)

        # ---------- TOP: inputs + prices ----------
        l_hpP = plot_if_nonzero(axs[0], T, hp_P, "Heat Pump Power [MWₑ]", "C1")
        l_hrP = plot_if_nonzero(axs[0], T, hr_P, "Heat Resistor Power [MWₑ]", "C5")
        l_ebP = plot_if_nonzero(axs[0], T, eb_P, "E-Boiler Power [MWₑ]", "C11")
        l_tsP = plot_if_nonzero(axs[0], T, ts_P, "TES Heater Power [MWₑ]", "C19", "-.")

        l_ng  = plot_if_nonzero(axs[0], T, br_NG, "Boiler NG [MWₜₕ]", "C2")
        l_h2  = plot_if_nonzero(axs[0], T, br_H2, "Boiler H₂ [MWₜₕ]", "C3", "--")
        l_co  = plot_if_nonzero(axs[0], T, br_CO, "Boiler Coal [MWₜₕ]", "C7", "-.")

        l_pv  = plot_if_nonzero(axs[0], T, pv_P, "PV [MWₑ]", "C4")
        l_gr  = plot_if_nonzero(axs[0], T, grid, "Grid Import [MWₑ]", "C0", ":")

        axs[0].set_ylabel("Inputs")
        axs[0].set_title("Unit Inputs & Energy Prices")
        axs[0].grid(True, which="both", axis="both")

        axp = axs[0].twinx()
        lp = []
        if elec_price:
            lp += axp.plot(T, elec_price, label="Elec Price [€/MWhₑ]", color="C6", linestyle="--")
        if ng_price:
            lp += axp.plot(T, ng_price, label="NG Price [€/MWhₜₕ]", color="C8", linestyle=":")
        if h2_price:
            lp += axp.plot(T, h2_price, label="H₂ Price [€/MWhₜₕ]", color="C9", linestyle="-.")
        axp.set_ylabel("Energy Price", color="gray")
        axp.tick_params(axis="y", labelcolor="gray")

        handles = [h for h in [l_hpP, l_hrP, l_ebP, l_tsP, l_ng, l_h2, l_co, l_pv, l_gr] if h is not None] + lp
        if handles:
            axs[0].legend(handles, [h.get_label() for h in handles], loc="upper left", frameon=True)

        # ---------- MIDDLE: TES operation ----------
        p0 = plot_if_nonzero(axs[1], T, ts_ch, "TES Charge [MWₜₕ]", "C2", "-")
        p1 = plot_if_nonzero(axs[1], T, ts_ds, "TES Discharge [MWₜₕ]", "C3", "--")
        if ts_soc and any(abs(v) > 1e-9 for v in ts_soc):
            axs[1].fill_between(T, ts_soc, 0, color="C0", alpha=0.25, label="TES SOC [MWhₜₕ]")

        axs[1].set_ylabel("MWₜₕ / MWhₜₕ")
        axs[1].set_title("Thermal Storage Operation")
        axs[1].grid(True, which="both", axis="both")
        if any([p0, p1]):
            axs[1].legend(loc="upper right", frameon=True)

        # ---------- BOTTOM: Reserve bids & operational impact ----------
        fcr_present = hasattr(instance, "fcr_blocks")
        fcr_len = getattr(self, "_FCR_BLOCK_LENGTH", 4)
        is_symmetric = bool(getattr(self, "fcr_symmetric", False))

        def zero_or_list(x, n):
            return [0.0] * n if x is None else x

        fcr_blocks_list = list(getattr(instance, "fcr_blocks", [])) if fcr_present else []
        cap_up_map, cap_dn_map = {}, {}
        # price maps (one for symmetric FCR; two for asymmetric aFRR)
        price_map_sym = {}
        price_map_pos = {}
        price_map_neg = {}

        if fcr_present and fcr_blocks_list:
            # capacities
            for b in fcr_blocks_list:
                if hasattr(instance, "cap_up"):
                    cap_up_map[b] = safe_value(instance.cap_up[b])
                if hasattr(instance, "cap_dn"):
                    cap_dn_map[b] = safe_value(instance.cap_dn[b])

            # prices
            if is_symmetric:
                for b in fcr_blocks_list:
                    if hasattr(instance, "fcr_block_price"):
                        price_map_sym[b] = safe_value(instance.fcr_block_price[b])
            else:
                for b in fcr_blocks_list:
                    if hasattr(instance, "afrr_block_price_pos"):
                        price_map_pos[b] = safe_value(instance.afrr_block_price_pos[b])
                    if hasattr(instance, "afrr_block_price_neg"):
                        price_map_neg[b] = safe_value(instance.afrr_block_price_neg[b])

            # expand to hourly 'stairs'
            H = len(T)
            cap_up_stairs = stairs_from_blocks(fcr_blocks_list, cap_up_map, fcr_len, H) if cap_up_map else [0.0]*H
            cap_dn_stairs = stairs_from_blocks(fcr_blocks_list, cap_dn_map, fcr_len, H) if cap_dn_map else [0.0]*H
            price_stairs_sym = stairs_from_blocks(fcr_blocks_list, price_map_sym, fcr_len, H) if price_map_sym else None
            price_stairs_pos = stairs_from_blocks(fcr_blocks_list, price_map_pos, fcr_len, H) if price_map_pos else None
            price_stairs_neg = stairs_from_blocks(fcr_blocks_list, price_map_neg, fcr_len, H) if price_map_neg else None

            # Baseline & capability envelope
            base = zero_or_list(total_elec, len(T))
            axs[2].plot(T, base, color="0.3", lw=2.0, label="Baseline Elec Load [MWₑ]")
            if max_cap is not None:
                axs[2].plot(T, [max_cap]*len(T), color="0.7", ls=":", label="Max Capability [MWₑ]")
            if min_cap is not None:
                axs[2].plot(T, [min_cap]*len(T), color="0.7", ls="--", label="Min Capability [MWₑ]")

            # Capacity bands
            lower_up = [max(min_cap if min_cap is not None else -1e9, base[i] - cap_up_stairs[i]) for i in range(len(T))]
            axs[2].fill_between(T, lower_up, base, color="#d62728", alpha=0.25, step="pre", label="UP Capacity [MW]")

            upper_dn = [min(max_cap if max_cap is not None else 1e9, base[i] + cap_dn_stairs[i]) for i in range(len(T))]
            axs[2].fill_between(T, base, upper_dn, color="#2ca02c", alpha=0.25, step="pre", label="DOWN Capacity [MW]")

            # Prices on twin axis
            axp2 = axs[2].twinx()
            if is_symmetric and price_stairs_sym is not None:
                axp2.plot(T, price_stairs_sym, color="#9467bd", ls="--", lw=2,
                          label="FCR Block Price [€/MW·4h]")
            else:
                if price_stairs_pos is not None:
                    axp2.plot(T, price_stairs_pos, color="#1f77b4", ls="--", lw=2,
                              label="aFRR+ Price [€/MW·4h]")
                if price_stairs_neg is not None:
                    axp2.plot(T, price_stairs_neg, color="#ff7f0e", ls="--", lw=2,
                              label="aFRR− Price [€/MW·4h]")

            axp2.set_ylabel("Reserve Price", color="gray")
            axp2.tick_params(axis="y", labelcolor="gray")

            # Shade active blocks
            for b in fcr_blocks_list:
                active = (cap_up_map.get(b, 0.0) > 0.0) or (cap_dn_map.get(b, 0.0) > 0.0)
                if active:
                    axs[2].axvspan(b, min(b + fcr_len, len(T)), color="k", alpha=0.04)

            # Legend from both axes
            h1, l1 = axs[2].get_legend_handles_labels()
            h2, l2 = axp2.get_legend_handles_labels()
            axs[2].legend(h1 + h2, l1 + l2, loc="upper left", frameon=True)

            axs[2].set_ylabel("MW / MWₑ")
            axs[2].set_title(
                "Reserve Bids (4h blocks) — Red=UP, Green=DOWN; prices on right axis"
                if not is_symmetric else
                "FCR Bids (4h blocks) — Red=UP, Green=DOWN; FCR price on right axis"
            )
            axs[2].grid(True, which="both", axis="both")

        else:
            _ = plot_if_nonzero(axs[2], T, total_elec, "Baseline Elec Load [MWₑ]", "C1", "-")
            if max_cap is not None:
                axs[2].plot(T, [max_cap]*len(T), color="0.7", ls=":", label="Max Capability [MWₑ]")
            if min_cap is not None:
                axs[2].plot(T, [min_cap]*len(T), color="0.7", ls="--", label="Min Capability [MWₑ]")
            axs[2].set_ylabel("MW / MWₑ")
            axs[2].set_title("Operational Baseline (no reserve structures on instance)")
            axs[2].grid(True, which="both", axis="both")
            axs[2].legend(loc="upper left", frameon=True)


        axs[-1].set_xlabel("Time step")
        fig.autofmt_xdate()  # if your time steps are datetime-indexed

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

        # -------- CSV export (includes FCR columns if present) --------
        df = pd.DataFrame({
            "t": T,
            "HP Power [MW_e]": hp_P if hp_P else None,
            "Resistor Power [MW_e]": hr_P if hr_P else None,
            "E-Boiler Power [MW_e]": eb_P if eb_P else None,
            "TES Heater Power [MW_e]": ts_P if ts_P else None,
            "Boiler NG [MW_th]": br_NG if br_NG else None,
            "Boiler H2 [MW_th]": br_H2 if br_H2 else None,
            "Boiler Coal [MW_th]": br_CO if br_CO else None,
            "PV [MW_e]": pv_P if pv_P else None,
            "Grid Import [MW_e]": grid if grid else None,
            "TES Charge [MW_th]": ts_ch if ts_ch else None,
            "TES Discharge [MW_th]": ts_ds if ts_ds else None,
            "TES SOC [MWh_th]": ts_soc if ts_soc else None,
            "Elec Price [€/MWh_e]": elec_price if elec_price else None,
            "NG Price [€/MWh_th]": ng_price if ng_price else None,
            "H2 Price [€/MWh_th]": h2_price if h2_price else None,
            "Total Elec Load [MW_e]": total_elec if total_elec else None,
        })

                # --- CSV export additions for reserve prices ---
        if fcr_present and fcr_blocks_list:
            # ensure these names exist if you want them in the CSV scope
            try:
                df["Reserve UP Cap [MW]"] = cap_up_stairs
                df["Reserve DOWN Cap [MW]"] = cap_dn_stairs
            except NameError:
                pass

            if is_symmetric:
                try:
                    df["FCR Block Price [€/MW·4h]"] = price_stairs_sym
                except NameError:
                    pass
            else:
                try:
                    if price_stairs_pos is not None:
                        df["aFRR+ Price [€/MW·4h]"] = price_stairs_pos
                    if price_stairs_neg is not None:
                        df["aFRR− Price [€/MW·4h]"] = price_stairs_neg
                except NameError:
                    pass


        if save_path:
            csv_path = save_path.rsplit(".", 1)[0] + ".csv"
            df.to_csv(csv_path, index=False)


    def dashboard(
        self,
        instance,
        baseline_instance=None,  # optional: BAU/no-TES run for comparison
        html_path="./outputs/steam_full_dashboard.html",
        sankey_max_steps=168,  # limit time-slider sankey to avoid huge HTML; set None for all
    ):
        """
        One-stop interactive dashboard for the steam plant:
        • Dynamic operation (electric & thermal balances)
        • Time-slider Sankey (per-step flows)
        • Storage analytics (duration curve, FIFO delivered-cost vs. displaced cost)
        • Emissions analytics (scope-1/2, intensity)
        • Economics analytics (cost stack, LCOH, variable-cost time series)
        Works with any subset of: heat_pump, boiler, heat_resistor, pv_plant, thermal_storage.
        """

        # ------------------------- helpers -------------------------
        def safe(v):
            try:
                return float(pyo.value(v))
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

        def total(a):
            return float(np.nansum(a)) if a is not None else 0.0

        T = list(instance.time_steps)
        dt = 1.0  # hour per time-step (adjust if different)

        B = instance.dsm_blocks
        hp = B["heat_pump"] if "heat_pump" in B else None
        br = B["boiler"] if "boiler" in B else None
        hr = B["heat_resistor"] if "heat_resistor" in B else None
        pv = B["pv_plant"] if "pv_plant" in B else None
        ts = B["thermal_storage"] if "thermal_storage" in B else None

        # ------------------------- time series -------------------------
        # electricity price & fuels
        elec_price = vec_param(instance, "electricity_price")
        ng_price = vec_param(instance, "natural_gas_price")
        h2_price = vec_param(instance, "hydrogen_price")
        coal_price = vec_param(instance, "coal_price")

        # demand profile
        demand_th = vec_param(instance, "thermal_demand")

        # electric supply
        pv_P = series_or_none(pv, "power")  # MW_e
        grid = (
            [safe(instance.grid_power[t]) for t in T]
            if hasattr(instance, "grid_power")
            else [0.0] * len(T)
        )

        # electric loads
        hp_P = series_or_none(hp, "power_in")
        hr_P = series_or_none(hr, "power_in")
        eb_P = (
            series_or_none(br, "power_in") if (br and hasattr(br, "power_in")) else None
        )  # e-boiler
        ts_P = series_or_none(ts, "power_in")  # 0 unless generator mode

        # thermal outputs
        hp_Q = series_or_none(hp, "heat_out")
        hr_Q = series_or_none(hr, "heat_out")
        br_Q = series_or_none(br, "heat_out")

        # boiler fuel inputs (MW_th)
        br_NG = series_or_none(br, "natural_gas_in")
        br_H2 = series_or_none(br, "hydrogen_in")
        br_CO = series_or_none(br, "coal_in")

        # TES
        ts_ch = series_or_none(ts, "charge")  # MW_th
        ts_ds = series_or_none(ts, "discharge")  # MW_th
        ts_soc = series_or_none(ts, "soc")  # MWh_th
        is_gen_param = getattr(ts, "is_generator_mode", None)
        is_gen_mode = (
            bool(safe(is_gen_param))
            if is_gen_param is not None
            else (
                getattr(self.components.get("thermal_storage", {}), "storage_type", "")
                == "short-term_with_generator"
            )
        )

        # efficiencies / COP (fallbacks if not attached on blocks)
        # Boiler efficiency for fuel → heat
        eta_boiler = safe(getattr(br, "eta_fossil", 0.90)) if br else 0.90
        # Heat pump COP (electric → heat)
        hp_cop = safe(getattr(hp, "cop", 3.0)) if hp else 3.0
        # TES round-trip efficiency (charge→discharge). If not explicit, infer from charge/discharge params
        eta_c = safe(getattr(ts, "eta_charge", 0.95)) if ts else 0.95
        eta_d = safe(getattr(ts, "eta_discharge", 0.95)) if ts else 0.95
        eta_rt = eta_c * eta_d

        # ------------------------- OPERATION PANELS -------------------------
        # (1) Electric balance + prices
        fig_op = ps.make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            subplot_titles=(
                "Electric balance (PV + Grid vs. Electric Loads)",
                "Thermal balance (Producers + TES vs Demand)",
                "Price signals",
            ),
        )

        # Row 1: bars for supply; lines for loads
        if exists(pv_P):
            fig_op.add_trace(
                go.Bar(x=T, y=pv_P, name="PV [MWₑ]", marker=dict(opacity=0.85)),
                row=1,
                col=1,
            )
        if exists(grid):
            fig_op.add_trace(
                go.Bar(x=T, y=grid, name="Grid [MWₑ]", marker=dict(opacity=0.85)),
                row=1,
                col=1,
            )

        def zero_if_none(a):
            return a if a is not None else [0.0] * len(T)

        load_sum = [
            zero_if_none(hp_P)[i]
            + zero_if_none(hr_P)[i]
            + zero_if_none(eb_P)[i]
            + zero_if_none(ts_P)[i]
            for i in range(len(T))
        ]
        fig_op.add_trace(
            go.Scatter(
                x=T, y=load_sum, name="Electric loads total [MWₑ]", mode="lines"
            ),
            row=1,
            col=1,
        )
        if exists(hp_P):
            fig_op.add_trace(
                go.Scatter(x=T, y=hp_P, name="HP power [MWₑ]", mode="lines"),
                row=1,
                col=1,
            )
        if exists(hr_P):
            fig_op.add_trace(
                go.Scatter(x=T, y=hr_P, name="Resistor power [MWₑ]", mode="lines"),
                row=1,
                col=1,
            )
        if exists(eb_P):
            fig_op.add_trace(
                go.Scatter(x=T, y=eb_P, name="E-boiler power [MWₑ]", mode="lines"),
                row=1,
                col=1,
            )
        if exists(ts_P):
            fig_op.add_trace(
                go.Scatter(x=T, y=ts_P, name="TES heater power [MWₑ]", mode="lines"),
                row=1,
                col=1,
            )

        # Row 2: thermal producers stacked; TES charge as negative; demand line
        prod_hp = zero_if_none(hp_Q)
        prod_hr = zero_if_none(hr_Q)
        prod_br = zero_if_none(br_Q)
        prod_ds = zero_if_none(ts_ds)
        if exists(hp_Q):
            fig_op.add_trace(
                go.Bar(x=T, y=prod_hp, name="HP heat [MWₜₕ]"), row=2, col=1
            )
        if exists(hr_Q):
            fig_op.add_trace(
                go.Bar(x=T, y=prod_hr, name="Resistor heat [MWₜₕ]"), row=2, col=1
            )
        if exists(br_Q):
            fig_op.add_trace(
                go.Bar(x=T, y=prod_br, name="Boiler heat [MWₜₕ]"), row=2, col=1
            )
        if exists(ts_ds):
            fig_op.add_trace(
                go.Bar(x=T, y=prod_ds, name="TES discharge [MWₜₕ]"), row=2, col=1
            )
        if exists(ts_ch):
            fig_op.add_trace(
                go.Bar(x=T, y=[-v for v in ts_ch], name="TES charge (−) [MWₜₕ]"),
                row=2,
                col=1,
            )
        if exists(demand_th):
            fig_op.add_trace(
                go.Scatter(x=T, y=demand_th, name="Demand [MWₜₕ]", mode="lines"),
                row=2,
                col=1,
            )

        # Row 3: prices
        if exists(elec_price):
            fig_op.add_trace(
                go.Scatter(x=T, y=elec_price, name="Elec price [€/MWhₑ]", mode="lines"),
                row=3,
                col=1,
            )
        if exists(ng_price):
            fig_op.add_trace(
                go.Scatter(x=T, y=ng_price, name="NG price [€/MWhₜₕ]", mode="lines"),
                row=3,
                col=1,
            )
        if exists(h2_price):
            fig_op.add_trace(
                go.Scatter(x=T, y=h2_price, name="H₂ price [€/MWhₜₕ]", mode="lines"),
                row=3,
                col=1,
            )
        if exists(coal_price):
            fig_op.add_trace(
                go.Scatter(
                    x=T, y=coal_price, name="Coal price [€/MWhₜₕ]", mode="lines"
                ),
                row=3,
                col=1,
            )

        fig_op.update_layout(barmode="stack", height=920, title="Dynamic Operation")
        fig_op.update_yaxes(title_text="MWₑ", row=1, col=1)
        fig_op.update_yaxes(title_text="MWₜₕ", row=2, col=1)
        fig_op.update_yaxes(title_text="€/MWh", row=3, col=1)

        # ------------------------- TIME-SLIDER SANKEY -------------------------
        # Build a per-step Sankey (limit to sankey_max_steps for file size)
        maxN = len(T) if sankey_max_steps is None else min(len(T), sankey_max_steps)
        steps = list(range(maxN))

        # Node registry
        nodes = [
            "Electricity",
            "Natural Gas",
            "Hydrogen",
            "Coal",
            "PV",
            "Grid",
            "Heat Pump",
            "Heat Resistor",
            "E-Boiler",
            "TES (discharge)",
            "TES (charge)",
            "Thermal Bus",
            "Demand",
        ]
        nid = {n: i for i, n in enumerate(nodes)}

        sankey_traces = []
        for k in steps:
            links_src = []
            links_tgt = []
            links_val = []
            links_lbl = []

            def add(s, t, v, l):
                if v > 1e-9:
                    links_src.append(nid[s])
                    links_tgt.append(nid[t])
                    links_val.append(float(v))
                    links_lbl.append(l)

            # electricity to loads (MW_e treated as same units for visualization)
            if exists(hp_P):
                add("Electricity", "Heat Pump", hp_P[k], "E→HP [MWₑ]")
            if exists(hr_P):
                add("Electricity", "Heat Resistor", hr_P[k], "E→Res [MWₑ]")
            if exists(eb_P):
                add("Electricity", "E-Boiler", eb_P[k], "E→E-Boiler [MWₑ]")
            if exists(ts_P) and ts_P[k] > 1e-9:
                add("Electricity", "TES (charge)", ts_P[k], "E→TES heater [MWₑ]")

            # PV and Grid split (purely cosmetic, both feed the Electricity node)
            if exists(pv_P):
                add("PV", "Electricity", pv_P[k], "PV→E [MWₑ]")
            add("Grid", "Electricity", grid[k], "Grid→E [MWₑ]")

            # fuels to boiler (MW_th)
            if exists(br_NG):
                add("Natural Gas", "E-Boiler", br_NG[k], "NG→Boiler [MWₜₕ]")
            if exists(br_H2):
                add("Hydrogen", "E-Boiler", br_H2[k], "H₂→Boiler [MWₜₕ]")
            if exists(br_CO):
                add("Coal", "E-Boiler", br_CO[k], "Coal→Boiler [MWₜₕ]")

            # thermal bus: outputs of units + TES discharge
            q_hp = hp_Q[k] if hp_Q else 0.0
            q_hr = hr_Q[k] if hr_Q else 0.0
            q_br = br_Q[k] if br_Q else 0.0
            q_ts = ts_ds[k] if ts_ds else 0.0
            if q_hp > 1e-9:
                add("Heat Pump", "Thermal Bus", q_hp, "HP→Bus [MWₜₕ]")
            if q_hr > 1e-9:
                add("Heat Resistor", "Thermal Bus", q_hr, "Res→Bus [MWₜₕ]")
            if q_br > 1e-9:
                add("E-Boiler", "Thermal Bus", q_br, "Boiler→Bus [MWₜₕ]")
            if q_ts > 1e-9:
                add("TES (discharge)", "Thermal Bus", q_ts, "TES→Bus [MWₜₕ]")

            # TES heat-charge (if you model heat-charged mode visually)
            if ts_ch is not None and (not is_gen_mode) and ts_ch[k] > 1e-9:
                add("Thermal Bus", "TES (charge)", ts_ch[k], "Bus→TES [MWₜₕ]")

            # bus to demand
            q_dem = demand_th[k] if demand_th else 0.0
            add("Thermal Bus", "Demand", q_dem, "Bus→Demand [MWₜₕ]")

            sankey_traces.append(
                go.Sankey(
                    node=dict(label=nodes, pad=12, thickness=18),
                    link=dict(
                        source=links_src,
                        target=links_tgt,
                        value=links_val,
                        label=links_lbl,
                    ),
                    visible=False,
                    valueformat=".2f",
                )
            )
        if sankey_traces:
            sankey_traces[0].visible = True

        # Slider steps
        sliders = [
            dict(
                active=0,
                currentvalue={"prefix": "t = "},
                pad={"t": 10},
                steps=[
                    dict(
                        method="update",
                        args=[
                            {"visible": [(i == k) for i in range(len(sankey_traces))]}
                        ],
                        label=str(T[k]),
                    )
                    for k in range(len(sankey_traces))
                ],
            )
        ]

        fig_sk = go.Figure(data=sankey_traces)
        fig_sk.update_layout(
            title="Per-step Sankey (use slider)", sliders=sliders, height=540
        )

        # ------------------------- STORAGE ANALYTICS -------------------------
        figs_tes = []
        if ts is not None:
            # Duration curve of discharge
            if exists(ts_ds):
                dc = sorted([v for v in ts_ds if v > 1e-9], reverse=True)
                fig_dc = go.Figure(go.Scatter(y=dc, mode="lines", name="Discharge"))
                fig_dc.update_layout(
                    title="TES Discharge Duration Curve",
                    yaxis_title="MWₜₕ",
                    xaxis_title="Ranked hours",
                )
                figs_tes.append(fig_dc)

            # Simple cycle amplitude via SOC extrema (approximate)
            if exists(ts_soc):
                y = np.array(ts_soc, dtype=float)
                ext_idx = [0]
                for i in range(1, len(y) - 1):
                    if (y[i - 1] <= y[i] and y[i] > y[i + 1]) or (
                        y[i - 1] >= y[i] and y[i] < y[i + 1]
                    ):
                        ext_idx.append(i)
                ext_idx.append(len(y) - 1)
                amps = [
                    abs(y[ext_idx[i + 1]] - y[ext_idx[i]])
                    for i in range(len(ext_idx) - 1)
                ]
                fig_ch = go.Figure(
                    go.Histogram(x=[a for a in amps if a > 1e-6], nbinsx=20)
                )
                fig_ch.update_layout(
                    title="Cycle Amplitude Histogram (SOC extrema deltas)",
                    xaxis_title="MWhₜₕ",
                    yaxis_title="Count",
                )
                figs_tes.append(fig_ch)

            # FIFO delivered cost vs displaced burner cost (hourly scatter)
            # build a FIFO pool of charge quanta with their effective €/MWh_th at discharge
            discharge_cost = [np.nan] * len(T)  # €/MWh_th delivered
            charge_pool = []  # list of (remaining_MWh_th_after_losses, €/MWh_th_delivered)
            # charge costing: if generator mode, charging cost comes from electricity_price and e2h efficiency to storage
            # delivered cost = (€/MWh_e) / (eta_electric_to_heat * eta_d)  OR if heat-charged → unknown: use 0, since taken from bus
            eta_e2h = (
                safe(getattr(ts, "eta_electric_to_heat", 0.95))
                if ts is not None
                else 0.95
            )
            for k in range(len(T)):
                # put new charges into pool
                ch = (
                    ts_P[k] * eta_e2h
                    if (is_gen_mode and ts_P is not None)
                    else (ts_ch[k] if ts_ch is not None else 0.0)
                )
                if ch > 1e-9:
                    if is_gen_mode and elec_price is not None:
                        # cost per MWh_th delivered from this charged MWh_th:
                        c_del = elec_price[k] / max(
                            eta_d, 1e-9
                        )  # already thermalized into store; only discharge loss remains
                        # (if you prefer to include eta_c explicitly: c_del = elec_price[k] / max(eta_e2h*eta_d,1e-9))
                    else:
                        # heat-charged from bus (no explicit energy price here) → set 0 for marginal delivered cost
                        c_del = 0.0
                    charge_pool.append(
                        [ch * dt, c_del]
                    )  # store energy (MWh_th) and its €/MWh_th delivered

                # service discharge from FIFO pool
                ds = ts_ds[k] if ts_ds is not None else 0.0
                need = ds * dt
                if need > 1e-9 and charge_pool:
                    acc_cost = 0.0
                    acc_energy = 0.0
                    while need > 1e-9 and charge_pool:
                        q, c = charge_pool[0]
                        take = min(q, need)
                        acc_cost += c * take
                        acc_energy += take
                        q -= take
                        need -= take
                        if q <= 1e-9:
                            charge_pool.pop(0)
                        else:
                            charge_pool[0][0] = q
                    if acc_energy > 1e-9:
                        discharge_cost[k] = acc_cost / acc_energy

            # displaced burner cost (cheapest marginal provider at each hour)
            # thermal €/MWh_th:
            cost_HP = [
                (elec_price[i] / max(hp_cop, 1e-9)) if elec_price else np.inf
                for i in range(len(T))
            ]
            cost_RES = [elec_price[i] if elec_price else np.inf for i in range(len(T))]
            cost_BO_NG = [
                (ng_price[i] / max(eta_boiler, 1e-9)) if ng_price else np.inf
                for i in range(len(T))
            ]
            cost_BO_H2 = [
                (h2_price[i] / max(eta_boiler, 1e-9)) if h2_price else np.inf
                for i in range(len(T))
            ]
            cost_BO_CO = [
                (coal_price[i] / max(eta_boiler, 1e-9)) if coal_price else np.inf
                for i in range(len(T))
            ]
            displaced = [
                min(
                    cost_HP[i], cost_RES[i], cost_BO_NG[i], cost_BO_H2[i], cost_BO_CO[i]
                )
                for i in range(len(T))
            ]

            # scatter for hours with TES discharge
            x = []
            y = []
            col = []
            for i in range(len(T)):
                if (
                    ts_ds is not None
                    and ts_ds[i] > 1e-9
                    and (not np.isnan(discharge_cost[i]))
                ):
                    x.append(discharge_cost[i])
                    y.append(displaced[i])
                    col.append(ts_ds[i])
            if x:
                fig_sc = go.Figure(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="markers",
                        marker=dict(
                            size=6, color=col, colorbar=dict(title="Discharge [MWₜₕ]")
                        ),
                        name="hourly points",
                    )
                )
                fig_sc.add_shape(
                    type="line",
                    x0=0,
                    y0=0,
                    x1=max(x + y) * 1.05,
                    y1=max(x + y) * 1.05,
                    line=dict(dash="dash"),
                )
                fig_sc.update_layout(
                    title="TES Delivered Cost vs. Displaced Burner Cost",
                    xaxis_title="TES delivered cost [€/MWhₜₕ]",
                    yaxis_title="Cheapest alternative cost [€/MWhₜₕ]",
                    annotations=[
                        dict(
                            text="Points above diagonal → economically favorable discharge",
                            xref="paper",
                            yref="paper",
                            x=0.02,
                            y=0.98,
                            showarrow=False,
                        )
                    ],
                )
                figs_tes.append(fig_sc)

            # TES operation panel (SOC & flows)
            fig_ts = ps.make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                subplot_titles=("TES charge/discharge", "TES state of charge"),
            )
            if exists(ts_ch):
                fig_ts.add_trace(
                    go.Scatter(x=T, y=ts_ch, name="Charge [MWₜₕ]"), row=1, col=1
                )
            if exists(ts_ds):
                fig_ts.add_trace(
                    go.Scatter(x=T, y=ts_ds, name="Discharge [MWₜₕ]"), row=1, col=1
                )
            if exists(ts_soc):
                fig_ts.add_trace(
                    go.Scatter(x=T, y=ts_soc, name="SOC [MWhₜₕ]", fill="tozeroy"),
                    row=2,
                    col=1,
                )
            fig_ts.update_layout(title="TES Operation", height=520)
            figs_tes.append(fig_ts)

                    # ---------------- TES scatter analytics vs electricity price ----------------
            if elec_price and ts_ch and ts_ds:
                # Charge scatter (MW_th on x, €/MWh on y)
                fig_chg_price = go.Figure()
                fig_chg_price.add_trace(go.Scatter(
                    x=ts_ch, y=elec_price,
                    mode="markers",
                    marker=dict(color="blue", size=6, opacity=0.6),
                    name="Charge"
                ))
                fig_chg_price.update_layout(
                    title="TES Charge vs. Electricity Price",
                    xaxis_title="Charge Power [MWₜₕ]",
                    yaxis_title="Electricity Price [€/MWhₑ]",
                    height=400
                )

                # Discharge scatter (MW_th on x, €/MWh on y)
                fig_dis_price = go.Figure()
                fig_dis_price.add_trace(go.Scatter(
                    x=ts_ds, y=elec_price,
                    mode="markers",
                    marker=dict(color="red", size=6, opacity=0.6),
                    name="Discharge"
                ))
                fig_dis_price.update_layout(
                    title="TES Discharge vs. Electricity Price",
                    xaxis_title="Discharge Power [MWₜₕ]",
                    yaxis_title="Electricity Price [€/MWhₑ]",
                    height=400
                )

                # Add to figs for export
                figs_tes.append(fig_chg_price)
                figs_tes.append(fig_dis_price)

            # ---------------- TES EXTENDED KPIs & CO2 AVOIDANCE ----------------
            # Energy integrals
            E_ch = (
                float(np.sum(ts_ch) * dt) if ts_ch is not None else 0.0
            )  # MWh_th charged (heat-charged)
            E_ds = (
                float(np.sum(ts_ds) * dt) if ts_ds is not None else 0.0
            )  # MWh_th discharged

            # Rated (or observed) energy capacity
            rated_E = None
            if hasattr(ts, "energy_capacity"):
                rated_E = safe(ts.energy_capacity)
            elif ts_soc is not None and len(ts_soc) > 0:
                rated_E = float(np.max(ts_soc) - np.min(ts_soc))
            else:
                rated_E = np.nan

            # Equivalent full cycles (EFC) & utilisation
            efc = (E_ds / rated_E) if (rated_E and rated_E > 1e-9) else np.nan
            util_time = (
                (np.count_nonzero(np.array(ts_ds) > 1e-9) / len(T))
                if ts_ds is not None
                else 0.0
            )
            util_energy = (
                (E_ds / (rated_E * len(T))) if (rated_E and rated_E > 1e-9) else np.nan
            )  # over the horizon, normalised by capacity-hours

            # Observed round-trip efficiency (from flows)
            #   - E-TES: charge is electrical; thermalised into store via eta_e2h*eta_c; we observe only thermal charge ts_ch when non-gen mode.
            #   - Here, compute a pragmatic observed value as E_ds / max(E_in_to_store, ε).
            #   If you track the thermalised energy into store explicitly, replace the denominator with that variable.
            if is_gen_mode:
                # electrical charging to delivered MWh_th:
                eta_e2h = safe(getattr(ts, "eta_electric_to_heat", 0.95))
                # approximate thermal energy injected into store from electricity (to compare with discharge)
                E_in_store = float(
                    np.sum([max(v, 0.0) for v in (ts_P or [0.0] * len(T))])
                    * dt
                    * eta_e2h
                    * eta_c
                )
            else:
                # heat-charged: we already have thermal charge flow
                E_in_store = E_ch
            eta_rt_obs = (E_ds / max(E_in_store, 1e-9)) if E_in_store > 1e-9 else np.nan

            # Hourly counterfactual emissions intensity (tCO2/MWh_th)
            # Provider intensities:
            grid_ef = (
                safe(getattr(instance, "electricity_emission_factor", 0.0))
                if hasattr(instance, "electricity_emission_factor")
                else 0.0
            )
            ng_ef = 0.202
            coal_ef = 0.341
            h2_ef = 0.0
            # tech params
            eta_boiler = safe(getattr(br, "eta_fossil", 0.90)) if br else 0.90
            hp_cop = safe(getattr(hp, "cop", 3.0)) if hp else 3.0

            cf_HP = (
                [(grid_ef / max(hp_cop, 1e-9))] * len(T) if hp else [np.inf] * len(T)
            )
            cf_RES = [grid_ef] * len(T) if hr else [np.inf] * len(T)
            cf_BO_NG = (
                [(ng_ef / max(eta_boiler, 1e-9))] * len(T)
                if (br and br_NG and any(br_NG))
                else [np.inf] * len(T)
            )
            cf_BO_CO = (
                [(coal_ef / max(eta_boiler, 1e-9))] * len(T)
                if (br and br_CO and any(br_CO))
                else [np.inf] * len(T)
            )
            cf_BO_H2 = (
                [(h2_ef / max(eta_boiler, 1e-9))] * len(T)
                if (br and br_H2 and any(br_H2))
                else [np.inf] * len(T)
            )

            cf_intensity = [
                min(cf_HP[i], cf_RES[i], cf_BO_NG[i], cf_BO_CO[i], cf_BO_H2[i])
                for i in range(len(T))
            ]

            # Emissions intensity of TES discharge (tCO2/MWh_th delivered)
            if is_gen_mode:
                # Electrical → thermal to store (eta_e2h), then discharge (eta_d)
                eta_e2h = safe(getattr(ts, "eta_electric_to_heat", 0.95))
                tes_intensity = [(grid_ef / max(eta_e2h * eta_d, 1e-9))] * len(T)
            else:
                # Heat-charged from bus (no additional direct emissions accounted here)
                tes_intensity = [0.0] * len(T)

            # Hourly avoided emissions and cumulative
            avoid_h = []
            for i in range(len(T)):
                ds = ts_ds[i] if ts_ds else 0.0
                if ds > 1e-9:
                    delta = max(
                        cf_intensity[i] - tes_intensity[i], 0.0
                    )  # never penalise; clamp at 0
                    avoid_h.append(ds * dt * delta)
                else:
                    avoid_h.append(0.0)
            avoid_total = float(np.sum(avoid_h))
            avoid_cum = np.cumsum(avoid_h).tolist()

            # Decarbonisation %
            # If a baseline instance is provided we compare total emissions; else we compare to counterfactual sum(cf_intensity*ds).
            if baseline_instance is not None and hasattr(
                baseline_instance, "time_steps"
            ):
                # reuse totals() helper above if declared; otherwise recompute quickly:
                def totals(inst):
                    BB = inst.dsm_blocks

                    def g(b, n):
                        return (
                            [safe(getattr(BB[b], n)[t]) for t in inst.time_steps]
                            if (b in BB and hasattr(BB[b], n))
                            else [0.0] * len(inst.time_steps)
                        )

                    T0 = list(inst.time_steps)
                    gp = (
                        [safe(inst.grid_power[t]) for t in T0]
                        if hasattr(inst, "grid_power")
                        else [0.0] * len(T0)
                    )
                    pE = (
                        [safe(inst.electricity_price[t]) for t in T0]
                        if hasattr(inst, "electricity_price")
                        else [0.0] * len(T0)
                    )
                    # scope-2 electricity emissions
                    geF = safe(getattr(inst, "electricity_emission_factor", grid_ef))
                    elP = g("heat_pump", "power_in")
                    elP = [
                        elP[i]
                        + g("heat_resistor", "power_in")[i]
                        + g("boiler", "power_in")[i]
                        + g("thermal_storage", "power_in")[i]
                        for i in range(len(T0))
                    ]
                    elCO2 = [elP[i] * geF for i in range(len(T0))]
                    # scope-1 fuels
                    brNG, brCO, brH2 = (
                        g("boiler", "natural_gas_in"),
                        g("boiler", "coal_in"),
                        g("boiler", "hydrogen_in"),
                    )
                    E = float(
                        (
                            np.sum(brNG) * ng_ef
                            + np.sum(brCO) * coal_ef
                            + np.sum(brH2) * h2_ef
                        )
                        * dt
                        + np.sum(elCO2) * dt
                    )
                    return E

                E_base = totals(baseline_instance)
                decarb_pct = (avoid_total / E_base) * 100.0 if E_base > 1e-9 else np.nan
            else:
                # counterfactual emissions that TES displaced (only for discharged energy)
                cf_emis = (
                    float(
                        np.sum(
                            [(ts_ds[i] * dt * cf_intensity[i]) for i in range(len(T))]
                        )
                    )
                    if ts_ds
                    else 0.0
                )
                decarb_pct = (
                    (avoid_total / cf_emis) * 100.0 if cf_emis > 1e-9 else np.nan
                )

            # Hourly "CO₂ merit" of discharge (y = cf_intensity - tes_intensity for hours with discharge)
            merit_x = []
            merit_y = []
            for i in range(len(T)):
                if ts_ds and ts_ds[i] > 1e-9:
                    merit_x.append(i)
                    merit_y.append(max(cf_intensity[i] - tes_intensity[i], 0.0))

            # KPI table & two plots
            kpi_rows = [
                ("Discharged energy [MWhₜₕ]", f"{E_ds:,.1f}"),
                ("Charged energy (to store) [MWhₜₕ]", f"{E_in_store:,.1f}"),
                (
                    "Observed round-trip efficiency [–]",
                    f"{eta_rt_obs:.3f}" if eta_rt_obs == eta_rt_obs else "—",
                ),
                (
                    "Rated capacity [MWhₜₕ]",
                    f"{rated_E:,.1f}" if rated_E == rated_E else "—",
                ),
                (
                    "Equivalent full cycles [cycles]",
                    f"{efc:.2f}" if efc == efc else "—",
                ),
                ("Utilisation (time) [% of hours]", f"{util_time*100:,.1f}%"),
                (
                    "Utilisation (energy) [E_ds/(Cap·H)]",
                    f"{util_energy:.4f}" if util_energy == util_energy else "—",
                ),
                ("CO₂ avoided (total) [t]", f"{avoid_total:,.2f}"),
                (
                    "Decarbonisation [%]",
                    f"{decarb_pct:,.1f}%" if decarb_pct == decarb_pct else "—",
                ),
            ]
            fig_tes_kpi = go.Figure(
                go.Table(
                    header=dict(values=["TES KPI", "Value"], align="left"),
                    cells=dict(
                        values=[[r[0] for r in kpi_rows], [r[1] for r in kpi_rows]],
                        align="left",
                    ),
                )
            )
            fig_tes_kpi.update_layout(title="TES Key Performance Indicators")

            fig_avoid = ps.make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                subplot_titles=(
                    "Cumulative CO₂ avoided",
                    "Hourly CO₂ merit of discharge",
                ),
            )
            fig_avoid.add_trace(
                go.Scatter(x=T, y=avoid_cum, name="Cum. CO₂ avoided [t]"), row=1, col=1
            )
            if merit_x:
                fig_avoid.add_trace(
                    go.Scatter(
                        x=merit_x, y=merit_y, name="Merit [tCO₂/MWhₜₕ]", mode="markers"
                    ),
                    row=2,
                    col=1,
                )
            fig_avoid.update_yaxes(title_text="t", row=1, col=1)
            fig_avoid.update_yaxes(title_text="tCO₂/MWhₜₕ", row=2, col=1)
            fig_avoid.update_layout(title="TES Decarbonisation Analytics")

            # add to list for writing
            figs_tes.append(fig_tes_kpi)
            figs_tes.append(fig_avoid)

        # ------------------------- EMISSIONS ANALYTICS -------------------------
        # Scope-1: NG/Coal/H2 combustion; Scope-2: electricity (grid EF param if available)
        grid_ef = (
            safe(getattr(instance, "electricity_emission_factor", 0.0))
            if hasattr(instance, "electricity_emission_factor")
            else 0.0
        )
        ng_ef = 0.202  # tCO2/MWh_th
        coal_ef = 0.341  # tCO2/MWh_th
        h2_ef = 0.0  # direct

        ph_e = (
            [zero_if_none(br_NG)[i] * ng_ef for i in range(len(T))]
            if br_NG
            else [0.0] * len(T)
        )
        ph_c = (
            [zero_if_none(br_CO)[i] * coal_ef for i in range(len(T))]
            if br_CO
            else [0.0] * len(T)
        )
        ph_h = (
            [zero_if_none(br_H2)[i] * h2_ef for i in range(len(T))]
            if br_H2
            else [0.0] * len(T)
        )

        # scope-2 electric emissions allocated to loads
        el_load = zero_if_none(hp_P)
        el_load = [
            el_load[i]
            + zero_if_none(hr_P)[i]
            + zero_if_none(eb_P)[i]
            + zero_if_none(ts_P)[i]
            for i in range(len(T))
        ]
        el_co2 = [el_load[i] * grid_ef for i in range(len(T))]

        co2_ts = [ph_e[i] + ph_c[i] + ph_h[i] + el_co2[i] for i in range(len(T))]
        E_total = float(np.sum(co2_ts) * dt)
        Q_deliv = float(np.sum(demand_th) * dt) if demand_th else 0.0
        intensity = (E_total / Q_deliv) if Q_deliv > 1e-9 else np.nan

        fig_em = ps.make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            subplot_titles=("Emissions by source (tCO₂/h)", "Emissions intensity"),
        )
        fig_em.add_trace(go.Scatter(x=T, y=ph_e, name="Boiler NG"), row=1, col=1)
        fig_em.add_trace(go.Scatter(x=T, y=ph_c, name="Boiler Coal"), row=1, col=1)
        fig_em.add_trace(go.Scatter(x=T, y=ph_h, name="Boiler H₂"), row=1, col=1)
        fig_em.add_trace(
            go.Scatter(x=T, y=el_co2, name="Electricity (scope-2)"), row=1, col=1
        )
        # intensity vs time (tCO2/MWh_th to demand)
        intens_ts = [
            (co2_ts[i] / max(demand_th[i], 1e-9)) if demand_th else np.nan
            for i in range(len(T))
        ]
        fig_em.add_trace(
            go.Scatter(x=T, y=intens_ts, name="Intensity [tCO₂/MWhₜₕ]"), row=2, col=1
        )
        fig_em.update_layout(
            title=f"Emissions (Total={E_total:,.1f} t; Intensity={intensity:.3f} tCO₂/MWhₜₕ)",
            height=700,
        )
        fig_em.update_yaxes(title_text="tCO₂/h", row=1, col=1)
        fig_em.update_yaxes(title_text="tCO₂/MWhₜₕ", row=2, col=1)

        # ------------------------- ECONOMICS ANALYTICS -------------------------
        # Unit operating_cost streams (if exposed); add grid energy cost
        cost_ts = (
            [safe(instance.variable_cost[t]) for t in T]
            if hasattr(instance, "variable_cost")
            else None
        )
        c_hp = series_or_none(hp, "operating_cost")
        c_hr = series_or_none(hr, "operating_cost")
        c_br = series_or_none(br, "operating_cost")
        c_pv = series_or_none(pv, "operating_cost")
        grid_cost = [
            (elec_price[i] * grid[i]) if elec_price else 0.0 for i in range(len(T))
        ]

        fig_ec = ps.make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            subplot_titles=("Variable cost stack (€/h)", "KPIs"),
        )
        # stack
        if exists(c_hp):
            fig_ec.add_trace(go.Bar(x=T, y=c_hp, name="HP €"), row=1, col=1)
        if exists(c_hr):
            fig_ec.add_trace(go.Bar(x=T, y=c_hr, name="Resistor €"), row=1, col=1)
        if exists(c_br):
            fig_ec.add_trace(go.Bar(x=T, y=c_br, name="Boiler €"), row=1, col=1)
        if exists(c_pv):
            fig_ec.add_trace(go.Bar(x=T, y=c_pv, name="PV €"), row=1, col=1)
        if exists(grid_cost):
            fig_ec.add_trace(go.Bar(x=T, y=grid_cost, name="Grid €"), row=1, col=1)
        if exists(cost_ts):
            fig_ec.add_trace(
                go.Scatter(
                    x=T, y=cost_ts, name="Total variable cost [€/h]", mode="lines"
                ),
                row=1,
                col=1,
            )
        fig_ec.update_layout(barmode="stack")

        total_cost = (
            float(np.sum(cost_ts) * dt) if cost_ts else (float(np.sum(grid_cost) * dt))
        )
        LCOH = (total_cost / Q_deliv) if Q_deliv > 1e-9 else np.nan
        # table = go.Table(
        #     header=dict(values=["Economic KPI","Value"], align="left"),
        #     cells=dict(values=list(zip(*[
        #         ("Total variable cost [€]", f"{total_cost:,.0f}"),
        #         ("Delivered heat [MWhₜₕ]", f"{Q_deliv:,.0f}"),
        #         ("LCOH_var [€/MWhₜₕ]", f"{LCOH:,.2f}" if LCOH==LCOH else "—"),
        #     ])), align="left")
        # )
        rows = [
            ("Total variable cost [€]", f"{total_cost:,.0f}"),
            ("Delivered heat [MWhₜₕ]", f"{Q_deliv:,.0f}"),
            ("LCOH_var [€/MWhₜₕ]", f"{LCOH:,.2f}" if LCOH == LCOH else "—"),
        ]
        table_trace = go.Table(
            header=dict(values=["Economic KPI", "Value"], align="left"),
            cells=dict(
                values=[[r[0] for r in rows], [r[1] for r in rows]], align="left"
            ),
        )

        # --- make the subplots with proper specs ---
        fig_ec = ps.make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=False,  # tables don't share x-axes
            vertical_spacing=0.07,
            specs=[[{"type": "xy"}], [{"type": "table"}]],
            subplot_titles=("Variable cost stack (€/h)", "KPIs"),
        )

        # row 1: your stacked bars/line stay the same
        if c_hp:
            fig_ec.add_trace(go.Bar(x=T, y=c_hp, name="HP €"), row=1, col=1)
        if c_hr:
            fig_ec.add_trace(go.Bar(x=T, y=c_hr, name="Resistor €"), row=1, col=1)
        if c_br:
            fig_ec.add_trace(go.Bar(x=T, y=c_br, name="Boiler €"), row=1, col=1)
        if c_pv:
            fig_ec.add_trace(go.Bar(x=T, y=c_pv, name="PV €"), row=1, col=1)
        if grid_cost:
            fig_ec.add_trace(go.Bar(x=T, y=grid_cost, name="Grid €"), row=1, col=1)
        if cost_ts:
            fig_ec.add_trace(
                go.Scatter(
                    x=T, y=cost_ts, name="Total variable cost [€/h]", mode="lines"
                ),
                row=1,
                col=1,
            )
        fig_ec.update_layout(barmode="stack")

        # row 2: place the table
        fig_ec.add_trace(table_trace, row=2, col=1)

        fig_ec.update_layout(title="Economics", height=780)

        # ------------------------- BASELINE COMPARISON (optional) -------------------------
        figs_cmp = []
        if baseline_instance is not None and hasattr(baseline_instance, "time_steps"):
            # quick totals for baseline: emissions, cost, demand
            def totals(inst):
                BB = inst.dsm_blocks

                def g(b, n):
                    return (
                        [safe(getattr(BB[b], n)[t]) for t in inst.time_steps]
                        if (b in BB and hasattr(BB[b], n))
                        else [0.0] * len(inst.time_steps)
                    )

                T0 = list(inst.time_steps)
                # loads & prices
                gp = (
                    [safe(inst.grid_power[t]) for t in T0]
                    if hasattr(inst, "grid_power")
                    else [0.0] * len(T0)
                )
                pE = (
                    [safe(inst.electricity_price[t]) for t in T0]
                    if hasattr(inst, "electricity_price")
                    else [0.0] * len(T0)
                )
                cost = (
                    [safe(inst.variable_cost[t]) for t in T0]
                    if hasattr(inst, "variable_cost")
                    else [pE[i] * gp[i] for i in range(len(T0))]
                )
                # emissions again
                brNG, brCO, brH2 = (
                    g("boiler", "natural_gas_in"),
                    g("boiler", "coal_in"),
                    g("boiler", "hydrogen_in"),
                )
                elP = g("heat_pump", "power_in")
                elP = [
                    elP[i]
                    + g("heat_resistor", "power_in")[i]
                    + g("boiler", "power_in")[i]
                    + g("thermal_storage", "power_in")[i]
                    for i in range(len(T0))
                ]
                elCO2 = [elP[i] * grid_ef for i in range(len(T0))]
                E = float(
                    (
                        np.sum(brNG) * ng_ef
                        + np.sum(brCO) * coal_ef
                        + np.sum(brH2) * h2_ef
                    )
                    * dt
                    + np.sum(elCO2) * dt
                )
                C = float(np.sum(cost) * dt)
                Q = (
                    float(np.sum([safe(inst.thermal_demand[t]) for t in T0]) * dt)
                    if hasattr(inst, "thermal_demand")
                    else 0.0
                )
                return E, C, Q

            E0, C0, Q0 = totals(baseline_instance)
            dE, dC, dQ = E_total - E0, total_cost - C0, Q_deliv - Q0
            mac = (dC / dE) if abs(dE) > 1e-9 else np.nan
            fig_b = go.Figure(
                data=[
                    go.Table(
                        header=dict(
                            values=["Baseline vs Scenario", "Value"], align="left"
                        ),
                        cells=dict(
                            values=list(
                                zip(
                                    *[
                                        ("Δ Emissions [tCO₂]", f"{dE:,.1f}"),
                                        ("Δ Variable cost [€]", f"{dC:,.0f}"),
                                        ("Δ Heat delivered [MWhₜₕ]", f"{dQ:,.1f}"),
                                        (
                                            "Marginal abatement cost [€/tCO₂]",
                                            f"{mac:,.1f}" if mac == mac else "—",
                                        ),
                                    ]
                                )
                            ),
                            align="left",
                        ),
                    )
                ]
            )
            fig_b.update_layout(title="Baseline Comparison (Economy & CO₂)")
            figs_cmp.append(fig_b)

                # ------------------------- PROSUMER / RESERVE (FCR / aFRR) ANALYTICS -------------------------
        figs_res = []
        is_prosumer = bool(getattr(self, "is_prosumer", False))
        if is_prosumer and hasattr(instance, "time_steps"):
            blk_len   = int(getattr(self, "_FCR_BLOCK_LENGTH", 4))
            step_mw   = float(getattr(self, "_FCR_STEP_MW", 1.0))
            min_bid   = float(getattr(self, "_FCR_MIN_BID_MW", 1.0))
            symmetric = bool(getattr(self, "fcr_symmetric", True))   # True → FCR; False → aFRR (+/−)

            # Handles exposed by your model
            fcr_blocks   = list(getattr(instance, "fcr_blocks", []))
            cap_up_v     = getattr(instance, "cap_up", None)           # per-block capacity [MW]
            cap_dn_v     = getattr(instance, "cap_dn", None)
            price_fcr_v  = getattr(instance, "fcr_block_price", None)  # €/MW per 4h (symmetric)
            price_pos_v  = getattr(instance, "afrr_block_price_pos", None)  # €/MW per 4h (asymmetric +)
            price_neg_v  = getattr(instance, "afrr_block_price_neg", None)  # €/MW per 4h (asymmetric −)

            if fcr_blocks and (cap_up_v or cap_dn_v):
                # helpers
                def s(v):
                    try: return float(pyo.value(v))
                    except Exception: return float(v) if v is not None else 0.0

                def pull_map(pyobj):
                    d = {}
                    if pyobj is None: return d
                    for b in fcr_blocks: d[b] = s(pyobj[b])
                    return d

                def stairs(blocks, block_map, L, H):
                    y = [0.0]*H
                    for b in blocks:
                        val = float(block_map.get(b, 0.0))
                        for k in range(L):
                            i = b + k
                            if 0 <= i < H: y[i] = val
                    return y

                up_map  = pull_map(cap_up_v)
                dn_map  = pull_map(cap_dn_v)
                H       = len(T)

                # ---- prices (symmetric or asymmetric) ----
                pr_map_sym = pull_map(price_fcr_v) if symmetric else {}
                pr_map_pos = pull_map(price_pos_v) if not symmetric else {}
                pr_map_neg = pull_map(price_neg_v) if not symmetric else {}

                up_stairs = stairs(fcr_blocks, up_map, blk_len, H) if up_map else [0.0]*H
                dn_stairs = stairs(fcr_blocks, dn_map, blk_len, H) if dn_map else [0.0]*H
                pr_sym    = stairs(fcr_blocks, pr_map_sym, blk_len, H) if pr_map_sym else None
                pr_pos    = stairs(fcr_blocks, pr_map_pos, blk_len, H) if pr_map_pos else None
                pr_neg    = stairs(fcr_blocks, pr_map_neg, blk_len, H) if pr_map_neg else None

                # ---- Block ladder (bars) + price line(s) ----
                x_blk = fcr_blocks
                y_up  = [up_map.get(b, 0.0) for b in x_blk]
                y_dn  = [dn_map.get(b, 0.0) for b in x_blk]

                fig_blk = ps.make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
                fig_blk.add_trace(go.Bar(x=x_blk, y=y_up, name="UP capacity [MW]",   marker_color="#d62728"), secondary_y=False)
                fig_blk.add_trace(go.Bar(x=x_blk, y=y_dn, name="DOWN capacity [MW]", marker_color="#2ca02c"), secondary_y=False)

                if symmetric:
                    y_pr = [pr_map_sym.get(b, 0.0) for b in x_blk]
                    fig_blk.add_trace(
                        go.Scatter(x=x_blk, y=y_pr, name="FCR price [€/MW·4h]",
                                   line=dict(color="#9467bd", dash="dash")),
                        secondary_y=True
                    )
                    fig_blk.update_layout(title="FCR Blocks: Capacity & Price", barmode="stack", height=420)
                else:
                    y_pr_pos = [pr_map_pos.get(b, 0.0) for b in x_blk]
                    y_pr_neg = [pr_map_neg.get(b, 0.0) for b in x_blk]
                    fig_blk.add_trace(
                        go.Scatter(x=x_blk, y=y_pr_pos, name="aFRR+ price [€/MW·4h]",
                                   line=dict(color="#1f77b4", dash="dash")),
                        secondary_y=True
                    )
                    fig_blk.add_trace(
                        go.Scatter(x=x_blk, y=y_pr_neg, name="aFRR− price [€/MW·4h]",
                                   line=dict(color="#ff7f0e", dash="dot")),
                        secondary_y=True
                    )
                    fig_blk.update_layout(title="aFRR Blocks: Capacity & Price (+/−)", barmode="stack", height=420)

                fig_blk.update_yaxes(title_text="MW", secondary_y=False)
                fig_blk.update_yaxes(title_text="€/MW·4h", secondary_y=True)
                figs_res.append(fig_blk)

                # ---- Hourly impact: bands around baseline + price line(s) on right axis ----
                base    = [safe(instance.total_power_input[t]) for t in T] if hasattr(instance, "total_power_input") else [0.0]*H
                max_cap = float(getattr(self, "max_plant_capacity", np.nan))
                min_cap = float(getattr(self, "min_plant_capacity", 0.0))

                lower_up = [max(min_cap, base[i] - up_stairs[i]) for i in range(H)]
                upper_dn = [min(max_cap if not np.isnan(max_cap) else 1e9, base[i] + dn_stairs[i]) for i in range(H)]

                fig_imp = ps.make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
                fig_imp.add_trace(go.Scatter(x=T, y=base, name="Baseline load [MWₑ]", line=dict(color="grey")), secondary_y=False)
                fig_imp.add_trace(go.Scatter(x=T, y=lower_up, name="Baseline−UP", line=dict(color="#d62728", width=0), showlegend=False), secondary_y=False)
                fig_imp.add_trace(go.Scatter(x=T, y=base,     name="UP capacity band",   fill="tonexty", mode="lines",
                                             line=dict(color="#d62728"), fillcolor="rgba(214,39,40,0.25)"), secondary_y=False)
                fig_imp.add_trace(go.Scatter(x=T, y=upper_dn, name="DOWN capacity band", fill="tonexty", mode="lines",
                                             line=dict(color="#2ca02c"), fillcolor="rgba(44,160,44,0.25)"), secondary_y=False)

                if symmetric and pr_sym is not None:
                    fig_imp.add_trace(go.Scatter(x=T, y=pr_sym, name="FCR price [€/MW·4h]",
                                                 line=dict(color="#9467bd", dash="dot")), secondary_y=True)
                else:
                    if pr_pos is not None:
                        fig_imp.add_trace(go.Scatter(x=T, y=pr_pos, name="aFRR+ price [€/MW·4h]",
                                                     line=dict(color="#1f77b4", dash="dot")), secondary_y=True)
                    if pr_neg is not None:
                        fig_imp.add_trace(go.Scatter(x=T, y=pr_neg, name="aFRR− price [€/MW·4h]",
                                                     line=dict(color="#ff7f0e", dash="dot")), secondary_y=True)

                fig_imp.update_layout(title="Reserve Impact: Capacity Bands (red=UP, green=DOWN)", height=460)
                fig_imp.update_yaxes(title_text="MW / MWₑ", secondary_y=False)
                fig_imp.update_yaxes(title_text="Price [€/MW·4h]", secondary_y=True)
                figs_res.append(fig_imp)

                # ---- Revenue (this matches your model payout) ----
                # model: if symmetric → cap_up == cap_dn and paid ONCE: revenue = price_fcr * cap_up
                #        if asymmetric → revenue = price_pos*cap_up + price_neg*cap_dn
                if symmetric:
                    y_pr = [pr_map_sym.get(b, 0.0) for b in x_blk]
                    rev_blk = [y_pr[i] * y_up[i] for i in range(len(x_blk))]
                    rev_up  = rev_blk
                    rev_dn  = [0.0]*len(x_blk)
                else:
                    y_pr_pos = [pr_map_pos.get(b, 0.0) for b in x_blk]
                    y_pr_neg = [pr_map_neg.get(b, 0.0) for b in x_blk]
                    rev_up   = [y_pr_pos[i] * y_up[i] for i in range(len(x_blk))]
                    rev_dn   = [y_pr_neg[i] * y_dn[i] for i in range(len(x_blk))]
                rev_tot = [rev_up[i] + rev_dn[i] for i in range(len(x_blk))]
                rev_cum = np.cumsum(rev_tot).tolist()

                # MW-weighted price (uses what you’re actually paid on each side)
                denom = sum(y_up) + sum(y_dn)
                if symmetric:
                    numer = sum([y_pr[i] * y_up[i] for i in range(len(x_blk))])
                else:
                    numer = sum([y_pr_pos[i]*y_up[i] for i in range(len(x_blk))]) + \
                            sum([y_pr_neg[i]*y_dn[i] for i in range(len(x_blk))])
                mw_w_price = numer / max(denom, 1e-9)

                fig_rev = ps.make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
                fig_rev.add_trace(go.Bar(x=x_blk, y=rev_up, name=("Revenue UP [€]" if not symmetric else "Revenue [€]"),
                                         marker_color="#d62728"), secondary_y=False)
                if not symmetric:
                    fig_rev.add_trace(go.Bar(x=x_blk, y=rev_dn, name="Revenue DOWN [€]",
                                             marker_color="#2ca02c"), secondary_y=False)
                fig_rev.add_trace(go.Scatter(x=x_blk, y=rev_cum, name="Cumulative revenue [€]",
                                             line=dict(color="#1f77b4", width=2)), secondary_y=True)
                fig_rev.update_layout(barmode=("overlay" if symmetric else "stack"),
                                      title="Reserve Revenue by Block (and cumulative)", height=420)
                fig_rev.update_yaxes(title_text="€/block", secondary_y=False)
                fig_rev.update_yaxes(title_text="€ cumulative", secondary_y=True)
                figs_res.append(fig_rev)

                # KPIs table (names adapt to symmetric/asymmetric)
                kpi_rows = [
                    ("Blocks with any bid", f"{sum(1 for i in range(len(x_blk)) if (y_up[i]+y_dn[i])>0)} / {len(x_blk)}"),
                    ("Total UP capacity [MW·blocks]",   f"{sum(y_up):,.0f}"),
                    ("Total DOWN capacity [MW·blocks]", f"{sum(y_dn):,.0f}"),
                    ("Total reserve revenue [€]",       f"{sum(rev_tot):,.0f}"),
                    ("MW-weighted price [€/MW·4h]",     f"{mw_w_price:,.1f}"),
                    ("Min bid [MW] / Step [MW]",        f"{min_bid:g} / {step_mw:g}"),
                    ("Symmetric product",               "Yes" if symmetric else "No (aFRR +/−)")
                ]
                fig_res_kpi = go.Figure(go.Table(
                    header=dict(values=["Reserve KPI","Value"], align="left"),
                    cells=dict(values=[[r[0] for r in kpi_rows],[r[1] for r in kpi_rows]], align="left")
                ))
                fig_res_kpi.update_layout(title="Reserve Market KPIs")
                figs_res.append(fig_res_kpi)

                # ---- Compliance check: min bid & step ----
                viol = []
                for i,b in enumerate(x_blk):
                    if (y_up[i] > 1e-9) and ((y_up[i] < min_bid) or (abs(y_up[i]/step_mw - round(y_up[i]/step_mw)) > 1e-9)):
                        viol.append((b, "UP", y_up[i]))
                    if (y_dn[i] > 1e-9) and ((y_dn[i] < min_bid) or (abs(y_dn[i]/step_mw - round(y_dn[i]/step_mw)) > 1e-9)):
                        viol.append((b, "DOWN", y_dn[i]))
                if viol:
                    fig_v = go.Figure(go.Table(
                        header=dict(values=["Block start","Side","Capacity [MW] (violates min/step)"], align="left"),
                        cells=dict(values=[[v[0] for v in viol],[v[1] for v in viol],[f"{v[2]:.1f}" for v in viol]], align="left")
                    ))
                    fig_v.update_layout(title="Bid Compliance Issues")
                    figs_res.append(fig_v)


        # ------------------------- WRITE SINGLE HTML -------------------------
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("<html><head><meta charset='utf-8'><title>Steam Plant – Full Dashboard</title></head><body>")
            f.write("<h2>Steam Generation Plant – Full Dashboard</h2>")

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

            # NEW: reserve analytics section
            if figs_res:
                f.write("<hr><h3>Reserve Market (FCR / aFRR) Analytics</h3>")
                for g in figs_res:
                    f.write(g.to_html(full_html=False, include_plotlyjs=False))

            f.write("</body></html>")
        print(f"Full dashboard saved to: {html_path}")


