# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import pyomo.environ as pyo

from assume.common.base import SupportsMinMax
from assume.common.forecasts import Forecaster
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

    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        forecaster: Forecaster,
        components: dict[str, dict] = None,
        technology: str = "steel_plant",
        objective: str = "min_variable_cost",
        flexibility_measure: str = "cost_based_load_shift",
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

        self.natural_gas_price = self.forecaster["fuel_price_natural_gas"]
        self.hydrogen_price = self.forecaster["price_hydrogen"]
        self.electricity_price = self.forecaster["price_EOM"]
        self.steel_demand_per_time_step = self.forecaster[f"{self.id}_steel_demand"]
        self.iron_ore_price = self.forecaster.get_price("iron_ore")
        self.steel_demand = demand
        self.steel_price = self.forecaster.get_price("steel")
        self.lime_price = self.forecaster.get_price("lime")
        self.co2_price = self.forecaster.get_price("co2")

        # Calculate congestion forecast and set it as a forecast column in the forecaster
        self.congestion_signal = self.forecaster[f"{node}_congestion_severity"]
        self.renewable_utilisation_signal = self.forecaster[
            f"{node}_renewable_utilisation"
        ]

        self.objective = objective
        self.flexibility_measure = flexibility_measure
        self.cost_tolerance = cost_tolerance
        self.congestion_threshold = congestion_threshold
        self.peak_load_cap = peak_load_cap

        # Check for the presence of components
        self.has_h2storage = "hydrogen_buffer_storage" in self.components.keys()
        self.has_dristorage = "dri_storage" in self.components.keys()
        self.has_electrolyser = "electrolyser" in self.components.keys()

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
        self.model.steel_demand = pyo.Param(initialize=self.steel_demand)
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
        # expects: self.steel_demand_per_time_step is a list/array length = len(time_steps)
        self.model.steel_demand_per_time_step = pyo.Param(
            self.model.time_steps,
            initialize={
                t: value for t, value in enumerate(self.steel_demand_per_time_step)
            },
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

        @self.model.Constraint(self.model.time_steps)
        def steel_output_association_constraint(m, t):
            if "eaf" not in m.dsm_blocks:
                return pyo.Constraint.Skip
            return m.dsm_blocks["eaf"].steel_output[t] >= m.steel_demand_per_time_step[t]


        # Constraint for total power input
        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
            """
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

    def plot_1(self, instance, save_name=None, out_dir="./outputs", show=True):
        """
        Two-panel figure + CSV export for SteelPlant.

        Top:
        - Electric inputs: EAF, DRI plant, Electrolyser (if present)
        - Fuel inputs to DRI: H2, Natural Gas (if present / nonzero)
        - Prices on twin axis (electricity + optional NG/H2/CO2)

        Bottom:
        - Material flows: DRI output, Steel output
        - Optional storages: H2 buffer (charge/discharge/SOC), DRI storage (charge/discharge/SOC)
        - Optional: electrolyser H2 output

        If save_name is given, saves a PNG and CSV to out_dir with that stem.
        Presence-aware: draws only what exists and is nonzero.
        """

        # local imports (keeps steel_plant.py clean)
        from pathlib import Path
        import pandas as pd
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import pyomo.environ as pyo

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

        def plot_if_nonzero(ax, x, y, label, color=None, style="-", eps=1e-9):
            if y is None:
                return None
            if any(abs(v) > eps for v in y):
                if color is None:
                    return ax.plot(x, y, label=label, linestyle=style)[0]
                return ax.plot(x, y, label=label, color=color, linestyle=style)[0]
            return None

        # ------- blocks (IndexedBlock-style access) -------
        B = instance.dsm_blocks
        dri = B["dri_plant"] if "dri_plant" in B else None
        eaf = B["eaf"] if "eaf" in B else None
        elz = B["electrolyser"] if "electrolyser" in B else None
        h2s = B["hydrogen_buffer_storage"] if "hydrogen_buffer_storage" in B else None
        dris = B["dri_storage"] if "dri_storage" in B else None

        # ------- series: electric loads (MW_e) -------
        dri_P = series_or_none(dri, "power_in")
        eaf_P = series_or_none(eaf, "power_in")
        elz_P = series_or_none(elz, "power_in")

        total_power = (
            [safe_value(instance.total_power_input[t]) for t in T]
            if hasattr(instance, "total_power_input")
            else None
        )

        # ------- series: DRI fuels (MW_th-equivalent in your model units) -------
        # (these are "in" streams already in MWh/step or MW_th depending on your timestep convention;
        #  we just plot them as provided)
        dri_H2 = series_or_none(dri, "hydrogen_in")
        dri_NG = series_or_none(dri, "natural_gas_in")

        # ------- series: material flows -------
        dri_out = series_or_none(dri, "dri_output")
        steel_out = series_or_none(eaf, "steel_output")
        eaf_dri_in = series_or_none(eaf, "dri_input")

        # ------- series: storages (charge/discharge in MW, soc in MWh) -------
        h2_ch = series_or_none(h2s, "charge")
        h2_ds = series_or_none(h2s, "discharge")
        h2_soc = series_or_none(h2s, "soc")

        dris_ch = series_or_none(dris, "charge")
        dris_ds = series_or_none(dris, "discharge")
        dris_soc = series_or_none(dris, "soc")

        # ------- electrolyser output (optional) -------
        elz_H2_out = series_or_none(elz, "hydrogen_out")

        # ------- prices & costs (optional) -------
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

        # €/t steel (per step): variable_cost[t] / steel_out[t]
        marginal_cost = None
        if variable_cost is not None and steel_out is not None:
            eps = 1e-9
            marginal_cost = [variable_cost[i] / max(steel_out[i], eps) for i in range(len(T))]

        # component emissions (if present)
        dri_co2 = series_or_none(dri, "co2_emission")
        eaf_co2 = series_or_none(eaf, "co2_emission")
        total_co2 = None
        if dri_co2 is not None or eaf_co2 is not None:
            total_co2 = [
                (dri_co2[i] if dri_co2 is not None else 0.0)
                + (eaf_co2[i] if eaf_co2 is not None else 0.0)
                for i in range(len(T))
            ]

        # ------- figure -------
        fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex=True, constrained_layout=True)

        # ---------- TOP: electric + fuels + prices ----------
        lines_top = []
        lines_top += [
            plot_if_nonzero(axs[0], T, eaf_P, "EAF Power [MWₑ]", color="C0"),
            plot_if_nonzero(axs[0], T, dri_P, "DRI Plant Power [MWₑ]", color="C1", style="-"),
            plot_if_nonzero(axs[0], T, elz_P, "Electrolyser Power [MWₑ]", color="C2", style="--"),
            plot_if_nonzero(axs[0], T, total_power, "Total Power [MWₑ]", color="C3", style="-."),
        ]

        lines_top += [
            plot_if_nonzero(axs[0], T, dri_H2, "DRI H₂ In [MWₜₕ eq.]", color="C4", style="--"),
            plot_if_nonzero(axs[0], T, dri_NG, "DRI NG In [MWₜₕ eq.]", color="C5", style=":"),
        ]

        axs[0].set_ylabel("MWₑ / MWₜₕ(eq.)")
        axs[0].set_title("Steel Plant – Electric Inputs, DRI Fuel Inputs, and Prices")
        axs[0].grid(True, which="both", axis="both")

        # prices (twin axis)
        axp = axs[0].twinx()
        lp = []
        if elec_price is not None:
            lp += axp.plot(T, elec_price, label="Elec Price", linestyle="--")
        if ng_price is not None:
            lp += axp.plot(T, ng_price, label="NG Price", linestyle=":")
        if h2_price is not None:
            lp += axp.plot(T, h2_price, label="H₂ Price", linestyle="-.")
        if co2_price is not None:
            lp += axp.plot(T, co2_price, label="CO₂ Price", linestyle="-")
        axp.set_ylabel("Price", color="gray")
        axp.tick_params(axis="y", labelcolor="gray")

        handles = [h for h in lines_top if h is not None] + lp
        labels = [h.get_label() for h in handles]
        if handles:
            axs[0].legend(handles, labels, loc="upper left", frameon=True, ncol=2)

        # ---------- BOTTOM: material + storages ----------
        bl = []
        bl += [
            plot_if_nonzero(axs[1], T, dri_out, "DRI Output [t/step]", color="C1"),
            plot_if_nonzero(axs[1], T, eaf_dri_in, "EAF DRI Input [t/step]", color="C6", style="--"),
            plot_if_nonzero(axs[1], T, steel_out, "Steel Output [t/step]", color="C0", style="-"),
            plot_if_nonzero(axs[1], T, elz_H2_out, "Electrolyser H₂ Out [MWₜₕ eq.]", color="C2", style="--"),
        ]

        # H2 storage
        bl += [
            plot_if_nonzero(axs[1], T, h2_ch, "H₂ Storage Charge [MW]", color="C4", style="-"),
            plot_if_nonzero(axs[1], T, h2_ds, "H₂ Storage Discharge [MW]", color="C4", style="--"),
        ]
        if h2_soc is not None and any(abs(v) > 1e-9 for v in h2_soc):
            axs[1].fill_between(T, h2_soc, 0, alpha=0.20, label="H₂ Storage SOC [MWh]")

        # DRI storage
        bl += [
            plot_if_nonzero(axs[1], T, dris_ch, "DRI Storage Charge [t/step]", color="C7", style="-"),
            plot_if_nonzero(axs[1], T, dris_ds, "DRI Storage Discharge [t/step]", color="C7", style="--"),
        ]
        if dris_soc is not None and any(abs(v) > 1e-9 for v in dris_soc):
            axs[1].fill_between(T, dris_soc, 0, alpha=0.20, label="DRI Storage SOC [t]")

        axs[1].set_ylabel("t/step or MW/MWh")
        axs[1].set_title("Steel Plant – Material Flows and Storage Operation")
        axs[1].grid(True, which="both", axis="both")
        axs[1].set_xlabel("Time step")

        blines = [h for h in bl if h is not None]
        if blines:
            axs[1].legend(loc="upper left", frameon=True, ncol=2)

        plt.tight_layout()

        # -------- CSV export (include what we compute, even if not plotted) --------
        df = pd.DataFrame(
            {
                "t": T,
                # power
                "EAF Power [MW_e]": eaf_P,
                "DRI Plant Power [MW_e]": dri_P,
                "Electrolyser Power [MW_e]": elz_P,
                "Total Power [MW_e]": total_power,
                # fuels to DRI
                "DRI H2 In [MW_th eq]": dri_H2,
                "DRI NG In [MW_th eq]": dri_NG,
                # materials
                "DRI Output [t/step]": dri_out,
                "EAF DRI Input [t/step]": eaf_dri_in,
                "Steel Output [t/step]": steel_out,
                # storages
                "H2 Storage Charge [MW]": h2_ch,
                "H2 Storage Discharge [MW]": h2_ds,
                "H2 Storage SOC [MWh]": h2_soc,
                "DRI Storage Charge [t/step]": dris_ch,
                "DRI Storage Discharge [t/step]": dris_ds,
                "DRI Storage SOC [t]": dris_soc,
                # electrolyser output
                "Electrolyser H2 Out [MW_th eq]": elz_H2_out,
                # emissions
                "DRI CO2 [tCO2/step]": dri_co2,
                "EAF CO2 [tCO2/step]": eaf_co2,
                "Total CO2 [tCO2/step]": total_co2,
                # prices & economics
                "Elec Price": elec_price,
                "NG Price": ng_price,
                "H2 Price": h2_price,
                "CO2 Price": co2_price,
                "Variable Cost [€]": variable_cost,
                "Marginal Cost [€/t steel]": marginal_cost,
            }
        )

        # ---------- SAVE: one name -> both .png and .csv ----------
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

