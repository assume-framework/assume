# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# %%
import logging
import os
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import pandas as pd

    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

from assume import World
from assume.scenario.loader_csv import load_scenario_folder, run_learning

log = logging.getLogger(__name__)

# define the path where the simulation results in form of CSV files will be stored, for example: "examples/outputs"
# "" means no CSV files will be stored
csv_path = ""

os.makedirs("./examples/local_db", exist_ok=True)

available_examples = {
    # small examples for easier understanding of different features and configurations
    "small": {"scenario": "example_01a", "study_case": "base"},
    "small_with_exchanges": {
        "scenario": "example_01a",
        "study_case": "base_with_exchanges",
    },
    "small_dam": {"scenario": "example_01a", "study_case": "dam"},
    "small_with_opt_clearing": {
        "scenario": "example_01a",
        "study_case": "dam_with_complex_clearing",
    },
    "small_with_vre": {"scenario": "example_01b", "study_case": "base"},
    "small_with_vre_and_storage": {
        "scenario": "example_01c",
        "study_case": "eom_only",
    },
    "small_with_BB_and_LB": {
        "scenario": "example_01c",
        "study_case": "dam_with_complex_opt_clearing",
    },
    "small_with_vre_and_storage_and_complex_clearing": {
        "scenario": "example_01c",
        "study_case": "dam_with_complex_opt_clearing",
    },
    "small_with_crm": {
        "scenario": "example_01c",
        "study_case": "eom_and_crm",
    },
    "small_with_redispatch": {
        "scenario": "example_01d",
        "study_case": "base",
    },
    "small_with_zonal_clearing": {
        "scenario": "example_01d",
        "study_case": "zonal_case",
    },
    "small_with_nodal_clearing": {
        "scenario": "example_01d",
        "study_case": "nodal_case",
    },
    # example_01e is used in the tutorial notebook #3: Custom unit and bidding strategy example
    "market_study_eom": {
        "scenario": "example_01f",
        "study_case": "eom_case",
    },
    "market_study_eom_and_ltm": {
        "scenario": "example_01f",
        "study_case": "ltm_case",
    },
    # example_01f is used in the tutorial notebook #5: Market configuration comparison example
    # example_01g is used in the tutorial notebook #6: Advanced order types example
    "small_household": {
        "scenario": "example_01h",
        "study_case": "eom",
    },
    #
    # DRL references case for learning advancement testing
    "small_learning_1": {"scenario": "example_02a", "study_case": "base"},
    "small_learning_2": {"scenario": "example_02b", "study_case": "base"},
    "small_learning_3": {"scenario": "example_02c", "study_case": "base"},
    # DRL cases with lstm instead of mlp as actor neural network architecture
    "small_learning_1_lstm": {"scenario": "example_02a", "study_case": "base_lstm"},
    "small_learning_2_lstm": {"scenario": "example_02b", "study_case": "base_lstm"},
    # Further DRL example simulation showcasing learning features
    "small_learning_with_storage": {"scenario": "example_02e", "study_case": "base"},
    "small_learning_with_renewables": {"scenario": "example_02d", "study_case": "base"},
    #
    # full year examples to show real-world scenarios
    "large_2019_eom": {"scenario": "example_03", "study_case": "base_case_2019"},
    "large_2019_eom_crm": {
        "scenario": "example_03",
        "study_case": "eom_crm_case_2019",
    },
    "large_2019_day_ahead": {
        "scenario": "example_03",
        "study_case": "dam_case_2019",
    },
    "large_2019_with_DSM": {
        "scenario": "example_03",
        "study_case": "base_case_2019_with_DSM",
    },
    "large_2019_rl": {"scenario": "example_03a", "study_case": "base_case_2019"},
    "large_2021_rl": {"scenario": "example_03b", "study_case": "base_case_2021"},
    "large_2019_storage": {
        "scenario": "example_03c",
        "study_case": "base_case_2019_with_storage",
    },
}


# %%
if __name__ == "__main__":
    """
    Available examples:
    - local_db: without database and grafana
    - timescale: with database and grafana (note: you need docker installed)
    """

    # select to store the simulation results in a local database or in timescale
    # when using timescale, you need to have docker installed and can access the grafana dashboard
    data_format = "local_db"  # "local_db" or "timescale"

    # select the example to run from the available examples above
    example = "large_2019_with_DSM"

    if data_format == "local_db":
        db_uri = "sqlite:///./examples/local_db/assume_db.db"
    elif data_format == "timescale":
        db_uri = "postgresql://assume:assume@localhost:5432/assume"

    # create world
    world = World(database_uri=db_uri, export_csv_path=csv_path)

    # load scenario
    load_scenario_folder(
        world,
        inputs_path="examples/inputs",
        scenario=available_examples[example]["scenario"],
        study_case=available_examples[example]["study_case"],
    )

    # to add custom bidding strategies, you need to import them
    # and add them to the world as follows:
    # from custom_bidding_strategy import CustomBiddingStrategy
    # world.bidding_strategies["custom_bidding_strategy"] = CustomBiddingStrategy

    # to add a custom unit type, you need to import it
    # and add it to the world as follows:
    # from custom_unit import CustomUnit
    # world.unit_types["custom_unit"] = CustomUnit

    # next you need to load and add the custom units to the scenario
    # from assume import load_custom_units
    # load_custom_units(
    #     world,
    #     inputs_path="examples/inputs",
    #     scenario=availabe_examples[example]["scenario"],
    #     file_name="custom_units",
    #     unit_type="custom_unit",
    # )

    if world.learning_mode:
        # run learning if learning mode is enabled
        run_learning(world)

    world.run()

    # Extract and save steel plant component operations data if available
    print("\n" + "=" * 100)
    print("EXTRACTING STEEL PLANT COMPONENT DATA")
    print("=" * 100)

    steel_unit = None
    for unit_id, unit in world.units.items():
        if hasattr(unit, "technology") and unit.technology == "steel_plant":
            steel_unit = unit
            break

    if (
        steel_unit
        and hasattr(steel_unit, "_component_operations")
        and len(steel_unit._component_operations) > 0
    ):
        output_dir = Path("steel_plant_component_analysis")
        output_dir.mkdir(exist_ok=True)

        # Save component data to CSV
        df = pd.DataFrame(steel_unit._component_operations)
        csv_file = output_dir / "steel_plant_component_hourly.csv"
        df.to_csv(csv_file, index=False)
        print(f"[OK] Saved component data to: {csv_file.absolute()}")
        print(f"[INFO] CSV contains {len(df)} rows and {len(df.columns)} columns")
        print(f"[INFO] Columns: {list(df.columns)}")

        # Print first 10 rows for validation
        print("\n[DATA SAMPLE] First 10 rows of steel production data:")
        print(
            df[["global_t", "timestamp", "eaf_steel_output", "eaf_power_input"]]
            .head(10)
            .to_string(index=False)
        )

        # Generate visualization if matplotlib is available
        if HAS_VISUALIZATION and "eaf_steel_output" in df.columns:
            fig, axes = plt.subplots(3, 1, figsize=(14, 10))
            fig.suptitle(
                "Steel Plant - Component Operations Analysis",
                fontsize=16,
                fontweight="bold",
            )

            # Plot 1: Steel Production (EAF Output)
            ax1 = axes[0]
            colors = ["green" if x > 0 else "lightgray" for x in df["eaf_steel_output"]]
            ax1.bar(
                df["global_t"],
                df["eaf_steel_output"],
                color=colors,
                alpha=0.7,
                width=0.8,
            )
            ax1.set_ylabel("Steel Output (MWh)", fontsize=11, fontweight="bold")
            ax1.set_title("EAF Steel Production Per Hour", fontsize=12)
            ax1.grid(True, alpha=0.3, axis="y")
            ax1.set_xlim(0, len(df))

            # Plot 2: Component Power Consumption
            ax2 = axes[1]
            width = 0.6
            ax2.bar(
                df["global_t"],
                df["eaf_power_input"],
                width=width,
                label="EAF Power",
                alpha=0.7,
                color="blue",
            )
            ax2.bar(
                df["global_t"],
                df["dri_power_input"],
                width=width,
                bottom=df["eaf_power_input"],
                label="DRI Power",
                alpha=0.7,
                color="orange",
            )
            ax2.bar(
                df["global_t"],
                df["electrolyser_power"],
                width=width,
                bottom=df["eaf_power_input"] + df["dri_power_input"],
                label="Electrolyser Power",
                alpha=0.7,
                color="purple",
            )
            ax2.set_ylabel("Power Input (MWh)", fontsize=11, fontweight="bold")
            ax2.set_title("Component Electricity Consumption Stacked View", fontsize=12)
            ax2.legend(loc="upper right")
            ax2.grid(True, alpha=0.3, axis="y")
            ax2.set_xlim(0, len(df))

            # Plot 3: Cumulative Steel Production
            cumulative_steel = df["eaf_steel_output"].cumsum()
            ax3 = axes[2]
            total_demand = df["eaf_steel_output"].sum()
            ax3.plot(
                df["global_t"],
                cumulative_steel,
                marker="o",
                linestyle="-",
                color="darkgreen",
                linewidth=2,
                markersize=4,
                label="Cumulative Steel",
            )
            ax3.fill_between(df["global_t"], cumulative_steel, alpha=0.3, color="green")
            ax3.axhline(
                y=total_demand,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Total Demand ({total_demand:.0f} MWh)",
            )
            ax3.set_ylabel("Cumulative Steel (MWh)", fontsize=11, fontweight="bold")
            ax3.set_xlabel("Hour", fontsize=11, fontweight="bold")
            ax3.set_title("Cumulative Steel Production vs Demand", fontsize=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, len(df))

            plt.tight_layout()

            # Save figure
            png_file = output_dir / "component_operations_visualization.png"
            plt.savefig(png_file, dpi=150, bbox_inches="tight")
            print(f"[OK] Saved visualization to: {png_file.absolute()}")
            plt.close()
        elif HAS_VISUALIZATION:
            print(
                "[WARNING] Matplotlib available but 'eaf_steel_output' column not found in data. Skipping visualization."
            )
        else:
            print("[INFO] Matplotlib not available - skipping visualization generation")

        # Print summary statistics
        print("\n[SUMMARY]")
        if "eaf_steel_output" in df.columns:
            total_steel = df["eaf_steel_output"].sum()
            production_hours = (df["eaf_steel_output"] > 0).sum()
            peak_hour = df["eaf_steel_output"].max()
            avg_hourly = (
                df[df["eaf_steel_output"] > 0]["eaf_steel_output"].mean()
                if production_hours > 0
                else 0
            )
            print(f"  Total Steel Output: {total_steel:.2f} MWh")
            print(f"  Production Hours: {production_hours}")
            print(f"  Peak Hour Output: {peak_hour:.2f} MWh")
            print(f"  Average per Production Hour: {avg_hourly:.2f} MWh")
        else:
            print(
                "  [WARNING] eaf_steel_output column not found in component operations data."
            )
    else:
        print(
            "[INFO] No steel plant with component operations data found in this simulation"
        )

    print("=" * 100)

# %%
