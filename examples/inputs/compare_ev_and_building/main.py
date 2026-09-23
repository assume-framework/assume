# Import
# TestCase 1: A single EV unidirectional charging scenario
# ElectricVehicle Unit.

import time

import numpy as np

from assume.world import World
from assume.scenario.loader_csv import load_config_and_create_forecaster, setup_world
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import matplotlib.dates as mdates


input_dir = Path(__file__).parent / "scenario_data"
output_dir = Path(__file__).parent / "output_data"

world_1 = World(export_csv_path=str(output_dir))
world_1.scenario_data = load_config_and_create_forecaster(
    inputs_path=str(input_dir),
    scenario="test_case_2_ev",
    study_case="no_tariff"  # What is the study_case needed for?
)

setup_world(world_1)
time_0 = time.time()
world_1.run()
time_1 = time.time()


# Building.
world_2 = World(export_csv_path=str(output_dir))
world_2.scenario_data = load_config_and_create_forecaster(
    inputs_path=str(input_dir),
    scenario="test_case_2_building",
    study_case="no_tariff"
)

setup_world(world_2)
time_2 = time.time()
world_2.run()
time_3 = time.time()
print(f"EV scenario run time: {time_1 - time_0} seconds")
print(f"Building scenario run time: {time_3 - time_2} seconds")



# Results
### Scenario: Building
market_orders_df_building = pd.read_csv(
    output_dir / "test_case_2_building_no_tariff" / "market_orders.csv",
    parse_dates=["start_time"], index_col="start_time")
ev0_bids_building = market_orders_df_building.query("unit_id == 'aggregator'")

unit_dispatch_df_building = pd.read_csv(
    output_dir / "test_case_2_building_no_tariff" / "unit_dispatch.csv",
    parse_dates=["time"])
unit_dispatch_df_building = unit_dispatch_df_building[unit_dispatch_df_building["time"].isin(market_orders_df_building.index)]

forecast_df_building = pd.read_csv(
    input_dir / "test_case_2_building" / "forecasts_df.csv",
    parse_dates=["datetime"], index_col="datetime")
forecast_df_building = forecast_df_building[forecast_df_building.index.isin(market_orders_df_building.index)]

### Scenario: EV
market_orders_df_ev = pd.read_csv(
    output_dir / "test_case_2_ev_no_tariff" / "market_orders.csv",
    parse_dates=["start_time"], index_col="start_time")
ev0_bids_ev = market_orders_df_ev.query("unit_id == 'EV_0'")

unit_dispatch_df_ev = pd.read_csv(
    output_dir / "test_case_2_ev_no_tariff" / "unit_dispatch.csv",
    parse_dates=["time"])
unit_dispatch_df_ev = unit_dispatch_df_ev[unit_dispatch_df_ev["time"].isin(market_orders_df_ev.index)]

forecast_df_ev = pd.read_csv(
    input_dir / "test_case_2_ev" / "forecasts_df.csv",
    parse_dates=["datetime"], index_col="datetime")
forecast_df_ev = forecast_df_ev[forecast_df_ev.index.isin(market_orders_df_ev.index)]

ev0_dispatch_df_ev = unit_dispatch_df_ev.query("unit == 'EV_0'")
expected_costs_ev = (ev0_dispatch_df_ev["power"].values * forecast_df_ev["price_EOM"].values)
actual_costs_ev = (ev0_bids_ev["accepted_volume"].values * ev0_bids_ev["accepted_price"].values)

ev0_dispatch_df_building = unit_dispatch_df_building.query("unit == 'aggregator'")
expected_costs_building = (ev0_dispatch_df_building["power"].values * forecast_df_building["price_EOM"].values)
actual_costs_building = (ev0_bids_building["accepted_volume"].values * ev0_bids_building["accepted_price"].values)

plot_df = pd.DataFrame({
    "time": ev0_dispatch_df_ev["time"],
    "expected_costs_ev": expected_costs_ev,
    "actual_costs_ev": actual_costs_ev,
    "expected_costs_building": expected_costs_building,
    "actual_costs_building": actual_costs_building,
})

"""sns.ecdfplot(data=plot_df, x="time", weights="expected_costs_building", ax=axes[0], label="Expected Costs")
sns.ecdfplot(data=plot_df, x="time", weights="actual_costs_building", ax=axes[0], label="Actual Costs")
axes[0].legend()

sns.ecdfplot(data=plot_df, x="time", weights="expected_costs_ev", ax=axes[1], label="Expected Costs")
sns.ecdfplot(data=plot_df, x="time", weights="actual_costs_ev", ax=axes[1], label="Actual Costs")
axes[1].legend()"""

trip_energy_ev = forecast_df_ev["EV_0_trip_energy_consumption"].values
trip_energy_building = forecast_df_building["aggregator_electric_vehicle_EV_0_trip_energy_consumption"].values

soc_building = [0.6]
for idx in range(len(ev0_dispatch_df_ev["power"].values)):
    delta = 10 * ev0_dispatch_df_ev["power"].values[idx] + 10 * trip_energy_building[idx]
    soc_building.append(soc_building[-1] - delta)

soc_ev = [0.6]
for idx in range(len(ev0_dispatch_df_ev["power"].values)):
    delta = 10 * ev0_dispatch_df_ev["power"].values[idx] + 10 * trip_energy_ev[idx]
    soc_ev.append(soc_ev[-1] - delta)

soc_final_ev = ev0_dispatch_df_ev["soc"].iloc[-1] - 10 * ev0_dispatch_df_ev["power"].iloc[-1]

plot_df["soc_building"] = soc_building[1:]
plot_df["soc_ev"] = soc_ev[1:]

prices = forecast_df_ev["price_EOM"].values
plot_df["price"] = prices

"""
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
sns.lineplot(data=plot_df, x="time", y="soc_building", ax=axes[0], label="SOC Building")
sns.lineplot(data=plot_df, x="time", y="soc_ev", ax=axes[1], label="SOC EV")
sns.lineplot(data=plot_df, x="time", y=[soc_final_ev]*len(plot_df), ax=axes[1], label="SOC Final EV")

import matplotlib.dates as mdates

# Set ticks every 3 hours (adjust 'hours' to 'days' if applicable)
axes[0].xaxis.set_major_locator(mdates.HourLocator(interval=4))
axes[1].xaxis.set_major_locator(mdates.HourLocator(interval=4))

# Format how the date/time string is displayed (e.g., "12:00")
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H'))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H'))

axes[0].set_ylabel("SOC")
axes[1].set_ylabel("SOC")"""
# Ensure chronological order
# import pdb; pdb.set_trace()
"""
plot_df = plot_df.sort_values("time")
axes[0].set_title("Using Building")
# Building Costs
axes[0].step(
    plot_df["time"],
    plot_df["expected_costs_building"].cumsum(),
    where="post",
    label="Expected Costs",
)
axes[0].step(
    plot_df["time"],
    plot_df["actual_costs_building"].cumsum(),
    where="post",
    label="Actual Costs",
)
axes[0].legend()

axes[1].set_title("Using EV")

# EV Costs
axes[1].step(
    plot_df["time"],
    plot_df["expected_costs_ev"].cumsum(),
    where="post",
    label="Expected Costs",
)
axes[1].step(
    plot_df["time"],
    plot_df["actual_costs_ev"].cumsum(),
    where="post",
    label="Actual Costs",
)
axes[1].legend()
"""
# sns.lineplot(data=market_orders_df, x="start_time", y="accepted_volume", hue="bid_id", ax=axes[0])
# sns.lineplot(data=market_orders_df, x="start_time", y="volume", hue="bid_id", ax=axes[1])
# Create a 4-row layout: [LinePlot1, Heatmap1, LinePlot2, Heatmap2]
# height_ratios give thin rows (0.15) for the heatmaps

def plot_soc_and_price(plot_df, soc_final_ev):
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 0.5]},
    )

    # 1. Top Subplot: SOC Building Line Plot
    sns.lineplot(
        data=plot_df, x="time", y="soc_building", ax=axes[0], label="SOC Building"
    )
    axes[0].set_ylabel("SOC")
    axes[0].set_xlabel("")

    # Convert x-limits to numerical dates for imshow extent alignment
    x_min = mdates.date2num(plot_df["time"].min())
    x_max = mdates.date2num(plot_df["time"].max())
    price_matrix = plot_df["price"].values.reshape(1, -1)

    # 3. Bottom Subplot: SOC EV Line Plots
    sns.lineplot(
        data=plot_df, x="time", y="soc_ev", ax=axes[1], label="SOC EV"
    )
    sns.lineplot(
        data=plot_df,
        x="time",
        y=[soc_final_ev] * len(plot_df),
        ax=axes[1],
        label="SOC Final EV",
    )
    axes[1].set_ylabel("SOC")
    axes[1].set_xlabel("")

    # 3. Bottom Heatmap: Price strip below second plot
    sns.lineplot(data=plot_df, x="time", y="price", ax=axes[2], label="Price")
    axes[2].set_ylabel("Price", rotation=0, labelpad=25, va="center", fontsize=9)

    # Set tick marks every 3 hours
    axes[2].xaxis.set_major_locator(mdates.HourLocator(interval=3))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    axes[2].set_xlabel("Time", fontsize=9)

    # Clean up layout padding and remove extra margins
    for ax in axes:
        ax.margins(x=0)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.12)
    plt.show()

plot_soc_and_price(plot_df, soc_final_ev)

# TestCase 2: A single EV bidirectional charging scenario
# TestCase 3: 2 EVs unidirectional charging scenario
# TestCase 4: 2 EVs bidirectional charging scenario
# TestCase 5: Multiple EVs unidirectional charging scenario
# TestCase 6: Multiple EVs mixed charging scenario