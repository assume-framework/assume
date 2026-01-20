#!/usr/bin/env python3
"""
Simple test to extract energy flow data from depot
"""

import sys
import pandas as pd
from pathlib import Path

# Test if model already exists
db_path = Path("test1_battery_pv.db")
if not db_path.exists():
    print(f"ERROR: Database {db_path} not found!")
    print("Run the test_three_scenarios.py first")
    sys.exit(1)

# Import after checking database
from assume import World
from assume.scenario.loader_csv import load_scenario_folder

print("Creating world...")
world = World(database_uri=f"sqlite:///./{db_path}", export_csv_path="")

print("Loading scenario...")
load_scenario_folder(
    world,
    inputs_path="examples/inputs",
    scenario="backup",
    study_case="eom",
)

print("Finding depot unit...")
depot_units = [u for u in world.units if hasattr(u, 'technology') and u.technology == 'bus_depot']

if not depot_units:
    print("ERROR: No depot units found!")
    sys.exit(1)

depot = depot_units[0]
print(f"Found depot: {depot.id}")
print(f"  - Has utility battery: {depot.has_utility_battery}")
print(f"  - Has PV: {depot.has_pv}")
print(f"  - Is prosumer: {depot.is_prosumer}")

# Extract data from model
model = depot.model
time_steps = list(model.time_steps)

print(f"\nExtracting energy flow data for {len(time_steps)} time steps...")

data = {"time_step": time_steps}

# Grid power
try:
    data["grid_power"] = [model.grid_power[t].value for t in time_steps]
except:
    data["grid_power"] = [0] * len(time_steps)

# PV data
if depot.has_pv:
    try:
        data["pv_generation"] = [model.dsm_blocks["pv_plant"].power[t].value for t in time_steps]
        data["pv_used"] = [model.pv_used[t].value for t in time_steps]
        data["pv_surplus"] = [model.pv_surplus[t].value for t in time_steps]
    except Exception as e:
        print(f"Warning: Could not extract PV data: {e}")
        data["pv_generation"] = [0] * len(time_steps)
        data["pv_used"] = [0] * len(time_steps)
        data["pv_surplus"] = [0] * len(time_steps)
else:
    data["pv_generation"] = [0] * len(time_steps)
    data["pv_used"] = [0] * len(time_steps)
    data["pv_surplus"] = [0] * len(time_steps)

# Total CS charge (from EV discharge)
cs_charge_total = [0] * len(time_steps)
for cs_key in model.dsm_blocks:
    if cs_key.startswith("charging_station"):
        try:
            for i, t in enumerate(time_steps):
                cs_charge_total[i] += model.dsm_blocks[cs_key].charge[t].value
        except:
            pass
data["total_cs_charge"] = cs_charge_total

# Total power output
try:
    data["total_power_output"] = [model.total_power_output[t].value for t in time_steps]
except:
    data["total_power_output"] = [0] * len(time_steps)

# Utility battery
if depot.has_utility_battery:
    try:
        data["utility_battery_charge"] = [model.dsm_blocks["utility_battery"].charge[t].value for t in time_steps]
        data["utility_battery_discharge"] = [model.dsm_blocks["utility_battery"].discharge[t].value for t in time_steps]
        data["utility_battery_soc"] = [model.dsm_blocks["utility_battery"].soc[t].value for t in time_steps]
    except Exception as e:
        print(f"Warning: Could not extract battery data: {e}")
        data["utility_battery_charge"] = [0] * len(time_steps)
        data["utility_battery_discharge"] = [0] * len(time_steps)
        data["utility_battery_soc"] = [0] * len(time_steps)
else:
    data["utility_battery_charge"] = [0] * len(time_steps)
    data["utility_battery_discharge"] = [0] * len(time_steps)
    data["utility_battery_soc"] = [0] * len(time_steps)

# Grid feed-in
if depot.is_prosumer and hasattr(model, 'grid_feed_in'):
    try:
        data["grid_feed_in"] = [model.grid_feed_in[t].value for t in time_steps]
    except:
        data["grid_feed_in"] = [0] * len(time_steps)
else:
    data["grid_feed_in"] = [0] * len(time_steps)

# Create DataFrame and save
df = pd.DataFrame(data)
output_dir = Path("test_energy_flow_results")
output_dir.mkdir(exist_ok=True)

csv_path = output_dir / "simple_energy_flow.csv"
df.to_csv(csv_path, index=False)

print(f"\nSaved to: {csv_path}")
print(f"\nKey Statistics:")
print(f"  Total Grid Power:          {df['grid_power'].sum():.2f} kWh")
print(f"  Total PV Generation:       {df['pv_generation'].sum():.2f} kWh")
print(f"  Total PV Used:             {df['pv_used'].sum():.2f} kWh")
print(f"  Total PV Surplus:          {df['pv_surplus'].sum():.2f} kWh")
print(f"  Total CS Charge:           {df['total_cs_charge'].sum():.2f} kWh")
print(f"  Total Power Output:        {df['total_power_output'].sum():.2f} kWh")
print(f"  Total Battery Charge:      {df['utility_battery_charge'].sum():.2f} kWh")
print(f"  Total Grid Feed-in:        {df['grid_feed_in'].sum():.2f} kWh")

print(f"\nVerification (first 5 timesteps):")
print(f"  Formula: total_power_output = CS charge + PV surplus")
for i in range(min(5, len(time_steps))):
    t = time_steps[i]
    cs_charge = data['total_cs_charge'][i]
    pv_surplus = data['pv_surplus'][i]
    total_output = data['total_power_output'][i]
    calculated = cs_charge + pv_surplus
    match = "OK" if abs(calculated - total_output) < 0.001 else "MISMATCH"
    print(f"  t={t}: {cs_charge:.4f} + {pv_surplus:.4f} = {calculated:.4f} vs {total_output:.4f} [{match}]")
