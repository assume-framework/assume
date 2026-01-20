#!/usr/bin/env python3
"""
Test script for three scenarios:
1. Utility Battery + PV
2. No Utility Battery + PV (prosumer, grid feed-in)
3. Utility Battery + No PV
"""

import logging
import sys
import os
import pandas as pd
import shutil
from pathlib import Path
from assume import World
from assume.scenario.loader_csv import load_scenario_folder

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Create output directory
OUTPUT_DIR = Path("test_energy_flow_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_scenario_csv(scenario_name: str, has_battery: bool, has_pv: bool):
    """Create a custom residential_dsm_units.csv for the scenario"""

    # Base components (EVs and charging stations)
    rows = [
        "name,unit_type,technology,node,bidding_EOM,fuel_type,unit_operator,objective,flexibility_measure,congestion_threshold,cost_tolerance,power_flow_directionality,mileage,charging_profile,availability_profile,demand,max_power,ramp_up,ramp_down,max_capacity,min_capacity,max_power_charge,max_power_discharge,initial_soc,efficiency_charge,efficiency_discharge,is_prosumer,uses_power_profile",
        "bangkok_1,bus_depot,electric_vehicle_1,north,naive_da_dsm,,dsm_operator_1,max_net_income,,0.8,10,bidirectional,0.0012,,Yes,,0.12,0.03,0.03,0.2,,,,0,1,1,Yes,",
        "bangkok_1,bus_depot,electric_vehicle_2,north,naive_da_dsm,,dsm_operator_1,max_net_income,,0.8,10,bidirectional,0.0012,,Yes,,0.12,0.03,0.03,0.2,,,,0,1,1,Yes,",
        "bangkok_1,bus_depot,charging_station_1,north,naive_da_dsm,,,,,,,bidirectional,,,Yes,,0.15,0.04,0.04,,,,,,,,,",
        "bangkok_1,bus_depot,charging_station_2,north,naive_da_dsm,,,,,,,bidirectional,,,Yes,,0.15,0.04,0.04,,,,,,,,,",
    ]

    # Add PV if needed
    if has_pv:
        rows.append("bangkok_1,bus_depot,pv_plant,north,naive_da_dsm,,dsm_operator_1,,,,,,,,Yes,,0.05,0,0,,,,,,,,,true")

    # Add utility battery if needed
    if has_battery:
        rows.append("bangkok_1,bus_depot,utility_battery,north,naive_da_dsm,,,,,,,,,,,,,,,0.5,0.0,0.25,0.25,0.5,0.95,0.95,,")

    # Write to temporary file
    temp_csv = OUTPUT_DIR / f"{scenario_name}_residential_dsm_units.csv"
    with open(temp_csv, "w") as f:
        f.write("\n".join(rows))

    return temp_csv


def save_energy_flow(world, scenario_name: str, depot):
    """Extract and save detailed energy flow data"""

    log.info(f"Extracting energy flow data for {scenario_name}...")

    # Get model instance
    model = depot.model
    time_steps = list(model.time_steps)

    # Initialize data dictionary
    data = {"time_step": time_steps}

    # Grid power
    data["grid_power"] = [model.grid_power[t].value for t in time_steps]

    # PV data (if exists)
    if depot.has_pv:
        data["pv_generation"] = [model.dsm_blocks["pv_plant"].power[t].value for t in time_steps]
        data["pv_used"] = [model.pv_used[t].value for t in time_steps]
        data["pv_surplus"] = [model.pv_surplus[t].value for t in time_steps]
    else:
        data["pv_generation"] = [0] * len(time_steps)
        data["pv_used"] = [0] * len(time_steps)
        data["pv_surplus"] = [0] * len(time_steps)

    # EV data
    for ev_key in model.dsm_blocks:
        if ev_key.startswith("electric_vehicle"):
            data[f"{ev_key}_charge"] = [model.dsm_blocks[ev_key].charge[t].value for t in time_steps]
            data[f"{ev_key}_discharge"] = [model.dsm_blocks[ev_key].discharge[t].value for t in time_steps]
            data[f"{ev_key}_soc"] = [model.dsm_blocks[ev_key].soc[t].value for t in time_steps]

    # CS data
    for cs_key in model.dsm_blocks:
        if cs_key.startswith("charging_station"):
            data[f"{cs_key}_discharge"] = [model.dsm_blocks[cs_key].discharge[t].value for t in time_steps]
            data[f"{cs_key}_charge"] = [model.dsm_blocks[cs_key].charge[t].value for t in time_steps]

    # Total CS charge (from EV discharge)
    cs_charge_total = [0] * len(time_steps)
    for cs_key in model.dsm_blocks:
        if cs_key.startswith("charging_station"):
            for i, t in enumerate(time_steps):
                cs_charge_total[i] += model.dsm_blocks[cs_key].charge[t].value
    data["total_cs_charge"] = cs_charge_total

    # Total power output
    data["total_power_output"] = [model.total_power_output[t].value for t in time_steps]

    # Utility battery (if exists)
    if depot.has_utility_battery:
        data["utility_battery_charge"] = [model.dsm_blocks["utility_battery"].charge[t].value for t in time_steps]
        data["utility_battery_discharge"] = [model.dsm_blocks["utility_battery"].discharge[t].value for t in time_steps]
        data["utility_battery_soc"] = [model.dsm_blocks["utility_battery"].soc[t].value for t in time_steps]
    else:
        data["utility_battery_charge"] = [0] * len(time_steps)
        data["utility_battery_discharge"] = [0] * len(time_steps)
        data["utility_battery_soc"] = [0] * len(time_steps)

    # Grid feed-in (if prosumer)
    if depot.is_prosumer and hasattr(model, 'grid_feed_in'):
        data["grid_feed_in"] = [model.grid_feed_in[t].value for t in time_steps]
    else:
        data["grid_feed_in"] = [0] * len(time_steps)

    # Financial data
    data["electricity_price"] = [model.electricity_price[t] for t in time_steps]
    data["variable_cost"] = [model.variable_cost[t].value for t in time_steps]
    data["variable_revenue"] = [model.variable_rev[t].value for t in time_steps]
    data["net_income"] = [model.net_income[t].value for t in time_steps]

    # Create DataFrame and save
    df = pd.DataFrame(data)
    csv_path = OUTPUT_DIR / f"{scenario_name}_energy_flow.csv"
    df.to_csv(csv_path, index=False)

    log.info(f"✓ Saved energy flow data to: {csv_path}")

    # Print key statistics
    log.info(f"\n{'=' * 60}")
    log.info(f"Key Statistics for {scenario_name}:")
    log.info(f"{'=' * 60}")
    log.info(f"Total Grid Power:          {df['grid_power'].sum():.2f} kWh")
    log.info(f"Total PV Generation:       {df['pv_generation'].sum():.2f} kWh")
    log.info(f"Total PV Used:             {df['pv_used'].sum():.2f} kWh")
    log.info(f"Total PV Surplus:          {df['pv_surplus'].sum():.2f} kWh")
    log.info(f"Total CS Charge:           {df['total_cs_charge'].sum():.2f} kWh")
    log.info(f"Total Power Output:        {df['total_power_output'].sum():.2f} kWh")
    log.info(f"Total Battery Charge:      {df['utility_battery_charge'].sum():.2f} kWh")
    log.info(f"Total Grid Feed-in:        {df['grid_feed_in'].sum():.2f} kWh")
    log.info(f"Total Cost:                {df['variable_cost'].sum():.2f} EUR")
    log.info(f"Total Revenue:             {df['variable_revenue'].sum():.2f} EUR")
    log.info(f"Net Income:                {df['net_income'].sum():.2f} EUR")
    log.info(f"{'=' * 60}\n")

    # Verify the formula: total_power_output = CS charge + PV surplus
    log.info(f"Verification:")
    for i in range(min(5, len(time_steps))):  # Show first 5 timesteps
        t = time_steps[i]
        cs_charge = data['total_cs_charge'][i]
        pv_surplus = data['pv_surplus'][i]
        total_output = data['total_power_output'][i]
        calculated = cs_charge + pv_surplus
        log.info(f"  t={t}: CS_charge={cs_charge:.4f} + PV_surplus={pv_surplus:.4f} = {calculated:.4f}, total_output={total_output:.4f}")

    return df


def test_scenario(scenario_name: str, has_battery: bool, has_pv: bool):
    """Test a specific scenario"""

    log.info("\n" + "=" * 80)
    log.info(f"TESTING: {scenario_name}")
    log.info(f"  - Utility Battery: {'YES' if has_battery else 'NO'}")
    log.info(f"  - PV System: {'YES' if has_pv else 'NO'}")
    log.info("=" * 80)

    try:
        # Create scenario CSV
        csv_file = create_scenario_csv(scenario_name, has_battery, has_pv)
        log.info(f"✓ Created scenario CSV: {csv_file}")

        # Copy to backup location temporarily
        backup_dir = Path("examples/inputs/backup")
        original_csv = backup_dir / "residential_dsm_units.csv"
        backup_csv = backup_dir / "residential_dsm_units.csv.temp_backup"

        # Backup original
        if original_csv.exists():
            shutil.copy(original_csv, backup_csv)

        # Copy scenario CSV to backup location
        shutil.copy(csv_file, original_csv)

        # Create world and load scenario
        db_uri = f"sqlite:///./{scenario_name}.db"
        world = World(database_uri=db_uri, export_csv_path="")

        load_scenario_folder(
            world,
            inputs_path="examples/inputs",
            scenario="backup",
            study_case="eom",
        )

        log.info("✓ Scenario loaded successfully")

        # Run simulation
        world.run()
        log.info("✓ Simulation completed successfully")

        # Find depot unit
        depot_units = [u for u in world.units if hasattr(u, 'technology') and u.technology == 'bus_depot']

        if depot_units:
            depot = depot_units[0]
            log.info(f"✓ Found depot unit: {depot.id}")
            log.info(f"  - Has utility battery: {depot.has_utility_battery}")
            log.info(f"  - Has PV: {depot.has_pv}")
            log.info(f"  - Is prosumer: {depot.is_prosumer}")

            # Save energy flow
            save_energy_flow(world, scenario_name, depot)

            # Restore original CSV
            if backup_csv.exists():
                shutil.copy(backup_csv, original_csv)
                backup_csv.unlink()

            return True
        else:
            log.error("✗ No depot units found!")
            return False

    except Exception as e:
        log.error(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

        # Restore original CSV on error
        backup_csv = Path("examples/inputs/backup/residential_dsm_units.csv.temp_backup")
        if backup_csv.exists():
            original_csv = Path("examples/inputs/backup/residential_dsm_units.csv")
            shutil.copy(backup_csv, original_csv)
            backup_csv.unlink()

        return False


def main():
    """Run all three test scenarios"""

    log.info("\n" + "=" * 80)
    log.info("STARTING THREE-SCENARIO TEST SUITE")
    log.info("=" * 80)

    results = {}

    # Test 1: Utility Battery + PV
    results["Test1_Battery_PV"] = test_scenario(
        scenario_name="test1_battery_pv",
        has_battery=True,
        has_pv=True
    )

    # Test 2: No Utility Battery + PV (grid feed-in)
    results["Test2_NoBattery_PV"] = test_scenario(
        scenario_name="test2_nobattery_pv",
        has_battery=False,
        has_pv=True
    )

    # Test 3: Utility Battery + No PV
    results["Test3_Battery_NoPV"] = test_scenario(
        scenario_name="test3_battery_nopv",
        has_battery=True,
        has_pv=False
    )

    # Summary
    log.info("\n" + "=" * 80)
    log.info("TEST SUITE SUMMARY")
    log.info("=" * 80)
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        log.info(f"{test_name}: {status}")
    log.info("=" * 80)
    log.info(f"\nResults saved to: {OUTPUT_DIR.absolute()}")

    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
