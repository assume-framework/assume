#!/usr/bin/env python3
"""
Sequential test for three scenarios
"""

import sys
import pandas as pd
from pathlib import Path
import shutil

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

from assume import World
from assume.scenario.loader_csv import load_scenario_folder

OUTPUT_DIR = Path("test_energy_flow_results")
OUTPUT_DIR.mkdir(exist_ok=True)

def create_scenario_csv(scenario_name, has_battery, has_pv):
    """Create CSV for scenario"""
    rows = [
        "name,unit_type,technology,node,bidding_EOM,fuel_type,unit_operator,objective,flexibility_measure,congestion_threshold,cost_tolerance,power_flow_directionality,mileage,charging_profile,availability_profile,demand,max_power,ramp_up,ramp_down,max_capacity,min_capacity,max_power_charge,max_power_discharge,initial_soc,efficiency_charge,efficiency_discharge,is_prosumer,uses_power_profile",
        "bangkok_1,bus_depot,electric_vehicle_1,north,naive_da_dsm,,dsm_operator_1,max_net_income,,0.8,10,bidirectional,0.0012,,Yes,,0.12,0.03,0.03,0.2,,,,0,1,1,Yes,",
        "bangkok_1,bus_depot,electric_vehicle_2,north,naive_da_dsm,,dsm_operator_1,max_net_income,,0.8,10,bidirectional,0.0012,,Yes,,0.12,0.03,0.03,0.2,,,,0,1,1,Yes,",
        "bangkok_1,bus_depot,charging_station_1,north,naive_da_dsm,,,,,,,bidirectional,,,Yes,,0.15,0.04,0.04,,,,,,,,,",
        "bangkok_1,bus_depot,charging_station_2,north,naive_da_dsm,,,,,,,bidirectional,,,Yes,,0.15,0.04,0.04,,,,,,,,,",
    ]

    if has_pv:
        rows.append("bangkok_1,bus_depot,pv_plant,north,naive_da_dsm,,dsm_operator_1,,,,,,,,Yes,,0.05,0,0,,,,,,,,,true")

    if has_battery:
        rows.append("bangkok_1,bus_depot,utility_battery,north,naive_da_dsm,,,,,,,,,,,,,,,0.5,0.0,0.25,0.25,0.5,0.95,0.95,,")

    csv_path = OUTPUT_DIR / f"{scenario_name}_dsm_units.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(rows))

    return csv_path


def extract_energy_flow(depot, scenario_name):
    """Extract energy flow from depot model"""
    model = depot.model
    time_steps = list(model.time_steps)

    data = {"timestep": time_steps}

    # Grid
    data["grid_import_kW"] = [model.grid_power[t].value if model.grid_power[t].value else 0 for t in time_steps]

    # PV
    if depot.has_pv:
        data["pv_available_kW"] = [model.dsm_blocks["pv_plant"].power[t].value for t in time_steps]
        data["pv_used_kW"] = [model.pv_used[t].value for t in time_steps]
        data["pv_surplus_kW"] = [model.pv_surplus[t].value for t in time_steps]
    else:
        data["pv_available_kW"] = [0] * len(time_steps)
        data["pv_used_kW"] = [0] * len(time_steps)
        data["pv_surplus_kW"] = [0] * len(time_steps)

    # CS charge (from EV discharge)
    cs_charge = [0] * len(time_steps)
    for cs_key in model.dsm_blocks:
        if cs_key.startswith("charging_station"):
            for i, t in enumerate(time_steps):
                cs_charge[i] += model.dsm_blocks[cs_key].charge[t].value
    data["total_cs_charge_kW"] = cs_charge

    # Total power output
    data["total_power_output_kW"] = [model.total_power_output[t].value for t in time_steps]

    # Utility battery
    if depot.has_utility_battery:
        data["battery_charge_kW"] = [model.dsm_blocks["utility_battery"].charge[t].value for t in time_steps]
        data["battery_discharge_kW"] = [model.dsm_blocks["utility_battery"].discharge[t].value for t in time_steps]
        data["battery_soc_kWh"] = [model.dsm_blocks["utility_battery"].soc[t].value for t in time_steps]
    else:
        data["battery_charge_kW"] = [0] * len(time_steps)
        data["battery_discharge_kW"] = [0] * len(time_steps)
        data["battery_soc_kWh"] = [0] * len(time_steps)

    # Grid feed-in
    if depot.is_prosumer and hasattr(model, 'grid_feed_in'):
        data["grid_feedin_kW"] = [model.grid_feed_in[t].value for t in time_steps]
    else:
        data["grid_feedin_kW"] = [0] * len(time_steps)

    # Financial
    data["variable_cost_EUR"] = [model.variable_cost[t].value for t in time_steps]
    data["variable_revenue_EUR"] = [model.variable_rev[t].value for t in time_steps]
    data["net_income_EUR"] = [model.net_income[t].value for t in time_steps]

    # Create DataFrame
    df = pd.DataFrame(data)
    csv_path = OUTPUT_DIR / f"{scenario_name}_energy_flow.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nSaved: {csv_path}")
    print(f"\nKey Statistics for {scenario_name}:")
    print(f"  Total Grid Import:        {df['grid_import_kW'].sum():.4f} kWh")
    print(f"  Total PV Available:       {df['pv_available_kW'].sum():.4f} kWh")
    print(f"  Total PV Used:            {df['pv_used_kW'].sum():.4f} kWh")
    print(f"  Total PV Surplus:         {df['pv_surplus_kW'].sum():.4f} kWh")
    print(f"  Total CS Charge:          {df['total_cs_charge_kW'].sum():.4f} kWh")
    print(f"  Total Power Output:       {df['total_power_output_kW'].sum():.4f} kWh")
    print(f"  Total Battery Charge:     {df['battery_charge_kW'].sum():.4f} kWh")
    print(f"  Total Grid Feed-in:       {df['grid_feedin_kW'].sum():.4f} kWh")
    print(f"  Total Cost:               {df['variable_cost_EUR'].sum():.2f} EUR")
    print(f"  Total Revenue:            {df['variable_revenue_EUR'].sum():.2f} EUR")
    print(f"  Net Income:               {df['net_income_EUR'].sum():.2f} EUR")

    # Verification
    print(f"\n  VERIFICATION: total_power_output = CS charge + PV surplus")
    for i in range(min(3, len(time_steps))):
        calculated = data['total_cs_charge_kW'][i] + data['pv_surplus_kW'][i]
        actual = data['total_power_output_kW'][i]
        match = "OK" if abs(calculated - actual) < 0.0001 else "FAIL"
        print(f"    t={i}: {data['total_cs_charge_kW'][i]:.4f} + {data['pv_surplus_kW'][i]:.4f} = {calculated:.4f} vs {actual:.4f} [{match}]")

    return df


def test_scenario(scenario_name, has_battery, has_pv):
    """Test one scenario"""
    print(f"\n{'='*80}")
    print(f"TEST: {scenario_name}")
    print(f"  Battery: {'YES' if has_battery else 'NO'}")
    print(f"  PV: {'YES' if has_pv else 'NO'}")
    print(f"{'='*80}")

    try:
        # Create CSV
        csv_file = create_scenario_csv(scenario_name, has_battery, has_pv)
        print(f"Created CSV: {csv_file}")

        # Backup original and replace
        backup_dir = Path("examples/inputs/backup")
        original = backup_dir / "residential_dsm_units.csv"
        temp_backup = backup_dir / "residential_dsm_units.csv.tempbackup"

        if original.exists():
            shutil.copy(original, temp_backup)
        shutil.copy(csv_file, original)

        # Create world
        db = f"{scenario_name}.db"
        world = World(database_uri=f"sqlite:///./{db}", export_csv_path="")

        print("Loading scenario...")
        load_scenario_folder(world, inputs_path="examples/inputs", scenario="backup", study_case="eom")

        # Find depot
        depot_units = [u for u in world.units if hasattr(u, 'technology') and u.technology == 'bus_depot']

        if not depot_units:
            print("ERROR: No depot found")
            return False

        depot = depot_units[0]
        print(f"Found depot: {depot.id}")
        print(f"  has_utility_battery: {depot.has_utility_battery}")
        print(f"  has_pv: {depot.has_pv}")
        print(f"  is_prosumer: {depot.is_prosumer}")

        # Run simulation
        print("\nRunning simulation...")
        world.run()
        print("Simulation complete!")

        # Extract data
        extract_energy_flow(depot, scenario_name)

        # Restore original
        if temp_backup.exists():
            shutil.copy(temp_backup, original)
            temp_backup.unlink()

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

        # Restore on error
        temp_backup = Path("examples/inputs/backup/residential_dsm_units.csv.tempbackup")
        if temp_backup.exists():
            original = Path("examples/inputs/backup/residential_dsm_units.csv")
            shutil.copy(temp_backup, original)
            temp_backup.unlink()

        return False


def main():
    print("SEQUENTIAL SCENARIO TESTING")
    print("="*80)

    results = {}

    # Test 1: Battery + PV
    results["Test1_Battery_PV"] = test_scenario("test1_battery_pv", has_battery=True, has_pv=True)

    # Test 2: Battery + No PV
    results["Test2_Battery_NoPV"] = test_scenario("test2_battery_nopv", has_battery=True, has_pv=False)

    # Test 3: No Battery + PV (last as requested)
    results["Test3_NoBattery_PV"] = test_scenario("test3_nobattery_pv", has_battery=False, has_pv=True)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {name}: {status}")
    print(f"\nResults saved to: {OUTPUT_DIR.absolute()}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
