#!/usr/bin/env python3
"""
Debug script to analyze depot logic and identify potential issues
"""

import pandas as pd
import numpy as np
import pyomo.environ as pyo

def debug_depot_logic():
    """Debug the depot logic by analyzing the data and constraints"""
    
    print("=== DEPOT LOGIC DEBUGGING ===\n")
    
    # 1. Analyze input data
    print("1. INPUT DATA ANALYSIS:")
    print("-" * 30)
    
    # Check unit configuration
    units_df = pd.read_csv("examples/inputs/example_bus/residential_dsm_units.csv")
    print("Unit Configuration:")
    for _, row in units_df.iterrows():
        if row['unit_type'] == 'bus_depot':
            print(f"  {row['technology']}: capacity={row['max_capacity']}kWh, power={row['max_power']}kW, mileage={row['mileage']}kWh/km, initial_soc={row['initial_soc']}")
    
    # Check forecasts
    forecasts_df = pd.read_csv("examples/inputs/example_bus/forecasts_df.csv")
    print(f"\nForecast Data ({len(forecasts_df)} timesteps):")
    print(f"  Electricity price range: {forecasts_df['electricity_price'].min():.2f} - {forecasts_df['electricity_price'].max():.2f} €/MWh")
    
    # Check range data
    range_cols = [col for col in forecasts_df.columns if 'range' in col and 'electric_vehicle' in col]
    print(f"  Range columns found: {len(range_cols)}")
    
    for col in range_cols[:2]:  # Check first 2 EVs
        range_values = forecasts_df[col].values
        print(f"  {col}: constant value = {range_values[0]} km")
        print(f"    Unique values: {np.unique(range_values)}")
    
    # Check availability profiles
    avail_cols = [col for col in forecasts_df.columns if 'availability_profile' in col and 'electric_vehicle' in col]
    for col in avail_cols[:2]:  # Check first 2 EVs
        avail_values = forecasts_df[col].values
        available_times = np.sum(avail_values)
        total_times = len(avail_values)
        print(f"  {col}: available {available_times}/{total_times} timesteps ({available_times/total_times*100:.1f}%)")
    
    # 2. Logic Analysis
    print(f"\n2. LOGIC ANALYSIS:")
    print("-" * 30)
    
    # Calculate required energy for travel
    mileage = 0.0012  # kWh/km from config
    range_km = 5  # km from forecast data
    required_energy = range_km * mileage
    
    print(f"Required energy calculation:")
    print(f"  Range: {range_km} km")
    print(f"  Mileage: {mileage} kWh/km")
    print(f"  Required energy: {required_energy} kWh")
    
    # Check if this makes sense with battery capacity
    battery_capacity = 0.35  # kWh from config
    initial_soc = 0  # from config
    
    print(f"\nBattery analysis:")
    print(f"  Battery capacity: {battery_capacity} kWh")
    print(f"  Initial SOC: {initial_soc} kWh")
    print(f"  Required energy: {required_energy} kWh")
    print(f"  Energy needed to charge: {required_energy - initial_soc} kWh")
    print(f"  Percentage of battery needed: {required_energy/battery_capacity*100:.1f}%")
    
    # 3. Potential Issues
    print(f"\n3. POTENTIAL ISSUES IDENTIFIED:")
    print("-" * 30)
    
    issues = []
    
    # Issue 1: Constant range values
    if range_km == 5:  # All timesteps have same range
        issues.append("WARNING: Range is constant (5 km) for all timesteps - EVs always need same energy")
    
    # Issue 2: Very low mileage
    if mileage < 0.01:
        issues.append(f"WARNING: Very low mileage ({mileage} kWh/km) - required energy is tiny ({required_energy} kWh)")
    
    # Issue 3: Required energy vs battery capacity
    if required_energy < battery_capacity * 0.1:
        issues.append(f"WARNING: Required energy ({required_energy} kWh) is only {required_energy/battery_capacity*100:.1f}% of battery - EVs need very little charging")
    
    # Issue 4: Initial SOC
    if initial_soc == 0:
        issues.append("INFO: Initial SOC is 0 - EVs start completely empty")
    
    # Issue 5: Always available
    all_available = all(forecasts_df[avail_cols[0]].values == 1)
    if all_available:
        issues.append("INFO: EVs are always available (never driving) - no dynamic behavior")
    
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  ✅ No obvious issues found")
    
    # 4. Recommendations
    print(f"\n4. RECOMMENDATIONS:")
    print("-" * 30)
    
    recommendations = []
    
    if required_energy < battery_capacity * 0.1:
        recommendations.append("FIX: Increase mileage value or range to create meaningful charging requirements")
    
    if range_km == 5:
        recommendations.append("FIX: Add varying range values across timesteps to simulate dynamic travel needs")
    
    if all_available:
        recommendations.append("FIX: Add periods where EVs are driving (availability=0) to test queue logic")
    
    if initial_soc == 0 and required_energy < 0.01:
        recommendations.append("FIX: Increase initial SOC or required energy to test charging continuation logic")
    
    recommendations.append("FIX: Test with one EV having higher travel requirements than the other")
    recommendations.append("FIX: Add time periods where EVs need different amounts of charging")
    
    for rec in recommendations:
        print(f"  {rec}")
    
    # 5. Expected vs Actual Behavior
    print(f"\n5. EXPECTED vs ACTUAL BEHAVIOR:")
    print("-" * 30)
    
    print("Expected behavior:")
    print("  - EV1 should start charging at timestep 0")
    print("  - EV2 should wait in queue")
    print("  - EV1 should continue charging until sufficient for travel")
    print("  - Then switch to EV2")
    
    print(f"\nActual behavior analysis:")
    print(f"  - With {required_energy} kWh needed and {units_df.loc[1, 'max_power']} kW charging rate")
    charging_time = required_energy / units_df.loc[1, 'max_power']
    print(f"  - Charging time needed: {charging_time:.2f} hours")
    print(f"  - At 15-min intervals: {charging_time * 4:.1f} timesteps")
    
    if charging_time < 0.25:  # Less than 15 minutes
        print("  WARNING: Charging completes in <1 timestep - may not see continuation behavior")

if __name__ == "__main__":
    try:
        debug_depot_logic()
    except Exception as e:
        print(f"Error during debugging: {e}")
        import traceback
        traceback.print_exc()