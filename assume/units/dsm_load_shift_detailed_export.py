"""
Export detailed energy flow function for DSM_load_shift.py
This should be added to the DSM_load_shift class
"""

def export_detailed_energy_flow(self, instance, save_path="detailed_energy_flow.csv"):
    """
    Export detailed energy flow data to CSV

    Includes:
    - Timesteps
    - Electricity prices
    - PV generation, usage, and surplus
    - Battery charge/discharge/SOC
    - Grid import/feed-in
    - Charging station discharge/charge
    - EV charge/discharge/SOC/usage/availability
    - Total power input/output
    - Financial data (cost, revenue, net income)
    """
    import pandas as pd
    import pyomo.environ as pyo

    # Get time steps
    time_steps = list(instance.time_steps)

    # Initialize data dictionary
    data = {"timestep": time_steps}

    # Helper function to safely get values
    def safe_value(var):
        try:
            return float(pyo.value(var))
        except:
            return 0.0

    # ===== ELECTRICITY PRICES =====
    data["electricity_price_EUR_MWh"] = [
        safe_value(instance.electricity_price[t]) for t in time_steps
    ]
    data["electricity_price_flex_EUR_MWh"] = [
        safe_value(instance.electricity_price_flex[t]) for t in time_steps
    ]

    # ===== PV DATA =====
    if hasattr(self, 'has_pv') and self.has_pv:
        data["pv_available_kW"] = [
            safe_value(instance.dsm_blocks["pv_plant"].power[t]) for t in time_steps
        ]
        data["pv_used_kW"] = [
            safe_value(instance.pv_used[t]) for t in time_steps
        ]
        # NEW: Use pv_surplus instead of pv_curtail
        data["pv_surplus_kW"] = [
            safe_value(instance.pv_surplus[t]) for t in time_steps
        ]
        # Calculate PV utilization
        data["pv_utilization_percent"] = [
            (data["pv_used_kW"][i] / data["pv_available_kW"][i] * 100)
            if data["pv_available_kW"][i] > 0 else 0
            for i in range(len(time_steps))
        ]
    else:
        data["pv_available_kW"] = [0] * len(time_steps)
        data["pv_used_kW"] = [0] * len(time_steps)
        data["pv_surplus_kW"] = [0] * len(time_steps)
        data["pv_utilization_percent"] = [0] * len(time_steps)

    # ===== UTILITY BATTERY DATA =====
    if hasattr(self, 'has_utility_battery') and self.has_utility_battery:
        battery = instance.dsm_blocks["utility_battery"]
        data["battery_charge_kW"] = [safe_value(battery.charge[t]) for t in time_steps]
        data["battery_discharge_kW"] = [safe_value(battery.discharge[t]) for t in time_steps]
        data["battery_soc_kWh"] = [safe_value(battery.soc[t]) for t in time_steps]

        # Calculate battery SOC percentage
        max_capacity = float(battery.max_capacity) if hasattr(battery, 'max_capacity') else 1.0
        data["battery_soc_percent"] = [
            (data["battery_soc_kWh"][i] / max_capacity * 100)
            for i in range(len(time_steps))
        ]

        # Battery net flow (negative = charging, positive = discharging)
        data["battery_net_flow_kW"] = [
            data["battery_discharge_kW"][i] - data["battery_charge_kW"][i]
            for i in range(len(time_steps))
        ]
    else:
        data["battery_charge_kW"] = [0] * len(time_steps)
        data["battery_discharge_kW"] = [0] * len(time_steps)
        data["battery_soc_kWh"] = [0] * len(time_steps)
        data["battery_soc_percent"] = [0] * len(time_steps)
        data["battery_net_flow_kW"] = [0] * len(time_steps)

    # ===== GRID DATA =====
    data["grid_import_kW"] = [safe_value(instance.grid_power[t]) for t in time_steps]

    # Grid feed-in
    if hasattr(self, 'is_prosumer') and self.is_prosumer and hasattr(instance, 'grid_feed_in'):
        data["grid_feedin_kW"] = [safe_value(instance.grid_feed_in[t]) for t in time_steps]
    else:
        data["grid_feedin_kW"] = [0] * len(time_steps)

    # Grid net flow (negative = import, positive = export)
    data["grid_net_flow_kW"] = [
        data["grid_feedin_kW"][i] - data["grid_import_kW"][i]
        for i in range(len(time_steps))
    ]

    # ===== CHARGING STATION DATA =====
    cs_discharge_total = [0] * len(time_steps)
    cs_charge_total = [0] * len(time_steps)

    for cs_key in instance.dsm_blocks:
        if cs_key.startswith("charging_station"):
            cs_block = instance.dsm_blocks[cs_key]

            # CS discharge (to EVs)
            cs_discharge = [safe_value(cs_block.discharge[t]) for t in time_steps]
            data[f"{cs_key}_discharge_kW"] = cs_discharge

            # CS charge (from EVs, V2G)
            cs_charge = [safe_value(cs_block.charge[t]) for t in time_steps]
            data[f"{cs_key}_charge_kW"] = cs_charge

            # Add to totals
            for i in range(len(time_steps)):
                cs_discharge_total[i] += cs_discharge[i]
                cs_charge_total[i] += cs_charge[i]

    data["total_cs_discharge_kW"] = cs_discharge_total
    data["total_cs_charge_kW"] = cs_charge_total

    # ===== ELECTRIC VEHICLE DATA =====
    for ev_key in instance.dsm_blocks:
        if ev_key.startswith("electric_vehicle"):
            ev_block = instance.dsm_blocks[ev_key]

            # EV charge
            data[f"{ev_key}_charge_kW"] = [safe_value(ev_block.charge[t]) for t in time_steps]

            # EV discharge
            data[f"{ev_key}_discharge_kW"] = [safe_value(ev_block.discharge[t]) for t in time_steps]

            # EV SOC
            data[f"{ev_key}_soc_kWh"] = [safe_value(ev_block.soc[t]) for t in time_steps]

            # EV SOC percentage
            max_capacity = float(ev_block.max_capacity) if hasattr(ev_block, 'max_capacity') else 1.0
            data[f"{ev_key}_soc_percent"] = [
                (data[f"{ev_key}_soc_kWh"][i] / max_capacity * 100)
                for i in range(len(time_steps))
            ]

            # EV usage
            data[f"{ev_key}_usage_kW"] = [safe_value(ev_block.usage[t]) for t in time_steps]

            # EV availability
            ev_availability = getattr(instance, f"{ev_key}_availability", None)
            if ev_availability is not None:
                data[f"{ev_key}_available"] = [ev_availability[t] for t in time_steps]
            else:
                data[f"{ev_key}_available"] = [1] * len(time_steps)

    # ===== TOTAL POWER INPUT/OUTPUT =====
    data["total_power_input_kW"] = [safe_value(instance.total_power_input[t]) for t in time_steps]
    data["total_power_output_kW"] = [safe_value(instance.total_power_output[t]) for t in time_steps]

    # Power balance check (should be close to 0)
    data["power_balance_check_kW"] = [
        data["total_power_input_kW"][i] - data["total_power_output_kW"][i]
        for i in range(len(time_steps))
    ]

    # ===== FINANCIAL DATA =====
    data["variable_cost_EUR"] = [safe_value(instance.variable_cost[t]) for t in time_steps]
    data["variable_revenue_EUR"] = [safe_value(instance.variable_rev[t]) for t in time_steps]
    data["net_income_EUR"] = [safe_value(instance.net_income[t]) for t in time_steps]

    # ===== INPUT SOURCE BREAKDOWN =====
    # Calculate percentage of input from each source
    for i in range(len(time_steps)):
        total_input = data["total_power_input_kW"][i]
        if total_input > 0:
            data.setdefault("input_from_pv_percent", []).append(
                (data["pv_used_kW"][i] / total_input * 100)
            )
            data.setdefault("input_from_grid_percent", []).append(
                (data["grid_import_kW"][i] / total_input * 100)
            )
            data.setdefault("input_from_battery_percent", []).append(
                (data["battery_discharge_kW"][i] / total_input * 100)
            )
        else:
            data.setdefault("input_from_pv_percent", []).append(0)
            data.setdefault("input_from_grid_percent", []).append(0)
            data.setdefault("input_from_battery_percent", []).append(0)

    # ===== CREATE DATAFRAME AND SAVE =====
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)

    print(f"\nDetailed energy flow exported to: {save_path}")

    # Print summary statistics
    print("\nEnergy Flow Summary:")
    print(f"  Total PV Available:       {df['pv_available_kW'].sum():.2f} kWh")
    print(f"  Total PV Used:            {df['pv_used_kW'].sum():.2f} kWh")
    print(f"  Total PV Surplus:         {df['pv_surplus_kW'].sum():.2f} kWh")
    print(f"  Total Grid Import:        {df['grid_import_kW'].sum():.2f} kWh")
    print(f"  Total Grid Feed-in:       {df['grid_feedin_kW'].sum():.2f} kWh")
    print(f"  Total Battery Charge:     {df['battery_charge_kW'].sum():.2f} kWh")
    print(f"  Total Battery Discharge:  {df['battery_discharge_kW'].sum():.2f} kWh")
    print(f"  Total CS Discharge:       {df['total_cs_discharge_kW'].sum():.2f} kWh")
    print(f"  Total CS Charge (V2G):    {df['total_cs_charge_kW'].sum():.2f} kWh")
    print(f"  Total Cost:               {df['variable_cost_EUR'].sum():.2f} EUR")
    print(f"  Total Revenue:            {df['variable_revenue_EUR'].sum():.2f} EUR")
    print(f"  Net Income:               {df['net_income_EUR'].sum():.2f} EUR")

    return df
