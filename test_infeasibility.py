#!/usr/bin/env python3
"""Detailed infeasibility investigation"""

import logging
import sys
import pyomo.environ as pyo
from assume import World
from assume.scenario.loader_csv import load_scenario_folder

# Reduce logging noise
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

try:
    # Create world
    db_uri = "sqlite:///./test_infeas.db"
    world = World(database_uri=db_uri, export_csv_path="")

    # Load scenario
    load_scenario_folder(
        world,
        inputs_path="examples/inputs",
        scenario="backup",
        study_case="eom",
    )

    print("SUCCESS: Loaded without errors")

except Exception as e:
    print(f"\nERROR: {e}\n")

    # Try to inspect the depot before it fails
    try:
        import traceback
        tb_lines = traceback.format_exc().split('\n')

        # Find if we can get to the depot object
        print("=" * 80)
        print("INVESTIGATING INFEASIBILITY")
        print("=" * 80)

        # Check if it's a Pyomo solver issue
        if "infeasible" in str(e).lower():
            print("\n⚠️  MODEL IS INFEASIBLE - This means constraints are contradictory")
            print("\nPossible causes:")
            print("1. Grid feed-in variable not created (check is_prosumer)")
            print("2. PV curtailment not working")
            print("3. Energy balance equation issue")
            print("4. Missing constraint")

            # Try to get depot info from world
            if hasattr(world, 'units'):
                depot_units = [u for u in world.units if hasattr(u, 'technology') and str(u.technology).startswith('bus_depot')]

                if depot_units:
                    depot = depot_units[0]
                    print(f"\n📊 DEPOT CONFIGURATION:")
                    print(f"  - ID: {depot.id}")
                    print(f"  - is_prosumer: {depot.is_prosumer}")
                    print(f"  - has_pv: {depot.has_pv}")
                    print(f"  - has_utility_battery: {depot.has_utility_battery}")
                    print(f"  - Components: {list(depot.components.keys())}")

                    # Check model variables
                    if hasattr(depot, 'model'):
                        m = depot.model
                        print(f"\n🔍 MODEL VARIABLES:")
                        print(f"  - grid_power: {hasattr(m, 'grid_power')}")
                        print(f"  - grid_feedin: {hasattr(m, 'grid_feedin')}")
                        print(f"  - pv_used: {hasattr(m, 'pv_used')}")
                        print(f"  - pv_curtail: {hasattr(m, 'pv_curtail')}")
                        print(f"  - total_power_input: {hasattr(m, 'total_power_input')}")
                        print(f"  - total_power_output: {hasattr(m, 'total_power_output')}")

                        # Check constraints
                        print(f"\n🔒 MODEL CONSTRAINTS:")
                        constraint_names = [name for name in dir(m) if not name.startswith('_')]
                        relevant_constraints = [c for c in constraint_names if 'constraint' in c.lower() or 'balance' in c.lower()]
                        for c in relevant_constraints[:10]:  # First 10
                            print(f"  - {c}")

                        # Try to write model to file for inspection
                        try:
                            m.pprint(filename="model_debug.txt")
                            print(f"\n✅ Model written to: model_debug.txt")
                        except:
                            pass

    except Exception as e2:
        print(f"\nCouldn't inspect depot: {e2}")

    traceback.print_exc()
    sys.exit(1)
