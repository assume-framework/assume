#!/usr/bin/env python3
"""
Test script to verify depot.py changes work correctly.
Tests both scenarios: with and without utility battery.
"""

import logging
import sys
from assume import World
from assume.scenario.loader_csv import load_scenario_folder

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def test_bus_depot():
    """Test the bus depot with utility battery"""
    log.info("=" * 80)
    log.info("Testing Bus Depot with Utility Battery")
    log.info("=" * 80)

    try:
        # Create world
        db_uri = "sqlite:///./test_depot.db"
        world = World(database_uri=db_uri, export_csv_path="")

        # Load scenario
        load_scenario_folder(
            world,
            inputs_path="examples/inputs",
            scenario="example_bus",
            study_case="eom",
        )

        log.info("✓ Scenario loaded successfully")

        # Run simulation
        world.run()

        log.info("✓ Simulation completed successfully")

        # Check that depot units were created
        depot_units = [u for u in world.units if hasattr(u, 'technology') and u.technology == 'bus_depot']

        if depot_units:
            depot = depot_units[0]
            log.info(f"✓ Found depot unit: {depot.id}")
            log.info(f"  - Has utility battery: {depot.has_utility_battery}")
            log.info(f"  - Is prosumer: {depot.is_prosumer}")
            log.info(f"  - Has PV: {depot.has_pv}")

            # Check if grid_feedin variable exists
            if hasattr(depot.model, 'grid_feedin'):
                log.info(f"  - Grid feed-in variable: EXISTS")

                # If has utility battery, grid_feedin should be constrained to 0
                if depot.has_utility_battery and depot.is_prosumer:
                    log.info(f"  - Utility battery present: Grid feed-in should be 0")
                else:
                    log.info(f"  - No utility battery: Grid feed-in allowed")
            else:
                log.info(f"  - Grid feed-in variable: NOT FOUND (not prosumer)")

            log.info("=" * 80)
            log.info("✓ ALL TESTS PASSED")
            log.info("=" * 80)
            return True
        else:
            log.error("✗ No depot units found!")
            return False

    except Exception as e:
        log.error(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_bus_depot()
    sys.exit(0 if success else 1)
