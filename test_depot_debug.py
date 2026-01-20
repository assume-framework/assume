#!/usr/bin/env python3
"""Debug script to understand infeasibility"""

import logging
import sys
from assume import World
from assume.scenario.loader_csv import load_scenario_folder

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

try:
    # Create world
    db_uri = "sqlite:///./test_debug.db"
    world = World(database_uri=db_uri, export_csv_path="")

    # Load scenario
    load_scenario_folder(
        world,
        inputs_path="examples/inputs",
        scenario="backup",
        study_case="eom",
    )

except Exception as e:
    log.error(f"Failed: {e}")

    # Try to get depot info
    try:
        depot_units = [u for u in world.units if hasattr(u, 'technology') and u.technology == 'bus_depot']
        if depot_units:
            depot = depot_units[0]
            log.info(f"Depot ID: {depot.id}")
            log.info(f"Is Prosumer: {depot.is_prosumer}")
            log.info(f"Has PV: {depot.has_pv}")
            log.info(f"Has Utility Battery: {depot.has_utility_battery}")
            log.info(f"Components: {list(depot.components.keys())}")

            # Check if grid_feedin exists
            if hasattr(depot.model, 'grid_feedin'):
                log.info("Grid feedin variable EXISTS")
            else:
                log.info("Grid feedin variable DOES NOT EXIST")
    except Exception as e2:
        log.error(f"Cannot inspect depot: {e2}")

    import traceback
    traceback.print_exc()
    sys.exit(1)
