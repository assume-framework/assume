import pyomo.environ as pyo

from assume.world import World
from assume.scenario.loader_csv import (
    load_config_and_create_forecaster,
    setup_world,
)


w = World()

w.scenario_data = load_config_and_create_forecaster(
    "examples/inputs",
    "example_04b",
    "base",
)

setup_world(w)


for unit_id in ["hs_north", "hs_east", "hs_west"]:

    unit = w.units[unit_id]

    # Create an independent Pyomo instance.
    # This does not modify the HeatSystem model used by ASSUME.
    instance = unit.model.create_instance()

    # Put the copy into normal cost-minimisation mode.
    instance = unit.switch_to_opt(instance)

    # Solve only this diagnostic copy.
    results = unit.solver.solve(instance)

    print(f"\n{'=' * 100}")
    print(f"HeatSystem: {unit_id}")
    print(f"{'=' * 100}")

    print(
        "datetime | demand | HP_el | HP_heat | "
        "boiler_gas | boiler_heat | "
        "TES_ch | TES_dis | TES_soc | "
        "P_el_total | balance"
    )

    for t in instance.time_steps:

        timestamp = unit.index[t]

        hp_power = 0.0
        hp_heat = 0.0

        boiler_gas = 0.0
        boiler_heat = 0.0

        tes_charge = 0.0
        tes_discharge = 0.0
        tes_soc = 0.0

        # Heat pumps
        for hp in unit.heat_pumps:
            block = instance.dsm_blocks[hp]

            hp_power += pyo.value(block.power_in[t])
            hp_heat += pyo.value(block.heat_out[t])

        # Boilers
        for boiler in unit.boilers:
            block = instance.dsm_blocks[boiler]

            boiler_heat += pyo.value(block.heat_out[t])

            if hasattr(block, "natural_gas_in"):
                boiler_gas += pyo.value(
                    block.natural_gas_in[t]
                )

        # Thermal storage
        for storage in unit.thermal_storages:
            block = instance.dsm_blocks[storage]

            tes_charge += pyo.value(block.charge[t])
            tes_discharge += pyo.value(block.discharge[t])
            tes_soc += pyo.value(block.soc[t])

        heat_demand = pyo.value(
            instance.heat_demand[t]
        )

        total_power = pyo.value(
            instance.total_power_input[t]
        )

        heat_supply = (
            hp_heat
            + boiler_heat
            + tes_discharge
        )

        heat_required = (
            heat_demand
            + tes_charge
        )

        balance = heat_supply - heat_required

        print(
            f"{timestamp} | "
            f"{heat_demand:.3f} | "
            f"{hp_power:.3f} | "
            f"{hp_heat:.3f} | "
            f"{boiler_gas:.3f} | "
            f"{boiler_heat:.3f} | "
            f"{tes_charge:.3f} | "
            f"{tes_discharge:.3f} | "
            f"{tes_soc:.3f} | "
            f"{total_power:.3f} | "
            f"{balance:.6f}"
        )

print(f"\nFuel prices for {unit_id}:")

if hasattr(unit.forecaster, "fuel_prices"):
    print("Available fuels:", unit.forecaster.fuel_prices.keys())

    if "natural_gas" in unit.forecaster.fuel_prices:
        print(
            "Natural gas first values:",
            unit.forecaster.fuel_prices["natural_gas"]
        )
    else:
        print("natural_gas NOT FOUND")


print(f"\n--- Prices seen by {unit_id} ---")

print(
    "Available fuel price keys:",
    list(unit.forecaster.fuel_prices.keys())
)

print(
    "natural_gas lookup:",
    list(unit.forecaster.get_price("natural_gas"))[:3]
)

print(
    "natural gas lookup:",
    list(unit.forecaster.get_price("natural gas"))[:3]
)

print(
    "electricity price:",
    list(unit.forecaster.electricity_price)[:3]
)