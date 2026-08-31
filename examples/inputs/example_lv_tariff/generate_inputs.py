# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generate the compact synthetic inputs for the ``example_lv_tariff`` scenario.

The scenario is a plumbing test for the grid-fee market (see
``assume/strategies/grid_tariff.py``): a Building acting purely as an EV
charging hub bids into the EOM, while a DSO unit publishes a grid fee on a
second ``pay_as_clear`` market that opens ahead of it.

Run from the repository root::

    python examples/inputs/example_lv_tariff/generate_inputs.py

The EV availability / trip-energy schema mirrors the ``dist_grid_DE`` inputs, so
that dataset can be dropped in unchanged for a realistic run.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent

START = "2019-03-01 00:00"
END = "2019-03-05 00:00"
FREQ = "1h"

N_EVS = 6
EV_CAPACITY = 0.03  # MWh
EV_POWER = 0.011  # MW
DAILY_TRIP_ENERGY = 0.015  # MWh per EV per weekday

rng = np.random.default_rng(42)
index = pd.date_range(START, END, freq=FREQ)
index.name = "datetime"


def write(name: str, df: pd.DataFrame) -> None:
    df.to_csv(HERE / name)
    print(f"wrote {name}: {df.shape}")


# --------------------------------------------------------------------------
# System side: a small merit order plus a day/night demand profile, so the EOM
# price the aggregator optimises against actually has a shape.
# --------------------------------------------------------------------------
hour = index.hour.to_numpy()

solar = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None) ** 1.5
wind = 0.35 + 0.25 * np.sin(np.arange(len(index)) / 17.0)
write(
    "availability_df.csv",
    pd.DataFrame({"Solar": solar, "Wind Onshore": wind}, index=index),
)

demand_shape = (
    0.78
    + 0.13 * np.sin((hour - 8) / 24 * 2 * np.pi)
    + 0.09 * np.exp(-(((hour - 19) / 2.5) ** 2))
)
write(
    "demand_df.csv",
    pd.DataFrame({"demand_EOM": 45000 * demand_shape}, index=index),
)

fuel = pd.DataFrame(
    {
        "uranium": 0.9,
        "lignite": 1.8,
        "hard coal": 8.5,
        "natural gas": 26.0,
        "oil": 21.7,
        "co2": 24.9,
    },
    index=index,
)
write("fuel_prices_df.csv", fuel)

powerplants = pd.DataFrame(
    [
        # name, technology, fuel, emission, max_power, efficiency, extra cost
        ("Solar", "solar", "renewable", 0.0, 14000, 1.0, 0.0, "renewables_operator"),
        (
            "Wind_Onshore",
            "wind_onshore",
            "renewable",
            0.0,
            10000,
            1.0,
            0.0,
            "renewables_operator",
        ),
        ("Nuclear", "nuclear", "uranium", 0.0, 8000, 0.33, 10.3, "conv_operator"),
        ("Lignite", "lignite", "lignite", 0.406, 18000, 0.43, 1.65, "conv_operator"),
        (
            "Hard_Coal",
            "hard coal",
            "hard coal",
            0.335,
            20000,
            0.45,
            1.3,
            "conv_operator",
        ),
        (
            "Natural_Gas",
            "open cycle gas turbine",
            "natural gas",
            0.201,
            30000,
            0.32,
            5.5,
            "conv_operator",
        ),
        ("Oil", "oil", "oil", 0.776, 8000, 0.31, 5.5, "conv_operator"),
    ],
    columns=[
        "name",
        "technology",
        "fuel_type",
        "emission_factor",
        "max_power",
        "efficiency",
        "additional_cost",
        "unit_operator",
    ],
)
powerplants["min_power"] = 0
powerplants["bidding_EOM"] = "powerplant_energy_heuristic_flexable"
powerplants["bidding_GridTariff"] = ""

# The DSO: an ordinary powerplant that only ever bids on the grid-fee market.
# It carries no meaningful physics - the tariff lives entirely in its strategy.
dso = {
    "name": "DSO",
    "technology": "grid_operator",
    "fuel_type": "natural gas",
    "emission_factor": 0.0,
    "max_power": 0.001,
    "efficiency": 1.0,
    "additional_cost": 0.0,
    "unit_operator": "dso_operator",
    "min_power": 0,
    "bidding_EOM": "",
    "bidding_GridTariff": "powerplant_grid_fee_dso",
}
powerplants = pd.concat([powerplants, pd.DataFrame([dso])], ignore_index=True)
write("powerplant_units.csv", powerplants.set_index("name"))

write(
    "demand_units.csv",
    pd.DataFrame(
        [
            {
                "name": "demand_EOM",
                "technology": "inflex_demand",
                "bidding_EOM": "demand_energy_naive",
                "bidding_GridTariff": "",
                "max_power": 1000000.0,
                "min_power": 0.0,
                "unit_operator": "eom_operator",
            }
        ]
    ).set_index("name"),
)

# --------------------------------------------------------------------------
# The aggregator: one Building used purely as an EV charging hub.
# Every non-EV building technology is left out, so `total_power_input` is the
# fleet's net node withdrawal and nothing else.
# --------------------------------------------------------------------------
rows = [
    {
        "name": "aggregator",
        "unit_type": "building",
        "technology": "",
        "node": "lv_node",
        "unit_operator": "aggregator_operator",
        "bidding_EOM": "household_energy_optimization",
        "bidding_GridTariff": "household_grid_fee_announcement",
        "objective": "min_variable_cost",
        "flexibility_measure": "electricity_price_signal",
        "cost_tolerance": 10.0,
        "is_prosumer": "No",
        "horizon_mode": "rolling_horizon",
        "look_ahead_horizon": "48h",
        "commit_horizon": "1h",
        "rolling_step": "1h",
    }
]
for i in range(N_EVS):
    rows.append(
        {
            "name": "aggregator",
            "unit_type": "building",
            "technology": f"electric_vehicle_EV_{i}",
            "capacity": EV_CAPACITY,
            "min_soc": 0.2,
            "max_soc": 1.0,
            "initial_soc": 0.6,
            "max_power_charge": EV_POWER,
            "max_power_discharge": EV_POWER,
            "efficiency_charge": 0.95,
            "efficiency_discharge": 0.95,
            "ramp_up": EV_POWER,
            "ramp_down": EV_POWER,
            "power_flow_directionality": "bidirectional",
        }
    )
write("residential_dsm_units.csv", pd.DataFrame(rows).set_index("name"))

# --------------------------------------------------------------------------
# EV availability and trip energy. Each EV leaves in the morning and returns in
# the evening; the daily trip energy is spread over the hours it is away.
# --------------------------------------------------------------------------
forecasts = {}
for i in range(N_EVS):
    leave = 7 + int(rng.integers(0, 3))  # 7, 8 or 9
    ret = 16 + int(rng.integers(0, 4))  # 16..19
    away = (hour >= leave) & (hour < ret) & (index.dayofweek < 5)

    availability = np.where(away, 0.0, 1.0)
    trip = np.zeros(len(index))
    # usage[t] == (1 - availability[t]) * trip_energy[t], so the energy has to be
    # spread over the away hours rather than booked at departure.
    trip[away] = DAILY_TRIP_ENERGY / max(ret - leave, 1)

    forecasts[f"aggregator_electric_vehicle_EV_{i}_availability_profile"] = availability
    forecasts[f"aggregator_electric_vehicle_EV_{i}_trip_energy_consumption"] = trip

write("forecasts_df.csv", pd.DataFrame(forecasts, index=index))
