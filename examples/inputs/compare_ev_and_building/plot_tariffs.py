"""Plot the summed EV fleet load (MW) of all tariff variants of a study case.

Usage: python plot_tariffs.py <study_case>

A tariff variant is a scenario named ``<study_case>_<tariff>_ev``, e.g.
``test_case_5_tou_tariff_ev`` for study case ``test_case_5``.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot import OUTPUT_DIR, PLOT_DIR, load_config, load_orders, load_parameter_file


def tariff_scenarios(config, study_case):
    """Return {tariff name: scenario name} for every EV variant of the case."""
    prefix = f"{study_case}_"
    return {
        name[len(prefix) : -len("_ev")]: name
        for name in config
        if name.startswith(prefix) and name.endswith("_ev")
    }


def available_study_cases(config):
    """Return study cases that define at least two tariff variants."""
    candidates = {
        "_".join(name.split("_")[:3])
        for name in config
        if name.endswith("_ev") and name.count("_") >= 4
    }
    return sorted(
        case for case in candidates if len(tariff_scenarios(config, case)) > 1
    )


def fleet_load(config, scenario_name):
    """Accepted volume summed over all EVs per time step (charging positive)."""
    scenario = config[scenario_name]
    parameters = load_parameter_file(config, scenario_name, "electric_vehicle_units")
    orders = load_orders(
        OUTPUT_DIR / scenario["simulation_id"], set(parameters["name"].astype(str))
    )
    load = -orders.groupby("start_time")["accepted_volume"].sum()
    index = pd.date_range(
        scenario["start_date"], scenario["end_date"], freq=scenario["time_step"]
    )
    return load.reindex(index.union(load.index), fill_value=0.0)


def plot_tariff_comparison(study_case):
    import seaborn as sns

    config = load_config()
    scenarios = tariff_scenarios(config, study_case)
    if not scenarios:
        raise ValueError(f"No tariff variants found for {study_case!r}")

    fig, ax = plt.subplots(figsize=(12, 6))
    for tariff, scenario_name in scenarios.items():
        load = fleet_load(config, scenario_name)
        load = load[load.values != 0]
        
        random_minutes = np.random.uniform(-30, 30, size=len(load))
        print(random_minutes)
        x_evasion = pd.to_timedelta(random_minutes, unit="m")
        
        sns.scatterplot(x=load.index + x_evasion, y=load.values, label=tariff, ax=ax)
        # ax.scatter(load.index, load.values, label=tariff)

    ax.set_title(f"{study_case}: total EV fleet load by tariff")
    ax.set_xlabel("Time")
    ax.set_ylabel("Fleet load (MW)")
    ax.legend(title="Tariff")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_file = PLOT_DIR / f"{study_case}_tariff_comparison.png"
    fig.savefig(plot_file)
    print(f"Saved tariff comparison plot to {plot_file}")
    plt.show()

def plot_energy_curves(study_case):
    config = load_config()
    scenarios = tariff_scenarios(config, study_case)
    if not scenarios:
        raise ValueError(f"No tariff variants found for {study_case!r}")

    fig, ax = plt.subplots(figsize=(12, 6))
    for tariff, scenario_name in scenarios.items():
        load = fleet_load(config, scenario_name)
        ax.plot(load.index, load.cumsum(), label=tariff)

    ax.set_title(f"{study_case}: cumulative EV fleet energy by tariff")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative energy (MWh)")
    ax.legend(title="Tariff")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_file = PLOT_DIR / f"{study_case}_energy_curves.png"
    fig.savefig(plot_file)
    print(f"Saved energy curves plot to {plot_file}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study_case",
        choices=available_study_cases(load_config()),
        help="Study case with several tariff variants, e.g. test_case_5.",
    )
    args = parser.parse_args()
    plot_energy_curves(args.study_case)
    plot_tariff_comparison(args.study_case)

    

if __name__ == "__main__":
    main()
