"""Plot reconstructed EV state of charge for a study case.

Usage: python plot.py <study_case>
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.lines import Line2D

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "scenario_data"
OUTPUT_DIR = BASE_DIR / "output_data"
PLOT_DIR = OUTPUT_DIR / "plots"
EV_TECHNOLOGY_PREFIX = "electric_vehicle_"


def load_config():
    with (INPUT_DIR / "config.yaml").open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def available_study_cases(config):
    """Return cases with both EV and building scenario definitions."""
    return sorted(
        case_name[:-3]
        for case_name in config
        if case_name.endswith("_ev")
        and f"{case_name[:-3]}_building" in config
    )


def load_parameter_file(config, scenario_name, parameter_name):
    scenario = config.get(scenario_name)
    if scenario is None:
        raise ValueError(f"Scenario {scenario_name!r} is not defined in config.yaml")
    relative_path = scenario.get(parameter_name)
    if not relative_path:
        raise ValueError(
            f"Scenario {scenario_name!r} does not define {parameter_name!r}"
        )

    parameter_file = INPUT_DIR / relative_path
    if not parameter_file.is_file():
        raise FileNotFoundError(f"Parameter file not found: {parameter_file}")
    return pd.read_csv(parameter_file)


def load_orders(output_dir, unit_ids):
    """Load accepted EOM volumes for the selected EV or building units."""
    orders_file = output_dir / "market_orders.csv"
    if not orders_file.is_file():
        raise FileNotFoundError(f"Market orders file not found: {orders_file}")

    orders = pd.read_csv(orders_file)
    required_columns = {
        "market_id",
        "unit_id",
        "start_time",
        "end_time",
        "accepted_volume",
    }
    missing_columns = required_columns - set(orders.columns)
    if missing_columns:
        raise ValueError(
            f"{orders_file} is missing required columns: {', '.join(sorted(missing_columns))}"
        )

    orders = orders[
        orders["market_id"].eq("EOM") & orders["unit_id"].astype(str).isin(unit_ids)
    ].copy()
    if orders.empty:
        raise ValueError(
            f"No EOM market orders for units {', '.join(sorted(unit_ids))} in {orders_file}"
        )

    orders["start_time"] = pd.to_datetime(orders["start_time"], errors="raise")
    orders["end_time"] = pd.to_datetime(orders["end_time"], errors="raise")
    orders["accepted_volume"] = pd.to_numeric(
        orders["accepted_volume"], errors="raise"
    )
    if not orders["accepted_volume"].map(math.isfinite).all():
        raise ValueError(f"{orders_file} contains a non-finite accepted volume")
    if (orders["end_time"] <= orders["start_time"]).any():
        raise ValueError(f"{orders_file} contains an invalid delivery interval")

    return orders.sort_values(["unit_id", "start_time", "end_time"])


def parameter_value(row, name, source):
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid or missing {name!r} in {source}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Invalid or missing {name!r} in {source}")
    return value


@dataclass
class SOCTrace:
    values: pd.Series
    trip_steps: pd.Series


def load_trip_energy(config, scenario_name, curve_columns):
    """Load battery-side trip energy for each EV or household EV fleet."""
    scenario = config.get(scenario_name)
    if scenario is None:
        raise ValueError(f"Scenario {scenario_name!r} is not defined in config.yaml")
    relative_path = scenario.get("forecasts_df")
    if not relative_path:
        raise ValueError(
            f"Scenario {scenario_name!r} does not define 'forecasts_df'"
        )

    forecast_file = INPUT_DIR / relative_path
    if not forecast_file.is_file():
        raise FileNotFoundError(f"Forecast file not found: {forecast_file}")
    forecasts = pd.read_csv(forecast_file, parse_dates=["datetime"])
    if forecasts["datetime"].duplicated().any():
        raise ValueError(f"{forecast_file} contains duplicate forecast timestamps")

    result = {}
    for curve_name, columns in curve_columns.items():
        missing = set(columns) - set(forecasts.columns)
        if missing:
            raise ValueError(
                f"{forecast_file} is missing trip energy columns: "
                f"{', '.join(sorted(missing))}"
            )
        energy = forecasts[columns].apply(pd.to_numeric, errors="raise").sum(axis=1)
        if not energy.map(math.isfinite).all() or (energy < 0).any():
            raise ValueError(
                f"{forecast_file} contains invalid trip energy for {curve_name!r}"
            )
        result[curve_name] = pd.Series(
            energy.to_numpy(),
            index=pd.DatetimeIndex(forecasts["datetime"]),
            name=curve_name,
        )
    return result


def reconstruct_soc(
    orders,
    trip_energy,
    capacity,
    initial_soc,
    efficiency_charge,
    efficiency_discharge,
    min_soc=0.0,
    max_soc=1.0,
    start_time=None,
):
    """Reconstruct SOC from accepted energy volumes and EV trip consumption.

    Accepted demand (negative volume) charges the battery; accepted supply
    (positive volume) discharges it. Both accepted volumes and trip consumption
    are battery-side MWh per delivery interval.
    """
    if capacity <= 0:
        raise ValueError("Battery capacity must be positive")
    if efficiency_charge <= 0 or efficiency_discharge <= 0:
        raise ValueError("Battery charge and discharge efficiencies must be positive")
    if min_soc > initial_soc or initial_soc > max_soc:
        raise ValueError("Initial SOC must be within the configured SOC bounds")

    trip_energy = trip_energy.sort_index()
    if trip_energy.empty:
        raise ValueError("Trip energy forecast is empty")
    if trip_energy.index.has_duplicates:
        raise ValueError("Trip energy forecast contains duplicate timestamps")
    if len(trip_energy) < 2:
        raise ValueError("At least two forecast timestamps are required")
    if start_time is not None and pd.Timestamp(start_time) != trip_energy.index[0]:
        raise ValueError("Scenario start time does not match the first forecast")

    orders = orders.copy()
    if not orders.empty:
        order_energy = orders.groupby("start_time")["accepted_volume"].sum()
        order_ends = orders.groupby("start_time")["end_time"].agg(
            lambda ends: ends.iloc[0] if ends.nunique() == 1 else pd.NaT
        )
        if order_ends.isna().any():
            raise ValueError("Orders for the same delivery time have different ends")
        unknown_starts = order_energy.index.difference(trip_energy.index[:-1])
        if not unknown_starts.empty:
            raise ValueError(
                "Market orders contain delivery times not present in the trip forecast"
            )
    else:
        order_energy = pd.Series(dtype=float)
        order_ends = pd.Series(dtype="datetime64[ns]")

    energy = initial_soc * capacity
    min_energy = min_soc * capacity
    max_energy = max_soc * capacity
    times = [trip_energy.index[0]]
    values = [initial_soc]
    step_trip_energy = []

    for i, delivery_start in enumerate(trip_energy.index[:-1]):
        delivery_end = trip_energy.index[i + 1]
        if delivery_end <= delivery_start:
            raise ValueError("Trip forecast timestamps must be strictly increasing")
        if delivery_start in order_ends.index and order_ends[delivery_start] != delivery_end:
            raise ValueError(
                f"Market order interval for {delivery_start} does not match the "
                "trip forecast interval"
            )

        accepted_volume = float(order_energy.get(delivery_start, 0.0))
        if accepted_volume < 0:
            energy += -accepted_volume * efficiency_charge
        else:
            energy -= accepted_volume / efficiency_discharge
        energy = min(max(energy, min_energy), max_energy)

        trip = float(trip_energy.iloc[i])
        energy = max(min_energy, energy - trip)
        step_trip_energy.append(trip > 0)
        times.append(delivery_end)
        values.append(energy / capacity)

    return SOCTrace(
        values=pd.Series(values, index=pd.DatetimeIndex(times), name="SOC"),
        trip_steps=pd.Series(
            step_trip_energy, index=pd.DatetimeIndex(times[:-1]), name="trip_energy"
        ),
    )


def reconstruct_ev_soc(ev_parameters, orders, trip_energy, start_time):
    """Return one reconstructed SOC curve per EV."""
    curves = {}
    for row in ev_parameters.to_dict(orient="records"):
        ev_name = str(row["name"])
        ev_orders = load_unit_orders(orders, ev_name)
        curves[ev_name] = reconstruct_soc(
            ev_orders,
            trip_energy=trip_energy[ev_name],
            capacity=parameter_value(row, "capacity", ev_name),
            initial_soc=parameter_value(row, "initial_soc", ev_name),
            efficiency_charge=parameter_value(row, "efficiency_charge", ev_name),
            efficiency_discharge=parameter_value(row, "efficiency_discharge", ev_name),
            min_soc=parameter_value(row, "min_soc", ev_name),
            max_soc=parameter_value(row, "max_soc", ev_name),
            start_time=start_time,
        )
    return curves


def load_unit_orders(orders, unit_id):
    """Select the already-loaded order rows for one unit."""
    selected = orders[orders["unit_id"].astype(str).eq(unit_id)]
    if selected.empty:
        raise ValueError(f"No EOM market orders found for unit {unit_id!r}")
    return (
        selected.groupby(["start_time", "end_time"], as_index=False)["accepted_volume"]
        .sum()
        .sort_values(["start_time", "end_time"])
    )


def reconstruct_building_soc(building_parameters, orders, trip_energy, start_time):
    """Return one capacity-weighted aggregate SOC curve per building.

    Building market orders contain the household's combined EV volume, so a
    multi-EV household is represented by one fleet SOC curve. Its energy changes
    use capacity-weighted efficiency, equivalent to allocating volume by EV
    capacity when individual dispatch is unavailable.
    """
    child_rows = building_parameters[
        building_parameters["technology"]
        .fillna("")
        .astype(str)
        .str.startswith(EV_TECHNOLOGY_PREFIX)
    ]
    if child_rows.empty:
        raise ValueError("Building parameter file contains no attached EV units")

    curves = {}
    for building_name, ev_rows in child_rows.groupby("name", sort=True):
        fleet_name = str(building_name)
        capacities = [
            parameter_value(row, "capacity", fleet_name)
            for row in ev_rows.to_dict(orient="records")
        ]
        total_capacity = sum(capacities)
        if total_capacity <= 0:
            raise ValueError(f"EV capacity for building {fleet_name!r} must be positive")

        def weighted_average(parameter):
            values = [
                parameter_value(row, parameter, fleet_name)
                for row in ev_rows.to_dict(orient="records")
            ]
            return sum(capacity * value for capacity, value in zip(capacities, values)) / total_capacity

        initial_soc = weighted_average("initial_soc")
        min_soc = weighted_average("min_soc")
        max_soc = weighted_average("max_soc")
        building_orders = load_unit_orders(orders, fleet_name)
        curves[fleet_name] = reconstruct_soc(
            building_orders,
            trip_energy=trip_energy[fleet_name],
            capacity=total_capacity,
            initial_soc=initial_soc,
            efficiency_charge=weighted_average("efficiency_charge"),
            efficiency_discharge=weighted_average("efficiency_discharge"),
            min_soc=min_soc,
            max_soc=max_soc,
            start_time=start_time,
        )
    return curves


def plot_curves(ax, curves, title):
    for name, trace in curves.items():
        color = None
        for i, trip_step in enumerate(trace.trip_steps):
            (line,) = ax.plot(
                trace.values.index[i : i + 2],
                trace.values.iloc[i : i + 2],
                color=color,
                linestyle="--" if trip_step else "-",
                label=name if i == 0 else "_nolegend_",
            )
            color = line.get_color()
    ax.set_title(title)
    ax.set_ylabel("State of charge")
    ax.set_ylim(0, 2)
    ax.legend(
        handles=[
            *ax.get_legend_handles_labels()[0],
            Line2D([0], [0], color="black", linestyle="--", label="Driving loss"),
        ]
    )


def plot_soc_comparison(study_case):
    config = load_config()
    ev_scenario_name = f"{study_case}_ev"
    building_scenario_name = f"{study_case}_building"
    ev_parameters = load_parameter_file(
        config, ev_scenario_name, "electric_vehicle_units"
    )
    building_parameters = load_parameter_file(
        config, building_scenario_name, "residential_dsm_units"
    )
    start_time = config[ev_scenario_name].get("start_date")
    ev_trip_energy = load_trip_energy(
        config,
        ev_scenario_name,
        {
            str(ev_name): [f"{ev_name}_trip_energy_consumption"]
            for ev_name in ev_parameters["name"]
        },
    )

    ev_orders = load_orders(
        OUTPUT_DIR / config[ev_scenario_name]["simulation_id"],
        set(ev_parameters["name"].astype(str)),
    )
    building_ids = set(
        building_parameters.loc[
            building_parameters["technology"]
            .fillna("")
            .astype(str)
            .str.startswith(EV_TECHNOLOGY_PREFIX),
            "name",
        ].astype(str)
    )
    ev_rows = building_parameters[
        building_parameters["technology"]
        .fillna("")
        .astype(str)
        .str.startswith(EV_TECHNOLOGY_PREFIX)
    ]
    building_trip_columns = {
        str(building_name): [
            f"{building_name}_{technology}_trip_energy_consumption"
            for technology in building_ev_rows["technology"].astype(str)
        ]
        for building_name, building_ev_rows in ev_rows.groupby("name", sort=True)
    }
    building_trip_energy = load_trip_energy(
        config, building_scenario_name, building_trip_columns
    )
    building_orders = load_orders(
        OUTPUT_DIR / config[building_scenario_name]["simulation_id"], building_ids
    )

    ev_curves = reconstruct_ev_soc(
        ev_parameters, ev_orders, ev_trip_energy, start_time
    )
    building_curves = reconstruct_building_soc(
        building_parameters, building_orders, building_trip_energy, start_time
    )

    fig, (ev_ax, building_ax) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 8)
    )
    plot_curves(ev_ax, ev_curves, f"{study_case}: EV SOC")
    plot_curves(
        building_ax,
        building_curves,
        f"{study_case}: Building EV fleet SOC",
    )
    building_ax.set_xlabel("Time")
    fig.tight_layout()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_file = PLOT_DIR / f"{study_case}_soc_comparison.png"
    fig.savefig(plot_file)
    print(f"Saved SOC comparison plot to {plot_file}")
    plt.show()


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study_case",
        choices=available_study_cases(config),
        help="Study case to compare, e.g. test_case_1_no_tariff.",
    )
    args = parser.parse_args()
    plot_soc_comparison(args.study_case)


if __name__ == "__main__":
    main()
