# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Controlled EV dispatch experiments (no market clearing loop).

Run from the repository root, either way:
    python -m examples.inputs.example_lv_tariff.run_experiments
    python examples/inputs/example_lv_tariff/run_experiments.py
"""

import argparse
import hashlib
import io
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Running the file by path puts this directory on sys.path instead of the
# repository root, so `assume` is not importable.  Add the root rather than
# making the caller remember PYTHONPATH.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from assume.scenario.loader_csv import load_config_and_create_forecaster, setup_world
from assume.strategies.ev_aggregator import EVPortfolioStrategy
from assume.units.electric_vehicle import ElectricVehicleUnit
from assume.world import World

try:
    from .generate_inputs import generate
except ImportError:
    from generate_inputs import generate

HERE = Path(__file__).resolve().parent
CASES = {
    "independent",
    "aggregated",
    "constrained",
    "static",
    "tou",
    "dynamic",
    "peak",
    "capacity",
    "ex_post_peak",
}


def validate(config):
    for key in ("n_evs", "n_aggregators"):
        if (
            not isinstance(config[key], int)
            or isinstance(config[key], bool)
            or config[key] < 1
        ):
            raise ValueError(f"{key} must be a positive integer")
    if config["n_aggregators"] > config["n_evs"]:
        raise ValueError("n_aggregators cannot exceed n_evs")
    if not isinstance(config["v2g"], bool):
        raise ValueError("v2g must be true or false")
    for key, value in config.items():
        if key.endswith(("_mw", "_mwh")):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite/nonnegative")
    if any(not isinstance(h, int) or h < 0 or h > 23 for h in config["tou_hours"]):
        raise ValueError("tou_hours must contain integer hours 0..23")
    if config["horizon_mode"] not in ("perfect_foresight", "rolling_horizon"):
        raise ValueError("horizon_mode must be perfect_foresight or rolling_horizon")
    for key in ("look_ahead_horizon", "rolling_step"):
        if pd.Timedelta(config[key]) <= pd.Timedelta(0):
            raise ValueError(f"{key} must be positive")
    if (
        not config["cases"]
        or set(config["cases"]) - CASES
        or len(set(config["cases"])) != len(config["cases"])
    ):
        raise ValueError("cases must be a nonempty list of unique supported cases")


def solve_groups(units, groups, index, config, case, fees):
    """Solve one plan per portfolio over the full horizon.

    ``horizon_mode`` chooses what the optimiser is allowed to see.  Under
    ``perfect_foresight`` each portfolio is solved once over the whole run.
    Under ``rolling_horizon`` it is re-solved every ``rolling_step`` over a
    window of ``look_ahead_horizon`` and only the first step of each solve is
    kept, which is how the study cases in ``config.yaml`` plan.  Same fleet,
    same prices, same fees - only the foresight differs, so the two modes are
    directly comparable and the gap between them is the cost of not knowing the
    future.
    """
    mode = config["horizon_mode"]
    look_ahead = pd.Timedelta(config["look_ahead_horizon"])
    step = pd.Timedelta(config["rolling_step"])
    end = index[-1] + index.freq
    plans = {}
    for members in groups.values():
        share = len(members) / len(units)
        params = {}
        if case == "constrained":
            params.update(
                ev_import_limit_mw=config["connection_import_limit_mw"] * share,
                ev_export_limit_mw=config["connection_export_limit_mw"] * share,
            )
        if case in ("peak", "capacity"):
            params["ev_peak_price"] = config[
                "peak_price_eur_per_mw"
                if case == "peak"
                else "capacity_excess_price_eur_per_mw"
            ]
            if case == "capacity":
                params["ev_observed_peak_mw"] = config["contracted_capacity_mw"] * share
        strategy = EVPortfolioStrategy(
            ev_horizon_mode=mode,
            ev_look_ahead_horizon=config["look_ahead_horizon"],
            **params,
        )
        background = pd.Series(config["background_load_mw"] * share, index=index)
        if mode == "perfect_foresight":
            plans.update(
                strategy.optimize(
                    members,
                    index[0],
                    end,
                    grid_fees=fees,
                    background_load=background,
                )
            )
            continue
        # Rolling: replan every `step`, commit the steps up to the next replan.
        # The optimiser plans from each vehicle's realised energy, so the
        # committed dispatch has to be written back before the next solve.
        committed = {unit.id: {} for unit in members}
        # A contracted capacity is already-paid-for headroom, so it is where the
        # billed peak starts, not zero.
        floor = float(params.get("ev_observed_peak_mw", 0.0))
        # `index` here is a plain DatetimeIndex; the strategy wants the unit's
        # own FastIndex, which is the one that slices by timestamp.
        unit_index = members[0].index
        for start in pd.date_range(index[0], end - index.freq, freq=step):
            window = strategy._horizon_end(
                members, unit_index, start, min(end, start + look_ahead)
            )
            plan = strategy.optimize(
                members,
                start,
                window,
                grid_fees=fees,
                background_load=background,
                observed_peak=floor,
            )
            keep = list(
                pd.date_range(
                    start, min(start + step, end) - index.freq, freq=index.freq
                )
            )
            for unit in members:
                for t in keep:
                    power = float(plan[unit.id][t])
                    committed[unit.id][t] = power
                    # optimize() reads history back through energy_at(), which
                    # follows outputs["energy"]; see ElectricVehicleUnit.
                    unit.outputs["energy"].at[t] = power
            withdrawal = sum(
                -unit.outputs["energy"].loc[index[0] : keep[-1]] for unit in members
            )
            floor = max(floor, float(max(0.0, withdrawal.max() + background.iloc[0])))
        plans.update(committed)
    return pd.DataFrame(plans, index=index)


def run_suite(config, output_dir):
    """Write a self-contained run; refuse to mix it with an existing directory."""
    # Configs written before the horizon switch existed could only ever have run
    # under perfect foresight.
    config.setdefault("horizon_mode", "perfect_foresight")
    config.setdefault("look_ahead_horizon", "48h")
    config.setdefault("rolling_step", "1h")
    validate(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "experiments.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    inputs = output_dir / "inputs"
    generate(inputs, config["n_evs"], config["seed"])
    (inputs / "config.yaml").write_bytes((HERE / "config.yaml").read_bytes())
    world = World()
    world.scenario_data = load_config_and_create_forecaster(
        inputs_path=str(output_dir), scenario="inputs", study_case="no_tariff"
    )
    setup_world(world)
    units = [u for u in world.units.values() if isinstance(u, ElectricVehicleUnit)]
    if not config["v2g"]:
        for unit in units:
            unit.max_power_discharge = 0
    # Treat end_date as an exclusive boundary: 4 days = 96 hourly steps.
    full_index = units[0].index.as_datetimeindex()
    index = full_index[full_index < pd.Timestamp(world.scenario_data["end"])]
    index = pd.DatetimeIndex(index, freq=pd.Timedelta(units[0].index.freq))
    prices = pd.Series(
        [units[0].forecaster.price["EOM"].at[t] for t in index], index=index
    )
    groups = {
        f"aggregator_{i}": units[i :: config["n_aggregators"]]
        for i in range(config["n_aggregators"])
    }
    independent = {f"operator_{unit.id}": [unit] for unit in units}
    zero = pd.Series(0.0, index=index)
    baseline = solve_groups(units, groups, index, config, "aggregated", zero)
    # Frozen ex ante expectation from the no-fee forecast-optimal baseline.
    expected = (-baseline.sum(axis=1) + config["background_load_mw"]).clip(lower=0)
    dynamic = config["static_fee_eur_per_mwh"] + config[
        "dynamic_adder_eur_per_mwh"
    ] * expected / max(expected.max(), 1e-12)
    schedules = pd.DataFrame(
        {
            "price_eur_per_mwh": prices,
            "background_load_mw": config["background_load_mw"],
            "expected_connection_import_mw": expected,
            "dynamic_fee_eur_per_mwh": dynamic,
        }
    )
    schedules.to_csv(output_dir / "signals.csv", index_label="datetime")
    summaries, connection_rows = [], []
    hours = pd.Timedelta(index.freq).total_seconds() / 3600
    baseline_cost = float((-baseline.sum(axis=1) * prices).sum() * hours)
    mapping = []
    for case in config["cases"]:
        print(f"Running {case}", flush=True)
        case_groups = independent if case == "independent" else groups
        fees = zero.copy()
        if case in ("static", "tou", "dynamic"):
            fees[:] = config["static_fee_eur_per_mwh"]
        if case == "tou":
            fees += (
                np.isin(index.hour, config["tou_hours"])
                * config["tou_adder_eur_per_mwh"]
            )
        if case == "dynamic":
            fees = dynamic
        plan = (
            baseline.copy()
            if case in ("aggregated", "ex_post_peak")
            else solve_groups(units, case_groups, index, config, case, fees)
        )
        case_dir = output_dir / case
        case_dir.mkdir()
        unit_rows = []
        for op, members in case_groups.items():
            for unit in members:
                mapping.append({"case": case, "unit_id": unit.id, "operator": op})
                energy = unit.initial_soc * unit.capacity
                for t in index:
                    power = float(plan.at[t, unit.id])
                    trip = unit.trip_energy_consumption.at[t]
                    energy += (
                        hours
                        * (
                            max(-power, 0) * unit.efficiency_charge
                            - max(power, 0) / unit.efficiency_discharge
                        )
                        - trip
                    )
                    if (
                        not unit.min_soc * unit.capacity - 1e-7
                        <= energy
                        <= unit.max_soc * unit.capacity + 1e-7
                    ):
                        raise RuntimeError(
                            f"Invalid battery energy for {unit.id} at {t}"
                        )
                    unit_rows.append(
                        {
                            "datetime": t,
                            "unit_id": unit.id,
                            "operator": op,
                            "power_mw": power,
                            "soc_end": energy / unit.capacity,
                            "trip_mwh": trip,
                            "availability": unit.forecaster.availability.at[t],
                        }
                    )
                if energy < unit.initial_soc * unit.capacity - 1e-7:
                    raise RuntimeError(f"Terminal energy shortfall for {unit.id}")
        pd.DataFrame(unit_rows).to_csv(case_dir / "ev_dispatch.csv", index=False)
        operator_rows = []
        total_fee, total_energy_cost = 0.0, 0.0
        for op, members in case_groups.items():
            net = -plan[[u.id for u in members]].sum(axis=1)
            # The DSO meters the connection, not the vehicles: bill the grid fee
            # and the capacity charge on the operator's share of the whole
            # withdrawal, matching what EVPortfolioStrategy optimises against.
            share = len(members) / len(units)
            connection_net = net + config["background_load_mw"] * share
            imports = connection_net.clip(lower=0)
            exports = (-connection_net).clip(lower=0)
            peak = float(imports.max())
            energy_cost = float((net * prices).sum() * hours)
            volumetric = float((imports * fees).sum() * hours)
            capacity_fee = 0.0
            if case in ("peak", "ex_post_peak"):
                capacity_fee = peak * config["peak_price_eur_per_mw"]
            if case == "capacity":
                contracted = (
                    config["contracted_capacity_mw"] * len(members) / len(units)
                )
                capacity_fee = (
                    max(0, peak - contracted)
                    * config["capacity_excess_price_eur_per_mw"]
                )
            total_fee += volumetric + capacity_fee
            total_energy_cost += energy_cost
            operator_rows.append(
                {
                    "operator": op,
                    "n_evs": len(members),
                    "peak_import_mw": peak,
                    "ev_peak_import_mw": max(0.0, float(net.max())),
                    "import_mwh": imports.sum() * hours,
                    "export_mwh": exports.sum() * hours,
                    "energy_cost_eur": energy_cost,
                    "volumetric_fee_eur": volumetric,
                    "capacity_fee_eur": capacity_fee,
                    "total_cost_eur": energy_cost + volumetric + capacity_fee,
                }
            )
        pd.DataFrame(operator_rows).to_csv(case_dir / "operators.csv", index=False)
        net = -plan.sum(axis=1)
        connection = net + config["background_load_mw"]
        import_excess = (connection - config["connection_import_limit_mw"]).clip(
            lower=0
        )
        export_excess = (-connection - config["connection_export_limit_mw"]).clip(
            lower=0
        )
        frame = pd.DataFrame(
            {
                "case": case,
                "datetime": index,
                "ev_net_import_mw": net.to_numpy(),
                "connection_net_import_mw": connection.to_numpy(),
                "fee_eur_per_mwh": fees.to_numpy(),
                "import_excess_mw": import_excess.to_numpy(),
                "export_excess_mw": export_excess.to_numpy(),
            }
        )
        connection_rows.append(frame)
        summaries.append(
            {
                "case": case,
                "n_evs": len(units),
                "n_operators": len(case_groups),
                "energy_cost_eur": total_energy_cost,
                "grid_fee_eur": total_fee,
                "total_cost_eur": total_energy_cost + total_fee,
                "energy_cost_delta_vs_aggregated_eur": total_energy_cost
                - baseline_cost,
                "total_cost_delta_vs_aggregated_eur": total_energy_cost
                + total_fee
                - baseline_cost,
                "fleet_peak_import_mw": max(0, net.max()),
                "connection_peak_import_mw": max(0, connection.max()),
                "connection_peak_export_mw": max(0, -connection.min()),
                "overload_hours": float(
                    ((import_excess > 1e-7) | (export_excess > 1e-7)).sum() * hours
                ),
                "overload_mwh": float((import_excess + export_excess).sum() * hours),
                "trip_mwh": sum(r["trip_mwh"] for r in unit_rows),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "summary.csv", index=False)
    timeseries = pd.concat(connection_rows, ignore_index=True)
    timeseries.to_csv(output_dir / "timeseries.csv", index=False)
    pd.DataFrame(mapping).to_csv(output_dir / "ownership.csv", index=False)
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in inputs.iterdir()
        if p.is_file()
    }
    manifest = {
        "python": sys.version,
        "packages": {
            name: version(name) for name in ("numpy", "pandas", "pyomo", "highspy")
        },
        "config": config,
        "input_sha256": hashes,
        "mode": (
            "full_horizon_fixed_forecast_dispatch"
            if config["horizon_mode"] == "perfect_foresight"
            else "rolling_horizon_fixed_forecast_dispatch"
        ),
        "horizon_mode": config["horizon_mode"],
        "start": str(index[0]),
        "end_exclusive": str(index[-1] + index.freq),
        "baseline_energy_cost_eur": baseline_cost,
        "dynamic_expectation": "frozen no-fee aggregated schedule plus background",
        "limits": "per-operator fleet-size shares of shared connection and background",
        "billing": "positive net operator import; no export fee credit; whole-run peak period",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Self-contained HTML comparison, with no plotting dependency.
    report = "<html><head><meta charset='utf-8'><title>EV experiments</title></head><body><h1>EV dispatch experiments</h1><p>Fixed forecasts; full horizon; fees settled on dispatch. Positive costs are expenditure, negative costs are revenue. See manifest.json for assumptions.</p>"
    report += summary.round(6).to_html(index=False)
    report += comparison_plot(summary, timeseries, config, output_dir)
    report += "<p>Compare constrained energy cost with aggregated to measure foregone arbitrage. Grid fees are transfers, not a social welfare loss. Independent and aggregated optimal energy costs should agree; tied schedules can have different peaks.</p></body></html>"
    (output_dir / "report.html").write_text(report, encoding="utf-8")
    return summary


def comparison_plot(summary, timeseries, config, output_dir):
    """Add an exportable figure when the optional plotting package is installed."""
    try:
        from matplotlib.figure import Figure
    except ImportError:
        return "<p>Install matplotlib to include comparison plots.</p>"
    figure = Figure(figsize=(13, 10), layout="constrained")
    cost_ax, peak_ax, load_ax = figure.subplots(3, 1)
    labels = summary["case"].tolist()
    cost_ax.bar(
        labels, summary.energy_cost_delta_vs_aggregated_eur, label="Energy cost change"
    )
    cost_ax.bar(
        labels,
        summary.grid_fee_eur,
        bottom=summary.energy_cost_delta_vs_aggregated_eur,
        label="Grid fee",
    )
    cost_ax.set_ylabel("EUR vs aggregated")
    cost_ax.legend()
    peak_ax.bar(labels, summary.connection_peak_import_mw * 1000)
    peak_ax.axhline(
        config["connection_import_limit_mw"] * 1000,
        color="black",
        linestyle="--",
        label="Connection import limit",
    )
    peak_ax.set_ylabel("Peak connection import (kW)")
    peak_ax.legend()
    for case, frame in timeseries.groupby("case", sort=False):
        load_ax.step(
            frame.datetime,
            frame.connection_net_import_mw * 1000,
            where="post",
            label=case,
            alpha=0.75,
        )
    load_ax.axhline(
        config["connection_import_limit_mw"] * 1000, color="black", linestyle="--"
    )
    load_ax.axhline(
        -config["connection_export_limit_mw"] * 1000, color="black", linestyle="--"
    )
    load_ax.set_ylabel("Connection net import (kW)")
    load_ax.legend(ncol=3, fontsize=8)
    load_ax.tick_params(axis="x", rotation=20)
    buffer = io.StringIO()
    figure.savefig(buffer, format="svg")
    svg = buffer.getvalue()
    (output_dir / "comparison.svg").write_text(svg, encoding="utf-8")
    return svg[svg.index("<svg") :]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "experiments.yaml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/outputs/lv_ev_experiments")
        / pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f"),
    )
    parser.add_argument("--n-evs", type=int)
    parser.add_argument("--n-aggregators", type=int)
    parser.add_argument(
        "--horizon-mode",
        choices=("perfect_foresight", "rolling_horizon"),
        help="What the optimiser may see. Use rolling_horizon to make these "
        "numbers comparable with the config.yaml study cases, which roll.",
    )
    parser.add_argument(
        "--look-ahead-horizon",
        help="Planning window in rolling mode, e.g. 48h.",
    )
    parser.add_argument(
        "--rolling-step",
        help="How often the plan is redone in rolling mode, e.g. 1h.",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    for name in (
        "n_evs",
        "n_aggregators",
        "horizon_mode",
        "look_ahead_horizon",
        "rolling_step",
    ):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    result = run_suite(config, args.output_dir)
    print(result.to_string(index=False))
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
