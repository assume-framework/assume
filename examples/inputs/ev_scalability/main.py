"""Find out how many EVs the EV optimization can simulate in reasonable time.

Builds one-day EOM scenarios with an increasing number of EVs (one aggregator,
no grid tariff) and runs them one after another, each in its own process.
If a case exceeds the timeout (default 25 min) it is killed and all cases with
more EVs are skipped. Runtime plots are written to output_data/plots.

Usage: python main.py [--ev-counts 1 5 10 ...] [--timeout-min 25]
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "scenario_data"
OUTPUT_DIR = BASE_DIR / "output_data"
PLOT_DIR = OUTPUT_DIR / "plots"
RESULTS_FILE = OUTPUT_DIR / "runtimes.csv"
SOURCE_DIR = BASE_DIR.parent / "compare_ev_and_building" / "scenario_data"

DEFAULT_EV_COUNTS = [2**k for k in range(13)]
SIM_START = "2019-03-01 00:00"
SIM_END = "2019-03-02 00:00"  # one simulated day
FORECAST_HOURS = 48  # same data length as the existing 2-day study cases


def case_name(n_evs: int) -> str:
    return f"ev_scalability_{n_evs}_evs"


def build_ev_profiles(n_evs: int, index: pd.DatetimeIndex, seed: int = 0):
    """Random daily commute: each EV is away for 1-3 hours, once per day."""
    rng = np.random.default_rng(seed)
    columns = {}
    for i in range(n_evs):
        availability = np.ones(len(index))
        consumption = np.zeros(len(index))
        for day_start in range(0, len(index), 24):
            start = day_start + int(rng.integers(4, 19))
            duration = int(rng.integers(1, 4))
            availability[start : start + duration] = 0.0
            consumption[start : start + duration] = 0.01
        columns[f"EV_{i}_availability_profile"] = availability
        columns[f"EV_{i}_trip_energy_consumption"] = consumption
    return pd.DataFrame(columns, index=index)


def generate_case(n_evs: int):
    """Write the input files and config entry for a case with n_evs EVs."""
    case_dir = INPUT_DIR / "study_cases" / case_name(n_evs)
    case_dir.mkdir(parents=True, exist_ok=True)

    source_prices = pd.read_csv(
        SOURCE_DIR / "study_cases" / "test_case_6_ev" / "forecasts_df.csv",
        index_col="datetime",
        parse_dates=True,
        usecols=["datetime", "price_EOM"],
    ).iloc[:FORECAST_HOURS]
    forecasts = pd.concat(
        [source_prices, build_ev_profiles(n_evs, source_prices.index)], axis=1
    )
    forecasts.to_csv(case_dir / "forecasts_df.csv", index_label="datetime")

    pd.DataFrame(
        {
            "name": [f"EV_{i}" for i in range(n_evs)],
            "technology": "electric_vehicle",
            "node": "lv_node",
            "unit_operator": "aggregator_operator",
            "bidding_EOM": "ev_portfolio_member",
            "bidding_GridTariff": "ev_portfolio_member",
            "capacity": 0.1,
            "min_soc": 0.2,
            "max_soc": 1.0,
            "initial_soc": 0.6,
            "max_power_charge": -0.022,
            "max_power_discharge": 0.0,
            "efficiency_charge": 1.0,
            "efficiency_discharge": 1.0,
        }
    ).to_csv(case_dir / "electric_vehicle_units.csv", index=False)

    pd.DataFrame(
        {"name": ["aggregator_operator"], "bidding_EOM": ["units_operator_ev"]}
    ).to_csv(case_dir / "unit_operators.csv", index=False)

    rel = f"study_cases/{case_name(n_evs)}"
    return {
        "simulation_id": case_name(n_evs),
        "availability_df": "common/availability_df.csv",
        "demand_df": "common/demand_df.csv",
        "demand_units": "common/demand_units.csv",
        "fuel_prices_df": "common/fuel_prices_df.csv",
        "powerplant_units": "common/powerplant_units.csv",
        "start_date": SIM_START,
        "end_date": SIM_END,
        "time_step": "1h",
        "save_frequency_hours": 24,
        "electric_vehicle_units": f"{rel}/electric_vehicle_units.csv",
        "forecasts_df": f"{rel}/forecasts_df.csv",
        "unit_operators": f"{rel}/unit_operators.csv",
        "bidding_strategy_params": {
            "ev_look_ahead_horizon": "48h",
            "ev_background_load_mw": 0.0,
        },
        "markets_config": {
            "EOM": {
                "operator": "eom_operator",
                "product_type": "energy",
                "products": [{"duration": "1h", "count": 1, "first_delivery": "1h"}],
                "opening_frequency": "1h",
                "opening_duration": "1h",
                "volume_unit": "MWh",
                "maximum_bid_volume": 1000000,
                "maximum_bid_price": 3000,
                "minimum_bid_price": -500,
                "price_unit": "EUR/MWh",
                "market_mechanism": "pay_as_clear",
            }
        },
    }


def generate_inputs(ev_counts):
    """Create scenario_data with common files and one study case per EV count."""
    shutil.copytree(SOURCE_DIR / "common", INPUT_DIR / "common", dirs_exist_ok=True)
    config = {case_name(n): generate_case(n) for n in ev_counts}
    with open(INPUT_DIR / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def run_single_case(n_evs: int):
    """Worker: simulate one case and write its timings to a JSON file."""
    # use the repository's assume rather than a possibly outdated installed one
    sys.path.insert(0, str(BASE_DIR.parents[2]))
    from assume.scenario.loader_csv import load_config_and_create_forecaster, setup_world
    from assume.world import World

    t0 = time.perf_counter()
    world = World(export_csv_path=str(OUTPUT_DIR))
    world.scenario_data = load_config_and_create_forecaster(
        inputs_path=str(INPUT_DIR.parent),
        scenario=INPUT_DIR.name,
        study_case=case_name(n_evs),
    )
    setup_world(world)
    t1 = time.perf_counter()
    world.run()
    t2 = time.perf_counter()
    (OUTPUT_DIR / f"{case_name(n_evs)}_timing.json").write_text(
        json.dumps({"setup_s": t1 - t0, "run_s": t2 - t1, "total_s": t2 - t0})
    )


def kill_tree(proc: subprocess.Popen):
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True
        )
    else:
        proc.kill()
    proc.wait()


def run_with_timeout(n_evs: int, timeout_s: float):
    """Run one case in a subprocess. Returns a result dict."""
    timing_file = OUTPUT_DIR / f"{case_name(n_evs)}_timing.json"
    timing_file.unlink(missing_ok=True)
    start = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__)), "--run-case", str(n_evs)]
    )
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        return {"n_evs": n_evs, "status": "timeout", "total_s": timeout_s}
    wall = time.perf_counter() - start
    if proc.returncode != 0 or not timing_file.exists():
        return {"n_evs": n_evs, "status": "failed", "total_s": wall}
    timing = json.loads(timing_file.read_text())
    return {"n_evs": n_evs, "status": "ok", **timing}


def plot_results(results: pd.DataFrame, timeout_s: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    ok = results[results["status"] == "ok"]
    bad = results[results["status"] != "ok"]
    limit_min = timeout_s / 60

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    ax.plot(ok["n_evs"], ok["run_s"] / 60, "o-", label="simulation run")
    ax.plot(ok["n_evs"], ok["setup_s"] / 60, "s--", label="setup")
    ax.plot(ok["n_evs"], ok["total_s"] / 60, "^-", label="total")
    if len(bad):
        ax.scatter(bad["n_evs"], [limit_min] * len(bad), c="red", marker="x", s=80,
                   label="aborted / failed", zorder=5)
    ax.axhline(limit_min, color="red", linestyle=":", label=f"{limit_min:g} min limit")
    ax.set_xlabel("Number of EVs")
    ax.set_ylabel("Runtime [min]")
    ax.set_title("Runtime vs. number of EVs (1 simulated day)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.loglog(ok["n_evs"], ok["run_s"], "o-", label="simulation run")
    if len(ok) >= 2:
        slope, intercept = np.polyfit(np.log(ok["n_evs"]), np.log(ok["run_s"]), 1)
        xs = np.array([ok["n_evs"].min(), ok["n_evs"].max()])
        ax.loglog(xs, np.exp(intercept) * xs**slope, "k--",
                  label=f"fit: runtime ~ N^{slope:.2f}")
    ax.axhline(timeout_s, color="red", linestyle=":", label="limit")
    ax.set_xlabel("Number of EVs")
    ax.set_ylabel("Runtime [s]")
    ax.set_title("Scaling (log-log)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    ax.plot(ok["n_evs"], ok["run_s"] / ok["n_evs"], "o-")
    ax.set_xscale("log")
    ax.set_xlabel("Number of EVs")
    ax.set_ylabel("Run time per EV [s]")
    ax.set_title("Runtime per EV")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "ev_scalability_runtime.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ev-counts", type=int, nargs="+", default=DEFAULT_EV_COUNTS)
    parser.add_argument("--timeout-min", type=float, default=25.0)
    parser.add_argument("--run-case", type=int, help="internal: run a single case")
    parser.add_argument("--plot-only", action="store_true",
                        help="re-create plots from the saved runtimes.csv")
    args = parser.parse_args()

    if args.run_case is not None:
        run_single_case(args.run_case)
        return

    timeout_s = args.timeout_min * 60
    if args.plot_only:
        plot_results(pd.read_csv(RESULTS_FILE), timeout_s)
        return

    ev_counts = sorted(set(args.ev_counts))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_inputs(ev_counts)

    results = []
    for n_evs in ev_counts:
        print(f"=== {n_evs} EVs (timeout {args.timeout_min:g} min) ===", flush=True)
        result = run_with_timeout(n_evs, timeout_s)
        results.append(result)
        print(f"{n_evs} EVs -> {result['status']} after {result['total_s']:.1f} s",
              flush=True)
        df = pd.DataFrame(results)
        df.to_csv(RESULTS_FILE, index=False)
        if result["status"] != "ok":
            print(f"Stopping: skipping all cases with more than {n_evs} EVs.")
            break

    df = pd.DataFrame(results)
    plot_results(df, timeout_s)
    print(df.to_string(index=False))
    print(f"Plot: {PLOT_DIR / 'ev_scalability_runtime.png'}")


if __name__ == "__main__":
    main()
