# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the ``example_lv_tariff`` study cases and compare them side by side.

This is the market-path counterpart of ``run_experiments.py``.  That harness
calls ``EVPortfolioStrategy.optimize()`` directly under perfect foresight with
no market at all; this one runs each study case through ``World.run()`` -- real
EOM clearing, a real tariff market, real agent messaging -- and then reduces the
per-unit CSV output to the same ``summary.csv`` / ``connection.csv`` shape, so
the two sets of numbers can be laid next to each other.

Run from the repository root, either way::

    python -m examples.inputs.example_lv_tariff.compare_study_cases
    python examples/inputs/example_lv_tariff/compare_study_cases.py

Selected cases only, without re-running what is already there::

    python examples/inputs/example_lv_tariff/compare_study_cases.py \
        --cases no_tariff independent constrained --reuse
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

# Running the file by path puts this directory on sys.path instead of the
# repository root, so `assume` is not importable.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from assume import World
from assume.scenario.loader_csv import load_scenario_folder

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCENARIO = HERE.name
INPUTS = str(HERE.parent)

# `ex_post_peak` is charged on `no_tariff` dispatch after the fact, so it cannot
# have changed anybody's behaviour.  See "dynamic and ex_post_peak" in
# EXPERIMENTS.md.
EX_POST_PEAK_BASE = "no_tariff"
EX_POST_PEAK_RATE_FROM = "peak_price"


def study_cases(path=HERE / "config.yaml"):
    """Every study case defined in the scenario, in file order."""
    return list(yaml.safe_load(path.read_text()))


def settlement(case, path=HERE / "config.yaml"):
    """What the study case told the optimiser, read back for billing.

    The peak and capacity charges steer the schedule through the optimiser's
    objective, but no market settles them, so the bill has to be computed here.
    Reading the rates back out of ``config.yaml`` rather than restating them
    keeps the bill and the incentive the fleet responded to the same numbers.
    """
    params = yaml.safe_load(path.read_text())[case].get("bidding_strategy_params", {})
    return {
        "background_load_mw": float(params.get("ev_background_load_mw", 0.0) or 0.0),
        "peak_price_eur_per_mw": float(params.get("ev_peak_price", 0.0) or 0.0),
        "contracted_capacity_mw": float(params.get("ev_observed_peak_mw", 0.0) or 0.0),
        "import_limit_mw": params.get("ev_import_limit_mw"),
        "export_limit_mw": params.get("ev_export_limit_mw"),
    }


def run_case(case, output_dir, reuse):
    """Simulate one study case and return its CSV output directory."""
    simulation_id = f"{SCENARIO}_{case}"
    result = Path(output_dir) / simulation_id
    if reuse and (result / "unit_dispatch.csv").exists():
        print(f"reusing {case}", flush=True)
        return result
    print(f"running {case}", flush=True)
    world = World(database_uri=None, export_csv_path=str(output_dir))
    load_scenario_folder(world, inputs_path=INPUTS, scenario=SCENARIO, study_case=case)
    world.run()
    return result


def read_case(case, result):
    """Reduce one run's CSV output to per-step fleet quantities.

    Returns ``(frame, n_operators)``.  In ``unit_dispatch`` an EV's ``power`` is
    signed the way the plan is: negative charging, positive discharging.  The
    connection sees the fleet's net withdrawal plus whatever else is behind the
    meter, so ``net`` is negated and the background load added back.
    """
    dispatch = pd.read_csv(result / "unit_dispatch.csv", parse_dates=["time"])
    meta = pd.read_csv(result / "electric_vehicle_meta.csv", index_col=0)
    n_evs = len(meta)
    evs = dispatch[dispatch.unit.isin(meta.index)]
    if evs.empty:
        raise ValueError(f"{case}: no EV dispatch in {result}")
    power = evs.pivot_table(
        index="time", columns="unit", values="power", aggfunc="last"
    )
    market = pd.read_csv(result / "market_meta.csv", parse_dates=["product_start"])
    eom = market[market.market_id == "EOM"].set_index("product_start")["price"]
    price = eom.reindex(power.index).ffill().bfill()

    fee = pd.Series(0.0, index=power.index)
    orders = result / "market_orders.csv"
    if orders.exists():
        book = pd.read_csv(orders, parse_dates=["start_time"])
        tariff = book[book.market_id == "GridTariff"]
        if not tariff.empty:
            # pay_as_clear writes accepted_price on every order, so the last
            # announcement for a delivery step carries the cleared fee.
            cleared = tariff.groupby("start_time")["accepted_price"].last()
            fee = cleared.reindex(power.index).ffill().fillna(0.0)

    background = settlement(case)["background_load_mw"]
    net = -power.sum(axis=1)
    frame = pd.DataFrame(
        {
            "case": case,
            "ev_net_import_mw": net,
            "connection_net_import_mw": net + background,
            "price_eur_per_mwh": price,
            "fee_eur_per_mwh": fee,
        }
    )
    frame.index.name = "datetime"
    return frame.reset_index(), n_evs, int(meta.unit_operator.nunique())


def settle(case, frame, n_evs, n_operators, hours, baseline_cost):
    """Bill one case the way its study case is specified.

    Volumetric fees fall on positive net *connection* import, never credited on
    export; the capacity charge falls on the highest such hour of the run.  Both
    are what the DSO meters, which is the whole connection rather than the
    vehicles alone.
    """
    rules = settlement(case)
    net = frame.ev_net_import_mw
    connection = frame.connection_net_import_mw
    imports = connection.clip(lower=0)
    energy_cost = float((net * frame.price_eur_per_mwh).sum() * hours)
    volumetric = float((imports * frame.fee_eur_per_mwh).sum() * hours)
    peak = float(max(0.0, connection.max()))
    billed_peak = max(0.0, peak - rules["contracted_capacity_mw"])
    capacity_fee = billed_peak * rules["peak_price_eur_per_mw"]

    import_limit = rules["import_limit_mw"]
    export_limit = rules["export_limit_mw"]
    import_excess = (
        (connection - import_limit).clip(lower=0)
        if import_limit is not None
        else pd.Series(0.0, index=frame.index)
    )
    export_excess = (
        (-connection - export_limit).clip(lower=0)
        if export_limit is not None
        else pd.Series(0.0, index=frame.index)
    )
    return {
        "case": case,
        "n_evs": n_evs,
        "n_operators": n_operators,
        # Cases that declare no background load report a connection peak equal
        # to their fleet peak; compare peaks only across cases that agree here.
        "background_load_mw": rules["background_load_mw"],
        "energy_cost_eur": energy_cost,
        "grid_fee_eur": volumetric + capacity_fee,
        "volumetric_fee_eur": volumetric,
        "capacity_fee_eur": capacity_fee,
        "total_cost_eur": energy_cost + volumetric + capacity_fee,
        "energy_cost_delta_vs_no_tariff_eur": energy_cost - baseline_cost,
        "total_cost_delta_vs_no_tariff_eur": energy_cost
        + volumetric
        + capacity_fee
        - baseline_cost,
        "fleet_peak_import_mw": float(max(0.0, net.max())),
        "connection_peak_import_mw": peak,
        "connection_peak_export_mw": float(max(0.0, -connection.min())),
        "overload_hours": float(
            ((import_excess > 1e-7) | (export_excess > 1e-7)).sum() * hours
        ),
        "overload_mwh": float((import_excess + export_excess).sum() * hours),
    }


def compare(cases, output_dir, reuse):
    output_dir = Path(output_dir)
    frames, operators, fleet = {}, {}, {}
    for case in cases:
        result = run_case(case, output_dir, reuse)
        frames[case], fleet[case], operators[case] = read_case(case, result)
    reference = frames[cases[0]]
    hours = float(
        pd.Timedelta(
            reference.datetime.iloc[1] - reference.datetime.iloc[0]
        ).total_seconds()
        / 3600
    )
    baseline = None
    if "no_tariff" in frames:
        base = frames["no_tariff"]
        baseline = float((base.ev_net_import_mw * base.price_eur_per_mwh).sum() * hours)

    rows = []
    for case in cases:
        row = settle(
            case,
            frames[case],
            fleet[case],
            operators[case],
            hours,
            baseline or 0.0,
        )
        if baseline is None:
            row.pop("energy_cost_delta_vs_no_tariff_eur")
            row.pop("total_cost_delta_vs_no_tariff_eur")
        rows.append(row)

    # An unanticipated peak bill on dispatch that was chosen without knowing
    # about it. It cannot change behaviour, so it is a settlement of an existing
    # case rather than a case of its own.
    rate = settlement(EX_POST_PEAK_RATE_FROM)["peak_price_eur_per_mw"]
    if EX_POST_PEAK_BASE in frames and rate:
        row = settle(
            EX_POST_PEAK_BASE,
            frames[EX_POST_PEAK_BASE],
            fleet[EX_POST_PEAK_BASE],
            operators[EX_POST_PEAK_BASE],
            hours,
            baseline or 0.0,
        )
        fee = row["connection_peak_import_mw"] * rate
        row.update(
            case="ex_post_peak",
            capacity_fee_eur=fee,
            grid_fee_eur=row["volumetric_fee_eur"] + fee,
            total_cost_eur=row["energy_cost_eur"] + row["volumetric_fee_eur"] + fee,
        )
        if baseline is not None:
            row["total_cost_delta_vs_no_tariff_eur"] = row["total_cost_eur"] - baseline
        rows.append(row)

    summary = pd.DataFrame(rows)
    connection = pd.concat(frames.values(), ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    connection.to_csv(output_dir / "connection.csv", index=False)
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(
            {
                "scenario": SCENARIO,
                "cases": cases,
                "mode": "market_path_rolling_horizon",
                "step_hours": hours,
                "baseline_case": "no_tariff" if baseline is not None else None,
                "baseline_energy_cost_eur": baseline,
                "settlement": {case: settlement(case) for case in cases},
                "billing": "positive net connection import; no export fee credit; "
                "whole-run peak period",
            },
            indent=2,
        )
        + "\n"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        help="Study cases to compare. Defaults to every case in config.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "examples" / "outputs" / "lv_tariff_study_cases",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Skip cases whose output is already in --output-dir.",
    )
    args = parser.parse_args()
    available = study_cases()
    cases = args.cases or available
    unknown = [c for c in cases if c not in available]
    if unknown:
        raise SystemExit(
            f"Unknown study case(s): {', '.join(unknown)}. "
            f"Available: {', '.join(available)}"
        )
    summary = compare(cases, args.output_dir, args.reuse)
    print(summary.round(4).to_string(index=False))
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
