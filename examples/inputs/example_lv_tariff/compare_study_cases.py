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

Two optimisation paths, one output shape
----------------------------------------

A study case is scheduled either by ``EVPortfolioStrategy`` -- one portfolio
MILP over six independent ``ElectricVehicleUnit``s -- or by ``Building``, one
MILP over a multi-asset building holding the same six vehicles as
``dst_components.ElectricVehicle`` components.  The two write very different
things to disk: the EV path exports one dispatch row per vehicle, the building
exports a single aggregate ``power`` and keeps its per-vehicle state inside the
Pyomo instance it throws away after each window.

So the per-vehicle schedule is collected the same way for both and then put
through *one* replay of the battery physics (:func:`replay`), rather than
trusting either model's own account of what it did.  Unserved driving energy,
terminal SOC and time spent at ``min_soc`` are all read off that replay, so they
mean the same thing on both sides.  For the building the schedule is captured by
:func:`_spy_on_building`, which records the committed step of every rolling
window as it is solved.

Run from the repository root, either way::

    python -m examples.inputs.example_lv_tariff.compare_study_cases
    python examples/inputs/example_lv_tariff/compare_study_cases.py

Selected cases only, without re-running what is already there::

    python examples/inputs/example_lv_tariff/compare_study_cases.py \
        --cases no_tariff independent constrained --reuse
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
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


def case_config(case, path=HERE / "config.yaml"):
    """The raw study case block."""
    return yaml.safe_load(path.read_text())[case]


def optimiser(case, path=HERE / "config.yaml"):
    """Which model schedules this case: ``"building"`` or ``"ev_portfolio"``.

    Read off the study case rather than from a list of names, so a new building
    case is picked up by declaring ``residential_dsm_units`` and nothing else.
    """
    return "building" if case_config(case, path).get("residential_dsm_units") else "ev_portfolio"


def settlement(case, path=HERE / "config.yaml"):
    """What the study case told the optimiser, read back for billing.

    The peak and capacity charges steer the schedule through the optimiser's
    objective, but no market settles them, so the bill has to be computed here.
    Reading the rates back out of ``config.yaml`` rather than restating them
    keeps the bill and the incentive the fleet responded to the same numbers.

    The building expresses none of them: ``ev_import_limit_mw`` and
    ``ev_peak_price`` are ``EVPortfolioStrategy`` parameters with no Building
    equivalent, and its background load is an ordinary inflexible demand read
    from the forecast rather than a strategy parameter.  So a building case
    reports its background load from ``{building}_load_profile`` and zeroes the
    rest, which is the honest answer: those mechanisms cannot be configured on
    that path at all.
    """
    cfg = case_config(case, path)
    params = cfg.get("bidding_strategy_params") or {}
    if optimiser(case, path) == "building":
        return {
            "background_load_mw": float(building_load_profile().mean()),
            "peak_price_eur_per_mw": 0.0,
            "contracted_capacity_mw": 0.0,
            "import_limit_mw": None,
            # `is_prosumer: No` puts total_power_input >= 0 on the building,
            # which is a zero-export connection limit by another name.
            "export_limit_mw": (
                None if is_prosumer(building_units_file(case, path)) else 0.0
            ),
        }
    return {
        "background_load_mw": float(params.get("ev_background_load_mw", 0.0) or 0.0),
        "peak_price_eur_per_mw": float(params.get("ev_peak_price", 0.0) or 0.0),
        "contracted_capacity_mw": float(params.get("ev_observed_peak_mw", 0.0) or 0.0),
        "import_limit_mw": params.get("ev_import_limit_mw"),
        "export_limit_mw": params.get("ev_export_limit_mw"),
    }


# ---------------------------------------------------------------------------
# The fleet, and the battery physics both paths are replayed through
# ---------------------------------------------------------------------------

#: Name of the `building` unit in residential_dsm_units.csv, which is also the
#: prefix BuildingForecaster expects on every one of its component profiles.
BUILDING_ID = "aggregator"


def building_units_file(case=None, path=HERE / "config.yaml"):
    """The residential_dsm_units CSV a building case uses, or the default one."""
    name = case and case_config(case, path).get("residential_dsm_units")
    return HERE / (name or "residential_dsm_units.csv")


def _building_components(path=None):
    """The building's EV component rows, indexed by the EV name they mirror."""
    rows = pd.read_csv(path or building_units_file())
    rows = rows[rows.technology.astype(str).str.startswith("electric_vehicle_")]
    return rows.set_index(rows.technology.str.removeprefix("electric_vehicle_"))


def is_prosumer(path=None):
    """Whether the building may export. ``No`` means total_power_input >= 0."""
    value = pd.read_csv(path or building_units_file()).is_prosumer.dropna().iloc[0]
    return str(value).strip().lower() in ("yes", "true", "1")


def building_load_profile(path=HERE / "forecasts_df.csv"):
    """The building's inflexible demand, i.e. its background load, in MW."""
    return pd.read_csv(path, index_col=0)[f"{BUILDING_ID}_load_profile"]


def fleet(path=HERE / "electric_vehicle_units.csv"):
    """Battery parameters of the standalone EV units, one row per vehicle.

    Also the parameters the building's components are checked against.  The two
    CSVs are meant to describe the same six vehicles; if they have drifted apart
    the comparison measures the input conversion rather than the aggregation, so
    raise here instead of reporting a difference that is not about the models.
    ``max_power_charge`` is negative on the EV units (the Storage sign
    convention) and positive on the building components, so compare magnitudes.
    """
    evs = pd.read_csv(path, index_col=0)
    components = _building_components()
    mismatched = []
    for field in (
        "capacity",
        "min_soc",
        "max_soc",
        "initial_soc",
        "max_power_charge",
        "max_power_discharge",
        "efficiency_charge",
        "efficiency_discharge",
    ):
        left = evs[field].abs().round(9)
        right = components[field].abs().round(9).reindex(left.index)
        if not left.equals(right):
            mismatched.append(field)
    if mismatched:
        raise ValueError(
            "The EV units and the building's EV components no longer describe "
            "the same fleet, so the two paths are not comparable; differing "
            f"fields: {', '.join(mismatched)}"
        )
    return evs


def profiles(path=HERE / "forecasts_df.csv"):
    """``(availability, trip_energy)`` per vehicle, indexed by timestamp.

    generate_inputs.py writes each vehicle's two profiles twice, once under
    ``EV_i_*`` for the standalone units and once under
    ``aggregator_electric_vehicle_EV_i_*`` for the building components, from the
    same arrays.  Check that here rather than assume it: if the two copies ever
    diverge the runs are being driven by different inputs.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    names = list(fleet().index)
    for name in names:
        for kind in ("availability_profile", "trip_energy_consumption"):
            mirror = f"{BUILDING_ID}_electric_vehicle_{name}_{kind}"
            if mirror in df and not df[f"{name}_{kind}"].equals(df[mirror]):
                raise ValueError(
                    f"{name}: the EV and building copies of {kind} differ; "
                    "re-run generate_inputs.py"
                )
    return (
        df[[f"{n}_availability_profile" for n in names]].set_axis(names, axis=1),
        df[[f"{n}_trip_energy_consumption" for n in names]].set_axis(names, axis=1),
    )


def replay(power, availability, trip, ev, hours):
    """Put one vehicle's signed dispatch through the EV unit's own physics.

    *power* is signed the way both paths report it: positive discharging,
    negative charging.  Returns ``(soc, unmet)`` -- state of charge at the start
    of each step, and the driving energy the battery could not supply.

    This is a transcription of ``ElectricVehicleUnit._transition``, applied
    identically to whichever optimiser produced the schedule.  It is the only
    place the two paths are judged on service, because it is the only account of
    what happened that does not come from the model being measured.

    The third return value is the state of charge *after* the final step.  The
    series itself holds the SOC at the start of each step, so its last entry is
    the state before the run's last hour has been dispatched -- which would
    charge the EV path, whose terminal-SOC target sits exactly one step past the
    end of the series, with a shortfall it does not have.
    """
    capacity = float(ev.capacity)
    floor, ceiling = ev.min_soc * capacity, ev.max_soc * capacity
    energy = float(ev.initial_soc) * capacity
    socs, unmets = [], []
    for t in power.index:
        socs.append(energy / capacity)
        plug = float(availability.at[t])
        p = float(power.at[t])
        p = min(max(p, -abs(ev.max_power_charge) * plug), ev.max_power_discharge * plug)
        p = min(p, max(0.0, energy - floor) * ev.efficiency_discharge / hours)
        p = max(p, -max(0.0, ceiling - energy) / ev.efficiency_charge / hours)
        energy += (
            max(-p, 0.0) * ev.efficiency_charge - max(p, 0.0) / ev.efficiency_discharge
        ) * hours
        drive = float(trip.at[t])
        unmets.append(max(0.0, floor - (energy - drive)))
        energy = max(floor, energy - drive)
    return (
        pd.Series(socs, index=power.index),
        pd.Series(unmets, index=power.index),
        energy / capacity,
    )


# ---------------------------------------------------------------------------
# Running a case, and getting a per-vehicle schedule out of either path
# ---------------------------------------------------------------------------

#: Filenames written next to the run's CSV output, so --reuse can skip a case
#: without losing what only existed while it was running.
SCHEDULE_FILE = "building_schedule.csv"
DIAGNOSTICS_FILE = "run_diagnostics.json"


def _building_unit(world):
    """The `building` unit in a loaded world, or ``None`` for an EV-path case."""
    for operator in world.unit_operators.values():
        for unit in operator.units.values():
            if getattr(unit, "evs", None):
                return unit
    return None


def _spy_on_building(unit, records):
    """Record the committed step of every rolling window the building solves.

    ``Building`` keeps its per-vehicle schedule inside the Pyomo instance for the
    current window and writes only the building's aggregate power to its
    outputs, so the moment the window is solved is the only place a per-vehicle
    number exists.  ``_update_init_states`` is the last hook that runs while the
    instance is still in scope, so wrap that.

    A window with no feasible solution never reaches the hook.  That is not a
    gap in the instrumentation but the measurement itself: the building models
    driving demand as a hard constraint with no slack, so an unservable trip can
    only show up as an infeasible window -- and the run then silently keeps the
    previous schedule.
    """
    original = unit._update_init_states

    def record(instance, commit_local, init_states):
        start = unit._rh_window_start
        for local in range(commit_local + 1):
            for ev in unit.evs:
                block = instance.dsm_blocks[ev]
                records.append(
                    {
                        "step": start + local,
                        "unit": ev.removeprefix("electric_vehicle_"),
                        "charge": pyo.value(block.charge[local]),
                        "discharge": pyo.value(block.discharge[local]),
                        "soc_planned": pyo.value(block.soc[local]),
                    }
                )
        return original(instance, commit_local, init_states)

    unit._update_init_states = record


class _WindowWatcher(logging.Handler):
    """Watch a run for the two ways a unit can stop scheduling without saying so.

    A window the building cannot solve is left uncommitted and the run carries on
    with the previous schedule.  Worse, an exception raised inside a bidding
    strategy is caught by the agent scheduler and logged: the simulation runs to
    completion, the unit simply stops bidding, and the CSV output looks like a
    unit that chose to do nothing.  Either one invalidates a comparison, so both
    are counted and :func:`run_case` refuses to summarise a run that hit the
    second.
    """

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.infeasible = 0
        self.errors = 0
        self.messages = []

    def emit(self, record):
        text = record.getMessage()
        if "no feasible solution" in text:
            self.infeasible += 1
        if record.levelno >= logging.ERROR:
            self.errors += 1
            if record.exc_info is not None:
                text = f"{text}: {record.exc_info[1]}"
        self.messages.append(f"{record.levelname} {record.name}: {text}")


def run_case(case, output_dir, reuse):
    """Simulate one study case; return its CSV output directory and diagnostics.

    Diagnostics are the things that only exist while the simulation is running --
    wall-clock solve time, the building's per-vehicle schedule, the windows it
    failed to solve -- so they are written next to the CSV output and read back
    under ``--reuse``.
    """
    simulation_id = f"{SCENARIO}_{case}"
    result = Path(output_dir) / simulation_id
    if reuse and (result / DIAGNOSTICS_FILE).exists():
        print(f"reusing {case}", flush=True)
        return result, json.loads((result / DIAGNOSTICS_FILE).read_text())

    print(f"running {case}", flush=True)
    watcher = _WindowWatcher()
    # The root logger, not "assume": an exception raised out of a bidding
    # strategy is reported by the agent framework under its own logger name.
    logging.getLogger().addHandler(watcher)
    records = []
    try:
        world = World(database_uri=None, export_csv_path=str(output_dir))
        load_scenario_folder(
            world, inputs_path=INPUTS, scenario=SCENARIO, study_case=case
        )
        building = _building_unit(world)
        if building is not None:
            _spy_on_building(building, records)
        started = time.perf_counter()
        world.run()
        elapsed = time.perf_counter() - started
    finally:
        logging.getLogger().removeHandler(watcher)

    if watcher.errors:
        raise RuntimeError(
            f"{case}: {watcher.errors} error(s) were logged during the run, so at "
            "least one agent stopped scheduling part-way through and the output "
            "is not a result. First: "
            + next(m for m in watcher.messages if m.startswith("ERROR"))
        )

    result.mkdir(parents=True, exist_ok=True)
    if records:
        pd.DataFrame(records).to_csv(result / SCHEDULE_FILE, index=False)
    diagnostics = {
        "case": case,
        "optimiser": optimiser(case),
        # Wall clock for the whole run, not solver time alone: it includes market
        # clearing and agent messaging, which are the same on both paths, so the
        # difference between two cases is the optimisation.
        "run_seconds": round(elapsed, 2),
        "infeasible_windows": watcher.infeasible,
        "logged_errors": watcher.errors,
        "warnings": watcher.messages[:20],
        "warning_count": len(watcher.messages),
    }
    (result / DIAGNOSTICS_FILE).write_text(json.dumps(diagnostics, indent=2) + "\n")
    return result, diagnostics


def building_schedule(result, index):
    """Per-vehicle signed dispatch recorded from the building's solved windows.

    Positive is discharging, the same convention the EV units export, so the two
    paths produce the same shape of frame.  Steps the building never optimised --
    the first hour, before any market has opened, and any window it could not
    solve -- are zero.
    """
    records = pd.read_csv(result / SCHEDULE_FILE)
    records["power"] = records.discharge - records.charge
    power = records.pivot_table(
        index="step", columns="unit", values="power", aggfunc="last"
    )
    power.index = [index[int(step)] for step in power.index]
    return power.reindex(index).fillna(0.0)


def read_case(case, result):
    """Reduce one run's output to per-step connection quantities and a schedule.

    Both paths are signed the same way once read: ``power`` is positive
    discharging and negative charging per vehicle, ``net`` is the fleet's net
    withdrawal, and the connection is that plus whatever else sits behind the
    meter.  Where they differ is what is on disk.

    On the EV path each vehicle has its own dispatch row and the background load
    is never bid at all -- it only ever existed inside the optimiser -- so it is
    added back here.  On the building path the six vehicles are components of
    one unit, which exports a single aggregate ``power`` that *already* contains
    its inflexible demand, so that series is the connection directly and the
    fleet is what remains after the background load is taken out.  The
    per-vehicle split for the building comes from the schedule recorded while it
    ran.

    Returns ``(frame, power, n_evs, n_operators)``.
    """
    dispatch = pd.read_csv(result / "unit_dispatch.csv", parse_dates=["time"])
    background = settlement(case)["background_load_mw"]

    if optimiser(case) == "building":
        aggregate = dispatch[dispatch.unit == BUILDING_ID]
        if aggregate.empty:
            raise ValueError(f"{case}: no building dispatch in {result}")
        # A DSM unit reports demand negative, so the connection's net import is
        # the negated aggregate. Overlapping save windows can repeat a step, so
        # take the last write for each one, as the EV branch does below.
        connection = -aggregate.groupby("time")["power"].last().sort_index()
        net = connection - background
        power = building_schedule(result, connection.index)
        n_evs, n_operators = len(power.columns), 1
    else:
        meta = pd.read_csv(result / "electric_vehicle_meta.csv", index_col=0)
        evs = dispatch[dispatch.unit.isin(meta.index)]
        if evs.empty:
            raise ValueError(f"{case}: no EV dispatch in {result}")
        power = evs.pivot_table(
            index="time", columns="unit", values="power", aggfunc="last"
        )
        net = -power.sum(axis=1)
        connection = net + background
        n_evs, n_operators = len(meta), int(meta.unit_operator.nunique())

    market = pd.read_csv(result / "market_meta.csv", parse_dates=["product_start"])
    eom = market[market.market_id == "EOM"].set_index("product_start")["price"]
    price = eom.reindex(net.index).ffill().bfill()

    fee = pd.Series(0.0, index=net.index)
    orders = result / "market_orders.csv"
    if orders.exists():
        book = pd.read_csv(orders, parse_dates=["start_time"])
        tariff = book[book.market_id == "GridTariff"]
        if not tariff.empty:
            # pay_as_clear writes accepted_price on every order, so the last
            # announcement for a delivery step carries the cleared fee.
            cleared = tariff.groupby("start_time")["accepted_price"].last()
            fee = cleared.reindex(net.index).ffill().fillna(0.0)

    frame = pd.DataFrame(
        {
            "case": case,
            "optimiser": optimiser(case),
            "ev_net_import_mw": net,
            "connection_net_import_mw": connection,
            "price_eur_per_mwh": price,
            "fee_eur_per_mwh": fee,
        }
    )
    frame.index.name = "datetime"
    return frame.reset_index(), power, n_evs, n_operators


def service(case, power, hours):
    """Replay the fleet's schedule and report what the driving actually got.

    The same physics for both paths (:func:`replay`), so ``unserved_driving_mwh``
    means one thing across the comparison.  ``terminal_soc`` is reported because
    the two optimisers disagree about it by construction: ``EVPortfolioStrategy``
    targets each vehicle's initial SOC at the end of every planning window, the
    building has no terminal-SOC term at all and so values energy left in a
    battery at zero.  A fleet that ends the run flat has borrowed against the
    days after it, and the cost figures alone will not say so.
    """
    evs = fleet()
    availability, trip = profiles()
    unserved, terminal, floored = 0.0, {}, 0
    for name, ev in evs.iterrows():
        if name not in power.columns:
            raise ValueError(f"{case}: no schedule for {name}")
        column = power[name].reindex(availability.index).fillna(0.0)
        soc, unmet, final = replay(column, availability[name], trip[name], ev, hours)
        unserved += float(unmet.sum())
        terminal[name] = float(final)
        floored += int((soc <= float(ev.min_soc) + 1e-9).sum())
    initial = float(evs.initial_soc.mean())
    mean_terminal = sum(terminal.values()) / len(terminal)
    return {
        "unserved_driving_mwh": unserved,
        "mean_terminal_soc": mean_terminal,
        # Energy the fleet ends the run holding, relative to where it started.
        "terminal_energy_delta_mwh": (mean_terminal - initial)
        * float(evs.capacity.mean())
        * len(evs),
        "vehicle_hours_at_min_soc": floored * hours,
    }


def _vwap(volume, price):
    """Volume-weighted average price, zero when nothing was bought."""
    total = float(volume.sum())
    return float((volume * price).sum() / total) if total > 1e-12 else 0.0


def settle(case, frame, n_evs, n_operators, hours, baseline_cost):
    """Bill one case the way its study case is specified.

    Volumetric fees fall on positive net *connection* import, never credited on
    export; the capacity charge falls on the highest such hour of the run.  Both
    are what the DSO meters, which is the whole connection rather than the
    vehicles alone.

    ``energy_cost_eur`` is the fleet's bill and excludes the background load,
    which is what makes it comparable across the two paths: the EV portfolio
    never buys that load on the EOM while the building does, and the difference
    is bookkeeping rather than behaviour.  ``connection_energy_cost_eur`` is the
    whole meter, for when the connection rather than the fleet is the subject.
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
        "optimiser": optimiser(case),
        "n_evs": n_evs,
        "n_operators": n_operators,
        # Cases that declare no background load report a connection peak equal
        # to their fleet peak; compare peaks only across cases that agree here.
        "background_load_mw": rules["background_load_mw"],
        "energy_cost_eur": energy_cost,
        "connection_energy_cost_eur": float(
            (connection * frame.price_eur_per_mwh).sum() * hours
        ),
        # What a MWh of charging actually cost this fleet, fee included, over
        # the hours it chose to charge in. Used below to price the energy the
        # two paths leave in their batteries at different levels.
        "fleet_import_vwap_eur_per_mwh": _vwap(
            net.clip(lower=0), frame.price_eur_per_mwh + frame.fee_eur_per_mwh
        ),
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


def equalise_terminal_energy(row):
    """Charge a case for the battery energy it ends the run without.

    A four-day run that ends with flatter batteries than it started has paid for
    part of its energy out of stock rather than out of the market, and the two
    paths do this to very different degrees: ``EVPortfolioStrategy`` targets each
    vehicle's initial SOC at the end of every planning window, while the Building
    MILP has no terminal-SOC term at all and so values energy left in a battery
    at exactly zero.  Comparing their bills without this adjustment compares a
    fleet that kept its charge against one that sold it.

    The missing energy is priced at what this fleet actually paid per MWh
    charged, grossed up by the charging efficiency, since that is what refilling
    would have cost it over the same hours.  Reported separately rather than
    folded into ``total_cost_eur``: it is an adjustment, not a bill anyone sends.
    """
    evs = fleet()
    replacement = (
        -row["terminal_energy_delta_mwh"]
        / float(evs.efficiency_charge.mean())
        * row["fleet_import_vwap_eur_per_mwh"]
    )
    return {
        "terminal_energy_replacement_eur": replacement,
        "cost_at_equal_terminal_energy_eur": row["total_cost_eur"] + replacement,
    }


def compare(cases, output_dir, reuse):
    output_dir = Path(output_dir)
    frames, operators, sizes, schedules, diagnostics = {}, {}, {}, {}, {}
    for case in cases:
        result, diagnostics[case] = run_case(case, output_dir, reuse)
        frames[case], schedules[case], sizes[case], operators[case] = read_case(
            case, result
        )
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
            sizes[case],
            operators[case],
            hours,
            baseline or 0.0,
        )
        # What the driving got, and what it cost the batteries to give it.
        row.update(service(case, schedules[case], hours))
        row.update(equalise_terminal_energy(row))
        row["infeasible_windows"] = diagnostics[case]["infeasible_windows"]
        row["run_seconds"] = diagnostics[case]["run_seconds"]
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
            sizes[EX_POST_PEAK_BASE],
            operators[EX_POST_PEAK_BASE],
            hours,
            baseline or 0.0,
        )
        row.update(service(EX_POST_PEAK_BASE, schedules[EX_POST_PEAK_BASE], hours))
        row["infeasible_windows"] = diagnostics[EX_POST_PEAK_BASE][
            "infeasible_windows"
        ]
        row["run_seconds"] = diagnostics[EX_POST_PEAK_BASE]["run_seconds"]
        fee = row["connection_peak_import_mw"] * rate
        row.update(
            case="ex_post_peak",
            capacity_fee_eur=fee,
            grid_fee_eur=row["volumetric_fee_eur"] + fee,
            total_cost_eur=row["energy_cost_eur"] + row["volumetric_fee_eur"] + fee,
        )
        if baseline is not None:
            row["total_cost_delta_vs_no_tariff_eur"] = row["total_cost_eur"] - baseline
        # After the peak bill has replaced total_cost_eur, not before it.
        row.update(equalise_terminal_energy(row))
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
                "optimiser": {case: optimiser(case) for case in cases},
                "settlement": {case: settlement(case) for case in cases},
                "diagnostics": diagnostics,
                "billing": "positive net connection import; no export fee credit; "
                "whole-run peak period",
                "service": "every case replayed through ElectricVehicleUnit physics, "
                "so unserved driving energy means the same on both paths",
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
