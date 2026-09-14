# EV aggregator — status and TODOs

Scope: `assume/units/electric_vehicle.py`, `assume/strategies/ev_aggregator.py`,
and the `examples/inputs/example_lv_tariff` scenario.

Full test suite green: **543 passed** (39 of them in `tests/test_ev_aggregator.py`).

---

## Where this stands (handoff)

Section 1 is **done**: the scenario now has ten study cases that all run
end-to-end through `World.run()`, and the offline harness can be run on the same
footing for comparison.

```bash
# every study case, through the real market path -> summary.csv / connection.csv
python examples/inputs/example_lv_tariff/compare_study_cases.py
# the offline harness on the same footing (no PYTHONPATH needed)
python examples/inputs/example_lv_tariff/run_experiments.py --horizon-mode rolling_horizon
```

The ten cases, and what each isolates:

| case | isolates |
|---|---|
| `no_tariff` | reference |
| `independent` | one owner per EV -- the aggregation benefit |
| `flat_tariff` | a flat volumetric fee |
| `tou_tariff` | a time-of-use fee, fully announced (**the reference tariff case**) |
| `capacity_tariff` | a fee priced off announced aggregate withdrawal |
| `tou_tariff_myopic` | ablation: announcement *length* (`count: 1`) |
| `tou_tariff_daily_announcement` | ablation: announcement *frequency* (`opening_frequency: 24h`) |
| `constrained` | a hard connection limit |
| `peak_price` | a capacity charge on the run's peak hour |
| `capacity_charge` | the same above a contracted capacity |

Invariants worth not breaking, each pinned by a test in
`tests/test_ev_aggregator.py`: every case shares `&ev_defaults`
(`ev_look_ahead_horizon: 27h`, `ev_background_load_mw: 0.005`); every tariff case
announces `count: 24` except the named myopic one; the announcement reaches the
end of the planning window; the frequency ablation differs from `tou_tariff` in
exactly one key.

Section 2 is **done** as well: the Building comparison has been run, and the
two optimisers agree to within 3 % on cost and 2 % on peak once export and
terminal energy are matched. It left one open defect behind -- exports from a
`Building` are bid as supply at the price cap and never clear -- which blocks
the V2G half of that comparison. **Next session starts there, or at section 3**,
which holds the smaller loose ends.

---

## Done

### Correctness

- [x] **Shared planning horizon is now safe for every vehicle.**
  The terminal-SOC target is imposed on all vehicles at one shared horizon end,
  but the old code picked that end as a plain `max` over each vehicle's
  individually-safe end. That maximum routinely landed inside *another*
  vehicle's trip and made the whole portfolio MILP infeasible — one slow-charging
  car could take the entire fleet down. `_horizon_end` now iterates to a fixed
  point: extend, re-ask every vehicle about the new end, repeat.
  Covered by `test_shared_horizon_stays_feasible_for_every_vehicle`.

- [x] **Grid fees are no longer credited to V2G exports.**
  `calculate_bids` used to leave the tariff to `price_plus_grid_fee`, which folds
  the fee into the price forecast and therefore applies it *symmetrically* — a
  280 EUR/MWh peak-hour fee became a 280 EUR/MWh export subsidy, the opposite of
  what a volumetric import tariff does, and flatly contradicting the class
  docstring. The strategy now reads the announced fee itself
  (`_connection_fees`) and charges it on net import only, taking the energy price
  from the new `grid_fee_free_price` helper so it is never counted twice.
  Covered by `test_grid_fee_is_not_credited_to_v2g_exports`.

- [x] **`grid_fees` and `background_load` are live in the simulation path.**
  They were only ever passed by the offline harness, so connection limits in a
  real ASSUME run silently ignored the co-located background load.
  (`background_load` was in fact still missing from `calculate_bids` when this
  was written; it was finished under TODO 1 below, via `ev_background_load_mw`.)

- [x] **Fee and capacity charge are billed on the whole connection.**
  `imports` now includes `background_load`, so the grid fee and the peak see what
  the DSO actually meters. Previously the 60-EV `capacity` case reported *zero*
  billable excess while the connection sat 25 % over contract (0.25 MW against a
  0.20 MW contract). It now holds the total at exactly 0.20 MW.
  `run_experiments.py` settlement was realigned to match, and reports
  `ev_peak_import_mw` alongside the connection peak.
  Covered by `test_peak_and_fee_are_billed_on_the_whole_connection`.

- [x] **Capacity charge carries the realised peak across rolling rounds.**
  `_realised_peak` feeds the peak already delivered back in as the floor, so a
  rolling plan stops re-buying the same peak every opening. Billing period is
  currently the whole run — see the TODO below.
  Covered by `test_realised_peak_carries_across_rolling_rounds`.

### Robustness

- [x] **Driving demand and terminal SOC are penalised slacks, not hard constraints.**
  A fleet merely short of energy used to raise `RuntimeError` and abort the
  simulation, while the vehicle model itself floors at `min_soc` and books
  `unmet_driving_energy`. The MILP now mirrors the physics: slacks at
  `DEFAULT_SLACK_PENALTY` (1e6 EUR/MWh) stay at zero whenever a feasible schedule
  exists, and a nonzero slack is reported through
  `outputs["ev_planned_unmet_energy"]` plus a warning.
  Covered by `test_impossible_trip_degrades_instead_of_raising`.

- [x] **Solver is configurable and bounded.**
  `ev_solver`, `ev_solver_time_limit`, `ev_mip_gap`. A solve that stops on a time
  limit with an incumbent is used with a warning rather than aborting the run — a
  slightly suboptimal schedule beats no schedule in a rolling simulation.
  Measured: a 6-vehicle / 24 h all-negative-price instance went from **523 s
  unbounded to 34 s**.

- [x] **Typo'd `ev_*` parameters are rejected.**
  `bidding_strategy_params` is shared across strategies and `**kwargs` swallowed
  everything, so `ev_look_ahead_horizen` silently did nothing. Unknown `ev_`-
  prefixed keys now raise; other strategies' parameters still pass through.
  Covered by `test_typo_in_an_ev_parameter_is_rejected`.

- [x] **Minimum-power and ramp limits are rejected instead of ignored.**
  Neither the unit dispatch nor the MILP models them, but `Storage` accepts them,
  so a scenario could set a limit that was silently dropped everywhere it
  mattered. Covered by `test_unmodelled_storage_limits_are_rejected`.

### Performance

- [x] **LP relaxation solved first, MILP only when it is actually needed.**
  The charge/discharge binary only binds when the *effective* price (market price
  plus grid fee) can go negative; otherwise the relaxation is exactly tight
  (measured gap 4.6e-14 on a 60-vehicle instance) and roughly 2-3x faster. The
  strategy solves the relaxation, checks the *solution* for simultaneous
  charge/discharge, and re-solves as a MILP only if it finds any — a test on the
  answer rather than on a sufficient condition, so no combination of fees, peak
  prices or limits can fool it. `ev_force_milp=True` opts out.
  Covered by `test_relaxation_and_milp_agree_on_the_plan` and
  `test_negative_prices_still_force_the_milp`.

### Documentation

- [x] Module-level docstring in `ev_aggregator.py` covering the MILP structure
      (variable table), why the binary exists and when it binds, the soft-constraint
      rationale, and who owns the grid fee.
- [x] Module-level docstring in `electric_vehicle.py` covering sign/availability/
      trip conventions, the degrade-don't-fail contract, and an output table.
- [x] Docstrings on every non-trivial method, and `Args:`/`Raises:` on both
      constructors.
- [x] `grid_fee_free_price` documented in `assume/common/forecast_algorithms.py`,
      including *why* an optimiser that models the fee on net import must use it.

### Incidental

- [x] Fixed `import nest_asyncio2` → `nest_asyncio` in `assume/world.py:172`.
      This was an unrelated typo that broke `World()` construction entirely, so no
      scenario could run at all. Fixing it is what unblocked the end-to-end runs
      below.

---

## Open TODOs

### 1. ~~Run the full case matrix as normal ASSUME scenarios~~ — done

All five original study cases plus four new ones run end-to-end through
`World.run()`. The harness cases that had no scenario equivalent are now
ordinary study cases in `config.yaml`, and a post-processing script puts the two
paths into the same output shape.

- [x] **`constrained`, `peak_price` and `capacity_charge` are study cases.**
      Pure config — `ev_import_limit_mw`, `ev_export_limit_mw`, `ev_peak_price`
      and `ev_observed_peak_mw` in `bidding_strategy_params`, no plumbing. Each
      lands where it should: `constrained` holds the connection at exactly its
      0.025 MW rating, `capacity_charge` sits flat on its 0.02 MW contract.
- [x] **`independent` is a study case.** `generate_inputs.py` now also writes
      `electric_vehicle_units_independent.csv` and
      `unit_operators_independent.csv` — the same six vehicles, one owner each.
      The aggregation benefit is measurable inside the market path: 0.74 EUR and
      a connection peak of 0.071 MW against 0.049 MW aggregated.
- [x] **`dynamic` and `ex_post_peak` decided — neither becomes a study case.**
      `dynamic`'s live counterpart is `capacity_tariff`, which prices the fee off
      the withdrawal aggregators actually announced instead of off a frozen
      ex-ante baseline. `ex_post_peak` is unanticipated by construction, so it is
      a *settlement* of `no_tariff` dispatch rather than a scenario;
      `compare_study_cases.py` reports it as such. Written up in EXPERIMENTS.md.
- [x] **`--horizon-mode` switch.** `run_experiments.py` takes
      `--horizon-mode`, `--look-ahead-horizon` and `--rolling-step` (also
      `horizon_mode` / `look_ahead_horizon` / `rolling_step` in
      `experiments.yaml`, recorded in `manifest.json`). Under
      `rolling_horizon` the harness and the study cases agree on the physical
      quantities — `peak_price` gives a 0.0088 MW connection peak either way —
      while euro figures still differ because the harness settles at the
      forecast price and the simulation at the cleared price.
- [x] **`run_experiments.py` no longer needs `PYTHONPATH=.`.** It puts the
      repository root on `sys.path` when run by file path. Documented in
      EXPERIMENTS.md.
- [x] **`compare_study_cases.py`** runs the study cases through `World.run()`
      and writes `summary.csv` / `connection.csv` in the harness's shape, plus
      `comparison_manifest.json`. It settles the peak and capacity charges in
      post-processing, since no market clears them.

Incidental, found while doing the above:

- [x] **`background_load` was *not* live in the simulation path.** It was listed
      as done above, but `calculate_bids` only ever passed `grid_fees` —
      `optimize()` accepted a `background_load` the simulation never supplied, so
      a connection limit in a real ASSUME run still saw the vehicles alone. Added
      `ev_background_load_mw`, passed it from `calculate_bids`, and included it
      in the `_realised_peak` floor so the rolling peak is the connection's and
      not the fleet's. Covered by
      `test_declared_background_load_reaches_the_simulation_path` and
      `test_background_load_raises_the_realised_peak_floor`.

Also done, decided after the port:

- [x] **Every tariff case announces the full day (`count: 24`).** The shared
      `grid_tariff` anchor was `count: 1`, so only about three hours of each
      planning window carried a published fee and the rest persisted one repeated
      value -- the fee was flat inside every round and could not reallocate
      anything within it. `tou_tariff` and `capacity_tariff` returned
      byte-identical numbers under that setting, which is the tell; they separate
      now. The old behaviour survives as `tou_tariff_myopic` (was
      `tou_tariff_day_ahead`, which became an exact duplicate of `tou_tariff`
      once the default changed).
- [x] **`ev_look_ahead_horizon: 27h` everywhere**, via a shared `&ev_defaults`
      block merged into all ten cases with `<<: *ev_defaults`. 27h is exactly
      `first_delivery` (3h) + `count` (24h), so the announcement covers the
      planning window end to end with no persisted tail. Non-tariff cases take
      the same 27h: a case that plans over a different window is not comparable
      with one that does not. This also closes the old "48h window, 24h
      announced" gap -- that was the same defect as a short `count`, moved to the
      other end of the window.
- [x] **Every case declares `ev_background_load_mw: 0.005`** (same shared block),
      tariff cases included, so the connection is one physical thing across the
      whole comparison. In the tariff cases this changes the schedule and not
      only the reported peak: the fee falls on net connection import, so the
      background load makes it bite at the margin during hours the fleet would
      otherwise export itself to zero net import. That is what the DSO meters.
- [x] **`opening_frequency` is an ablation, defaulting to hourly.**
      `tou_tariff_daily_announcement` publishes the same `count: 24` fee once a
      day instead of republishing it hourly, and differs from `tou_tariff` in
      that one key and nothing else (pinned by
      `test_announcement_frequency_ablation_differs_only_in_frequency`).

      The result is worth knowing and was not obvious: republished hourly, the
      aggregator always holds a fee for the next 24h, so its 27h window is always
      covered. Published once a day, the *announced depth decays through the
      day* -- 26h just after the auction, ~3h just before the next -- so for most
      of the day it is again optimising against a persisted tail. Covering a 27h
      window at all times from a once-a-day auction needs `count` near 48.

Measured over the four-day run, same 30/280 EUR/MWh time-of-use fee
(`compare_study_cases.py`, plus the penalised-hour share computed from
`connection.csv`):

| case | announcement | import in penalised hours | grid fee |
|---|---|---|---|
| `no_tariff` | none | 38.2 % | -- |
| `tou_tariff` | 24h, hourly | **11.5 %** | **18.61 EUR** |
| `tou_tariff_daily_announcement` | 24h, once a day | 25.2 % | 60.72 EUR |
| `tou_tariff_myopic` | 1h, hourly | 32.3 % | 53.61 EUR |

Follow-up the frequency ablation surfaced:

- [ ] `tou_tariff_daily_announcement` conflates two things it would be better to
      separate: publishing *less often*, and publishing *less depth on average*.
      A third case at `opening_frequency: 24h` with `count: 48` would hold the
      announced depth at >= 27h for the whole day and isolate the frequency
      effect alone. Worth adding if the once-a-day result is going to be quoted
      as "daily announcement is worse" rather than "a 24-product daily
      announcement leaves a persisted tail for most of the day".

### 2. ~~Compare against the Building class as aggregator~~ — done

Four study cases pair the Building MILP against the portfolio MILP on the same
six vehicles, and `compare_study_cases.py` runs both paths into one summary.
Written up in full in EXPERIMENTS.md ("The Building comparison"); the headline:

```bash
python examples/inputs/example_lv_tariff/compare_study_cases.py --cases \
    no_tariff no_tariff_no_export building_no_tariff \
    tou_tariff tou_tariff_no_export building_tou_tariff
```

**The two optimisers agree on the answer and disagree on the route to it.**
Once export is matched and terminal energy is priced, total cost differs by 3 %
without a tariff and 2.5 % with one, and the connection peak by under 2 %. Hour
by hour the connection profiles correlate at only 0.50 and 0.61, and differ by
up to 31 kW and 54 kW on connections peaking at 35 kW and 70 kW.

| | `no_tariff_no_export` | `building_no_tariff` | `tou_tariff` | `building_tou_tariff` |
|---|---|---|---|---|
| total cost, EUR | 4.15 | 2.57 | 24.18 | 19.48 |
| **at equal terminal energy** | **4.94** | **4.79** | **23.87** | **23.29** |
| connection peak, MW | 0.0356 | 0.0353 | 0.0710 | 0.0695 |
| import in 280 EUR/MWh hours | -- | -- | 0.0 % | 0.0 % |
| mean terminal SOC | 0.458 | 0.200 | 0.629 | 0.229 |
| unserved driving, MWh | 0.0 | 0.0 | 0.0 | 0.0 |
| run wall clock, s | ~9 | ~16 | ~18 | ~19 |

- [x] **The Building path runs against the same EOM and GridTariff markets.**
      `building_no_tariff` and `building_tou_tariff` set
      `residential_dsm_units: residential_dsm_units.csv`,
      `electric_vehicle_units: null` and `unit_operators: null` (no portfolio
      strategy). `generate_inputs.py` now also writes the building's component
      profiles and its `aggregator_load_profile`, which is the same 0.005 MW
      the EV cases declare as `ev_background_load_mw`.
- [x] **The two EV models are equivalent, and it is now checked rather than
      assumed.** `dst_components.ElectricVehicle` takes
      `external_trip_energy_consumption` as well as `external_trip_distance`,
      and that mode is battery-side energy per step exactly as
      `ElectricVehicleUnit` wants it, so the mapping is the identity -- no
      mileage conversion anywhere. `compare_study_cases.fleet()` compares every
      battery parameter across the two CSVs and `profiles()` compares the two
      copies of each availability/trip series; either raises rather than
      reporting a difference that is about the inputs. The `ramp_up`/`ramp_down`
      columns on the building's components equal its max power, so they never
      bind -- which is what makes the EV unit's refusal to model ramps
      equivalent rather than merely different.
- [x] **The binary is there; the fee asymmetry is real and now demonstrated.**
      `dst_components.ElectricVehicle` carries the same non-simultaneity binary
      (`status`, with `charge <= max_power_charge * status` and
      `discharge <= max_power_discharge * (1 - status)`), and no solved schedule
      charges and discharges a vehicle at once (0 of 570 vehicle-steps). The fee
      is a genuine defect: `variable_cost[t] == total_power_input[t] *
      electricity_price[t]` with the fee folded in by `price_plus_grid_fee` is
      symmetric, and `building_tou_tariff_prosumer` plans export in 32 hours,
      23 of them at 280 EUR/MWh, with 111.58 EUR of fee credited to 0.571 MWh
      of export. `is_prosumer: No` masks it; it does not fix it.
- [x] **Horizons aligned.** The CSV's `look_ahead_horizon` went from 48h to the
      scenario's 27h; `commit_horizon` and `rolling_step` were already 1h, which
      is how `EVPortfolioStrategy` replans.
- [x] **The connection limit and capacity charge have no Building equivalent,
      and none was invented.** `ev_import_limit_mw`, `ev_export_limit_mw`,
      `ev_peak_price` and `ev_observed_peak_mw` are portfolio-strategy
      parameters. The building's only connection constraint is the binary
      `is_prosumer` (`total_power_input >= 0`) and it has no peak term at all,
      so `constrained`, `peak_price` and `capacity_charge` have no counterpart.
      Giving them one means adding constraints to `Building`, not configuring it.
- [x] **Reported on the same metrics.** Total cost, connection peak, overload
      hours, unserved driving energy and solve time, all in `summary.csv`.
      Unserved driving is measured by replaying *both* paths' schedules through
      one transcription of `ElectricVehicleUnit._transition`
      (`compare_study_cases.replay`), so it means the same thing on both sides
      rather than being taken from the model under test. It is 0.0 MWh
      everywhere: neither path ever failed a trip, and the building solved every
      window (0 infeasible), so its hard driving constraint never bit.

Found while doing it, all three in the Building path:

- [x] **Numerical noise in the carried SOC killed the unit** — fixed in
      `dsm_load_shift._update_init_states`. A solver returning
      `1.0000000000000007` for a full battery had that handed back as the next
      window's `initial_soc`, where `GenericStorage` rejects anything above 1.0.
      The exception was raised inside the bidding strategy, caught by the agent
      scheduler and logged, so the run *completed* with the building not bidding
      for its last 92 hours and the CSV output looking like a unit that chose to
      do nothing. The carried state is now clamped to the bounds it came from.
      Only reachable when the building actually fills a battery to `max_soc`,
      which is why the non-prosumer cases never hit it.
- [x] **`compare_study_cases.py` refuses to summarise a run that logged an
      error.** The above produced a complete, plausible, entirely wrong summary
      row before anyone noticed. The harness now watches the root logger -- an
      exception out of a bidding strategy is reported under the agent
      framework's logger, not `assume`'s -- and raises instead of reporting.
- [ ] **`DsmEnergyOptimizationStrategy` bids exports as supply at the price
      cap.** It prices every order at 3000 EUR/MWh whatever the sign of the
      volume, so an exporting building offers supply at the cap and never
      clears: 0 of 32 export offers accepted in either prosumer case. The unit's
      internal state still advances along the plan it did not get, so schedule
      and dispatch diverge by up to 61 kW. Until this is fixed no bidirectional
      Building case is meaningful, and the V2G half of this comparison cannot be
      run. `EVPortfolioStrategy` avoids it by bidding `minimum_bid_price` when
      discharging.
- [x] **`assume/world.py:172` said `import nest_asyncio2` again.** Listed as
      fixed under "Incidental" below, but the typo was back on disk and broke
      `World()` construction, so nothing could run at all. Fixed again. Worth
      checking why it returned before trusting that entry.

**Can the match be improved? No -- it is already at the solver-noise floor.**
`no_tariff_no_export_drain` and `tou_tariff_no_export_drain` set
`ev_terminal_soc: 0.2` (`min_soc`), i.e. the EV portfolio asked to value
end-of-window energy the way the Building MILP already does. That one change
drops the cost gap from 1.58 to 0.15 EUR without a tariff and from 4.69 to
0.15 EUR with one, matches terminal energy to four decimals and the fee bill to
0.0007 EUR. The control -- `tou_tariff` against `tou_tariff_no_export`, the same
optimiser with a config change that provably does not bind -- still scatters by
3.76 kW mean / 38.2 kW max at 0.803 correlation, and the matched pairs sit at or
inside that. Fixing the two open defects would not tighten any of this: both are
masked by `is_prosumer: No`. See EXPERIMENTS.md, "Could the match be better?".

Left open deliberately:

- [ ] Compare the two under V2G. Blocked on the export-bid pricing above; the
      `*_prosumer` cases are in `config.yaml` and ready for it.
- [ ] **Give `dst_components.ElectricVehicle` a terminal-SOC term.** The
      ablation above shows this is the whole remaining difference between the
      two optimisers, and it is a defect in its own right rather than a
      comparability nuisance: over a rolling run the building empties the fleet
      to `min_soc` and holds it there for 140 of 576 vehicle-hours, ending every
      run flat with no reserve for an unplanned trip.
      `cost_at_equal_terminal_energy_eur` prices it after the fact; a building
      used for longer than a window wants the term itself.

### 3. Smaller leftovers

- [ ] Configurable peak billing period. `_realised_peak` treats the whole
      simulation as one billing period, which matches `experiments.yaml`
      ("peak rates cover this whole run") but not a real monthly capacity charge.
      Add `ev_peak_billing_period` (e.g. `"30d"`) and reset the floor per period.
- [ ] `_connection_fees` reads the announced fee from the first vehicle that has
      one, on the grounds that a `pay_as_clear` tariff is uniform behind one
      connection. Validate that assumption if aggregators ever span nodes.
- [ ] `energy_at` replays the full index on every call, so `calculate_bids` is
      O(units x steps) per opening and O(steps²) over a run. Fine at 6-60 vehicles
      and 96 steps; cache the replay if annual runs are wanted.
- [ ] Price-taking bids (`maximum_bid_price` when charging, `minimum_bid_price`
      when discharging) commit the fleet regardless of how far the clearing price
      lands from the forecast. Acceptable for a price-taker study; revisit if the
      fleet ever becomes large enough to move the price.

---

## Behaviour changes to be aware of

- An impossible trip no longer raises. Check
  `outputs["ev_planned_unmet_energy"]` and `outputs["unmet_driving_energy"]`
  instead of relying on an exception. `run_experiments.py` still asserts on
  terminal energy and will raise there if a slack ever activates.
- Setting `min_power_charge`, `min_power_discharge` or any `ramp_*` on an
  `ElectricVehicleUnit` now raises instead of being ignored.
- Unknown `ev_*` keys in `bidding_strategy_params` now raise.
- `run_experiments.py` capacity/peak figures change: they are now billed on the
  connection rather than on the vehicles alone, so previously-generated outputs in
  `examples/outputs/lv_ev_experiments/` are stale and should be regenerated.
