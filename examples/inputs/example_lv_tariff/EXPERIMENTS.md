# Comparable EV experiments

The experiment suite reuses `ElectricVehicleUnit`, `EVPortfolioStrategy`, the
scenario input generator, and the EOM forecast calculated by the normal CSV
loader. It solves charging and V2G dispatch over the full four-day horizon.
This is a **fixed-forecast, price-taking optimisation benchmark**, separate from
the hourly market-clearing simulations in `config.yaml`. Dispatch is assumed
fully accepted and settled at the supplied forecast prices. There is no forecast
error, endogenous energy-price impact, or rolling forecast update in this suite.

## Run

From the repository root:

```powershell
conda activate assume
python -m examples.inputs.example_lv_tariff.run_experiments
python -m examples.inputs.example_lv_tariff.run_experiments --n-evs 60 --n-aggregators 3
```

Running the file by path works too; it puts the repository root on `sys.path`
itself, so no `PYTHONPATH=.` is needed:

```powershell
python examples/inputs/example_lv_tariff/run_experiments.py
```

Configure all cases together in `experiments.yaml`. `--config PATH` selects a
different configuration. `--output-dir examples/outputs/lv_ev_experiments/my_run`
chooses a named run. An existing run directory is refused, so results cannot
silently mix. Defaults put every case in the same timestamped subfolder under
`examples/outputs/lv_ev_experiments/`.

The fleet is generated once per run. Increasing `n_evs` preserves existing EV
profiles with the same seed and appends additional ones. `n_aggregators` splits
the same fleet round-robin into nonempty portfolios for every coordinated case.
The independent case always solves one portfolio per EV. `v2g: false` disables
discharging for all cases. Battery sizes, charger ratings, trip consumption and
dates are central constants in `generate_inputs.py`; change those there and,
for date changes, keep the baseline dates in `config.yaml` aligned.

## Foresight, and comparing with the study cases

`horizon_mode` decides what the optimiser is allowed to see. `perfect_foresight`
solves each portfolio once over the whole run; `rolling_horizon` replans every
`rolling_step` over a `look_ahead_horizon` window and keeps only the steps up to
the next replan, which is how the `config.yaml` study cases plan. Same fleet,
same prices, same fees — only the foresight differs, so the gap between the two
modes is the cost of not knowing the future.

```powershell
python examples/inputs/example_lv_tariff/run_experiments.py --horizon-mode rolling_horizon
python examples/inputs/example_lv_tariff/run_experiments.py --horizon-mode rolling_horizon --look-ahead-horizon 24h --rolling-step 1h
```

Set them in `experiments.yaml` to make a run reproducible; the chosen mode is
recorded in `manifest.json`. Configs written before the switch existed default
to `perfect_foresight`, which is the only behaviour they could have had.

Four of these cases now also exist as ordinary ASSUME study cases in
`config.yaml`, with a real EOM clearing and real agent messaging instead of a
direct call to `optimize()`:

| harness case | study case | what differs |
|---|---|---|
| `independent` | `independent` | one unit operator per EV in both |
| `constrained` | `constrained` | the study case names the connection rating and the background load separately rather than pre-allocating shares |
| `peak` | `peak_price` | same 1000 EUR/MW on the run's peak hour |
| `capacity` | `capacity_charge` | same 2000 EUR/MW above a 0.02 MW contract |

Run the study-case side with `compare_study_cases.py`, which writes the same
`summary.csv` / `connection.csv` shape. Under `--horizon-mode rolling_horizon`
the two agree on the physical quantities — the connection peak of `peak_price`
is 0.0088 MW either way — while the euro figures still differ, because the
harness settles at the supplied forecast price and the simulation settles at the
price the EOM actually cleared.

### `dynamic` and `ex_post_peak`

Neither becomes a study case, for the same underlying reason: both are defined
against a frozen ex-ante baseline the harness computes for itself, which has no
counterpart once prices are endogenous.

- `dynamic` sets the fee from a no-fee baseline the DSO solves once and freezes.
  In a live simulation the DSO does not need to guess: `capacity_tariff` already
  prices the fee off the withdrawal the aggregators actually announced, which is
  the endogenous version of the same idea and is a market mechanism rather than
  a frozen forecast. Use `capacity_tariff` as the live counterpart.
- `ex_post_peak` is unanticipated by construction, so it cannot change
  behaviour. That makes it a *settlement* of `no_tariff` dispatch, not a case
  with a scenario of its own, and `compare_study_cases.py` reports it that way —
  the `no_tariff` run billed at the `peak_price` rate.

The original scenario generator also accepts `--n-evs`, `--seed` and
`--output-dir`, so fleet growth can be used in the existing market simulations.
The benchmark copies its generated inputs into each run and leaves the original
scenario CSVs untouched. Its portfolio assignments are in `ownership.csv`;
the copied baseline scenario CSVs describe forecast construction, not the
ownership in every experiment.

## Cases and billing

| Case | Optimisation and settlement |
|---|---|
| `independent` | Each EV solves separately with only its own battery, availability, trips and the common EOM price forecast. |
| `aggregated` | Joint optimisation within each aggregator; no common power limit or fee. |
| `constrained` | Same as aggregated, subject to import and export limits including background load. |
| `static` | Flat EUR/MWh fee on positive net aggregator import. |
| `tou` | Static fee plus an adder during configured hours; the complete schedule is known ex ante. |
| `dynamic` | Static fee plus an adder proportional to expected connection imports, normalised by their maximum. Expected load is the no-fee aggregated forecast-optimal dispatch plus background. This schedule is frozen before the tariff response. |
| `peak` | Known EUR/MW charge on maximum positive net aggregator import across the whole run, included in optimisation and billed afterwards. |
| `capacity` | Known EUR/MW charge on the positive excess of that peak above an allocated contracted capacity. No separate fixed subscription charge. |
| `ex_post_peak` | An **unanticipated** peak bill applied to the exact aggregated baseline dispatch. This cannot alter behaviour. For an anticipated charge settled ex post, use `peak`. |

Energy settlement is signed net EV import times the common EOM price times
step duration. Positive costs are expenditure; negative costs are net revenue.
Volumetric grid fees apply to positive net import **at each aggregator meter**;
exports earn energy revenue but receive no grid-fee credit. Charging by one EV
can net against V2G from another in the same portfolio. Separate aggregators
are billed separately. Background consumption contributes to connection loading
and dynamic fee design, and — since the fee and the capacity charge are billed on
the whole connection, which is what the DSO meters — it is also part of the
volumetric fee and of the aggregator peak. It is *not* part of the EV energy
bill, which settles only what the vehicles drew on the EOM. The `config.yaml`
study cases follow the same convention through `ev_background_load_mw`.

Peak rates are EUR/MW **for the entire four-day experiment**, not annual rates.
Do not compare them with annual EUR/kW tariffs without converting both unit and
billing period. Static fees need not reduce peaks; even dynamic/TOU schedules
can create new coincident peaks. The output measures the response rather than
assuming every fee works.

With multiple aggregators, connection limits, background load and contracted
capacity are allocated by each aggregator's fraction of fleet size. Their
allocated limits sum to the physical connection limit. This is a conservative
static capacity allocation; aggregators do not trade unused headroom. Increasing
fleet size does **not** automatically increase the physical limit. Severe
constraints can make trips and terminal energy infeasible; the solve then raises
an error rather than silently dropping trips or relaxing the limit.

## How each mechanism works

Sign convention: in the plan and in `ev_dispatch.csv`, `power > 0` is export
(V2G to grid) and `power < 0` is import (charging). `net = -sum(power)` is
positive when the portfolio imports. Fees and peak charges act on
`imports = net.clip(lower=0)`, i.e. positive net import only, with no export
credit (`run_experiments.py` `solve_groups`/operator loop;
`ev_aggregator.py` `optimize`).

`static` is not dispatch-neutral. A flat EUR/MWh adder would leave the charging
schedule unchanged only if it were a symmetric price shift. It is not: it is
levied on positive net import with no export credit, so it does not move the
timing of charging between hours (the rate is constant), but it removes
grid-directed V2G arbitrage whose price spread is below roughly the fee plus
round-trip losses. In the sample results the export peak collapses to zero in
`static`, `tou` and `dynamic`, and the positive `energy_cost_delta_vs_aggregated`
is the foregone arbitrage revenue, reported separately from `grid_fee_eur`.
V2G between EVs in the same portfolio stays fee-free because the fee is on
portfolio net import.

`tou` schedule, from `tou_hours` and the two rate keys: hours listed in
`tou_hours` (`[23, 0, 1, 2, 3, 4, 5, 6]`, i.e. 23:00-06:59) are charged
`static_fee_eur_per_mwh + tou_adder_eur_per_mwh`; all other hours are charged
`static_fee_eur_per_mwh`. With the default 30 and 250 that is 280 EUR/MWh
overnight and 30 EUR/MWh otherwise. Deterministic and known ex ante. It can
create a new peak in the now-cheaper daytime shoulder.

`dynamic` is a single forecast, not an iteration. The DSO solves the no-fee
aggregated baseline once, takes expected connection import as that baseline net
import plus background clipped at zero, and sets
`fee(t) = static_fee + dynamic_adder * expected(t) / max(expected)`. That
schedule is frozen to `signals.csv`; each aggregator then re-optimises against
it once. The DSO does not re-forecast after the response, so load shifted into a
formerly cheap hour can produce a new peak there.

`peak` has no target threshold. It is a linear price on the single highest
net-import hour over the whole run. In `optimize`, `model.peak >= imports[k]`
for every step makes `peak` the horizon maximum, and the objective carries
`ev_peak_price * (peak - observed_peak)` with `observed_peak = 0`. Every MW on
the tallest hour costs `peak_price_eur_per_mw` for the period, so the aggregator
flattens its profile even into pricier hours. Billed afterwards as
`peak_import_mw * peak_price_eur_per_mw` on the realised plan; anticipated and
ex-post values agree under perfect foresight. `ex_post_peak` applies the same
rate to the frozen baseline the aggregator never sees, so it is a pure transfer.

`capacity` is `peak` with a free allowance. Same machinery, but the rate is
`capacity_excess_price_eur_per_mw` and `observed_peak` is set to the allocated
contracted capacity (`contracted_capacity_mw * fleet share`), which becomes the
lower bound of `model.peak`. The objective term is then zero for any peak at or
below the contract, so the aggregator flattens down to the contracted level and
stops; below it the marginal peak MW is free and is spent on cheaper energy
timing. Billed as `max(0, peak - contracted) * capacity_excess_price_eur_per_mw`.
The per-period charge for the contracted band itself is not modelled, only the
exceedance penalty.

Multiple aggregators do not co-optimise. `solve_groups` splits the fleet
round-robin (`units[i::n_aggregators]`) so each portfolio gets a similar profile
mix, then solves one independent model per portfolio with no shared constraint
or communication. `constrained` stays within the physical connection because
each aggregator is given a hard sub-limit equal to its fleet-size share of the
connection and of the background load; since the shares sum to one, the
per-aggregator limits sum exactly to the physical limit, so independent
compliance guarantees aggregate compliance. The cost is that headroom is not
tradable: an idle aggregator's unused share cannot be lent to a busy one, so the
decentralised `constrained` solution is weakly worse than the single-optimiser
`constrained` and the gap widens as the limit binds harder. `peak` and
`capacity` receive their price signal but no hard limit, so their connection
peaks can still exceed `connection_import_limit_mw`; only `constrained` enforces
it.

## The Building comparison

`Building` can hold the same six vehicles as `dst_components.ElectricVehicle`
components and schedule them with one MILP over the whole building, which is the
natural alternative to a portfolio MILP over six independent
`ElectricVehicleUnit`s. `residential_dsm_units.csv` defines exactly that
building; four study cases pair it against the EV path:

| case | optimiser | connection | tariff |
|---|---|---|---|
| `no_tariff` | portfolio MILP | export free | none |
| `no_tariff_no_export` | portfolio MILP | `ev_export_limit_mw: 0` | none |
| `building_no_tariff` | Building MILP | `is_prosumer: No` | none |
| `tou_tariff` | portfolio MILP | export free | ToU |
| `tou_tariff_no_export` | portfolio MILP | `ev_export_limit_mw: 0` | ToU |
| `building_tou_tariff` | Building MILP | `is_prosumer: No` | ToU |

Run them together:

```powershell
python examples/inputs/example_lv_tariff/compare_study_cases.py --cases \
    no_tariff no_tariff_no_export building_no_tariff \
    tou_tariff tou_tariff_no_export building_tou_tariff
```

### Making the two comparable

The vehicles are the same by construction and `compare_study_cases.py` refuses
to run if they stop being: `fleet()` compares every battery parameter in
`electric_vehicle_units.csv` against the building's components, and `profiles()`
compares the two copies of each availability and trip-energy series that
`generate_inputs.py` writes. `look_ahead_horizon` was aligned to the scenario's
27h on both sides.

Two differences could not be configured away and are the reason the
`*_no_export` cases exist:

- **Export.** `is_prosumer: No` puts `total_power_input[t] >= 0` on the
  building, which is a hard zero-export limit at the connection.
  `ev_export_limit_mw: 0.0` is the same constraint on the portfolio. Without it
  the aggregator is simply allowed to do something the building cannot, and the
  comparison measures the permission rather than the optimisation. It is worth
  1.67 EUR of energy cost over four days on its own.
- **Terminal SOC.** `EVPortfolioStrategy` targets each vehicle's initial SOC at
  the end of every planning window. The Building MILP has no terminal-SOC term,
  so energy left in a battery is worth exactly zero to it and every window ends
  by emptying the fleet: the building finishes all four days at a mean SOC of
  0.200, which is `min_soc` to three decimals, against 0.458 for the portfolio.
  That is 0.072 MWh of battery sold rather than bought, and it is most of the
  building's apparent cost advantage. `cost_at_equal_terminal_energy_eur` prices
  it back in at what each fleet actually paid per MWh charged.

Service is measured the same way on both sides rather than taken from either
model: `replay()` puts each vehicle's schedule through a transcription of
`ElectricVehicleUnit._transition`. For the building the per-vehicle schedule has
to be captured while it runs, since it exports only an aggregate `power`; the
harness records the committed step of every rolling window as it is solved.

### What the run shows

Four days, six vehicles, 0.005 MW of other load behind the same meter:

| | `no_tariff_no_export` | `building_no_tariff` | `tou_tariff` | `building_tou_tariff` |
|---|---|---|---|---|
| total cost, EUR | 4.15 | 2.57 | 24.18 | 19.48 |
| **at equal terminal energy, EUR** | **4.94** | **4.79** | **23.87** | **23.29** |
| grid fee, EUR | -- | -- | 18.61 | 16.29 |
| connection peak, MW | 0.0356 | 0.0353 | 0.0710 | 0.0695 |
| import in 280 EUR/MWh hours | -- | -- | 0.0 % | 0.0 % |
| mean terminal SOC | 0.458 | 0.200 | 0.629 | 0.229 |
| unserved driving, MWh | 0.0 | 0.0 | 0.0 | 0.0 |
| run wall clock, s | ~9 | ~16 | ~18 | ~19 |

**The two optimisers agree on the answer.** Once export is matched and terminal
energy is priced, total cost differs by 3 % without a tariff and 2.5 % with one,
and the connection peak by under 2 %. Under the time-of-use fee both drive the
connection to *exactly* zero net import in all 29 penalised hours -- the fleet
discharges precisely enough to cover the background load -- so neither pays the
280 EUR/MWh rate at all, and the whole fee bill is the 30 EUR/MWh base on the
remaining hours. The building's smaller bill there is its smaller purchase
(0.621 against 0.701 MWh), which is the energy it does not put back in the
batteries.

**They do not agree on the hourly path to it.** Hour by hour the two connection
profiles correlate at only 0.50 without a tariff and 0.61 with one, and differ
by up to 31 kW and 54 kW respectively -- on a connection that peaks at 35 kW and
70 kW. Aggregates match; the allocation within the day does not. Anything that
prices *when* energy is taken, rather than how much, should not assume these two
paths are interchangeable.

**Solve cost.** The building is roughly 1.7x slower without a tariff (~16 s
against ~9 s of whole-run wall clock, which varies a little between runs) and
level with it under the tariff. `Building` solves a
full MILP every window; `EVPortfolioStrategy` solves the LP relaxation first and
re-solves as a MILP only when the relaxed solution actually violates the
charge/discharge disjunction, which on this scenario it rarely does. Both models
carry the non-simultaneity binary, and neither ever charges and discharges the
same vehicle at once (0 of 570 vehicle-steps).

### Three defects in the Building path this turned up

The V2G half of the comparison cannot be run today. `building_no_tariff_prosumer`
and `building_tou_tariff_prosumer` (`is_prosumer: Yes`) exist to show why.

1. **The grid fee is credited to exports.** `Building.define_constraints` writes
   `variable_cost[t] == total_power_input[t] * electricity_price[t]`, and
   `price_plus_grid_fee` folds the announced fee into `electricity_price`. The
   fee is therefore symmetric. Given the chance, the building takes it:
   `building_tou_tariff_prosumer` plans to export in 32 hours, **23 of them in a
   280 EUR/MWh hour**, and its objective credits 111.58 EUR of grid fee to
   0.571 MWh of export -- a payment no volumetric import tariff makes. This is
   the same defect `EVPortfolioStrategy` was moved off via `grid_fee_free_price`;
   `is_prosumer: No` masks it rather than fixing it, which is why the
   non-prosumer cases above are unaffected.
2. **Exports are bid as supply at the price cap.**
   `DsmEnergyOptimizationStrategy` prices every order at 3000 EUR/MWh whatever
   the sign of the volume, so an exporting building offers supply at the cap.
   None of the 32 export offers cleared, in either prosumer case. The building's
   internal state still advances along the plan it did not get, so its schedule
   and its dispatch diverge -- by up to 61 kW here. Treat every number from the
   prosumer cases as a diagnostic, not a result.
3. **Numerical noise in the carried SOC aborted the unit** (fixed). A solver
   returning `1.0000000000000007` for a full battery had that value handed back
   as the next window's `initial_soc`, where `GenericStorage` rejects anything
   above 1.0. The exception was raised inside the bidding strategy, caught by the
   agent scheduler and logged, so the simulation ran to completion with the
   building simply not bidding for its last 92 hours and the CSV output looking
   like a unit that chose to do nothing. `_update_init_states` now clamps the
   carried state to the bounds it came from. `compare_study_cases.py` now also
   refuses to summarise any run that logged an error, so a silently dead agent
   cannot reach a results table again.

### Could the match be better? The terminal-SOC ablation

It cannot, and fixing the three defects above is not what would do it. Two of
them are masked by `is_prosumer: No` -- a building that cannot export is never
credited the fee and never bids an export at the cap -- so neither can be moving
the matched numbers at all. What fixing them buys is the *ability to run the V2G
comparison*, not a closer one of this comparison.

What closes the gap is the terminal-SOC term. `no_tariff_no_export_drain` and
`tou_tariff_no_export_drain` set `ev_terminal_soc: 0.2`, which is `min_soc`:
the EV portfolio asked to value end-of-window energy the way the Building MILP
already does. Everything else is unchanged.

| | EV, targets initial SOC | EV, drains | Building |
|---|---|---|---|
| no tariff: total cost, EUR | 4.150 | **2.715** | **2.569** |
| no tariff: terminal energy delta, MWh | -0.0255 | **-0.0720** | **-0.0720** |
| ToU: total cost, EUR | 24.176 | **19.629** | **19.483** |
| ToU: grid fee, EUR | 18.615 | **16.2918** | **16.2911** |
| ToU: terminal energy delta, MWh | +0.0053 | **-0.0667** | **-0.0667** |

The cost gap falls from 1.58 EUR to 0.15 EUR without a tariff and from 4.69 EUR
to 0.15 EUR with one; terminal energy matches to four decimals and the fee bill
to 0.0007 EUR. One term accounts for essentially the whole difference.

The remaining 0.15 EUR is not a modelling difference. `tou_tariff` and
`tou_tariff_no_export` are the control: the same optimiser, the same scenario,
and a config change that provably does not bind (the case never exports, so
cost, fee, peak and terminal SOC are identical to four decimals). Their hourly
profiles still diverge, and that is the floor any comparison here can reach:

| | corr | mean diff | max diff | hours >1 kW apart |
|---|---|---|---|---|
| **control** -- same optimiser, non-binding change | **0.803** | **3.76 kW** | **38.2 kW** | **21** |
| no tariff, EV targets initial SOC | 0.497 | 3.22 kW | 30.6 kW | 33 |
| no tariff, EV drains like the building | 0.572 | 2.59 kW | 27.5 kW | 26 |
| ToU, EV targets initial SOC | 0.538 | 5.14 kW | 66.0 kW | 24 |
| ToU, EV drains like the building | **0.761** | **3.33 kW** | **35.1 kW** | **20** |

With terminal SOC matched, both pairs sit at or inside that floor on every
magnitude measure. Six identical vehicles, a flat 30 EUR/MWh fee and a smooth
price curve leave a very flat optimal face, so many distinct schedules cost the
same and the solver -- `appsi_highs` on both paths, so this is not a solver
difference -- picks among them arbitrarily. The earlier "correlate at only 0.50
and 0.61" reading was measuring that scatter, not a disagreement between the
models.

So the finding inverts. The two optimisers agree to within solver noise, and the
Building's missing terminal-SOC target is not a comparability nuisance to be
adjusted around but a defect of its own: over a rolling run it empties the fleet
and holds it at `min_soc` for 140 of 576 vehicle-hours, ending every run flat
with no reserve for an unplanned trip. Adding the term to
`dst_components.ElectricVehicle` is the change worth making; matching the two
paths more closely is not.

### What has no Building equivalent

`constrained`, `peak_price` and `capacity_charge` have no building counterpart
and none is attempted. `ev_import_limit_mw`, `ev_export_limit_mw`,
`ev_peak_price` and `ev_observed_peak_mw` are `EVPortfolioStrategy` parameters;
the building's only connection constraint is the binary `is_prosumer`, and it
has no peak term in its objective at all. Expressing them would mean adding
constraints to `Building`, not configuring it.

## Read results together

- `report.html` contains a comparison table and embedded plots when matplotlib
  is available; `comparison.svg` is the standalone figure.
- `summary.csv` compares energy costs, bills, cost changes against the aggregated
  baseline, connection/fleet peaks, overload hours and overload energy.
- `timeseries.csv` puts all case connection loads, fees and limit exceedances on
  the same hourly timeline; `signals.csv` records EOM prices, background,
  expected loads and the frozen dynamic tariff.
- Each case contains `ev_dispatch.csv` (positive power means export; SOC is
  **end-of-step**) and `operators.csv` with separate energy/volumetric/capacity
  bills, imports, exports and peaks. `ownership.csv` links EVs to portfolios.
- `experiments.yaml` records the resolved settings; `manifest.json` records
  assumptions, the time boundary and SHA256 hashes of generated inputs.

The simulation end is exclusive: March 1 through March 4 comprises 96 steps.
All EVs have the same initial conditions in every case and must end with at
least their initial battery energy. Exported SOC and trips allow energy-balance
checks. Fees are settled once on dispatch; the GridTariff announcement cashflows
from the market examples are not used here.

Under these price-taking, separable objectives, independent and aggregated
minimum energy costs should match. Solver ties can produce different schedules
and peaks despite the same optimal cost. The aggregated case is therefore a
consistency reference, not an assumed source of savings. The constrained
energy-cost delta measures the aggregator's lost arbitrage opportunity for
the same served mobility. Fee-inclusive cost changes measure private bills;
fees are transfers and should not be counted as social resource costs.

## What already existed and what was added

The initial workspace supported individual EV physics, portfolio optimisation,
independent operators through the ordinary UnitsOperator interface, and flat,
TOU and scarcity-step GridTariff announcements. Its `capacity_tariff` is a
volumetric scarcity step, **not** a billing-period peak charge or hard limit.

`EVPortfolioStrategy` now also accepts `ev_import_limit_mw`,
`ev_export_limit_mw`, `ev_peak_price` and `ev_observed_peak_mw`. Its `optimize`
method accepts timestamp-keyed `grid_fees` and `background_load` profiles.
Limits constrain signed portfolio power plus background; import fees and peak
pricing use positive net portfolio import. This is one connection constraint,
not an AC load flow or a model of multiple internal lines, voltage or losses.
Use the binding trafo/line rating for the connection or extend the constraints
for an explicit network.

The new tariff settlement and experiment orchestration are in
`run_experiments.py`. They are not automatically wired into the old GridTariff
market. `grid_fees` must not be combined with an EOM forecast that already
includes those fees. For rolling peak optimisation, a caller must maintain the
observed peak and reset it at billing-period boundaries; the benchmark covers
one complete period so does not require that state. Forecast horizon quality
and imperfect dispatch acceptance remain separate market-simulation experiments.

For a later local capacity market, replace static allocation in `solve_groups`
with cleared capacity entitlements and add capacity payments to operator bills.
For a learning DSO, expose tariff/limit schedules as actions and connection
overload, mobility and private-cost KPIs as observations/reward components.
Keep the baseline, settlement definitions, inputs and output schema for those
comparisons. Neither the capacity market nor a learning DSO is implemented yet.
