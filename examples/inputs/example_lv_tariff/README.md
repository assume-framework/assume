<!--
SPDX-FileCopyrightText: ASSUME Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# `example_lv_tariff` — a grid fee published on a market ahead of the EOM

A first prototype of grid-sensitive control of an EV fleet behind one LV node.
It exercises the full signal path — DSO publishes a fee → the fee reaches the
aggregator → the fee enters the aggregator's optimisation objective → the
schedule changes — without adding a market mechanism.

## Who is who

| Agent | Units | Markets |
|---|---|---|
| `aggregator_operator` | six independent `electric_vehicle` units | portfolio bids on `EOM` and `GridTariff` |
| `DSO` | tariff-publishing powerplant | `GridTariff` |
| system operators | generation and inflexible demand | `EOM` |

`ElectricVehicleUnit` extends Storage with binary plug availability and battery-side
trip consumption. It has no Building or DSM optimisation model. The existing
`UnitsOperator` runs `EVPortfolioStrategy` (`units_operator_ev`); no new agent type
or market protocol is needed. Orders retain each EV's id and node for settlement.
Unit strategies in the CSV provide market registration and standard reward hooks;
scheduling is performed by the operator's portfolio strategy.

## EV inputs and scheduling

`electric_vehicle_units.csv` contains one row per EV. Power is MW (negative for
charging, positive for export), capacity is MWh, and SOC is a fraction. Setting
`max_power_discharge: 0` disables V2G. Forecast columns are
`EV_0_availability_profile` and `EV_0_trip_energy_consumption`, etc. Driving
consumption is MWh **per timestep**, must be nonnegative, and occurs only while
unplugged. When converting power profiles to these inputs, multiply by the step
length in hours. The generator preserves the old fleet's trips and availability.
The legacy `residential_dsm_units.csv` is disabled through YAML and retained for
reference.

Operator configuration is in `unit_operators.csv`. Optional settings go in each
case's `bidding_strategy_params`:

```yaml
bidding_strategy_params:
  ev_horizon_mode: rolling_horizon  # or perfect_foresight
  ev_look_ahead_horizon: 27h        # the strategy default is 48h; every study
                                    # case in this scenario overrides it to 27h
                                    # so the announced fee covers the window
  ev_terminal_soc: 0.6
```

All study cases share one `&ev_defaults` block (`ev_look_ahead_horizon`,
`ev_background_load_mw`) merged in with `<<: *ev_defaults`, so the fleet, the
connection and the planning window are one physical setup throughout and the
cases differ only in the mechanism being tested.

Rolling mode replans at every opening. Perfect-foresight mode uses the remaining
simulation horizon; its prices are still the supplied forecasts, so experiments
requiring actual perfect foresight must supply the full price trajectory.
Both modes reconstruct SOC from accepted dispatch, fix already-cleared energy
products, and leave future plans provisional. The terminal SOC lower bound
defaults to each EV's initial SOC. Rolling windows extend beyond the nominal look-ahead when needed to include
return and recharge after a trip. The simulation data boundary remains final. Infeasible optimisation
raises an explicit error. If market rejection makes driving impossible, execution
clips battery power and records the missing trip energy in
`outputs["unmet_driving_energy"]` rather than inventing battery energy.

The small MILP excludes simultaneous charging/discharging, including at negative
prices. It is not a convex model suitable for exact transformer shadow prices.
It supports full products of one simulation timestep and one energy market
(`EOM`). Plans are stored as signed MW in `outputs["ev_planned_energy"]`.

## Congestion interpretation

Congestion in this study is the loading of the single wholesale-to-LV connection
(the transformer equivalent). Its signed import is LV inflexible demand minus
local generation minus the sum of EV energy dispatch; compare its absolute value
with the connection rating if reverse-flow congestion matters. This example
currently uses a tariff/headroom signal, not an explicit two-node network clearing
or a hard line constraint. The tariff can encourage lower loading but does not
guarantee the connection limit. Fleet scheduling is the first step; enforcing and
measuring the connection limit is a separate congestion-management experiment.

## How the fee is set

`GridTariff` is an ordinary `pay_as_clear` market with `product_type: grid_fee`.
The DSO submits a stepwise supply curve; the aggregator submits a demand bid at
the price cap. The clearing price *is* the tariff, and `pay_as_clear` writes
`accepted_price` on accepted and rejected orders alike, so it reaches every
participant through the standard clearing path and lands in
`outputs["grid_fee_accepted_price"]`.

`product_type` must not be `energy` — `set_dispatch_plan` accumulates
`outputs[product_type]`, so an announced volume would otherwise be added to the
unit's energy dispatch and EOM cashflow.

## Timing

```
market round T          EOM opens  -> delivery T+1, clears at T+1
                  GridTariff opens -> delivery T+3, clears at T+1
```

The fee for a delivery hour is therefore cleared two market rounds before the
EOM round that dispatches it.

## The horizon mismatch, and why `count` matters

The numerical observations below are from the former Building prototype; they
are historical reference results, not validation results for the EV operator.

The tariff is published a few hours ahead, but the aggregator optimises over a
long look-ahead window. `price_plus_grid_fee` builds a full-horizon fee series:
announced hours carry the published fee, hours beyond it carry a naive
persistence forecast (the most recently published value).

With `count: 1` the announced prefix is about three hours, so **essentially the
whole planning window carries a single repeated value.** Dumping the fee series
the former Building implementation optimised against, per round:

```
count: 1     round 40  ->  look-ahead [40:64] distinct fees: [30.0]
             round 50  ->  look-ahead [50:74] distinct fees: [280.0]

count: 24    round 40  ->  fee[38:48] =  30  30  30  30  30  30  30  30  30 280
             round 50  ->  fee[48:58] = 280 280 280 280 280 280 280  30  30  30
```

Under `count: 1` the fee cannot reallocate anything *within* a round - it is
flat there. What moves the dispatch is the fee *level* changing between rounds,
which is a myopic controller artefact, not a response to a known tariff. Under
`count: 24` the real time-of-use shape is inside every window and the response
is foresighted: the same avoidance of the penalised hours, but at roughly half
the peak and at no extra EOM cost (see the table below).

**Every tariff study case therefore announces `count: 24`**, the way a day-ahead
market publishes a whole day at once, and the EV operator plans over
`ev_look_ahead_horizon: 27h` — exactly `first_delivery` (3h) + `count` (24h), so
the announcement covers the planning window end to end and no hour of it falls
back on the persisted value.

Two constraints drive that arithmetic:

- `look_ahead_horizon >= first_delivery + count`, otherwise the far products are
  announced against a plan that does not exist yet.
- Anything beyond the announced range persists the last published value, so for
  the *whole* window to be announced you also need
  `count >= look_ahead - first_delivery`.

Both hold with equality at 27h / 3h / 24h. A 48h window with `count: 24` would
satisfy the first and fail the second — the last 21 hours would carry a
persisted value, which is the same defect as a short `count` moved to the other
end of the window.

Measured on the EV operator over the four-day run, same 30/280 EUR/MWh fee:

| case | announcement | import in penalised hours | grid fee |
|---|---|---|---|
| `no_tariff` | — (reference) | 38.2 % | — |
| `tou_tariff` | 24h, republished hourly | **11.5 %** | **18.61 EUR** |
| `tou_tariff_daily_announcement` | 24h, published once a day | 25.2 % | 60.72 EUR |
| `tou_tariff_myopic` | 1h, republished hourly | 32.3 % | 53.61 EUR |

Read any `count: 1` result as a controller artefact rather than a tariff design.

### Why hourly republication is the default

`opening_frequency` is an ablation in its own right, and the result is not
obvious. With `count: 24` republished **hourly**, the aggregator always holds a
fee published for the next 24 hours, so its 27h window is always covered. With
the same `count: 24` published **once a day**, the announced depth *decays
through the day*: 26h just after the auction, about 3h just before the next one.
The window is fully covered only immediately after an auction, and for most of
the day the aggregator is again optimising against a persisted tail.

That is why the daily case recovers less than half the response of the hourly one
and pays over three times the fee. Covering a 27h window at all times from a
once-a-day auction needs `count` near 48, not 24. Hourly is the default because
it is the setting that keeps the fee current with the plan; the once-a-day case
is kept as `tou_tariff_daily_announcement` so the cost of not doing so is
measurable rather than assumed.

Note this is a different question from the announcement-versus-realisation drift
below: here the fee *schedule* is fully known and static, and the only thing
varying is how much of it the aggregator can see at the moment it plans.

## Comparable EV experiment suite

See [EXPERIMENTS.md](EXPERIMENTS.md) for independent/aggregated EVs, hard
connection limits, static/TOU/dynamic fees, peak/capacity pricing and ex-post
billing, with one shared fleet and a combined output report. Run from the repo
root in the `assume` conda environment:

```bash
python -m examples.inputs.example_lv_tariff.run_experiments
```

This is a controlled fixed-forecast dispatch benchmark. The study cases below
remain the separate hourly market-announcement simulations.

## Study cases

| case | DSO bids | what it is for |
|---|---|---|
| `no_tariff` | — | reference |
| `flat_tariff` | one unlimited block at 30 EUR/MWh | plumbing reference. A flat fee applies to net withdrawal. With lossy V2G, a flat fee can also change arbitrage and total throughput; schedule invariance is not a general correctness test. |
| `tou_tariff` | 30 EUR/MWh, +250 overnight | the first case that can shift load, and it does: 11.5 % of import in the penalised hours against 38.2 % unpriced |
| `capacity_tariff` | 0.012 MW at 30 EUR/MWh, the rest at 430 | the fee becomes a function of the announced aggregate withdrawal |
| `tou_tariff_myopic` | same fee as `tou_tariff`, but `count: 1` | the deliberate contrast: the announcement is shorter than the planning horizon, so the response is a controller artefact |
| `tou_tariff_daily_announcement` | same fee and `count: 24`, but `opening_frequency: 24h` | ablation on announcement *frequency*: published once a day instead of republished hourly |
| `constrained` | — | the connection is physically limited: 0.025 MW rating, 0.005 MW of other load behind the same meter, so 0.020 MW is left for the fleet |
| `peak_price` | — | 1000 EUR/MW on the single highest hour of the run, no free allowance, so the profile flattens |
| `capacity_charge` | — | 2000 EUR/MW, but only above a 0.02 MW contracted capacity, so the profile sits flat just under the contract |
| `independent` | — | the same six EVs, one owner each: no aggregation. The reference the aggregation benefit is measured against |

The last four have no DSO and no `GridTariff` market — the limit or the charge
is a property of the connection, not something cleared on a market, so it is set
directly in `bidding_strategy_params`:

```yaml
bidding_strategy_params:
  ev_import_limit_mw: 0.025       # the connection's own rating
  ev_export_limit_mw: 0.025
  ev_background_load_mw: 0.005    # other load behind the same meter
  ev_peak_price: 1000.0           # EUR/MW on the peak hour of the run
  ev_observed_peak_mw: 0.02       # free allowance: only the rise above it is charged
```

`ev_background_load_mw` is not an EV and is never dispatched, but it occupies the
connection, so the limits, the grid fee and the capacity charge all see it. Quote
the connection's real rating in the limits and put the other load here rather
than netting it off by hand.

**Every study case in this scenario declares the same 0.005 MW**, so the
connection is one physical thing throughout and peaks are comparable across
cases. It is not only a reporting quantity: in the tariff cases the fee falls on
net *connection* import, so the background load makes the fee bite at the margin
during hours the fleet would otherwise export itself down to zero net import,
and the DSO bills the whole connection rather than the vehicles alone. In
`no_tariff` and `independent`, where nothing prices or limits the connection, it
provably cannot move a single MWh.

`ev_peak_price` steers the schedule through the optimiser's objective, but no
market settles it — the bill is computed in post-processing by
`compare_study_cases.py`. The rate is EUR/MW **for these four days**, not an
annual EUR/kW tariff.

### Comparing the study cases

`compare_study_cases.py` runs the study cases through `World.run()` and reduces
the output to the same `summary.csv` / `connection.csv` shape `run_experiments.py`
produces, so the two can be laid side by side:

```bash
python examples/inputs/example_lv_tariff/compare_study_cases.py
python examples/inputs/example_lv_tariff/compare_study_cases.py \
    --cases no_tariff independent constrained --reuse
```

It also reports `ex_post_peak`: the peak charge applied to `no_tariff` dispatch
after the fact. That is a settlement of an existing case rather than a case of
its own — being unanticipated, it cannot have changed anybody's behaviour.

Rebuild the inputs with:

```bash
python examples/inputs/example_lv_tariff/generate_inputs.py
```

`examples/examples.py` registers the cases as `lv_tariff_none`,
`lv_tariff_flat`, `lv_tariff_tou`, `lv_tariff_tou_myopic`, `lv_tariff_tou_daily`,
`lv_tariff_capacity`, `lv_tariff_constrained`, `lv_tariff_peak_price`,
`lv_tariff_capacity_charge` and `lv_tariff_independent`; set `example` to one of
them and run `python examples/examples.py`.

## Where the numbers are

- The canonical tariff series is the `GridTariff` rows of `market_orders`.
  `grid_fee_accepted_price` matches none of the keys exported by
  `get_actual_dispatch`, so it never reaches `unit_dispatch`.
- `grid_fee_cashflow` **is** exported to `unit_dispatch` (it contains
  `cashflow`). It is the bill for what the aggregator *announced*. The bill for
  what it actually consumed has to be computed in post-processing as realised
  energy times fee; report both, and the gap between them.
- Do not sum `grid_fee_cashflow` and the EOM cashflow into one welfare number
  without care — that counts the fee twice.

## Known limitation of this prototype

**Announcement drift is not fixed by widening `count`.** These are two separate
problems and it is easy to conflate them:

- *What fee does the optimiser see over its horizon?* - solved by `count: 24`.
- *Does the announced quantity match the dispatched one?* - unaffected. The
  aggregator announces its provisional plan for `T+3`, receives a fee priced on
  it, re-optimises against the fee, and dispatches something else.

In the former Building implementation, measured hour-by-hour drift is ~180 % of realised energy in every case,
`count: 24` included, and it is almost entirely misallocation across hours
rather than a level error: the LP re-picks among near-tied hours each round.
This is the one-shot fixed point the design accepts in exchange for having no
iteration loop. The resolutions are on the billing side, not the announcement
horizon - bill on the announcement so truthful announcing has value, or add a
deviation penalty. Report the announcement-versus-realisation gap as a KPI in
its own right.

Note that `market_orders` carries no submission timestamp, so with `count > 1`
the announcement made closest to delivery has to be recovered from row order.

The independent EV scheduler also uses non-simultaneity binaries. Exact shadow
prices would require a justified convex formulation or a separate pricing solve.
