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

| Agent | Unit type | Markets |
|---|---|---|
| `aggregator` | `building`, containing only `electric_vehicle_*` components | bids `energy` on `EOM`, announces its withdrawal on `GridTariff` |
| `DSO` | `powerplant` (no physics; the tariff lives in its strategy) | sole supplier on `GridTariff` |
| system units | conventional + renewable powerplants, inflexible demand | `EOM` |

The Building is used purely as a charging hub: no heat pump, no boiler, no PV,
so `total_power_input` is exactly the fleet's net withdrawal at the LV node.

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

The tariff is published a few hours ahead, but the aggregator optimises over a
long look-ahead window. `price_plus_grid_fee` builds a full-horizon fee series:
announced hours carry the published fee, hours beyond it carry a naive
persistence forecast (the most recently published value).

With `count: 1` the announced prefix is about three hours, so **essentially the
whole planning window carries a single repeated value.** Dumping the fee series
the Building actually optimises against, per round:

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

Two constraints follow:

- `look_ahead_horizon >= first_delivery + count`, otherwise the far products are
  announced against a plan that does not exist yet. The unit csv uses `48h`.
- Anything beyond the announced range still persists the last published value.
  To have the *whole* window announced, `count >= look_ahead - first_delivery`.

With `opening_frequency: 1h` and `count: 24` each delivery hour is announced 24
times and the fee is overwritten each round (`accepted_price` is written by
assignment). That is a rolling announcement and is self-consistent; use
`opening_frequency: 24h` instead for a true once-a-day announcement.

## Study cases

| case | DSO bids | what it is for |
|---|---|---|
| `no_tariff` | — | reference |
| `flat_tariff` | one unlimited block at 30 EUR/MWh | plumbing only. A fee that is constant over time adds the same amount to charging cost and discharging revenue in every hour, so total energy and EOM cost must be unchanged against `no_tariff`. The hour-by-hour schedule may still differ where the price forecast has ties. |
| `tou_tariff` | 30 EUR/MWh, +250 overnight, `count: 1` | the first case that can shift load - but myopically, see above |
| `tou_tariff_day_ahead` | same fee, `count: 24` | the same fee announced across the planning horizon |
| `capacity_tariff` | 0.012 MW at 30 EUR/MWh, the rest at 430 | the fee becomes a function of the announced aggregate withdrawal |

Rebuild the inputs with:

```bash
python examples/inputs/example_lv_tariff/generate_inputs.py
```

`examples/examples.py` registers the cases as `lv_tariff_none`,
`lv_tariff_flat`, `lv_tariff_tou` and `lv_tariff_capacity`; set `example` to one
of them and run `python examples/examples.py`.

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

Measured hour-by-hour drift is ~180 % of realised energy in every case,
`count: 24` included, and it is almost entirely misallocation across hours
rather than a level error: the LP re-picks among near-tied hours each round.
This is the one-shot fixed point the design accepts in exchange for having no
iteration loop. The resolutions are on the billing side, not the announcement
horizon - bill on the announcement so truthful announcing has value, or add a
deviation penalty. Report the announcement-versus-realisation gap as a KPI in
its own right.

Note that `market_orders` carries no submission timestamp, so with `count > 1`
the announcement made closest to delivery has to be recovered from row order.

The Building's EV components carry non-simultaneity binaries, so this scenario
is a MILP and cannot produce exact shadow prices of a transformer constraint.
That needs a convex `ElectricVehicleUnit`.
