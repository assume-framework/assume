# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generate the `p1`..`p7` demand series: a ladder in how often the stage game
sits in its *pivotal* Nash equilibrium.

Why
---
Run 14b found that a shared critic learns the equilibrium it sees often and
leaves the rare one unfitted: on `sb02b` the `bertrand` regime is 67 % of hours
and its `argmax Q1` is a tight line on marginal cost, while `pivotal` is 13 %
and its `argmax Q1` is noise across the whole action range. That is one case at
three seeds, and "rare" and "pivotal" are confounded in it -- the regime that
failed is also the only regime that was rare.

This ladder separates them. Every case is `powerplant_units_02b.csv` -- the same
five 500 MW learners, the same naive fleet, the same fuel prices and market --
and only the demand series differs. If regime frequency is what matters, the
pivotal critic should sharpen monotonically from `p1` to `p7`. If instead the
pivotal equilibrium is intrinsically harder to represent, it should stay noisy
even at 87.5 % of hours.

The transform
-------------
Each series is an **affine compression toward the pivotal band's midpoint**::

    d'(t) = c + alpha * (d(t) - c),   c = (7000 + 7500) / 2 = 7250 MW

with `alpha` solved by bisection so that the pivotal band [7000, 7500] holds
exactly k/8 of the hours of the simulated horizon (March 2019). One parameter,
monotone in `alpha`, and the *shape* of the daily and weekly profile is
preserved exactly -- every hour keeps its rank.

A pure shift would have been preferable on the face of it, since it moves the
distribution across the band edges without touching its spread. It cannot reach
the top of the ladder: the pivotal band is 500 MW wide against a demand sd of
~1000 MW, so no translation puts more than about a fifth of the hours inside it.
Reaching 7/8 *requires* compression.

What the agent actually sees
----------------------------
Compression costs less than it looks like it should, because
`learning_strategies.py:126-133` min-max scales residual load by **that series'
own** min and max::

    min_max_scale(c + alpha*(d - c)) = alpha*(d - min d) / (alpha*(max d - min d))
                                     = (d - min d) / (max d - min d)

`alpha` cancels exactly. Measured across the horizon, the largest difference
between any rung's scaled observation and `p1`'s is 1e-4 -- which is this file's
`%.1f` rounding, not the transform. Mean 0.543 and sd 0.266 to three decimals in
all seven. The rest of the 50-dim observation is invariant too: the price
forecast is a flat 50.0 in every rung, and the marginal-cost slot is fixed by
the unchanged fuel prices. The one slot that does move is
`current_volume / max_power`, which is endogenous -- it differs because dispatch
differs, i.e. downstream of the manipulation rather than alongside it.

So the ladder holds the exogenous observation **bit-identical** and moves only
where the regime boundary falls inside it: 7000 MW sits at scaled 0.868 in `p1`
and at 0.231 in `p7`. That is a clean manipulation of the reward landscape, and
it is the property that makes the ladder worth running.

What genuinely does change: one unit of scaled observation is 3436 MW at `p1`
and 352 MW at `p7`, so the same observation resolution buys ten times finer
discrimination near the boundary at the top of the ladder. Frequency and
boundary sharpness therefore move together -- but they move together *by
construction*, since the transform preserves every hour's rank, so this is one
variable seen twice rather than two variables confounded. Read the ladder as a
trend in "how much of the buffer is pivotal", which is what it manipulates.

Run from anywhere::

    python examples/inputs/example_02_single_bid/make_demand_ladder.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

#: the horizon study cases 02a-02c and p1-p7 all simulate
START, END = "2019-03-01 00:00", "2019-04-01 00:00"

#: sb02b's merit order, from eom_critic_film.merit_order("sb02b"):
#: 5000 MW of cheap naive capacity, then 5 x 500 MW of learners, then the
#: backup. Demand in (7000, 7500] leaves one learner marginal -> pivotal NE.
PIVOTAL_LO, PIVOTAL_HI = 7000.0, 7500.0
CENTRE = (PIVOTAL_LO + PIVOTAL_HI) / 2

LICENSE = """SPDX-FileCopyrightText: ASSUME Developers

SPDX-License-Identifier: AGPL-3.0-or-later
"""


def pivotal_share(hourly: pd.Series, alpha: float) -> float:
    d = CENTRE + alpha * (hourly - CENTRE)
    return float(((d > PIVOTAL_LO) & (d <= PIVOTAL_HI)).mean())


def solve_alpha(hourly: pd.Series, target: float) -> float:
    """Bisect for the compression that puts `target` of the hours in the band.

    `pivotal_share` is monotone decreasing in alpha over (0, 1]: at alpha -> 0
    every hour collapses onto CENTRE, which is inside the band.
    """
    lo, hi = 1e-4, 1.0
    if pivotal_share(hourly, hi) >= target:
        return hi                      # already at or above target undistorted
    for _ in range(200):
        mid = (lo + hi) / 2
        if pivotal_share(hourly, mid) >= target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> None:
    src = pd.read_csv(HERE / "demand_df.csv", index_col=0, parse_dates=True)
    full = src["demand_EOM"]
    hourly = full.resample("1h").mean().loc[START:END]

    print(f"source: {len(full)} rows, horizon {START} .. {END} "
          f"({len(hourly)} hours)")
    print(f"  min {hourly.min():.0f}  mean {hourly.mean():.0f}  "
          f"max {hourly.max():.0f}  sd {hourly.std():.0f}")
    print(f"  pivotal band ({PIVOTAL_LO:.0f}, {PIVOTAL_HI:.0f}] holds "
          f"{pivotal_share(hourly, 1.0):.1%} of hours undistorted\n")

    print(f"  {'case':<6} {'alpha':>7} {'target':>7} {'actual':>7} "
          f"{'mean':>7} {'sd':>7}   regime shares")
    for k in range(1, 8):
        target = k / 8
        alpha = solve_alpha(hourly, target)
        # the transform is applied to the WHOLE series, at native resolution,
        # so the file stays a drop-in replacement for demand_df.csv
        out = src.copy()
        out["demand_EOM"] = CENTRE + alpha * (full - CENTRE)

        got = out["demand_EOM"].resample("1h").mean().loc[START:END]
        shares = {
            "idle": float((got <= 5000).mean()),
            "bertrand": float(((got > 5000) & (got <= PIVOTAL_LO)).mean()),
            "pivotal": float(((got > PIVOTAL_LO) & (got <= PIVOTAL_HI)).mean()),
            "backup": float((got > PIVOTAL_HI).mean()),
        }
        name = f"demand_df_p{k}.csv"
        out.to_csv(HERE / name, float_format="%.1f")
        (HERE / f"{name}.license").write_text(LICENSE, encoding="utf-8")

        pretty = "  ".join(f"{n} {v:.0%}" for n, v in shares.items() if v > 0.005)
        print(f"  p{k:<5} {alpha:7.4f} {target:7.1%} {shares['pivotal']:7.1%} "
              f"{got.mean():7.0f} {got.std():7.0f}   {pretty}")

    print(f"\nwrote 7 series to {HERE}")


if __name__ == "__main__":
    main()
