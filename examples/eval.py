# packages needed: seaborn, plotly, kaleido

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from assume import World
from assume.scenario.loader_csv import load_scenario_folder

# assume module imports
#import examples.examples as examples
from joblib import Parallel, delayed, parallel_config
from numba import njit


@njit(cache=True)
def _clear_with_inserted(sp_r, sv_r, dp, dv, idx, new_price, new_volume):
    """Clear with one bid virtually inserted at position idx in (sp_r, sv_r).
    Two-pointer sweep, no allocations, no merged arrays."""
    n_r = sp_r.shape[0]
    n_d = dp.shape[0]
    j = 0
    dem_avail = 0.0
    cum_s = 0.0
    last_ok = -1
    last_ok_price = 0.0
    cleared = 0.0
    cum_before_k = 0.0

    for i in range(n_r + 1):
        # Virtual supply at position i: either inserted bid or original residual
        if i < idx:
            sp_i = sp_r[i]; sv_i = sv_r[i]
        elif i == idx:
            sp_i = new_price; sv_i = new_volume
        else:
            sp_i = sp_r[i - 1]; sv_i = sv_r[i - 1]

        # Pull in demand bids clearing against sp_i (dp descending)
        while j < n_d and dp[j] >= sp_i:
            dem_avail += dv[j]
            j += 1

        if cum_s < dem_avail:
            last_ok = i
            last_ok_price = sp_i
            cum_before_k = cum_s
            s_total = cum_s + sv_i
            cleared = dem_avail if s_total > dem_avail else s_total
        cum_s += sv_i

    return last_ok_price, last_ok, cleared, cum_before_k


@njit(cache=True)
def _probe_classify(sp_r, sv_r, dp, dv, new_price, new_volume):
    """Returns (clear_price, accepted_volume, state).
    state: -1 fully inframarginal, 0 marginal, +1 fully extramarginal."""
    n = sp_r.shape[0]
    idx = 0
    while idx < n and sp_r[idx] < new_price:
        idx += 1

    cp, k, cleared, cum_before = _clear_with_inserted(
        sp_r, sv_r, dp, dv, idx, new_price, new_volume
    )

    if k < 0 or idx > k:
        return cp, 0.0, 1
    if idx < k:
        return cp, new_volume, -1
    # idx == k → unit is marginal
    accepted = cleared - cum_before
    if accepted >= new_volume - 1e-9:
        return cp, accepted, -1
    if accepted <= 1e-9:
        return cp, accepted, 1
    return cp, accepted, 0


@njit(cache=True)
def _walk_outward(sp_r, sv_r, dp, dv, unit_price, unit_volume,
                  out_probe_prices, out_clear_prices, out_accepted_vols):
    """Walk outward from unit_price. Fills the three output buffers in ascending
    price order. Returns the number of probes written."""
    n_r = sp_r.shape[0]
    start = 0
    while start < n_r and sp_r[start] < unit_price:
        start += 1
    n_cand = n_r + 1

    # Scratch buffers indexed by candidate position
    pr_buf = np.empty(n_cand)
    cp_buf = np.empty(n_cand)
    av_buf = np.empty(n_cand)

    # Probe at start (the unit's own price)
    cp, av, st = _probe_classify(sp_r, sv_r, dp, dv, unit_price, unit_volume)
    pr_buf[start] = unit_price
    cp_buf[start] = cp
    av_buf[start] = av
    lo = start
    hi = start

    # Walk down
    i = start - 1
    while i >= 0:
        p = sp_r[i]
        cp, av, st = _probe_classify(sp_r, sv_r, dp, dv, p, unit_volume)
        pr_buf[i] = p; cp_buf[i] = cp; av_buf[i] = av
        lo = i
        if st == -1:
            break
        i -= 1

    # Walk up
    i = start + 1
    while i < n_cand:
        p = sp_r[i - 1]
        cp, av, st = _probe_classify(sp_r, sv_r, dp, dv, p, unit_volume)
        pr_buf[i] = p; cp_buf[i] = cp; av_buf[i] = av
        hi = i
        if st == 1:
            break
        i += 1

    n_out = hi - lo + 1
    for m in range(n_out):
        out_probe_prices[m] = pr_buf[lo + m]
        out_clear_prices[m] = cp_buf[lo + m]
        out_accepted_vols[m] = av_buf[lo + m]
    return n_out


def _process_timestep(timestep, ts_data):
    """Module-level, pickles cleanly across workers."""
    sp, sv, su, dp, dv, cum_dv = ts_data
    # cum_dv unused here — the JIT version computes demand on the fly
    out = {}
    max_probes = sp.size + 1
    pr_buf = np.empty(max_probes)
    cp_buf = np.empty(max_probes)
    av_buf = np.empty(max_probes)
    # print(timestep, end="__")
    for unit_id in np.unique(su):
        mask = su == unit_id
        unit_volume = float(sv[mask][0])
        unit_price  = float(sp[mask][0])
        sp_r = np.ascontiguousarray(sp[~mask])
        sv_r = np.ascontiguousarray(sv[~mask])

        n = _walk_outward(sp_r, sv_r, dp, dv,
                          unit_price, unit_volume,
                          pr_buf, cp_buf, av_buf)

        out[(timestep, str(unit_id))] = [
            {"price": float(pr_buf[i]),
             "volume": unit_volume,
             "accepted_price": float(cp_buf[i]),
             "accepted_volume": float(av_buf[i])}
            for i in range(n)
        ]
    return out

def _process_chunk(items_chunk):
    out = {}
    for t, data in items_chunk:
        out.update(_process_timestep(t, data))
    return out

class FastClearing:
    def __init__(self, orders_df):
        self.ts = {}
        for t, g in orders_df.groupby("start_time", sort=False):
            p = g["price"].to_numpy(dtype=np.float64)
            v = g["volume"].to_numpy(dtype=np.float64)
            u = g["unit_id"].to_numpy()
            s = v > 0; d = v < 0
            sp, sv, su = p[s], v[s], u[s]
            dp, dv = p[d], -v[d]
            i = np.argsort(sp, kind="stable")
            j = np.argsort(-dp, kind="stable")
            self.ts[t] = (sp[i], sv[i], su[i],
                          dp[j], dv[j], np.cumsum(dv[j]))

    def run_all_probes(self):
        """Serial version — walk-outward, single core."""
        merged = {}
        for t, data in self.ts.items():
            merged.update(_process_timestep(t, data))
        return merged


db_uri = "postgresql://assume:assume@localhost:5432/assume"
db = create_engine(db_uri)

input_path = "examples/inputs/"
scenario = "example_03"  # "example_03_naive"  # "example_01a"
#scenario = "example_03_naive"
study_case = "base_case_2019"  # "base"

query = f"SELECT * FROM market_orders where simulation = '{scenario}_{study_case}'"
# market_orders_df = pd.read_sql(query, db)
#print("orders", market_orders_df.columns)

if False:
    orders = market_orders_df[["start_time", "unit_id", "accepted_price", "accepted_volume", "price", "volume"]]
    print(len(orders))
    orders = orders[:100]
    print("cropped", len(orders))
    #orders = orders.groupby(["start_time"])


    fc = FastClearing(market_orders_df)
    results = fc.run_all_probes(dedupe_consecutive=False)

    # Inspect a single (timestep, unit):
    ts = next(iter(fc.ts))
    #print(pd.DataFrame(results[(ts, "Unit 3")]))

    # Flatten everything into one DataFrame if you'd rather work that way:
    rows = [
        {"start_time": t, "unit_id": u, **r}
        for (t, u), probes in results.items()
        for r in probes
    ]
    df = pd.DataFrame(rows)
    #print(df)


import time
import cProfile
import pstats
from pstats import SortKey

# 1. Load real data once
print("Read database")
t0 = time.perf_counter()
market_orders_df = pd.read_sql(query, db)
market_orders_df = market_orders_df[["start_time", "unit_id", "accepted_price", "accepted_volume", "price", "volume"]]
read_time = time.perf_counter() - t0
print(f"Setup: {read_time*1000:.1f} ms  "
      f"({len(market_orders_df)} entries)")
n = len(market_orders_df)  #10**5
print("Reduce size to", n)
market_orders_df = market_orders_df[:n]
print("starting")


# 2. Build the precomputed structure (this cost is one-time, measure separately)
t0 = time.perf_counter()
fc = FastClearing(market_orders_df)
build_time = time.perf_counter() - t0
print(f"Setup: {build_time*1000:.1f} ms  "
      f"({len(fc.ts)} timesteps)")

print("compiling")
n_jobs = 1

ts0, data0 = next(iter(fc.ts.items()))
t0 = time.perf_counter()
_process_timestep(ts0, data0)
elapsed = time.perf_counter() - t0
print("compile time:", elapsed)


# 3. Time the full probe sweep. Run a few times — first call may pay
#    one-off costs (numpy import warmup, allocator settling).
from timeit import Timer
ts = next(iter(fc.ts))
sp, sv, su, dp, dv, cum_dv = fc.ts[ts]
mask = su == "Unit 3"
sp_r, sv_r = sp[~mask], sv[~mask]

t = Timer(
    lambda: _probe_classify(sp_r, sv_r, dp, dv, 50.0, 1000.0)
)
n, total = t.autorange()
print(f"_probe_in_residual: {total/n*1e6:.2f} µs per call")




profiler = cProfile.Profile()
profiler.enable()
results = fc.run_all_probes()
#results = fc.run_all_probes()
profiler.disable()
stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
stats.print_stats(20)   # top 20 by cumulative time


for run in range(1):
    t0 = time.perf_counter()
    results = fc.run_all_probes()
    #results = fc.run_all_probes()
    elapsed = time.perf_counter() - t0

    n_probes = sum(len(v) for v in results.values())
    n_pairs  = len(results)
    print(f"Run {run+1}: {elapsed:.3f} s  "
          f"{n_pairs} (timestep, unit) pairs, "
          f"{n_probes} probes total, "
          f"{n_probes/elapsed:,.0f} probes/sec")


if False:
    world = World(database_uri=db_uri, export_csv_path="")

    # load scenario
    load_scenario_folder(
        world,
        inputs_path="examples/inputs",
        scenario="example_01a",
        study_case="base",
    )