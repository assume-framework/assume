# packages needed: seaborn, plotly, kaleido

import os
import time
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from assume import World
from assume.scenario.loader_csv import load_scenario_folder
from timeit import Timer

# assume module imports
#import examples.examples as examples
from joblib import Parallel, delayed, parallel_config
from numba import njit


@njit(cache=True)
def _clear_with_inserted(sp_r, sv_r, dp, dv, idx, new_price, new_volume):
    """Clear with one bid virtually inserted at position idx in (sp_r, sv_r).
    Two-pointer sweep, no allocations, no merged arrays."""
    n_r = sp_r.shape[0]  # num residual supplies
    n_d = dp.shape[0]  # num demands
    j = 0  # pointer on demand position
    dem_avail = 0.0  # available demand volume at current price
    cum_s = 0.0  # supply that gets cumulated
    last_ok = -1  #  last valid supply id
    last_ok_price = 0.0  # last valid clearing price
    cleared = 0.0  # cleared volume
    cum_before_k = 0.0  # last supply volume before current volume addition

    # go through redisual supply bids + virtually inserted new bid
    for i in range(n_r + 1):
        # Virtual supply at position i: either inserted bid or original residual
        if i < idx:  # normal supply bids
            sp_i = sp_r[i]; sv_i = sv_r[i]
        elif i == idx:  # virtually insert new bid
            sp_i = new_price; sv_i = new_volume
        else:  # normal supply bids (with adjusted index i)
            sp_i = sp_r[i - 1]; sv_i = sv_r[i - 1]

        # Accumulate demands volume dv as long as its price dp is above supply price sp_i (dp descending)
        while j < n_d and dp[j] >= sp_i:
            dem_avail += dv[j]
            j += 1

        # if available demand > supply from last round we can still add some supply
        # at this market price
        if cum_s < dem_avail:
            last_ok = i
            last_ok_price = sp_i
            cum_before_k = cum_s
            s_total = cum_s + sv_i
            cleared = dem_avail if s_total > dem_avail else s_total
        # FIXME: if last round supply is greater than demand that I can use now -->
        #           cannot add anything (and next round I can use less (descending dp))
        # else: break
        cum_s += sv_i
        # FIXME: Shouldn't j get reset to 0 
        # --> as dp descending to small sp_i, in next iteration this will not be valid anymore!!!!?
        

    return last_ok_price, last_ok, cleared, cum_before_k


@njit(cache=True)
def _probe_classify(sp_r, sv_r, dp, dv, new_price, new_volume):
    """Returns (clear_price, accepted_volume, state).
    state: -1 fully inframarginal, 0 marginal, +1 fully extramarginal."""
    n = sp_r.shape[0]
    idx = 0
    # find index before new price
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
    """
    sp_r: residual supply prices (excluding unit price)
    sv_r: residual supply volumes (excluding unit volume)
    dp: demand prices
    dv: demand volumes
    out_...: buffers for results

    Walk outward from unit_price. Fills the three output buffers in ascending
    price order. Returns the number of probes written."""

    # get starting position: start from unit price????????!!!!!!
    # FIXME: ---> maybe go to market price instead
    #             put start after: "probe at units own price" --> get cp from that
    #                   --> do while until: sp_r[start] < cp:
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

    # for every unit
    for unit_id in np.unique(su):
        # get mask for unit specific elements
        mask = su == unit_id

        # do stuff with unit specific elements
        # --> in this simple version just 1 bid per unit with price and volume
        unit_volume = float(sv[mask][0])
        unit_price  = float(sp[mask][0])

        # Get residual prices and volumes
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

class FastClearing:
    def __init__(self, orders_df):
        self.ts = {}

        # sort data by timestep
        for t, g in orders_df.groupby("start_time", sort=False):
            # get bids of units (keys: "price" and "volume")
            p = g["price"].to_numpy(dtype=np.float64)
            v = g["volume"].to_numpy(dtype=np.float64)
            u = g["unit_id"].to_numpy()

            # separate supply and demand
            s = v > 0; d = v < 0
            sp, sv, su = p[s], v[s], u[s]
            dp, dv = p[d], -v[d]

            # sort units based on prices (neg for demand to get highest bidding)
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
#scenario, study_case = "example_03", "base_case_2019"  # "example_03_naive"  # "example_01a"
#scenario, study_case = "example_03_naive", "base_case_2019"
scenario, study_case = "example_01a", "base"

query = f"SELECT * FROM market_orders where simulation = '{scenario}_{study_case}'"



def init_database(db, query):
    # 1. Load real data once
    print("Read database")
    t0 = time.perf_counter()
    market_orders_df = pd.read_sql(query, db)
    market_orders_df = market_orders_df[["start_time", "unit_id", "accepted_price", "accepted_volume", "price", "volume"]]
    read_time = time.perf_counter() - t0
    print(f"Read db: {read_time*1000:.1f} ms  "
        f"({len(market_orders_df)} entries)")

    # n = len(market_orders_df)  #10**5
    #print("Reduce size to", n)
    #market_orders_df = market_orders_df[:n]
    #print("starting")


    # 2. Build the precomputed structure (this cost is one-time, measure separately)
    t0 = time.perf_counter()
    fc = FastClearing(market_orders_df)
    build_time = time.perf_counter() - t0
    print(f"Setup fast clearing: {build_time*1000:.1f} ms  "
        f"({len(fc.ts)} timesteps)")
    print(fc.ts)

    ts0, data0 = next(iter(fc.ts.items()))
    t0 = time.perf_counter()
    _process_timestep(ts0, data0)
    compile_time = time.perf_counter() - t0
    print(f"JIT compile: {compile_time*1000:.1f} ms  ")




    # 3. Time the full probe sweep. Run a few times — first call may pay
    #    one-off costs (numpy import warmup, allocator settling).
    ts = next(iter(fc.ts))
    sp, sv, su, dp, dv, cum_dv = fc.ts[ts]
    mask = su == "Unit 3"
    sp_r, sv_r = sp[~mask], sv[~mask]

    t = Timer(
        lambda: _probe_classify(sp_r, sv_r, dp, dv, 50.0, 1000.0)
    )
    n, total = t.autorange()
    print(f"_probe_in_residual: {total/n*1e6:.2f} µs per call")
    return market_orders_df, fc


df, fc = init_database(db, query)

for ts in fc.ts:
    sp, sv, su, dp, dv, cum_dv = fc.ts[ts]
    print("supply price", sp)
    print("supply vol", sv)
    print("supply unit", su)
    print("demand price", dp)
    print("demand vol", dv)
    break

    






bench = False

if bench:
    import cProfile
    import pstats
    from pstats import SortKey

    profiler = cProfile.Profile()
    profiler.enable()
    results = fc.run_all_probes()
    #results = fc.run_all_probes()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)   # top 20 by cumulative time


if False:
    world = World(database_uri=db_uri, export_csv_path="")

    # load scenario
    load_scenario_folder(
        world,
        inputs_path="examples/inputs",
        scenario="example_01a",
        study_case="base",
    )