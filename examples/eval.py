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
# from joblib import Parallel, delayed, parallel_config
from numba import njit


@njit(cache=True)
def _clear_with_inserted(sp_r, sv_r, dp, dv, idx, new_price, new_volume):
    """Clear with one bid virtually inserted at position idx in (sp_r, sv_r).
    Two-pointer sweep, no allocations, no merged arrays."""
    n_r = sp_r.shape[0]  # num residual supplies
    n_d = dp.shape[0]  # num demands
    j = 0  # pointer on demand position
    # available demand volume at current price — starts at total demand and is
    # decremented as we sweep past demand bids that can no longer pay sp_i
    dem_avail = 0.0
    for k in range(n_d):
        dem_avail += dv[k]
    cum_s = 0.0  # supply that gets cumulated
    last_ok = -1  #  last valid supply id
    last_ok_price = 0.0  # last valid clearing price
    cleared = 0.0  # cleared volume
    cum_before_k = 0.0  # last supply volume before current volume addition
    vol_at_price = 0.0


    # go through redisual supply bids + virtually inserted new bid
    for i in range(n_r + 1):
        # Virtual supply at position i: either inserted bid or original residual
        if i < idx:  # normal supply bids
            sp_i = sp_r[i]; sv_i = sv_r[i]
        elif i == idx:  # virtually insert new bid
            sp_i = new_price; sv_i = new_volume
        else:  # normal supply bids (with adjusted index i)
            sp_i = sp_r[i - 1]; sv_i = sv_r[i - 1]

        # Reduce available demand by bids that cant be used anymore (dp ascending)
        while j < n_d and dp[j] < sp_i:
            dem_avail -= dv[j]
            j += 1

        # if available demand > supply from last round we can still add some supply
        # at this market price
        if cum_s < dem_avail:
            last_ok = i
            if sp_i != last_ok_price:
                vol_at_price = 0.0
            last_ok_price = sp_i
            cum_before_k = cum_s
            s_total = cum_s + sv_i
            cleared = dem_avail if s_total > dem_avail else s_total
            vol_at_price += cleared - cum_before_k  # add what got added to cleared
        else:  # if last round cannot satisfy demand, this round is worse (sp ascending)
            break
        cum_s += sv_i


    return last_ok_price, last_ok, cleared, cum_before_k, vol_at_price


@njit(cache=True)
def _probe_classify(sp_r, sv_r, dp, dv, new_price, new_volume):
    """Returns (clear_price, accepted_volume, state).
    state: -1 fully inframarginal, 0 marginal, +1 fully extramarginal.

    Splitting rule: give credit to current unit first
    """
    n = sp_r.shape[0]
    idx = 0
    # find index before new price
    while idx < n and sp_r[idx] < new_price:
        idx += 1

    # cp: clearing price, k: index of last dispatched unit (marginal unit)
    # cleared: total cleared volume, cum_before: last fully cleared volume
    # vol_at_price: dispatched volume at this price
    cp, k, cleared, cum_before, vol_at_price = _clear_with_inserted(
        sp_r, sv_r, dp, dv, idx, new_price, new_volume
    )

    # cp < new_price: unit not accepted --> extramarginal, k < 0: nothing got accepted
    if cp < new_price or k < 0:
        return cp, 0.0, 1  # extramarginal
    
    # cp > new_price: unit got fully accepted --> inframarginal
    if cp > new_price:
        return cp, new_volume, -1

    #cp == new_price: --> unit is marginal (at correct price)

    # accepted = cleared - cum_before  # deterministic based on sort
    
    # assume that current unit got all volume at this price
    accepted = new_volume if vol_at_price >= new_volume else vol_at_price
    return cp, accepted, 0

    #if k < 0 or idx > k:
    #    return cp, 0.0, 1
    
    # idx < k: unit got fully accepted 
    #if idx < k:
    #    return cp, new_volume, -1
    # idx == k → unit is marginal
    #accepted = cleared - cum_before
    #if accepted >= new_volume - 1e-9:
    #    return cp, accepted, -1
    #if accepted <= 1e-9:
    #    return cp, accepted, 1
    #return cp, accepted, 0


@njit(cache=True)
def _walk_outward(sp_r, sv_r, dp, dv, unit_price, unit_volume,
                  out_probe_prices, out_clear_prices, out_accepted_vols):
    """
    sp_r: residual supply prices (excluding unit price, ascending)
    sv_r: residual supply volumes (excluding unit volume, ascending)
    dp: demand prices  (ascending)
    dv: demand volumes  (ascending)
    out_...: buffers for results

    Walk outward from unit_price. Fills the three output buffers in ascending
    price order. Returns the number of probes written."""
    n_r = sp_r.shape[0]
    n_cand = n_r + 1

    # Scratch buffers indexed by candidate position
    pr_buf = np.empty(n_cand)  # probing price
    cp_buf = np.empty(n_cand)  # clearing price
    av_buf = np.empty(n_cand)  # accepted volume

    # Probe at start (the unit's own price)
    cp, av, st = _probe_classify(sp_r, sv_r, dp, dv, unit_price, unit_volume)

    # start at current clearing price
    start = 0
    while start < n_r and sp_r[start] < cp:
        start += 1
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
    sp, sv, su, dp, dv, cum_dv, cp, av = ts_data
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
            # get accepted volume and clearing price ("accepted_price") of initial bid
            cp = g["accepted_price"].to_numpy(dtype=np.float64)
            av = g["accepted_volume"].to_numpy(dtype=np.float64)

            # separate supply and demand
            s = v > 0; d = v < 0
            sp, sv, su = p[s], v[s], u[s]
            dp, dv = p[d], -v[d]

            # sort units based on prices in **ascending** order
            i = np.argsort(sp, kind="stable")
            j = np.argsort(dp, kind="stable")
            self.ts[t] = (sp[i], sv[i], su[i],
                          dp[j], dv[j], np.cumsum(dv[j]),
                          cp[i], av[i])

    def run_all_probes(self):
        """Serial version — walk-outward, single core."""
        merged = {}
        for t, data in self.ts.items():
            merged.update(_process_timestep(t, data))
        return merged

    def calculate_exploitability(self, units, results):
        """Per-(timestep, unit) exploitability.

        marginal_costs: iterable of (unit_id, marginal_cost) pairs, or dict.
        results: output of run_all_probes().

        For each unit, exploitability = best_response_profit - original_profit,
        where profit = (accepted_price - marginal_cost) * accepted_volume,
        best_response is taken over all probed bid prices, and "original"
        refers to the probe at the unit's actual orderbook bid price.
        """
        #mc = dict(marginal_costs)
        per_unit = {}
        for (t, unit_id), probes in results.items():
            if unit_id not in units:
                continue
            c = units[unit_id].marginal_cost[t] if len(units[unit_id].marginal_cost) > 1 else units[unit_id].marginal_cost
            su, cp, av = self.ts[t][2], self.ts[t][6], self.ts[t][7]
            original_accepted_price = float(cp[su == unit_id][0])
            original_accepted_volume = float(av[su == unit_id][0])

            original_profit = (original_accepted_price - c) * original_accepted_volume
            best_profit = -np.inf
            for p in probes:
                profit = (p["accepted_price"] - c) * p["accepted_volume"]
                if profit > best_profit:
                    best_profit = profit
            per_unit[(t, unit_id)] = best_profit - original_profit
        return per_unit

    def aggregate_per_timestep(self, per_unit):
        """Sum per-unit exploitability across units for each timestep."""
        num_units = int(len(per_unit) / len(self.ts))
        agg = {}
        for (t, _), e in per_unit.items():
            agg[t] = agg.get(t, 0.0) + e / num_units
        return agg

    @staticmethod
    def aggregate_total(per_unit):
        """Single value: mean across all timesteps and units."""
        return float(sum(per_unit.values())/len(per_unit.values())) 

    @staticmethod
    def plot_exploitability(per_unit, path=None):
        """Time-series plot of total exploitability with a per-unit breakdown."""
        import matplotlib.pyplot as plt

        df = pd.DataFrame(
            [(t, u, e) for (t, u), e in per_unit.items()],
            columns=["time", "unit", "exploitability"],
        )
        per_t = df.groupby("time")["exploitability"].sum().sort_index()
        per_u = df.groupby("unit")["exploitability"].sum().sort_values(ascending=False)
        total = float(per_t.sum())

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [2, 1]}
        )
        per_t.plot(ax=ax1, marker=".")
        ax1.set_xlabel("time")
        ax1.set_ylabel("exploitability (sum over units)")
        ax1.set_title(f"Exploitability over time — total: {total:.2f}")
        ax1.grid(True, alpha=0.3)

        per_u.plot(kind="bar", ax=ax2)
        ax2.set_ylabel("exploitability (sum over time)")
        ax2.set_title("Per-unit exploitability")
        ax2.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        if path:
            fig.savefig(path)
        return fig


# ---------------------------------------------------------------------------
# Tests for _clear_with_inserted
# ---------------------------------------------------------------------------
# Convention used by FastClearing (see __init__):
#   sp_r, sv_r  : residual supply prices/volumes, sp_r sorted ascending
#   dp,  dv     : demand prices/volumes, dp sorted ascending, dv positive
#   idx         : position to virtually insert (new_price, new_volume)
# Returns:
#   (clearing_price, last_ok_index, cleared_volume,
#    cum_supply_before_marginal, volume_at_clearing_price)
# A "no-op" insertion (used when we only want to clear the residual stack)
# uses idx = len(sp_r), new_price = +inf, new_volume = 0.0.


def _no_insert():
    return float("inf"), 0.0


def _assert_clear(result, *, price, last_ok, cleared, cum_before, vol_at_price, tol=1e-9):
    cp, k, cl, cb, vap = result
    assert abs(cp - price) < tol, f"clearing price: got {cp}, expected {price}"
    assert k == last_ok, f"last_ok index: got {k}, expected {last_ok}"
    assert abs(cl - cleared) < tol, f"cleared volume: got {cl}, expected {cleared}"
    assert abs(cb - cum_before) < tol, f"cum_before_k: got {cb}, expected {cum_before}"
    assert abs(vap - vol_at_price) < tol, f"vol_at_price: got {vap}, expected {vol_at_price}"


def test_clear_single_supply_single_demand():
    """One supply bid fully meets a willing demand bid."""
    sp = np.array([50.0])
    sv = np.array([100.0])
    dp = np.array([200.0])
    dv = np.array([100.0])
    np_, nv = _no_insert()
    result = _clear_with_inserted(sp, sv, dp, dv, len(sp), np_, nv)
    _assert_clear(result, price=50.0, last_ok=0, cleared=100.0,
                  cum_before=0.0, vol_at_price=100.0)


def test_clear_demand_binding_marginal_supply():
    """Demand cuts off in the middle of the most expensive supply bid."""
    sp = np.array([20.0, 50.0, 100.0])
    sv = np.array([100.0, 200.0, 300.0])
    dp = np.array([200.0])
    dv = np.array([400.0])
    np_, nv = _no_insert()
    result = _clear_with_inserted(sp, sv, dp, dv, len(sp), np_, nv)
    # cum supply: 100, 300, 600 — demand 400 — clears at 100 with 100 MW of
    # the 300 MW marginal bid actually dispatched.
    _assert_clear(result, price=100.0, last_ok=2, cleared=400.0,
                  cum_before=300.0, vol_at_price=100.0)


def test_clear_supply_binding_demand_unmet():
    """Total supply is below demand — all supply dispatches, price = most expensive."""
    sp = np.array([50.0, 100.0])
    sv = np.array([100.0, 100.0])
    dp = np.array([200.0])
    dv = np.array([1000.0])
    np_, nv = _no_insert()
    result = _clear_with_inserted(sp, sv, dp, dv, len(sp), np_, nv)
    _assert_clear(result, price=100.0, last_ok=1, cleared=200.0,
                  cum_before=100.0, vol_at_price=100.0)


def test_clear_demand_willingness_filters_supply():
    """A demand bid that can't pay the next supply price exits the market."""
    sp = np.array([50.0, 200.0])
    sv = np.array([100.0, 100.0])
    dp = np.array([100.0, 300.0])  # ascending
    dv = np.array([100.0, 100.0])
    np_, nv = _no_insert()
    result = _clear_with_inserted(sp, sv, dp, dv, len(sp), np_, nv)
    # At sp=50 all 200 MW demand is in play, 100 MW supply clears.
    # At sp=200 the 100 MW @ 100 demand drops out, only 100 MW demand remains,
    # cum_s already 100 -> no further clearing. Marginal = first bid @ 50.
    _assert_clear(result, price=50.0, last_ok=0, cleared=100.0,
                  cum_before=0.0, vol_at_price=100.0)


def test_clear_multiple_supply_bids_same_price_accumulate():
    """Two supply bids at the same price contribute to vol_at_price together."""
    sp = np.array([50.0, 50.0, 100.0])
    sv = np.array([100.0, 100.0, 100.0])
    dp = np.array([200.0])
    dv = np.array([250.0])
    np_, nv = _no_insert()
    result = _clear_with_inserted(sp, sv, dp, dv, len(sp), np_, nv)
    # cum supply 100, 200, 300 — demand 250 — clears at 100, marginal piece 50.
    _assert_clear(result, price=100.0, last_ok=2, cleared=250.0,
                  cum_before=200.0, vol_at_price=50.0)


def test_insert_cheap_bid_becomes_marginal():
    """Inserted cheap bid at idx=0 satisfies all demand alone."""
    sp = np.array([100.0, 200.0])
    sv = np.array([200.0, 200.0])
    dp = np.array([500.0])
    dv = np.array([200.0])
    result = _clear_with_inserted(sp, sv, dp, dv, 0, 50.0, 300.0)
    # Effective supply: (50,300), (100,200), (200,200). Demand 200 < 300 -> the
    # inserted bid is the marginal bid; cleared all from it.
    _assert_clear(result, price=50.0, last_ok=0, cleared=200.0,
                  cum_before=0.0, vol_at_price=200.0)


def test_insert_expensive_bid_does_not_clear():
    """Inserted bid above clearing price is extramarginal — clearing unaffected."""
    sp = np.array([50.0, 100.0])
    sv = np.array([100.0, 100.0])
    dp = np.array([200.0])
    dv = np.array([150.0])
    # Insert at end of residual stack, above current clearing price (100).
    result = _clear_with_inserted(sp, sv, dp, dv, 2, 300.0, 500.0)
    # cum supply at price 100 reaches 200, demand 150 already met at 100.
    # The 300 bid is extramarginal — function should break before counting it.
    _assert_clear(result, price=100.0, last_ok=1, cleared=150.0,
                  cum_before=100.0, vol_at_price=50.0)


def test_insert_in_middle_becomes_marginal():
    """Inserted bid sits between residual bids and is the marginal one."""
    sp = np.array([50.0, 100.0])
    sv = np.array([100.0, 100.0])
    dp = np.array([200.0])
    dv = np.array([150.0])
    result = _clear_with_inserted(sp, sv, dp, dv, 1, 75.0, 50.0)
    # Effective supply: (50,100), (75,50), (100,100). cum 100, 150, 250.
    # Demand 150 — marginal bid is the inserted (75, 50), fully accepted.
    _assert_clear(result, price=75.0, last_ok=1, cleared=150.0,
                  cum_before=100.0, vol_at_price=50.0)


def test_insert_in_middle_inframarginal():
    """Inserted bid is cleared but a more expensive residual bid sets price."""
    sp = np.array([50.0, 100.0])
    sv = np.array([100.0, 100.0])
    dp = np.array([200.0])
    dv = np.array([250.0])
    result = _clear_with_inserted(sp, sv, dp, dv, 1, 75.0, 50.0)
    # Effective supply: (50,100), (75,50), (100,100). cum 100, 150, 250.
    # Demand 250 — marginal at 100, inserted bid fully dispatched.
    _assert_clear(result, price=100.0, last_ok=2, cleared=250.0,
                  cum_before=150.0, vol_at_price=100.0)


def test_demand_completely_unwilling_no_clearing():
    """Every demand bid is below the cheapest supply price — nothing clears."""
    sp = np.array([100.0, 200.0])
    sv = np.array([100.0, 100.0])
    dp = np.array([10.0, 50.0])
    dv = np.array([100.0, 100.0])
    np_, nv = _no_insert()
    result = _clear_with_inserted(sp, sv, dp, dv, len(sp), np_, nv)
    cp, k, cl, _, _ = result
    assert k == -1, f"expected no clearing (last_ok=-1), got {k}"
    assert cl == 0.0, f"expected cleared=0, got {cl}"
    assert cp == 0.0, f"expected clearing price 0, got {cp}"


def test_matches_pay_as_clear_market():
    """Cross-check against PayAsClearRole on the same merit order.

    Mirrors the orderbook from test_market_pay_as_clear in
    tests/test_simple_market_mechanisms.py: two demand bids (-400 @ 3000,
    -100 @ 3000) and two supply bids (300 @ 100, 200 @ 50). Expected clearing
    price 100, total cleared volume 500.
    """
    sp = np.array([50.0, 100.0])
    sv = np.array([200.0, 300.0])
    dp = np.array([3000.0, 3000.0])
    dv = np.array([400.0, 100.0])
    np_, nv = _no_insert()
    result = _clear_with_inserted(sp, sv, dp, dv, len(sp), np_, nv)
    _assert_clear(result, price=100.0, last_ok=1, cleared=500.0,
                  cum_before=200.0, vol_at_price=300.0)


def test_matches_pay_as_clears_single_demand_more_generation():
    """Cross-check the test_market_pay_as_clears_single_demand_more_generation
    scenario: demand -400 @ 3000, supply (300 @ 100, 200 @ 50, 230 @ 60).
    Expected clearing price 60, cleared 400 MW.
    """
    sp = np.array([50.0, 60.0, 100.0])
    sv = np.array([200.0, 230.0, 300.0])
    dp = np.array([3000.0])
    dv = np.array([400.0])
    np_, nv = _no_insert()
    result = _clear_with_inserted(sp, sv, dp, dv, len(sp), np_, nv)
    # cum supply: 200, 430, 730 — demand 400 — marginal at 60.
    _assert_clear(result, price=60.0, last_ok=1, cleared=400.0,
                  cum_before=200.0, vol_at_price=200.0)


# ---------------------------------------------------------------------------
# Tests for _walk_outward, _process_timestep and end-to-end run_all_probes
# ---------------------------------------------------------------------------


def test_walk_outward_hits_relevant_niveaus_and_stops():
    """walk_outward probes every in-play merit-order slot and short-circuits
    once probes leave the in-play region in either direction.

    Residual supply (sorted ascending) — big units at the borders, small ones
    in between so the best bid is non-obvious:

        10  @ 100  (big cheap baseload)
        50  @  20
        55  @  20
        65  @  20
        80  @ 100  (big mid peaker)
       200  @ 200  (big expensive)
       300  @ 100  (most expensive)

    Demand: 200 MW @ 1000. Without the test unit, clearing is at 80 (the
    cheap stack 10+50+55+65 = 160 MW covers the first 160, then 40 of the
    100 MW @ 80 bid).

    Test unit bids 60 @ 50, sitting between the small middle bids. Expected
    walk:

      walk-down: probe 65 -> marginal (full 50)
                 probe 55 -> inframarginal -> break (does NOT probe 50 or 10)
      walk-up:   probe 80 -> marginal (40 of 50 dispatched, cp moves to 80)
                 probe 200 -> extramarginal -> break (does NOT probe 300)

    Profit-wise the best bid is non-trivial: 55, 60 and 65 all yield
    (65 - c) * 50; bidding 80 yields (80 - c) * 40 — which one wins depends
    on c. The test pins down the exact (probe, clear, accepted) triples and
    asserts that the un-probed buffer slots are left untouched.
    """
    sp_r = np.array([10.0, 50.0, 55.0, 65.0, 80.0, 200.0, 300.0])
    sv_r = np.array([100.0, 20.0, 20.0, 20.0, 100.0, 200.0, 100.0])
    dp = np.array([1000.0])
    dv = np.array([200.0])
    n_cand = len(sp_r) + 1  # 8

    SENTINEL = -999.0
    pr_buf = np.full(n_cand, SENTINEL)
    cp_buf = np.full(n_cand, SENTINEL)
    av_buf = np.full(n_cand, SENTINEL)

    n = _walk_outward(sp_r, sv_r, dp, dv, 60.0, 50.0,
                      pr_buf, cp_buf, av_buf)

    # Five probes in positional order — the cheap end (10, 50) and the
    # extreme top (300) are correctly skipped.
    expected = [
        (55.0,  65.0, 50.0),   # walk-down step: inframarginal -> stop
        (60.0,  65.0, 50.0),   # unit's own bid: inframarginal at clearing 65
        (65.0,  65.0, 50.0),   # marginal, unit grabs full 50 MW
        (80.0,  80.0, 40.0),   # marginal at higher slot, only 40 of 50 dispatched
        (200.0, 80.0,  0.0),   # extramarginal -> stop
    ]
    assert n == len(expected), f"expected {len(expected)} probes, got {n}"
    actual = [(pr_buf[i], cp_buf[i], av_buf[i]) for i in range(n)]
    assert actual == expected, f"expected {expected}, got {actual}"

    # Slots beyond what was probed must remain untouched (sentinel value).
    for k in range(n, n_cand):
        assert pr_buf[k] == SENTINEL, f"pr_buf[{k}] should be untouched, got {pr_buf[k]}"
        assert cp_buf[k] == SENTINEL, f"cp_buf[{k}] should be untouched, got {cp_buf[k]}"
        assert av_buf[k] == SENTINEL, f"av_buf[{k}] should be untouched, got {av_buf[k]}"


def test_walk_outward_marginal_and_extramarginal():
    """walk_outward probes every merit-order slot around the unit's bid.

    Setup: residual stack [(20,100), (50,100)], demand 150 MW @ 200. Unit
    under test bids 80 @ 100 — extramarginal at its own price. Walking
    outward should also probe prices 50 (marginal, 50 MW dispatched) and 20
    (marginal, full 100 MW dispatched).
    """
    sp_r = np.array([20.0, 50.0])
    sv_r = np.array([100.0, 100.0])
    dp = np.array([200.0])
    dv = np.array([150.0])
    pr_buf = np.empty(3)
    cp_buf = np.empty(3)
    av_buf = np.empty(3)
    n = _walk_outward(sp_r, sv_r, dp, dv, 80.0, 100.0,
                      pr_buf, cp_buf, av_buf)
    assert n == 3
    by_price = {pr_buf[i]: (cp_buf[i], av_buf[i]) for i in range(n)}
    assert by_price[20.0] == (20.0, 100.0), by_price
    assert by_price[50.0] == (50.0, 50.0), by_price
    assert by_price[80.0] == (50.0, 0.0), by_price  # extramarginal at own bid


def test_process_timestep_one_entry_per_unit():
    """_process_timestep keys results by (timestep, unit_id) and emits one
    probe entry per merit-order slot, each with the unit's own volume."""
    timestep = pd.Timestamp("2025-01-01 00:00")
    # ts_data layout matches FastClearing.__init__: sp/sv/su sorted ascending
    # by sp; cum_dv/cp/av aren't read by _process_timestep itself.
    sp = np.array([20.0, 50.0, 80.0])
    sv = np.array([100.0, 100.0, 100.0])
    su = np.array(["U1", "U2", "U3"])
    dp = np.array([200.0])
    dv = np.array([250.0])
    cum_dv = np.cumsum(dv)
    cp = np.zeros_like(sp)
    av = np.zeros_like(sp)

    out = _process_timestep(timestep, (sp, sv, su, dp, dv, cum_dv, cp, av))

    assert set(out.keys()) == {(timestep, "U1"), (timestep, "U2"), (timestep, "U3")}
    for key, probes in out.items():
        assert len(probes) == 3, f"{key}: expected 3 probes, got {len(probes)}"
        for p in probes:
            assert set(p.keys()) == {"price", "volume", "accepted_price", "accepted_volume"}
            assert p["volume"] == 100.0
        # Probe prices should cover the three merit-order slots (own + 2 residual).
        probe_prices = {p["price"] for p in probes}
        assert probe_prices == {20.0, 50.0, 80.0}


def test_run_all_probes_integration_3_units_2_timesteps():
    """End-to-end: build a small market from a DataFrame and check that
    FastClearing.run_all_probes returns the expected merit-order probes.

    Three supply units U1/U2/U3 priced 20/50/80 (vol 100 each) plus one
    demand bid 'D' at 200. Demand is 250 MW in t1 (clears at 80) and 150 MW
    in t2 (clears at 50). For every (timestep, unit) the probes are fully
    determined by these prices/volumes.
    """
    t1 = pd.Timestamp("2025-01-01 00:00")
    t2 = pd.Timestamp("2025-01-01 01:00")

    def row(t, unit, price, volume, ap, av):
        return {"start_time": t, "unit_id": unit, "price": price,
                "volume": volume, "accepted_price": ap, "accepted_volume": av}

    rows = [
        # t1 — clearing at 80, demand 250: U1+U2 full, U3 marginal at 50 MW
        row(t1, "U1", 20.0, 100.0, 80.0, 100.0),
        row(t1, "U2", 50.0, 100.0, 80.0, 100.0),
        row(t1, "U3", 80.0, 100.0, 80.0, 50.0),
        row(t1, "D",  200.0, -250.0, 80.0, -250.0),
        # t2 — clearing at 50, demand 150: U1 full, U2 marginal at 50 MW, U3 out
        row(t2, "U1", 20.0, 100.0, 50.0, 100.0),
        row(t2, "U2", 50.0, 100.0, 50.0, 50.0),
        row(t2, "U3", 80.0, 100.0, 50.0, 0.0),
        row(t2, "D",  200.0, -150.0, 50.0, -150.0),
    ]
    fc = FastClearing(pd.DataFrame(rows))
    results = fc.run_all_probes()

    expected_keys = {(t1, "U1"), (t1, "U2"), (t1, "U3"),
                     (t2, "U1"), (t2, "U2"), (t2, "U3")}
    assert set(results.keys()) == expected_keys

    def by_price(probes):
        return {p["price"]: (p["accepted_price"], p["accepted_volume"]) for p in probes}

    # ---- t1 (clearing price 80, demand 250) ----
    # U1 (20) inframarginal at every probe — full 100 MW dispatch at price 80 due to splitting rule!
    assert by_price(results[(t1, "U1")]) == {
        20.0: (80.0, 100.0), 50.0: (80.0, 100.0), 80.0: (80.0, 100.0),
    }
    # U2 (50) likewise inframarginal at every probe.
    assert by_price(results[(t1, "U2")]) == {
        20.0: (80.0, 100.0), 50.0: (80.0, 100.0), 80.0: (80.0, 100.0),
    }
    # U3 (80) is marginal at its own bid (50 MW). Underbidding drops the
    # clearing price to 50 but lets U3 fully dispatch (100 MW).
    assert by_price(results[(t1, "U3")]) == {
        20.0: (50.0, 100.0), 50.0: (50.0, 100.0), 80.0: (80.0, 50.0),
    }

    # ---- t2 (clearing price 50, demand 150) ----
    # U1 (20) fully dispatched at the low probes; becomes marginal if bidding up to 80.
    assert by_price(results[(t2, "U1")]) == {
        20.0: (50.0, 100.0), 50.0: (50.0, 100.0), 80.0: (80.0, 50.0),
    }
    # U2 (50, marginal at own bid) gets 50 MW; underbidding to 20 wins it the
    # full 100 MW at the new clearing price 20.
    assert by_price(results[(t2, "U2")]) == {
        20.0: (20.0, 100.0), 50.0: (50.0, 50.0), 80.0: (80.0, 50.0),
    }
    # U3 (80) extramarginal at its own bid (0 MW); underbidding to 50 yields
    # 50 MW, to 20 yields the full 100 MW.
    assert by_price(results[(t2, "U3")]) == {
        20.0: (20.0, 100.0), 50.0: (50.0, 50.0), 80.0: (50.0, 0.0),
    }


def _run_all_tests():
    """Run every test_* defined in this module and report pass/fail."""
    import traceback
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
        except Exception:
            failures += 1
            print(f"  ERROR {fn.__name__}:")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures == 0


# ---------------------------------------------------------------------------
# Main script (DB-backed benchmark / exploitability pipeline)
# ---------------------------------------------------------------------------


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

    ts0, data0 = next(iter(fc.ts.items()))
    t0 = time.perf_counter()
    _process_timestep(ts0, data0)
    compile_time = time.perf_counter() - t0
    print(f"JIT compile: {compile_time*1000:.1f} ms  ")




    # 3. Time the full probe sweep. Run a few times — first call may pay
    #    one-off costs (numpy import warmup, allocator settling).
    ts = next(iter(fc.ts))
    sp, sv, su, dp, dv, cum_dv, cp, av = fc.ts[ts]
    mask = su == "Unit 3"
    sp_r, sv_r = sp[~mask], sv[~mask]

    t = Timer(
        lambda: _probe_classify(sp_r, sv_r, dp, dv, 50.0, 1000.0)
    )
    n, total = t.autorange()
    print(f"_probe_classify: {total/n*1e6:.2f} µs per call")
    return market_orders_df, fc


def _main():
    db_uri = "postgresql://assume:assume@localhost:5432/assume"
    db = create_engine(db_uri)

    input_path = "examples/inputs"
    #scenario, study_case = "example_03", "base_case_2019"  # "example_03_naive"  # "example_01a"
    scenario, study_case = "example_03_naive", "base_case_2019"
    #scenario, study_case = "example_01a", "base"

    query = f"SELECT * FROM market_orders where simulation = '{scenario}_{study_case}'"

    df, fc = init_database(db, query)

    results = fc.run_all_probes()
    bench = False
    calc_exploitability = True

    if calc_exploitability:
        world = World(database_uri=db_uri, export_csv_path="")

        load_scenario_folder(
            world,
            inputs_path=input_path,
            scenario=scenario,
            study_case=study_case,
        )

        units = world.units

        per_unit = fc.calculate_exploitability(units, results)
        per_t = fc.aggregate_per_timestep(per_unit)
        total = fc.aggregate_total(per_unit)
        print(f"Exploitability — total: {total:.2f}  "
            f"(mean per timestep: {total / max(len(per_t), 1):.2f})")

        fig = fc.plot_exploitability(per_unit, path="exploitability.png")
        print("Saved plot to exploitability.png")

    if bench:
        import cProfile
        import pstats
        from pstats import SortKey

        profiler = cProfile.Profile()
        profiler.enable()
        results = fc.run_all_probes()
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(20)   # top 20 by cumulative time


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        ok = _run_all_tests()
        sys.exit(0 if ok else 1)
    _main()