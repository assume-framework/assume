# Handoff — inc-dec RL benchmark

One page. Read this to resume in a fresh session; everything else is detail.

## What this is

A fast surrogate of the `2_nodes_paper_small` inc-dec reward landscape
(`reward_landscape.png`, unit `diesel_0`), used to test which RL algorithms can
find its optimum and why they fail — plus, since run 07, probes that read
ASSUME's **own** MATD3 rather than an SB3 analogue. Closed-form reward, no market
clearing, so a 10 000-step surrogate run takes ~3 min instead of a HiGHS solve
per action.

**If you read one finding, read 17, then 19.** Run 12 answers what runs 07–11 were
circling: on the true reward the critic fails because the action is 1 of 75 of its
input dimensions, and raising that share solves the task 3/3 at the same budget —
the archive's first solve, and without the reward shaping. **Run 13 then takes it
to the multi-agent case and finds the effect is real but is on *rate*, not
feasibility** — with eleven learners even the untouched baseline eventually
descends. Anything in this file phrased as "MATD3 cannot learn X" is a
single-agent statement with a budget attached.

**Code** (tracked): `examples/inputs/2_nodes_paper_small/rl_benchmark/`
**Run log** (tracked): `RUNS.md`, next to this file — describes every run in
order, with its commands, numbers and corrections. **Read it for detail; this file
is only the summary.** It used to live in the archive and was moved because
`examples/outputs` is gitignored.
**Run archive** (gitignored): `examples/outputs/2_nodes_paper_small/rl_benchmark/runs/`
— the `.npz` files and figures `RUNS.md` links to; four headline figures sit at
its top level. Runs 01b, 08, 09 and 12 have no `img/NN-*` file because their
figure *is* one of those four, which is the usual reason a run looks like it has
no plot.

⚠️ **A fresh clone has no starting buffer.**
`learned_strategies/buffers/single_10ep_standard.npz` — 24 KB, SHA256 `5f1b80b4…`,
checksum-guarded by both sweep runners — is gitignored by `.gitignore:142`, and
without it those runners refuse to start. It is a byte-identical copy of
`single_10ep.npz`. **It is recreatable and contains no policy** (no gradient step
runs during collection, and the actor is never queried — the actions are uniform
draws), so a fresh one is statistically equivalent; it is just not bit-identical,
so the guards would need updating and runs 09–12 re-running. `RUNS.md` §9 has the
tracked/not-tracked table and the exact recipe — note it needs a **72 h horizon**
(`end_date: 2019-01-04`), not the 24 h `inc_dec_collect_buffer` currently
inherits, and the file wanted is the one written after **episode 10** (620 = 62 ×
10; running all 11 gives 682).

Needs `gymnasium` + `stable-baselines3` in the `assume` conda env (installed, not
in `pyproject.toml`). Run scripts with `conda run -n assume python`.

## Layout

```
rl_benchmark/
├── HANDOFF.md       this file — the summary
├── RUNS.md          the run log: every run in order, with its numbers
├── _layout.py       sys.path + OUT_DIR + resolve(); every script imports it
├── surrogate/       the closed-form landscape and the Gymnasium env
├── sweeps/          training drivers: run_benchmark.py, td3_stability.py
├── analysis/        reads a recorded run and explains it; makes the figures
├── real_matd3/      probes ASSUME's own MATD3, live or from saved .pt files
│                    (runs 07–12 single-agent, run 13 the `assume_multiagent_*`
│                     quartet for the 11-agent `inc_dec_learning` case)
└── test_rl_benchmark.py   the three things that would fail silently
```

`analysis/critic_coherence.py` owns the observation-disagreement statistics
(`argmax_disagreement`, `argmax_range`). Every run from 10 to 13 imports them from
there; they used to be reimplemented per script with two different definitions,
which is `RUNS.md` correction 14.

Run the tests with

```bash
conda run -n assume python -m pytest \
    examples/inputs/2_nodes_paper_small/rl_benchmark/test_rl_benchmark.py -v
```

Four groups, all covering things that would fail *silently* — the figures would
keep rendering, from the wrong input:

* **`MultiAgentRecorder` against the real `TD3.update_policy`.** One real gradient
  step with real networks, capturing the tensors the critic was actually handed,
  asserting the recorder reproduces the observation assembly (`matd3.py:585-591`),
  the agent-major action ordering, and the swept column — end to end. This is the
  convention run 13's whole archive rests on and it is otherwise pinned only by a
  comment quoting line numbers.
* **The run-13 action-scale lever**, which degenerates to a no-op if
  `CriticTD3.act_dim` ever stops meaning `act_dim * n_agents`.
* **`act_share`**, against the critic input matrix it describes.
* **The coherence statistic and the per-episode transition count.**

No simulation, no archive needed; ~8 s.

`sweeps/run_benchmark.py` owns the house palette — `COLORS`, `INK`, `MUTED` and
the diverging `DIVERGING` ramp used for every signed gradient field. Import
colours from there. `analysis/critic_evolution.py` re-exports `DIVERGING` only
because `descent_window.py` and `real_matd3/assume_film.py` already import it
from that module; defining it there instead would make the import cycle real,
since `critic_evolution` imports `COLORS` from `run_benchmark`.

Every script runs directly from any working directory —
`python analysis/descent_window.py` — because importing `_layout` puts all four
folders on `sys.path`. Defaults resolve archived runs automatically via
`_layout.resolve()`, so the figures redraw with no arguments.

## The landscape

```
b > 49          not dispatched            reward  0        (27 % of action space, flat)
30 <= b <= 49   dispatched, then dec'd    reward (49-b)/100 -> peak +0.190 at b=30
b < 30          dispatched, not dec'd     reward -0.170    (62 % of action space, flat)
```

Optimum sits **on a cliff edge**: 0.36 drop one tick below it. Only 9.5 % of the
action space carries any gradient.

## Established findings

1. **Exploration is not the bottleneck.** 1000 uniform warmup steps put ~10 % of
   samples in the profitable band; with `buffer_size == timesteps` nothing is
   evicted. Same in ASSUME: `buffers/single_10ep_standard.npz` has 620
   transitions, 10 % in band, best in-band reward +0.199.
2. **The critics learn the landscape correctly** in the surrogate — argmax at
   32.7–37.6 vs the true 30.
3. **SB3's tanh actor is what breaks TD3/DDPG.** float32 `tanh` pins to exactly
   1.0 at `z >= 9.011`, so `1-a² = 0` and *every* actor gradient is exactly zero.
   Frozen permanently, 8/8 seeds at exactly 100.00. **ASSUME's `Actor` defaults to
   softsign** (`neural_network_architecture.py:131`) and is not affected.
4. **Softsign unfreezes the actor but does not fix the problem** (run 08). It does
   reliably break the deadlock — 2.1e-2 of headroom survives at bid ~99 and the
   actor climbs back. But at 8 seeds the baseline solves only **3/8**, and the new
   failure mode is **overshooting through the band to −100**, onto the loss shelf.
5. **The result is chaotic, not just noisy** (run 08). Run 05's "31.73 ± 0.13" is
   bit-identically reproducible under its own conditions — it was never a lucky
   draw. But running the same command with **one BLAS thread instead of 14** moves
   seed 1 from +31.60 to **−60.49**. Treat any single-configuration number here as
   a sample from a bimodal distribution.
6. **Step size is the lever, not activation** (run 08). `lr-1e-4` solves **8/8 at
   32.10 ± 0.37**; `batch-128` is second. Both shrink or smooth the actor step.
   Inside a 19-EUR band bounded by a 0.36 cliff, a full-sized Adam step is simply
   too big to stay there — the same scale-invariance that rescues the actor from
   the plateau is what throws it off the cliff.
7. **The escape is a window, not a standing option** (run 06). Three phases, in
   all 12 runs: *(a)* the critic knows only the coarse shape, so `Q` rises
   monotonically with the bid, `argmax Q` = +100, and both actors run to the
   ceiling — following a correct gradient on an incomplete critic; *(b)* the
   plateau slope flips at step **1225–1325 in every run**, opening an unbroken
   descent path for ~600 steps; *(c)* it fragments again, because a *correctly
   learned* flat region has a noise gradient. Softsign crosses in 190–410 actor
   updates; tanh gets the same clean path at the same time and cannot take a step.
8. **SAC needs its entropy floor lowered, and is then the best learner here.**
   Default `target_entropy = -1` forces σ ≈ ±9 EUR, as wide as the whole band.
   With `--ent-coef 0.001` it lands at **31.35 ± 0.31 (+0.177), 4/4 seeds** — the
   tightest result in the archive. At best-known settings (run 01b): SAC 31.35,
   **DDPG softsign 31.86 ± 0.51**, TD3 softsign 1.59 ± 52.26, PPO 100.00. Note
   the ordering: **DDPG is more reliable than TD3 on this landscape**, despite
   having none of TD3's stabilisers.
9. **The window is the landscape's, but spending it is the algorithm's** (run 01b,
   critic row). Across all 12 runs (TD3/DDPG/SAC × 4 seeds) the plateau slope
   flips at **step 1400** — 11/12 plateau-wide, 12/12 measured at the actor's own
   action. Identical for all three algorithms, so the window is a property of the
   landscape and the buffer. What differs is only how fast each actor uses it:
   first probe back inside the band is **SAC 1400–1800, DDPG 1600–1800, TD3
   2000** (and never, for the seed that fails). **`policy_delay = 2` is the
   cause** — TD3 takes half as many actor updates per environment step, so its
   600-step crossing is ~300 actor updates against DDPG's 200–400, both inside
   finding 7's 190–410. Same actor updates, twice the environment steps, against
   a window counted in environment steps. This explains the DDPG > TD3 ordering
   in finding 8 and reframes finding ❌ "`policy_delay` fixes it": run 04 turned
   the knob the wrong way. **Untested: TD3 at `--policy-delay 1` should match
   DDPG.**
10. **A converged critic here is correct exactly where the reward has a gradient.**
    After step 5000, inside the band `[30, 49)` **85–94 %** of grid cells carry the
    true negative slope at median `|dQ/d(bid)| ≈ 9.5e-3`; on both flat regions it
    is `~1.5e-4` and the sign is a coin flip (47–50 % down on `[70,100]`, 51–54 %
    up on the `<30` shelf). So the "fragmentation" of finding 7 phase (c) is the
    flat regions being fitted *correctly*, and the `<30` shelf has **no restoring
    force** — which is why an actor that steps past the band never returns.
11. **A stochastic policy should not aim at 30.** With σ = 1 EUR, centring on the
    optimum earns 0.006 vs 0.163 at the constrained optimum of 32.31 — half the
    samples fall off the cliff. SAC's 32.34 *is* correct behaviour.

## Findings about ASSUME's own MATD3

12. **The budget is the first-order problem** (run 07). Saved optimizer state, all
    three study cases: `critic_optimizer step = 640`, `actor_optimizer step = 80`,
    against the 190–410 actor updates a crossing costs in the surrogate. At that
    budget the *unshaped* case is frozen in phase 1 — `Q1` monotone in the bid,
    `argmax Q` = 100.0, actor at 95.2, `dQ/da` still positive. Not a converged
    critic the actor failed to follow: **the critic never reached the flip.**
13. **On the true reward the critic never leaves phase 1 — the shaping is what
    moves it** (run 09, the films). Over 40 episodes = **2560 critic updates**,
    4× the default budget:

    | film | argmax `Q1` first → last | actor first → last |
    |---|---|---|
    | shaped | 100.0 → **48.8** | 62.9 → **49.0** |
    | unshaped | 100.0 → **100.0** | 61.3 → **94.4** |

    The unshaped field is **mottled noise across the whole bid axis** with no
    coherent sign structure — the critic never develops a preference to follow,
    and the actor parks at the ceiling. The shaped run's field is smooth and
    single-signed and converges to the band's rim. So this is *not* the budget
    of finding 12; it is a different failure, and it is the one that made the
    shaping necessary in the first place.

    Leading explanation, untested: the surrogate is single-context, so its critic
    only learns `Q(a)`; ASSUME's must fit `Q(s, a)` across **548 distinct
    observations** from ~600–1200 transitions, and on a reward that is 90 % flat
    it never resolves the action dependence. The shaping makes `Q` depend on the
    action everywhere, which is a far easier regression — that may be the real
    mechanism behind the cheat, rather than the gradient story of finding 14.
14. **The shaping creates a decoy** (run 07). `learning_strategies.py:1583-1589`
    fires only when `reward <= 0`, so it does not apply inside the band and
    re-enters at full height at bid 49. The shaped landscape has **two local
    maxima — +0.190 at bid 30, and +0.170 at bid 49, where the true reward is
    0.000** — still separated by the same cliff. It does what it was designed to
    do (a permanent ramp replacing run 06's transient window, walking the actor
    down from +100), but it terminates one euro above the band.
15. **Both of finding 13's outcomes reproduce across 6 seeds — but drop the
    "100.0"** (run 10). ASSUME has a real seed knob: `loader_csv.py:555` calls
    `set_random_seed(config.get("seed", 42))` once while the scenario is read,
    and nothing re-seeds after. `assume_training_probe.py --seed` re-applies that
    call after the load; since the CSVs and the forecaster contain no RNG draws,
    what is left downstream is exactly network init, exploration noise and the
    batch draws. 6 seeds per condition, 40 episodes each:

    | condition | argmax `Q1` last | disagreement over the 6 obs | range | actor last | reaches the band at any frame |
    |---|---|---|---|---|---|
    | shaped | 49.5 ± 4.4 | 4.2 | 10.4 | **50.6 ± 3.0** | **6/6** |
    | unshaped | 89.7 ± 10.6 | **24.5** | 56.4 | 94.3 ± 7.0 | **1/6** |

    The two conditions do not overlap at all on final actor bid — shaped
    44.9–54.1, unshaped 79.3–99.2 — so finding 13's *conclusion* holds. Its
    *number* does not: only 2/6 unshaped seeds end at exactly 100.0. The median
    `argmax` is a weak statistic here, because the six probed observations of a
    single unshaped run disagree about the preferred bid by 24.5 EUR between an
    average pair, spanning 56.4 EUR end to end (shaped: 4.2 and 10.4). **Quote the
    disagreement, not the range, whenever comparing across runs** — the two used to
    be computed differently in different scripts and were once compared with each
    other; see `RUNS.md` correction 14 and
    [`analysis/critic_coherence.py`](analysis/critic_coherence.py). Say **the
    unshaped critic never forms a coherent
    preference**, not that it prefers the ceiling. Five of the six shaped seeds
    stop just *above* the band, as finding 14 predicts; one ends inside at 44.9
    for +0.041, which is one seed and not a rate.

    Side result: unshaped seed 42 reproduces run 09's film to **0.119 EUR** over
    all 80 frames while running at one torch thread where run 09 did not. **Run
    08's thread-chaos does not transfer to ASSUME** — treat it as a property of
    the SB3 surrogate until shown otherwise.
16. **No simple configuration change rescues true-reward MATD3; update budget is
    the only lever that visibly helps** (run 11). The broad screen is 30
    configurations × 3 seeds, 40 episodes, starting from the same immutable
    unshaped replay buffer and then growing private per-run buffers. Baseline is
    the requested `lr=1e-3`, 10 gradient steps, batch 128 and policy delay 2
    (`gamma=.99`, `tau=.005`, exploration sigma `.1`):

    | configuration | final actor bid | final true reward | solved |
    |---|---:|---:|---:|
    | baseline | 99.4 ± 0.1 | +0.000 | 0/3 |
    | `lr-1e-4` | 95.5 ± 0.3 | +0.000 | 0/3 |
    | `policy-delay-1` | 99.8 ± 0.1 | +0.000 | 0/3 |
    | **`grad-32`** | **78.7 ± 16.1** (60.7–99.8) | **+0.002 ± 0.003** | 0/3 |
    | `lr-1e-4-grad-32` | 98.2 | +0.000 | 0/3 |

    Every one of the 30 configurations is 0/3 solved. Low learning rate does not
    transfer from the surrogate at 800 critic updates; policy delay 1, gamma 0,
    target smoothing, exploration noise, batch size and tau do not rescue it
    either. Only `grad-32` (2560 critic updates) forms a substantial descending
    critic region and moves actors materially off the ceiling, but it is
    seed-unstable and does not solve the task. Its low-LR interaction removes the
    movement again, so more updates are not interchangeable with smaller steps
    in this horizon.

    All **90/90 final archives** contain Q1/Q2 and both autograd action-gradient
    fields on 401 bids × 6 fixed observations at all 80 training blocks. The
    all-config critic plot shows nearly every field staying positive over most of
    the action range — phase 1 — with `grad-32` the conspicuous but inconsistent
    exception. This is a 3-seed broad screen, not a stability-rate claim, and
    changing gradient steps also changes total optimisation work.
17. **The cause is the action's share of the critic's input, and raising it solves
    the true-reward task** (run 12). The action is **1 of 75 critic inputs**; the
    other 74 carry 97 % of the input variation. Define

        act_share = sd(a) / (sd(a) + sum_j sd(obs_j))      = 0.030 for ASSUME
        var_share = sd(a)^2 / (sd(a)^2 + sum_j sd(obs_j)^2) = 0.055

    (`var_share` is the one with a mechanism: Xavier makes the first-layer weights
    iid across inputs, so contributions to the pre-activation variance add in
    quadrature. Both order every sweep identically.) Live, 40 episodes, true
    reward, run 11's `BASELINE`, 3 seeds:

    | condition | act_share | obs_dim | final bid | true reward | solved |
    |---|---:|---:|---|---:|---:|
    | baseline | 0.030 | 74 | 99.4 ± 0.1 | +0.000 | 0/3 |
    | `foresight-6` | 0.108 | 20 | 63.9 ± 24.2 | +0.012 | 0/3 |
    | `foresight-3` | 0.191 | 11 | 40.4 ± 2.5 | +0.087 | 0/3 |
    | `act-x10` | 0.234 | 74 | −7.4 ± 61.0 | +0.033 | 0/3 |
    | **`act-x30`** | 0.479 | 74 | **33.0 ± 0.2** | **+0.160** | **3/3** |

    `baseline` reproduces run 11's own cell, so the table is the levers. **`act-x30`
    is the first solve in this archive** against 0/90 in run 11, and it lands on
    §6's constrained optimum (32.31 for σ ≈ 1), not on 30. Two mechanically
    unrelated levers — removing observation dimensions, scaling the critic's action
    input — land on one curve, so `act_share` is the variable, not either lever.
    Run 10's 24.5-EUR disagreement between probed observations falls to **1.8**
    (`baseline` here is 21.7, i.e. the same failure as run 10's unshaped).
    `act-x10` is bimodal: 2 seeds converge, seed 42 overshoots to −93.7, which is
    **run 08's failure mode arriving in ASSUME** now that the critic works.
18. **The window it creates is not run 06's transient one** (run 12). Measured with
    `descent_window.py`'s definitions: at `act_share ≥ 0.23` the first unbroken
    descent path appears at update **20** — the second frame — instead of 585, and
    the share of frames carrying one *rises* with `act_share` rather than decaying.
    Phase 1 is shortened, not skipped: peak bid falls from 99.5 to ~74.5. Crossing
    costs 180–580 updates, comfortably inside finding 12's 190–410, because it no
    longer has to be paid inside a closing window. `pulled left` (sign of `dQ1/d(bid)`
    at the actor's own action, last frame) separates converged from still-moving:
    `act-x30` 50 % (a coin flip, i.e. converged), `foresight-3` **100 %** — it was
    still descending at bid 40.4 when the budget ran out.

## Findings about the multi-agent case (run 13)

`inc_dec_learning` — all 11 units of `powerplant_units_learning.csv` learn, 72 h
horizon, so each agent's critic sees **94 observation + 11 action = 105 inputs**.
6 conditions × 3 seeds, true reward, the study case as written.

19. **`act_share` orders the outcome, but it buys rate rather than feasibility.**
    The measured own-action share is **0.016**, half the single-agent 0.030, as
    §7 predicted. Final `diesel_0` bid, by own share, at two budgets:

    | updates | own 0.015 | own 0.024 | own 0.065 / 0.137 |
    |---|---:|---:|---:|
    | 1200 | 97.1 ± 0.2 | 79.2 ± 14.1 | **47.1 ± 1.7** (0.065) |
    | 2700 | 80.1 ± 12.3 | 51.9 ± 7.5 | **29.0 ± 7.4** (0.137) |

    Monotone at each budget separately, and mean reward follows. **But the
    baseline is slow, not stuck**: 97.1 ± 0.2 at 1200 updates, 80.1 ± 12.3 by
    2700, two of three seeds reaching ~70, with a coherent leftward field forming
    from ~2000 updates. `act-all-x2` reaches 79.2 in 1200 updates where the
    baseline needs ~2700 — roughly a **2.25× budget multiplier** for a shift of
    0.015 → 0.024. Runs 09–12's "the critic never forms a preference" does not
    transfer.
20. **It is the *own* action's share, not the action block's.** Run 12's lever
    scales the whole action vector; with N agents that raises everyone together
    and **caps each agent's own share at 1/N** (0.091 here) for any S. Scaling
    only critic *i*'s own action column has no such cap. The two move own share
    and block share in opposite proportions, which separates them: `act-all-x15`
    carries 3.2× `act-own-x15`'s block share (0.728 vs 0.230) and ends 18 EUR
    further out; the block-matched control `act-all-x2` (block 0.302 vs 0.329,
    own 0.027) tracks the ladder by its own share, not its block. **Any principled
    fix inside `CriticTD3` therefore has to be per-agent.**
21. **Run 10's incoherence statistic inverts here — do not carry it over.** The
    best condition (`act-own-x15`) has the *highest* disagreement between probed
    observations about `argmax Q1` (47.7 EUR, mean over the eleven agents at the
    final frame) and the failing baselines the lowest (13.7 at 1200 updates, 21.2
    at 2700), the reverse of run 12's 21.7 → 1.8. With eleven agents the
    critic's preferred bid genuinely should depend on the observation, so
    disagreement stops being evidence of a broken critic. The multi-agent
    baseline's failure is also not run 09's: `diesel_0` parks at 97.1 with
    `argmax Q1` exactly 100.0, but the northern units go to the **floor** and the
    critic's preference is coherent and bang-bang, not incoherent.
22. **Both short conditions are bit-identical prefixes of their long
    counterparts**, on `greedy`, `critic_q`, `critic_grad`, `rewards` and `steps`,
    3/3 seeds each: `baseline-25` == `baseline`[:120 frames] and `act-all-x2` ==
    `act-all-x2-50`[:120 frames]. So nothing depends on `training_episodes` (no
    schedule is active, early stopping off), the runs are deterministic given seed
    and thread count, and the budget-doubling comparison is **the same
    trajectories continued**, not a fresh sample. Confirms finding 15's side
    result: ASSUME does not inherit the surrogate's BLAS-thread chaos.
23. **Fleet reward runs opposite to `diesel_0`'s in every condition.** Best for
    `diesel_0` (`act-all-x15`, +0.526) is second-lowest for the fleet;
    `act-all-x2-50` is lowest at +4.20. Agents compete, so falling bids mean
    falling prices. **"Solved" cannot be defined by fleet profit here**, and the
    closed-form `incdec_reward` landscape does not apply at all — it was derived
    with the rest of the fleet bidding naively. Run 13 reads rewards from each
    run's own buffer for that reason.

## Refuted or revised — do not re-test

- ❌ "The critic smooths the cliff into a ramp." It doesn't; see finding 2.
- ⚠️ "`policy_delay` fixes it." Raising it does not: 2 → 100.0, 8 → 100.0,
  64 → 68 ± 28 (tanh); with softsign, `policy-delay-8` is **0/8**. **But the knob
  is not inert — run 04 turned it the wrong way.** Finding 9: at `policy_delay 2`
  TD3 already crosses the plateau 200–600 environment steps later than DDPG and
  SAC, purely because it takes half as many actor updates per step, and that is
  what loses it the window. Lowering it to 1 remains untested in the surrogate;
  on real ASSUME MATD3 it is now tested and is **0/3** at 40 episodes / 800 critic
  updates (run 11).
- ❌ "The surrogate's `lr-1e-4` result transfers directly to ASSUME." Run 11's
  real-MATD3 screen ends at actor 95.5 ± 0.3 and 0/3 solved. That rejects an easy
  transfer at 40 episodes / 800 critic updates, not a longer-horizon benefit.
- ❌ "A 1e-5 gradient is too small to move the actor." Adam steps by
  `lr·G/(G+eps)`; anything two orders above `eps = 1e-8` gives a full-sized step.
  Only a hard zero kills it.
- ⚠️ "The fragmented gradient field is not the binding constraint, because
  softsign scores *worse* on every path metric and crosses anyway."
  **The evidence was pooled over the wrong interval** — see finding 7 and
  `RUNS.md` correction 6. During the crossing the path is clean 91–100 %
  of the time for *both* activations; the pooled numbers are dominated by the
  post-crossing phase, which for tanh is a consequence of the actor parking at
  the ceiling and feeding the buffer there (after step 2000 the tanh runs place
  100 % of their bids on [50, 100], softsign 0–52 %).
- ⚠️ "Run 06's window predicts a double dissociation: `warmup` should not break
  the result, `policy_delay` and low `lr` should." **Half wrong** (run 08).
  `policy-delay-8` fails as predicted, but `lr-1e-4` is the *best* configuration
  in the sweep and `warmup-3000` is no worse than baseline.
- ⚠️ "Without shaping, ASSUME's `argmax Q1` stays at 100.0." **The conclusion
  survives 6 seeds, the number does not** (run 10, finding 15). 89.7 ± 10.6, and
  only 2/6 seeds end at 100.0. Do not quote a single unshaped `argmax` — quote
  the 24.5 EUR disagreement between probed observations, or the fact that 5/6
  runs never reach the band at all (the sixth grazes it for 13 of 480 cells and
  still ends at 94.4).
- ⚠️ "Any single-configuration result here is a sample from a bimodal
  distribution, because BLAS thread count alone flipped a seed" (finding 5).
  **True of the SB3 surrogate, not shown for ASSUME.** Run 10's unshaped seed 42
  reproduces run 09's film to 0.119 EUR across a change of thread count.
- ❌ "The real critic fails because it must fit `Q(s, a)` across 548 observations
  rather than `Q(a)` at one" (finding 13's leading explanation). **Wrong in that
  form** — run 12 has the controls. The reward is 95 % a function of the bid
  alone; the observation carries no generalising reward information at all
  (leave-one-out 1-NN R² = **−1.07**); an offline critic fed **shuffled**
  observations fails *identically* to one fed the real ones, so it is not real
  contextual structure; and a critic that memorises harder (`obs-x0.1`, train MSE
  → 0) learns the band correctly, so it is not memorisation capacity. The number
  of observations was standing in for the number of input **dimensions**. Keep all
  548 contexts and just weight the action more, and the learner solves it.
  Rerun the controls with `real_matd3/assume_offline_critic.py` — γ = 0 on the
  frozen buffer, no simulation, ~15 min. `shuffled-obs` is the one row that is not
  seed-stable between scripts (the permutation consumes RNG), so quote it as
  "fails like `full-obs`", not by its digits.
- ❌ "Nothing but the update budget helps on the true reward" (finding 16). True of
  every knob in run 11's 30 configurations, and none of them was `act_share`.
  Finding 17 solves it at the *same* 800-update budget.
- ⚠️ "Without the shaping or a raised `act_share`, ASSUME's MATD3 never forms a
  usable action preference" (findings 13, 15, 17). **A single-agent statement, and
  budget-dependent.** With 11 learners the untouched baseline descends from
  97.1 ± 0.2 at 1200 updates to 80.1 ± 12.3 at 2700 (finding 19). Raising
  `act_share` still orders the outcome, but it multiplies the budget rather than
  enabling learning. Attach a budget to any "cannot learn" claim.
- ⚠️ "Scaling the critic's action input raises `act_share`" (finding 17's second
  lever). **Only at N = 1.** Applied to the whole action vector with N agents it
  saturates at 1/N — 0.091 at N = 11 (finding 20). Scale each critic's *own*
  action column instead.

## Things that did not work, and cost time

- **`--train-freq 1h` on this scenario dies** partway through with
  `AssumeException: No rewards were collected during evaluation run`. Leave
  `train_freq` at **12h**. That caps the film resolution at one frame per
  training block (32 critic updates); finer needs a hook inside the gradient loop
  at `matd3.py:510`, which `assume_training_probe.py` exists to avoid.
- **`algorithm.n_updates` is useless as a time axis.** The world — and with it the
  algorithm object — is rebuilt every episode, so the counter restarts: it reads
  32, 64, 32, 64, … The recorder keeps its own cumulative count instead.
- **A fresh `run_learning` deletes `trained_policies_save_path`**
  (`common/utils.py:885`, and `interactive_input` defaults to *yes*). The probe
  forces a separate `learned_strategies/probe_<case>` folder and its own SQLite
  file. Never point it at a folder holding results you want. If the folder already
  exists and there is **no TTY**, `interactive_input` raises `EOFError` rather
  than taking the default, so an unattended rerun dies at startup — delete the
  folder first.
- **`trained_policies_save_path` must be relative** to the scenario inputs path —
  `replace_paths()` prefixes it on every `setup_world()`, so an absolute path
  becomes `.../inputs/2_nodes_paper_small/C:/...` and `os.makedirs` fails.
- **`assume_training_probe.py` refuses to film a multi-agent run**, by design —
  its recorder sweeps `critics(obs, a)` with a single action column. Run 13's
  `MultiAgentRecorder` replaces it through the same monkeypatch route the run 12
  sweep uses (`probe.Recorder = ...`), so `assume_training_probe.py` is untouched
  and runs 09–12 stay reproducible. It builds each critic's observation input the
  way `matd3.py:584-591` does and sweeps agent *i*'s own action with the others
  held at their actors' outputs.
- **Run 13 is memory-bound, not core-bound.** Each 11-agent trial holds ~0.85 GB;
  six concurrent left 1.4 GB free on this 16 GB machine. Four is comfortable.
  Cores are not the constraint — do not raise `--threads` to compensate, since
  every recorded run used one torch thread and run 08 found thread count alone can
  flip an outcome.
- **The single-agent starting buffer cannot seed a multi-agent run** — its arrays
  are shaped `(n, 1, obs_dim)`. Run 13 therefore collects its own 5 exploration
  episodes per trial, so unlike runs 09–12 the trials do not share a start. That
  also means no checksum guard applies and none is needed.
- **16 parallel workers is past the knee.** Throughput saturates near 11
  short-runs/min on this 20-thread machine; 16 single-thread workers each run ~6×
  slower than one solo run. Run 11 also tried 10 long one-thread probes, which
  pushed this 16 GB machine to **97.8 % memory** before producing results. Six
  workers left about 2 GB free, kept at least two CPU cores available, and
  completed the sweep. Budget from memory and throughput, not solo timing.
- **`conda run python -c` rejects newlines** in the command. Write a file.
- **`conda run` buffers the child's stdout to the end**, so a backgrounded run's
  log stays *empty* for the whole run and looks like a hang — `python -u` does not
  help, the buffering is conda's. Pass `--no-capture-output` when you want to
  watch progress, or poll for the output file instead of the log.
- **Two `conda run` calls at once can collide** on `%TEMP%\__conda_tmp_*.txt`
  ("Der Prozess kann nicht auf die Datei zugreifen"). Harmless for a one-off;
  just retry. For anything that launches runs in **parallel** it is not a retry
  problem — call the env interpreter directly instead:
  `C:/Users/finnr/miniconda3/envs/assume/python.exe`. Three concurrent probes
  under `conda run` killed two of three; the same three under the direct
  interpreter all completed, and it also removes conda's output buffering.
  `assume_stability.py` launches its children with `sys.executable` for this
  reason.
- **The shaped and unshaped conditions cannot run concurrently.** The shaping is
  a source edit, not a config flag, so the whole process tree is in one condition
  at a time; run 10's two 6-seed batches had to go serially (29 min + 17 min).
  `assume_stability.py` checks `learning_strategies.py` against `--condition` and
  refuses to start on a mismatch — worth the five lines, since the failure is
  silent and only shows up in the table at the end.
- **`--critic-grid` is nearly free at training time and expensive to add later.**
  It is one extra forward+backward pass on a batch of `N` actions per probe —
  seconds across a whole run — but a run recorded without it cannot be explained
  afterwards without repeating the training. Run 01b was originally recorded
  without it and cost a **~55 min** re-run to get the critic row onto its figure.
  Record `--critic-grid 401` on anything you may later want to explain.
- **Overriding a path-valued `learning_config` entry needs *both* forms.**
  `replace_paths()` prefixes them with the scenario inputs path on every
  `setup_world()`, so `world.scenario_data` must hold the *relative* one; but
  `run_learning` reads the live `learning_role.learning_config` value directly,
  so that one must be absolute. Setting the relative form on both makes
  `run_learning` exit with `no buffer file found` — and, in this environment,
  without printing a traceback, which cost two runs to track down.
- **`run_learning` writes `tensorboard/` and `assume.log` into the current
  working directory**, so running the probes from this folder drops them here.
  Both are gitignored, but they are not wanted output.
- **Concurrent TensorBoard writers can fail before learning completes.** Six run
  11 trials hit an async-writer `FileNotFoundError`; this was a logging race, not
  an algorithm result. `assume_training_probe.py --disable-tensorboard` replaces
  only TensorBoard with a no-op inside that probe process while preserving the
  database address, update cadence and learning dynamics. The runner now uses it.
  The six retries completed; their two-frame partial archives are retained under
  `partial_failures_before_retry/` rather than mixed into the final 90 files.

## Known caveats in the current results

- ⚠️ **The surrogate is not the scenario's reward, and four scripts used it to
  score real runs.** `reward_from_bid` agrees with the frozen buffer's 620 stored
  rewards on **24.8 %** of transitions (MAE 0.038, R² 0.78). The real EOM price
  varies hour to hour — three loss shelves (−0.20/−0.25/−0.30) against the
  surrogate's one (−0.17) — and `diesel_0` costs 68, not 66. **Bids, critics,
  `act_share` and the offline fits are measured and stand; every reconstructed
  "true reward", `regret`, `+0.15 solved` and the `32.31` optimum do not.** Those
  columns now read `recon`. `RUNS.md` §12 tabulates the measured reward beside
  them: run 12's headline **survives**, `act-x30` measuring +0.167 ± 0.005 against
  the reconstructed +0.160. Do not retune `PAPER_SMALL` — the surrogate is exact
  for runs 01–08 by construction. `RUNS.md` correction 15.
- ⚠️ **The evaluation database holds two hours per episode**, 10:00 and 11:00 of
  14 — an unflushed async write at shutdown (`RUNS.md` correction 16). Training is
  unaffected; best-policy selection, early stopping and any measured reward read
  from `rl_params` are an early-hours sample. **The clean fix is to have the
  single-agent `Recorder` snapshot buffer rewards the way `MultiAgentRecorder`
  already does, then re-run** — deferred to a cluster.
- ⚠️ **Run 13's recorded critic field is not matd3's actor objective.**
  `matd3.py:704` holds the other agents at their *stored* actions; the recorder
  holds them at their *current actors'* outputs. A valid critic slice, but the
  window / `pulled left` / coherence readings describe the current joint policy.
  Empty at N = 1, so runs 09–12 are untouched. `RUNS.md` correction 17.
- **Two `ReplayBuffer` defects in `assume/` are open and untriggered** — an
  early-`full` wrap that would sample unwritten zero rows, and episode-boundary
  bootstrapping. No result here is affected (largest buffer 3 450 against a 50 000
  capacity). See `RUNS.md` §8; fixing them is a library change.

- **`assume_probe_unshaped.npz` is mislabelled** — it preloads
  `buffers/single_10ep_gradient.npz`, whose 280 stored rewards are *shaped*, so
  about a third of its training data is shaped. It is kept only for the record;
  **use `assume_probe_unshaped_clean.npz`**, which passes
  `--load-buffer buffers/single_10ep_standard.npz`. The difference is not
  cosmetic: the mislabelled run appears to flip, the clean one never does.
- **`summarize()` in `run_benchmark.py` still prints `hit (<0.5 EUR)` and
  `regret`** against the *deterministic* optimum. Both penalise a correctly
  behaving stochastic policy (finding 11). Replace with expected reward under the
  policy's own spread.
- **All surrogate runs are single-context**: the 74-dim observation is a constant
  vector, so `Q(s,a)` is effectively `Q(a)`. The real scenario varies hour to hour
  — 548 distinct observations in a 620-transition buffer — and that gap is a
  candidate explanation for why the real critic needs far more updates.
  `IncDecEnv(params=[...])` takes a context list for the harder version.
- **The reward shaping is currently commented out** in
  `assume/strategies/learning_strategies.py:1583-1589`, deliberately, so the repo
  is in "true reward" state. Run 10 uncommented it for its shaped batch and
  restored it with `git checkout --`. Runs 11 and 12 both ran in this state, and
  `assume_actshare_sweep.py`'s `preflight()` refuses to start if it is live —
  finding 17 is a claim about the *true* reward and would be worthless otherwise.
- **`argmax Q1` is a brittle summary when `Q` is nearly flat.** Over a 401-point
  grid it is decided by differences of ~1e-2 in `Q`, which is why run 09's
  unshaped film reads 100.0 and its bit-near-identical rerun reads 99.2. Run 10
  reports the spread of `argmax` across probed observations alongside it; use
  both or neither.
- **Run 10 is 6 seeds per condition.** Enough to separate the two conditions
  (they do not overlap on final actor bid at all), not enough for any finer
  claim — in particular the 1/6 shaped seeds ending *inside* the band is one
  seed, not a rate.
- **Run 11 is 3 seeds per configuration and 40 episodes.** Its 0/3 cells are a
  broad-screen result, not stable failure rates. `gradient_steps` configurations
  also receive different total critic-update budgets: baseline 800, `grad-32`
  2560. Compare fixed-update and fixed-exposure designs before attributing the
  difference specifically to gradient steps.
- **Run 12 is 3 seeds per condition on one scenario.** `act-x30`'s 3/3 is the
  archive's first solve, not a success rate; S = 30 is one arbitrary point.
  Reduced `foresight` also discards forecast dimensions — free here, where they
  carry no reward information, and not in general. Recorded `dQ/d(bid)` carries
  the factor S in the action-scale runs, so **signs are comparable across
  conditions and magnitudes are not**; the figure draws each facet on its own
  robust scale for that reason.
- **The debug prints at `matd3.py:618-628` are now commented out** — run 13 did
  that before starting, as this list previously recommended. They drew `th.rand(1)`
  per gradient step per agent, shifting the exploration and batch-sampling stream,
  and their `target_Q_values[0] > 0` branch fires on most steps in the multi-agent
  case. **Runs 09–12 were recorded with them live**, so a bit-identity claim
  against those archives now has to account for their *absence*; run 13 is
  internally consistent. Restore with
  `git checkout -- assume/reinforcement_learning/algorithms/matd3.py`.
- ⚠️ **Run 13 used a working-tree `inc_dec_learning` that is not the committed
  one.** `config.yaml` was modified but uncommitted during the runs: 72 h horizon
  (committed: 5 h), `learning_rate` 1e-4 (0.001), 50 episodes (8), 5 collecting
  (2), `train_freq` 12h (**1h**). The horizon drives everything — 72 h at 12h
  gives 6 blocks/episode, hence 270 frames and 2700 updates — and the committed
  `train_freq: 1h` is the setting that dies with "No rewards were collected".
  `RUNS.md` §13 has the full table. **Restore it before re-running, or commit it.**
- **Run 13's `act_share` figures are measured on each run's final buffer** and are
  lower than the value its lever was set from: `act-own-x15` is 0.197 on the
  collection buffer and reads 0.137 at the end, because the policy concentrates
  and `sd(a)` falls. Run 12 quotes the initialisation value throughout and has the
  same drift unmeasured. Say which one you mean.
- The archive is gitignored; committing it needs an exception or LFS.

## Open

- ~~**Why does the real critic learn nothing from the true reward?**~~ **Answered
  by finding 17** — the action is 1 of 75 critic inputs. Candidate *(a)*, the
  contextual regression, is refuted; candidate *(b)*, coverage, was already
  refuted by finding 1. What remains open is how to fix it *properly*: run 12's
  two levers are experiment monkeypatches, not an API. The principled version is
  input scaling or normalisation inside `CriticTD3` — a library change affecting
  every scenario, so it needs checking against the other examples before it goes
  anywhere near `assume/`.
- **Where does `act_share` saturate, and where does the overshoot stop?** The
  ladder jumps 0.234 → 0.479 with the bimodal `act-x10` in between. Run
  0.28/0.33/0.40 at more seeds to find whether the overshoot region is real or
  seed luck, and test above 0.5 for a downside.
- **Does `foresight-3` solve it with more episodes?** Finding 18 says it was still
  descending at the budget's end (`pulled left` 100 %, bid 40.4). 128 episodes
  would settle it, and it is the cheaper lever of the two.
- ~~**Does raising `act_share` remove the need for the shaping in general?**~~
  **Answered for the multi-agent case by run 13** (findings 19–23): the own share
  does halve at 11 agents and does order the outcome, but the effect is a ~2.25×
  budget multiplier rather than the difference between learning and not learning,
  and uniform action scaling caps at 1/N. What is left open is *why* the budgets
  differ so much between the two settings — the single-agent unshaped critic never
  moved in 2560 updates, the 11-agent one moves by ~2000. More agents means more
  transitions per episode into the shared buffer (62 vs 14 per episode) and a
  non-stationary opponent set; neither has been isolated.
- **How far does run 13's ladder go?** Only `act-own-x15` was run at own share
  0.137, and it ends at 29.0 ± 7.4 — still above the single-agent constrained
  optimum. `act-own-x30` (own share ≈ 0.33) and a 100-episode budget are the two
  obvious next points, and `act-own` above 0.5 is untested for a downside, as it
  is in run 12.
- **Is the multi-agent outcome an equilibrium?** Fleet reward moves *opposite* to
  `diesel_0`'s in every run 13 condition (finding 23). Nothing here distinguishes
  "the agents learned to bid better" from "the agents competed the price down",
  and the archive has no equilibrium analysis at all. This is the gap that most
  limits what run 13 can be used to claim.
- **Why does even the shaped run stop at the band's rim (49) rather than 30?**
  The decoy of finding 14 explains where it stops; whether fixing the decoy is
  enough to get to 30 is untested. Run 10 sharpens the target: all 6 shaped seeds
  *do* dip into the band during the run (best true reward at any frame +0.036 on
  average, best single frame +0.065 at bid ≈ 42.5), they just do not stay. So the
  question is retention inside the band, not reaching it.
- **Fix the shaping decoy.** Untested candidate: make the shaped branch
  continuous with the true reward at the band edges — `(49 - price)/100` above 49
  (zero at 49, −0.51 at 100) and `-0.17 - (30 - price)/100` below 30 — so both
  ramps point *into* the band instead of terminating at its rim.
- **Is `grad-32` a real escape or only delayed movement?** It is run 11's only
  material signal, but 0/3 solve and one seed still ends at 99.8. Extend baseline,
  `grad-20` and `grad-32` to 128 episodes / more seeds, and compare two designs:
  fixed episode exposure with different updates, then fixed total updates with
  different gradient steps. That separates optimisation budget from fresh data.
- **Does low LR help ASSUME only on a longer horizon?** `lr-1e-4` and its
  `grad-32` interaction fail within 40 episodes, but the screen does not match
  update distance. If retried, give it the same actor/critic optimisation budget,
  rather than repeating the already-negative 800-update cell.
- **Does surrogate TD3 at `--policy-delay 1` match DDPG?** Finding 9's surrogate
  hypothesis is still open. Run 11 only settles the real-ASSUME version at the
  baseline horizon: delay 1 is 0/3 and does not create a coherent critic field.

## Commands

```bash
cd examples/inputs/2_nodes_paper_small/rl_benchmark
O=../../../outputs/2_nodes_paper_small/rl_benchmark

# the three headline figures, all redrawn from archived data
# (--critic-seed 3 is TD3's failing seed, which is what the archived figure draws)
python sweeps/run_benchmark.py --replot --critic-seed 3 \
    --results $O/runs/data/01b-best-known/headline_comparison.npz \
    --out    $O/runs/01-algorithms-best-known-settings.png
python sweeps/td3_stability.py --replot \
    --results <runs>/data/08-stability/td3_stability.npz \
               <runs>/data/08-stability/td3_stability_10k.npz
python real_matd3/assume_film.py
python real_matd3/assume_stability.py --report   # run 10, from the archive
python real_matd3/assume_config_sweep.py --phase broad --report-only \
    --critic-out $O/runs/img/11-assume-config-critic-evolution-broad.png  # run 11

# the mechanism figures
python analysis/descent_window.py         # when the descent path is open
python analysis/activation_comparison.py  # tanh vs softsign
python analysis/critic_landscape.py       # final critic vs true reward
python analysis/actor_saturation.py       # regenerates actor_saturation.md

# new runs
python sweeps/td3_stability.py --seeds 8 --workers 15      # ~40 min
python real_matd3/assume_critic_probe.py                   # saved .pt files
python real_matd3/assume_training_probe.py --episodes 40 --label <what-reward>

# run 10's sweep. One condition per invocation: the shaping is a source edit,
# so uncomment learning_strategies.py:1583-1589 between the two and put it back
# afterwards. ~30 min per batch at 6 workers.
python real_matd3/assume_stability.py --condition unshaped --seeds 42 1 2 3 4 5
python real_matd3/assume_stability.py --condition shaped   --seeds 42 1 2 3 4 5

# run 11's broad screen: 30 configs x 3 seeds, 40 episodes, six one-thread
# workers. Each child loads the same checksum-guarded clean buffer and then owns
# its online buffer. The archived full launch manifest is manifest-initial-broad.json.
python real_matd3/assume_config_sweep.py --phase broad --workers 6

# temporal Q1/Q2 gradients plus final twin-critic landscape from any run 11 npz
python real_matd3/assume_run_diagnostics.py --results <run-11.npz> --out <plot.png>

# run 12's act_share ladder: 5 conditions x 3 seeds, 40 episodes, ~15 min each.
# Both levers are patches installed in the child before the scenario loads, so
# assume/ is untouched -- but the shaping must stay commented out and preflight()
# refuses to start otherwise. The truncated buffers it derives are cached next to
# the shared one as buffers/single_10ep_standard_f{k}.npz.
python real_matd3/assume_actshare_sweep.py --workers 5
python real_matd3/assume_actshare_sweep.py --report-only

# run 12's two figures and the descent-window table, from the archive
python real_matd3/assume_actshare_film.py

# run 12's offline gamma=0 fits -- where act_share was found. No simulation, no
# archive needed, ~15 min for all three rounds from the frozen buffer alone.
python real_matd3/assume_offline_critic.py
python real_matd3/assume_offline_critic.py --round conditions   # just round 1

# run 13, the 11-agent case. No shared starting buffer exists at N > 1, so each
# trial collects its own 5 exploration episodes. Memory-bound: ~0.85 GB per trial,
# so --workers 4 on a 16 GB machine. Leave --threads at 1. ~31 min per 25-episode
# trial, ~59 min per 50-episode trial.
python real_matd3/assume_multiagent_actshare.py                  # baseline + act-own-x15
python real_matd3/assume_multiagent_actshare.py \
    --conditions baseline act-own-x15 --seeds 1 2 --workers 3
python real_matd3/assume_multiagent_actshare.py --report-only \
    --conditions baseline act-all-x2 act-all-x15 act-own-x15

# pick S for a target own-action share, from any recorded run's buffer statistics
python real_matd3/assume_multiagent_actshare.py --measure <run.npz> --target 0.2

# run 13's figures, from the archive
python real_matd3/assume_multiagent_grids.py    # critic grid, bid grid, summary
python real_matd3/assume_multiagent_film.py     # pooled four-condition view
python real_matd3/assume_multiagent_window.py   # run 06's window statistics
```

Results and figures always write to the **outputs** folder, never the tracked
input folder.
