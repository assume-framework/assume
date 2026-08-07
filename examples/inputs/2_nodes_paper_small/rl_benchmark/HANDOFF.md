# Handoff — inc-dec RL benchmark

One page. Read this to resume in a fresh session; everything else is detail.

## What this is

A fast surrogate of the `2_nodes_paper_small` inc-dec reward landscape
(`reward_landscape.png`, unit `diesel_0`), used to test which RL algorithms can
find its optimum and why they fail — plus, since run 07, probes that read
ASSUME's **own** MATD3 rather than an SB3 analogue. Closed-form reward, no market
clearing, so a 10 000-step surrogate run takes ~3 min instead of a HiGHS solve
per action.

**Code** (tracked): `examples/inputs/2_nodes_paper_small/rl_benchmark/`
**Run archive** (gitignored): `examples/outputs/2_nodes_paper_small/rl_benchmark/runs/`
— three headline figures sit at its top level; `README.md` there describes every
run in order, with the commands.

Needs `gymnasium` + `stable-baselines3` in the `assume` conda env (installed, not
in `pyproject.toml`). Run scripts with `conda run -n assume python`.

## Layout

```
rl_benchmark/
├── _layout.py       sys.path + OUT_DIR + resolve(); every script imports it
├── surrogate/       the closed-form landscape and the Gymnasium env
├── sweeps/          training drivers: run_benchmark.py, td3_stability.py
├── analysis/        reads a recorded run and explains it; makes the figures
└── real_matd3/      probes ASSUME's own MATD3, live or from saved .pt files
```

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
9. **A stochastic policy should not aim at 30.** With σ = 1 EUR, centring on the
   optimum earns 0.006 vs 0.163 at the constrained optimum of 32.31 — half the
   samples fall off the cliff. SAC's 32.34 *is* correct behaviour.

## Findings about ASSUME's own MATD3

10. **The budget is the first-order problem** (run 07). Saved optimizer state, all
    three study cases: `critic_optimizer step = 640`, `actor_optimizer step = 80`,
    against the 190–410 actor updates a crossing costs in the surrogate. At that
    budget the *unshaped* case is frozen in phase 1 — `Q1` monotone in the bid,
    `argmax Q` = 100.0, actor at 95.2, `dQ/da` still positive. Not a converged
    critic the actor failed to follow: **the critic never reached the flip.**
11. **On the true reward the critic never leaves phase 1 — the shaping is what
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
    of finding 10; it is a different failure, and it is the one that made the
    shaping necessary in the first place.

    Leading explanation, untested: the surrogate is single-context, so its critic
    only learns `Q(a)`; ASSUME's must fit `Q(s, a)` across **548 distinct
    observations** from ~600–1200 transitions, and on a reward that is 90 % flat
    it never resolves the action dependence. The shaping makes `Q` depend on the
    action everywhere, which is a far easier regression — that may be the real
    mechanism behind the cheat, rather than the gradient story of finding 12.
12. **The shaping creates a decoy** (run 07). `learning_strategies.py:1583-1589`
    fires only when `reward <= 0`, so it does not apply inside the band and
    re-enters at full height at bid 49. The shaped landscape has **two local
    maxima — +0.190 at bid 30, and +0.170 at bid 49, where the true reward is
    0.000** — still separated by the same cliff. It does what it was designed to
    do (a permanent ramp replacing run 06's transient window, walking the actor
    down from +100), but it terminates one euro above the band.

## Refuted or revised — do not re-test

- ❌ "The critic smooths the cliff into a ramp." It doesn't; see finding 2.
- ❌ "`policy_delay` fixes it." 2 → 100.0, 8 → 100.0, 64 → 68 ± 28 (tanh); with
  softsign, `policy-delay-8` is **0/8**.
- ❌ "A 1e-5 gradient is too small to move the actor." Adam steps by
  `lr·G/(G+eps)`; anything two orders above `eps = 1e-8` gives a full-sized step.
  Only a hard zero kills it.
- ⚠️ "The fragmented gradient field is not the binding constraint, because
  softsign scores *worse* on every path metric and crosses anyway."
  **The evidence was pooled over the wrong interval** — see finding 7 and
  `runs/README.md` correction 6. During the crossing the path is clean 91–100 %
  of the time for *both* activations; the pooled numbers are dominated by the
  post-crossing phase, which for tanh is a consequence of the actor parking at
  the ceiling and feeding the buffer there (after step 2000 the tanh runs place
  100 % of their bids on [50, 100], softsign 0–52 %).
- ⚠️ "Run 06's window predicts a double dissociation: `warmup` should not break
  the result, `policy_delay` and low `lr` should." **Half wrong** (run 08).
  `policy-delay-8` fails as predicted, but `lr-1e-4` is the *best* configuration
  in the sweep and `warmup-3000` is no worse than baseline.

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
  file. Never point it at a folder holding results you want.
- **`trained_policies_save_path` must be relative** to the scenario inputs path —
  `replace_paths()` prefixes it on every `setup_world()`, so an absolute path
  becomes `.../inputs/2_nodes_paper_small/C:/...` and `os.makedirs` fails.
- **16 parallel workers is past the knee.** Throughput saturates near 11
  short-runs/min on this 20-thread machine; 16 single-thread workers each run ~6×
  slower than one solo run. Budget from throughput, not from the solo timing.
- **`conda run python -c` rejects newlines** in the command. Write a file.
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

## Known caveats in the current results

- **`assume_probe_unshaped.npz` is mislabelled** — it preloads
  `buffers/single_10ep_gradient.npz`, whose 280 stored rewards are *shaped*, so
  about a third of its training data is shaped. It is kept only for the record;
  **use `assume_probe_unshaped_clean.npz`**, which passes
  `--load-buffer buffers/single_10ep_standard.npz`. The difference is not
  cosmetic: the mislabelled run appears to flip, the clean one never does.
- **`summarize()` in `run_benchmark.py` still prints `hit (<0.5 EUR)` and
  `regret`** against the *deterministic* optimum. Both penalise a correctly
  behaving stochastic policy (finding 9). Replace with expected reward under the
  policy's own spread.
- **All surrogate runs are single-context**: the 74-dim observation is a constant
  vector, so `Q(s,a)` is effectively `Q(a)`. The real scenario varies hour to hour
  — 548 distinct observations in a 620-transition buffer — and that gap is a
  candidate explanation for why the real critic needs far more updates.
  `IncDecEnv(params=[...])` takes a context list for the harder version.
- **The reward shaping is currently commented out** in
  `assume/strategies/learning_strategies.py:1583-1589`, deliberately, so the repo
  is in "true reward" state.
- The archive is gitignored; committing it needs an exception or LFS.

## Open

- **Why does the real critic learn nothing from the true reward?** Finding 11 —
  the single most important open question, since it is what forced the shaping.
  Two testable candidates: *(a)* the contextual regression is the problem — run
  the surrogate with `IncDecEnv(params=[...])` so it too must fit `Q(s, a)` over
  many contexts, and see whether its flip disappears; *(b)* it is coverage — seed
  the buffer with in-band transitions and see whether `argmax Q1` moves off the
  ceiling.
- **Why does even the shaped run stop at the band's rim (49) rather than 30?**
  The decoy of finding 12 explains where it stops; whether fixing the decoy is
  enough to get to 30 is untested.
- **Fix the shaping decoy.** Untested candidate: make the shaped branch
  continuous with the true reward at the band edges — `(49 - price)/100` above 49
  (zero at 49, −0.51 at 100) and `-0.17 - (30 - price)/100` below 30 — so both
  ramps point *into* the band instead of terminating at its rim.
- **Does `lr-1e-4` transfer to ASSUME?** It is the only stable setting found
  (finding 6) and has not been tried on the real MATD3.

## Commands

```bash
cd examples/inputs/2_nodes_paper_small/rl_benchmark
O=../../../outputs/2_nodes_paper_small/rl_benchmark

# the three headline figures, all redrawn from archived data
python sweeps/run_benchmark.py --replot     --results $O/runs/data/01b-best-known/headline_comparison.npz
python sweeps/td3_stability.py --replot \
    --results <runs>/data/08-stability/td3_stability.npz \
               <runs>/data/08-stability/td3_stability_10k.npz
python real_matd3/assume_film.py

# the mechanism figures
python analysis/descent_window.py         # when the descent path is open
python analysis/activation_comparison.py  # tanh vs softsign
python analysis/critic_landscape.py       # final critic vs true reward
python analysis/actor_saturation.py       # regenerates actor_saturation.md

# new runs
python sweeps/td3_stability.py --seeds 8 --workers 15      # ~40 min
python real_matd3/assume_critic_probe.py                   # saved .pt files
python real_matd3/assume_training_probe.py --episodes 40 --label <what-reward>
```

Results and figures always write to the **outputs** folder, never the tracked
input folder.
