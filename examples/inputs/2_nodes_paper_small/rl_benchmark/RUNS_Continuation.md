# Run log, continued — run 14 onwards

Runs 01–13 are closed and live in [`RUNS.md`](RUNS.md). **Everything new goes
here**, cluster runs included. Findings that change the picture get promoted into
[`HANDOFF.md`](HANDOFF.md); this file keeps the numbers and the commands.

Read [`HANDOFF.md`](HANDOFF.md) before adding a run — it carries the forward plan
and the two things most likely to waste a batch: the surrogate is **not** the
simulator's reward (so do not score real runs with it), and `inc_dec_learning`'s
working-tree config differs from the committed one.

**What the next runs are for** (detail in `HANDOFF.md` § Forward plan):

* **A — fix the recording, then re-run.** Record the actor's *actual* objective
  (other agents at stored actions, as `matd3.py:704` does) alongside the current
  sweep, and record rewards from the buffer rather than the two-hours-per-episode
  database. Everything downstream is provisional until this lands, so tag any run
  that predates it.
* **B — literature-grounded critic changes, replacing `act_share`.** Late action
  injection (DDPG) and SimBa (RSNorm + pre-LN residual blocks + post-LN).
  **Screen these offline first** on `real_matd3/assume_offline_critic.py` — 5
  seeds, ~15 min, no simulation, and it already reproduces the live failure. Only
  what survives there is worth a cluster array.
* **C — do the conclusions generalise?** Port the probes to `example_02a`–`02c`
  (plain EOM, no redispatch) and read exploitability alongside them.
  **Exploitability is already on this branch** (ported 2026-08-11, not merged —
  the branch's diff against `main` is ~176 files and deletes test fixtures):
  `assume/reinforcement_learning/exploitability.py` plus the `WriteOutput` hooks
  in `assume/common/outputs.py`, with `test_exploitability.py` and
  `exploitability_two_bid_walkthrough.py` next to this file and the scenario at
  `examples/inputs/exploit_example/`. ⚠️ It required a `world.py` change so
  evaluation episodes write market output at all — see the note below.
  ⚠️ **Exploitability is only correct for a single day-ahead market with no
  storages**, which is why it is read on `example_02a`–`02c` and **not** on the
  inc-dec cases: `WriteOutput._write_exploitability` selects the whole orderbook
  without filtering by market, so EOM and Redispatch bids are grouped by
  `start_time` and cleared together as if they were one auction, and even
  filtered to the EOM a unit's day-ahead profit is not its total profit. The
  SCOPE note at the top of `assume/reinforcement_learning/exploitability.py` has
  the full statement. **Do not read the `exploitability` table off a run 13-style
  run.**

⚠️ **The exploitability port changed when market output is written.**
`world.py`'s `add_market_operator` used to withhold `output_agent_addr` whenever
`learning_mode` **or** `evaluation_mode` was set; it now withholds it only during
*training* episodes. Exploitability needs this — it reads the evaluation
orderbook from `rl_market_orders`, which is empty otherwise — but it means
**evaluation episodes now write full orderbooks**. Three consequences for cluster
runs:

* **Database size grows**, and orderbooks are the largest table. Check
  `assume_db.db` per variant on the first array before scaling up.
* **More work at evaluation shutdown**, which is exactly where correction 16
  (the two-rows-per-episode truncation) already bites. This may worsen it or
  expose it. **Count rows per episode before and after rather than assuming** —
  it is entangled with workstream A.
* **No contamination**: evaluation orders go to `rl_market_orders`, not
  `market_orders`, and rows carry `evaluation_mode` / `eval_episode` tags.
  Training is untouched — these are database writes and consume no torch RNG, so
  runs 09–13 stay comparable on the learning side.

It is isolated in its own commit (`world: write market output during evaluation
episodes`) so it can be reverted without losing the rest.

## House rules

- **One section per run**, numbered continuously from 14, in this shape:
  *Why* (one paragraph — what question forces this run) · *Command* · *Data* (the
  archive path) · *Result* (a table) · *Reading* (2–4 sentences) · *⚠️ Caveats*.
  Copy the template at the bottom.
- **Recompute the numbers from the archived `.npz`**, never from a console log.
  Every table in `RUNS.md` was, which is why its corrections could be made at all.
- **Record `--critic-grid 401`.** It is nearly free at training time and a run
  without it cannot be explained afterwards without retraining.
- **State the budget** in critic updates, not episodes, in any claim about what a
  learner can or cannot do. Run 13 is the reason.
- **Name the recorder version.** Say whether a run predates or postdates
  workstream A, and which critic slice its window/coherence numbers come from.
- **Screen architectures offline before booking cluster time.** The γ = 0 harness
  reproduces the live failure at 1/60th the cost.
- **Corrections go in `RUNS.md` §4**, appended, keeping the existing numbering.
- **Nothing gets edited in [`archive/`](archive/).** It is a frozen snapshot.

---

## Cluster runs (SLURM)

The local machine is the binding constraint on every open question in
`HANDOFF.md` — they are all "more seeds, more episodes, more points on a curve",
which is what a job array is for. Locally, run 13 was memory-bound at ~0.85 GB
per 11-agent trial and 4 concurrent workers on 16 GB; on the cluster each trial
gets its own task and that ceiling disappears.

### No GPUs

ASSUME learning is two small MLPs against a CPU-bound simulator. A GPU buys
nothing here and only lengthens the queue. Use a CPU partition (`cpu_il`, `cpu`,
`highmem`), leave `gres` unset, and take the parallelism from the array width.
Per task, **4 cores and 8 GB** is the right starting point; a 50-episode 11-agent
trial peaked near 0.9 GB locally.

⚠️ **Thread count is not free.** Run 08 found that BLAS thread count alone flipped
a surrogate seed from +31.60 to −60.49. ASSUME itself reproduced bit-identically
across a thread-count change (runs 10, 13), but every archived ASSUME run used
**one torch thread**. If a cluster run is meant to be comparable to the archive,
pin `cpus_per_task: 1` and pass `--threads 1`; if it is a fresh question, use 4
and say so in the run's section.

### The harness

[`examples/sweep/`](../../../sweep/) — see
[`Readme_sweep.md`](../../../sweep/Readme_sweep.md) for the full description.

```bash
# on the cluster, from the repo root
$EDITOR examples/sweep/sweep.yaml          # base scenario, study case, parameters, slurm block
bash    examples/sweep/submit.sh examples/sweep/sweep.yaml
```

`expand.py` materialises one self-contained scenario folder per variant (patched
`config.yaml`, symlinked CSVs), writes `manifest.tsv`, and creates the log
directories. `submit.sh` then submits a single array job with the resources and
log paths from the sweep file, so `run_array.sh` needs no per-sweep editing —
**except `--chdir`, which must point at your repo checkout on the cluster.**

### Where the logs go

Beside the outputs they describe, not in a global `slurm/` folder:

```
examples/outputs/<sweep_name>/
├── 0001_<timestamp>_<variant>/
│   ├── config.yaml        the patched config that produced this run
│   ├── assume_db.db       this variant's own database
│   ├── logs/run.log       this variant's stdout+stderr
│   └── *.csv              symlinks to the source scenario
├── 0002_.../
├── logs/                  the SLURM streams for the array
│   └── sweep-<jobid>_<task>.{out,err}
└── manifest.tsv           the record of what was run
```

So a variant folder is self-contained: config, database and log travel together
and can be copied off the cluster on their own. `output.logs_dir` in `sweep.yaml`
moves the array-level directory if you want it elsewhere; it defaults to
`<scenarios_dir>/logs`. Both directories are created before submission —
**sbatch does not create missing log directories** and a job whose `--output`
path does not exist fails at launch with nothing to read.

### Two kinds of cluster job

**(a) Plain config sweeps of a study case** — learning rate, batch size, episodes,
seeds, `act_share` via config where it is expressible. This is exactly what
`examples/sweep/` is for: one array task per variant, no code involved.

**(b) The probe sweeps** (`real_matd3/assume_*.py`) — these install monkeypatches
before the scenario loads and record critic films, so they cannot be expressed as
a config override. They already have their own `--conditions/--seeds/--workers`
process pool. **Do not give one task the whole sweep and let it fan out**; submit
**one array task per (condition, seed)** with `--workers 1`, so SLURM does the
scheduling and one crashed trial does not take the batch with it.

Two launchers already do this, each in **one call** — see
[`cluster/README.md`](cluster/README.md):

```bash
# run 13 again: 6 conditions x 3 seeds on inc_dec_learning
bash examples/inputs/2_nodes_paper_small/rl_benchmark/cluster/rerun_run13.sh

# workstream C: critic films of the learning units on example_02a-c
bash examples/inputs/2_nodes_paper_small/rl_benchmark/cluster/eom_critic_evolution.sh
```

Each preflights the checkout (shaping commented out, `inc_dec_learning` is run
13's config, log directories exist), submits the array, and chains a collector
job on `afterany` that reruns `--report-only` and the figure scripts and packs
films + logs + `sacct` + figures into **one tarball** under `runs/exports/`. Both
scripts are their own array body and their own collector, so `--chdir` is derived
rather than edited, and both skip a trial whose `.npz` already validates —
resubmitting the same command re-runs only the failures.

### Preflight checklist

Cheap to check, expensive to discover after a 200-task batch:

1. **The reward shaping is commented out** (`learning_strategies.py:1583-1589`)
   unless the run is deliberately shaped. `assume_actshare_sweep.py`'s
   `preflight()` enforces this; a plain config sweep does not.
2. **`inc_dec_learning`'s config is the one you mean.** The committed version is
   a 5 h horizon at `train_freq: 1h`, and `train_freq: 1h` **dies** with
   `No rewards were collected during evaluation run`. Run 13's working-tree table
   is in `RUNS.md` §3. Commit whichever you intend to use — a sweep patches the
   *committed* file on the cluster.
3. **The starting buffer exists** where the runner expects it, with SHA256
   `5f1b80b4…`, if the script guards on it. It is gitignored, so a fresh clone on
   the cluster does **not** have it — copy it across or recreate it (`RUNS.md` §8).
4. **`trained_policies_save_path` does not point at results you want.** A fresh
   `run_learning` rmtree's it, and with no TTY the confirmation raises `EOFError`
   rather than defaulting — so an unattended task dies at startup instead.
5. **`--disable-tensorboard`** on anything parallel. Run 11 lost six trials to a
   concurrent TensorBoard async-writer race that looked like failed learning.
6. **One short test task first.** `sbatch --array=1` (or a `dev_*` partition)
   before the full array. Walltime is per task, not for the array.

### After a batch

- `sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS` — `MaxRSS`
  tells you what to request next time.
- Resubmit only the failures: `sbatch --array=3,7,9 ...` (pass `--output`/`--error`
  explicitly so the retries land beside the rest).
- Audit the file count before believing a table. Run 11's "90/90 complete files,
  30 configs × 3 seeds × 80 frames, all carrying the same buffer checksum" is the
  shape of check that caught its six silent failures.
- Keep `manifest.tsv` with the results. It is the only record of what each
  `simulation_id` means.

---

## Runs

### 13-rerun — run 13 on the cluster, from the committed config

**Why:** run 13's numbers came from a working-tree `inc_dec_learning`; that
config is now committed (`d717a2a5`), the `matd3.py:618-628` debug prints that
were live for runs 09–12 are commented out, and the recorder's `REWARD_WINDOW`
is 69 rather than the 62 inherited from the single-agent case. So the archive's
bid / critic / `act_share` columns should reproduce **bit-identically** and its
`rewards` column should not. That is the check: anything else moving means
something changed that nobody meant to change. It also lifts the local 4-worker
memory ceiling, so all 18 trials run at once.

```bash
bash examples/inputs/2_nodes_paper_small/rl_benchmark/cluster/rerun_run13.sh
```

- data: `runs/data/13-multiagent-actshare-cluster/` (kept **beside** the local
  archive in `13-multiagent-actshare/`, not over it — the comparison is the point)
- tarball: `runs/exports/run13_6262973.tar.gz`
- 6 conditions × 3 seeds, 1200 / 2700 critic updates, cluster job **6262973**,
  18/18 `COMPLETED`, **22–27 min per task**, `MaxRSS` 1.1–1.4 GB (so `--mem 4G`
  was generous; 2 GB is enough)

`diesel_0`'s final bid, median over the 6 probed observations, per seed:

| condition | upd | own share | local archive | cluster rerun | Δ mean |
|---|---:|---:|---|---|---:|
| `baseline-25` | 1200 | 0.015 | 97.3, 97.2, 96.8 → **97.1 ± 0.3** | 97.3, 97.2, 96.8 → **97.1 ± 0.3** | 0.0 |
| `act-all-x2` | 1200 | 0.025 | 81.6, 60.9, 95.3 → 79.2 ± 17.3 | 87.9, 66.6, 95.2 → 83.3 ± 14.8 | +4.1 |
| `act-all-x15` | 1200 | 0.065 | 47.3, 49.1, 44.8 → **47.1 ± 2.1** | 47.8, 48.7, 48.3 → **48.3 ± 0.4** | +1.2 |
| `baseline` | 2700 | 0.013 | 97.3, 73.8, 69.3 → 80.1 ± 15.0 | 97.2, 75.3, 67.6 → 80.0 ± 15.4 | −0.1 |
| `act-all-x2-50` | 2700 | 0.022 | 62.5, 45.6, 47.6 → 51.9 ± 9.2 | 59.7, 43.8, 44.9 → 49.5 ± 8.9 | −2.4 |
| `act-own-x15` | 2700 | 0.141 | 22.1, 39.2, 25.7 → **29.0 ± 9.0** | 18.6, 35.3, 27.6 → **27.2 ± 8.3** | −1.8 |

**Reading — the trajectories do not reproduce, the table does.** Every
`greedy` array differs, by up to **177 EUR** on a 200-wide action range, and the
divergence starts at **frame 0** at float32 rounding scale (max |Δ| ≈ 2e-5 EUR
on the bid, ≈ 1e-7 relative) and amplifies monotonically — 3e-5 at frame 0,
8e-3 by frame 7, tens of EUR by the end. `steps` and `buffer_fill` are
bit-identical, so it is not a budget or schedule difference; the probed
`observations`, taken from the collection episodes *before any gradient step*,
already differ, so the perturbation enters through the simulator (HiGHS, the
forecaster) and torch, not through a different RNG stream. **This learner is
chaotic in the Lyapunov sense on this scenario**, and per-seed bids from a single
machine are not a reproducible quantity.

**What survives is everything the run was claiming.** `act_share` matches to
three decimals in all six conditions (it is a buffer statistic, not a trajectory).
The condition means move by at most 4.1 EUR, all well inside the seed spreads, and
the **ordering is preserved exactly at both budgets** — monotone in the own share,
`act-all-x15` (47.1 → 48.3) and `act-own-x15` (29.0 → 27.2) reproducing tightest,
and `baseline` still landing at 80 for 2.25× `act-all-x2`'s budget. Run 13's
headline is a property of the conditions, not of three lucky seeds.

⚠️ **Caveats:**

- **`RUNS.md` §13's per-seed columns should be read as one draw, not as the
  numbers.** The condition means and the ordering are the reportable part.
- `rewards` differ partly **by design**: `REWARD_WINDOW` went 62 → 69 (the
  archive averaged the last 62 of each episode's 69 transitions). Fleet reward
  moves ≤ 0.4 and the `diesel_0` reward ≤ 0.06; both are consistent with the
  window change plus the trajectory divergence, and neither is separately
  identified here.
- This also **retires the archive's claim that runs 10 and 13 "reproduce
  bit-identically across thread-count changes"** as a general statement. It holds
  within one machine; it does not survive a change of platform. `--threads 1` was
  pinned on both sides, so thread count is not what moved.
- The `world.py` exploitability change (evaluation episodes now write market
  output) is **not** exonerated or implicated by this: platform and code changed
  together. Separating them needs one local re-run on the current tree, which has
  not been done.
- Predates workstream A — the critic slice is still the current-policy sweep, not
  the stored-action one (`RUNS.md` correction 17).
- Figures were **not** produced on the cluster (SB3 missing there); they were
  redrawn locally. See [`cluster/README.md`](cluster/README.md).

---

### 14 — critic evolution on the plain-EOM examples (workstream C)

**Why:** every finding in `RUNS.md` is one redispatch/multi-market scenario with
a 90 %-flat reward. These are the same learner on an ordinary energy-only market
at 1, 5 and 10 learning agents, which is the cheapest test of whether the
`act_share` ordering and the "critic never forms a preference" readings are about
MATD3 or about the inc-dec landscape. It is also the only setting exploitability
is valid on, so both readings come off the same runs.

**Six cases, in two ladders.** `02a`–`02c` are the stock examples with
`EnergyLearningStrategy` (`act_dim` 2: inflexible + flexible block).
`sb02a`–`sb02c` are the same three fleets with
`EnergyLearningSingleBidStrategy` (`act_dim` 1: one bid for the whole
`max_power`), as three study cases of the new folder
`examples/inputs/example_02_single_bid/`, differing only in `powerplant_units`.
Everything else — demand, fuel prices, naive fleet, market — is byte-identical to
the originals, so each pair is an A/B on the bid structure. **The `sb*` trio is
what runs by default**, and it is the cleaner half for both readings: one bid
axis per agent, the same shape runs 09–13 recorded, and one bid per unit is
exactly what the exploitability probe handles with no ordered-bids decomposition.

```bash
# the sb* trio, 9 tasks
bash examples/inputs/2_nodes_paper_small/rl_benchmark/cluster/eom_critic_evolution.sh
# the two-bid originals, or both ladders as one 18-task array
CASES="02a 02b 02c" bash .../cluster/eom_critic_evolution.sh
CASES="02a 02b 02c sb02a sb02b sb02c" bash .../cluster/eom_critic_evolution.sh

# locally, or to redraw from a downloaded tarball
python real_matd3/eom_critic_film.py --report-only
python analysis/eom_critic_evolution.py
```

- data: `runs/data/14-eom-critic-evolution/eom_film_<case>_seed<seed>.npz`
- img: `runs/img/14-eom-<case>-seed<seed>.png`, `runs/img/14-eom-summary.png`,
  `runs/img/14-eom-exploitability.png` (mean ± sd across units, per episode,
  one subplot per case) and `runs/img/14-eom-exploitability-by-regime.png`
- open: **the per-regime critic views are not recorded yet.** The recorder can
  now stratify its probed observations by demand regime
  (`eom_critic_film.py --obs-regimes`, `OBS_REGIMES=1` on the cluster script),
  which is what would show whether one critic holds both equilibria at once —
  but that needs a re-run. Three views are built and smoke-tested against it:
  `--only regime-heatmap` (the critic's field per regime on a shared scale, plus
  the difference — the one that tells a critic failure from an actor failure),
  `--only separation` (actor bid and `argmax Q1` per regime against both
  equilibrium prices) and `--regime each --only film` (the full per-unit film,
  one PNG per regime)
- tarball: `runs/exports/eom_6262975.tar.gz`
- 3 cases × 3 seeds, 100 episodes at `train_freq: 100h`, `lr 1e-3`, **190 frames,
  7600 critic updates**, recorder at **grid 201, 4 observations, every 4th block**
- cluster job **6262975**, 9/9 `COMPLETED`, **9 min** (`sb02a`/`sb02b`) to
  **16 min** (`sb02c`) per task, `MaxRSS` 1.2–1.5 GB
- **marginal cost of every learning unit: 55.7 EUR/MWh** (CCGT, gas 26, CO₂ 25,
  η 0.60, `additional_cost` 4.0), read off the recorded observations
- ⚠️ **the two-bid cases `02a`–`02c` have not been run** — only the single-bid
  trio is below

Final bid = median over units of each unit's median over probed observations.
Exploitability = **learning units only**, first vs. last evaluation episode, EUR/h
per unit, and the same divided by that unit's capacity:

| case | learners | MW each | final bid, per seed | mean | markup | act_share | reward | exploit. ep1 → ep19 | per MW |
|---|---:|---:|---|---:|---:|---:|---:|---|---:|
| `sb02a` | 1 | 2500 | 73.3, 72.7, 72.7 | **72.9** | **+17.2** | 0.0044 | **+0.072** | 29 549 → 4 749–6 744 | 1.9–2.7 |
| `sb02b` | 5 | 500 | 57.0, 55.8, 56.9 | **56.6** | **+0.8** | 0.0101 | −0.002 | 3 793–5 298 → 577–741 | 1.2–1.5 |
| `sb02c` | 10 | 500 | 62.5, 63.4, 64.7 | **63.5** | **+7.8** | 0.0059 | −0.002 | 8 108–10 232 → 132–155 | **0.26–0.31** |

**Reading — MATD3 works here, cleanly, and that is the result.** Nothing in
runs 09–13's failure picture transfers. All nine trials converge, seeds agree to
within **0.6 EUR** on `sb02a` and ~1 EUR on the others (against run 13's 15–17
EUR spreads), and the `sb02a` film is textbook: the actor starts pinned at the
100 cap, crashes through to ~58 by update 1200 as the reward goes from −0.11 to
+0.04, then climbs steadily to ~73 while the reward rises monotonically to
+0.075, tracking the sign flip of `dQ1/d(bid)` — which sits right at the
marginal-cost line. **`act_share` is 0.004–0.010 throughout, an order of
magnitude *below* run 13's failing baseline (0.013–0.015), and the task is solved
anyway.** Whatever `act_share` orders, it is not a general predictor of whether
MATD3 can learn.

**The markup tracks learning capacity, not learner count.** `sb02a`'s single
2500 MW unit is pivotal and marks up +17.2; five 500 MW learners compete each
other down to +0.8, essentially marginal-cost bidding; ten of them go back *up*
to +7.8. Reading the middle column as "more agents ⇒ more competition" is wrong —
what changes from `sb02b` to `sb02c` is that the learners collectively hold
5000 MW instead of 2500 MW against the same naive fleet, so the joint outcome
tilts back toward the pivotal case.

**The stage game has two equilibria and the learners find only one of them.**
Every non-learning unit bids marginal cost, so the merit order is known and the
NE switches with demand — with `C` = 5000 MW of cheap naive capacity below the
learners, `L` their total and `u` one unit's:

| regime | demand | equilibrium | `sb02a` | `sb02b` | `sb02c` |
|---|---|---|---:|---:|---:|
| `idle` | `D ≤ C` | not dispatched | 18 % | 20 % | 20 % |
| `bertrand` | `C < D ≤ C+L−u` | **marginal cost** — an undispatched learner undercuts | — | 67 % | 80 % |
| `pivotal` | `C+L−u < D ≤ C+L` | **backup's marginal cost (85.7)** — the partly-dispatched unit is marginal and irreplaceable | 82 % | 13 % | — |

`sb02a` has `L − u = 0`, so `bertrand` cannot exist: its one unit is marginal
whenever it runs. `sb02c`'s `pivotal` threshold is 9500 MW, which March demand
(4018–7454) never reaches. **`sb02b` is the only case containing both**, which
makes it the one within-run test. Exploitability per learning unit, first to last
evaluation episode (`analysis/eom_exploitability.py --table`):

| case | `idle` | `bertrand` | `pivotal` | all |
|---|---:|---:|---:|---:|
| `sb02a` | 0 → 0 | — | 36 256 → **7 209** (0.20×) | 29 549 → 5 875 |
| `sb02b` | 271 → 4 | 5 240 → **424** (0.08×) | 8 000 → **2 698** (0.34×) | 4 604 → 637 |
| `sb02c` | 11 → 0 | 11 910 → **183** (0.02×) | — | 9 524 → 147 |

**The Bertrand equilibrium is learned; the pivotal one is not.** Within `sb02b`
the same policy ends **6.4× more exploitable in the pivotal hours than in the
Bertrand hours** (2 698 vs 424), and the by-regime figure shows the pivotal trace
**flat from about episode 5** while the Bertrand trace keeps falling. Across
cases the pattern repeats: `sb02c`, which is Bertrand at every hour, gets to
0.02× of where it started; `sb02a`, which is pivotal at every hour, only to
0.20× — and its residual 7 209 EUR/h is almost exactly the markup it failed to
take, `(85.7 − 72.9) × ~1000 MW` averaged over the hours it runs. **Marking up
to marginal cost is easy because a mistake is punished immediately; marking up
to the backup's cost requires finding a price 30 EUR above your own cost with no
local gradient telling you where to stop.**

⚠️ `pivotal` is 13 % of `sb02b`'s hours and the case is n = 3 seeds. The
within-run contrast is the strong part; the cross-case one shares the confound
that `sb02a`'s learner is also 5× larger.

**Exploitability falls monotonically in all nine runs and is the sharpest of the
three statistics.** Per MW of learner capacity it goes 2.3 → 1.3 → 0.29 EUR/MWh
across the cases: the ten-learner case ends **closest to Nash** even though it
bids *further above* marginal cost than the five-learner case. That is the metric
doing its job — it measures distance to best response, not distance to
competitive pricing, and the two separate exactly where market power is real. It
is also the first evidence that the `WriteOutput` hook works end to end on a live
learning run: 19 evaluation episodes, 96 k–226 k rows per trial, no gaps.

⚠️ **Caveats:**

- **The two ladders differ by more than the action count.**
  `EnergyLearningSingleBidStrategy` defaults `foresight` to **24** against the
  two-bid strategy's 12, so its observation is 50-dimensional against 26. That is
  the strategy's own default, not a choice of this scenario — but it means the
  critic's input count moves for two reasons at once, and an `act_share`
  comparison **across** ladders has to say so. Within a ladder it is clean.
- **"final bid" means different things per ladder.** At `act_dim` 2 it is the
  inflexible price (`min` of the two actions, as `calculate_bids` assigns them);
  at `act_dim` 1 there is only one price. The films likewise carry three sweeps
  per agent (`a0`, `a1`, `diag`) in the two-bid cases and one in the single-bid
  cases — named `diag` in both, so the figures' default reads either.
- `02a` and `sb02a` end a day earlier than the other four (2019-03-31 vs
  2019-04-01), inherited from upstream `example_02a` and kept so each pair stays
  a faithful A/B.
- The recorder grid is 201, not the house rule's 401, because 401 × the ten
  agents of `02c` is ~1.5 GB per seed and the results have to be scp-able.
- `lr` is the study cases' own 1e-3, not the 1e-4 runs 11–13 used, so a
  difference against run 13 is not a clean single-variable comparison. **The
  "MATD3 works on a plain EOM" claim is therefore not a single-variable contrast
  with run 13** — scenario, bid structure, learning rate and horizon all differ
  at once. It refutes "MATD3 cannot learn this kind of task"; it does not
  isolate *which* of those the inc-dec failure needs.
- ⚠️ **The frame index is aliased against the calendar, and the reward panel's
  sawtooth was the month, not the learner.** `train_freq: 100h` is snapped by
  `learning_role.sync_train_freq_with_simulation_horizon` to divide the horizon
  evenly — **90h** for `sb02a`'s 30 days, **93h** for the 31-day cases, eight
  blocks either way — and the recorder snapshots every 4th block, i.e. **twice
  per episode**. Since every episode replays the same March, even frames always
  average 03-01→03-16 and odd frames always 03-16→03-31:

  | case | train_freq | blocks/ep | frames/ep | even frames | odd frames |
  |---|---|---:|---:|---|---|
  | `sb02a` | 90h | 8 | 2 | 03-01→03-16, **6026 MW** | 03-16→03-31, **5818 MW** |
  | `sb02b`/`sb02c` | 93h | 8 | 2 | 03-01→03-16 12:00, **5999 MW** | →04-01, **5767 MW** |

  The signature is unmissable once looked for: the reward trace **reverses
  direction on 77–95 % of consecutive frames**, and the even/odd gap is
  systematic and reproduces across all three seeds of each case — `sb02a`
  +0.0083 ± 0.0005, `sb02b` +0.0047 ± 0.0003, `sb02c` −0.0007 ± 0.0002 (the
  ten-learner case is the one that earns slightly *less* in the high-demand
  half). Greedy bids are almost unaffected (±0.3 EUR), as they should be — they
  are probed at fixed observations. `analysis/eom_critic_evolution.py` now
  splits the reward panel by phase and labels each with its window and mean
  demand, and `eom_critic_film.py` records `frame_time`/`frame_episode` so future
  runs do not need the derivation. **`--every 8` would sample one fixed phase and
  `--every 1` all eight; `--every 4` was the one choice that aliases to two.**
- **`sb02b` seed 2 has two of five learners collapsed to −12.6 and −12.7**, well
  below marginal cost and losing money, while the other three converge normally.
  Averages hide it — the per-unit table in the report does not. One failure mode
  survives even here, and it is per-unit, not per-run.
- **Exploitability levels are not comparable across cases** without the per-MW
  column: `sb02a`'s learner is 2500 MW against 500 MW elsewhere. Ratios to
  `best_profit` were tried and are unusable — summed over timesteps the
  denominator goes small or negative and the "gap" exceeds 100 %.
- Exploitability is read at the **evaluation** episodes only (19 of them), and
  `RUNS.md` correction 16 (the two-hours-per-episode truncation of the evaluation
  database) has **not** been re-checked since the `world.py` change. If it still
  bites, these are an early-hours sample of each episode.
- Predates workstream A.

---

<!-- ------------------------------------------------------------------------
### 15 — <short title>

**Why:** what question forces this run, and what the alternative answers would
mean. One paragraph.

```bash
<the exact command, including --workers/--threads and any sbatch array line>
```

- data: `runs/data/15-<name>/`
- img: `runs/img/15-<name>.png`
- <N conditions × M seeds>, <episodes> episodes = **<K> critic updates**,
  <where it ran: local / cluster jobid>, <wall time>

| condition | final bid | measured reward | solved |
|---|---:|---:|---:|
|  |  |  |  |

**Reading:** 2–4 sentences. What it shows, and what it does *not*.

⚠️ **Caveats:** seed count, budget, anything that differed from the archive
(thread count, working-tree config, shaping state).
------------------------------------------------------------------------- -->
