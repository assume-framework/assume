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

- data: `runs/data/13-multiagent-actshare/` (same folder — delete or move the
  archived `.npz` first, or the runner will validate them and skip)
- tarball: `runs/exports/run13_<jobid>.tar.gz`
- 6 conditions × 3 seeds, 1200 / 2700 critic updates, cluster, ~1 h per task

| condition | final bid | vs. `RUNS.md` §13 | reward |
|---|---:|---:|---:|
|  |  |  |  |

**Reading:** *(after the batch)*

⚠️ **Caveats:** predates workstream A — the critic slice is still the
current-policy sweep, not the stored-action one (`RUNS.md` correction 17).

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
- img: `runs/img/14-eom-<case>-seed<seed>.png`, `runs/img/14-eom-summary.png`
- tarball: `runs/exports/eom_<jobid>.tar.gz`
- 3 cases × 3 seeds, 100 episodes at `train_freq: 100h`, `lr 1e-3`, recorder at
  **grid 201, 4 observations, every 4th block**

| case | learners | act_dim | final bid | act_share | reward | exploitability |
|---|---:|---:|---:|---:|---:|---:|
| `sb02a` | 1 | 1 |  |  |  |  |
| `sb02b` | 5 | 1 |  |  |  |  |
| `sb02c` | 10 | 1 |  |  |  |  |
| `02a` | 1 | 2 |  |  |  |  |
| `02b` | 5 | 2 |  |  |  |  |
| `02c` | 10 | 2 |  |  |  |  |

**Reading:** *(after the batch)*

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
  difference against run 13 is not a clean single-variable comparison.
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
