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
scheduling and one crashed trial does not take the batch with it. A minimal array
script for that pattern:

```bash
#!/bin/bash
#SBATCH --job-name=rl_probe
#SBATCH --partition=cpu_il
#SBATCH --time=02:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G
#SBATCH --chdir=<repo checkout>
set -euo pipefail

O=examples/outputs/2_nodes_paper_small/rl_benchmark/runs
LOGS="$O/logs/14-<name>"          # beside the run's data, as above
mkdir -p "$LOGS"

# one line per trial: "<condition> <seed>"
TRIAL=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" trials.txt)
COND=${TRIAL% *}; SEED=${TRIAL#* }

"$HOME/miniconda3/envs/assume/bin/python" \
    examples/inputs/2_nodes_paper_small/rl_benchmark/real_matd3/assume_multiagent_actshare.py \
    --conditions "$COND" --seeds "$SEED" --workers 1 --threads 1 \
    2>&1 | tee -a "$LOGS/${COND}_seed${SEED}.log"
```

Submit with `sbatch --array=1-$(wc -l < trials.txt) --output=$LOGS/%A_%a.out ...`.
Then run the `--report-only` / figure scripts once, locally or in a single task,
against the collected archive.

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

*None yet — run 14 is the next number.*

<!-- ------------------------------------------------------------------------
### 14 — <short title>

**Why:** what question forces this run, and what the alternative answers would
mean. One paragraph.

```bash
<the exact command, including --workers/--threads and any sbatch array line>
```

- data: `runs/data/14-<name>/`
- img: `runs/img/14-<name>.png`
- <N conditions × M seeds>, <episodes> episodes = **<K> critic updates**,
  <where it ran: local / cluster jobid>, <wall time>

| condition | final bid | measured reward | solved |
|---|---:|---:|---:|
|  |  |  |  |

**Reading:** 2–4 sentences. What it shows, and what it does *not*.

⚠️ **Caveats:** seed count, budget, anything that differed from the archive
(thread count, working-tree config, shaping state).
------------------------------------------------------------------------- -->
