# ASSUME sweep helper

Run many config variants as a single SLURM job array. You keep your
`config.yaml` as the source of truth; this layer just patches and submits.

**No GPU is needed.** ASSUME learning is two small MLPs against a CPU-bound
simulator — a GPU buys nothing and only lengthens the queue. Use a CPU partition
and get your parallelism from the array width.

## Files

| file | role |
| --- | --- |
| `sweep.yaml` | define the sweep: what to vary, how to combine, where output goes, what resources each task gets |
| `expand.py` | materializes variant scenario folders + a TSV manifest + the log dirs |
| `run_array.sh` | SLURM array script — one task per variant |
| `run_simulation.py` | thin wrapper that runs ONE variant — adapt it to match your `examples.py` |
| `submit.sh` | one command: expand + sbatch, with resources and log paths read from `sweep.yaml` |

This folder lives at `examples/sweep/` and the paths in `submit.sh` and
`expand.py` are resolved relative to it, so it works in place. The **one** thing
you must edit in `run_array.sh` is `--chdir`, which has to point at your repo
checkout on the cluster.

## Quick start

```bash
# 1) Edit sweep.yaml — base scenario, study case, mode, parameters, resources
$EDITOR examples/sweep/sweep.yaml

# 2) Submit
bash examples/sweep/submit.sh examples/sweep/sweep.yaml
```

SLURM gives you ONE array job ID (e.g. `12345_[1-12%8]`). `sacct -j 12345` shows
all tasks; `scancel 12345` kills the whole batch.

## Three modes

```yaml
# grid: cartesian product (3 * 2 * 1 = 6 variants)
mode: grid
parameters:
  learning_config.learning_rate: [0.0001, 0.0005, 0.001]
  learning_config.batch_size:    [128, 256]
  learning_config.algorithm:     [matd3]

# zip: lockstep, same length required (3 variants, paired)
mode: zip
parameters:
  learning_config.learning_rate: [0.0001, 0.0005, 0.001]
  learning_config.batch_size:    [64,    128,    256]

# scenarios: explicit named runs, each with its own overrides
mode: scenarios
scenarios:
  - name: ppo_baseline
    params:
      learning_config.algorithm: ppo
      learning_config.learning_rate: 0.0003
```

`seed` is a top-level study-case key, so replicates are just another parameter:
`seed: [1, 2, 3, 4, 5]` in grid mode.

## How it works

1. `expand.py` reads `sweep.yaml` + your base `config.yaml`.
2. For each variant it deep-copies the base config and applies overrides via
   dotted paths (e.g. `learning_config.learning_rate`) inside the study-case
   sub-tree, and sets a `simulation_id` that encodes array index, timestamp,
   variant name and study case.
3. It writes each patched config to `<scenarios_dir>/<NNNN>_<timestamp>_<name>/config.yaml`
   and symlinks every other file from your source scenario (CSVs etc.) into
   that folder. Each variant is a real, self-contained ASSUME scenario.
4. A TSV manifest is written — one row per variant, one array task per row. Keep
   it next to your results; it is the record of what you ran.
5. `submit.sh` calls `sbatch --array=1-N[%K]` with the log paths and resources
   from `sweep.yaml`.
6. Each task reads its manifest row and calls `run_simulation.py` with the right
   scenario folder and study case.

## Outputs

Everything a run produced — its config, its database and its log — lands in one
folder, so a variant can be inspected or copied on its own:

```
examples/outputs/MY_SCENARIO_OUTPUT/
├── 0001_20260811_142300__learning_rate-0.0001__batch_size-128/
│   ├── config.yaml          # patched
│   ├── assume_db.db         # this variant's sqlite output
│   ├── logs/run.log         # this variant's stdout+stderr
│   └── *.csv                # symlinks to the source scenario
├── 0002_.../
├── logs/                    # the SLURM streams for the array
│   ├── sweep-<jobid>_1.out
│   ├── sweep-<jobid>_1.err
│   └── ...
└── manifest.tsv
```

The array-level log directory is `output.logs_dir` in `sweep.yaml` and defaults
to `<scenarios_dir>/logs`. Both directories are created by `expand.py` before
submission — sbatch does **not** create missing log directories, and a job whose
`--output` path does not exist fails silently at launch.

## Resources

The `slurm:` block in `sweep.yaml` is passed straight to sbatch by `submit.sh`
and overrides the `#SBATCH` defaults in `run_array.sh`:

```yaml
slurm:
  partition: cpu_il
  time: "04:00:00"     # per task, not for the whole array
  cpus_per_task: 4
  mem: 8G
  # gres: gpu:1        # leave commented out
```

`OMP_NUM_THREADS` is set to `cpus_per_task` inside `run_array.sh`. If you are
reproducing a run whose result depends on thread count, pin `cpus_per_task: 1`.

## Useful SLURM commands

| Need | Command |
| --- | --- |
| See task status | `squeue -j <jobid> -t all` |
| Detailed exit codes | `sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS` |
| Resubmit only failed tasks | `sbatch --array=3,7,9 run_array.sh manifest.tsv` |
| Cap concurrency at submit time | `sbatch --array=1-50%8 ...` |
| Kill one task | `scancel <jobid>_<task>` |
| Kill whole sweep | `scancel <jobid>` |
| Which variant was task 7? | `awk -F'\t' '$1==7' manifest.tsv` |

Note that a manual `sbatch --array=3,7,9 run_array.sh manifest.tsv` uses
`run_array.sh`'s fallback `#SBATCH` log paths, not the sweep's. Pass
`--output`/`--error` explicitly if you want the retries beside the rest.

## Adapting `run_simulation.py`

Its `run_simulation()` body is the standard ASSUME pattern —
`World(...)` → `load_scenario_folder(...)` → `run_learning(...)` if
`world.learning_mode` → `world.run()`. If your `examples/examples.py` does
anything beyond that (custom output paths, postprocessing, plotting), copy that
logic in here so every sweep variant gets the same treatment.

## Extending later

- **A different sweep file** — `bash submit.sh path/to/other.yaml`.
- **Different walltimes per variant** — split into multiple sweep files.
- **Result collection** — iterate `manifest.tsv` and load each `assume_db.db`;
  `simulation_id` ties every row back to its array task.
