# ASSUME sweep helper

Run many config variants as a single SLURM job array. You keep your
`config.yaml` as the source of truth; this layer just patches and submits.

## Files
- `sweep.yaml`    — define your sweep (what to vary, how to combine)
- `expand.py`     — materializes variant scenario folders + a TSV manifest
- `run_array.sh`  — SLURM array script (one task per variant)
- `run_one.py`    — thin wrapper that runs ONE variant — adapt to match your `examples.py`
- `submit.sh`     — one-command: expand + sbatch

Recommended location: put this folder at `<repo root>/hpc_sweep/` so paths in
`run_array.sh` and `submit.sh` resolve cleanly. (Adjust the `--chdir` line in
`run_array.sh` if your repo lives elsewhere.)

## Quick start

```bash
# 1) Edit sweep.yaml — choose mode (grid/zip/scenarios) and parameters
$EDITOR hpc_sweep/sweep.yaml

# 2) Submit
bash hpc_sweep/submit.sh hpc_sweep/sweep.yaml
```

That's it. SLURM gives you ONE array job ID (e.g. `12345_[1-12%8]`).
`sacct -j 12345` shows all tasks; `scancel 12345` kills the whole batch.

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

## How it works

1. `expand.py` reads `sweep.yaml` + your base `config.yaml`.
2. For each variant it deep-copies the base config and applies overrides via
   dotted paths (e.g. `learning_config.learning_rate`) inside the study-case
   sub-tree.
3. It writes each patched config to `<scenarios_dir>/<NNNN__<run_id>>/config.yaml`
   and symlinks every other file from your source scenario (CSVs etc.) into
   that folder. So each variant is a real, self-contained ASSUME scenario.
4. A TSV manifest is written. Each row = one variant = one array task.
5. `sbatch --array=1-N[%K] run_array.sh manifest.tsv` submits everything.
6. Each task reads its manifest row, then calls `run_one.py` with the right
   scenario folder and study case.

## Outputs

Each variant writes to its own folder:
```
examples/inputs/_sweeps/march_sweep/
├── 0001__learning_rate-0.0001__batch_size-128/
│   ├── config.yaml          # patched
│   ├── outputs.db           # this variant's sqlite output
│   └── *.csv                # symlinks to the source scenario
├── 0002__learning_rate-0.0001__batch_size-256/
├── ...
└── manifest.tsv

examples/outputs/slurm/
├── sweep-<jobid>_1.out
├── sweep-<jobid>_1.err
├── sweep-<jobid>_2.out
├── ...
```

The manifest is also a clean record of "what did I run" — keep it next to your
results.

## Useful SLURM tips for sweeps

| Need | Command |
| --- | --- |
| See task status | `squeue -j <jobid> -t all` |
| Resubmit only failed tasks | `sbatch --array=3,7,9 run_array.sh manifest.tsv` |
| Cap concurrency at submit time | `sbatch --array=1-50%8 ...` |
| Kill one task | `scancel <jobid>_<task>` |
| Kill whole sweep | `scancel <jobid>` |
| Detailed exit codes | `sacct -j <jobid> --format=JobID,State,ExitCode` |

## Adapting `run_one.py`

The default `run_simulation()` body uses the standard ASSUME pattern. If your
`examples/examples.py` does anything beyond `World.load_scenario_folder` +
`world.run()` (custom output paths, postprocessing, plotting, etc.), copy that
logic into `run_one.py` so every sweep variant gets the same treatment.

## Extending later

- **Other config files to override** — pass `--sweep <other.yaml>` to use a
  different sweep config in the same folder.
- **Random seeds as a parameter** — just add `seed: [1, 2, 3, 4, 5]` under
  `parameters:` (works in grid mode for replicates).
- **Different walltimes per variant** — split into multiple sweep files; each
  `submit.sh` call can set `--time` via `sbatch --time=... ...`.
- **Result collection** — iterate `manifest.tsv` and load each `outputs.db`.