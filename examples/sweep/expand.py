#!/usr/bin/env python
"""Expand a sweep.yaml into per-variant scenario folders + a manifest.

For each variant the script:
  1. Deep-copies the base config.
  2. Applies parameter overrides at dotted paths INSIDE the study_case sub-tree.
  3. Writes the patched config to a fresh scenario folder.
  4. Symlinks the base scenario's CSV input files into the new folder.
  5. Appends one row to the manifest TSV.

The SLURM array job uses the manifest — one row per array task.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import sys#
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required. Install with: pip install pyyaml")


def deep_set(d: dict, dotted: str, value) -> None:
    """Assign d[a][b][c] = value for dotted path 'a.b.c'. Creates dicts as needed."""
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def slug(value) -> str:
    """Make a parameter value safe for use inside a folder name."""
    s = str(value)
    for bad in ("/", " ", ":", "\\", "\n", "\t"):
        s = s.replace(bad, "_")
    return s


def build_variants(sweep: dict) -> list[dict]:
    """Return [{name, overrides}, ...] given a sweep spec."""
    mode = sweep.get("mode", "grid")

    if mode == "scenarios":
        scenarios = sweep.get("scenarios") or []
        if not scenarios:
            sys.exit("mode=scenarios but no `scenarios:` entries provided.")
        return [
            {"name": s["name"], "overrides": dict(s.get("params", {}))}
            for s in scenarios
        ]

    params = sweep.get("parameters") or {}
    if not params:
        sys.exit(f"mode={mode} requires a non-empty `parameters:` block.")

    keys = list(params.keys())
    value_lists = [
        params[k] if isinstance(params[k], list) else [params[k]] for k in keys
    ]

    if mode == "grid":
        combos = list(itertools.product(*value_lists))
    elif mode == "zip":
        lengths = {len(v) for v in value_lists}
        if len(lengths) > 1:
            sys.exit(
                f"mode=zip requires all parameter lists to have the same length, "
                f"got lengths {lengths}."
            )
        combos = list(zip(*value_lists))
    else:
        sys.exit(f"Unknown mode: {mode!r}. Use grid, zip, or scenarios.")

    variants = []
    for combo in combos:
        overrides = dict(zip(keys, combo))
        # Build a short, descriptive name from the leaf keys + values
        name_bits = [f"{k.split('.')[-1]}-{slug(v)}" for k, v in overrides.items()]
        variants.append({"name": "__".join(name_bits), "overrides": overrides})
    return variants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="sweep.yaml", help="Path to sweep config")
    args = ap.parse_args()

    sweep_path = Path(args.sweep).resolve()
    sweep = yaml.safe_load(sweep_path.read_text())

    base = sweep["base"]
    out = sweep["output"]
    input_path = Path(base["input_path"]).resolve()
    src_scenario = input_path / base["scenario"]
    config_filename = base.get("config_filename", "config.yaml")
    study_case = base["study_case"]

    src_config_path = src_scenario / config_filename
    if not src_config_path.exists():
        sys.exit(f"Base config not found: {src_config_path}")

    base_config = yaml.safe_load(src_config_path.read_text())
    if study_case not in base_config:
        sys.exit(
            f"Study case {study_case!r} not found in {src_config_path}. "
            f"Top-level keys: {list(base_config.keys())}"
        )

    scenarios_dir = Path(out["scenarios_dir"]).resolve()
    manifest_path = Path(out["manifest"]).resolve()
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    variants = build_variants(sweep)
    if not variants:
        sys.exit("No variants generated.")

    # Union of override keys across all variants (for the manifest header).
    all_param_keys: list[str] = []
    for v in variants:
        for k in v["overrides"]:
            if k not in all_param_keys:
                all_param_keys.append(k)

    with manifest_path.open("w") as mf:
        header = (
            ["array_idx", "run_id", "scenario_dir", "study_case", "simulation_id"]
            + all_param_keys
        )
        mf.write("\t".join(header) + "\n")

        current_time = time.strftime("%Y%m%d_%H%M%S")


        for idx, v in enumerate(variants, start=1):
            variant_name = v["name"] or "default"
            run_id = f"{idx:04d}_{current_time}_{variant_name}"
            variant_dir = scenarios_dir / run_id
            variant_dir.mkdir(parents=True, exist_ok=True)

            # simulation_id encodes the three things you need to trace a row in
            # outputs.db back to a specific run: array_idx, variant scenario,
            # and study case. Edit this format if you want a different layout.
            simulation_id = f"{idx:04d}_{current_time}_{variant_name}_{study_case}"

            # Patch the config (overrides apply inside the study_case sub-tree)
            cfg = copy.deepcopy(base_config)
            for dotted, value in v["overrides"].items():
                deep_set(cfg[study_case], dotted, value)
            cfg[study_case]["simulation_id"] = simulation_id

            (variant_dir / config_filename).write_text(
                yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
            )

            # Symlink every non-config file from the source scenario.
            # CSVs are shared; the only thing that varies per run is config.yaml.
            for f in src_scenario.iterdir():
                if f.name == config_filename:
                    continue
                target = variant_dir / f.name
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(f.resolve())

            row = [str(idx), run_id, str(variant_dir), study_case, simulation_id]
            row += [str(v["overrides"].get(k, "")) for k in all_param_keys]
            mf.write("\t".join(row) + "\n")

    print(f"Generated {len(variants)} variants -> {scenarios_dir}")
    print(f"Manifest: {manifest_path}")
    print()

    max_par = sweep.get("max_parallel")
    array_spec = f"1-{len(variants)}"
    if isinstance(max_par, int) and max_par > 0:
        array_spec += f"%{max_par}"
    print("Submit with:")
    print(f"  sbatch --array={array_spec} run_array.sh {manifest_path}")


if __name__ == "__main__":
    main()