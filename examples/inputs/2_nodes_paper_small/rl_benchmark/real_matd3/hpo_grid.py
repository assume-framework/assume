# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
The hyperparameter grid both HPO studies run, defined once.

Two hosts sweep the same cells, so the two results are directly comparable:

* **inc-dec** -- ``inc_dec_learning_single``, one agent on the synthetic
  inc-dec landscape, through ``real_matd3/assume_arch_sweep.py --round hpo``
* **p1** -- ``example_02_single_bid`` study case ``p1``, five learners on a
  real EOM, through ``real_matd3/eom_critic_film.py --hp <cell>``

A coordinate sweep, not a full cross
------------------------------------
Four axes at 3-4 values each is 108 cells crossed and 20 as a coordinate sweep,
and the crossed version is not 5x more informative here: nothing in runs 09-17
suggests these four *interact* -- the failure they are being pointed at is a
critic that never develops a slope, which is an optimization-landscape problem,
not a schedule-tuning one. So each axis moves alone from the study case's own
setting, and a cell that looks live is worth crossing afterwards on its own.

The one axis that is genuinely 2-D is learning rate against its schedule -- a
schedule *is* a statement about the learning rate -- so those two are crossed:
3 rates x {const, linear, cosine} = 9 cells, which is the grid the request
asked for.

What each cell is relative to
-----------------------------
The study case's own ``learning_config``, which for **both** hosts is
``learning_rate 1e-3`` with **no** schedule. So ``lr1e-3-const`` reproduces
``default`` exactly and is the grid's internal control -- a 3x3 grid read with
a hole where its centre should be is harder to read than one redundant cell.

The two hosts differ in ``batch_size`` (inc-dec 256, p1 128) and in
``policy_delay`` (inc-dec 8, p1 the ``LearningConfig`` default of 2), so the
``bs*`` and ``pd*`` cells are absolute values rather than multipliers and the
cell that matches a host's own setting is named in ``baseline_cell()``.

Weight decay is the axis with no config field
---------------------------------------------
``matd3.py`` constructs ``AdamW(params, lr=...)`` and passes nothing else, so
every run in this benchmark's archive trained with torch's default
``weight_decay = 0.01`` -- not with none. ``default`` therefore carries 0.01,
and the axis reaches down to 0 and up to 0.1. Applied by the monkeypatch in
``optim_patches.py``; see its docstring for why it is not a ``LearningConfig``
change.
"""

from __future__ import annotations

from optim_patches import TORCH_DEFAULT_WEIGHT_DECAY

#: the three learning rates, bracketing both study cases' own 1e-3
LRS: list[float] = [3e-4, 1e-3, 3e-3]

#: ASSUME's schedules. ``None`` is a constant rate; the other two decay toward
#: ``min_learning_rate`` over the training run (``learning_role.py:87-99``).
SCHEDULES: list[tuple[str, str | None]] = [
    ("const", None), ("linear", "linear"), ("cosine", "cosine"),
]

BATCH_SIZES: list[int] = [64, 128, 256, 512]
POLICY_DELAYS: list[int] = [1, 2, 4, 8]
WEIGHT_DECAYS: list[float] = [0.0, TORCH_DEFAULT_WEIGHT_DECAY, 0.1]

#: how far a decaying schedule is allowed to fall. A tenth of the starting rate
#: rather than ASSUME's default of 0, because a schedule that reaches exactly
#: zero stops learning before the last episodes are recorded and the film's
#: final frames would then show a frozen critic rather than a converged one.
MIN_LR_FRACTION = 0.1


def _cells() -> dict[str, dict]:
    cells: dict[str, dict] = {
        "default": {"overrides": {}, "weight_decay": TORCH_DEFAULT_WEIGHT_DECAY,
                    "axis": "centre"},
    }

    for lr in LRS:
        for label, sched in SCHEDULES:
            over: dict = {"learning_rate": lr, "learning_rate_schedule": sched}
            if sched is not None:
                # rounded: the value is written into a JSON command line and
                # into every log, and 2.9999999999999997e-05 reads as a bug
                over["min_learning_rate"] = float(f"{lr * MIN_LR_FRACTION:.3g}")
            cells[f"lr{lr:g}-{label}"] = {
                "overrides": over,
                "weight_decay": TORCH_DEFAULT_WEIGHT_DECAY,
                "axis": "lr",
            }

    for bs in BATCH_SIZES:
        cells[f"bs{bs}"] = {"overrides": {"batch_size": bs},
                            "weight_decay": TORCH_DEFAULT_WEIGHT_DECAY,
                            "axis": "batch"}

    for pd in POLICY_DELAYS:
        cells[f"pd{pd}"] = {"overrides": {"policy_delay": pd},
                            "weight_decay": TORCH_DEFAULT_WEIGHT_DECAY,
                            "axis": "delay"}

    for wd in WEIGHT_DECAYS:
        if wd == TORCH_DEFAULT_WEIGHT_DECAY:
            continue                      # that is 'default'
        cells[f"wd{wd:g}"] = {"overrides": {}, "weight_decay": wd, "axis": "wd"}

    return cells


#: every cell, keyed by name. ``overrides`` are ``LearningConfig`` fields for
#: ``assume_training_probe.py --overrides-json``; ``weight_decay`` goes through
#: ``optim_patches.install_weight_decay``.
CELLS: dict[str, dict] = _cells()

#: cells grouped by the axis they move, so one axis can be submitted alone
GROUPS: dict[str, list[str]] = {
    axis: [n for n, c in CELLS.items() if c["axis"] == axis]
    for axis in ("centre", "lr", "batch", "delay", "wd")
}


def resolve(names: list[str]) -> list[str]:
    """Expand group names (``lr``, ``batch``, ``all``, ...) to cell names.

    Lets a cluster script say ``CELLS=lr`` for the nine-cell grid or
    ``CELLS="default bs512"`` for two specific ones, without the caller having
    to know which is which.
    """
    out: list[str] = []
    for name in names:
        if name == "all":
            out.extend(CELLS)
        elif name in GROUPS:
            out.extend(GROUPS[name])
        elif name in CELLS:
            out.append(name)
        else:
            raise SystemExit(
                f"unknown hyperparameter cell or group {name!r}; cells: "
                f"{', '.join(CELLS)}; groups: {', '.join(GROUPS)}, all"
            )
    return list(dict.fromkeys(out))       # de-duplicate, keep order


def describe(name: str) -> str:
    """One line for a table header or a log."""
    cell = CELLS[name]
    bits = [f"{k}={v!r}" for k, v in cell["overrides"].items()]
    if cell["weight_decay"] != TORCH_DEFAULT_WEIGHT_DECAY:
        bits.append(f"weight_decay={cell['weight_decay']}")
    return ", ".join(bits) or "the study case's own settings, unchanged"


def baseline_cell(batch_size: int, policy_delay: int) -> str | None:
    """Which ``bs*``/``pd*`` cell reproduces a host's own configuration.

    Worth printing: on inc-dec ``bs256`` and ``pd8`` are re-runs of ``default``
    at a different seed-independent path, so agreement between them is a check
    that the sweep machinery is not itself changing the run, and disagreement
    is a bug rather than a result.
    """
    names = [n for n, want in ((f"bs{batch_size}", batch_size),
                               (f"pd{policy_delay}", policy_delay))
             if n in CELLS]
    return " and ".join(names) if names else None


if __name__ == "__main__":
    print(f"{len(CELLS)} cells\n")
    for axis, names in GROUPS.items():
        print(f"  {axis} ({len(names)})")
        for n in names:
            print(f"    {n:<16} {describe(n)}")
