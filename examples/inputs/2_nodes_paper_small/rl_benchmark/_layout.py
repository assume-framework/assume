# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Where things live, and how the scripts find each other.

This benchmark is a set of scripts that import one another, grouped into folders
by role rather than packaged. Importing this module puts every one of those
folders on ``sys.path``, so any script runs directly::

    python analysis/descent_window.py
    python sweeps/run_benchmark.py --algos TD3

from any working directory, with no install and no ``PYTHONPATH``. Each script
bootstraps with two lines::

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _layout import OUT_DIR  # noqa: E402

The folders
-----------
``surrogate/``   the closed-form landscape and the Gymnasium environment
``sweeps/``      training drivers -- one run, or a configuration sweep
``analysis/``    reads a recorded run and explains it; produces the figures
``real_matd3/``  probes ASSUME's own MATD3 rather than an SB3 analogue

Results never go to any of them: they go to ``OUT_DIR``, in the scenario's
*outputs* folder, which is gitignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: the benchmark's own root -- the folder holding this file
ROOT = Path(__file__).resolve().parent

#: role folders, all of which go on sys.path so the scripts can import flat
FOLDERS = ("surrogate", "sweeps", "analysis", "real_matd3")

for _folder in FOLDERS:
    _path = str(ROOT / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

#: the scenario this benchmark is a fast surrogate of
SCENARIO = ROOT.parent

#: Where every result and figure is written. The code is tracked in the
#: scenario's *inputs* folder; results belong in *outputs*, which is gitignored,
#: so a run never dirties a tracked directory.
OUT_DIR = SCENARIO.parents[1] / "outputs" / SCENARIO.name / "rl_benchmark"

#: The archive of recorded runs, with its own README describing each one.
RUNS = OUT_DIR / "runs"


def resolve(name: str) -> Path:
    """Find a recorded run by file name.

    A fresh run sitting in ``OUT_DIR`` wins; otherwise the archive under
    ``runs/data/*/`` is searched, so the scripts' default arguments keep working
    after a run has been filed away. Returns the live path unchanged when nothing
    is found, so the caller fails with the path a new run would write to.
    """
    live = OUT_DIR / name
    if live.exists():
        return live
    for archived in sorted((RUNS / "data").glob(f"*/{name}")):
        return archived
    return live
