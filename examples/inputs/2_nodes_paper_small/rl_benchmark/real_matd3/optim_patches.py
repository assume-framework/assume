# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Optimizer knobs the hyperparameter sweeps need and ``LearningConfig`` does not have.

Everything else the sweeps move -- ``learning_rate``, ``learning_rate_schedule``,
``min_learning_rate``, ``batch_size``, ``policy_delay`` -- is already a
``LearningConfig`` field and goes through ``assume_training_probe.py
--overrides-json``. **Weight decay is not.** ``matd3.py:366`` and ``:407``
construct ``AdamW`` with ``lr`` alone, so both networks silently take torch's
default of ``0.01``:

    AdamW(strategy.critics.parameters(), lr=...)

That default is not nothing. 0.01 decoupled weight decay on a critic fitting a
piecewise-constant reward is a real prior toward small weights, and it has never
been varied in this benchmark -- so "the current runs use no regularization" is
false, and a sweep that never touches it cannot say whether the setting helps.

The patch rebinds the ``AdamW`` name inside ``matd3``'s module namespace before
the world is built, so both optimizers pick it up with no edit to ``assume/``.
It is deliberately *not* a change to ``LearningConfig``: this is a probe, and
adding the field upstream is a decision for whoever reads the sweep.

Usage, in the child process before the scenario loads::

    from optim_patches import install_weight_decay
    install_weight_decay(0.0)          # AdamW as plain Adam
    install_weight_decay(0.1)          # ten times the current default
"""

from __future__ import annotations

import functools

#: torch's own AdamW default, i.e. what every run in this benchmark's archive
#: was trained with, because nothing ever passed the argument
TORCH_DEFAULT_WEIGHT_DECAY = 0.01


def install_weight_decay(weight_decay: float) -> None:
    """Give ``matd3``'s ``AdamW`` a fixed ``weight_decay``.

    Idempotent in effect but not in intent: calling it twice leaves the second
    value in force, wrapping the first wrapper. Call it once, in the child, and
    print what it did -- a run whose optimizer differs from the config is
    exactly the kind of thing that has to be visible in the log.
    """
    from assume.reinforcement_learning.algorithms import matd3

    original = getattr(matd3, "_ASSUME_ORIGINAL_ADAMW", matd3.AdamW)
    matd3._ASSUME_ORIGINAL_ADAMW = original

    @functools.wraps(original)
    def AdamW(*args, **kwargs):  # noqa: N802 - it stands in for the class
        kwargs["weight_decay"] = weight_decay
        return original(*args, **kwargs)

    matd3.AdamW = AdamW
    print(f"  patched matd3.AdamW: weight_decay={weight_decay} "
          f"(torch default {TORCH_DEFAULT_WEIGHT_DECAY})")
