# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
How much the critic's preferred bid depends on which observation you ask from.

The critic is swept from several real observations at once, so ``argmax Q1`` is a
*set* of answers per frame, not one. Run 10 found that the median of that set is
close to meaningless on its own -- an unshaped critic's six answers were spread
across most of the bid axis, so the median moved a long way on very little -- and
the disagreement between them became the statistic that carried the finding.

**This module exists because that statistic was computed two different ways.**
Run 10 (``assume_stability.py``) and run 11 (``assume_config_sweep.py``) used the
*range*, ``max - min``; runs 12 and 13 used a mean pairwise difference. Both are
reasonable, they are not the same number -- on run 10's own archive the unshaped
condition reads 56.4 as a range and 24.5 as a mean pairwise difference -- and
``RUNS.md`` at one point quoted one against the other as if they were comparable.
Everything now imports from here so that cannot recur.

``argmax_disagreement`` is the primary statistic. ``argmax_range`` is kept because
it is what runs 10 and 11 originally reported, and because on a small sample it
says something the mean does not: whether *any* two observations disagree wildly.
Quote whichever you like, but say which one, and never mix them in one comparison.

Beware carrying either across settings. Run 13 (finding 21) shows the statistic
*inverting* in a genuine multi-agent scenario: with eleven learners the critic's
preferred bid genuinely should depend on the observation, so high disagreement
stops being evidence of a broken critic. It is a coherence diagnostic only where
the observation is known to carry no reward information.
"""

from __future__ import annotations

import numpy as np

__all__ = ["argmax_disagreement", "argmax_range", "peak_bids"]


def peak_bids(bids: np.ndarray, q: np.ndarray, axis: int = -1) -> np.ndarray:
    """``argmax Q`` in EUR/MWh, over the action grid ``axis`` of a sweep."""
    return np.asarray(bids)[np.argmax(q, axis=axis)]


def argmax_disagreement(peaks: np.ndarray, axis: int = 0) -> np.ndarray | float:
    """Mean absolute difference between two *distinct* probed observations.

    ``peaks`` holds one ``argmax Q1`` per probed observation along ``axis``; every
    other axis is kept and broadcast over, so this works on a single frame's six
    answers and on a whole ``(agents, obs, frames)`` film alike.

    The mean is over the ``n * (n - 1) / 2`` unordered pairs of *different*
    observations. The self-pairs are excluded deliberately: they are exactly zero
    and only dilute the answer, by a factor ``(n - 1) / n`` that depends on how
    many observations happened to be probed -- which would make two runs with
    different ``--n-obs`` silently incomparable. An earlier version of this
    statistic divided by ``n**2`` and did include them, so numbers computed before
    this module are 5/6 of these at the usual ``--n-obs 6``.

    Returns a scalar when ``peaks`` is one-dimensional, and ``nan`` for a single
    observation, where the quantity is undefined rather than zero.
    """
    peaks = np.asarray(peaks, dtype=float)
    n = peaks.shape[axis]
    if n < 2:
        return np.nan if peaks.ndim == 1 else np.full(
            np.delete(peaks.shape, axis), np.nan
        )

    moved = np.moveaxis(peaks, axis, 0)
    # |p_i - p_j| summed over all ordered pairs is twice the unordered sum, and
    # the n zero self-pairs contribute nothing to it -- so only the divisor differs
    total = np.abs(moved[:, None, ...] - moved[None, :, ...]).sum(axis=(0, 1))
    result = total / (n * (n - 1))
    return float(result) if np.ndim(result) == 0 else result


def argmax_range(peaks: np.ndarray, axis: int = 0) -> np.ndarray | float:
    """Spread between the most and least favourable probed observation.

    Runs 10 and 11's original statistic. More sensitive than
    ``argmax_disagreement`` to a single outlying observation, which is why it
    reads roughly twice as high on the same data, and why it is the better summary
    when the question is "do any two of these disagree at all".
    """
    peaks = np.asarray(peaks, dtype=float)
    result = peaks.max(axis=axis) - peaks.min(axis=axis)
    return float(result) if np.ndim(result) == 0 else result
