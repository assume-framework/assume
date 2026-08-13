# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
SimBa's Fourier simplicity-bias score, for ASSUME's critics.

Lee et al. (ICLR 2025) §2.1 define a complexity measure as the
frequency-weighted average of a function's Fourier coefficients::

    c(f) = sum_k |f~(k)| * k / sum_k |f~(k)|            (their Eq. 1 / 10)
    s(f) = E_theta [ 1 / c(f_theta) ]                   (their Eq. 2 / 15)

Large ``c`` means high-frequency content dominates -- rapid changes, intricate
detail. ``s``, the simplicity score, is its reciprocal, so **higher ``s`` is
simpler**. Their claim is that architectures with a higher ``s`` at
initialization converge to simpler functions and scale better with parameter
count (their Figs. 4, 5, 17).

What is a definition and what is an estimator
---------------------------------------------
The *definition* is general: §2.1 quantifies functions in
``F = {f | f : X -> Y}`` for ``X`` a subset of ``R^n`` and ``Y`` of ``R^m``.
Nothing there is 2-dimensional.

The *estimator* they use is 2-dimensional, for one reason: it is a discrete
Fourier transform on a **uniform grid over the input domain**, which is also
what lets them pick the cutoff ``K`` by Nyquist-Shannon. A grid of ``G``
divisions per axis costs ``G^n`` points -- at their ``G = 300`` that is 90 000
for ``n = 2``, 27 million for ``n = 3``, and nothing at all for the ~57-D input
of the critics here. So the dimensionality limit is in the estimator, not the
measure. Their output is scalar (``m = 1``), and so is a Q head, so ``m``
constrains nothing for us.

There is a second thing worth knowing before reading any number below: their
Fig. 4(a) scores are **not** measured on a trained RL critic. They instantiate
each architecture *template* with a 2-input / 1-output head at random
initialization and measure that. It is a property of the architecture's
inductive bias before training. It has no time axis in their paper.

The two measurements here
-------------------------
``grid`` (measurement A) -- **the paper's protocol, reproduced.**
    Instantiate each critic variant with a 1-D observation and a 1-D action,
    i.e. exactly 2 inputs, at random initialization; evaluate Q1 on a
    ``300 x 300`` grid over ``[-100, 100]^2``; 2-D DFT; average ``1/c`` over
    100 seeds. This is Appendix B step for step, and it answers one question:
    do *our* variants order the way *their* variants did? It says nothing
    about the trained critic.

``lines`` (measurement C) -- **the same measure, at full input dimension.**
    Sample ``L`` random directions through a reference point in the critic's
    real input space, evaluate Q1 along each at ``S`` samples, take a 1-D DFT
    per line, and average. Same weighting formula, no grid, so it works at any
    ``n``, has no privileged axes and needs no dimensions frozen at arbitrary
    values. ``L = 300`` lines at ``S = 300`` samples is 90 000 evaluations --
    deliberately the same budget as one grid in measurement A.

    This is a **deviation**, and an honest reading needs it stated: a 1-D
    slice of an n-D function does not have the n-D spectrum, so ``c`` from
    ``lines`` is not ``c`` from ``grid`` measured more cheaply. It is the same
    functional applied to directional restrictions, averaged over directions.
    Compare ``lines`` numbers to other ``lines`` numbers only.

Conventions, so numbers here are comparable to each other
---------------------------------------------------------
* The output is **mean-centred** before the transform. The DC coefficient
  otherwise carries the function's mean, which is unrelated to its complexity
  and, being typically far larger than every other coefficient, would drive
  ``c`` toward 0 for every architecture alike. After centring the DC term is
  exactly 0 and drops out of both sums on its own.
* Frequency is **normalized so that Nyquist = 1**, radially in 2-D. So ``c``
  lies in ``[0, ~1.41]`` for grids and ``[0, 1]`` for lines regardless of grid
  resolution, and ``s = 1/c`` is comparable across settings.
* Absolute values are therefore **not** comparable to the paper's printed
  5.8-6.5, which use their own normalization. **The ordering is the claim**,
  and the ordering is what transfers.

Usage::

    python analysis/simplicity_bias.py                    # A, all variants
    python analysis/simplicity_bias.py --measure lines    # C, on the buffer
    python analysis/simplicity_bias.py --measure both
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _layout import OUT_DIR  # noqa: E402  (also sets sys.path)
from critic_architectures import (  # noqa: E402
    REGISTRY,
    build,
    describe,
    match_width,
    param_count,
)

#: Appendix B: 300 divisions per axis over [-100, 100]^2, 100 initializations
GRID_DIVISIONS = 300
GRID_EXTENT = 100.0
N_INIT = 100

#: measurement C: 300 lines x 300 samples = the same 90 000 evaluations
N_LINES = 300
N_SAMPLES = 300
#: how far along each direction to walk, in units of the input's own std
LINE_RADIUS = 3.0


# ------------------------------------------------------------ the measure


def complexity_2d(image: np.ndarray) -> float:
    """``c(f)`` for a function sampled on a 2-D grid (their Eq. 10).

    ``image`` is the ``(G, G)`` array of scalar outputs. Radial frequency is
    normalized so 1.0 is Nyquist along an axis; the grid corners reach
    ``sqrt(2)``.
    """
    image = image - image.mean()
    mag = np.abs(np.fft.fft2(image))
    fy = np.fft.fftfreq(image.shape[0]) / 0.5  # [-1, 1), 1 == Nyquist
    fx = np.fft.fftfreq(image.shape[1]) / 0.5
    k = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    total = mag.sum()
    if total <= 0:  # an exactly constant function has no frequency content
        return float("nan")
    return float((mag * k).sum() / total)


def complexity_1d(lines: np.ndarray) -> np.ndarray:
    """``c(f)`` per row for functions sampled along 1-D lines.

    ``lines`` is ``(L, S)``; returns ``(L,)``. Same formula as
    :func:`complexity_2d` with ``k`` the absolute normalized frequency.
    """
    lines = lines - lines.mean(axis=1, keepdims=True)
    mag = np.abs(np.fft.fft(lines, axis=1))
    k = np.abs(np.fft.fftfreq(lines.shape[1])) / 0.5
    total = mag.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (mag * k[None, :]).sum(axis=1) / total, np.nan)


def _summarize(cs: np.ndarray) -> dict[str, float]:
    """Mean complexity and the simplicity score ``E[1/c]`` (their Eq. 15).

    ``s`` is the mean of the reciprocals, not the reciprocal of the mean --
    the paper takes the expectation outside, and for a spread of ``c`` the two
    differ by Jensen's inequality.
    """
    cs = np.asarray(cs, dtype=float)
    cs = cs[np.isfinite(cs) & (cs > 0)]
    if cs.size == 0:
        return {"c": float("nan"), "s": float("nan"), "s_sd": float("nan"), "n": 0}
    s = 1.0 / cs
    return {
        "c": float(cs.mean()),
        "s": float(s.mean()),
        # 95% CI half-width, the interval the paper's figures show
        "s_sd": float(1.96 * s.std(ddof=1) / np.sqrt(s.size)) if s.size > 1 else 0.0,
        "n": int(s.size),
    }


# ------------------------------------------- measurement A: the paper's grid


def grid_score(
    cls,
    n_init: int = N_INIT,
    divisions: int = GRID_DIVISIONS,
    extent: float = GRID_EXTENT,
    seed0: int = 0,
    fit_stats: bool = True,
) -> dict[str, float]:
    """Appendix B, applied to one of our critic variants.

    The variant is built with ``obs_dim = 1`` and ``act_dim = 1`` so its input
    is the 2-D plane the protocol needs. Axis 0 of the grid is fed as the
    observation and axis 1 as the action, which is the only way to give a
    critic a 2-D input without inventing a projection.

    **RSNorm makes this protocol ambiguous, and the choice moves the
    table.** A freshly constructed RSNorm is exactly the identity, so with
    ``fit_stats = False`` every RSNorm row is bit-identical to its
    un-normalized twin and the normalizer cannot be scored at all. With
    ``fit_stats = True`` the grid is passed through once in training mode to
    fold the input domain into the running statistics -- but the domain is
    ``[-100, 100]^2``, so RSNorm then divides the input by its own standard
    deviation of about 58, and the network sees a domain ~58x smaller. Much of
    what that row measures is therefore an input rescaling, not an
    architecture.

    Neither reading is obviously the paper's. Their Fig. 4(a) does separate
    ``MLP`` from ``MLP + RSNorm``, which implies something like
    ``fit_stats = True``; but their Appendix C.1 then reports RSNorm's score
    moving the *opposite* way to its effect on return, and explains it exactly
    this way -- a uniform input distribution is not the distribution RSNorm is
    for. Treat the RSNorm rows of measurement A as the least trustworthy in
    the table, and read ``lines`` on a real buffer instead.
    """
    axis = np.linspace(-extent, extent, divisions, dtype=np.float32)
    o = th.as_tensor(np.repeat(axis, divisions)[:, None])
    a = th.as_tensor(np.tile(axis, divisions)[:, None])

    cs = []
    for i in range(n_init):
        th.manual_seed(seed0 + i)
        critic = cls(
            n_agents=1, obs_dim=1, act_dim=1,
            float_type=th.float32, unique_obs_dim=0,
        )
        # only a variant that *has* running statistics needs the extra pass;
        # for the rest it would double the cost of the whole measurement
        needs_fit = fit_stats and any(
            not p.requires_grad for p in critic.parameters()
        )
        if needs_fit:
            critic.train()
            with th.no_grad():
                critic(o, a)  # fold the input domain into the running statistics
        critic.eval()
        with th.no_grad():
            q = critic.q1_forward(o, a).numpy().reshape(divisions, divisions)
        cs.append(complexity_2d(q))
    return _summarize(np.array(cs))


# --------------------------------------- measurement C: random lines, full n


def line_score(
    critic,
    reference: np.ndarray,
    scale: np.ndarray,
    n_lines: int = N_LINES,
    n_samples: int = N_SAMPLES,
    radius: float = LINE_RADIUS,
    seed: int = 0,
    obs_dim: int | None = None,
) -> dict[str, float]:
    """``c`` averaged over random 1-D directions through the critic's inputs.

    Args:
        critic: any module exposing ``q1_forward(obs, actions)``.
        reference: ``(n_inputs,)`` centre of the probe, in the critic's own
            input units -- the buffer mean is the natural choice, since it is
            the region the critic was actually fitted on.
        scale: ``(n_inputs,)`` per-dimension step size, normally the buffer
            standard deviation. A direction is walked in units of this, so
            dimensions that barely vary in the data are barely varied here.
            Without it a raw unit direction would sweep every dimension
            equally and mostly measure the critic far outside its data.
        obs_dim: how many leading entries of ``reference`` are the
            observation; the rest are the action. Defaults to all-but-one.

    Directions are drawn isotropically (normalized Gaussians), so the action
    dimension gets no special treatment -- that is the point of this
    measurement as against the critic films, which sweep the action alone.
    """
    n = reference.shape[0]
    obs_dim = n - 1 if obs_dim is None else obs_dim

    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n_lines, n)).astype(np.float32)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    t = np.linspace(-radius, radius, n_samples, dtype=np.float32)

    # (n_lines, n_samples, n): reference + t * (u scaled per dimension)
    step = (u * scale[None, :])[:, None, :] * t[None, :, None]
    pts = reference[None, None, :] + step
    flat = pts.reshape(n_lines * n_samples, n)

    was_training = critic.training
    critic.eval()
    with th.no_grad():
        x = th.as_tensor(flat, dtype=th.float32)
        q = critic.q1_forward(x[:, :obs_dim], x[:, obs_dim:]).numpy()
    critic.train(was_training)

    return _summarize(complexity_1d(q.reshape(n_lines, n_samples)))


def buffer_reference(obs: np.ndarray, act: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Probe centre and per-dimension scale from a replay buffer.

    Zero-variance dimensions get a scale of 0, so they stay pinned at their
    constant value instead of being swept through a range the critic has never
    seen. That is a deliberate choice: it measures the function's complexity
    *on the data manifold's bounding box*, which is where the critic's fit
    matters, not over a box it was never asked about.
    """
    x = np.concatenate([obs, act], axis=1)
    return x.mean(axis=0).astype(np.float32), x.std(axis=0).astype(np.float32)


# --------------------------------------------------------------------- CLI


def _table(rows: list[tuple[str, int, dict, float]], title: str, note: str) -> None:
    print(f"\n{title}")
    header = (f"{'variant':<16}{'params':>10}{'c (complexity)':>16}"
              f"{'s = E[1/c]':>13}{'95% CI':>9}{'sec':>7}  description")
    print(header)
    print("-" * len(header))
    for name, params, m, secs in sorted(rows, key=lambda r: -r[2]["s"]):
        print(
            f"{name:<16}{params:>10,}{m['c']:>16.4f}{m['s']:>13.3f}"
            f"{m['s_sd']:>9.3f}{secs:>7.1f}  {describe(name)}"
        )
    print(f"\n{note}")

    # Identical rows are a result, not a glitch, and saying so beats letting a
    # reader discover it. Two ways they arise: an unfitted RSNorm is exactly
    # the identity, so at initialization every RSNorm row equals its
    # un-normalized twin; and matching parameter counts collapses a pure width
    # ladder (simba / simba-small / simba-tiny) onto one network.
    groups: dict[tuple, list[str]] = {}
    for name, params, m, _ in rows:
        groups.setdefault((params, round(m["c"], 9)), []).append(name)
    tied = [names for names in groups.values() if len(names) > 1]
    if tied:
        print("\nidentical rows (same network, by construction):")
        for names in tied:
            print(f"  {' == '.join(names)}")


def _resolve(arch: str, target: int | None, obs_dim: int, act_dim: int,
             unique_obs_dim: int) -> tuple[type, int]:
    """The class to measure and its parameter count, matched if asked."""
    kw = dict(obs_dim=obs_dim, act_dim=act_dim, n_agents=1,
              unique_obs_dim=unique_obs_dim)
    if target is None:
        cls = build(arch)
        return cls, param_count(cls, **kw)
    cls, _, got = match_width(arch, target, **kw)
    return cls, got


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--measure", nargs="+", default=["grid"],
                   choices=["grid", "lines", "both"])
    p.add_argument("--arch", nargs="+", default=list(REGISTRY))
    p.add_argument("--n-init", type=int, default=N_INIT)
    p.add_argument("--divisions", type=int, default=GRID_DIVISIONS)
    p.add_argument("--lines", type=int, default=N_LINES)
    p.add_argument("--samples", type=int, default=N_SAMPLES)
    p.add_argument(
        "--match-params", action="store_true",
        help="size every variant to the baseline's parameter count by "
             "bisecting its width. This is SimBa's own protocol (their "
             "Appendix D holds all twelve architectures within 1%% of 4.5 M) "
             "and is the only setting in which the table compares "
             "architectures rather than architecture-and-capacity at once.",
    )
    p.add_argument(
        "--no-fit-rsnorm", action="store_true",
        help="leave RSNorm's running statistics unset in measurement A, which "
             "makes it the exact identity, so its rows collapse onto their "
             "un-normalized twins. The default folds the probe grid into them "
             "instead -- see grid_score's docstring, neither choice is clearly "
             "the paper's and the two disagree",
    )
    p.add_argument("--threads", type=int, default=0,
                   help="torch threads; 0 = leave torch's default")
    args = p.parse_args()
    if args.threads:
        th.set_num_threads(args.threads)
    which = set(args.measure)
    if "both" in which:
        which = {"grid", "lines"}
    matched = " at matched params" if args.match_params else ""

    if "grid" in which:
        # the protocol's network has exactly two inputs, so the matching
        # target is the baseline measured at that same probe size
        target = (
            param_count("baseline", obs_dim=1, act_dim=1, n_agents=1,
                        unique_obs_dim=0)
            if args.match_params else None
        )
        rows = []
        for name in args.arch:
            cls, params = _resolve(name, target, 1, 1, 0)
            t0 = time.perf_counter()
            m = grid_score(cls, n_init=args.n_init, divisions=args.divisions,
                           fit_stats=not args.no_fit_rsnorm)
            rows.append((name, params, m, time.perf_counter() - t0))
            print(f"  done: {name}", flush=True)
        _table(
            rows,
            f"measurement A -- SimBa Appendix B, reproduced{matched}  "
            f"({args.divisions}x{args.divisions} grid on [-100,100]^2, "
            f"{args.n_init} inits)",
            "higher s = simpler. Absolute values use k normalized to Nyquist = 1 "
            "and are\nNOT comparable to the paper's printed 5.8-6.5; the ordering "
            "is the claim.",
        )

    if "lines" in which:
        from assume_offline_critic import load  # noqa: E402  (lazy: needs the buffer)

        obs, act, _ = load()
        ref, scale = buffer_reference(obs, act)
        target = (
            param_count("baseline", obs_dim=obs.shape[1], act_dim=act.shape[1],
                        n_agents=1, unique_obs_dim=2)
            if args.match_params else None
        )
        rows = []
        for name in args.arch:
            cls, params = _resolve(name, target, obs.shape[1], act.shape[1], 2)
            t0 = time.perf_counter()
            th.manual_seed(0)
            critic = cls(
                n_agents=1, obs_dim=obs.shape[1], act_dim=act.shape[1],
                float_type=th.float32, unique_obs_dim=2,
            )
            m = line_score(critic, ref, scale, n_lines=args.lines,
                           n_samples=args.samples, obs_dim=obs.shape[1])
            rows.append((name, params, m, time.perf_counter() - t0))
            print(f"  done: {name}", flush=True)
        _table(
            rows,
            f"measurement C -- random 1-D lines at full input dimension, at "
            f"initialization{matched}\n({args.lines} lines x {args.samples} "
            f"samples through the buffer mean, +-{LINE_RADIUS} sd)",
            "1-D restrictions of an n-D function: compare these to other 'lines' "
            "numbers only,\nnever to measurement A's.",
        )


if __name__ == "__main__":
    main()
