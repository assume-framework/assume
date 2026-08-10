# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Run 12's offline experiment -- the critic's regression with the RL loop removed.

Sets ``gamma = 0``, so the target is the stored reward: no bootstrap, no moving
target, no actor feedback, no growing buffer. What is left is exactly the
regression ASSUME's critic has to solve, and it is enough to reproduce the live
failure. Everything else is ASSUME's own -- the real ``CriticTD3``, AdamW at
lr 1e-3, batch 128, gradient clipping at 1.0, both Q heads, and the frozen
true-reward buffer ``single_10ep_standard.npz``.

Three rounds, each answering the objection the previous one raises.

``--round conditions``
    Does the failure survive without the RL loop, and is the observation what
    causes it? ``const-obs`` is the surrogate's single-context setting;
    ``shuffled-obs`` keeps the dimensionality and the near-uniqueness of the
    states while destroying any real state-reward association; ``obs-x0.1``
    keeps every dimension and only shrinks them.

``--round ladder``
    Is it the action's *relative* weight, or just input magnitude? Scales the
    observation down and the action up, and compares at matched ``act_share``.
    If magnitude were the cause, scaling the action up -- which makes the inputs
    *larger* -- would not help.

``--round foresight``
    Does removing dimensions work as well as rescaling them? This is the offline
    version of the lever ``assume_actshare_sweep.py`` then applies live, and it
    reuses that module's ``truncate_obs`` so the observation matches what
    ``create_observation`` builds at that foresight.

Usage::

    python real_matd3/assume_offline_critic.py                # all three rounds
    python real_matd3/assume_offline_critic.py --round ladder
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch as th
from torch.nn import functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _layout import SCENARIO  # noqa: E402  (also sets sys.path)
from assume_actshare_sweep import BUFFER, FULL_FORESIGHT, truncate_obs  # noqa: E402

sys.path.insert(0, str(SCENARIO.parents[2]))
from assume.reinforcement_learning.neural_network_architecture import (  # noqa: E402
    CriticTD3,
)

MAX_BID, BAND, PLATEAU = 100.0, (30.0, 49.0), (50.0, 100.0)
LR, BATCH, CLIP, GRID = 1e-3, 128, 1.0, 401
#: 800 is the live 40-episode budget, 2560 is grad-32's
BUDGETS = (800, 2560)
N_SEEDS = 5

th.set_num_threads(1)


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(BUFFER)
    n = int(d["pos"][0])
    return (d["observations"][:n, 0, :].astype(np.float32),
            d["actions"][:n, 0, :].astype(np.float32),
            d["rewards"][:n, 0].astype(np.float32))


def act_share_of(obs: np.ndarray, sd_a: float, scale: float) -> float:
    a = sd_a * scale
    return a / (a + float(obs.std(axis=0).sum()))


def metrics(critic, probe: np.ndarray, bids: np.ndarray, act_scale: float) -> dict:
    """argmax, band slope and preference, swept at the probed observations.

    ``band_neg`` is the share of grid cells in ``[30, 49]`` carrying the true
    negative slope; 0.50 is a coin flip. The gradient is taken w.r.t. the *raw*
    action so the action-scale rows stay comparable in sign.
    """
    band = (bids >= BAND[0]) & (bids <= BAND[1])
    plat = (bids >= PLATEAU[0]) & (bids <= PLATEAU[1])
    argmax, band_neg, plat_neg, pref = [], [], [], []
    for row in probe:
        o = th.tensor(np.tile(row, (len(bids), 1)), dtype=th.float32)
        raw = th.tensor((bids / MAX_BID)[:, None], dtype=th.float32, requires_grad=True)
        q = critic.q1_forward(o, raw * act_scale)
        g = th.autograd.grad(q.sum(), raw)[0].numpy().ravel()
        qn = q.detach().numpy().ravel()
        argmax.append(bids[int(np.argmax(qn))])
        band_neg.append(float((g[band] < 0).mean()))
        plat_neg.append(float((g[plat] < 0).mean()))
        pref.append(float(qn[np.argmin(abs(bids - 32.0))]
                          - qn[np.argmin(abs(bids - 100.0))]))
    return {
        "argmax": float(np.median(argmax)),
        "band_neg": float(np.mean(band_neg)),
        "plat_neg": float(np.mean(plat_neg)),
        "pref": float(np.mean(pref)),
        "in_band": float(np.mean([BAND[0] <= a <= BAND[1] for a in argmax])),
    }


def fit(x: np.ndarray, act: np.ndarray, rew: np.ndarray, act_scale: float,
        seed: int, probe_idx: np.ndarray, bids: np.ndarray,
        budgets=BUDGETS) -> dict[int, dict]:
    rng = np.random.default_rng(seed)
    th.manual_seed(seed)
    critic = CriticTD3(n_agents=1, obs_dim=x.shape[1], act_dim=1,
                       unique_obs_dim=2, float_type=th.float32)
    opt = AdamW(critic.parameters(), lr=LR)

    # held-out split: memorisation shows up as train << test
    perm = rng.permutation(len(x))
    te, tr = perm[: len(x) // 5], perm[len(x) // 5:]
    X, A, R = th.tensor(x), th.tensor(act) * act_scale, th.tensor(rew)[:, None]

    out: dict[int, dict] = {}
    for step in range(1, max(budgets) + 1):
        idx = tr[rng.integers(0, len(tr), BATCH)]
        q1, q2 = critic(X[idx], A[idx])
        loss = F.mse_loss(q1, R[idx]) + F.mse_loss(q2, R[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        th.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=CLIP)
        opt.step()
        if step in budgets:
            with th.no_grad():
                trn = F.mse_loss(critic(X[tr], A[tr])[0], R[tr]).item()
                tst = F.mse_loss(critic(X[te], A[te])[0], R[te]).item()
            m = metrics(critic, x[probe_idx], bids, act_scale)
            m.update(train_mse=trn, test_mse=tst)
            out[step] = m
    return out


def table(rows: list[tuple[str, float, list[dict]]], title: str) -> None:
    print(f"\n{title}")
    header = (f"{'condition':<20}{'act_share':>10}{'argmax Q1':>16}{'in_band':>9}"
              f"{'band_neg':>10}{'plat_neg':>10}{'Q32-Q100':>11}{'train':>10}{'test':>10}")
    print(header)
    print("-" * len(header))
    for label, share, rs in rows:
        g = lambda k: np.array([r[k] for r in rs])  # noqa: E731
        am = g("argmax")
        print(f"{label:<20}{share:>10.3f}{am.mean():>10.1f} +-{am.std():>4.1f}"
              f"{g('in_band').mean():>9.2f}{g('band_neg').mean():>10.2f}"
              f"{g('plat_neg').mean():>10.2f}{g('pref').mean():>+11.4f}"
              f"{g('train_mse').mean():>10.5f}{g('test_mse').mean():>10.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", nargs="+", default=["conditions", "ladder",
                                                       "foresight"],
                        choices=["conditions", "ladder", "foresight"])
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()

    obs, act, rew = load()
    sd_a = float(act[:, 0].std())
    probe_idx = np.linspace(0, len(obs) - 1, 6).astype(int)  # the run 09/10 six
    bids = np.linspace(-MAX_BID, MAX_BID, GRID)
    seeds = range(args.seeds)

    print(f"gamma = 0 offline critic fit on {BUFFER.name}: n = {len(obs)}, "
          f"obs_dim = {obs.shape[1]}, sd(a) = {sd_a:.3f}")
    print(f"CriticTD3 + AdamW lr {LR}, batch {BATCH}, clip {CLIP}, {args.seeds} seeds")
    print("the true slope inside [30, 49] is negative, so band_neg should approach "
          "1.00; 0.50 is a coin flip")

    if "conditions" in args.round:
        def transform(name, rng):
            if name == "full-obs":
                return obs.copy()
            if name == "const-obs":
                return np.tile(obs.mean(axis=0, keepdims=True), (len(obs), 1))
            if name == "shuffled-obs":
                return obs[rng.permutation(len(obs))]
            if name == "obs-x0.1":
                return obs * 0.1
            if name == "obs-2dim":
                out = np.tile(obs.mean(axis=0, keepdims=True), (len(obs), 1))
                out[:, -2:] = obs[:, -2:]
                return out
            raise ValueError(name)

        names = ["full-obs", "const-obs", "shuffled-obs", "obs-x0.1", "obs-2dim"]
        by_budget: dict[int, list] = {b: [] for b in BUDGETS}
        for name in names:
            runs = [fit(transform(name, np.random.default_rng(s)), act, rew, 1.0,
                        s, probe_idx, bids) for s in seeds]
            x0 = transform(name, np.random.default_rng(0))
            for b in BUDGETS:
                by_budget[b].append((name, act_share_of(x0, sd_a, 1.0),
                                     [r[b] for r in runs]))
            print(f"  done: {name}", flush=True)
        for b in BUDGETS:
            table(by_budget[b], f"round 1 -- is it the observation?  ({b} updates)")

    if "ladder" in args.round:
        ladder = [("obs x1    act x1", 1.0, 1.0), ("obs x0.5  act x1", 0.5, 1.0),
                  ("obs x0.2  act x1", 0.2, 1.0), ("obs x0.1  act x1", 0.1, 1.0),
                  ("obs x0.03 act x1", 0.03, 1.0), ("obs x1    act x2", 1.0, 2.0),
                  ("obs x1    act x5", 1.0, 5.0), ("obs x1    act x10", 1.0, 10.0),
                  ("obs x1    act x30", 1.0, 30.0), ("obs z-score", "z", 1.0)]
        rows = []
        for label, os_, as_ in ladder:
            if os_ == "z":
                sd = obs.std(axis=0, keepdims=True)
                sd[sd < 1e-6] = 1.0
                x = (obs - obs.mean(axis=0, keepdims=True)) / sd
            else:
                x = obs * os_
            runs = [fit(x, act, rew, as_, s, probe_idx, bids, budgets=(2560,))
                    for s in seeds]
            rows.append((label, act_share_of(x, sd_a, as_), [r[2560] for r in runs]))
            print(f"  done: {label}", flush=True)
        table(rows, "round 2 -- relative weight, or magnitude?  (2560 updates)")

    if "foresight" in args.round:
        rows = []
        for k in (24, 12, 6, 3, 1):
            x = truncate_obs(obs, k) if k != FULL_FORESIGHT else obs
            runs = [fit(x, act, rew, 1.0, s, probe_idx, bids, budgets=(2560,))
                    for s in seeds]
            rows.append((f"foresight {k:>2} ({x.shape[1]} dims)",
                         act_share_of(x, sd_a, 1.0), [r[2560] for r in runs]))
            print(f"  done: foresight {k}", flush=True)
        table(rows, "round 3 -- removing dimensions, not rescaling  (2560 updates)")


if __name__ == "__main__":
    main()
