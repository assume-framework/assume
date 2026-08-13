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

``--round arch``
    **Workstream B's screen.** Same fit, same buffer, same budget -- only the
    critic architecture changes, over the variants in
    ``real_matd3/critic_architectures.py``: late action injection (Lillicrap
    et al. 2016), RSNorm, and SimBa (Lee et al., ICLR 2025).

    This round exists to settle a contradiction the rounds above created.
    ``ladder`` found that **z-scoring the observation was the worst cell in
    the table** -- ``argmax Q1`` pinned at exactly 100.0 in 5/5 seeds -- while
    observation standardization is SimBa's single most important component
    (their §7.1). Both cannot be the whole story. Either RSNorm differs from
    z-scoring in a way that matters, or normalization is only safe in company
    with the residual path and the LayerNorms. Running ``rsnorm`` alone,
    ``rsnorm+late`` and full ``simba`` on the same buffer says which.

    Each row also carries a **simplicity score** (SimBa's Fourier measure, via
    ``analysis/simplicity_bias.py``) taken on the *fitted* critic, so the
    table is a direct test of the paper's central correlation -- their Fig. 17
    reports Pearson 0.79 between simplicity score and return -- on a case
    where the outcome is known to be a failure.

Usage::

    python real_matd3/assume_offline_critic.py                # the first three
    python real_matd3/assume_offline_critic.py --round arch   # workstream B
    python real_matd3/assume_offline_critic.py --round arch --arch baseline simba
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch as th
from torch.nn import functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _layout import OUT_DIR, SCENARIO  # noqa: E402  (also sets sys.path)
from assume_actshare_sweep import BUFFER, FULL_FORESIGHT, truncate_obs  # noqa: E402
from critic_architectures import REGISTRY, build, describe, param_count  # noqa: E402

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


def sweep(critic, probe: np.ndarray, bids: np.ndarray,
          act_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """``Q1`` and ``dQ1/d(bid)`` over the bid grid, at each probed observation.

    Returns two ``(n_probe, n_bids)`` arrays. The gradient is taken w.r.t. the
    *raw* action, before ``act_scale``, so the action-scale rows of the ladder
    stay comparable in sign.
    """
    q_rows, g_rows = [], []
    for row in probe:
        o = th.tensor(np.tile(row, (len(bids), 1)), dtype=th.float32)
        raw = th.tensor((bids / MAX_BID)[:, None], dtype=th.float32, requires_grad=True)
        q = critic.q1_forward(o, raw * act_scale)
        g = th.autograd.grad(q.sum(), raw)[0].numpy().ravel()
        q_rows.append(q.detach().numpy().ravel())
        g_rows.append(g)
    return np.array(q_rows), np.array(g_rows)


def metrics(critic, probe: np.ndarray, bids: np.ndarray, act_scale: float) -> dict:
    """argmax, band slope and preference, swept at the probed observations.

    ``band_neg`` is the share of grid cells in ``[30, 49]`` carrying the true
    negative slope; 0.50 is a coin flip.
    """
    band = (bids >= BAND[0]) & (bids <= BAND[1])
    plat = (bids >= PLATEAU[0]) & (bids <= PLATEAU[1])
    q, g = sweep(critic, probe, bids, act_scale)

    argmax = bids[np.argmax(q, axis=1)]
    i32, i100 = np.argmin(abs(bids - 32.0)), np.argmin(abs(bids - 100.0))
    return {
        "argmax": float(np.median(argmax)),
        "band_neg": float((g[:, band] < 0).mean()),
        "plat_neg": float((g[:, plat] < 0).mean()),
        "pref": float((q[:, i32] - q[:, i100]).mean()),
        "in_band": float(np.mean((argmax >= BAND[0]) & (argmax <= BAND[1]))),
    }


def fit(x: np.ndarray, act: np.ndarray, rew: np.ndarray, act_scale: float,
        seed: int, probe_idx: np.ndarray, bids: np.ndarray,
        budgets=BUDGETS, arch: str = "baseline",
        simplicity: bool = False,
        film_every: int = 0) -> dict[int, dict]:
    """Fit the critic and read it out at each budget.

    With ``film_every`` set, the Q1 field and its gradient over the bid grid
    are also snapshotted every ``film_every`` updates and returned under the
    key ``"film"`` -- the offline equivalent of the live critic films, and the
    only way to see *when* during the fit a variant commits to the ceiling.
    Costs one sweep per snapshot, about a fifth of a training step's forward
    cost each.
    """
    rng = np.random.default_rng(seed)
    th.manual_seed(seed)
    critic = build(arch)(n_agents=1, obs_dim=x.shape[1], act_dim=1,
                         unique_obs_dim=2, float_type=th.float32)
    # AdamW's weight decay must not reach a normalizer's running statistics:
    # they are Parameters (so Polyak can sync them live) but not learned.
    opt = AdamW([p for p in critic.parameters() if p.requires_grad], lr=LR)

    # held-out split: memorisation shows up as train << test
    perm = rng.permutation(len(x))
    te, tr = perm[: len(x) // 5], perm[len(x) // 5:]
    X, A, R = th.tensor(x), th.tensor(act) * act_scale, th.tensor(rew)[:, None]

    out: dict[int, dict] = {}
    frames, frame_steps = [], []
    for step in range(1, max(budgets) + 1):
        if film_every and (step == 1 or step % film_every == 0):
            critic.eval()
            frames.append(sweep(critic, x[probe_idx], bids, act_scale))
            frame_steps.append(step - 1)  # updates completed before this sweep
        critic.train()
        idx = tr[rng.integers(0, len(tr), BATCH)]
        q1, q2 = critic(X[idx], A[idx])
        loss = F.mse_loss(q1, R[idx]) + F.mse_loss(q2, R[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        th.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=CLIP)
        opt.step()
        if step in budgets:
            # eval mode throughout the readout: a normalizing variant would
            # otherwise fold the whole training set into its running
            # statistics here, which is a leak the baseline cannot have and
            # would make the architecture rows incomparable
            critic.eval()
            with th.no_grad():
                trn = F.mse_loss(critic(X[tr], A[tr])[0], R[tr]).item()
                tst = F.mse_loss(critic(X[te], A[te])[0], R[te]).item()
            m = metrics(critic, x[probe_idx], bids, act_scale)
            m.update(train_mse=trn, test_mse=tst)
            if simplicity:
                m.update(s=simplicity_of(critic, x, act * act_scale))
            out[step] = m
    if film_every:
        out["film"] = {                                   # type: ignore[index]
            "steps": np.array(frame_steps),
            "q": np.stack([f[0] for f in frames]),        # (frames, probe, bids)
            "grad": np.stack([f[1] for f in frames]),
        }
    return out


def simplicity_of(critic, x: np.ndarray, act: np.ndarray) -> float:
    """SimBa's Fourier simplicity score of the *fitted* critic.

    Measurement C from ``analysis/simplicity_bias.py`` -- random 1-D lines
    through the buffer mean at full input dimension -- because the critic's
    input here is ~50-D and the paper's 2-D grid estimator does not reach that
    far. Higher is simpler. See that module for what the number is and is not.
    """
    from simplicity_bias import buffer_reference, line_score

    ref, scale = buffer_reference(x, act)
    return line_score(critic, ref, scale, obs_dim=x.shape[1])["s"]


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


def arch_table(rows: list[tuple[str, int, list[dict]]], title: str) -> None:
    """Like ``table`` but keyed on architecture: parameters in, act_share out.

    ``act_share`` is dropped because it is identical across these rows by
    construction -- the buffer and the action scale never change here, only
    the network -- and printing a constant column would invite reading it as a
    result. ``params`` and the simplicity score take its place.
    """
    print(f"\n{title}")
    header = (f"{'architecture':<16}{'params':>10}{'argmax Q1':>16}{'in_band':>9}"
              f"{'band_neg':>10}{'plat_neg':>10}{'Q32-Q100':>11}{'simple':>8}"
              f"{'train':>9}{'test':>9}")
    print(header)
    print("-" * len(header))
    for label, params, rs in rows:
        g = lambda k: np.array([r[k] for r in rs])  # noqa: E731
        am = g("argmax")
        print(f"{label:<16}{params:>10,}{am.mean():>10.1f} +-{am.std():>4.1f}"
              f"{g('in_band').mean():>9.2f}{g('band_neg').mean():>10.2f}"
              f"{g('plat_neg').mean():>10.2f}{g('pref').mean():>+11.4f}"
              f"{g('s').mean():>8.2f}"
              f"{g('train_mse').mean():>9.5f}{g('test_mse').mean():>9.5f}")
    print(
        "\nband_neg -> 1.00 is the true slope inside [30, 49]; 0.50 is a coin "
        "flip.\nin_band is the share of probed observations whose argmax lands "
        "in [30, 49].\n'simple' is SimBa's Fourier simplicity score of the "
        "fitted critic -- higher is simpler."
    )


def merge_film(path: Path, bids: np.ndarray, seed: int,
               shape: tuple[int, ...] | None = None) -> dict[str, dict]:
    """Rows of an existing ``films.npz`` that this run may keep.

    Filming is incremental so a variant added to the registry later can be
    recorded on its own -- the four 8.5 M rows cost tens of minutes each and
    re-recording them to gain one narrow panel is pure waste.

    A row is only carried over if it is **comparable**: same bid grid, same
    seed, same frame count. Those three are what the figure puts on shared
    axes, so a mismatch would draw two different experiments as one panel row.
    Anything that does not match is dropped and must be re-recorded.
    """
    if not path.exists():
        return {}
    d = np.load(path, allow_pickle=False)
    if d["bids"].shape != bids.shape or not np.allclose(d["bids"], bids):
        print(f"  film: bid grid changed, discarding {len(d['arch'])} old rows")
        return {}
    if int(d["seed"]) != seed:
        print(f"  film: seed {int(d['seed'])} != {seed}, discarding old rows")
        return {}
    if shape is not None and d["q"].shape[1:] != shape:
        print(f"  film: frame layout {d['q'].shape[1:]} != {shape}, "
              f"discarding {len(d['arch'])} old rows")
        return {}
    return {
        str(a): {"steps": d["steps"][i], "q": d["q"][i], "grad": d["grad"][i]}
        for i, a in enumerate(d["arch"])
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", nargs="+", default=["conditions", "ladder",
                                                       "foresight"],
                        choices=["conditions", "ladder", "foresight", "arch"])
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--arch", nargs="+", default=list(REGISTRY),
                        choices=list(REGISTRY),
                        help="which critic architectures the 'arch' round runs")
    parser.add_argument(
        "--film", type=int, default=0, metavar="EVERY",
        help="snapshot the Q1 field and its bid gradient every EVERY updates "
             "during the 'arch' round and write runs/data/17-offline-arch/"
             "films.npz. 80 gives 32 frames over the 2560-update budget. "
             "First seed only. Draw with analysis/offline_arch_film.py",
    )
    parser.add_argument(
        "--threads", type=int, default=1,
        help="torch threads. The default of 1 is what runs 12's recorded "
             "tables used, and BLAS thread count changes float summation "
             "order, so raising it makes a row no longer bit-comparable with "
             "the archive. Raise it for the wide SimBa rows, which are "
             "compute-bound, and say so when quoting them.",
    )
    args = parser.parse_args()
    th.set_num_threads(args.threads)

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

    if "arch" in args.round:
        by_budget: dict[int, list] = {b: [] for b in BUDGETS}
        films: dict[str, dict] = {}
        for name in args.arch:
            t0 = time.perf_counter()
            runs = []
            for s in seeds:
                # the film is recorded for the first seed only: it is an
                # illustration of the trajectory, while the table beside it
                # already carries the across-seed statistics. Filming all five
                # would multiply the wide SimBa rows' cost for no new claim.
                every = args.film if (args.film and s == list(seeds)[0]) else 0
                r = fit(obs, act, rew, 1.0, s, probe_idx, bids,
                        arch=name, simplicity=True, film_every=every)
                if "film" in r:
                    films[name] = r.pop("film")
                runs.append(r)
            params = param_count(name, obs_dim=obs.shape[1], act_dim=1,
                                 n_agents=1, unique_obs_dim=2)
            for b in BUDGETS:
                by_budget[b].append((name, params, [r[b] for r in runs]))
            # the headline numbers as each variant lands, not only in the
            # tables at the end: the wide SimBa rows take tens of minutes and
            # a screen that prints nothing until all of them finish cannot be
            # watched, killed early, or read at all if it dies
            last = [r[max(BUDGETS)] for r in runs]
            am = np.array([r["argmax"] for r in last])
            print(f"  done: {name:<14} {time.perf_counter() - t0:6.1f}s  "
                  f"argmax {am.mean():6.1f} +-{am.std():4.1f}  "
                  f"in_band {np.mean([r['in_band'] for r in last]):.2f}  "
                  f"band_neg {np.mean([r['band_neg'] for r in last]):.2f}  "
                  f"simple {np.mean([r['s'] for r in last]):5.2f}",
                  flush=True)
        for b in BUDGETS:
            arch_table(
                by_budget[b],
                f"round 4 -- workstream B: critic architecture  ({b} updates, "
                f"{args.seeds} seeds)\nsame buffer, same optimizer, same "
                f"budget; only the network changes",
            )

        if films:
            path = OUT_DIR / "runs" / "data" / "17-offline-arch" / "films.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            first = films[next(iter(films))]["q"].shape
            kept = merge_film(path, bids, int(list(seeds)[0]), shape=first)
            for name, keep in kept.items():
                films.setdefault(name, keep)      # a re-recorded row wins
            np.savez_compressed(
                path,
                bids=bids,
                arch=np.array(list(films)),
                steps=np.stack([films[a]["steps"] for a in films]),
                # (arch, frames, probe, bids)
                q=np.stack([films[a]["q"] for a in films]),
                grad=np.stack([films[a]["grad"] for a in films]),
                seed=np.array(list(seeds)[0]),
                params=np.array([
                    param_count(a, obs_dim=obs.shape[1], act_dim=1,
                                n_agents=1, unique_obs_dim=2) for a in films
                ]),
            )
            print(f"\nwrote {path}  ({len(films)} architectures, "
                  f"{len(kept)} carried over, "
                  f"{path.stat().st_size / 1e6:.1f} MB)")
            print("draw it with:  python analysis/offline_arch_film.py")


if __name__ == "__main__":
    main()
