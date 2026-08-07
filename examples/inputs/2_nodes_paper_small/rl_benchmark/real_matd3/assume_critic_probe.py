# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
The same probe as ``critic_landscape.py``, but on ASSUME's own trained MATD3.

Everything else in this folder is an SB3 analogue. This reads the real thing:
the ``.pt`` files a learning run leaves in
``examples/inputs/<scenario>/learned_strategies/<case>/last_policies/``, sweeps
``Q1`` -- the quantity ``matd3.py`` has the actor ascend -- over the bid axis at
observations taken from that run's own replay buffer, and reports where the
critic's maximum sits versus where the actor ended up.

What it is for: run 06 (``descent_window.py``) predicts a specific end state for a
MATD3 run that fails on this landscape -- **a correct critic and an actor parked
at the ceiling**, because the descent path is only open for a few hundred critic
updates and the actor is not given enough updates to use it. The optimizer state
in the saved files makes the budget explicit: ``critic_optimizer`` has taken
``step: 640`` and ``actor_optimizer`` ``step: 80``, against the 190-410 actor
updates the crossing costs in run 06.

Usage::

    python assume_critic_probe.py                 # every case with saved policies
    python assume_critic_probe.py --cases inc_dec_learning_single_g0_2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _layout import (
    OUT_DIR,  # noqa: E402  (also puts the folders on sys.path)
    SCENARIO,  # noqa: E402
)
from incdec_reward import PAPER_SMALL  # noqa: E402
from run_benchmark import COLORS, INK, MUTED  # noqa: E402

POLICIES = SCENARIO / "learned_strategies"

#: ``max_bid_price`` from the scenario's ``learning_config``. The actor output is
#: scaled by it, exactly as in ``TorchLearningStrategy``.
MAX_BID_PRICE = 100.0
#: ``3 * foresight + unique_obs_dim`` for foresight 24 -- the layout the strategy
#: builds and the critic was trained on.
OBS_DIM = 74
UNIQUE_OBS_DIM = 2


def load_networks(case_dir: Path, unit: str = "diesel_0"):
    """Rebuild the actor and critic from a case's ``last_policies``.

    The architectures are ASSUME's own, so this fails loudly if the saved shapes
    stop matching ``neural_network_architecture.py`` rather than silently probing
    something else.
    """
    from assume.reinforcement_learning.neural_network_architecture import (
        CriticTD3,
        MLPActor,
    )

    actor_blob = th.load(
        case_dir / "actors" / f"actor_{unit}.pt", map_location="cpu", weights_only=False
    )
    critic_blob = th.load(
        case_dir / "critics" / f"critic_{unit}.pt",
        map_location="cpu",
        weights_only=False,
    )

    actor = MLPActor(OBS_DIM, 1, th.float32)
    actor.load_state_dict(actor_blob["actor"])
    actor.eval()

    critic = CriticTD3(
        n_agents=1,
        obs_dim=OBS_DIM,
        act_dim=1,
        float_type=th.float32,
        unique_obs_dim=UNIQUE_OBS_DIM,
    )
    critic.load_state_dict(critic_blob["critic"])
    critic.eval()

    budget = {
        "actor_updates": int(actor_blob["actor_optimizer"]["state"][0]["step"].item()),
        "critic_updates": int(
            critic_blob["critic_optimizer"]["state"][0]["step"].item()
        ),
    }
    return actor, critic, budget


#: Collection buffer to borrow observations from when a case saved none of its
#: own. Only the observations are used, and every case in this scenario observes
#: the same 74-vector layout over the same days, so the choice does not bias the
#: sweep -- the rewards, which *do* differ between the shaped and unshaped runs,
#: are not read from here.
SHARED_BUFFER = POLICIES / "buffers" / "single_10ep_standard.npz"


def read_buffer(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(observations, bids, rewards)`` from the filled part of a saved ring.

    Bids are actions scaled back to EUR/MWh, which makes the buffer an empirical
    picture of the landscape the critic was fitted to -- including any reward
    shaping that was active during that run.
    """
    data = np.load(path)
    n = len(data["observations"]) if bool(data["full"][0]) else int(data["pos"][0])
    return (
        data["observations"][:n, 0, :],
        data["actions"][:n, 0, 0] * MAX_BID_PRICE,
        data["rewards"][:n, 0],
    )


def observation_source(case_dir: Path) -> Path:
    own = case_dir / "replay_buffer.npz"
    return own if own.exists() else SHARED_BUFFER


def critic_curve(critic, obs: np.ndarray, bids: np.ndarray):
    """Sweep ``Q1`` and its autograd gradient over the bid axis at one observation.

    ``q1_forward`` is what ``matd3.py`` differentiates for the actor loss, so this
    is the surface the actor climbs -- not ``min(Q1, Q2)``, which forms the critic
    target only.
    """
    obs_batch = th.as_tensor(np.repeat(obs[None, :], len(bids), axis=0), dtype=th.float32)
    act = th.as_tensor((bids / MAX_BID_PRICE)[:, None], dtype=th.float32)
    act.requires_grad_(True)

    q = critic.q1_forward(obs_batch, act)
    (grad,) = th.autograd.grad(q.sum(), act)
    return q.detach().numpy().ravel(), grad.numpy().ravel() / MAX_BID_PRICE


def descent_stop(bids: np.ndarray, grad: np.ndarray, start: float) -> float:
    """Where an unbroken leftward descent from ``start`` runs out of gradient."""
    k = int(np.argmin(np.abs(bids - start)))
    while k > 0 and grad[k] < 0:
        k -= 1
    return float(bids[k])


def probe(case_dir: Path, bids: np.ndarray, n_obs: int) -> dict:
    actor, critic, budget = load_networks(case_dir)
    source = observation_source(case_dir)
    obs_pool, buf_bids, buf_rewards = read_buffer(source)

    # Real observations, never a hand-built one: the 74-vector varies hour to hour
    # in this scenario (548 distinct vectors in a 620-transition buffer), so a
    # constant context would probe a state the critic was never trained on.
    idx = np.linspace(0, len(obs_pool) - 1, n_obs).astype(int)
    observations = obs_pool[idx]

    qs, grads, actor_bids = [], [], []
    for obs in observations:
        q, g = critic_curve(critic, obs, bids)
        qs.append(q)
        grads.append(g)
        with th.no_grad():
            a = actor(th.as_tensor(obs[None, :], dtype=th.float32)).item()
        actor_bids.append(a * MAX_BID_PRICE)

    qs, grads, actor_bids = np.array(qs), np.array(grads), np.array(actor_bids)
    argmax = bids[qs.argmax(axis=1)]
    grad_at_actor = np.array(
        [np.interp(b, bids, g) for b, g in zip(actor_bids, grads)]
    )
    stop = np.array(
        [descent_stop(bids, g, b) for b, g in zip(actor_bids, grads)]
    )

    p = PAPER_SMALL
    return {
        "name": case_dir.parent.name,
        "budget": budget,
        "obs_source": source.name,
        "own_buffer": source.parent.parent.name == case_dir.parent.name,
        "buffer_in_band": float(
            np.mean((buf_bids >= p.dec_threshold) & (buf_bids <= p.eom_price))
        ),
        "buffer_n": len(buf_bids),
        "bids": bids,
        "q": qs,
        "grad": grads,
        "actor_bids": actor_bids,
        "argmax": argmax,
        "grad_at_actor": grad_at_actor,
        "stop": stop,
    }


def report(results: list[dict]) -> None:
    p = PAPER_SMALL
    print(
        f"\n  band {p.dec_threshold:.0f}-{p.eom_price:.0f} EUR/MWh, "
        f"optimum {p.optimal_bid:.0f}\n"
    )
    print(
        f"  {'case':<34}{'critic':>8}{'actor':>7}{'critic argmax':>16}"
        f"{'actor bids':>15}{'dQ/da @ actor':>15}{'descent reaches':>17}"
    )
    print("  " + "-" * 112)
    for r in results:
        argmax, actor_bids = r["argmax"], r["actor_bids"]
        in_band = np.mean((argmax >= p.dec_threshold) & (argmax <= p.eom_price))
        reaches = np.mean(r["stop"] <= p.eom_price)
        print(
            f"  {r['name']:<34}{r['budget']['critic_updates']:>8}"
            f"{r['budget']['actor_updates']:>7}"
            f"{np.median(argmax):>10.1f} ({in_band:>3.0%}){np.median(actor_bids):>15.1f}"
            f"{np.median(r['grad_at_actor']):>15.1e}{np.median(r['stop']):>12.1f}"
            f" ({reaches:>3.0%})"
        )
    print(
        "\n  critic/actor = optimizer steps taken.  '(%)' after the argmax is the\n"
        "  share of probed observations whose critic maximum falls inside the band;\n"
        "  after 'descent reaches', the share from which an unbroken leftward\n"
        "  descent from the actor's own bid arrives in the band.\n"
    )
    for r in results:
        own = "own" if r["own_buffer"] else "borrowed"
        print(
            f"  {r['name']:<46} observations: {own} {r['obs_source']} "
            f"({r['buffer_n']} transitions, {r['buffer_in_band']:.0%} in band)"
        )
    print()


def plot(results: list[dict], out: Path) -> None:
    p = PAPER_SMALL
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(6.4 * n, 8.4), squeeze=False)

    for col, r in enumerate(results):
        bids = r["bids"]
        ax_q, ax_g = axes[0][col], axes[1][col]

        for ax in (ax_q, ax_g):
            ax.axvspan(p.dec_threshold, p.eom_price, color="#1baf7a", alpha=0.10, lw=0)
            for b in r["actor_bids"]:
                ax.axvline(b, color=INK, lw=0.8, alpha=0.35)

        # --- Q1, one curve per probed observation, centred so they overlay ------
        for q in r["q"]:
            ax_q.plot(bids, q - q.mean(), lw=1.2, color=COLORS["TD3"], alpha=0.5)
        ax_q.plot(
            [np.median(r["argmax"])], [0], "v", ms=9, color=COLORS["TD3"],
            clip_on=False, zorder=5,
        )
        ax_q.set_title(
            f"{r['name']}\ncritic {r['budget']['critic_updates']} updates · "
            f"actor {r['budget']['actor_updates']}",
            loc="left", fontsize=10.5, color=INK,
        )
        ax_q.set_ylabel("Q1 (centred per observation)")
        ax_q.annotate(
            f"critic argmax {np.median(r['argmax']):.0f}",
            xy=(np.median(r["argmax"]), 0), xytext=(6, 14),
            textcoords="offset points", fontsize=8.5, color=COLORS["TD3"],
        )
        ax_q.annotate(
            f"actor bids {np.median(r['actor_bids']):.0f}",
            xy=(np.median(r["actor_bids"]), 0), xytext=(-6, -22),
            textcoords="offset points", ha="right", fontsize=8.5, color=INK,
        )

        # --- the gradient the actor would follow -------------------------------
        for g in r["grad"]:
            ax_g.plot(bids, g, lw=1.2, color=COLORS["TD3"], alpha=0.5)
        ax_g.axhline(0.0, lw=1.2, color=INK, zorder=0)
        ax_g.set_yscale("symlog", linthresh=1e-5)
        ax_g.set_title(
            "dQ1/d(bid) — negative pulls toward the band",
            loc="left", fontsize=10.5, color=INK,
        )
        ax_g.set_ylabel("dQ1/d(bid)")

        for ax in (ax_q, ax_g):
            ax.set_xlabel("bid price (EUR/MWh)")
            ax.set_xlim(bids[0], bids[-1])
            ax.grid(True, color=MUTED, alpha=0.2, lw=0.7)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=9)

    fig.suptitle(
        "ASSUME's own MATD3: where the critic points, and where the actor stands",
        x=0.006, y=1.0, ha="left", fontsize=13.5, fontweight="bold", color=INK,
    )
    fig.text(
        0.006, 0.963,
        "Q1 is the surface matd3.py has the actor ascend. Thin vertical lines are the "
        "actor's own bids, one per probed observation from the run's replay buffer.",
        fontsize=9, color=MUTED, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", nargs="+", default=None,
        help="study-case folder names under learned_strategies/; default = all "
        "that have last_policies/",
    )
    parser.add_argument("--grid", type=int, default=401)
    parser.add_argument(
        "--n-obs", type=int, default=12,
        help="observations sampled from the case's replay buffer",
    )
    parser.add_argument("--out", type=Path, default=OUT_DIR / "assume_critic_probe.png")
    args = parser.parse_args()

    names = args.cases or sorted(
        d.name for d in POLICIES.iterdir()
        if (d / "last_policies" / "critics").is_dir()
    )
    bids = np.linspace(-MAX_BID_PRICE, MAX_BID_PRICE, args.grid)

    results = []
    for name in names:
        case_dir = POLICIES / name / "last_policies"
        try:
            results.append(probe(case_dir, bids, args.n_obs))
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"  skipping {name}: {exc}")
    if not results:
        raise SystemExit("no saved policies found")

    report(results)
    plot(results, args.out)


if __name__ == "__main__":
    main()
