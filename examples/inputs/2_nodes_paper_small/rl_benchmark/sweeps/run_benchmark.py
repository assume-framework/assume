# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Compare RL algorithms on the inc-dec reward landscape.

Runs each algorithm for several seeds on :class:`incdec_env.IncDecEnv`, probing the
greedy (noise-free) policy at a fixed interval, and reports how close each one gets
to the cliff-edge optimum.

This code lives in the tracked scenario input folder; ``results.npz`` and
``benchmark.png`` are written to the scenario's *output* folder
(``examples/outputs/2_nodes_paper_small/rl_benchmark/``), which is gitignored.
Earlier runs are archived under ``runs/`` there, with a write-up in its README.

    python run_benchmark.py                       # all algorithms, 1 seed
    python run_benchmark.py --algos TD3 SAC --seeds 5 --timesteps 20000

    # sweep the thing that actually matters here -- exploration
    python run_benchmark.py --algos TD3 --seeds 5 --sigma 0.02
    python run_benchmark.py --algos TD3 --seeds 5 --noise-schedule linear
    python run_benchmark.py --algos TD3 --seeds 5 --warmup 250 --buffer-size 2000

Requires ``gymnasium`` and ``stable-baselines3``:

    pip install "gymnasium>=1.0" "stable-baselines3>=2.4"

Stable-Baselines3 is used rather than RLlib because the problem is a one-dimensional
contextual bandit -- SB3 gives the same TD3/DDPG/SAC/PPO implementations with a
fraction of the setup, and TD3 there is the closest single-agent analogue of the
MATD3 that ASSUME actually runs. Swapping in RLlib means replacing ``_train_one``
only; the env is a plain ``gymnasium.Env``.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from stable_baselines3.common.noise import ActionNoise

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _layout import OUT_DIR  # noqa: E402  (also puts the folders on sys.path)
from critic_probe import critic_curve  # noqa: E402
from incdec_env import IncDecEnv  # noqa: E402
from incdec_reward import PAPER_SMALL, reward_from_bid, sweep  # noqa: E402

HERE = Path(__file__).parent

# Categorical hues in fixed order -- an algorithm keeps its colour regardless of
# how many are plotted.
COLORS = {
    "TD3": "#2a78d6",
    "DDPG": "#eb6834",
    "SAC": "#1baf7a",
    "PPO": "#eda100",
    "Random search": "#e87ba4",
}
INK = "#0b0b0b"
MUTED = "#8a8985"

#: Diverging ramp for signed fields such as ``dQ/d(bid)``: two poles with a
#: *neutral* -- not a hue -- at the midpoint, so "no gradient" reads as absence of
#: colour rather than as a colour. It lives here, with the rest of the house
#: palette, because both ``critic_evolution.py`` and this module draw with it and
#: the analysis scripts already import their colours from here.
DIVERGING = LinearSegmentedColormap.from_list(
    "grad", ["#2a78d6", "#9dc2ec", "#f2f2f0", "#f4b79c", "#eb6834"]
)
#: Below this the symlog colour scale is linear. The field spans a ~1e-1 spike at
#: the cliff and a ~1e-5 background on the plateaus, so a plain linear scale would
#: show the spike and nothing else.
GRAD_LINTHRESH = 1e-4

DEFAULT_ALGOS = ["TD3", "DDPG", "SAC", "PPO", "Random search"]


@dataclass
class RunConfig:
    """One benchmark configuration. The exploration fields are the interesting
    ones: on this landscape 90% of the action space is exactly flat, so what an
    algorithm can learn is decided almost entirely by what its behaviour policy
    puts in the buffer."""

    timesteps: int = 10_000
    eval_every: int = 2_500
    learning_rate: float = 1e-3
    #: Uniform-random steps before any gradient step. SB3's ``learning_starts``;
    #: the counterpart of ASSUME's ``episodes_collecting_initial_experience``.
    warmup: int = 1_000
    #: Gaussian action-noise std in scaled action units. 0.1 == +-10 EUR/MWh.
    sigma: float = 0.1
    #: Sigma at the end of training when ``noise_schedule == "linear"``.
    sigma_end: float = 0.0
    #: ``"const"`` (SB3 default) or ``"linear"`` (ASSUME's ``action_noise_schedule``).
    noise_schedule: str = "const"
    buffer_size: int = 10_000
    batch_size: int = 256
    #: Environment steps between training blocks. 0 keeps SB3's default of one
    #: block per episode -- 24 steps here. ASSUME's counterpart is ``train_freq``,
    #: set to ``12h`` in the scenario config.
    train_freq: int = 0
    #: Gradient steps per training block. 0 keeps SB3's default of -1, i.e. one
    #: per environment step elapsed. ASSUME uses 32 per 12 steps, so the real
    #: setup takes ~2.7 gradient steps per environment step, not 1.
    gradient_steps: int = 0
    #: TD3 only. Critic updates per actor update. The idea was that letting the
    #: critic become accurate before the actor commits would keep it off the flat
    #: plateau. Measured: it does not (2 and 8 both end at the +100 ceiling,
    #: 64 lands at 68 +- 28). ASSUME's config uses 8.
    policy_delay: int = 2
    #: Output squashing of the TD3/DDPG actor. SB3 hardcodes ``tanh``, whose
    #: gradient underflows to *exactly* zero in float32 by |z| ~ 9. ASSUME's
    #: Actor defaults to ``softsign``, whose gradient decays polynomially and
    #: never reaches zero. See docs/actor_saturation.md.
    actor_activation: str = "tanh"
    save_models: bool = False
    #: Sweep the critic and its autograd gradient at every probe, on a grid of
    #: this many actions. 0 disables. Off-policy algorithms only -- PPO has no
    #: action-value critic to sweep.
    critic_grid: int = 0
    #: SAC only. ``"auto"`` tunes the entropy temperature against
    #: ``target_entropy``; a float pins it. See ``target_entropy``.
    ent_coef: str = "auto"
    #: SAC only. ``"auto"`` means ``-dim(A)`` == -1.0 here, which is a *floor* on
    #: policy entropy: it forbids a near-deterministic policy, so SAC cannot sit
    #: on a peak narrower than roughly its own spread. More negative == sharper
    #: policy permitted.
    target_entropy: str = "auto"


class LinearAnnealedNoise(ActionNoise):
    """Gaussian action noise whose sigma decays linearly after the warmup.

    SB3 ships only :class:`NormalActionNoise`, which is constant for the whole
    run. ASSUME anneals instead (``action_noise_schedule: linear``), and on a
    landscape whose informative band is 19 EUR wide that difference dominates:
    constant sigma=0.1 keeps kicking a converged actor 20 EUR across the cliff.

    Sigma is held at ``sigma_start`` through the warmup, then interpolated to
    ``sigma_end`` at ``total_steps``. The noise object is called exactly once per
    environment step, so the internal counter tracks ``num_timesteps``.
    """

    def __init__(
        self,
        sigma_start: float,
        sigma_end: float,
        warmup: int,
        total_steps: int,
        seed: int | None = None,
        shape: tuple[int, ...] = (1,),
    ):
        super().__init__()
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.warmup = warmup
        self.total_steps = total_steps
        self.shape = shape
        self.rng = np.random.default_rng(seed)
        self._t = 0
        self.sigma = sigma_start

    def __call__(self) -> np.ndarray:
        span = max(self.total_steps - self.warmup, 1)
        frac = float(np.clip((self._t - self.warmup) / span, 0.0, 1.0))
        self.sigma = self.sigma_start + frac * (self.sigma_end - self.sigma_start)
        self._t += 1
        return self.rng.normal(0.0, self.sigma, self.shape)

    def reset(self) -> None:  # noqa: D102 - episode boundaries carry no state here
        pass


def _maybe_float(value: str) -> str | float:
    """SB3 takes either the string ``"auto"`` or a number for these. Keep the
    CLI stringly-typed and convert here."""
    if isinstance(value, str) and value.startswith("auto"):
        return value
    return float(value)


def make_noise(cfg: RunConfig, seed: int) -> ActionNoise:
    from stable_baselines3.common.noise import NormalActionNoise

    if cfg.noise_schedule == "linear":
        return LinearAnnealedNoise(
            cfg.sigma, cfg.sigma_end, cfg.warmup, cfg.timesteps, seed=seed
        )
    if cfg.noise_schedule == "const":
        return NormalActionNoise(mean=np.zeros(1), sigma=cfg.sigma * np.ones(1))
    raise ValueError(f"unknown noise schedule {cfg.noise_schedule!r}")


def _set_actor_activation(model, activation: str):
    """Swap the TD3/DDPG actor's output squashing.

    SB3 builds the actor as ``create_mlp(..., squash_output=True)``, which appends
    a fixed ``nn.Tanh()``; there is no constructor argument for it. Replacing that
    last module is enough -- the rest of the network, the action bounds and the
    optimizer are all unchanged, since softsign shares tanh's ``(-1, 1)`` range.

    Both the actor and its Polyak target must be swapped, or the target would keep
    squashing differently from the online network.
    """
    import torch.nn as nn

    if activation == "tanh":
        return model
    if activation != "softsign":
        raise ValueError(f"unknown actor activation {activation!r}")

    for net in (model.actor, model.actor_target):
        if not isinstance(net.mu[-1], nn.Tanh):
            raise RuntimeError(
                f"expected a Tanh output layer, found {type(net.mu[-1]).__name__} "
                "-- SB3's actor layout has changed"
            )
        net.mu[-1] = nn.Softsign()
    return model


def make_model(algo: str, env, seed: int, cfg: RunConfig):
    """Build an SB3 model. Hyperparameters follow the ASSUME learning config where
    they have a counterpart (lr 1e-3, batch 256, sigma 0.1, gamma 0.99).

    Note which knobs reach which algorithm: ``warmup``/``buffer_size`` are
    off-policy only, and ``sigma`` reaches only TD3 and DDPG -- SAC explores via
    its entropy term and PPO via its own policy std, so neither takes external
    action noise.
    """
    from stable_baselines3 import DDPG, PPO, SAC, TD3

    off_policy = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg.learning_rate,
        buffer_size=cfg.buffer_size,
        learning_starts=cfg.warmup,
        batch_size=cfg.batch_size,
        gamma=0.99,
        tau=0.005,
        seed=seed,
        verbose=0,
        device="cpu",
    )
    # Left at SB3's defaults unless asked: train_freq=(1, "episode") with
    # gradient_steps=-1 gives one gradient step per environment step here.
    if cfg.train_freq:
        off_policy["train_freq"] = cfg.train_freq
    if cfg.gradient_steps:
        off_policy["gradient_steps"] = cfg.gradient_steps

    if algo == "TD3":
        return _set_actor_activation(
            TD3(
                action_noise=make_noise(cfg, seed),
                policy_delay=cfg.policy_delay,
                **off_policy,
            ),
            cfg.actor_activation,
        )
    if algo == "DDPG":
        return _set_actor_activation(
            DDPG(action_noise=make_noise(cfg, seed), **off_policy),
            cfg.actor_activation,
        )
    if algo == "SAC":
        # SAC has its own entropy-driven exploration, so no action noise.
        return SAC(
            ent_coef=_maybe_float(cfg.ent_coef),
            target_entropy=_maybe_float(cfg.target_entropy),
            **off_policy,
        )
    if algo == "PPO":
        # On-policy: no replay buffer and no warmup phase at all. It starts from
        # log_std_init=0.0, i.e. std 1.0 in scaled units, which after clipping is
        # close to the uniform warmup the off-policy algorithms get -- then
        # narrows as the entropy bonus decays.
        return PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            gamma=0.99,
            seed=seed,
            verbose=0,
            device="cpu",
        )
    raise ValueError(f"unknown algorithm {algo!r}")


def _train_one(
    algo: str, seed: int, cfg: RunConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Train one (algorithm, seed) and record its full bidding history.

    Returns
    -------
    (steps, greedy_bids, placed_bids, critic_q, critic_grad)
        ``steps`` are the probe timesteps and ``greedy_bids`` the noise-free bid
        the policy would place at each of them, both in EUR/MWh. ``placed_bids``
        is every bid actually placed, one per environment step -- the behaviour
        policy including exploration noise, which is what fills the replay buffer.
        The last two are ``(n_probes, critic_grid)`` sweeps of the actor's
        objective and its autograd gradient, or ``None`` when ``critic_grid`` is
        0 or the algorithm has no action-value critic.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    env = IncDecEnv()
    probe_obs, _ = env.reset(seed=seed)

    steps: list[int] = []
    bids: list[float] = []
    q_snaps: list[np.ndarray] = []
    g_snaps: list[np.ndarray] = []

    if algo == "Random search":
        # Reference line: what uniform exploration alone would find, if you could
        # argmax over everything it has seen. This is the bar a learner must beat.
        rng = np.random.default_rng(seed)
        best_bid, best_reward = 0.0, -np.inf
        placed: list[float] = []
        for t in range(1, cfg.timesteps + 1):
            bid = rng.uniform(-1, 1) * PAPER_SMALL.max_bid_price
            placed.append(bid)
            r = float(reward_from_bid(bid, PAPER_SMALL))
            if r > best_reward:
                best_reward, best_bid = r, bid
            if t % cfg.eval_every == 0:
                steps.append(t)
                bids.append(best_bid)
        return np.array(steps), np.array(bids), np.array(placed), None, None

    # PPO's critic is a state-value V(s), not Q(s,a), so there is no action
    # sweep to take.
    record_critic = cfg.critic_grid > 0 and algo != "PPO"
    grid = np.linspace(-1.0, 1.0, cfg.critic_grid) if record_critic else None

    class Probe(BaseCallback):
        def _on_step(self) -> bool:
            if self.num_timesteps % cfg.eval_every == 0:
                action, _ = self.model.predict(probe_obs, deterministic=True)
                steps.append(self.num_timesteps)
                bids.append(float(action[0]) * PAPER_SMALL.max_bid_price)
                if record_critic:
                    q, g = critic_curve(
                        self.model, probe_obs, grid, PAPER_SMALL.max_bid_price
                    )
                    q_snaps.append(q)
                    g_snaps.append(g)
            return True

    model = make_model(algo, env, seed, cfg)
    model.learn(total_timesteps=cfg.timesteps, callback=Probe(), progress_bar=False)

    if cfg.save_models:
        # keep the trained networks so the critic can be inspected afterwards
        # without paying for the run again -- see critic_landscape.py
        model_dir = OUT_DIR / "models"
        model_dir.mkdir(exist_ok=True)
        model.save(model_dir / f"{algo.replace(' ', '_')}_seed{seed}")

    # PPO always completes a full n_steps rollout, so it overshoots
    # total_timesteps and probes once more than the off-policy algorithms.
    # Trim to the common grid so every algorithm shares one x axis.
    steps_arr, bids_arr = np.array(steps), np.array(bids)
    keep = steps_arr <= cfg.timesteps
    return (
        steps_arr[keep],
        bids_arr[keep],
        np.array(env.bid_history[: cfg.timesteps]),
        np.array(q_snaps)[keep] if q_snaps else None,
        np.array(g_snaps)[keep] if g_snaps else None,
    )


def probe_grid(cfg: RunConfig) -> np.ndarray:
    """The canonical probe timesteps every algorithm is aligned to."""
    return np.arange(cfg.eval_every, cfg.timesteps + 1, cfg.eval_every)


def run(algos: list[str], seeds: int, cfg: RunConfig):
    """Returns ``(probe_steps, greedy_bids, placed_bids, critic)``. The bid dicts
    are keyed by algorithm with one row per seed; ``critic`` maps algorithm to
    ``(q, grad)`` arrays of shape ``(seeds, n_probes, critic_grid)``, and is empty
    unless ``cfg.critic_grid`` is set."""
    greedy: dict[str, np.ndarray] = {}
    placed: dict[str, np.ndarray] = {}
    critic: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    steps = probe_grid(cfg)

    for algo in algos:
        per_seed_greedy, per_seed_placed = [], []
        per_seed_q, per_seed_g = [], []
        t0 = time.perf_counter()
        for seed in range(seeds):
            algo_steps, bids, bid_history, q, g = _train_one(algo, seed, cfg)
            if not np.array_equal(algo_steps, steps):
                raise RuntimeError(
                    f"{algo} seed {seed} probed {len(algo_steps)} times, "
                    f"expected {len(steps)}"
                )
            per_seed_greedy.append(bids)
            per_seed_placed.append(bid_history)
            if q is not None:
                per_seed_q.append(q)
                per_seed_g.append(g)
        greedy[algo] = np.vstack(per_seed_greedy)
        placed[algo] = np.vstack(per_seed_placed)
        if per_seed_q:
            critic[algo] = (np.stack(per_seed_q), np.stack(per_seed_g))
        print(f"  {algo:<14} {time.perf_counter() - t0:6.1f}s")

    return steps, greedy, placed, critic


def save_results(
    path: Path,
    steps: np.ndarray,
    greedy: dict[str, np.ndarray],
    placed: dict[str, np.ndarray],
    cfg: RunConfig,
    critic: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> None:
    """Persist a run. Written before plotting, so a plotting failure never costs
    a training run -- recover it with ``--replot``."""
    critic = critic or {}
    path.parent.mkdir(parents=True, exist_ok=True)

    extra: dict[str, np.ndarray] = {}
    if critic:
        # the action grid the sweeps were taken on, in EUR/MWh
        extra["critic_bids"] = (
            np.linspace(-1.0, 1.0, cfg.critic_grid) * PAPER_SMALL.max_bid_price
        )
        for algo, (q, g) in critic.items():
            extra[f"critic_q/{algo}"] = q
            extra[f"critic_grad/{algo}"] = g

    np.savez(
        path,
        steps=steps,
        **{f"greedy/{k}": v for k, v in greedy.items()},
        **{f"placed/{k}": v for k, v in placed.items()},
        **{f"cfg/{k}": v for k, v in asdict(cfg).items()},
        **extra,
    )


def load_results(path: Path, fallback: RunConfig) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    RunConfig,
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    """Reload a saved run for re-plotting. Runs saved before the config was
    recorded fall back to the config given on the command line; runs recorded
    without ``--critic-grid`` come back with an empty ``critic`` dict."""
    data = np.load(path)
    steps = data["steps"]

    greedy, placed = {}, {}
    critic_q, critic_grad = {}, {}
    for key in data.files:
        prefix, _, algo = key.partition("/")
        if prefix == "greedy":
            # trim PPO's overshoot in runs saved before the grid was aligned
            greedy[algo] = data[key][:, : len(steps)]
        elif prefix == "placed":
            placed[algo] = data[key]
        elif prefix == "critic_q":
            critic_q[algo] = data[key]
        elif prefix == "critic_grad":
            critic_grad[algo] = data[key]

    critic = {a: (critic_q[a], g) for a, g in critic_grad.items() if a in critic_q}

    # ``from __future__ import annotations`` makes field.type a string, so take
    # the concrete type from the fallback instance instead of the annotation.
    values = {
        f.name: type(getattr(fallback, f.name))(data[f"cfg/{f.name}"].item())
        for f in fields(RunConfig)
        if f"cfg/{f.name}" in data.files
    }
    cfg = RunConfig(**{**asdict(fallback), **values}) if values else fallback

    return steps, greedy, placed, cfg, critic


def describe(cfg: RunConfig) -> str:
    """One-line summary of the exploration settings, for the console and the figure."""
    p = PAPER_SMALL
    if cfg.noise_schedule == "linear":
        sigma = f"sigma {cfg.sigma:g}->{cfg.sigma_end:g} linear"
    else:
        sigma = f"sigma {cfg.sigma:g} const"
    reach = 1.96 * cfg.sigma * p.max_bid_price
    parts = [
        f"warmup {cfg.warmup}",
        f"{sigma} (+-{reach:.0f} EUR at 95%, band is "
        f"{p.eom_price - p.dec_threshold:.0f} EUR)",
        f"buffer {cfg.buffer_size}",
        f"lr {cfg.learning_rate:g}",
    ]
    # the two settings that decide whether this landscape is solvable at all --
    # spell them out rather than leaving a figure that looks like a default run
    if cfg.actor_activation != "tanh":
        parts.append(f"actor {cfg.actor_activation} (TD3/DDPG)")
    if cfg.ent_coef != "auto":
        parts.append(f"ent_coef {cfg.ent_coef} (SAC)")
    if cfg.policy_delay != 2:
        parts.append(f"policy_delay {cfg.policy_delay}")
    return " | ".join(parts)


def summarize(results: dict[str, np.ndarray], tol: float = 0.5) -> None:
    """Print a table of final performance. ``tol`` is the bid tolerance in EUR/MWh
    within which a seed counts as having found the optimum."""
    opt_bid = PAPER_SMALL.optimal_bid
    opt_reward = PAPER_SMALL.optimal_reward

    print(f"\n  optimum: bid {opt_bid:.2f} EUR/MWh -> reward {opt_reward:+.3f}")
    print(
        f"\n  {'algorithm':<14} {'final bid':>18} {'reward':>16} "
        f"{'regret':>8} {f'hit (<{tol} EUR)':>16}"
    )
    print("  " + "-" * 76)

    for algo, bids in results.items():
        final = bids[:, -1]
        rewards = reward_from_bid(final, PAPER_SMALL)
        hit = np.mean(np.abs(final - opt_bid) <= tol)
        print(
            f"  {algo:<14} {np.mean(final):>8.2f} +- {np.std(final):<6.2f} "
            f"{np.mean(rewards):>+8.3f} +- {np.std(rewards):<5.3f} "
            f"{opt_reward - np.mean(rewards):>8.3f} "
            f"{hit:>15.0%}"
        )


def plot(
    steps: np.ndarray,
    results: dict[str, np.ndarray],
    placed: dict[str, np.ndarray],
    cfg: RunConfig,
    out: Path,
    critic: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    critic_seed: int = 0,
) -> None:
    """Draw the comparison figure.

    Two rows always: the landscape with the reward curves, then one learning
    history per algorithm. A run recorded with ``--critic-grid`` gets a third row
    showing the critic gradient field each actor was climbing, on the *same* axes
    as the history above it, so a column reads straight down.
    """
    p = PAPER_SMALL
    names = list(results)
    critic = critic or {}
    # Only algorithms that both appear in this figure and carry a sweep. PPO's
    # critic is a state-value V(s) and random search has no network, so the row
    # is narrower than the one above it -- see the note drawn in the gap.
    critic_names = [a for a in names if a in critic]

    if critic_names:
        fig = plt.figure(figsize=(16, 15))
        outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.05, 1.05], hspace=0.30)
    else:
        fig = plt.figure(figsize=(16, 9.5))
        outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.32)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.5], wspace=0.2)
    bottom = outer[1].subgridspec(1, len(names), wspace=0.1)

    def row_heading(spec, text: str) -> None:
        """Left-margin heading for a faceted row, placed from the grid rather
        than from a hardcoded figure fraction, so adding a row cannot slide it
        onto the panels above. The offset is in inches -- 0.3 clears the facets'
        own titles, which sit in the same band -- so it does not shrink as the
        figure grows taller."""
        fig.text(
            0.007,
            spec.get_position(fig).y1 + 0.3 / fig.get_figheight(),
            text,
            fontsize=11,
            color=INK,
            va="bottom",
        )

    ax_land = fig.add_subplot(top[0])
    ax_curve = fig.add_subplot(top[1])

    # --- panel A: the landscape the algorithms are searching -----------------
    bids, rewards = sweep(p)
    ax_land.axvspan(
        -p.max_bid_price, p.dec_threshold, color="#e34948", alpha=0.06, lw=0
    )
    ax_land.axvspan(p.dec_threshold, p.eom_price, color="#1baf7a", alpha=0.08, lw=0)
    ax_land.axvspan(p.eom_price, p.max_bid_price, color=MUTED, alpha=0.10, lw=0)
    ax_land.plot(bids, rewards, lw=2, color=COLORS["TD3"], solid_capstyle="round")
    ax_land.plot([p.optimal_bid], [p.optimal_reward], "o", ms=9, color=COLORS["TD3"])
    ax_land.annotate(
        f"optimum\nbid {p.optimal_bid:.0f} -> {p.optimal_reward:+.2f}",
        xy=(p.optimal_bid, p.optimal_reward),
        xytext=(p.optimal_bid + 22, p.optimal_reward - 0.02),
        color=INK,
        fontsize=9,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=1),
    )
    ax_land.text(
        -95, 0.15, "not dec'd\n(loss)", color="#b03a39", fontsize=8.5, va="top"
    )
    ax_land.text(
        p.eom_price + 4, 0.15, "out of market\n(no gradient)", color=MUTED, fontsize=8.5, va="top"
    )
    ax_land.set_title("the landscape", loc="left", fontsize=11, color=INK)
    ax_land.set_xlabel("bid price (EUR/MWh)")
    ax_land.set_ylabel("reward")

    # --- panel B: greedy bid quality over training ---------------------------
    for algo, seeds in results.items():
        r = reward_from_bid(seeds, p)
        color = COLORS.get(algo, MUTED)
        ax_curve.fill_between(
            steps, r.min(axis=0), r.max(axis=0), color=color, alpha=0.15, lw=0
        )
        ax_curve.plot(steps, r.mean(axis=0), lw=2, color=color, label=algo)
        # No direct end-labels here: the failing algorithms all converge to
        # exactly 0.0, so their labels would stack on one point. The legend and
        # the per-algorithm facets below carry identity instead.
    ax_curve.axhline(p.optimal_reward, ls="--", lw=1.2, color=INK, zorder=0)
    ax_curve.annotate(
        "optimum",
        xy=(steps[0], p.optimal_reward),
        xytext=(2, 4),
        textcoords="offset points",
        fontsize=8.5,
        color=INK,
    )
    ax_curve.axhline(0.0, ls=":", lw=1.2, color=MUTED, zorder=0)
    ax_curve.annotate(
        "zero plateau",
        xy=(steps[0], 0.0),
        xytext=(2, 4),
        textcoords="offset points",
        fontsize=8.5,
        color=MUTED,
    )
    ax_curve.set_title(
        "reward of the greedy policy (mean, min-max over seeds)",
        loc="left",
        fontsize=11,
        color=INK,
    )
    ax_curve.set_xlabel("environment steps")
    ax_curve.set_ylabel("reward")
    ax_curve.set_xlim(steps[0], steps[-1])
    ax_curve.legend(frameon=False, fontsize=9, loc="lower right", ncols=3)

    # --- row 2: the learning history, one panel per algorithm ----------------
    # Faceted rather than overlaid: five clouds of bids on one axis would be
    # unreadable, and the shape of each trajectory is the point.
    hist_axes = []
    n_steps = placed[names[0]].shape[1]
    step_axis = np.arange(1, n_steps + 1)

    for i, algo in enumerate(names):
        ax = fig.add_subplot(bottom[i], sharey=hist_axes[0] if hist_axes else None)
        hist_axes.append(ax)
        color = COLORS.get(algo, MUTED)

        ax.axhspan(p.dec_threshold, p.eom_price, color="#1baf7a", alpha=0.10, lw=0)
        ax.axhline(p.optimal_bid, ls="--", lw=1.2, color=INK, zorder=2)

        # the warmup boundary -- only off-policy algorithms have one
        if algo not in ("PPO", "Random search") and 0 < cfg.warmup < n_steps:
            ax.axvline(cfg.warmup, ls=":", lw=1.2, color=INK, zorder=2)
            ax.annotate(
                "warmup ends",
                xy=(cfg.warmup, -p.max_bid_price),
                xytext=(3, 6),
                textcoords="offset points",
                rotation=90,
                fontsize=7.5,
                color=INK,
            )

        # every bid actually placed, across all seeds -- the exploration cloud
        for seed_bids in placed[algo]:
            ax.plot(
                step_axis,
                seed_bids,
                ".",
                ms=1.6,
                alpha=0.25,
                color=color,
                rasterized=True,
                zorder=1,
            )
        # the noise-free policy on top
        for seed_greedy in results[algo]:
            ax.plot(steps, seed_greedy, lw=1.8, color=color, zorder=3)

        ax.set_ylim(-p.max_bid_price, p.max_bid_price)
        ax.set_xlim(0, n_steps)
        # Facets sit shoulder to shoulder, so spell the ticks short enough that
        # neighbouring axes cannot run their labels together.
        ticks = np.linspace(0, n_steps, 3)
        ax.set_xticks(ticks, [f"{t / 1000:g}k" if t else "0" for t in ticks])
        ax.set_title(algo, loc="left", fontsize=10, color=color)
        ax.set_xlabel("environment steps")
        if i == 0:
            ax.set_ylabel("bid price (EUR/MWh)")
            ax.annotate(
                f"dec'd band\n{p.dec_threshold:.0f}-{p.eom_price:.0f}",
                xy=(0.03, p.eom_price + 4),
                xycoords=("axes fraction", "data"),
                fontsize=8,
                color="#137f59",
            )
        else:
            ax.tick_params(labelleft=False)

    row_heading(
        outer[1], "learning history: every bid placed (dots) and the greedy policy (line)"
    )

    # --- row 3: the gradient field the actor was climbing --------------------
    # Same x (environment steps) and y (bid price) as the facet directly above,
    # and shared with it, so a column reads straight down: where the policy went,
    # then the field that sent it there. The actor never sees the reward -- it
    # ascends the critic -- so this is the landscape that actually drove row 2.
    heat_axes = []
    if critic_names:
        third = outer[2].subgridspec(1, len(names), wspace=0.1)
        # the action grid the sweeps were taken on, in EUR/MWh (see save_results)
        n_grid = critic[critic_names[0]][1].shape[-1]
        grid_bids = np.linspace(-1.0, 1.0, n_grid) * p.max_bid_price
        seed = min(critic_seed, critic[critic_names[0]][1].shape[0] - 1)
        # one colour scale for the whole row, so the panels are comparable
        vmax = max(np.abs(critic[a][1][seed]).max() for a in critic_names)
        norm = SymLogNorm(
            linthresh=GRAD_LINTHRESH, vmin=-vmax, vmax=vmax, base=10
        )

        for i, algo in enumerate(names):
            if algo not in critic:
                continue
            ax = fig.add_subplot(third[i], sharex=hist_axes[i], sharey=hist_axes[i])
            heat_axes.append(ax)
            grad = critic[algo][1][seed]  # (probes, grid)
            mesh = ax.pcolormesh(
                steps,
                grid_bids,
                grad.T,
                cmap=DIVERGING,
                norm=norm,
                shading="nearest",
                rasterized=True,
            )
            for y in (p.dec_threshold, p.eom_price):
                ax.axhline(y, color=INK, lw=0.9, alpha=0.45, ls="--")
            if 0 < cfg.warmup < steps[-1]:
                # left of this the critic has taken no gradient step at all --
                # the field there is the random initialisation
                ax.axvline(cfg.warmup, ls=":", lw=1.2, color=INK, zorder=2)
            # the greedy policy again, haloed so it stays readable over both poles
            ax.plot(steps, results[algo][seed], lw=3.2, color="white", zorder=3)
            ax.plot(steps, results[algo][seed], lw=1.4, color=INK, zorder=4)

            ax.set_title(algo, loc="left", fontsize=10, color=COLORS.get(algo, INK))
            ax.set_xlabel("environment steps")
            if i == 0:
                ax.set_ylabel("bid price (EUR/MWh)")
            else:
                ax.tick_params(labelleft=False)

        # --- the gap where PPO and random search would be -------------------
        # One shared colourbar for the row -- the panels are only comparable
        # because they share a scale, and per-panel bars would hide that -- plus
        # the reason those two algorithms have no panel. Better than a blank:
        # the absence is itself a fact about them.
        spare = [i for i, a in enumerate(names) if a not in critic]
        gap = cax = None
        # Only borrow the gap when it is one unbroken block of columns; an
        # algorithm without a critic sitting *between* two that have one would
        # otherwise put the legend on top of a panel.
        if spare and spare == list(range(spare[0], spare[-1] + 1)):
            gap = fig.add_subplot(third[0, spare[0] : spare[-1] + 1])
            gap.axis("off")
            cax = gap.inset_axes([0.04, 0.60, 0.66, 0.022])

        cbar = fig.colorbar(
            mesh,
            ax=None if cax is not None else heat_axes,
            cax=cax,
            orientation="horizontal",
            # pad clears the panels' own x labels, which sit in between
            **({} if cax is not None else {"fraction": 0.04, "pad": 0.22}),
        )
        cbar.ax.tick_params(colors=MUTED, labelsize=8)
        cbar.outline.set_visible(False)
        if cax is None:
            cbar.set_label(
                "dQ/d(bid) of the actor's own objective  (autograd; symlog below "
                "1e-4).  Black line: the greedy policy.",
                fontsize=9,
                color=MUTED,
            )

        if gap is not None:
            # Written out by hand rather than with set_label/annotations at one
            # height: three captions on the colourbar's own baseline collide.
            # Each line gets its own band, top to bottom.
            gap.text(
                0.04,
                0.94,
                "the black line is the greedy policy -- the same line as the "
                "facet directly above.",
                fontsize=9,
                color=INK,
                va="top",
                transform=gap.transAxes,
            )
            gap.text(
                0.04,
                0.86,
                "no critic panel for "
                + " or ".join(names[i] for i in spare)
                + ": PPO's critic is a state-value V(s), which has no gradient\n"
                "with respect to the action, and random search has no network "
                "at all.",
                fontsize=9,
                color=MUTED,
                va="top",
                transform=gap.transAxes,
            )
            gap.text(
                0.04,
                0.68,
                "dQ/d(bid) of the actor's own objective  (autograd; symlog below "
                "1e-4)",
                fontsize=9,
                color=MUTED,
                va="bottom",
                transform=gap.transAxes,
            )
            gap.text(
                0.04,
                0.50,
                "◀  pulls the bid down",
                fontsize=8.5,
                color=COLORS["TD3"],
                va="top",
                ha="left",
                transform=gap.transAxes,
            )
            gap.text(
                0.70,
                0.50,
                "pushes it up  ▶",
                fontsize=8.5,
                color=COLORS["DDPG"],
                va="top",
                ha="right",
                transform=gap.transAxes,
            )
            gap.text(
                0.04,
                0.36,
                "Reading a column: the actor climbs this field, never the reward.\n"
                "An all-orange panel means the critic prefers a higher bid "
                "everywhere,\n"
                "so the actor runs to the +100 ceiling on a correct gradient over an\n"
                "incomplete critic. Blue opening up over [49, 100] is the descent "
                "path\n"
                "back into the band. Mottling is a flat region fitted correctly: a\n"
                "converged critic on a plateau has only a noise gradient left.",
                fontsize=8.5,
                color=INK,
                va="top",
                linespacing=1.5,
                transform=gap.transAxes,
            )

        row_heading(
            outer[2],
            f"the field the actor is climbing: dQ/d(bid) of its own objective "
            f"(seed {seed} of {critic[critic_names[0]][1].shape[0]})",
        )

    for ax in (ax_land, ax_curve, *hist_axes, *heat_axes):
        # No grid over a heatmap: the mesh is the data, and rules on top of it
        # read as structure that is not there.
        ax.grid(ax not in heat_axes, color=MUTED, alpha=0.2, lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)

    # Title and strapline sit above the axes in *inches*, not in figure
    # fractions, so the header keeps its spacing when a row is added.
    height = fig.get_figheight()
    fig.suptitle(
        "Can RL find the inc-dec optimum?  diesel_0, 2_nodes_paper_small",
        x=0.007,
        y=1 + 0.05 / height,
        ha="left",
        fontsize=13.5,
        fontweight="bold",
        color=INK,
    )
    # the settings are the experiment -- record them so swept figures stay legible
    fig.text(
        0.007, 1 - 0.26 / height, describe(cfg), fontsize=9, color=MUTED, ha="left"
    )

    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--algos", nargs="+", default=DEFAULT_ALGOS)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=2_500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--warmup",
        type=int,
        default=1_000,
        help="uniform-random steps before the first gradient step (off-policy only)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.1,
        help="action-noise std in scaled units; 0.1 == +-10 EUR/MWh (TD3/DDPG only)",
    )
    parser.add_argument(
        "--sigma-end", type=float, default=0.0, help="target sigma for --noise-schedule linear"
    )
    parser.add_argument(
        "--noise-schedule",
        choices=("const", "linear"),
        default="linear",
        help="linear is default (contrary to SB3: const); linear matches ASSUME's action_noise_schedule",
    )
    parser.add_argument(
        "--buffer-size", type=int, default=10_000, help="replay buffer size (off-policy only)"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--train-freq",
        type=int,
        default=0,
        help="environment steps between training blocks; 0 = SB3 default (once "
        "per 24-step episode). ASSUME uses 12",
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=0,
        help="gradient steps per training block; 0 = SB3 default (-1). ASSUME uses 32",
    )
    parser.add_argument(
        "--ent-coef",
        default="auto",
        help="SAC entropy temperature: 'auto' or a float, e.g. 0.001",
    )
    parser.add_argument(
        "--target-entropy",
        default="auto",
        help="SAC entropy floor: 'auto' (== -1.0 here) or a float; more negative "
        "permits a sharper policy",
    )
    parser.add_argument(
        "--policy-delay",
        type=int,
        default=2,
        help="TD3 critic updates per actor update; ASSUME's config uses 8",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="write trained networks to models/ for critic_landscape.py",
    )
    parser.add_argument(
        "--actor-activation",
        choices=("tanh", "softsign"),
        default="tanh",
        help="TD3/DDPG output squashing. tanh is SB3's; softsign is ASSUME's "
        "default and keeps a non-zero gradient when saturated",
    )
    parser.add_argument(
        "--critic-grid",
        type=int,
        default=0,
        help="sweep the critic and its autograd gradient at every probe, on this "
        "many actions (e.g. 201); 0 disables. Off-policy algorithms only",
    )
    parser.add_argument(
        "--critic-seed",
        type=int,
        default=0,
        help="which seed's critic field the bottom row of the figure draws; the "
        "rest of the figure keeps all seeds",
    )
    parser.add_argument(
        "--out", type=Path, default=OUT_DIR / "benchmark.png", help="output figure path"
    )
    parser.add_argument(
        "--results", type=Path, default=OUT_DIR / "results.npz", help="run data path"
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help="re-draw the figure from --results without retraining",
    )
    args = parser.parse_args()

    cfg = RunConfig(
        timesteps=args.timesteps,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
        warmup=args.warmup,
        sigma=args.sigma,
        sigma_end=args.sigma_end,
        noise_schedule=args.noise_schedule,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        policy_delay=args.policy_delay,
        actor_activation=args.actor_activation,
        save_models=args.save_models,
        critic_grid=args.critic_grid,
        ent_coef=args.ent_coef,
        target_entropy=args.target_entropy,
    )

    if args.replot:
        steps, greedy, placed, cfg, critic = load_results(args.results, cfg)
        print(f"\n  replotting {args.results}\n  {describe(cfg)}\n")
        summarize(greedy)
        plot(steps, greedy, placed, cfg, args.out, critic, args.critic_seed)
        return

    print(f"\n  {args.seeds} seeds x {cfg.timesteps} steps\n  {describe(cfg)}\n")
    if cfg.buffer_size < cfg.timesteps:
        print(
            f"  note: buffer ({cfg.buffer_size}) < timesteps ({cfg.timesteps}) -- the "
            f"uniform warmup data is evicted after step {cfg.buffer_size}\n"
        )

    steps, greedy, placed, critic = run(args.algos, args.seeds, cfg)

    summarize(greedy)
    save_results(args.results, steps, greedy, placed, cfg, critic)
    if critic:
        print(
            f"  critic sweeps: {len(critic)} algo(s) x {len(steps)} probes "
            f"x {cfg.critic_grid} actions -> {args.results.name}"
        )
    plot(steps, greedy, placed, cfg, args.out, critic, args.critic_seed)


if __name__ == "__main__":
    main()
