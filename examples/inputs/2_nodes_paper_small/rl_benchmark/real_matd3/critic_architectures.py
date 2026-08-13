# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Drop-in replacements for ASSUME's ``CriticTD3``, from the literature.

Workstream B of ``HANDOFF.md``. ``act_share`` -- the quantity runs 12 and 13
arrived at -- is not a mechanism anyone else uses, and both levers that raise it
are experiment monkeypatches. The literature has two standard designs that act
on the same thing, and this module implements them so they can be screened
offline before any cluster time is spent.

Every variant here has the **same constructor signature and the same
``forward`` / ``q1_forward`` interface** as
``assume.reinforcement_learning.neural_network_architecture.CriticTD3``, so a
live sweep needs one line before the world loads::

    import assume.reinforcement_learning.algorithms.matd3 as matd3
    matd3.CriticTD3 = build("simba")

and **no change to ``assume/`` at all** (``matd3.py:388`` and ``:396``
construct the class by name from that module's globals).

The variants
------------
``baseline``
    ASSUME's own ``CriticTD3``, re-exported so tables have a control row.
    Observation and action are concatenated at the *input* layer, hidden
    ``[256, 128]``, Xavier, ReLU, no normalization anywhere.

``late``
    **Late action injection** -- the DDPG default since Lillicrap et al. (2016),
    whose supplementary §7 states plainly: *"Actions were not included until the
    2nd hidden layer of Q."* The action gets its own weight matrix instead of
    being 1 of ``obs_dim + act_dim`` columns of the first one. This is the
    literature's version of "make the critic notice the action", with no free
    scale parameter and no discarded observation dimensions. Cheapest and most
    defensible test in the plan.

``bn+late``
    The rest of what the same paper prescribes: batch normalization *"on the
    state input and all layers of the Q network prior to the action input"*.
    That is, it normalizes the observation and deliberately keeps the action
    out of the normalized path. Its own condition because it is a claim about
    observations, not about the action.

``rsnorm``
    SimBa's observation normalizer alone, on the baseline MLP. Per-dimension
    standardization by *running* mean/variance (Lee et al., ICLR 2025, §4).

``rsnorm+late``
    Both single-mechanism levers at once.

``simba``
    Full SimBa: RSNorm, a linear embedding, ``N`` pre-LayerNorm residual
    feedforward blocks with an inverted bottleneck (hidden ``4 * d_h``), and a
    post-LayerNorm before the output head. Paper defaults for the critic:
    ``N = 2`` blocks at width ``512`` (their Table 7).

``simba-small``
    Same topology at ``d_h = 128``, which lands near the baseline's parameter
    count. Separates "SimBa's shape helps" from "more parameters help" -- the
    paper's own claim is that the shape is what lets the parameters help, so
    both rows are needed to say anything.

``simba-nornorm``
    SimBa with RSNorm removed. The ablation that isolates whether the
    normalization or the residual path is carrying the result, which is exactly
    the open question ``HANDOFF.md`` §B.3 poses: run 12 found z-scoring the
    observation was the *worst* cell in its ladder, while observation
    standardization is SimBa's single most important component. Both cannot be
    the whole story.

``simba+late``
    SimBa with the action injected after the embedding rather than
    concatenated before it. Not in either paper; recorded because the two
    mechanisms are independent and the screen is cheap.

Nothing here changes the optimizer, the loss, ``gamma``, or the target update.
The SimBa paper pairs its architecture with AdamW at weight decay 1e-2 and
lr 1e-4, which is a *training* change and is deliberately not baked in --
``matd3.py`` already uses AdamW, and the learning rate is a config knob, so
mixing them into the architecture row would confound the screen.

Usage::

    from critic_architectures import build, REGISTRY, describe
    critic = build("simba")(n_agents=1, obs_dim=50, act_dim=1,
                            float_type=th.float32, unique_obs_dim=2)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch as th
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _layout import SCENARIO  # noqa: E402  (also sets sys.path)

sys.path.insert(0, str(SCENARIO.parents[2]))
from assume.reinforcement_learning.neural_network_architecture import (  # noqa: E402
    CriticTD3,
)

__all__ = ["REGISTRY", "build", "describe", "RSNorm", "SimBaBlock", "CriticTD3"]


# --------------------------------------------------------------- shared pieces


def _hidden_sizes(n_agents: int) -> list[int]:
    """``CriticTD3``'s own width schedule, so variants stay comparable to it."""
    if n_agents <= 20:
        return [256, 128]
    if n_agents <= 50:
        return [512, 256, 128]
    return [1024, 512, 256, 128]


def _xavier(module: nn.Module) -> None:
    """``CriticTD3``'s initialization, applied to the ``nn.Linear`` layers only.

    ``nn.LayerNorm`` keeps its own default (weight 1, bias 0), which is what
    SimBa expects -- Xavier on a normalization gain would be meaningless.
    """

    def init_layer(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    module.apply(init_layer)


def _stats_to_params(module: nn.Module) -> None:
    """Re-register a module's float running statistics as non-grad Parameters.

    **This is load-bearing, not tidying.** ``matd3.py:766`` syncs the target
    critic with ``polyak_update``, which zips ``critics.parameters()`` against
    ``target_critics.parameters()`` -- and ``parameters()`` does **not** yield
    buffers. A normalizer that keeps its running mean/variance in the usual
    ``register_buffer`` slot would therefore never reach the target critic: the
    online critic would standardize its inputs while the target critic
    normalized with the untouched initial statistics, so ``Q`` and ``Q_target``
    would be functions of differently scaled inputs and the TD target would be
    quietly wrong. Nothing would raise.

    ``nn.Parameter(..., requires_grad=False)`` fixes it with no change to
    ``assume/``: such tensors *are* yielded by ``parameters()``, so Polyak
    tracks them (the target's statistics then lag the online ones by ``tau``,
    which is the right semantics for a target network); they are still in
    ``state_dict()`` so save/load is unchanged; and ``AdamW`` skips them
    because ``p.grad is None``.

    Integer counters such as ``BatchNorm``'s ``num_batches_tracked`` are left
    as buffers -- ``lerp_`` is undefined on int64, and both critics are the
    same class so the two parameter lists still line up.
    """
    for name, buf in list(module.named_buffers(recurse=True)):
        if buf is None or not th.is_floating_point(buf):
            continue
        parent = module
        *path, leaf = name.split(".")
        for part in path:
            parent = getattr(parent, part)
        del parent._buffers[leaf]
        parent.register_parameter(leaf, nn.Parameter(buf.clone(), requires_grad=False))


class RSNorm(nn.Module):
    """Running Statistics Normalization -- SimBa (Lee et al., ICLR 2025) §4.

    Standardizes each input dimension by the running mean and variance seen so
    far, so no dimension dominates the first layer purely by scale::

        o_bar = (o - mu_t) / sqrt(sigma^2_t + eps)

    The paper's Eq. 3 is a per-sample recursion; this is its exact batch
    generalization (Chan et al.'s parallel Welford update), because the critic
    is fed a replay batch at a time and updating 128 times per gradient step
    would be both slower and a different filter.

    **``forward`` never updates.** The statistics are folded in by an explicit
    ``update`` call from the owning critic, once per replay batch, on the
    twin-Q path only. That matters for two reasons: ``q1_forward`` is called
    again on the *same* batch during the actor update (``matd3.py:713``), so
    updating inside ``forward`` would count every batch twice and, worse, make
    ``q1_forward(obs, a)`` differ from the ``q1`` that ``forward(obs, a)``
    just returned -- a critic that is not a function of its inputs. The critic
    films would then be measuring a moving target.

    The initial state ``mean = 0, var = 1, count = 0`` makes the untrained
    module the exact identity, and the Welford update is already correct at
    ``count = 0`` (it returns the batch's own mean and variance), so no
    special-casing is needed anywhere.

    Do **not** substitute ``nn.BatchNorm1d`` or an environment-wrapper
    normalizer here. SimBa §7.1 tests exactly those two and both underperform:
    BatchNorm because it normalizes by the batch rather than the history, and
    the env wrapper because an off-policy buffer then holds the same
    observation under whatever statistics happened to hold when it was
    collected, so identical states are stored with different values.
    """

    def __init__(self, dim: int, float_type=th.float32, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        # Parameters, not buffers -- see _stats_to_params for why.
        self.mean = nn.Parameter(th.zeros(dim, dtype=float_type), requires_grad=False)
        self.var = nn.Parameter(th.ones(dim, dtype=float_type), requires_grad=False)
        self.count = nn.Parameter(th.zeros(1, dtype=float_type), requires_grad=False)

    @th.no_grad()
    def update(self, x: th.Tensor) -> None:
        """Fold one batch into the running statistics (parallel Welford)."""
        n = x.shape[0]
        if n == 0:
            return
        batch_mean = x.mean(dim=0)
        # biased batch variance, matching the running estimate's convention
        batch_var = x.var(dim=0, unbiased=False)

        total = self.count + n
        delta = batch_mean - self.mean
        new_var = (
            self.var * self.count
            + batch_var * n
            + delta.pow(2) * self.count * n / total
        ) / total

        self.mean.copy_(self.mean + delta * (n / total))
        self.var.copy_(new_var)
        self.count.copy_(total)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return (x - self.mean) / th.sqrt(self.var + self.eps)


class SimBaBlock(nn.Module):
    """Pre-LayerNorm residual feedforward block with an inverted bottleneck.

    SimBa Eq. 6, ``x <- x + MLP(LayerNorm(x))``, with the MLP expanding to
    ``4 * d_h`` and a ReLU between its two linear layers (their §4, following
    Vaswani 2017). The residual is what gives a **direct linear pathway from
    input to output**: the network can pass its input through unchanged and
    only applies non-linearity where it needs to, which is the mechanism the
    paper credits for the simplicity bias.
    """

    def __init__(self, d_h: int, float_type=th.float32, expansion: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(d_h, dtype=float_type)
        self.fc1 = nn.Linear(d_h, expansion * d_h, dtype=float_type)
        self.fc2 = nn.Linear(expansion * d_h, d_h, dtype=float_type)

    def forward(self, x: th.Tensor) -> th.Tensor:
        h = self.norm(x)
        h = self.fc2(F.relu(self.fc1(h)))
        return x + h


class _TwinCritic(nn.Module):
    """Shared plumbing: the input widths, and the twin-Q / ``q1_forward`` API.

    Subclasses build ``self.q1`` and ``self.q2`` as callables taking the
    (already assembled) observation and action tensors. Keeping the two heads
    as separate modules rather than a shared trunk matches ``CriticTD3``, which
    builds two independent ``ModuleList``s -- a shared trunk would be a second
    change riding along with the architecture.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        float_type,
        unique_obs_dim: int,
    ):
        super().__init__()
        self.obs_dim = obs_dim + unique_obs_dim * (n_agents - 1)
        self.act_dim = act_dim * n_agents
        self.n_agents = n_agents
        self.float_type = float_type

    def forward(self, obs, actions):
        return self.q1(obs, actions), self.q2(obs, actions)

    def q1_forward(self, obs, actions):
        return self.q1(obs, actions)


# ------------------------------------------------------------ late injection


class _LateQ(nn.Module):
    """One Q head with the action injected at the second hidden layer.

    Lillicrap et al. (2016), supplementary §7. Layer 1 sees the observation
    only; the action enters concatenated with layer 1's activations, so it owns
    ``act_dim`` rows of layer 2's weight matrix rather than ``act_dim`` of
    ``obs_dim + act_dim`` columns of layer 1's.

    ``normalize_obs`` inserts the same paper's batch normalization *on the
    state input and every layer prior to the action input*, and nowhere after
    -- the action path is deliberately left un-normalized.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: list[int],
        float_type,
        normalize_obs: bool = False,
    ):
        super().__init__()
        if len(hidden) < 2:
            raise ValueError("late injection needs at least two hidden layers")
        self.obs_in = nn.BatchNorm1d(obs_dim, dtype=float_type) if normalize_obs else None
        self.fc_obs = nn.Linear(obs_dim, hidden[0], dtype=float_type)
        self.bn_obs = nn.BatchNorm1d(hidden[0], dtype=float_type) if normalize_obs else None

        layers = []
        in_dim = hidden[0] + act_dim  # <- the action joins here
        for h in hidden[1:]:
            layers.append(nn.Linear(in_dim, h, dtype=float_type))
            in_dim = h
        self.rest = nn.ModuleList(layers)
        self.head = nn.Linear(in_dim, 1, dtype=float_type)

    def forward(self, obs, actions):
        x = obs
        if self.obs_in is not None:
            x = self.obs_in(x)
        x = self.fc_obs(x)
        if self.bn_obs is not None:
            x = self.bn_obs(x)
        x = F.relu(x)

        x = th.cat([x, actions], dim=1)
        for layer in self.rest:
            x = F.relu(layer(x))
        return self.head(x)


class _SplitQ(nn.Module):
    """Two encoders of equal width, concatenated at hidden layer 2.

    Late injection (``_LateQ``) gives the action its own weight matrix but the
    action still arrives *raw* -- ``act_dim`` columns against ``hidden[0]``
    columns of observation activations, so it is outnumbered at layer 2 just as
    it was at layer 1, only less so. This head gives the action **its own
    hidden layer** and makes the two branches the same width:

        h_obs = relu(W_o . obs)     dim = hidden[0] // 2
        h_act = relu(W_a . act)     dim = hidden[0] // 2
        x     = cat(h_obs, h_act)   dim = hidden[0]

    From layer 2 on, the observation and the action occupy the same number of
    rows, so neither can dominate the layer's input by sheer count. That is the
    whole claim -- it fixes the *count*, not the *scale*, and the scale is what
    ``act_share`` moved in run 12. The two are separable exactly here: if
    equal-count is enough, this variant works where ``late`` does not.

    Splitting ``hidden[0]`` in half rather than giving each branch the full
    width keeps the concatenation exactly ``hidden[0]`` wide, so the rest of the
    network is untouched and the variant stays parameter-comparable to
    ``baseline`` -- one architecture change at a time.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: list[int],
        float_type,
        normalize_obs: bool = False,
    ):
        super().__init__()
        if len(hidden) < 2:
            raise ValueError("the split trunk needs at least two hidden layers")
        if normalize_obs:
            raise ValueError("BatchNorm is not defined for the split trunk")
        branch = max(1, hidden[0] // 2)
        self.fc_obs = nn.Linear(obs_dim, branch, dtype=float_type)
        self.fc_act = nn.Linear(act_dim, branch, dtype=float_type)

        layers = []
        in_dim = 2 * branch
        for h in hidden[1:]:
            layers.append(nn.Linear(in_dim, h, dtype=float_type))
            in_dim = h
        self.rest = nn.ModuleList(layers)
        self.head = nn.Linear(in_dim, 1, dtype=float_type)

    def forward(self, obs, actions):
        x = th.cat([F.relu(self.fc_obs(obs)), F.relu(self.fc_act(actions))], dim=1)
        for layer in self.rest:
            x = F.relu(layer(x))
        return self.head(x)


class _PlainQ(nn.Module):
    """``CriticTD3``'s own head, extracted so a normalizer can go in front."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int], float_type,
                 normalize_obs: bool = False):
        super().__init__()
        if normalize_obs:
            raise ValueError("BatchNorm on the state path requires late injection")
        layers = []
        in_dim = obs_dim + act_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h, dtype=float_type))
            in_dim = h
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(in_dim, 1, dtype=float_type)

    def forward(self, obs, actions):
        x = th.cat([obs, actions], dim=1)
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.head(x)


# ------------------------------------------------------------- the MLP family


class MLPCritic(_TwinCritic):
    """The plain-MLP family: ``CriticTD3`` plus two independent switches.

    ``late``        the action enters at hidden layer 2 instead of the input
    ``use_rsnorm``  RSNorm standardizes the observation first
    ``batchnorm``   BatchNorm on the state path (implies ``late``)
    ``hidden``      width override; ``None`` uses ``CriticTD3``'s own schedule

    At every switch off and ``hidden = None`` this **is** ``CriticTD3``, layer
    for layer and parameter for parameter -- ``_smoke`` asserts the count
    matches. It exists separately because the registry's ``baseline`` entry is
    ASSUME's real class (so a live sweep's control row is literally the
    shipped code), while the probes need a width-parameterizable stand-in.

    The action is **not** normalized in any configuration. It already lives in
    ``[-1, 1]`` by construction (the actor's ``softsign``), so standardizing it
    would only rescale it by its own replay-buffer spread -- which is
    ``act_share``'s lever wearing a normalizer's clothes, and the point of
    workstream B is to stop pulling that one.
    """

    late = False
    split = False
    use_rsnorm = False
    batchnorm = False
    hidden: list[int] | None = None

    def __init__(self, n_agents, obs_dim, act_dim, float_type, unique_obs_dim):
        super().__init__(n_agents, obs_dim, act_dim, float_type, unique_obs_dim)
        hidden = self.hidden or _hidden_sizes(n_agents)
        # one RSNorm shared by both Q heads: it is a property of the *input
        # distribution*, not of either head, and a copy each would make the
        # two heads see different inputs for no reason
        self.rsnorm = (
            RSNorm(self.obs_dim, float_type=float_type) if self.use_rsnorm else None
        )
        if self.split:
            head = _SplitQ
        elif self.late or self.batchnorm:
            head = _LateQ
        else:
            head = _PlainQ
        kw = dict(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden=list(hidden),
            float_type=float_type,
            normalize_obs=self.batchnorm,
        )
        self._q1 = head(**kw)
        self._q2 = head(**kw)
        _xavier(self)
        if self.batchnorm:
            # BatchNorm keeps running_mean/running_var as buffers, which Polyak
            # would skip; promote them so the target critic tracks them too.
            _stats_to_params(self)

    def _obs(self, obs):
        return self.rsnorm(obs) if self.rsnorm is not None else obs

    def forward(self, obs, actions):
        # the one place the running statistics advance: once per replay batch,
        # on the critic-loss path. See RSNorm's docstring for why not inside
        # RSNorm.forward, and why q1_forward must not.
        if self.training and self.rsnorm is not None:
            self.rsnorm.update(obs)
        return self.q1(obs, actions), self.q2(obs, actions)

    def q1(self, obs, actions):
        return self._q1(self._obs(obs), actions)

    def q2(self, obs, actions):
        return self._q2(self._obs(obs), actions)


class LateInjectionCritic(MLPCritic):
    """``CriticTD3`` with the action entering at the second hidden layer."""

    late = True


class SplitTrunkCritic(MLPCritic):
    """Observation and action get their own encoder, equal width, then merge.

    The natural next cell after ``late``: late injection gives the action its
    own weight matrix, this gives it its own *layer* and equal representation
    in the merged activation. See ``_SplitQ`` for the geometry and for why it
    isolates equal-count from equal-scale.
    """

    split = True


class BatchNormLateCritic(MLPCritic):
    """Lillicrap et al.'s full prescription: BN on the state path + late action.

    ``nn.BatchNorm1d`` normalizes by the *batch* in training mode and by its
    running statistics in eval mode, so unlike RSNorm this critic is not a
    pure function of one input row: a probe grid is normalized by the grid's
    own statistics. The critic films therefore read differently for this
    variant than for every other one, and it is included as a faithful
    reproduction of the 2016 paper rather than as a recommended design --
    SimBa §7.1 measures BatchNorm as one of the *weaker* normalizers.
    """

    late = True
    batchnorm = True


class RSNormCritic(MLPCritic):
    """RSNorm on the observation, then ``CriticTD3`` unchanged."""

    use_rsnorm = True


class RSNormLateCritic(MLPCritic):
    """RSNorm on the observation *and* the action injected at layer 2."""

    use_rsnorm = True
    late = True


# --------------------------------------------------------------------- SimBa


class _SimBaQ(nn.Module):
    """One SimBa Q head: embed, ``N`` residual blocks, post-LayerNorm, head.

    With ``late=True`` the action is concatenated onto the *embedding* instead
    of onto the input, so the residual trunk carries the observation and the
    action gets its own projection.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        d_h: int,
        n_blocks: int,
        float_type,
        late: bool = False,
    ):
        super().__init__()
        self.late = late
        in_dim = obs_dim if late else obs_dim + act_dim
        self.embed = nn.Linear(in_dim, d_h, dtype=float_type)
        self.inject = (
            nn.Linear(d_h + act_dim, d_h, dtype=float_type) if late else None
        )
        self.blocks = nn.ModuleList(
            SimBaBlock(d_h, float_type=float_type) for _ in range(n_blocks)
        )
        self.post = nn.LayerNorm(d_h, dtype=float_type)
        self.head = nn.Linear(d_h, 1, dtype=float_type)

    def forward(self, obs, actions):
        if self.late:
            x = self.embed(obs)
            x = self.inject(th.cat([x, actions], dim=1))
        else:
            x = self.embed(th.cat([obs, actions], dim=1))
        for block in self.blocks:
            x = block(x)
        return self.head(self.post(x))


class SimBaCritic(_TwinCritic):
    """SimBa (Lee et al., ICLR 2025) as a ``CriticTD3`` replacement.

    Defaults are the paper's critic settings, Table 7: ``n_blocks = 2`` at
    ``d_h = 512``. Note this is ~40x the baseline's parameter count, which is
    the paper's whole point (their Fig. 2b: an MLP *degrades* with width while
    SimBa improves) -- but it means ``simba`` against ``baseline`` moves two
    things at once. ``simba-small`` is the params-comparable row.
    """

    d_h = 512
    n_blocks = 2
    use_rsnorm = True
    late = False

    def __init__(self, n_agents, obs_dim, act_dim, float_type, unique_obs_dim):
        super().__init__(n_agents, obs_dim, act_dim, float_type, unique_obs_dim)
        self.rsnorm = (
            RSNorm(self.obs_dim, float_type=float_type) if self.use_rsnorm else None
        )
        kw = dict(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            d_h=self.d_h,
            n_blocks=self.n_blocks,
            float_type=float_type,
            late=self.late,
        )
        self._q1 = _SimBaQ(**kw)
        self._q2 = _SimBaQ(**kw)
        _xavier(self)

    def forward(self, obs, actions):
        if self.training and self.rsnorm is not None:
            self.rsnorm.update(obs)
        return self.q1(obs, actions), self.q2(obs, actions)

    def _obs(self, obs):
        return self.rsnorm(obs) if self.rsnorm is not None else obs

    def q1(self, obs, actions):
        return self._q1(self._obs(obs), actions)

    def q2(self, obs, actions):
        return self._q2(self._obs(obs), actions)


class SimBaSmallCritic(SimBaCritic):
    d_h = 128


class SimBaTinyCritic(SimBaCritic):
    """The width that lands nearest ``baseline``'s parameter count.

    Together with ``simba-small`` and ``simba`` this is a width ladder at fixed
    topology -- 0.15M / 0.54M / 8.5M against the baseline's 0.09M -- which is
    the comparison SimBa's Fig. 2b is about: an MLP degrades as it widens and
    SimBa does not. Without it, ``simba`` vs ``baseline`` moves architecture
    and parameter count at the same time and neither can be blamed.
    """

    d_h = 64


class SimBaNoNormCritic(SimBaCritic):
    """SimBa's residual trunk with RSNorm removed.

    The offline screen's headline row: at ``d_h = 512`` this is the only
    variant that recovers the reward band, while ``simba`` -- the identical
    network with RSNorm in front -- stays pinned at 100.0. That pair is a
    single-variable contrast (same width, same seeds, same budget), so it
    isolates **RSNorm as the component that breaks this landscape**.

    It is *not* a clean contrast against ``baseline``, which differs in both
    architecture and capacity (8.5 M against 105 k). ``-tiny`` and ``-small``
    below, plus ``mlp-wide``, are the cells that separate those two.
    """

    use_rsnorm = False


class SimBaNoNormSmallCritic(SimBaNoNormCritic):
    d_h = 128


class SimBaNoNormTinyCritic(SimBaNoNormCritic):
    d_h = 64


class MLPWideCritic(MLPCritic):
    """``CriticTD3``'s own topology widened to ``simba``'s parameter count.

    Part of the **matched width ladder**. Comparing two architectures at one
    parameter count answers "which is better at this size"; comparing their
    whole curves answers "how much capacity does each need for a given
    result", which is the question a scaling claim is actually about and the
    one SimBa's own Fig. 2b poses. Reading only the matched-parameter cells
    understates an architecture that gets the same result far cheaper.

    So the offline screen runs both families at the same three widths:

    ===============  ==========  ===================
    params           MLP         SimBa trunk
    ===============  ==========  ===================
    ~105 k           ``baseline``  --
    ~143 k           ``mlp-tiny``  ``simba-nornorm-tiny``
    ~548 k           ``mlp-small`` ``simba-nornorm-small``
    ~8.5 M           ``mlp-wide``  ``simba-nornorm``
    ===============  ==========  ===================

    Widths are bisected at import time against the SimBa variant they pair
    with, so the two families stay matched if either changes.
    """

    # 74 obs + 2 unique at n_agents = 1 is the shape of the frozen buffer the
    # offline harness fits on; the width is only a default and any caller
    # working at another shape should size it themselves via match_width
    _pair = "simba"
    hidden = None  # filled in below, once match_width exists


class MLPSmallCritic(MLPWideCritic):
    """MLP at ``simba-nornorm-small``'s parameter count (~548 k)."""

    _pair = "simba-small"
    hidden = None


class MLPTinyCritic(MLPWideCritic):
    """MLP at ``simba-nornorm-tiny``'s parameter count (~143 k)."""

    _pair = "simba-tiny"
    hidden = None


class SimBaLateCritic(SimBaCritic):
    late = True


# ------------------------------------------------------------------ registry


REGISTRY: dict[str, type] = {
    "baseline": CriticTD3,
    "late": LateInjectionCritic,
    "split": SplitTrunkCritic,
    "bn+late": BatchNormLateCritic,
    "rsnorm": RSNormCritic,
    "rsnorm+late": RSNormLateCritic,
    "simba": SimBaCritic,
    "simba-small": SimBaSmallCritic,
    "simba-tiny": SimBaTinyCritic,
    "simba-nornorm": SimBaNoNormCritic,
    "simba-nornorm-small": SimBaNoNormSmallCritic,
    "simba-nornorm-tiny": SimBaNoNormTinyCritic,
    "simba+late": SimBaLateCritic,
    "mlp-wide": MLPWideCritic,
    "mlp-small": MLPSmallCritic,
    "mlp-tiny": MLPTinyCritic,
}

#: one line per variant, for table headers and ``--help`` text
DESCRIPTIONS: dict[str, str] = {
    "baseline": "ASSUME's CriticTD3: concat at input, [256,128], ReLU, no norm",
    "late": "action injected at hidden layer 2 (Lillicrap 2016 §7)",
    "split": "separate obs and action encoders of equal width, merged at layer 2",
    "bn+late": "late injection + BatchNorm on the state path only (Lillicrap 2016)",
    "rsnorm": "running per-dim standardization of the observation (SimBa §4)",
    "rsnorm+late": "RSNorm + late action injection",
    "simba": "full SimBa, 2 blocks at d_h 512 (their critic default; ~90x baseline)",
    "simba-small": "full SimBa, 2 blocks at d_h 128 (~6x baseline params)",
    "simba-tiny": "full SimBa, 2 blocks at d_h 64 (nearest to baseline params)",
    "simba-nornorm": "SimBa residual trunk without RSNorm (isolates the normalizer)",
    "simba-nornorm-small": "SimBa trunk, no RSNorm, d_h 128 (is it the trunk or the width?)",
    "simba-nornorm-tiny": "SimBa trunk, no RSNorm, d_h 64 (is it the trunk or the width?)",
    "mlp-wide": "baseline topology at simba's ~8.5M params (width ladder)",
    "mlp-small": "baseline topology at ~548k params (width ladder)",
    "mlp-tiny": "baseline topology at ~143k params (width ladder)",
    "simba+late": "SimBa with the action injected after the embedding",
}


def build(arch: str) -> type:
    """The critic class registered under ``arch``.

    Returns the *class*, not an instance, so it can be assigned straight onto
    ``matd3.CriticTD3`` for a live sweep as well as constructed offline.
    """
    if arch not in REGISTRY:
        raise ValueError(
            f"unknown critic architecture {arch!r}; have: {', '.join(REGISTRY)}"
        )
    return REGISTRY[arch]


def describe(arch: str) -> str:
    return DESCRIPTIONS.get(arch, "")


def param_count(arch_or_cls, obs_dim: int, act_dim: int = 1, n_agents: int = 1,
                unique_obs_dim: int = 2) -> int:
    """Trainable parameters at this problem size, both Q heads."""
    cls = build(arch_or_cls) if isinstance(arch_or_cls, str) else arch_or_cls
    critic = cls(
        n_agents=n_agents,
        obs_dim=obs_dim,
        act_dim=act_dim,
        float_type=th.float32,
        unique_obs_dim=unique_obs_dim,
    )
    return sum(p.numel() for p in critic.parameters() if p.requires_grad)


# ----------------------------------------------------- width, as a free knob


#: ``baseline`` is ASSUME's own class and has no width knob, so sizing it uses
#: the family stand-in, which is parameter-for-parameter the same network.
_SIZEABLE: dict[str, type] = {"baseline": MLPCritic}


def hidden_at(width: int, depth: int) -> list[int]:
    """``depth`` hidden layers at ``width``, keeping ``CriticTD3``'s final taper.

    ``[width] * (depth - 1) + [width // 2]``. At ``depth = 2`` that is exactly
    ``[width, width // 2]``, i.e. ``CriticTD3``'s own shape, so the depth knob
    is backwards compatible with every width already recorded; deeper networks
    grow a constant-width body and keep the 2:1 step into the head.

    A geometric taper (halving every layer) was the alternative and is worse
    here: at depth 6 it would put the last hidden layer at ``width / 32``, so
    "depth" and "how narrow the network ends" would move together and neither
    could be blamed for the result.
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    if depth == 1:
        return [int(width)]
    return [int(width)] * (depth - 1) + [max(1, int(width) // 2)]


def sized(arch: str, width: int, depth: int | None = None) -> type:
    """``arch`` rebuilt at hidden width ``width`` and, optionally, ``depth``.

    Two knobs, because "more capacity" is two different experiments:

    * **width** -- MLP: ``hidden = [width, width // 2]``, keeping
      ``CriticTD3``'s 2:1 taper; SimBa: ``d_h``.
    * **depth** -- MLP: the number of hidden layers (see ``hidden_at``);
      SimBa: ``n_blocks``, the number of residual blocks. One SimBa block is
      two ``Linear`` layers plus a ``LayerNorm``, so the two families' depth
      units are **not** the same count of matrix multiplies. Compare each
      family's curve against itself, not depth 4 against depth 4.

    ``depth = None`` leaves the class's own depth alone, which is what every
    caller predating the depth knob wants.

    Needed because SimBa's own architecture comparison (their Fig. 4a, and
    Appendix D explicitly) is run at **matched parameter count** -- all twelve
    of their architectures sit within 1 % of 4.5 M. Comparing an 8.5 M SimBa
    against a 92 k MLP is a different experiment from theirs and answers a
    different question.
    """
    base = _SIZEABLE.get(arch) or build(arch)
    if issubclass(base, SimBaCritic):
        attrs = {"d_h": int(width)}
        if depth is not None:
            attrs["n_blocks"] = int(depth)
    elif issubclass(base, MLPCritic):
        attrs = {"hidden": hidden_at(width, 2 if depth is None else depth)}
    else:
        raise ValueError(f"{arch!r} has no width knob")
    suffix = f"_w{width}" + ("" if depth is None else f"_d{depth}")
    return type(f"{base.__name__}{suffix}", (base,), attrs)


def match_width(
    arch: str,
    target: int,
    obs_dim: int,
    act_dim: int = 1,
    n_agents: int = 1,
    unique_obs_dim: int = 2,
    bounds: tuple[int, int] = (2, 4096),
    depth: int | None = None,
) -> tuple[type, int, int]:
    """The width whose parameter count lands nearest ``target``, at fixed depth.

    Parameter count is monotone in width, so a bisection is exact up to the
    integer grid. Returns ``(class, width, params)`` -- the achieved count is
    returned rather than assumed, because widths are integers and an exact
    match usually does not exist.

    Width is bisected and depth is *given* rather than the other way round
    because depth is a small integer: at a fixed sensible width there is no
    depth that lands anywhere near 8 M without stacking a hundred layers, while
    at a fixed depth there is always a width. So a "is it depth or width"
    question is asked as a grid -- each depth run up the same parameter ladder
    -- not as two one-dimensional sweeps.
    """
    lo, hi = bounds
    kw = dict(obs_dim=obs_dim, act_dim=act_dim, n_agents=n_agents,
              unique_obs_dim=unique_obs_dim)
    while lo < hi:
        mid = (lo + hi) // 2
        if param_count(sized(arch, mid, depth), **kw) < target:
            lo = mid + 1
        else:
            hi = mid
    # bisection lands on the first width at or above target; the one below it
    # may be closer, so compare both
    best = min(
        (w for w in (lo - 1, lo) if w >= bounds[0]),
        key=lambda w: abs(param_count(sized(arch, w, depth), **kw) - target),
    )
    return sized(arch, best, depth), best, param_count(sized(arch, best, depth), **kw)


#: ``mlp-wide``'s width, resolved once now that ``match_width`` exists. The
#: shape is the frozen buffer's -- 74 observation dimensions, one action, one
#: agent -- which is what the offline screen fits on. A caller working at a
#: different shape should size it themselves rather than trust this default.
_WIDE_SHAPE = dict(obs_dim=74, act_dim=1, n_agents=1, unique_obs_dim=2)
for _cls in (MLPWideCritic, MLPSmallCritic, MLPTinyCritic):
    _cls.hidden = match_width(
        "baseline", param_count(_cls._pair, **_WIDE_SHAPE), **_WIDE_SHAPE
    )[0].hidden


# --------------------------------------------------- the two-family scaling grid


#: parameter counts the live sweep walks, and the label each gets in a name
LADDER_TARGETS: list[tuple[str, int]] = [
    ("100k", 100_000), ("500k", 500_000), ("2M", 2_000_000), ("8M", 8_000_000),
]

#: depths each rung is built at. MLP: hidden layers. SimBa: residual blocks --
#: 2 is the paper's critic default, so ``d2`` is their setting and ``d4`` the
#: deeper one. The two families' depth units differ (see ``sized``).
LADDER_DEPTHS: list[int] = [2, 4]

#: the two families the grid contrasts, as (name prefix, base architecture).
#: RSNorm is absent on purpose: run 17 measured six variants carrying it at a
#: mean in_band of 0.005 with the argmax pinned at exactly 100.0 at every width
#: from 143 k to 8.5 M, so putting it on the cluster would buy 24 more tasks of
#: the same answer.
LADDER_FAMILIES: list[tuple[str, str]] = [
    ("mlp", "baseline"), ("sbn", "simba-nornorm"),
]


def ladder_names() -> list[str]:
    """Every ``<family>-d<depth>-<size>`` name, in ladder order."""
    return [
        f"{fam}-d{d}-{label}"
        for fam, _ in LADDER_FAMILIES
        for d in LADDER_DEPTHS
        for label, _ in LADDER_TARGETS
    ]


def _build_ladder() -> None:
    """Resolve the grid's widths once and register the resulting classes.

    Widths are bisected at ``_WIDE_SHAPE`` -- 74 observation dimensions, one
    agent -- which is the offline harness's buffer and the single-agent inc-dec
    case. At a different observation width the achieved count drifts by the
    input layer only (``obs_dim * width`` against a body of ``width^2``), so a
    50-dim scenario lands within a couple of percent; the runner prints the
    real count for the shape it actually built, and that printed number is the
    one to quote.
    """
    for fam, base in LADDER_FAMILIES:
        for depth in LADDER_DEPTHS:
            for label, target in LADDER_TARGETS:
                cls, width, got = match_width(
                    base, target, depth=depth, **_WIDE_SHAPE
                )
                name = f"{fam}-d{depth}-{label}"
                cls.__name__ = f"{base}_d{depth}_{label}"
                REGISTRY[name] = cls
                DESCRIPTIONS[name] = (
                    f"{'plain MLP' if fam == 'mlp' else 'SimBa trunk, no RSNorm'}"
                    f", depth {depth}, width {width} -> {got:,} params"
                )


_build_ladder()


def _smoke(obs_dim: int = 50, act_dim: int = 1, n_agents: int = 1,
           unique: int = 2, batch: int = 8) -> None:
    """Every variant runs, is a function of its inputs, and Polyak-syncs.

    The third check is the one worth having: it constructs the variant twice,
    the way ``matd3.create_critics`` builds ``critics`` and ``target_critics``,
    and asserts ``polyak_update``'s ``zip`` is valid -- same parameter count,
    same shapes, in the same order. A variant whose normalizer statistics sit
    in buffers passes every other test here and silently desynchronizes the
    target critic at run time.
    """
    obs = th.randn(batch, obs_dim)
    act = th.rand(batch, act_dim * n_agents) * 2 - 1

    print(f"{'variant':<16}{'params':>10}{'stats':>7}  description")
    print("-" * 96)
    for name in REGISTRY:
        kw = dict(n_agents=n_agents, obs_dim=obs_dim, act_dim=act_dim,
                  float_type=th.float32, unique_obs_dim=unique)
        critic, target = build(name)(**kw), build(name)(**kw)
        target.load_state_dict(critic.state_dict())

        q1, q2 = critic(obs, act)
        assert q1.shape == (batch, 1) and q2.shape == (batch, 1), (name, q1.shape)

        # q1_forward is the actor's objective and must return exactly the q1
        # that forward just produced -- no statistic may advance between them
        assert th.allclose(critic.q1_forward(obs, act), q1), (
            f"{name}: q1_forward disagrees with forward on the same input"
        )

        # polyak_update zips these two lists; they must line up element-wise
        mine = list(critic.parameters())
        theirs = list(target.parameters())
        assert len(mine) == len(theirs), name
        assert all(a.shape == b.shape for a, b in zip(mine, theirs)), name

        # and any running statistic must be inside that list, not a buffer
        stats = sum(1 for p in critic.parameters() if not p.requires_grad)
        leftover = [
            n for n, b in critic.named_buffers() if th.is_floating_point(b)
        ]
        assert not leftover, f"{name}: float buffers skipped by Polyak: {leftover}"

        trainable = sum(p.numel() for p in critic.parameters() if p.requires_grad)
        print(f"{name:<16}{trainable:>10,}{stats:>7}  {describe(name)}")

    # MLPCritic with every switch off must BE CriticTD3, or the family stand-in
    # used for width matching is quietly a different network from the control
    kw = dict(obs_dim=obs_dim, act_dim=act_dim, n_agents=n_agents,
              unique_obs_dim=unique)
    assert param_count(MLPCritic, **kw) == param_count(CriticTD3, **kw), (
        "MLPCritic has drifted from CriticTD3"
    )

    # and the width knob must be able to hit a target
    target = param_count(CriticTD3, **kw)
    _, w, got = match_width("simba", target, obs_dim=obs_dim, act_dim=act_dim,
                            n_agents=n_agents, unique_obs_dim=unique)
    print(f"\nall variants: shapes ok, q1_forward consistent, Polyak-compatible")
    print(f"MLPCritic == CriticTD3 at {target:,} params; "
          f"simba matches it at d_h {w} ({got:,}, {got / target - 1:+.1%})")


if __name__ == "__main__":
    _smoke()
