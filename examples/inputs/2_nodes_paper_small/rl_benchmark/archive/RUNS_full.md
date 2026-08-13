# Inc-dec reward landscape: RL algorithm benchmark

> **This file is tracked; the data it describes is not.** The archive itself —
> every `.npz`, every figure — lives in `../../../outputs/2_nodes_paper_small/rl_benchmark/runs/`,
> which is gitignored (`.gitignore:143`), so the links below only resolve on a
> machine that has run the experiments. This document is kept here so the record
> of what was run, what it showed and what was refuted survives the archive.
> See §9 for what is and is not reproducible from git alone.

Archive of the runs behind the question *"can standard RL find the optimum of the
inc-dec reward landscape, and if not, why not?"* — the landscape measured in
`examples/outputs/2_nodes_paper_small/reward_landscape.png` for the learning
diesel unit `diesel_0`.

Run on **2026-08-06/07**. Every number in this document was recomputed from the
archived `.npz` files, not copied from a console log.

## Start here — the three headline figures

They sit at this folder's top level; everything under `data/` and `img/` is the
detail behind them.

| figure | what it answers |
|---|---|
| [`01-algorithms-best-known-settings.png`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/01-algorithms-best-known-settings.png) | **Which algorithms find the optimum**, each at the best settings this archive knows: TD3/DDPG with a softsign actor (run 05), SAC with `ent_coef 0.001` (run 03). SAC 31.35 ± 0.31 and DDPG 31.86 ± 0.51 both solve it 4/4; TD3 is unstable at 1.59 ± 52.26; PPO never leaves the ceiling. Its **bottom row** adds the critic gradient field each actor was climbing, on the same axes: all three algorithms get the same descent window at step 1400, and the ordering is decided by how fast each spends it. Run 01 is the same comparison at *defaults*, where every learner loses to random search. Run 01b. |
| [`02-td3-stability-sweep.png`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/02-td3-stability-sweep.png) | **How stable that is** — 17 configurations × 8 seeds. Answer: mostly not. The baseline solves 3/8; only `lr-1e-4` is tight (8/8 at 32.10 ± 0.37). Run 08. |
| [`03-assume-matd3-films.png`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/03-assume-matd3-films.png) | **What ASSUME's own MATD3 does**, filmed over a real 40-episode learning run, with the reward shaping on and off. Run 09, reproduced at 6 seeds per condition in run 10. |
| [`12-actshare-dose-response.png`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/12-actshare-dose-response.png) | **What finally fixes it**, and it is not a hyperparameter: the action carries 3 % of the variation across the critic's 75 input dimensions, and raising that share solves the true-reward task 3/3. Two mechanically unrelated levers land on one curve. Run 12. |

If you read one section, read **§6** (why the optimum is the wrong target for a
stochastic policy), **§7** (what any of this means for ASSUME) and **§3, run 12**
(the first configuration in this archive that solves the true reward).

### The short version

The landscape is 90 % flat with the optimum on a cliff edge. Three things go
wrong, in this order:

1. **The critic is right before it is complete.** Its first fit is "higher bids
   are better" — true of 89 % of the action space — so `argmax Q` is the +100
   ceiling and every actor runs there. This is not a bug and not exploration
   failure; the band is in the buffer the whole time.
2. **The way back is open only briefly.** Once the critic finds the band, the
   slope over the whole plateau flips and an unbroken descent path exists — for
   about 600 steps. Then the plateau is learned as genuinely flat and its
   gradient degenerates into noise. A *converged* critic offers no way back.
   (Run 06.)
3. **Crossing back is unstable.** Softsign lets the actor move at all — tanh
   saturates to a hard zero and cannot — but a full-sized Adam step is too big to
   stay inside a 19-EUR band bounded by a 0.36 cliff, so the actor overshoots to
   the far side. Only a smaller step (`lr-1e-4`) converges reliably. (Run 08.)

ASSUME's MATD3 fails at stage 1 on its default budget — 640 critic updates and
**80 actor updates**, against the 190–410 actor updates a crossing costs here.
Give it 4× the budget and, **on the true reward, it still never leaves stage 1**:
the critic's field stays incoherent for all 2560 updates — its preferred bid
disagrees by 24.5 EUR between an average pair of probed observations, 56 EUR end
to end — and 5 of 6 seeds never place a
bid in the band at all. With the reward shaping switched on, all 6 seeds leave the
ceiling within 96–192 updates and settle at the band's rim. So on the real
scenario the binding problem is not the crossing at all — it is that the critic
never learns a usable shape from the true reward. (Runs 07, 09, 10.)

**Run 12 says why, and it is none of the above.** The reward *is* almost a
function of the bid alone (R² = 0.95 on the buffer); the observation carries no
generalising information about it at all (leave-one-out 1-NN R² = **−1.07**). But
the action is **one of 75 critic inputs**, and the other 74 contribute 97 % of the
input variation. Raise the action's share — by cutting `foresight`, or by scaling
the critic's action input, two mechanically unrelated levers — and the same
learner, on the same true reward, with the same budget, **solves it 3/3 at
33.0 ± 0.2 for +0.160**. That is the first non-zero solve in this archive, against
0/90 in run 11.

---

## 1. The problem

`../incdec_reward.py` reproduces the measured landscape in closed form. Under
pay-as-clear EOM plus pay-as-bid redispatch, with the rest of the fleet fixed:

```
b > 49            not dispatched               reward  0
30 <= b <= 49     dispatched, then dec'd       reward (49 - b)/100   -> peak +0.190 at b=30
b < 30            dispatched, NOT dec'd        reward -0.170         (flat)
```

Parameters: `marginal_cost 66`, `eom_price 49`, `dec_threshold 30`,
`volume 1000 MW`, `max_bid_price 100`. The action is a single tanh output scaled
to a bid, so the agent can reach `[-100, +100]` EUR/MWh.

Why it is hard:

| property | value |
|---|---|
| optimum | bid **30.00** -> reward **+0.190** |
| cliff depth one tick below the optimum | **0.360** |
| informative (non-flat) share of the action space | **9.5 %** (19 of 200 EUR) |
| exactly-flat share of the action space | **~90 %** (62 % loss shelf, 27 % zero plateau) |

The optimum sits *on the cliff edge*, which turns out to matter enormously
(see §6).

> An independently hand-written version of this reward function (`rwd.py`, now in
> the sibling `rl_benchmark - Kopie/` folder) specifies the same three regimes and
> the same breakpoints. The closed form here agrees with it.

## 2. What the networks see

The observation is a **74-vector that never changes** within a run: `[0:24]`
residual-load forecast = 0.49, `[24:48]` price forecast = 0.49, `[48:72]` price
history = 0.49, `[72]` scaled marginal cost = 0.66, `[73]` availability = 1.0.

The layout mirrors `TorchLearningStrategy.create_observation`
(`3 * foresight + unique_obs_dim`), but in the single-context setup it carries
**zero information** — `Q(s,a)` is effectively `Q(a)`. This is the *charitable*
case; the real scenario adds 73 varying, mostly-irrelevant dimensions on top.
`IncDecEnv(params=[...])` takes a list of contexts for the harder version.
**Everything archived here is single-context.**

---

## 3. The runs

Common to all: `learning_rate 1e-3`, `batch_size 256`, `gamma 0.99`,
`warmup 1000`, `buffer_size 10000`, `sigma 0.1`, `noise_schedule linear`,
`device cpu`. Each run's full `RunConfig` is stored inside its own `.npz` under
`cfg/*` keys and is reloaded by `--replot`.

### 01 — baseline: which algorithms find it at all?

**Why:** the first question. Five contenders including a uniform random-search
reference, which is the bar a learner has to beat.

```bash
python run_benchmark.py --eval-every 200 --noise-schedule linear
```

- data: [`data/01-baseline/results.npz`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/data/01-baseline/results.npz)
- img: [`img/01-baseline.png`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/img/01-baseline.png)
- 1 seed, 10 000 steps

| algorithm | final bid | reward |
|---|---|---|
| TD3 | 100.00 | +0.000 |
| DDPG | 100.00 | +0.000 |
| SAC | 60.36 | +0.000 |
| PPO | 100.00 | +0.000 |
| **Random search** | **30.02** | **+0.190** |

**Uniform random search beats all four learners outright.** Every learner ends on
the zero plateau; three pin to the `+100` action ceiling.

Exploration is *not* the bottleneck: 1000 uniform warmup steps put ~10 % of
samples inside `[30, 49]`, the best within ~0.15 EUR of the cliff edge, and with
`buffer_size == timesteps` **nothing was ever evicted**. The critics had the
optimum in the buffer the whole time.

> This run predates config-persistence, so its `.npz` has no `cfg/*` keys.
> Re-plot it with the flags given above.

### 01b — the same comparison at best-known settings (**headline figure 1**)

**Why:** run 01 asks "can these algorithms find it out of the box", and the answer
is no. The more useful question, once runs 03 and 05 had found what each algorithm
needs, is "can they find it *at all*". This is run 01 with two changes, both
established elsewhere in this archive: TD3/DDPG get a softsign actor (run 05), SAC
gets `ent_coef 0.001` (run 03). PPO and random search are unchanged.

```bash
python sweeps/run_benchmark.py --algos TD3 DDPG SAC PPO "Random search" \
    --seeds 4 --timesteps 10000 --eval-every 200 --noise-schedule linear \
    --actor-activation softsign --ent-coef 0.001 --critic-grid 401

# the archived figure draws the critic row for seed 3 -- TD3's failing seed
python sweeps/run_benchmark.py --replot --critic-seed 3 \
    --results <runs>/data/01b-best-known/headline_comparison.npz \
    --out <runs>/01-algorithms-best-known-settings.png
```

- data: [`data/01b-best-known/headline_comparison.npz`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/data/01b-best-known/headline_comparison.npz)
- img: [`01-algorithms-best-known-settings.png`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/01-algorithms-best-known-settings.png)
- 4 seeds, 10 000 steps, probes every 200

| algorithm | final bid | reward |
|---|---|---|
| **SAC** (`ent_coef 0.001`) | **31.35 ± 0.31** | **+0.177 ± 0.003** |
| **DDPG** (softsign) | **31.86 ± 0.51** | **+0.171 ± 0.005** |
| TD3 (softsign) | 1.59 ± 52.26 | +0.087 ± 0.148 |
| PPO | 100.00 ± 0.00 | +0.000 |
| Random search | 30.02 ± 0.02 | +0.190 |

**Two of the four learners now solve it, and tightly.** SAC lands at 31.35 ± 0.31
and DDPG at 31.86 ± 0.51, both 4/4 seeds — against §6's constrained optimum of
31.30 for σ = 0.5. Compare run 01, where every learner ended on the zero plateau
and lost to random search.

**TD3 is the unstable one**, at 1.59 ± 52.26. Its four final bids are
`31.86, 31.60, 31.83, −88.93`: **three seeds converge into the band and one falls
through it** onto the −0.170 shelf, which is run 08's failure mode. (An earlier
version of this section said "two and two"; the archived `.npz` says 3/1.) On this
landscape DDPG — no twin critic, no delayed actor, no target smoothing — is
*more* reliable than TD3, and that is the opposite of the usual ordering.

#### The critic row: why the ordering comes out that way

The bottom row of the figure shows `dQ/d(bid)` of each actor's own objective —
autograd, on a 401-point action grid, at every one of the 50 probes — on the same
axes as the learning history above it. The actor never sees the reward; it ascends
this field, so this is the landscape that actually produced row 2. PPO and random
search have no panel: PPO's critic is a state-value `V(s)`, with no gradient with
respect to the action, and random search has no network.

Run 06 established the three phases on TD3/DDPG alone. Recomputed here across all
12 runs (3 algorithms × 4 seeds), they are **identical in timing for every
algorithm**:

- **Phase 1 — right but incomplete.** At the first probe with a trained critic
  (step 1200) `dQ/d(bid)` is positive across **100 % of the `[49, 100]` plateau in
  10 of the 12 runs**, and `argmax Q` = +100.0. Every actor runs to the ceiling
  following a correct gradient on a critic that has not yet learned the plateau is
  flat.
- **Phase 2 — the window.** The plateau slope goes near-unanimously negative at
  **step 1400 in 11 of 12 runs** (SAC seed 1 at 1600), and the sign at each
  actor's *own* action is negative at step 1400 in **12 of 12**. The window is a
  property of the landscape and the buffer, not of the algorithm: TD3, DDPG and
  SAC all get it, at the same step, on the same probe.
- **Phase 3 — fragmentation, but only where the reward is flat.** After step 5000
  the converged field is *correct where it matters*: inside the band `[30, 49)`
  **85–94 % of grid cells carry the negative slope the true reward has**, at a
  median `|dQ/d(bid)|` of **9.5 × 10⁻³**. On the two flat regions it is two orders
  of magnitude weaker (**~1.5 × 10⁻⁴**) and the sign is a coin flip — 47–50 % of
  cells point back down on `[70, 100]`, 51–54 % point back up on the `< 30` shelf.
  The mottling in the panels is a flat region fitted *correctly*, not a failure.

**What differs between the algorithms is only how fast the actor spends the
window.** First probe at or after the flip with a greedy bid back inside
`[30, 49]`:

| | seed 0 | seed 1 | seed 2 | seed 3 | steps after the flip |
|---|---|---|---|---|---|
| SAC | 1400 | 1400 | 1600 | 1800 | 0–400 |
| DDPG | 1600 | 1800 | 1800 | 1800 | 200–400 |
| TD3 | 2000 | 2000 | 2000 | **never** | 600 |

**This is `policy_delay`, and it explains the DDPG-beats-TD3 ordering.** TD3
updates its actor once per 2 critic updates; DDPG and SAC update every step. At
one gradient step per environment step, TD3's 600-step crossing is ≈ **300 actor
updates** and DDPG's 200–400-step crossing is 200–400 — both inside run 06's
measured 190–410. The two algorithms need the same number of *actor* updates and
TD3 needs twice as many *environment* steps to accumulate them, against a window
that is counted in environment steps. It is the only stabiliser TD3 adds here that
does anything, and what it does is make TD3 miss the window.

This also **explains run 04's negative result** rather than contradicting it.
Run 04 raised `policy_delay` to 8 and 64 looking for a fix and found none — 8 is
0/8 with softsign. It was searching in the wrong direction: on this landscape
every increment of `policy_delay` buys critic accuracy the critic does not need
(it is already right by step 1400) at the cost of actor updates it cannot spare.
**Untested prediction:** TD3 at `--policy-delay 1` should match DDPG. Nothing else
distinguishes them here — the twin critic and target smoothing neither help nor
hurt on a landscape whose critic is accurate long before the actor arrives.

**TD3 seed 3 is what missing it looks like**, and it is the seed the figure draws.
Its actor sits between +87 and +98 from step 1400 all the way to 5400, while
`dQ/d(bid)` at its own action decays to **−2.5 × 10⁻⁵** — an actor that had
reached the band would be feeling ~−6 × 10⁻³ — and then between probe 5400
(+91.97) and probe 5600 (**−93.65**) it crosses the entire action space in one
step and stays between −94.20 and −81.33 for the remaining 4400 steps. It never
crosses the band; it steps over it. And nothing brings it back, because the field on the `< 30` shelf has no
consistent direction: run 08's overshoot failure and this one are the same shelf,
reached in one step rather than by overshooting through the band.

> The archived `.npz` originally held no critic sweeps. It was re-recorded with
> `--critic-grid 401` (~55 min); the `greedy/*`, `placed/*` and `steps` arrays of
> the new file are **bit-identical** to the old one, so every number above and in
> the table is the same run. The critic sweep consumes no RNG, which is why adding
> it did not perturb training — a useful reproducibility check in its own right,
> given run 08's finding that this configuration is chaotic in the BLAS thread
> count.

**Random search still posts the best number, and it is still not a policy.** It is
an argmax over every action ever sampled, i.e. an oracle with perfect recall and no
generalisation; the learners are reporting a deployable policy. It belongs on the
figure as the bar to clear, not as a competitor.

### 02 — what did the critics actually learn?

**Why:** the actor update is gradient ascent on the critic
(`L_actor = -Q(s, pi(s))`), so the actor never sees the reward — only the
critic's approximation. This run saves the trained networks so the critic can be
swept directly.

```bash
python run_benchmark.py --algos TD3 DDPG SAC --seeds 1 --timesteps 10000 \
    --eval-every 200 --noise-schedule linear --save-models \
    --results critic_run.npz --out critic_run.png
python critic_landscape.py
```

- data: [`data/02-critic/critic_run.npz`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/data/02-critic/critic_run.npz),
  networks in [`data/02-critic/models/`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/data/02-critic/models/)
- img: [`img/02-critic-run.png`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/img/02-critic-run.png) (training),
  [`img/02-critic-landscape.png`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/img/02-critic-landscape.png) (**the key figure**)

Reproduces 01 exactly (same seed). The critic sweep:

| | critic argmax | actor sits at | `dQ/da` at the actor |
|---|---|---|---|
| TD3 | **34.2** | 100.0 | −4.4 × 10⁻⁵ |
| DDPG | **37.6** | 100.0 | +6.3 × 10⁻⁶ |
| SAC | **32.7** | 60.4 | −7.7 × 10⁻⁵ |

> These figures were **revised** after checking which Q each actor climbs. TD3's
> actor loss uses `critic.q1_forward` (`td3.py:199`) — **Q1 alone**, not
> `min(Q1, Q2)`; SAC uses the min (`sac.py:281`). The first version of this table
> swept the min for all three, which put TD3's argmax at 34.8 and its gradient at
> −1.5 × 10⁻⁵. Gradients are now `torch.autograd`, not `np.gradient`. The
> conclusion is unchanged — only the third digit moved.

**The critics are right.** All three place their maximum inside the profitable
band. `dQ/da` is ≈0 everywhere except a +0.05…+0.10 spike at 32 and a
−0.004…−0.011 leftward pull in `[40, 48]` — matching the true reward's shape.

**The actors are stuck at the ceiling**, where `dQ/da` is three to four orders of
magnitude below the signal at the cliff.

> The original reading of this table — "the gradient is too small, so the
> actor cannot move" — is **wrong**, see correction 4. Adam is scale-invariant
> for a consistently-signed gradient, so 10⁻⁵ would still give full-sized steps.
> What actually freezes these two actors is tanh saturation (§7, run 05); the
> small `dQ/da` here is a symptom of standing on the plateau, not the cause of
> being unable to leave it.

To re-inspect without retraining, just run `python critic_landscape.py` — it
searches `models/` then this archive, so the saved networks work in place.
Override with `--models <dir>`.

### 03 — SAC entropy: why does it keep leaving the band?

**Why:** in 01, SAC's greedy policy repeatedly entered the profitable band
(peaks +0.157) and fell back out, four times. SB3 defaults to `ent_coef="auto"`
with `target_entropy = -dim(A) = -1.0`, which is a **floor on policy entropy**
enforced by a feedback controller — roughly σ ≈ 0.089, i.e. ±9 EUR, about as wide
as the whole 19 EUR band. Hypothesis: SAC is structurally forbidden from
converging onto a peak narrower than its own mandated spread.

```bash
python run_benchmark.py --algos SAC --seeds 2 --timesteps 8000 --eval-every 200 \
    --ent-coef <EC> --target-entropy <TE> --results <...> --out <...>
```

- data: [`data/03-sac-entropy/`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/data/03-sac-entropy/)
- img: `img/03-sac-*.png`
- 2 seeds, 8000 steps each

| setting | file | final bid | reward |
|---|---|---|---|
| `auto` (target entropy −1) | `auto.npz` | 51.67 ± 14.55 | +0.059 |
| `--target-entropy -4` | `target-entropy-4.npz` | 33.50 ± 0.04 | +0.155 |
| `--target-entropy -8` | `target-entropy-8.npz` | 36.64 ± 1.69 | +0.124 |
| `--ent-coef 0.001` | `ent-coef-0.001.npz` | **32.34 ± 0.59** | **+0.167** |

**Confirmed, and the response is non-monotonic.** −4 converges tightly (±0.04
across seeds); −8 is *worse* than −4. Reading: too high a floor and it never
converges; too low and it collapses to near-deterministic early and inherits
TD3's failure mode. There is a window.

### 04 — TD3 `policy_delay`: a hypothesis that failed

**Why:** having seen that the actor commits to the ceiling the instant warmup
ends — while the critic has had only a handful of gradient steps — the obvious
guess was *early commitment*: slow the actor down and let the critic become
accurate first. `policy_delay` does exactly that, and ASSUME's own config already
uses 8.

```bash
python run_benchmark.py --algos TD3 --seeds 2 --timesteps 8000 --eval-every 200 \
    --policy-delay <2|8|64> --results <...> --out <...>
```

- data: [`data/04-td3-policy-delay/`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/data/04-td3-policy-delay/)
- img: `img/04-td3-pd-*.png`

| `policy_delay` | final bid | reward |
|---|---|---|
| 2 (SB3 default) | 100.00 ± 0.00 | +0.000 |
| 8 (ASSUME's setting) | 100.00 ± 0.00 | +0.000 |
| 64 | 68.46 ± 27.58 | +0.041 |

**The hypothesis is not supported.** Slowing the actor 32× moved it from 100 to
68 ± 28 — still on the plateau, now with large seed variance. Timing is not the
lever. Kept in the archive because a refuted hypothesis is worth not re-testing.

### 05 — softsign actor: the one-line fix

**Why:** runs 01–04 blamed TD3/DDPG. If the tanh saturation of §7 is the real
cause, swapping only the actor's output squashing — nothing else — should undo
the failure. `--actor-activation softsign` replaces `net.mu[-1]` on the actor and
its Polyak target; softsign shares tanh's `(-1, 1)` range, so bounds, optimizer
and architecture are untouched.

```bash
python run_benchmark.py --algos TD3 DDPG --seeds 2 --timesteps 10000 \
    --eval-every 200 --noise-schedule linear --critic-grid 201 \
    --actor-activation softsign --results <out>/softsign.npz
python activation_comparison.py
```

- data: [`data/05-softsign/softsign.npz`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/data/05-softsign/softsign.npz)
- img: [`img/05-activation-comparison.png`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/img/05-activation-comparison.png)
  (**the key figure**), `05-softsign-training.png`,
  `05-softsign-critic-evolution.png`
- 2 seeds, 10 000 steps

| actor output | TD3 | DDPG |
|---|---|---|
| `tanh` (SB3 default) | 100.00 ± 0.00 → **+0.000** | 100.00 ± 0.00 → **+0.000** |
| `softsign` (ASSUME's default) | **31.73 ± 0.13 → +0.173** | **31.51 ± 0.52 → +0.175** |

Complete reversal, both seeds — and better than the best SAC result in run 03
(+0.167). Against §6's table, a policy of width σ ≈ 0.5 has a constrained optimum
at 31.30 for +0.175; TD3 lands at 31.73. It is essentially solved.

The trajectories show the mechanism, and it is *not* that softsign avoids the
runaway:

| step | TD3 tanh | TD3 softsign |
|---|---|---|
| 1000 (warmup ends) | −4.32 | −4.14 |
| 1200 | **100.00** | **98.68** |
| 2200 | 100.00 | **34.69** |
| 10000 | 100.00 | 31.86 |

Both actors bolt for the ceiling the instant warmup ends. They differ only in
whether they can return: at bid 100.00 tanh gives `1 - a² = 0` **exactly** and the
actor is frozen for the remaining 8800 steps, while softsign at bid 98.68 still
holds `2.1e-2` of headroom and climbs back into the band within ~1000 steps.

### 06 — the escape window: *when* was the descent available?

**Why:** run 05 explains what changed, but the trajectory it produced is a single
burst between steps ~1200 and ~2200, not a steady descent. If crossing the plateau
is only possible while the critic is in a particular transient state, then §7's
path metrics — pooled over *every post-warmup probe of the whole run* — are
averaging over an interval in which the outcome had already been decided.

```bash
python run_benchmark.py --algos TD3 DDPG --seeds 3 --timesteps 4000 \
    --eval-every 25 --critic-grid 401 --actor-activation softsign \
    --results <o>/window_softsign.npz
python run_benchmark.py ... --actor-activation tanh --results <o>/window_tanh.npz
python descent_window.py               # TD3
python descent_window.py --algo DDPG   # same figure for DDPG
```

- data: [`data/06-window/`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/data/06-window/) — `window_softsign.npz`, `window_tanh.npz`
- img: [`img/06-descent-window.png`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/img/06-descent-window.png) (**the key figure**),
  `06-descent-window-ddpg.png`
- 3 seeds × 2 algorithms × 2 activations, 4000 steps, probed **every 25 steps** on a
  401-point action grid — 8× finer in time than run 05, which is what makes the
  phases visible at all

The run splits into three phases, in all twelve runs:

| phase | steps | mean `dQ/d(bid)` on [50, 100] | unbroken path to the band |
|---|---|---|---|
| warmup → flip | 1000 – 1250 | **+1.1e-3**, pulls *right* | 0 % |
| flip → crossed | 1250 – 1900 | −6.4e-4, pulls *left* | **91 – 100 %** |
| after crossing | 1900 – 4000 | −3.7e-4 | 27 – 72 %, 2 sign reversals |

**Phase 1 is why both actors run to the ceiling, and it is not a bug.** The first
thing the critic learns is the coarse shape — 62 % of the action space is the
−0.170 loss shelf, 27 % is the zero plateau — so `Q` rises monotonically with the
bid across the entire range and `argmax Q` is at +100. The actors are following a
correct gradient on an incomplete critic.

**Phase 2 is the window.** Once the critic finds the band's spike, the slope over
the *whole* plateau reverses into one coherent leftward pull. The flip happens at
step **1225 – 1325 in every one of the twelve runs** — 225 to 325 steps after
warmup ends — and for the next ~600 steps an unbroken descent path runs from the
ceiling into the band.

| | first clean path | pull flips | actor crosses | final bid |
|---|---|---|---|---|
| TD3 softsign, 3 seeds | 1275 – 1325 | 1250 – 1325 | 1575 – 2125 | 34.7 / 78.1 / 37.1 |
| TD3 tanh, 3 seeds | 1275 – 1300 | 1250 – 1300 | **never** | 100.00 ×3 |
| DDPG softsign, 3 seeds | 1225 – 1300 | 1225 – 1275 | 1450 – 1950 | 32.8 / 34.3 / 33.3 |
| DDPG tanh, 3 seeds | 1225 – 1275 | 1225 – 1250 | **never** | 100.00 ×3 |

**Both activations see the same clean field at the same time.** The tanh runs are
not deprived of a descent path — they are handed one, on schedule, and cannot take
a single step along it. That is run 05's conclusion confirmed on a much finer grid,
and it is also the cleanest available disproof of "the field was too fragmented".

**Phase 3 is not evidence about crossability, and it is not exogenous.** After the
crossing, the plateau's coherent pull decays and its sign fragments — because the
critic has by then learned that the plateau really *is* flat, and the gradient of a
correctly-learned flat region is numerical noise. Worse, which critic keeps
learning there depends on where its own actor went:

| after step 2000 | bids placed on [50, 100] | bids placed in the band |
|---|---|---|
| tanh (all 6 runs) | **100 %** | 0 % |
| softsign (6 runs) | 0 – 52 % | 39 – 88 % |

So the pooled path metric compares a plateau that one algorithm is still training on
against a plateau the other abandoned. It is a consequence of the outcome, not a
cause. See correction 7.

### 07 — ASSUME's own MATD3, at last

**Why:** everything above is an SB3 analogue, and §8 has listed "nothing has been
tested against the real MATD3" as the open item that matters. The scenario has
saved networks for three learning study cases, so the analogue can be checked
against the thing it stands in for.

```bash
python assume_critic_probe.py --n-obs 24
```

- networks: `examples/inputs/2_nodes_paper_small/learned_strategies/<case>/last_policies/`
- img: [`img/07-assume-critic-probe.png`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/img/07-assume-critic-probe.png)
- `Q1` swept over the bid axis at 24 **real** observations drawn from the run's own
  replay buffer — the 74-vector varies hour to hour here (548 distinct vectors in
  620 transitions), so a constant context would probe a state never trained on

**The budget is in the saved optimizer state, and it is exactly as predicted:**
every case has `critic_optimizer step = 640` and `actor_optimizer step = 80`.
Against run 06's 190–410 actor updates per crossing, 80 is not close.

| case | reward | critic argmax | actor bids | `dQ/da` at the actor |
|---|---|---|---|---|
| `..._inc_dec_learning_single` | true | **100.0** (4 % in band) | 95.2 | **+1.0e-3** (pulls right) |
| `..._inc_dec_learning_single_g0` | shaped | 62.8 (12 %) | 52.9 | +4.3e-3 |
| `inc_dec_learning_single_g0_2` | shaped | 51.0 (29 %) | 50.7 | +6.5e-4 |

**The unshaped run is frozen in phase 1.** Its `Q1` rises monotonically with the
bid over the entire range, `argmax Q` is the +100 ceiling, and `dQ/da` at the
actor is *positive* — it is still being pushed up. This is not a converged critic
that the actor failed to follow; the critic never reached the sign flip. With 640
updates it is where run 06's critics were at ~step 1200, and the run ends there.

Exploration is not the reason: `buffers/single_10ep_standard.npz` holds 620
transitions with **10 % inside the band** and a best in-band reward of +0.199 —
the same coverage the SB3 warmup gives. What differs from the benchmark is that
the real critic must fit `Q(s, a)` across 548 distinct observations rather than
`Q(a)` at one, on the same 640 updates.

**The shaped runs converged onto a decoy.** With the reward shaping at
`learning_strategies.py:1583-1589` active, `Q1` becomes a broad unimodal hill and
the actor sits essentially *on* its maximum (51.0 vs 50.7; 62.8 vs 52.9) — the
actor is tracking the critic accurately. The problem has moved into the reward:

| bid | true reward | shaped reward |
|---|---|---|
| 48.9 | +0.001 | +0.001 |
| **49.0** | **+0.000** | **+0.170** |
| 50.0 | +0.000 | +0.160 |
| 66.0 | +0.000 | +0.000 |
| 100.0 | +0.000 | −0.340 |

Because the shaping fires only when `reward <= 0`, it does not apply inside the
band, and it re-enters at bid 49 at full height. The shaped landscape therefore
has **two local maxima** — +0.190 at bid 30, and **+0.170 at bid 49, where the
real profit is zero** — separated by the same cliff as before. The ramp the
shaping adds above the band is a genuine, permanent replacement for run 06's
transient window, and it does exactly what it was meant to do: it walks the actor
down from +100. It just terminates one euro above the band instead of leading
through it, and both g0 runs stopped there.

### 08 — how stable is run 05? Not at all.

**Why:** run 05 changed the headline conclusion of this whole archive on the
strength of **two seeds of one configuration**. Run 06 then predicted the result
should be fragile in a specific direction: anything that slows the actor while the
critic keeps converging should break it, since the descent path is only open for a
few hundred updates.

```bash
python td3_stability.py --seeds 8 --workers 15          # 15 configs x 8 seeds
python td3_stability.py --configs run05-repro lr-1e-4-10k --seeds 8
```

- data: [`data/08-stability/`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/data/08-stability/) — `td3_stability.npz`,
  `td3_stability_10k.npz`
- img: [`img/08-td3-stability.png`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/img/08-td3-stability.png) (**the key figure**),
  `08-td3-stability-10k.png`
- 8 seeds each; 6000 steps except the two `-10k` configurations. 6000 is not a
  truncation of 10 000 — `noise_schedule linear` anneals sigma to zero at
  `timesteps`, so it is a different exploration schedule

| configuration | solved | final bid | | configuration | solved | final bid |
|---|---|---|---|---|---|---|
| baseline | **3/8** | −3.67 ± 55.73 | | **lr-1e-4** | **6/8** | **33.30 ± 0.69** |
| tanh | 0/8 | 100.00 ± 0.00 | | batch-128 | 4/8 | 33.89 ± 1.13 |
| policy-delay-8 | **0/8** | −9.44 ± 64.97 | | sigma-0.05 | 4/8 | 30.89 ± 29.55 |
| policy-delay-64 | 3/8 | 42.56 ± 14.02 | | sigma-0.3 | 1/8 | 2.59 ± 80.46 |
| warmup-250 | 3/8 | 26.11 ± 53.78 | | lr-3e-3 | 5/8 | −9.28 ± 56.11 |
| warmup-3000 | 2/8 | 18.70 ± 57.07 | | buffer-2000 | 3/8 | 56.62 ± 30.60 |
| noise-const | 3/8 | −25.03 ± 58.40 | | **assume-knobs** | **0/8** | 58.27 ± 49.31 |
| | | | | **assume-budget** | **0/8** | 93.11 ± 2.83 |

At run 05's own length: `run05-repro` **2/8** (−7.92 ± 62.14),
`lr-1e-4-10k` **8/8** (**32.10 ± 0.37**, reward +0.169 ± 0.004).

**Softsign does not fix this. It replaces a deterministic failure with a chaotic
one.** The trajectories show a new failure mode that runs 01–05 never saw: the
actor bolts to +100, the plateau pull flips on schedule, it comes back — and then
**overshoots straight through the 19-EUR band to −100**, landing on the −0.170
loss shelf. Failure has moved from the ceiling to the far side of the cliff.

#### How unstable: BLAS thread count decides it

`td3_stability.py` runs each seed in its own process at `torch.set_num_threads(1)`;
`run_benchmark.py` runs in-process at the torch default of 14. Nothing else
differs — same code, same seeds, same hyperparameters.

| run | seed 0 | seed 1 |
|---|---|---|
| archived run 05 (14 threads) | 31.86 | 31.60 |
| rerun, same command, 14 threads | 31.86 | 31.60 |
| rerun, same command, **1 thread** | **35.44** | **−60.49** |

Run 05 is *bit-identically reproducible* under its own conditions — it was never a
lucky draw. But a change with no mathematical content, the order in which a matrix
product accumulates, moves seed 1 from the constrained optimum to the loss shelf.
**Any single-configuration result on this landscape should be read as a sample
from a bimodal distribution, not as a measurement.**

#### What actually stabilises it

Not the window timing — run 06's prediction is **half wrong**. `policy-delay-8`
fails as predicted (0/8), but `lr-1e-4` is the *best* configuration in the sweep,
not a casualty of a missed window, and `warmup-3000` (which delays actor and
critic together, so the window merely opens later) is no worse than baseline.

The two configurations that converge tightly — `lr-1e-4` (8/8, ±0.37) and
`batch-128` (±1.13) — are the two that **shrink or smooth the actor's step**. Once
the actor is inside a 19-EUR band bounded by a 0.36 cliff, a full-sized Adam step
is simply too big to stay there, and the same scale-invariance that rescues the
actor from the plateau (correction 4) is what throws it off the cliff. Step size,
not activation, is the lever.

#### For ASSUME

`assume-knobs` (`policy_delay 8`, 32 gradient steps per 12 environment steps) is
**0/8 even with a full 6000-step budget**, and `assume-budget` (those knobs plus
the real 240 + 240-step budget) is 0/8 at 93.11 ± 2.83 — it never comes back at
all. So ASSUME's difficulty is not only the short budget: its update ratio sits in
the bad region independently.

### 09 — ASSUME's MATD3, filmed over a live run

**Why:** run 07 reads one frame — the networks a finished run left behind. Run 06
showed the frame that decides the outcome lasts a few hundred updates, so a single
end state cannot distinguish a critic that never flipped from one that flipped and
was not followed. This records the whole run.

```bash
python real_matd3/assume_training_probe.py --study-case inc_dec_learning_single_g0 \
    --episodes 40 --n-obs 6 --grid 401 --label shaped
# then comment out learning_strategies.py:1583-1589 and repeat with
#   --label unshaped --load-buffer buffers/single_10ep_standard.npz
python real_matd3/assume_film.py
```

- data: `assume_probe_{shaped,unshaped,unshaped_clean}.npz` in
  [`data/09-assume-films/`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/data/09-assume-films/)
- img: [`03-assume-matd3-films.png`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/03-assume-matd3-films.png) (**headline**)
- 40 episodes = **2560 critic gradient steps**, 4× the default budget, filmed at
  one frame per training block (32 updates), swept at 6 real observations

`assume_training_probe.py` wraps `matd3.TD3.update_policy` from the outside, so
**nothing in `assume/` is edited** and the learning dynamics are untouched. It
emits the same npz schema as the surrogate runs, with the *seeds* axis carrying
probed observations rather than random seeds.

| film | buffer | argmax `Q1` first → last | actor first → last |
|---|---|---|---|
| shaped | `single_10ep_gradient` (shaped) | 100.0 → **48.8** | 62.9 → **49.0** |
| unshaped | `single_10ep_standard` (true) | 100.0 → **100.0** | 61.3 → **94.4** |

**Phase 1 is confirmed on the real thing.** Both runs open with `argmax Q1` at the
+100 ceiling and the actor climbing — exactly the surrogate's phase 1, and exactly
where run 07's 640-update networks were frozen.

**Without shaping, the critic never leaves phase 1 — not in 2560 updates.**
`argmax Q1` sits at the ceiling for the entire run and the actor parks at 94.4.
The heatmap shows why and it is not subtle: the unshaped field is **mottled noise
across the whole bid axis**, with no coherent sign structure anywhere. The critic
never develops a preference to follow. This is 4× the default budget, so it is not
the budget of finding 10 — it is a different failure.

**With shaping, the same run converges cleanly to the band's rim.** The field is
smooth and single-signed, `argmax Q1` falls to ~48 within ~800 updates, and the
actor tracks it to 49.0. That is the decoy of run 07 doing exactly what it was
built for: it walks the actor down from the ceiling. It just stops one euro above
the band and never reaches 30.

**Why the surrogate flips and ASSUME does not** is the open question, and §2's
caveat is the leading candidate. The surrogate is single-context, so its critic
only has to learn `Q(a)`. ASSUME's critic must fit `Q(s, a)` across **548 distinct
observations** from ~600–1200 transitions, and on the true reward — 90 % of which
is flat — it apparently never resolves the action dependence at all. The shaped
reward makes `Q` a function of the action *everywhere*, which is a much easier
regression, and that may be the real reason the shaping works.

> **An earlier version of this section said both films flip.** That run
> (`assume_probe_unshaped.npz`, kept for the record) was labelled unshaped but
> preloaded `buffers/single_10ep_gradient.npz`, whose 280 stored rewards are
> *shaped* — about a third of its training data. Those shaped transitions were
> doing the work. `assume_probe_unshaped_clean.npz` is the corrected run and is
> what the table and figure above use.

### 10 — are those two films reproducible? 6 seeds each

**Why:** run 09 is one seed per condition, and run 08 is the standing reason not
to read one run on this landscape as a measurement. This reruns both films across
seeds and asks which parts of run 09's headline survive.

```bash
# repo as committed — the shaping is commented out
python real_matd3/assume_stability.py --condition unshaped --seeds 42 1 2 3 4 5
# then uncomment learning_strategies.py:1583-1589 and
python real_matd3/assume_stability.py --condition shaped   --seeds 42 1 2 3 4 5
# ...and comment it back out
python real_matd3/assume_stability.py --report
```

- data: [`data/10-assume-stability/`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/data/10-assume-stability/) —
  `assume_stab_{shaped,unshaped}_seed{42,1,2,3,4,5}.npz`
- img: [`img/10-assume-stability.png`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/img/10-assume-stability.png)
- **6 seeds per condition**, 40 episodes each, same 401-point grid and same 6
  probed observations as run 09. 29 min and 17 min wall for the two batches, 6
  concurrent processes at one torch thread each

**How the seed was varied.** ASSUME has a real seed knob and this uses it:
`loader_csv.py:555` calls `set_random_seed(config.get("seed", 42), ...)` once,
while the scenario is read, and nothing re-seeds afterwards.
`assume_training_probe.py --seed` re-applies that same call after the scenario is
loaded and before `run_learning()` builds the networks. Because the scenario's CSV
tables and forecaster contain no RNG draws, the only things downstream of it are
network initialisation, exploration noise and the replay-buffer batch draws — so
this varies the learner and holds the environment fixed. Two runs at `--seed 1`
are bit-identical in every recorded array; `--seed 2` is not. Seed 42 is ASSUME's
own default, i.e. a rerun of run 09.

| condition | seeds | argmax `Q1` last | disagreement over the 6 obs | range | actor last | true reward | positive reward at any frame |
|---|---|---|---|---|---|---|---|
| shaped | 6 | **49.5 ± 4.4** | 4.2 | 10.4 | **50.6 ± 3.0** | +0.007 ± 0.015 | **6/6** |
| unshaped | 6 | 89.7 ± 10.6 | **24.5** | 56.4 | 94.3 ± 7.0 | +0.000 ± 0.000 | **1/6** |

> **Two columns, because this run originally reported only the second and run 12
> compared its own *first* against it.** `disagreement` is the mean over distinct
> pairs of probed observations; `range` is `max − min`, roughly twice as large and
> more sensitive to one outlying observation. Both are now defined once, in
> [`analysis/critic_coherence.py`](analysis/critic_coherence.py), and every run
> from 10 to 13 reports the same pair. **Only `disagreement` is comparable across
> runs**; see correction 14.

**The shaped result reproduces.** All six seeds leave the ceiling within 96–192
critic updates and settle at the band's rim: final actor bids 44.9, 49.9, 50.2,
51.3, 53.4, 54.1. Run 09's 49.0 sits inside that. Five of the six stop just
*above* the band, which is finding 12's decoy behaving as described; only seed 2
ends inside it, at 44.9 for +0.041. All six do dip into the band transiently —
best true reward at any recorded frame averages +0.036, best single frame +0.065
(bid ≈ 42.5) — they simply do not stay.

**The unshaped conclusion reproduces; the unshaped *number* does not.** Run 09
reported `argmax Q1` sitting at exactly 100.0 for the whole run. Across six seeds
the final median `argmax` is 89.7 ± 10.6, and only two seeds end at 100.0 — the
others end at 73.2, 81.0, 84.8, 99.2. **Read on its own, that median is close to
meaningless here**, which is the more useful finding: at the last frame the six
probed observations of a single unshaped run disagree about the preferred bid by
**24.5 EUR between an average pair**, and span 56.4 EUR end to end (seed 4: 16.5
to 90.0). The shaped runs disagree by 4.2, spanning 10.4.
That is run 09's "mottled noise, no coherent sign structure" turned into a number,
and it says the unshaped critic has not learned a preference rather than that it
has learned to prefer +100.

What is not in doubt is the outcome. **Five of the six unshaped runs never place a
probed bid inside the 30–49 band at any frame of 2560 critic updates**, and the
sixth (seed 2) only grazes it — 13 of its 480 (observation, frame) cells, clustered
around frames 65–71, peaking at +0.124 — before ending at 94.4 like the rest. The
end states do not overlap at all: unshaped 79.3–99.2 against shaped 44.9–54.1. So
run 09's conclusion holds at 6 seeds; only its "100.0" should be dropped.

**ASSUME is not thread-chaotic the way the surrogate is.** Run 08's alarming
result was that one BLAS thread instead of 14 moved a surrogate seed from +31.60
to −60.49. These runs are pinned to one torch thread where run 09's were not, and
the unshaped seed 42 still reproduces run 09's film to **0.119 EUR maximum
deviation over all 80 frames** (final actor 94.45 vs 94.35). The shaped seed 42
is looser — up to 9.4 EUR mid-film, ending 51.3 against run 09's 49.0 — but well
inside the shaped condition's own ±3.0 seed spread. Run 08's chaos warning
appears to be a property of the SB3 surrogate, not of this scenario.

**Six seeds is six seeds.** The two conditions are cleanly separated here (no
overlap at all on final actor bid: 44.9–54.1 against 79.3–99.2), but nothing in
this run supports a claim finer than that, and in particular the shaped condition's
1/6 seeds that end *inside* the band is one seed, not a rate.

### 11 — ASSUME configuration stability on the true reward

**Why:** runs 09–10 established that unshaped MATD3 usually never forms a useful
action preference. This is the first broad sweep on the real ASSUME learner: vary
the most plausible optimisation, update-budget and noise knobs around the
requested baseline, while holding the scenario, initial data and diagnostics
fixed.

```bash
python real_matd3/assume_config_sweep.py --phase broad --workers 6

# redraw both multiseed views from the archived npz files, without retraining
python real_matd3/assume_config_sweep.py --phase broad --report-only \
  --critic-out ../../../outputs/2_nodes_paper_small/rl_benchmark/runs/img/11-assume-config-critic-evolution-broad.png
```

- data: [`data/11-assume-config-stability/broad/`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/data/11-assume-config-stability/broad/)
  — 90 `assume_cfg_<config>_seed<seed>.npz` files and the complete
  [`summary-complete.txt`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/data/11-assume-config-stability/broad/summary-complete.txt)
- actor img: [`img/11-assume-config-stability-broad.png`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/img/11-assume-config-stability-broad.png)
- critic img: [`img/11-assume-config-critic-evolution-broad.png`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/img/11-assume-config-critic-evolution-broad.png)
- **30 configurations × 3 seeds** (`42, 1, 2`), 40 episodes each, 80 recorded
  training blocks per run. The initial batch took 243.5 min wall time; six failed
  trials were recovered in a 14.6 min retry

The baseline is `learning_rate=1e-3`, `gradient_steps=10`, `batch_size=128`,
`policy_delay=2`, `gamma=0.99`, `tau=0.005`, exploration `sigma=0.1`, target
policy noise `0.2` clipped at `0.5`, and `train_freq=12h`. That is **800 critic
updates** over 40 episodes. The sweep changes one knob at a time where possible:

| group | configurations |
|---|---|
| learning rate | `3e-4`, `1e-4`, `3e-5` |
| batch size | `64`, `256`, `512` |
| policy delay | `1`, `4`, `8` |
| gradient steps | `4`, `20`, `32` |
| return / target update | `gamma={0,.90,.95}`, `tau={.001,.01}` |
| noise | target noise off or reduced; exploration `sigma={0,.05,.2}`; linear schedule |
| interactions | six combinations of low LR, delay 1, 32 steps, batch 256, gamma 0 and no target noise |

**One coherent starting point, not one live buffer shared between processes.**
Every trial loads the same immutable, true-reward
`buffers/single_10ep_standard.npz` (SHA256
`5f1b80b4a2cb98c1c166c35e48931e87ae24f06e92f88c46c44f768954c16a72`).
The runner verifies that checksum before launch. After loading, each process owns
and grows its own replay buffer, so trials cannot leak experience into each other.
The reward-shaping block was commented out throughout: there are no shaped or
"cheated" rewards in this sweep.

**Result: 0/3 solved for all 30 configurations.** Here "solved" means final mean
true reward over the six fixed observations at least +0.15. Almost every actor
still finishes on the upper zero-reward plateau.

| configuration | final actor bid | final true reward | reading |
|---|---:|---:|---|
| baseline | 99.4 ± 0.1 | +0.000 | phase 1 persists |
| `lr-1e-4` | 95.5 ± 0.3 | +0.000 | surrogate's stable LR does not transfer at this budget |
| `policy-delay-1` | 99.8 ± 0.1 | +0.000 | twice the actor-update rate is not enough |
| `grad-20` | 99.1 | +0.000 | extra updates still insufficient |
| **`grad-32`** | **78.7 ± 16.1** (60.7–99.8) | **+0.002 ± 0.003** | only setting with material descent; still unstable and unsolved |
| `lr-1e-4-grad-32` | 98.2 | +0.000 | lowering the step size cancels that movement in 40 episodes |

Gamma zero, target smoothing off, exploration-noise changes, batch size, tau and
the other interactions also fail to rescue the learner. The strongest lever is
therefore **critic-update budget**, not any stabiliser by itself: `grad-32`
performs 2560 critic updates and is the only configuration that develops a
substantial descending region and moves actors off the ceiling, but even it
does not reach a stable solution in this horizon.

**The critic evolution is archived for every run.** Each npz contains Q1 and Q2,
their exact autograd gradients over 401 bids at six fixed real observations, the
actor path, and exact config metadata; the critic and actor arrays cover every
one of the 80 training blocks.
The critic figure facets all 30 configurations; colour is the median Q1 spatial
gradient across 3 seeds × 6 observations over training time, and black lines are
the three per-seed median actor trajectories. Almost every facet stays positive
over most of the bid axis — phase 1 still pushes upward. `grad-32` is the visible
exception, but its descending field does not form reliably across seeds.

For a single run, including the temporal Q1/Q2 gradient films and final twin-
critic landscape, use:

```bash
python real_matd3/assume_run_diagnostics.py \
  --results <data/11-assume-config-stability/broad/assume_cfg_...npz> \
  --out <diagnostic.png>
```

**The six initial failures were logging failures, not failed learning
runs.** All three `target-noise-0.1-clip-0.2` and all three
`gamma-0-target-noise-0` processes hit a concurrent TensorBoard async-writer
`FileNotFoundError`. Their two-frame partial archives were moved to
`broad/partial_failures_before_retry/`; the retry disables only TensorBoard in
the probe processes and preserves database logging and learning dynamics. The
final audit is **90/90 complete files**, 30 configs with exactly 3 seeds and 80
frames each, all carrying the same starting-buffer checksum. The original full
launch manifest is retained as `manifest-initial-broad.json`.

**This is a broad screen, not a stability-rate estimate.** Three seeds
and 40 episodes can reject easy fixes, but cannot establish a success rate.
Moreover, changing `gradient_steps` changes total optimisation work, so the
`grad-32` comparison deliberately does not separate update budget from episode
exposure. That is the next experiment.

### 12 — the action's share of the critic's input (**headline figure 4**)

**Why:** run 11 exhausted the hyperparameters. Run 09's open question — why the
critic learns nothing from the true reward — had one standing candidate, §2's
"it must fit `Q(s, a)` across 548 observations rather than `Q(a)` at one". That
candidate turns out to be wrong in its obvious form, and the thing that replaces
it is not a hyperparameter at all.

#### The offline experiment that located it

Before touching the live learner, the RL loop was stripped off: γ = 0, so the
target is the stored reward — no bootstrap, no moving target, no actor feedback,
no growing buffer. What remains is exactly the regression the critic must solve,
with ASSUME's own `CriticTD3`, AdamW at lr 1e-3, batch 128, clip 1.0, on the
frozen `single_10ep_standard` buffer. 5 seeds, 2560 updates.

| condition | argmax Q1 | band_neg | Q(32) − Q(100) | train MSE | test MSE |
|---|---|---:|---:|---:|---:|
| `full-obs` — what ASSUME does | **95.3 ± 4.8** | **0.04** | **−0.086** | 0.00012 | 0.00486 |
| `const-obs` — the surrogate's setting | 32.9 ± 1.9 | 0.84 | +0.061 | 0.00151 | 0.00205 |
| `shuffled-obs` | 98.8 ± 1.8 | 0.02 | −0.120 | 0.00011 | 0.00804 |
| `obs-x0.1` | 34.0 ± 1.2 | 0.78 | +0.053 | 0.00000 | 0.00109 |
| `obs-2dim` | 32.8 ± 1.8 | 0.84 | +0.081 | 0.00155 | 0.00220 |

`band_neg` is the share of grid cells in `[30, 49]` carrying the true negative
slope; 0.50 is a coin flip. **The live failure reproduces in a plain supervised
fit**, so it is not the bootstrap, the actor or the budget. And two intuitive
explanations die here: `shuffled-obs`, whose observations have no association
with the reward at all, fails *identically*, so it is not real contextual
structure; and `obs-x0.1` drives train MSE to zero — memorising harder than
`full-obs` — while learning the band correctly, so it is not memorisation
capacity either.

A second round separated ratio from magnitude by scaling the *action up* instead,
at full-scale observations. Both sides of the ratio produce the same ordering:

| | act_share | argmax Q1 | band_neg | | | act_share | argmax Q1 | band_neg |
|---|---:|---|---:|---|---|---:|---|---:|
| obs ×1 (ASSUME) | 0.030 | 95.3 ± 4.8 | 0.04 | | act ×2 | 0.058 | 65.2 ± 12.3 | 0.10 |
| obs ×0.5 | 0.058 | 60.6 ± 13.7 | 0.22 | | act ×5 | 0.133 | 46.5 ± 6.7 | 0.28 |
| obs ×0.2 | 0.133 | 38.5 ± 1.4 | 0.61 | | act ×10 | 0.234 | 38.4 ± 2.2 | 0.57 |
| obs ×0.1 | 0.234 | 34.0 ± 1.2 | 0.78 | | act ×30 | 0.479 | 33.7 ± 1.6 | 0.79 |
| obs ×0.03 | 0.505 | 32.6 ± 1.0 | 0.85 | | obs z-scored | **0.008** | **100.0 ± 0.0** | 0.02 |

with

```
act_share = sd(a) / (sd(a) + sum_j sd(obs_j))       = 0.030 as ASSUME is configured
```

a std-share over the 75 critic inputs (`sd(a) = 0.592`, `sum_j sd(obs_j) = 19.3`).
The variance-share `sd(a)² / (sd(a)² + sum_j sd(obs_j)²) = 0.055` is the version
with a mechanism behind it, since Xavier init makes the first-layer weights iid
across inputs so their contributions to the pre-activation variance add in
quadrature. Both order the sweep identically. Note the direction: **fewer
observation dimensions raises `act_share`**, and z-scoring the observation — which
makes it *more* dominant — is the worst cell in the table, pinned at exactly 100.0
in 5/5 seeds.

A third round removes dimensions instead of rescaling them — the offline version
of the `foresight` lever, using the same truncation the live sweep applies, so
the observation matches what `create_observation` builds at that foresight:

| foresight | obs_dim | act_share | argmax Q1 | band_neg | Q(32) − Q(100) | test MSE |
|---:|---:|---:|---|---:|---:|---:|
| 24 | 74 | 0.030 | 95.3 ± 4.8 | 0.04 | −0.086 | 0.00486 |
| 12 | 38 | 0.057 | 66.4 ± 11.7 | 0.36 | −0.001 | 0.00427 |
| 6 | 20 | 0.108 | 37.2 ± 5.2 | 0.57 | +0.027 | 0.00193 |
| 3 | 11 | 0.191 | 33.2 ± 1.1 | 0.81 | +0.080 | 0.00176 |
| 1 | 5 | 0.356 | **31.1 ± 0.9** | 0.89 | +0.110 | 0.00113 |

Three mechanically unrelated levers — rescaling the observation, rescaling the
action, deleting observation dimensions — all track `act_share`. It predicts the
*ordering* across levers, not the exact value: at a matched 0.133, `obs ×0.2`
gives `band_neg` 0.61 against `act ×5`'s 0.28.

Reproduce all three rounds with
`python real_matd3/assume_offline_critic.py` (~15 min, 5 seeds, no simulation).

#### The live run

```bash
python real_matd3/assume_actshare_sweep.py --workers 5
python real_matd3/assume_actshare_film.py          # both figures, from the archive
```

- data: [`data/12-actshare/`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/data/12-actshare/) — 15 `assume_as_<condition>_seed<seed>.npz`
- headline: [`12-actshare-dose-response.png`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/12-actshare-dose-response.png)
- mechanism: [`img/12-actshare-descent-window.png`](../../../outputs/2_nodes_paper_small/rl_benchmark/runs/img/12-actshare-descent-window.png)
- **5 conditions × 3 seeds** (`42, 1, 2`), 40 episodes, true reward (shaping
  commented out), run 11's `BASELINE` config, the same checksum-guarded starting
  buffer, 80 recorded frames per run. ~15 min per trial, 5 workers

Neither lever edits `assume/`. `foresight` is forced through the strategy's own
kwarg (`learning_strategies.py:1338`), which changes `obs_dim = 3·foresight + 2`
(`base.py:952`); the action lever patches `CriticTD3` to fit `Q(s, S·a)`, leaving
the actor's output, the bid mapping and the environment untouched. Adam is
scale-invariant, so the actor's *step size* is unchanged even though its gradient
is S times larger. A reduced-foresight run cannot load the 74-dim buffer, so a
truncated copy is derived from the same transitions — first k of each forecast
block, **last** k of the price history, because `create_observation` builds that
one with `direction="backward"` (`fast_pandas.py:864-867`).

| condition | act_share | obs_dim | final bid | final true reward | argmax Q1 | obs spread | band_neg | solved |
|---|---:|---:|---|---:|---|---:|---:|---:|
| `baseline` | 0.030 | 74 | 99.4 ± 0.1 | +0.000 | 85.5 ± 11.2 | 21.7 | 0.11 | 0/3 |
| `foresight-6` | 0.108 | 20 | 63.9 ± 24.2 | +0.012 | 53.3 ± 13.2 | 22.5 | 0.38 | 0/3 |
| `foresight-3` | 0.191 | 11 | 40.4 ± 2.5 | +0.087 | 37.5 ± 1.8 | 1.8 | 0.61 | 0/3 |
| `act-x10` | 0.234 | 74 | −7.4 ± 61.0 | +0.033 | 35.2 ± 2.2 | 8.8 | 0.72 | 0/3 |
| **`act-x30`** | 0.479 | 74 | **33.0 ± 0.2** | **+0.160** | 33.2 ± 1.1 | 1.8 | 0.81 | **3/3** |

> The `obs spread` column is now the shared `argmax_disagreement` of
> [`analysis/critic_coherence.py`](analysis/critic_coherence.py): the mean over
> **distinct** pairs of probed observations. Earlier versions of this table
> divided by `n²` and so included the `n` zero self-pairs, reading 5/6 of these
> (18.1, 18.7, 1.5, 7.4, 1.5). See correction 14.

**`final true reward` above is *reconstructed*, not measured** — it applies the
surrogate curve to the recorded bid, and that curve matches what the simulator
actually paid on only 24.8 % of the frozen buffer's transitions (correction 15).
The measured counterpart, read from each trial's own `rl_params` table:

| condition | measured (eval) | reconstructed | per-seed measured | final bid |
|---|---:|---:|---|---:|
| `baseline` | +0.000 ± 0.000 | +0.000 | 0.000, 0.000, 0.000 | 99.4 |
| `foresight-6` | +0.041 ± 0.058 | +0.012 | 0.123, 0.000, 0.000 | 63.9 |
| `foresight-3` | +0.088 ± 0.015 | +0.087 | 0.105, 0.091, 0.067 | 40.4 |
| `act-x10` | +0.025 ± 0.159 | +0.033 | −0.200, 0.135, 0.139 | −7.4 |
| **`act-x30`** | **+0.167 ± 0.005** | +0.160 | 0.167, 0.173, 0.162 | 33.0 |

**The headline survives on measured reward and `act-x30` measures slightly better
than the reconstruction claimed.** The ordering, the monotonicity in `act_share`
and the 3/3 all hold. What is *not* established against the simulator is the
`+0.15` bar itself, the `regret` figures, the `32.31` constrained optimum and the
exact `[30, 49]` band — all four are properties of the surrogate curve.

Second caveat on that column: the `rl_params` table holds only the **first two
products of each episode** (10:00 and 11:00 of 14), so it is an early-hours
sample, not the episode mean (correction 16). Settling this properly needs the
probe to record buffer rewards the way `MultiAgentRecorder` already does, and a
re-run — deferred.

**`baseline` reproduces run 11's own baseline cell** — 99.4 ± 0.1, +0.000, 0/3,
with the incoherent critic (per-seed `argmax Q1` 100.0 / 83.8 / 72.8). The rest of
the table is therefore the levers and not the harness.

**`act-x30` is the first configuration in this archive to solve the true-reward
task**, against 0/90 in run 11. All 18 probed bids across its 3 seeds lie in
32.0–34.8; per-seed rewards +0.161 / +0.159 / +0.158. It does not sit on the
deterministic optimum of 30 and should not: §6's constrained optimum for a policy
of residual width σ ≈ 1 is 32.31, which is where SAC lands in the surrogate.

**Run 10's incoherence statistic collapses.** The disagreement between the six
probed observations about the preferred bid — 24.5 EUR for run 10's unshaped
condition and 21.7 for `baseline` here, i.e. *the same failure* — falls to **1.8**
at both `foresight-3` and `act-x30`. The critic forms a preference, and it is the
right one.

> An earlier version of this paragraph read "56.4 EUR for run 10's unshaped
> condition, 18.1 for `baseline` here", which compared run 10's **range** against
> run 12's **mean pairwise** number and made `baseline` look three times more
> coherent than run 10's unshaped critic. On either statistic the two are in fact
> indistinguishable — 24.5 vs 21.7 as a disagreement, 56.4 vs 45.3 as a range —
> which is what `baseline` reproducing run 11 should have predicted. Correction 14.

**`act-x10` is bimodal, and the failure is run 08's.** Its −7.4 ± 61.0 is not a
spread around a mean: seeds 1 and 2 converge into the band (+0.123, +0.146) while
seed 42 overshoots through it to −93.7 and lands on the −0.170 shelf. That is the
surrogate's overshoot failure appearing in ASSUME for the first time — the critic
is fixed and the actor's step is now the binding constraint. It does not recur at
`act-x30` (0/3), but 3 seeds cannot distinguish that from luck.

#### The window, and how it differs from run 06's

`img/12-actshare-descent-window.png` measures the field with run 06's own
definitions, so the two are comparable. "Clean path" is an unbroken `dQ/d(bid) < 0`
walk from bid 100 reaching the band; "settles" is the update from which the actor
never leaves the band ±3 again.

| condition | act_share | actor starts | peak bid | first clean | clean share | open at end | pulled left | settles | settles at |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 0.030 | 65.1 | 99.5 | 585 | 2 % | 11 % | 56 % | 0/18 | — |
| `foresight-6` | 0.108 | 44.7 | 96.4 | 300 | 21 % | 33 % | 39 % | 5/18 | 730 |
| `foresight-3` | 0.191 | 40.2 | 94.7 | 455 | 27 % | **89 %** | **100 %** | 18/18 | 580 |
| `act-x10` | 0.234 | 63.4 | 74.5 | **20** | 27 % | 22 % | 44 % | 12/18 | 180 |
| `act-x30` | 0.479 | −38.1 | 74.8 | **20** | 37 % | 33 % | 50 % | 18/18 | 280 |

Three things differ from run 06's surrogate window:

- **It is not transient.** Run 06's path opened at step ~1400 and fragmented after
  ~600 steps. Here `clean share` *rises* with `act_share` and an unbroken path is
  still present at the final frame in 33 % of `act-x30` traces and 89 % of
  `foresight-3`'s. There is nothing to race against.
- **At high `act_share` it is there from the start** — first clean path at update
  **20**, i.e. the second recorded frame, rather than after 585. Phase 1 is
  shortened rather than skipped: peak bid falls from 99.5 to ~74.5, so the actor
  still climbs on the incomplete critic, just not to the ceiling.
- **The crossing costs 180–580 updates**, comfortably inside the run 06 budget of
  190–410 actor updates that finding 12 said ASSUME's 80 could not pay — because
  the path no longer has to be paid for inside a closing window.

`pulled left` is the sign of `dQ1/d(bid)` at the actor's own action on the last
frame, and it separates "converged" from "still moving": `act-x30` is 50 %, a coin
flip, which is what a converged optimum looks like; `foresight-3` is **100 %**, so
it was still descending when the 800-update budget ran out at bid 40.4. That is a
prediction — `foresight-3` should improve with more episodes — not a result.

**Caveats.** Three seeds per condition and one scenario: `act-x30`'s 3/3
rejects "nothing simple works", it is not a success rate. `foresight-3` puts 3/3
seeds inside the band but is 0/3 against the +0.15 bar, so it is a partial rescue.
S = 30 is an arbitrary point on a curve that appears to saturate around
`act_share` 0.25–0.5; nothing beyond was tested. Cutting foresight discards
forecast information that is worthless *in this scenario* and would not be in
general. The recorded `dQ/d(bid)` carries the factor S in the action-scale rows,
so signs are comparable across conditions and magnitudes are not, and each facet
of the figure is drawn on its own robust scale for that reason. Finally, the debug
prints at `matd3.py:618-628` were live for all 15 trials; they consume a
`th.rand(1)` draw per gradient step, but they affect every condition equally and
`baseline` reproduced run 11, so comparability holds.

### 13 — eleven learning agents: does run 12 transfer?

**Why:** run 12 is one learning unit. §7 predicted the multi-agent case should be
*worse*, because a centralised critic adds `unique_obs_dim` observation dimensions
**and one action dimension** per agent while agent *i*'s own action stays a single
scalar — 0.030 at N = 1 falling to 0.017 at N = 16. `inc_dec_learning` is that
case: all 11 units of `powerplant_units_learning.csv` learn, on a 72 h horizon.
Each agent's critic sees **94 observation + 11 action = 105 inputs**.

```bash
python real_matd3/assume_multiagent_actshare.py                       # the two 50-episode runs
python real_matd3/assume_multiagent_actshare.py --conditions act-all-x2 --seeds 42 1 2
python real_matd3/assume_multiagent_actshare.py --report-only
python real_matd3/assume_multiagent_grids.py    # the three per-seed figures
python real_matd3/assume_multiagent_window.py   # the descent-window table
```

- data: `data/13-multiagent-actshare/` — 18 `assume_ma_<condition>_seed<seed>.npz`
- img: `img/13-multiagent-critic-grid.png`, `img/13-multiagent-bids-grid.png`,
  `img/13-multiagent-summary.png`, plus `img/13-multiagent-window.png`
- **6 conditions × 3 seeds** (`42, 1, 2`), true reward (shaping commented out), the
  study case as it stood in the working tree. Only early stopping is disabled
  (`early_stopping_steps` 1e6), so both budgets are guaranteed. ~31 min per
  25-episode trial, ~59 min per 50-episode trial

**The `inc_dec_learning` study case in the working tree is not the committed
one, and run 13 used the working-tree version.** At the time of the runs
`config.yaml` was modified but uncommitted:

| field | committed at `9cf080eb` | used by run 13 |
|---|---|---|
| `end_date` | 2019-01-01 **05:00** | 2019-01-**04** 00:00 (72 h) |
| `learning_rate` | 0.001 | **0.0001** |
| `training_episodes` | 8 | 50 |
| `episodes_collecting_initial_experience` | 2 | 5 |
| `train_freq` | **1h** | **12h** |
| `validation_episodes_interval` | 2 | 5 |

The horizon matters most: 72 h at `train_freq` 12h gives 6 training blocks per
episode, which is where 45 × 6 = 270 recorded frames and 2700 critic updates come
from. A fresh clone would instead run a 5 h horizon at `train_freq` 1h — the
setting §"things that did not work" records as dying with
`AssumeException: No rewards were collected during evaluation run`. **Restore the
table above before re-running run 13.**

#### The lever had to change, and that is itself a result

Run 12's action lever multiplies the critic's **whole** action input by S. With N
agents that raises every agent's action together, so an agent's own share
**saturates at 1/N — 0.091 here — for any S**. The requested 0.2 is unreachable
that way. So two levers are run:

| mode | what it scales | own share at S = 15 | action-block share |
|---|---|---:|---:|
| `act-all` | the whole action vector (run 12's patch verbatim) | 0.069 | 0.764 |
| `act-own` | only critic *i*'s own action column | 0.197 | 0.329 |

`act-own` is symmetric across agents (each critic upweights its own actor's
action, none is privileged), is the input `matd3.py:711` differentiates for the
actor loss, and reduces to run 12's patch at N = 1. Because the two move the own
share and the block share in *opposite* proportions, they separate two things run
12 moved together. `act-all-x2` was added as the block-matched control: block
0.302 against `act-own-x15`'s 0.329, own share only 0.027.

#### The result: the ordering is `act_share`, at both budgets

`act_share` here is measured on each run's **final** buffer, which is lower than
the value the lever was set from — the policy concentrates, `sd(a)` falls. Set
from the collection buffer, `act-own-x15` is 0.197; it reads 0.137 at the end.

| condition | updates | own share | `diesel_0` final bid, per seed | mean ± sd | `diesel_0` reward | fleet |
|---|---:|---:|---|---:|---:|---:|
| `baseline-25` | 1200 | 0.015 | 97.2, 96.8, 97.3 | **97.1 ± 0.2** | +0.018 | +6.26 |
| `act-all-x2` | 1200 | 0.024 | 60.9, 95.3, 81.6 | 79.2 ± 14.1 | +0.232 | +5.92 |
| `act-all-x15` | 1200 | 0.065 | 49.1, 44.8, 47.3 | **47.1 ± 1.7** | +0.526 | +5.01 |
| `baseline` | 2700 | 0.013 | 73.8, 69.3, 97.3 | 80.1 ± 12.3 | +0.232 | +5.93 |
| `act-all-x2-50` | 2700 | 0.023 | 45.6, 47.6, 62.5 | 51.9 ± 7.5 | +0.378 | +4.20 |
| `act-own-x15` | 2700 | 0.137 | 39.2, 25.7, 22.1 | **29.0 ± 7.4** | +0.493 | +5.00 |

**The ordering is monotone in the own share at each budget separately** —
97.1 → 79.2 → 47.1 at 1200 updates, 80.1 → 51.9 → 29.0 at 2700 — and mean reward
follows it. **The action-block share does not predict**: `act-all-x15` carries
3.2× `act-own-x15`'s block share (0.728 vs 0.230) and ends 18 EUR further from the
band. So it is not the action block the critic must notice; it is each critic's
own actor's action, which is also the only thing uniform scaling cannot deliver.

#### But it buys rate, not feasibility — run 12's framing does **not** transfer

`act-all-x2` reaches 79.2 in 1200 updates. **The baseline reaches 80.1 in 2700** —
the same place for ~2.25× the budget. And the baseline is not frozen: all three
seeds sit at 97.1 ± 0.2 at 1200 updates, but by 2700 two of three have descended
to 73.8 and 69.3, with a coherent leftward field forming from ~2000 updates
(`img/13-multiagent-critic-grid.png`, row 4).

That is a real departure from runs 09–12. There the unshaped single-agent critic
never formed a preference in 2560 updates and the shaping or `act_share` was what
made learning possible at all. Here **every condition descends eventually** and
`act_share` sets how fast. Any claim of the form "MATD3 cannot learn this without
X" must be stated with a budget attached.

#### The multi-agent baseline is also not the single-agent failure mode

`diesel_0` does park at 97.1 with `argmax Q1` exactly 100.0, which reproduces. But
the northern units go to the **floor** (wind and coals at ≈ −95 in several seeds)
and come back off it, and the critic's preference is coherent and bang-bang rather
than incoherent. **Run 10's incoherence statistic does not transfer**: the
condition that works best, `act-own-x15`, has the *highest* observation
disagreement about `argmax Q1` (47.7 EUR, mean over the eleven agents at the final
frame) and the baselines the lowest (13.7 at 1200 updates, 21.2 at 2700) —
the reverse of run 12, where the fix drove it from 21.7 to 1.8. With eleven agents
the critic's preferred bid genuinely *should* depend on the observation, so the
statistic stops being a failure diagnostic here.

#### The window is a slow opening, not a transient

Measured with run 06's definitions (`assume_multiagent_window.py`, which imports
`descent_stop` from `assume_actshare_film` so the numbers are on one footing):
at its 1200-update budget `act-all-x2` ends with `pulled left` — the sign of
`dQ1/d(bid)` at the actor's own bid — at **1.00 in 6/6 probed observations on two
of three seeds**, its plateau pull still rising and its descent path still
deepening. That is run 12's `foresight-3` signature, "still descending when the
budget ran out", and doubling the budget confirms it: 79.2 → **51.9 ± 7.5**, with
the seed spread halving.

#### Two reproducibility results, free

Both short conditions are **bit-identical prefixes of their long counterparts**,
on `greedy`, `critic_q`, `critic_grad`, `rewards` and `steps`, for all three seeds:

- `baseline-25` == `baseline` truncated to 120 frames, 3/3
- `act-all-x2` == `act-all-x2-50` truncated to 120 frames, 3/3

So nothing in this configuration depends on `training_episodes` (no learning-rate
or noise schedule is active and early stopping is off), the runs are deterministic
given seed and thread count, and the doubling comparison above is **the same three
trajectories continued**, not a fresh sample. It also confirms run 10's finding
that ASSUME does not inherit the surrogate's BLAS-thread chaos.

**Fleet reward runs opposite to `diesel_0`'s in every row.** The best condition
for `diesel_0` (`act-all-x15`, +0.526) has the second-lowest fleet total, and
`act-all-x2-50` the lowest at +4.20. Agents compete, so bids falling means prices
falling; total reward is not what any agent maximises. **"Solved" cannot be defined
by fleet profit here**, and none of these numbers should be read as a welfare
result without a separate equilibrium analysis.

**Three seeds and one scenario.** `act-all-x2` at 25 episodes spans 60.9–95.3
and the 50-episode baseline spans 69.3–97.3; those are wide. The ordering is
consistent across two independent budgets, which is the strongest thing on offer
at n = 3, but it is not a rate.

---

## 4. Corrections made along the way

Recorded so they are not silently reintroduced.

1. **"The critic smooths the cliff into a ramp, so the actor slides past the
   spike."** Wrong — run 02 shows the critics learn the cliff accurately. The
   real mechanism is the marooning: the actor sits where `dQ/da` ≈ 10⁻⁴.
2. **"Slowing the actor (`policy_delay`) should fix it."** Wrong — run 04.
3. **The `hit (<0.5 EUR)` and `regret` columns printed by `summarize()` are a
   bad metric.** See §6. They are still in the code; treat them with suspicion.
4. **"The gradient is 1e-4, so the actor is stuck."** Wrong, and the wrong kind
   of wrong — it confuses magnitude with direction. Adam steps by
   `lr·G/(G+eps)`, so any consistently-signed gradient two orders above
   `eps = 1e-8` gives essentially a full-sized step. The actual cause is the
   tanh output pinning to exactly 1.0 in float32 (§7, run 05).
5. **"The fragmented gradient field is the binding constraint."** Not supported —
   run 05's softsign actor crosses the field successfully while scoring worse on
   every path-consistency metric than the tanh run that never moved (§7).
   **Superseded by 6:** the evidence for this was pooled over the wrong interval.
6. **"Softsign scores worse on every path metric and crosses anyway, so spatial
   consistency is not binding."** The observation is real, the inference is not.
   Run 06 resolves both at 25-step resolution: during the interval in which the
   actor actually crosses, the path is clean **91–100 % of the time for both
   activations**. The pooled statistic is dominated by the ~2000 steps *after* the
   crossing, when the critic has learned the plateau is flat and its gradient there
   is noise — and, for the tanh runs, when the actor is parked at the ceiling
   feeding the buffer that produces exactly that noise. The right statement is
   narrower: **a clean path was necessary and it was present; only softsign could
   act on it.** Nothing here shows that a fragmented field would have been
   crossable.
7. **The critic sweep used `min(Q1, Q2)` for every algorithm.** Only SAC's actor
   climbs the min; TD3/DDPG climb Q1 alone. Fixed in `critic_probe.py`
   (`actor_objective`), which now dispatches per algorithm. Also, gradients were
   finite differences of the sampled curve; they are now `torch.autograd` — the
   same quantity `actor_loss.backward()` propagates. The two agree to ~0.1% in
   the flat regions and differ by up to 4 × 10⁻² at the kink near 32 EUR/MWh.
8. **"TD3 loses two seeds of four into the band and two through it" (run 01b).**
   Miscounted. The archived finals are `31.86, 31.60, 31.83, −88.93`: **three in,
   one through**. The mean and spread quoted, 1.59 ± 52.26, were always right —
   only the seed count in the prose was wrong.
9. **"`policy_delay` does nothing here" (correction 2, run 04).** Half wrong, and
   in an informative direction. Raising it does not help — that part stands. But
   run 01b's critic row shows `policy_delay = 2` is already *costing* TD3 the
   descent window: it crosses the plateau 200–600 environment steps later than
   DDPG and SAC because it takes half as many actor updates per step, and the one
   seed that fails is the one still on the plateau when the window closes. The
   knob matters; run 04 turned it the wrong way.
10. **"Without shaping, `argmax Q1` stays at 100.0" (run 09).** The conclusion
    survives 6 seeds, the number does not — across run 10's seeds the final
    median `argmax` is 89.7 ± 10.6 and only 2/6 end at 100.0. The median is a
    poor summary here: the six probed observations of one unshaped run disagree
    about the preferred bid by 24.5 EUR between an average pair, spanning 56.4 EUR
    end to end. The defensible statement is that the
    unshaped critic never develops a coherent preference, and that 5/6 runs never
    place a probed bid in the band at all while the sixth only grazes it (13 of
    480 cells) before ending at 94.4.
11. **"The real critic fails because it must fit `Q(s, a)` across 548 distinct
    observations, which is a harder regression" (runs 09, 13, and §2's caveat).**
    Wrong in that form, and run 12 has the controls. The reward is 95 % a function
    of the bid alone, so the regression is not made hard by being contextual; the
    observation carries *no* generalising information about the reward
    (LOO 1-NN R² = −1.07); an offline critic fed **shuffled** observations, which
    have no association with the reward whatsoever, fails identically to one fed
    the real ones; and a critic that memorises *harder* (`obs-x0.1`, train MSE → 0)
    learns the band correctly. What the number of observations was standing in for
    is the number of input **dimensions**: the action is 1 of 75, and the other 74
    carry 97 % of the input variation. Keep the observation, keep all 548 contexts,
    keep the memorisation — just weight the action more — and the same learner
    solves the task.
12. **"Without the shaping or a raised `act_share`, ASSUME's MATD3 never forms a
    usable action preference" (runs 09–12).** True of the single-agent case at
    the budgets tested; **false as stated for the multi-agent one** (run 13).
    With eleven learners the unmodified baseline sits at 97.1 ± 0.2 at 1200
    critic updates but descends to 80.1 ± 12.3 by 2700, two of three seeds
    reaching ~70. Raising `act_share` still orders the outcome monotonically at
    both budgets, but what it buys is **rate**: `act-all-x2` reaches 79.2 in 1200
    updates where the baseline needs ~2700 for the same place. State a budget
    with any "cannot learn" claim.
13. **"Run 10's `argmax Q1` disagreement measures critic incoherence" (runs 10,
    12).** It does in the single-agent case, where the observation carries no
    reward information. In run 13's genuine multi-agent setting the best
    condition has the *highest* disagreement and the failing baseline the lowest,
    so the statistic inverts. Do not carry it across without first checking that
    the observation is uninformative about the reward.
14. **"Run 10's unshaped critic disagrees by 56.4 EUR and run 12's `baseline` by
    18.1, so the baseline is markedly more coherent."** Those are **two different
    statistics**. Run 10 (and run 11) reported the *range*, `max − min` over the
    probed observations; runs 12 and 13 reported a *mean pairwise* difference —
    and divided it by `n²`, including the `n` zero self-pairs, so it was a further
    5/6 of the true pairwise mean. Matched, the two conditions are
    indistinguishable: **24.5 vs 21.7** as a disagreement, **56.4 vs 45.3** as a
    range. That is the right answer — `baseline` was built to reproduce run 11 and
    it does, on coherence as well as on the final bid — but it is not what the
    archive said. Both statistics now live in
    [`analysis/critic_coherence.py`](analysis/critic_coherence.py) and every run
    from 10 to 13 reports both, so they cannot drift apart again. The affected
    numbers, all recomputed from the archive rather than rescaled: run 10 unshaped
    24.5 (was 56.4) and shaped 4.2 (was 10.4); run 12 `baseline` 21.7, `foresight-6`
    22.5, `foresight-3` 1.8, `act-x10` 8.8, `act-x30` 1.8 (was 18.1, 18.7, 1.5, 7.4,
    1.5); run 13 `act-own-x15` 47.7 against the baselines' 13.7 and 21.2.
    **No conclusion changes** — the collapse at high `act_share` and the inversion
    of correction 13 both survive on either statistic.
15. **"`reward_from_bid` is the scenario's reward, so a recorded bid can be scored
    with it."** It is not. Against the 620 stored rewards of the frozen
    `single_10ep_standard` buffer, the closed form agrees on **24.8 %** of
    transitions: MAE 0.038, maximum error 0.369, R² 0.78, 466/620 rows differing.
    The mismatch is structural — the real EOM price varies hour to hour, so the
    loss shelf is **three** values (−0.20, −0.25, −0.30, implying clearing prices
    48/43/38) against the surrogate's single −0.17; `diesel_0` carries
    `additional_cost 68`, not 66; and the measured profitable region runs 28.1 to
    47.4, so bids below the cliff sometimes pay and the band's top does not.
    (`max_power` 5000 vs 1000 is *not* part of it — volume cancels between the
    profit legs and the reward normaliser.)
    **What this does and does not invalidate.** Bids, critic fields, `act_share`,
    the offline MSE fits and every trajectory remain measured quantities. What was
    never established against the simulator is the *reconstructed* reward: the
    `+0.15` solved bar, `regret`, the `32.31` constrained optimum and the exact
    `[30, 49]` band. §12 now tabulates the measured reward beside it and the
    headline holds; §11's and §10's reward columns have **not** been recomputed and
    are relabelled `recon` rather than corrected.
    The fix is *not* to retune `PAPER_SMALL` — one shelf cannot represent three,
    and the surrogate is exact for runs 01–08 by construction, since `IncDecEnv`
    is defined from it. The fix is to stop scoring real runs with it.
    Pinned by `test_rl_benchmark.py`.
16. **"The evaluation reward summarises the episode."** Every inspected run-12
    database holds exactly **two rows per episode**, at 10:00 and 11:00 — the
    first two of the episode's 14 delivery products — while the replay buffer
    grows by 14. `run_learning` (`loader_csv.py:1347`) calls `get_sum_reward()`
    immediately after `world.run()`, and the writes are scheduled as instant
    messages in `learning_role.py:647`, so the tail is not flushed before the
    query. Consequences: `compare_and_save_policies`, early stopping and any
    evaluation score depend on two particular hours. Runs 11–13 disabled early
    stopping, so their trajectories are unaffected; run 09/10's saved "best"
    policies and §12's measured column are not. **Training is unaffected** — it
    reads the cache directly, not the database.
17. **"Run 13's recorded critic field is the objective the actor differentiates."**
    It is a valid slice of the same critic, but not that objective. `matd3.py:704`
    clones the **replay batch's** actions and replaces only column *i*, so the
    other agents sit at stored behaviour actions; `MultiAgentRecorder` holds them
    at their **current actors' greedy outputs**. The recorder documents which it
    takes, but the window, `pulled left` and coherence readings of §13 therefore
    describe the critic's response to the current joint policy. **At N = 1 the
    distinction is empty** — there are no other columns — so runs 09–12 are
    untouched. Pinned by `test_rl_benchmark.py`.

## 5. Known bug fixed during this work

PPO always completes a full `n_steps=256` rollout, so it overshoots
`total_timesteps` (10 240 for a 10 000-step run) and probes 51 times where the
off-policy algorithms probe 50. This crashed plotting. `run_benchmark.py` now
trims every algorithm to a canonical `probe_grid()` and raises if one disagrees.

## 6. The result that reframes everything

Best achievable by a Gaussian policy of width σ (computed two independent ways —
direct quadrature and FFT convolution — agreeing to three decimals):

| σ (EUR) | best centre μ | E[r] at μ | E[r] if centred at **30** |
|---|---|---|---|
| 0 | 30.00 | 0.190 | 0.190 |
| 0.5 | 31.30 | 0.175 | **0.008** |
| 1.0 | 32.31 | 0.163 | **0.006** |
| 2.0 | 34.00 | 0.142 | 0.002 |
| 3.0 | 35.37 | 0.123 | −0.002 |
| 5.0 | 37.53 | 0.090 | −0.010 |
| 9.0 | 41.61 | 0.045 | −0.025 |

Because the optimum sits on a cliff edge, **centring a stochastic policy on it is
catastrophic** — at σ = 1 it earns 0.006 instead of 0.163, since half the samples
fall off a 0.36 cliff. The optimal centre for any σ > 0 is strictly *above* 30.

SAC with `--ent-coef 0.001` lands at **32.34**, against a constrained optimum of
**32.31 for σ ≈ 1**. It is not approximately solving this — it is sitting on the
constrained optimum for its own residual width. The "regret 0.023" reported
against the deterministic optimum measures it against a target a maximum-entropy
algorithm is forbidden to reach.

**Open task:** replace `hit`/`regret` in `summarize()` with expected reward under
the policy's actual spread.

## 7. What this means for ASSUME

MATD3 is TD3-family: deterministic actor, injected noise, and an actor update
`∇ₐQ(s, π(s))` read **at a single point**. SAC's objective is an expectation over
its own policy distribution, so its update sees a neighbourhood. That difference,
not exploration, separates them on this landscape.

It also retroactively justifies the reward-shaping block at
`assume/strategies/learning_strategies.py:1583`: replacing both flat regions with
slopes that point back at the band addresses the mechanism directly.

**Run 12 supersedes this as the practical recommendation.** The shaping works by
giving the reward a large action-correlated amplitude across the whole bid range,
which is what a critic at `act_share = 0.030` needs before it notices the action
at all — but it does so by changing the reward, and it terminates at the band's
rim (finding 14's decoy). Raising `act_share` achieves the same thing without
touching the reward, and reaches 33.0 rather than 49. Two ways to do it:

* **scale the critic's action input**, i.e. fit `Q(s, S·a)`. General, discards
  nothing, and is equivalent to rescaling the action block of `CriticTD3`'s first
  layer. The principled version is input normalisation in `CriticTD3` rather than
  a fixed S.
* **reduce `foresight`**, since `obs_dim = 3·foresight + 2`. Cheap and needs no
  library change, but it throws away forecast dimensions — free here, where they
  carry no reward information, and not in general.

The second lever is also a warning about the observation layout as a whole: any
change that adds observation dimensions lowers `act_share`, and for a centralised
critic **adding agents lowers each actor's own share** (0.030 at N = 1 to 0.017 at
N = 16), because every extra agent contributes `unique_obs_dim` observation
dimensions and one action dimension while agent *i*'s own action stays one.

**Run 13 tested that prediction on 11 agents and it holds, with two amendments.**
The measured own share is **0.016**, about half the single-agent 0.030, and the
outcome orders monotonically by it at two independent budgets. But:

* **Scaling the action input no longer generalises.** Applied to the whole action
  vector — run 12's patch as written — the own share saturates at **1/N**, 0.091
  here, for any S. Only scaling each critic's *own* action column reaches 0.2, and
  the block-share control shows the block is not what matters. Whatever the
  principled fix inside `CriticTD3` turns out to be, it has to be per-agent.
* **The effect is on rate, not feasibility.** With eleven agents even the
  untouched baseline descends given ~2700 updates. `act_share` is worth roughly a
  2.25× budget multiplier between 0.015 and 0.024, not the difference between
  learning and not learning. The single-agent framing does not survive here.

**But the TD3/DDPG results here do not transfer to MATD3 unqualified.** Two
findings changed this section:

1. **A small gradient is not the problem.** Adam steps by `lr·G/(G+eps)`, which is
   ~`lr` for any consistently-signed `G >> eps = 1e-8`. A `dQ/da` of 1e-5 still
   moves the actor at full speed. The original claim that the actors were "stuck
   because the gradient is 1e-4" was wrong.
2. **SB3's tanh actor is.** float32 `tanh(z)` rounds to exactly `1.0` for
   `z ≥ 9.011`, so autograd's `1-f²` is *identically zero* and all six actor
   tensors come back `max|grad| == 0.0`. The measured TD3 actor sat at
   `z = 10.198`. That is unrecoverable — `0/(0+eps) = 0`.
   **ASSUME's `Actor` defaults to `softsign`**
   ([`neural_network_architecture.py:131`](../../../../assume/reinforcement_learning/neural_network_architecture.py#L131)),
   whose gradient decays polynomially and does not pin until `z ≥ 2.1e7` — a
   2.3-million-fold larger margin. At `z = 10` softsign still takes 96.6% of a
   full step where tanh takes 0%.

Full derivation, thresholds and per-`z` tables:
[`actor_saturation.md`](analysis/actor_saturation.md),
regenerable with `python actor_saturation.py`.

### Is a fragmented gradient field the binding constraint? — the pooled test was wrong

A live gradient is necessary but not obviously sufficient. Two consistency
conditions look like they should both matter:

* **temporal** — the sign at the actor's action persisting across consecutive
  updates, or Adam's momentum `m̂` cancels;
* **spatial** — the sign persisting *along the path* from the actor to the
  optimum, or the actor descends only to the nearest stationary point.

The field really is fragmented. Measured on `critic_evolution.npz`, after warmup,
between bid 30 and 100:

| | median sign reversals | unbroken descent from 100 stops at | probes with a clean path to the band |
|---|---|---|---|
| TD3 tanh | 1 | 85.0 | 49 % |
| DDPG tanh | 3 | 93.0 | 20 % |
| SAC tanh | 1 | 65.0 | 47 % |
| TD3 softsign | 2 | 100.0 | 27 % |
| DDPG softsign | 2 | 100.0 | 13 % |

**Run 05's softsign TD3 nevertheless traverses from bid 98.7 to 34.7 in about 1000
steps, while scoring *worse* on every column of that table than the tanh run that
never moved.** The original reading of that — "so the spatial condition is not
binding, and Adam's momentum recovers a direction from a 67–71 %-consistent
field" — **does not follow, and run 06 shows why.**

Every number in the table above is a **median over all post-warmup probes of a
10 000-step run**, and the crossing occupies about 600 of those steps. Resolved at
25-step spacing (run 06), the run is three phases, and pooling them is what
produces the paradox:

* the actor crosses in phase 2, during which the path is clean **91–100 % of the
  time — for tanh and softsign alike**;
* the table is dominated by phase 3, i.e. the ~8000 steps *after* the outcome was
  settled, when the critic has correctly learned that the plateau is flat. **The
  gradient of a correctly-learned flat region is numerical noise**, so a fragmented
  field there is the signature of a converged critic, not of a broken one;
* and phase 3 is endogenous. After step 2000 the tanh runs place **100 %** of their
  bids on `[50, 100]` and the softsign runs place 0–52 %, so the two critics are
  not being scored on the same thing: one plateau is still being trained, the
  other has gone stale.

What survives is narrower and, for ASSUME, more useful: **a coherent path was
necessary, it existed, and it existed only briefly.** The escape is a race, not a
standing option — the actor has to move while the critic is still wrong about the
plateau. Nothing measured here shows that a genuinely fragmented field would have
been crossable, so "partial consistency is enough" is unsupported.

## 8. Caveats

- **Single-context only.** The observation is constant (§2). Nothing here has
  been checked against the varying-context variant, which is what the real
  scenario looks like.
- **Seed counts are small** — 1 seed for 01/02, 2 seeds for 03/04. The tight
  ±0.04 spreads are suggestive, not conclusive; the ±27.58 in 04 is not.
- **Run 11 is a 3-seed, 40-episode broad screen.** Its 0/3 cells reject simple
  rescues at that horizon; they are not failure-rate estimates. Configurations
  with different `gradient_steps` also receive different total update budgets.
- **Run 12 is 3 seeds per condition on one scenario.** `act-x30`'s 3/3 is the
  first solve in the archive, not a success rate, and S = 30 is one point on a
  curve that saturates somewhere above `act_share` ≈ 0.25. Its two levers are
  monkeypatches for the experiment, not a proposed API. The offline γ = 0 fits
  behind it are 5 seeds on a single frozen buffer.
- **Run 13 is 3 seeds per condition, and its spreads are wide** — `act-all-x2`
  spans 60.9–95.3, the 50-episode baseline 69.3–97.3. What is solid is that the
  ordering by own `act_share` reproduces at *two independent budgets*; the
  individual cells are not rates. Its **reward numbers are not welfare numbers**:
  eleven agents compete, fleet reward moves opposite to `diesel_0`'s in every
  condition, and no equilibrium analysis has been done. The closed-form
  `incdec_reward` landscape does **not** apply there — it was derived with the
  rest of the fleet bidding naively — so run 13 reads rewards from the run's own
  buffer and has no "solved ≥ +0.15" column.
- **`examples/outputs` is gitignored** (`.gitignore:143`), so this archive is
  untracked — the ~21 MB of `.npz` and model zips would need an exception or LFS
  to be committed. The source code is tracked, at
  `examples/inputs/2_nodes_paper_small/rl_benchmark/`.
- **Two `ReplayBuffer` defects in `assume/`, neither triggered here.**
  `add()` (`buffer.py:124`) increments `pos` *before* testing `pos + len_obs`, so
  it marks the buffer full early and resets `pos` to 0, discarding the tail;
  `sample()` then treats all `buffer_size` rows as valid, so after a wrap it can
  return unwritten zero rows as transitions. Every benchmark buffer stays far
  below the 50 000 capacity (run 13's largest is 3 450), so **no result here is
  affected** — but any longer run would train on zeros. Separately, `sample()`
  builds `next_observations` as `observations[idx + 1]` with no terminal flag, so
  roughly 1 transition in 69 bootstraps from the first observation of the next
  episode; the effect is softened because this task is a contextual bandit whose
  actions do not drive the next state. Left open deliberately: fixing them is an
  ASSUME change affecting every scenario and needs checking against the other
  examples.
- **Environment:** requires `gymnasium 1.3.0` and `stable-baselines3 2.9.0`,
  installed into the `assume` conda env for this work and *not* declared in
  `pyproject.toml`.
- A sibling `rl_benchmark - Kopie/` folder holds a duplicate snapshot plus
  `rwd.py`; it is not part of this archive.

## 9. Reproducing

### What survives in git, and what does not

This document and the code are tracked. **Nothing else here is.** Read this
before assuming a fresh clone can regenerate the archive.

| | tracked? | |
|---|---|---|
| this document, and `HANDOFF.md` | **yes** | `examples/inputs/2_nodes_paper_small/rl_benchmark/` |
| every benchmark script | **yes** | same folder, ~17 files |
| ASSUME source, incl. the commented-out shaping and the `matd3.py:618-628` debug prints | **yes** | the exact state these runs used |
| scenario CSVs, `config.yaml`, study cases | **yes** | |
| the run archive: 293 MB of `.npz` and figures | **no** | `.gitignore:143` ignores `examples/outputs` |
| **the starting replay buffer** `buffers/single_10ep_standard.npz` | **no** | `.gitignore:142` ignores `learned_strategies` |
| run 07's saved policies (`learned_strategies/<case>/last_policies/`) | **no** | same line; the folder is 532 MB in total |
| `gymnasium` and `stable-baselines3` | **no** | installed into the `assume` conda env, absent from `pyproject.toml` |

The consequential one is the **starting buffer**. It is 24 KB, it is the same file
for every run 09–12 trial, `assume_config_sweep.py` and `assume_actshare_sweep.py`
both verify its SHA256 `5f1b80b4…` before launching, and it is not in git. Without
it those runners exit at preflight. **Tracking those 24 KB is the single cheapest
thing that would make runs 09–12 re-runnable**, and it needs a narrow exception to
`.gitignore:142`. Failing that, it can be recreated — see below.

### Recreating the starting buffer

**It contains no policy.** `run_learning` only calls `update_policy()` once
`episodes_done >= episodes_collecting_initial_experience`
(`learning_role.py:328-331`), so during the collection episodes there are **zero
gradient steps**; and `EnergyLearningSingleBidRedispatchStrategy.get_actions`
returns `th.rand_like(noise) * 2 - 1` while `collect_initial_experience_mode` is
set (`learning_strategies.py:1362-1363`), so the actor network is never even
queried. The buffer is therefore uniform bids on `[-100, +100]` and the
deterministic market's response to them — nothing in it depends on network
initialisation, the learning rate, or any algorithm setting. Checked against the
archived file: its actions are indistinguishable from `Uniform(-1, 1)`
(Kolmogorov–Smirnov D = 0.051, p = 0.076, n = 620).

That makes a regenerated buffer **statistically equivalent, but not bit-identical**
— it is a fresh draw from the same distribution, with the same ~9.5 % in-band
coverage by construction. Runs 09–12 would have to be re-run against it, and the
two `BUFFER_SHA256` guards updated.

The `inc_dec_collect_buffer` study case exists for this, and its comment block
documents the episode-count arithmetic. **Two things it does not currently say,
both established by re-running it:**

- **The horizon must be 72 h, not the 24 h it inherits from `*single_case`.** The
  archived buffer is 62 transitions × exactly 10 episodes; the case as anchored
  today yields 14 × 11 = 154. Products per episode is `hours − 10` here (market
  opening offsets eat the first ten), so 24 h gives 14 and 72 h gives 62. Adding
  `end_date: 2019-01-04 00:00` to the case reproduces 62/episode exactly (verified:
  682 = 62 × 11).
- **The archived file is the state after episode 10, not 11.** 620 = 62 × 10. The
  buffer is rewritten every episode, so either interrupt after episode 10 — the
  comment's second option — or run all 11 and slice the arrays to the first 620.

```yaml
inc_dec_collect_buffer:
  <<: *single_case
  end_date: 2019-01-04 00:00        # 72 h -> 62 transitions/episode; 24 h gives 14
  learning_config:
    <<: *single_learning_config
    training_episodes: 11           # >= collecting + validation_interval
    episodes_collecting_initial_experience: 10
    validation_episodes_interval: 1
    load_replay_buffer: false
    save_replay_buffer: true
    replay_buffer_save_path: learned_strategies/buffers/single_10ep.npz
```

Do not point `trained_policies_save_path` at a folder holding results: a fresh
run rmtree's it (`common/utils.py:885`), and with no TTY the confirmation prompt
raises `EOFError` instead of defaulting.

The rest degrades gracefully: the figures regenerate from the `.npz` files, the
`.npz` files regenerate by re-running the sweeps, and the sweeps are deterministic
given a seed and a fixed thread count — with the caveat of run 08's finding 5 for
the SB3 surrogate, which is chaotic in the BLAS thread count and therefore not
reproducible across machines at all.

### Where new runs write

New runs write to `examples/outputs/2_nodes_paper_small/rl_benchmark/` by default,
so they land beside `runs/` without touching the tracked folder.

```bash
cd examples/inputs/2_nodes_paper_small/rl_benchmark

# redraw any archived run without retraining (config is read from the npz)
python run_benchmark.py --replot \
  --results ../../../outputs/2_nodes_paper_small/rl_benchmark/runs/data/03-sac-entropy/ent-coef-0.001.npz \
  --out /tmp/check.png

# the critic figure, from the archived networks (found automatically)
python critic_landscape.py

# run 11: redraw the actor comparison and the all-config critic evolution
python real_matd3/assume_config_sweep.py --phase broad --report-only \
  --critic-out ../../../outputs/2_nodes_paper_small/rl_benchmark/runs/img/11-assume-config-critic-evolution-broad.png

# run 11: one archived run's temporal gradients and final twin-critic landscape
python real_matd3/assume_run_diagnostics.py --results <run-11.npz> --out <plot.png>

# run 12: both figures and the window table, from the archive
python real_matd3/assume_actshare_film.py

# run 12: the offline gamma = 0 fits, from the frozen buffer -- no simulation
python real_matd3/assume_offline_critic.py --round conditions

# run 12: a fresh sweep. Both levers are monkeypatches applied in the child
# process, so nothing in assume/ is edited; the shaping must stay commented out
# and preflight() refuses to start otherwise.
python real_matd3/assume_actshare_sweep.py --workers 5
python real_matd3/assume_actshare_sweep.py --report-only

# run 13: the multi-agent sweep. No starting buffer exists for 11 agents, so each
# trial collects its own 5 exploration episodes and nothing is shared between
# trials. Memory, not cores, is the limit -- each trial holds ~0.85 GB, so
# --workers 4 is about right on a 16 GB machine. Leave --threads at 1 (the
# runner's default): every recorded run used one torch thread, and run 08 found
# thread count alone can flip an outcome.
python real_matd3/assume_multiagent_actshare.py                    # baseline + act-own-x15
python real_matd3/assume_multiagent_actshare.py \
    --conditions baseline act-own-x15 --seeds 1 2 --workers 3
python real_matd3/assume_multiagent_actshare.py --report-only \
    --conditions baseline act-all-x2 act-all-x15 act-own-x15

# run 13: pick S for a target act_share from any recorded run's buffer statistics
python real_matd3/assume_multiagent_actshare.py --measure <run.npz> --target 0.2

# run 13: its four figures, all from the archive
python real_matd3/assume_multiagent_grids.py    # critic grid, bid grid, summary
python real_matd3/assume_multiagent_film.py     # the pooled four-condition view
python real_matd3/assume_multiagent_window.py   # run 06's window table
```

## 10. Layout

```
runs/
├── README.md                     a stub pointing back at this document
├── 01-algorithms-best-known-settings.png   ┐
├── 02-td3-stability-sweep.png              ├ the four headline figures
├── 03-assume-matd3-films.png               │
├── 12-actshare-dose-response.png           ┘
├── data/
│   ├── 01-baseline/              results.npz  (defaults; every learner fails)
│   ├── 01b-best-known/           headline_comparison.npz  (+ critic sweeps, 401 grid)
│   ├── 02-critic/                critic_run.npz + models/{TD3,DDPG,SAC}_seed0.zip
│   ├── 03-sac-entropy/           auto | target-entropy-4 | target-entropy-8 | ent-coef-0.001
│   ├── 04-td3-policy-delay/      pd-2 | pd-8 | pd-64
│   ├── 05-softsign/              softsign.npz
│   ├── 06-window/                window_{softsign,tanh}.npz  (25-step probes)
│   ├── 08-stability/             td3_stability{,_10k}.npz
│   ├── 09-assume-films/          assume_probe_{shaped,unshaped,unshaped_clean}.npz
│   ├── 10-assume-stability/      assume_stab_{shaped,unshaped}_seed{42,1,2,3,4,5}.npz
│   ├── 11-assume-config-stability/broad/  30 configs × 3 seeds, Q1/Q2 films included
│   ├── 12-actshare/              5 conditions × 3 seeds, the act_share ladder
│   └── 13-multiagent-actshare/   6 conditions × 3 seeds, 11 learning agents
└── img/                          the detail figures, numbered by run
```

Runs 01b, 08, 09 and 12 have no `img/NN-*` entry because their figure is one of
the four **headline** figures at this folder's top level: `01-algorithms-…` is run
01b, `02-td3-stability-sweep` is run 08, `03-assume-matd3-films` is run 09, and
`12-actshare-dose-response` is run 12. Run 12 additionally has
`img/12-actshare-descent-window.png`; run 09's film is drawn by
`real_matd3/assume_film.py`.

Run 07 has no data folder — it reads ASSUME's own saved networks in place, from
`examples/inputs/2_nodes_paper_small/learned_strategies/`.

Each `.npz` holds `steps`, `greedy/<algo>` (probe timesteps × seeds — the
noise-free policy), `placed/<algo>` (every bid actually placed, one per env step)
and, except for 01, `cfg/*`. Runs recorded with `--critic-grid N` additionally
carry `critic_bids` (the action grid, in EUR/MWh) and
`critic_q/<algo>` / `critic_grad/<algo>`, shaped `(seeds, probes, N)` — the
actor's own objective and its **autograd** gradient at every probe. Runs 01b
(`N = 401`), 02 (`critic_evolution.npz`) and 05 have these. PPO and random search
never do: neither has an action-value critic to sweep, so `critic_grad/*` covers
only TD3, DDPG and SAC.

### Source files

Tracked at `examples/inputs/2_nodes_paper_small/rl_benchmark/`:

Every script runs directly from any working directory, because importing
`_layout` puts all four folders on `sys.path` and resolves archived runs:

```bash
python analysis/descent_window.py            # no arguments needed
python sweeps/run_benchmark.py --algos TD3
```

| file | role |
|---|---|
| `_layout.py` | folders, `OUT_DIR`, and `resolve()` — imported by every script |
| `surrogate/incdec_reward.py` | the closed-form landscape |
| `surrogate/incdec_env.py` | the Gymnasium environment |
| `sweeps/run_benchmark.py` | training driver, CLI, plotting |
| `sweeps/td3_stability.py` | run 08 — the configuration sweep and its figure |
| `test_rl_benchmark.py` | the three things that would fail silently: the run 13 lever, `act_share`, the per-episode transition count |
| `analysis/critic_probe.py` | reads a trained critic: `actor_objective`, autograd `critic_curve` |
| `analysis/critic_coherence.py` | `argmax_disagreement` / `argmax_range` — **the one definition runs 10–13 share**, see correction 14 |
| `analysis/critic_landscape.py` | final-critic figure |
| `analysis/critic_evolution.py` | critic gradient field over training |
| `analysis/activation_comparison.py` | tanh vs softsign |
| `analysis/actor_saturation.md` / `.py` | **the tanh/softsign derivation and its tables** |
| `analysis/descent_window.py` | run 06 — when the descent path is open, and for how long |
| `real_matd3/assume_critic_probe.py` | run 07 — reads ASSUME's *own* saved MATD3 networks |
| `real_matd3/assume_training_probe.py` | films both ASSUME critics and their action gradients over a live learning run, with no edit to `assume/` |
| `real_matd3/assume_film.py` | run 09 — draws a pair of those films |
| `real_matd3/assume_stability.py` | run 10 — the same films across seeds, both reward conditions |
| `real_matd3/assume_config_sweep.py` | run 11 — guarded shared-start buffer, parallel config sweep, summary and multiseed actor/critic plots |
| `real_matd3/assume_run_diagnostics.py` | run 11 — temporal twin-critic gradients and final landscape for any archived trial |
| `real_matd3/assume_actshare_sweep.py` | run 12 — the `act_share` ladder: forces `foresight` and/or scales the critic's action input, both as patches installed before the scenario loads |
| `real_matd3/assume_offline_critic.py` | run 12 — the γ = 0 offline fits that located `act_share`, all three rounds |
| `real_matd3/assume_actshare_film.py` | run 12 — the descent-window figure and the dose-response figure |
| `real_matd3/assume_multiagent_actshare.py` | run 13 — the multi-agent runner, the `act-own`/`act-all` patches, the multi-agent recorder, and `--measure` for choosing S |
| `real_matd3/assume_multiagent_film.py` | run 13 — the four-condition figure, seeds pooled with observations |
| `real_matd3/assume_multiagent_grids.py` | run 13 — the three per-seed views: critic grid, bid grid, summary row |
| `real_matd3/assume_multiagent_window.py` | run 13 — run 06's window statistics on the multi-agent runs |
