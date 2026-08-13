# Inc-dec reward landscape: RL algorithm benchmark — runs 01–13

> **Condensed.** The full-length version of this document (1 844 lines) is
> [`archive/RUNS_full.md`](archive/RUNS_full.md). **Do not read it by default**
> — it costs ~25 k tokens. `grep` it when a number here is not enough; see
> [`archive/README.md`](archive/README.md) for what only exists there.
>
> **New runs do not go in this file.** They go in
> [`RUNS_Continuation.md`](RUNS_Continuation.md). This one is closed at run 13.

> **This file is tracked; the data it describes is not.** Every `.npz` and figure
> lives in `../../../outputs/2_nodes_paper_small/rl_benchmark/runs/`, which is
> gitignored, so the links below only resolve on a machine that has run the
> experiments. §8 says what is and is not reproducible from git alone.

Runs 01–08 use a closed-form **surrogate** of the landscape driving SB3; runs
07–13 probe **ASSUME's own MATD3**. Run on **2026-08-06/07**; run 13 on
2026-08-10/11. Every number here was recomputed from the archived `.npz` files,
not copied from a console log.

## Headline figures

At the archive's top level; everything under `data/` and `img/` is detail.

| figure | what it answers | run |
|---|---|---|
| `01-algorithms-best-known-settings.png` | which algorithms find the optimum at their best-known settings, plus the critic field each actor was climbing | 01b |
| `02-td3-stability-sweep.png` | how stable that is — 17 configs × 8 seeds. Mostly not | 08 |
| `03-assume-matd3-films.png` | what ASSUME's own MATD3 does over a live 40-episode run, shaping on and off | 09 |
| `12-actshare-dose-response.png` | what finally fixes it, and it is not a hyperparameter | 12 |

### The short version

The landscape is 90 % flat with the optimum on a cliff edge. In the **surrogate**,
three things go wrong in order: *(1)* the critic is right before it is complete —
its first fit is "higher bids are better", true of 89 % of the action space, so
every actor runs to the ceiling; *(2)* the way back is open only briefly, ~600
steps, before the plateau is learned as genuinely flat and its gradient becomes
noise; *(3)* crossing back is unstable — a full-sized Adam step overshoots the
19-EUR band.

On **real ASSUME**, single-agent, the binding problem is different: the critic
never learns a usable shape from the true reward at all, not even at 4× budget.
Run 12's answer is that the action is **1 of 75 critic inputs** and the other 74
carry 97 % of the input variation; raising the action's share solves it 3/3 at the
same budget. Run 13 takes that to 11 agents and finds the effect real but weaker
in kind — it buys **rate**, not feasibility. `act_share` is a quantity invented
here rather than a literature one, and is under review; see `HANDOFF.md`
workstream B before building on runs 12–13.

---

## 1. The problem

`surrogate/incdec_reward.py` reproduces the measured landscape in closed form.
Pay-as-clear EOM plus pay-as-bid redispatch, rest of the fleet fixed:

```
b > 49            not dispatched               reward  0
30 <= b <= 49     dispatched, then dec'd       reward (49 - b)/100   -> peak +0.190 at b=30
b < 30            dispatched, NOT dec'd        reward -0.170         (flat)
```

`marginal_cost 66`, `eom_price 49`, `dec_threshold 30`, `volume 1000 MW`,
`max_bid_price 100`. The action is a single squashed output scaled to
`[-100, +100]` EUR/MWh.

| property | value |
|---|---|
| optimum | bid **30.00** → reward **+0.190** |
| cliff depth one tick below it | **0.360** |
| informative share of the action space | **9.5 %** (19 of 200 EUR) |
| exactly-flat share | **~90 %** (62 % loss shelf, 27 % zero plateau) |

## 2. What the networks see

The surrogate's observation is a **74-vector that never changes** within a run:
`[0:24]` residual-load forecast = 0.49, `[24:48]` price forecast = 0.49,
`[48:72]` price history = 0.49, `[72]` scaled marginal cost = 0.66, `[73]`
availability = 1.0. Layout mirrors `TorchLearningStrategy.create_observation`
(`3 · foresight + unique_obs_dim`), but carries **zero information** — `Q(s,a)` is
effectively `Q(a)`. **Everything in runs 01–08 is single-context.** The real
scenario has 548 distinct observations in a 620-transition buffer.

---

## 3. The runs

Common to the surrogate runs: `learning_rate 1e-3`, `batch_size 256`,
`gamma 0.99`, `warmup 1000`, `buffer_size 10000`, `sigma 0.1`,
`noise_schedule linear`, `device cpu`. Each run's `RunConfig` is stored inside its
own `.npz` under `cfg/*` and reloaded by `--replot`.

### 01 — baseline: which algorithms find it at all?

`python run_benchmark.py --eval-every 200 --noise-schedule linear` ·
`data/01-baseline/results.npz` · 1 seed, 10 000 steps

TD3 100.00, DDPG 100.00, SAC 60.36, PPO 100.00 — all **+0.000**. **Random search
30.02 → +0.190.** Uniform random search beats all four learners outright; every
learner ends on the zero plateau. Exploration is not the bottleneck: ~10 % of
warmup samples are in `[30, 49]` and nothing is ever evicted. *(No `cfg/*` keys —
this run predates config persistence.)*

### 01b — the same comparison at best-known settings (**headline 1**)

```bash
python sweeps/run_benchmark.py --algos TD3 DDPG SAC PPO "Random search" \
    --seeds 4 --timesteps 10000 --eval-every 200 --noise-schedule linear \
    --actor-activation softsign --ent-coef 0.001 --critic-grid 401
python sweeps/run_benchmark.py --replot --critic-seed 3 \
    --results <runs>/data/01b-best-known/headline_comparison.npz --out <runs>/01-...png
```

`data/01b-best-known/headline_comparison.npz` · 4 seeds, 10 000 steps

| algorithm | final bid | reward |
|---|---|---|
| **SAC** (`ent_coef 0.001`) | **31.35 ± 0.31** | **+0.177 ± 0.003** |
| **DDPG** (softsign) | **31.86 ± 0.51** | **+0.171 ± 0.005** |
| TD3 (softsign) | 1.59 ± 52.26 | +0.087 ± 0.148 |
| PPO | 100.00 ± 0.00 | +0.000 |
| Random search | 30.02 ± 0.02 | +0.190 |

Two of four now solve it tightly. **TD3 is the unstable one**: finals
`31.86, 31.60, 31.83, −88.93` — three converge, one falls through onto the loss
shelf. **DDPG is more reliable than TD3 here**, the opposite of the usual ordering.

**The critic row explains that.** Across all 12 runs the three phases are
identical in timing for every algorithm: `dQ/d(bid)` positive across 100 % of the
`[49, 100]` plateau at step 1200 in 10/12 runs; the plateau slope flips at step
**1400 in 11/12** (12/12 at the actor's own action); after step 5000 the field is
correct where it matters (85–94 % of cells in `[30, 49]` carry the true negative
slope, median `|dQ/d(bid)| ≈ 9.5e-3`) and a coin flip on the flat regions
(~1.5e-4). First probe back inside the band: **SAC 1400–1800, DDPG 1600–1800,
TD3 2000** (never, for the failing seed). That is `policy_delay = 2` — TD3 needs
the same *actor* updates and twice the *environment* steps, against a window
counted in environment steps. Untested: TD3 at `--policy-delay 1` should match DDPG.
*Random search is the bar to clear, not a competitor* — an oracle with perfect
recall and no generalisation.

### 02 — what did the critics actually learn?

`--save-models` then `python analysis/critic_landscape.py` ·
`data/02-critic/` (+ `models/`)

| | critic argmax | actor sits at | `dQ/da` at the actor |
|---|---|---|---|
| TD3 | **34.2** | 100.0 | −4.4e-5 |
| DDPG | **37.6** | 100.0 | +6.3e-6 |
| SAC | **32.7** | 60.4 | −7.7e-5 |

**The critics are right** — all three place their maximum inside the band, and
`dQ/da` matches the true reward's shape. The actors are stuck at the ceiling where
the field is 3–4 orders below the signal at the cliff; the original reading of
that ("too small to move") is **wrong**, correction 4. Sweeps use each actor's
*own* objective (TD3/DDPG climb Q1 alone, SAC the min) and `torch.autograd`,
correction 7.

### 03 — SAC entropy

`--ent-coef / --target-entropy` · `data/03-sac-entropy/` · 2 seeds, 8000 steps

| setting | final bid | reward |
|---|---|---|
| `auto` (target entropy −1) | 51.67 ± 14.55 | +0.059 |
| `--target-entropy -4` | 33.50 ± 0.04 | +0.155 |
| `--target-entropy -8` | 36.64 ± 1.69 | +0.124 |
| **`--ent-coef 0.001`** | **32.34 ± 0.59** | **+0.167** |

Default `target_entropy = -1` forces σ ≈ ±9 EUR, as wide as the whole band, so SAC
is structurally forbidden from converging onto it. The response is
**non-monotonic** — −8 is worse than −4: too high a floor and it never converges,
too low and it collapses early into TD3's failure mode.

### 04 — TD3 `policy_delay`: a hypothesis that failed

`data/04-td3-policy-delay/` · 2 seeds. `policy_delay` 2 → 100.00, 8 → 100.00,
64 → 68.46 ± 27.58. **Not supported**; slowing the actor 32× leaves it on the
plateau with large seed variance. Kept because a refuted hypothesis is worth not
re-testing — but see 01b: the knob matters, run 04 turned it the **wrong way**.

### 05 — softsign actor: the one-line fix

`--actor-activation softsign` (replaces `net.mu[-1]` and its Polyak target only) ·
`data/05-softsign/softsign.npz` · 2 seeds, 10 000 steps

| actor output | TD3 | DDPG |
|---|---|---|
| `tanh` (SB3 default) | 100.00 → **+0.000** | 100.00 → **+0.000** |
| `softsign` (ASSUME's default) | **31.73 ± 0.13 → +0.173** | **31.51 ± 0.52 → +0.175** |

Complete reversal, both seeds. The mechanism is *not* that softsign avoids the
runaway — both actors bolt to the ceiling the instant warmup ends (step 1200:
tanh 100.00, softsign 98.68). They differ only in whether they can **return**: at
100.00 tanh gives `1 − a² = 0` **exactly** and is frozen for the remaining 8800
steps, while softsign at 98.68 still holds 2.1e-2 of headroom.

### 06 — the escape window: *when* was the descent available?

3 seeds × 2 algorithms × 2 activations, 4000 steps, probed **every 25 steps** on a
401-point grid · `data/06-window/` · `python analysis/descent_window.py`

| phase | steps | mean `dQ/d(bid)` on [50,100] | unbroken path to the band |
|---|---|---|---|
| warmup → flip | 1000–1250 | **+1.1e-3**, pulls right | 0 % |
| flip → crossed | 1250–1900 | −6.4e-4, pulls left | **91–100 %** |
| after crossing | 1900–4000 | −3.7e-4 | 27–72 %, 2 sign reversals |

The flip happens at step **1225–1325 in every one of the twelve runs**. Crossings:
TD3 softsign 1575–2125, DDPG softsign 1450–1950, **both tanh conditions never** —
so both activations see the same clean field at the same time, and the tanh runs
are handed a descent path on schedule and cannot take a single step along it.
Phase 3 is not evidence about crossability and is **endogenous**: after step 2000
the tanh runs place **100 %** of their bids on `[50, 100]` and the softsign runs
0–52 %, so one plateau is still being trained and the other has gone stale
(correction 6).

### 07 — ASSUME's own MATD3, from saved networks

`python real_matd3/assume_critic_probe.py --n-obs 24` · reads
`learned_strategies/<case>/last_policies/` in place; no data folder

**The budget is in the saved optimizer state:** every study case has
`critic_optimizer step = 640` and `actor_optimizer step = 80`, against the 190–410
actor updates a crossing costs in the surrogate.

| case | reward | critic argmax | actor bids | `dQ/da` at the actor |
|---|---|---|---|---|
| `inc_dec_learning_single` | true | **100.0** (4 % in band) | 95.2 | **+1.0e-3** (pulls right) |
| `..._single_g0` | shaped | 62.8 (12 %) | 52.9 | +4.3e-3 |
| `..._single_g0_2` | shaped | 51.0 (29 %) | 50.7 | +6.5e-4 |

**The unshaped run is frozen in phase 1** — not a converged critic the actor
failed to follow; the critic never reached the flip. **The shaped runs converged
onto a decoy**: the shaping fires only when `reward <= 0`, so it does not apply
inside the band and re-enters at full height at bid 49 (true reward there +0.000,
shaped +0.170). Two local maxima separated by the same cliff. It does what it was
built for — walks the actor down from +100 — and stops one euro above the band.

### 08 — how stable is run 05? Not at all. (**headline 2**)

`python sweeps/td3_stability.py --seeds 8 --workers 15` · `data/08-stability/` ·
8 seeds each, 6000 steps except the `-10k` configs

| config | solved | final bid | | config | solved | final bid |
|---|---|---|---|---|---|---|
| baseline | **3/8** | −3.67 ± 55.73 | | **lr-1e-4** | **6/8** | **33.30 ± 0.69** |
| tanh | 0/8 | 100.00 | | batch-128 | 4/8 | 33.89 ± 1.13 |
| policy-delay-8 | **0/8** | −9.44 ± 64.97 | | sigma-0.05 | 4/8 | 30.89 ± 29.55 |
| policy-delay-64 | 3/8 | 42.56 ± 14.02 | | sigma-0.3 | 1/8 | 2.59 ± 80.46 |
| warmup-250 | 3/8 | 26.11 ± 53.78 | | lr-3e-3 | 5/8 | −9.28 ± 56.11 |
| warmup-3000 | 2/8 | 18.70 ± 57.07 | | buffer-2000 | 3/8 | 56.62 ± 30.60 |
| noise-const | 3/8 | −25.03 ± 58.40 | | **assume-knobs** | **0/8** | 58.27 ± 49.31 |
| | | | | **assume-budget** | **0/8** | 93.11 ± 2.83 |

At run 05's own length: `run05-repro` **2/8**, `lr-1e-4-10k` **8/8** (32.10 ± 0.37).

**Softsign replaces a deterministic failure with a chaotic one** — the new failure
mode is overshooting straight through the band to −100. **BLAS thread count
decides it**: the archived run 05 reproduces bit-identically at 14 threads
(31.86 / 31.60), and the same command at **1 thread** gives 35.44 / **−60.49**.
Read any single-configuration surrogate result as a sample from a bimodal
distribution. *(Not true of ASSUME — runs 10 and 13.)*

What stabilises it is **step size, not window timing**, so run 06's prediction is
half wrong: `policy-delay-8` fails as predicted, but `lr-1e-4` is the *best* config
and `warmup-3000` is no worse than baseline. The two tight configs are the two
that shrink or smooth the actor's step. **For ASSUME:** `assume-knobs`
(`policy_delay 8`, 32 gradient steps per 12 env steps) is 0/8 even at a full
6000-step budget, so its difficulty is not only the short budget — the update
ratio sits in the bad region independently.

### 09 — ASSUME's MATD3, filmed over a live run (**headline 3**)

```bash
python real_matd3/assume_training_probe.py --study-case inc_dec_learning_single_g0 \
    --episodes 40 --n-obs 6 --grid 401 --label shaped
# then comment out learning_strategies.py:1583-1589 and repeat with
#   --label unshaped --load-buffer buffers/single_10ep_standard.npz
python real_matd3/assume_film.py
```

`data/09-assume-films/` · 40 episodes = **2560 critic updates**, 4× the default
budget, one frame per training block (32 updates), 6 real observations

| film | argmax `Q1` first → last | actor first → last |
|---|---|---|
| shaped | 100.0 → **48.8** | 62.9 → **49.0** |
| unshaped | 100.0 → **100.0** | 61.3 → **94.4** |

**Phase 1 is confirmed on the real thing** — both open at the ceiling. **Without
shaping the critic never leaves it, not in 2560 updates:** the field is mottled
noise across the whole bid axis with no coherent sign structure. So this is not
run 07's budget; it is a different failure. **With shaping the same run converges
cleanly to the band's rim** within ~800 updates.

`assume_training_probe.py` wraps `matd3.TD3.update_policy` from the outside, so
**nothing in `assume/` is edited**. `assume_probe_unshaped.npz` is
**mislabelled** — it preloads a buffer whose 280 stored rewards are *shaped*. Use
`assume_probe_unshaped_clean.npz`; the mislabelled run appears to flip, the clean
one never does.

### 10 — are those two films reproducible? 6 seeds each

```bash
python real_matd3/assume_stability.py --condition unshaped --seeds 42 1 2 3 4 5
# uncomment learning_strategies.py:1583-1589, repeat with --condition shaped, put it back
python real_matd3/assume_stability.py --report
```

`data/10-assume-stability/` · **6 seeds per condition**, 40 episodes, 29 + 17 min
wall at 6 workers, one torch thread each

**How the seed was varied:** `loader_csv.py:555` calls
`set_random_seed(config.get("seed", 42))` once while the scenario is read and
nothing re-seeds after. `--seed` re-applies that call after the load; the CSVs and
forecaster contain no RNG draws, so what varies is network init, exploration noise
and batch draws — the learner varies and the environment is fixed.

| condition | argmax `Q1` last | disagreement | range | actor last | reaches the band at any frame |
|---|---|---|---|---|---|
| shaped | 49.5 ± 4.4 | 4.2 | 10.4 | **50.6 ± 3.0** | **6/6** |
| unshaped | 89.7 ± 10.6 | **24.5** | 56.4 | 94.3 ± 7.0 | **1/6** |

**The conclusion reproduces; the number does not.** Run 09's "argmax stays at
100.0" is wrong — only 2/6 seeds end there, and read on its own that median is
close to meaningless: the six probed observations of a single unshaped run
disagree about the preferred bid by **24.5 EUR between an average pair**. **Quote
the disagreement, not the range** — two statistics, once compared with each other
(correction 14), now both in `analysis/critic_coherence.py`. Say *the unshaped
critic never forms a coherent preference*, not that it prefers the ceiling.

Final actor bids do not overlap at all: shaped 44.9–54.1, unshaped 79.3–99.2. Five
of six unshaped runs never place a probed bid in the band; the sixth grazes it (13
of 480 cells) and still ends at 94.4. All six shaped seeds *do* dip into the band
transiently (best frame +0.065 at bid ≈ 42.5) — they just do not stay; the 1/6
ending inside at 44.9 is one seed, not a rate.

**ASSUME is not thread-chaotic the way the surrogate is:** unshaped seed 42
reproduces run 09's film to **0.119 EUR** over all 80 frames, at one torch thread
where run 09 ran at the default.

### 11 — ASSUME configuration stability on the true reward

`python real_matd3/assume_config_sweep.py --phase broad --workers 6` ·
`data/11-assume-config-stability/broad/` (90 files + `summary-complete.txt`) ·
**30 configs × 3 seeds** (`42, 1, 2`), 40 episodes, 80 frames each · 243.5 min
wall + a 14.6 min retry

Baseline: `lr 1e-3`, `gradient_steps 10`, `batch 128`, `policy_delay 2`,
`gamma .99`, `tau .005`, `sigma .1`, target noise `.2` clipped at `.5`,
`train_freq 12h` = **800 critic updates**. Swept one knob at a time: learning rate
(3e-4/1e-4/3e-5), batch (64/256/512), policy delay (1/4/8), gradient steps
(4/20/32), gamma/tau, noise, plus six interactions. Every trial loads the same
immutable, checksum-guarded true-reward buffer and then owns its own; the shaping
was commented out throughout.

**Result: 0/3 solved for all 30 configurations.**

| configuration | final actor bid | recon reward | reading |
|---|---:|---:|---|
| baseline | 99.4 ± 0.1 | +0.000 | phase 1 persists |
| `lr-1e-4` | 95.5 ± 0.3 | +0.000 | the surrogate's stable LR does not transfer |
| `policy-delay-1` | 99.8 ± 0.1 | +0.000 | twice the actor-update rate is not enough |
| **`grad-32`** | **78.7 ± 16.1** (60.7–99.8) | +0.002 ± 0.003 | only material descent; unstable, unsolved |
| `lr-1e-4-grad-32` | 98.2 | +0.000 | low LR cancels that movement |

The strongest lever is **critic-update budget**, not any stabiliser: `grad-32` does
2560 updates and is the only config that forms a substantial descending region.
All **90/90 archives** carry Q1/Q2 and both autograd gradient fields on 401 bids ×
6 observations at all 80 blocks. Six initial failures were a concurrent
TensorBoard async-writer race, **not** failed learning; retried with
`--disable-tensorboard`, partials kept under `partial_failures_before_retry/`.
3 seeds and 40 episodes reject easy fixes; they are not rates, and changing
`gradient_steps` also changes total optimisation work.

### 12 — the action's share of the critic's input (**headline 4**)

#### The offline experiment that located it

γ = 0 on the frozen buffer — no bootstrap, no moving target, no actor, no growing
buffer. ASSUME's own `CriticTD3`, AdamW lr 1e-3, batch 128, 5 seeds, 2560 updates.
`python real_matd3/assume_offline_critic.py` (~15 min, no simulation).

| condition | argmax Q1 | band_neg | train MSE | test MSE |
|---|---|---:|---:|---:|
| `full-obs` — what ASSUME does | **95.3 ± 4.8** | **0.04** | 0.00012 | 0.00486 |
| `const-obs` — the surrogate's setting | 32.9 ± 1.9 | 0.84 | 0.00151 | 0.00205 |
| `shuffled-obs` | 98.8 ± 1.8 | 0.02 | 0.00011 | 0.00804 |
| `obs-x0.1` | 34.0 ± 1.2 | 0.78 | 0.00000 | 0.00109 |

`band_neg` is the share of cells in `[30, 49]` carrying the true negative slope;
0.50 is a coin flip. **The live failure reproduces in a plain supervised fit**, so
it is not the bootstrap, the actor or the budget — which is what makes this harness
the right place to screen critic architectures. Two intuitive explanations die
here: `shuffled-obs`, whose observations have no association with the reward at
all, fails *identically*; and `obs-x0.1` drives train MSE to zero — memorising
harder — while learning the band correctly. *(`shuffled-obs` is the one row that is
not seed-stable between scripts, since the permutation consumes RNG — quote it as
"fails like `full-obs`", not by its digits.)*

Three mechanically unrelated levers — rescaling the observation, rescaling the
action, deleting observation dimensions — all track

```
act_share = sd(a) / (sd(a) + sum_j sd(obs_j))       = 0.030 as ASSUME is configured
var_share = sd(a)^2 / (sd(a)^2 + sum_j sd(obs_j)^2) = 0.055
```

(`sd(a) = 0.592`, `Σⱼ sd(obsⱼ) = 19.3`. `var_share` is the one with a mechanism —
Xavier makes the first-layer weights iid across inputs, so contributions to the
pre-activation variance add in quadrature. Both order every sweep identically.)
Selected rungs: `act ×2` 0.058 → 65.2, `act ×10` 0.234 → 38.4, `act ×30` 0.479 →
33.7; `foresight 12` 0.057 → 66.4, `foresight 3` 0.191 → 33.2, `foresight 1` 0.356
→ **31.1 ± 0.9**. Note the direction: **fewer observation dimensions raises
`act_share`**, and **z-scoring the observation is the worst cell in the table**
(0.008, pinned at exactly 100.0 in 5/5 seeds). `act_share` predicts the *ordering*
across levers, not the exact value.

> **That last cell is in direct tension with the literature and is the next
> thing to test.** SimBa (Lee et al., ICLR 2025) makes per-dimension observation
> standardization by running statistics its single most important component, and
> DDPG (Lillicrap et al., 2016) batch-normalizes the state input for the same
> reason. Both cannot be right. Either `act_share` is measuring the wrong thing, or
> normalization is only safe alongside the residual path and LayerNorms SimBa pairs
> it with. `HANDOFF.md` workstream B runs it on this harness.

#### The live run

```bash
python real_matd3/assume_actshare_sweep.py --workers 5
python real_matd3/assume_actshare_film.py          # both figures, from the archive
```

`data/12-actshare/` · **5 conditions × 3 seeds**, 40 episodes, true reward, run
11's `BASELINE`, same checksum-guarded buffer, ~15 min per trial

Neither lever edits `assume/`. `foresight` goes through the strategy's own kwarg
(`obs_dim = 3·foresight + 2`); the action lever patches `CriticTD3` to fit
`Q(s, S·a)`, leaving the actor, the bid mapping and the environment untouched
(Adam is scale-invariant, so the actor's *step size* is unchanged). A
reduced-foresight run cannot load the 74-dim buffer, so a truncated copy is
derived — first k of each forecast block, **last** k of the price history, because
`create_observation` builds that one with `direction="backward"`.

| condition | act_share | obs_dim | final bid | measured reward | recon | argmax Q1 | disagree | band_neg | solved |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|
| `baseline` | 0.030 | 74 | 99.4 ± 0.1 | +0.000 | +0.000 | 85.5 ± 11.2 | 21.7 | 0.11 | 0/3 |
| `foresight-6` | 0.108 | 20 | 63.9 ± 24.2 | +0.041 ± 0.058 | +0.012 | 53.3 ± 13.2 | 22.5 | 0.38 | 0/3 |
| `foresight-3` | 0.191 | 11 | 40.4 ± 2.5 | +0.088 ± 0.015 | +0.087 | 37.5 ± 1.8 | 1.8 | 0.61 | 0/3 |
| `act-x10` | 0.234 | 74 | −7.4 ± 61.0 | +0.025 ± 0.159 | +0.033 | 35.2 ± 2.2 | 8.8 | 0.72 | 0/3 |
| **`act-x30`** | 0.479 | 74 | **33.0 ± 0.2** | **+0.167 ± 0.005** | +0.160 | 33.2 ± 1.1 | 1.8 | 0.81 | **3/3** |

**`baseline` reproduces run 11's own cell** (99.4 ± 0.1, 0/3, per-seed argmax
100.0 / 83.8 / 72.8), so the rest of the table is the levers and not the harness.
**`act-x30` is the first configuration in this archive to solve the true-reward
task**, against 0/90 in run 11; all 18 probed bids lie in 32.0–34.8. It lands on
§5's constrained optimum for σ ≈ 1, not on 30, and should.

**Run 10's incoherence statistic collapses** — 24.5 (run 10 unshaped) and 21.7
(`baseline`, i.e. the same failure) fall to **1.8**. **`act-x10` is bimodal**:
two seeds converge, seed 42 overshoots to −93.7 — run 08's overshoot failure
arriving in ASSUME for the first time, now that the critic works.

**`recon` is reconstructed from the surrogate curve and is not the simulator's
reward** (correction 15); the `measured` column is read from each trial's own
`rl_params`, and **the headline survives on it**. That column is itself an
early-hours sample — `rl_params` holds only the first two products of each
episode (correction 16).

#### The window, and how it differs from run 06's

Measured with run 06's own definitions (`img/12-actshare-descent-window.png`):

| condition | act_share | peak bid | first clean | clean share | open at end | pulled left | settles |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 0.030 | 99.5 | 585 | 2 % | 11 % | 56 % | 0/18 |
| `foresight-6` | 0.108 | 96.4 | 300 | 21 % | 33 % | 39 % | 5/18 |
| `foresight-3` | 0.191 | 94.7 | 455 | 27 % | **89 %** | **100 %** | 18/18 |
| `act-x10` | 0.234 | 74.5 | **20** | 27 % | 22 % | 44 % | 12/18 |
| `act-x30` | 0.479 | 74.8 | **20** | 37 % | 33 % | 50 % | 18/18 |

Three differences from run 06: **it is not transient** (clean share *rises* with
`act_share`); **at high `act_share` it is there from the start** — first clean path
at update 20, the second frame, rather than 585, with phase 1 shortened rather
than skipped (peak bid 99.5 → ~74.5); **the crossing costs 180–580 updates**,
inside run 07's 190–410, because it no longer has to be paid inside a closing
window. `pulled left` separates converged from still-moving: `act-x30` 50 % (a
coin flip), `foresight-3` **100 %** — still descending at bid 40.4 when the budget
ran out.

3 seeds, one scenario. `act-x30`'s 3/3 rejects "nothing simple works"; it is not
a rate. S = 30 is one arbitrary point. Cutting foresight discards forecast
information that is worthless *in this scenario* and would not be in general.
Recorded `dQ/d(bid)` carries the factor S, so **signs are comparable across
conditions and magnitudes are not**. The `matd3.py:618-628` debug prints were live
for all 15 trials, but they affect every condition equally and `baseline`
reproduced run 11.

### 13 — eleven learning agents: does run 12 transfer?

```bash
python real_matd3/assume_multiagent_actshare.py                     # the two 50-episode runs
python real_matd3/assume_multiagent_actshare.py --report-only
python real_matd3/assume_multiagent_grids.py    # three per-seed figures
python real_matd3/assume_multiagent_window.py   # the descent-window table
python real_matd3/assume_multiagent_actshare.py --measure <run.npz> --target 0.2
```

`data/13-multiagent-actshare/` · **6 conditions × 3 seeds**, true reward, all 11
units of `powerplant_units_learning.csv` learning on a 72 h horizon, so each
critic sees **94 observation + 11 action = 105 inputs**. Only early stopping is
disabled. ~31 min per 25-episode trial, ~59 min per 50-episode trial.

**Run 13 used a working-tree `inc_dec_learning`, not the committed one:**

| field | committed at `9cf080eb` | used by run 13 |
|---|---|---|
| `end_date` | 2019-01-01 **05:00** | 2019-01-**04** 00:00 (72 h) |
| `learning_rate` | 0.001 | **0.0001** |
| `training_episodes` | 8 | 50 |
| `episodes_collecting_initial_experience` | 2 | 5 |
| `train_freq` | **1h** | **12h** |
| `validation_episodes_interval` | 2 | 5 |

The horizon drives everything — 72 h at 12h gives 6 blocks/episode, hence 270
frames and 2700 updates — and the committed `train_freq: 1h` is the setting that
dies with `No rewards were collected`. **Restore this table before re-running.**

**The lever had to change, and that is itself a result.** Run 12's patch scales the
critic's *whole* action input; with N agents that raises everyone together and
**caps each agent's own share at 1/N** (0.091 here) for any S. `act-own` scales
only critic *i*'s own action column — symmetric across agents, the input
`matd3.py:711` differentiates for the actor loss, and reduces to run 12's patch at
N = 1. The two move own share and block share in *opposite* proportions, which
separates them; `act-all-x2` is the block-matched control.

| condition | updates | own share | `diesel_0` final bid, per seed | mean ± sd | reward | fleet |
|---|---:|---:|---|---:|---:|---:|
| `baseline-25` | 1200 | 0.015 | 97.2, 96.8, 97.3 | **97.1 ± 0.2** | +0.018 | +6.26 |
| `act-all-x2` | 1200 | 0.024 | 60.9, 95.3, 81.6 | 79.2 ± 14.1 | +0.232 | +5.92 |
| `act-all-x15` | 1200 | 0.065 | 49.1, 44.8, 47.3 | **47.1 ± 1.7** | +0.526 | +5.01 |
| `baseline` | 2700 | 0.013 | 73.8, 69.3, 97.3 | 80.1 ± 12.3 | +0.232 | +5.93 |
| `act-all-x2-50` | 2700 | 0.023 | 45.6, 47.6, 62.5 | 51.9 ± 7.5 | +0.378 | +4.20 |
| `act-own-x15` | 2700 | 0.137 | 39.2, 25.7, 22.1 | **29.0 ± 7.4** | +0.493 | +5.00 |

**Monotone in the own share at each budget separately**, and mean reward follows.
**The action-block share does not predict**: `act-all-x15` carries 3.2×
`act-own-x15`'s block share (0.728 vs 0.230) and ends 18 EUR further out.

**But it buys rate, not feasibility.** `act-all-x2` reaches 79.2 in 1200 updates;
**the baseline reaches 80.1 in 2700** — the same place for ~2.25× the budget, with
a coherent leftward field forming from ~2000 updates. Runs 09–12's "the critic
never forms a preference" does not transfer; every condition descends eventually
and `act_share` sets how fast.

**The multi-agent baseline is not the single-agent failure mode either.**
`diesel_0` parks at 97.1 with `argmax Q1` exactly 100.0, which reproduces — but
the northern units go to the **floor** (≈ −95) and come back off it, and the
critic's preference is coherent and bang-bang, not incoherent. **Run 10's
incoherence statistic inverts**: the best condition has the *highest* disagreement
(47.7 EUR) and the failing baselines the lowest (13.7, 21.2). With eleven agents
the preferred bid genuinely *should* depend on the observation.

**Two reproducibility results, free.** Both short conditions are **bit-identical
prefixes of their long counterparts** on `greedy`, `critic_q`, `critic_grad`,
`rewards` and `steps`, 3/3 seeds each. So nothing depends on `training_episodes`,
the runs are deterministic given seed and thread count, and the budget-doubling
comparison is **the same trajectories continued**, not a fresh sample.

**Fleet reward runs opposite to `diesel_0`'s in every row** — agents compete, so
falling bids mean falling prices. **"Solved" cannot be defined by fleet profit
here**, and the closed-form `incdec_reward` landscape does not apply at all (it was
derived with the rest of the fleet bidding naively), which is why run 13 reads
rewards from each run's own buffer. Three seeds, wide spreads: `act-all-x2`
spans 60.9–95.3. The ordering reproducing at two independent budgets is the
strongest thing on offer at n = 3.

---

## 4. Corrections

Recorded so they are not silently reintroduced. Full evidence for each is in
[`archive/RUNS_full.md`](archive/RUNS_full.md) §4.

1. "The critic smooths the cliff into a ramp." Wrong — run 02, the critics learn it.
2. "Slowing the actor (`policy_delay`) fixes it." Wrong — run 04. **See 9.**
3. `summarize()`'s `hit (<0.5 EUR)` and `regret` columns are a bad metric (§5).
   Still in the code; treat with suspicion.
4. "The gradient is 1e-4, so the actor is stuck." Wrong, and the wrong *kind* of
   wrong — it confuses magnitude with direction. Adam steps by `lr·G/(G+eps)`; the
   actual cause is tanh pinning to exactly 1.0 in float32.
5. "The fragmented gradient field is the binding constraint." **Superseded by 6.**
6. "Softsign scores worse on every path metric and crosses anyway, so spatial
   consistency is not binding." Observation real, inference not — **the evidence
   was pooled over the wrong interval**. During the crossing the path is clean
   91–100 % of the time for *both* activations; the pooled statistic is dominated
   by the endogenous post-crossing phase. Right statement: a clean path was
   necessary and present; only softsign could act on it.
7. The critic sweep used `min(Q1, Q2)` for every algorithm — only SAC's actor
   climbs the min. Gradients were also finite differences, now `torch.autograd`.
   Only the third digit moved.
8. "TD3 loses two seeds into the band and two through it" (01b). Miscounted —
   **three in, one through**. The mean and spread were always right.
9. "`policy_delay` does nothing here." Half wrong: raising it does not help, but at
   `policy_delay 2` TD3 **already** crosses 200–600 env steps later than DDPG and
   SAC. The knob matters; run 04 turned it the wrong way.
10. "Without shaping, `argmax Q1` stays at 100.0" (09). Conclusion survives 6
    seeds, number does not — 89.7 ± 10.6, only 2/6 at 100.0.
11. "The real critic fails because it must fit `Q(s,a)` across 548 observations."
    Wrong in that form — run 12's controls: shuffled observations fail
    identically, harder memorisation learns the band. The observation *count* stood
    in for the input *dimension* count.
12. "Without shaping or a raised `act_share`, MATD3 never forms a usable
    preference." True single-agent at the budgets tested, **false as stated for the
    multi-agent case**. State a budget with any "cannot learn" claim.
13. "The `argmax Q1` disagreement measures critic incoherence." Only where the
    observation carries no reward information. It **inverts** at N = 11.
14. "Run 10's critic disagrees by 56.4 and run 12's `baseline` by 18.1, so the
    baseline is more coherent." **Two different statistics** — a *range* against a
    *mean pairwise* number additionally divided by `n²`. Matched, the two are
    indistinguishable: **24.5 vs 21.7** as a disagreement, 56.4 vs 45.3 as a range.
    Both now in `analysis/critic_coherence.py`. No conclusion changed.
15. "`reward_from_bid` is the scenario's reward, so a recorded bid can be scored
    with it." It is not — it agrees on **24.8 %** of the frozen buffer's 620
    transitions (MAE 0.038, R² 0.78). The mismatch is structural: the real EOM
    price varies hour to hour, so the loss shelf is **three** values
    (−0.20/−0.25/−0.30) against one, and `diesel_0` carries `additional_cost 68`,
    not 66. **Bids, critic fields, `act_share` and the offline fits stand**; the
    `+0.15` bar, `regret`, the `32.31` optimum and the exact `[30, 49]` band were
    never established against the simulator. §3's run 10/11 reward columns are
    relabelled `recon`, not corrected. Do **not** retune the surrogate — it is
    exact for runs 01–08 by construction. Pinned by `test_rl_benchmark.py`.
16. "The evaluation reward summarises the episode." Every inspected database holds
    exactly **two rows per episode** (10:00 and 11:00 of 14): `get_sum_reward()` is
    called immediately after `world.run()` and the writes are scheduled as instant
    messages, so the tail is not flushed. Consequences for
    `compare_and_save_policies`, early stopping and any measured reward.
    **Training is unaffected** — it reads the cache, not the database.
17. "Run 13's recorded critic field is the objective the actor differentiates." A
    valid slice of the same critic, but not that objective — `matd3.py:704` holds
    the other agents at **stored** actions, the recorder at their **current
    actors'** outputs. Empty at N = 1. Pinned by `test_rl_benchmark.py`.

**Known bug fixed during this work:** PPO always completes a full `n_steps=256`
rollout, so it overshoots `total_timesteps` and probes 51 times where the
off-policy algorithms probe 50. `run_benchmark.py` now trims every algorithm to a
canonical `probe_grid()` and raises if one disagrees.

## 5. The result that reframes everything

Best achievable by a Gaussian policy of width σ (two independent computations,
agreeing to three decimals):

| σ (EUR) | best centre μ | E[r] at μ | E[r] if centred at **30** |
|---|---|---|---|
| 0 | 30.00 | 0.190 | 0.190 |
| 0.5 | 31.30 | 0.175 | **0.008** |
| 1.0 | 32.31 | 0.163 | **0.006** |
| 2.0 | 34.00 | 0.142 | 0.002 |
| 5.0 | 37.53 | 0.090 | −0.010 |
| 9.0 | 41.61 | 0.045 | −0.025 |

Because the optimum sits on a cliff edge, **centring a stochastic policy on it is
catastrophic** — half the samples fall off a 0.36 cliff — and the optimal centre
for any σ > 0 is strictly *above* 30. SAC at `--ent-coef 0.001` lands at 32.34
against a constrained optimum of 32.31 for σ ≈ 1: it is not approximately solving
this, it is *sitting on* the constrained optimum for its own residual width.

Surrogate numbers, never validated against the simulator — correction 15.
**Open task:** replace `hit`/`regret` in `summarize()` with expected reward under
the policy's actual spread.

## 6. What this means for ASSUME

MATD3 is TD3-family: deterministic actor, injected noise, and an actor update
`∇ₐQ(s, π(s))` read **at a single point**, where SAC's objective is an expectation
over its own policy distribution and so sees a neighbourhood. That difference, not
exploration, separates them here. And **a small gradient is not the problem**
(correction 4) — **SB3's tanh actor is**, which ASSUME does not share: its `Actor`
defaults to softsign (`neural_network_architecture.py:131`), which does not pin
until `z ≥ 2.1e7` against tanh's 9.011. Derivation:
[`analysis/actor_saturation.md`](analysis/actor_saturation.md).

**What replaces the shaping is an open question, not run 12's answer.** The
shaping works by giving the reward a large action-correlated amplitude across the
whole bid range, but it changes the reward and stops at the band's rim. Run 12
reaches 33 rather than 49 without touching the reward — but by monkeypatching
`CriticTD3` or deleting forecast dimensions, neither of which is an API. What its
two levers do establish is a warning about the **input layout**: any change that
adds observation dimensions dilutes the action, and for a centralised critic
**adding agents dilutes each actor's own action** (0.030 at N = 1, 0.016 at
N = 11), with uniform action scaling saturating at 1/N so that any fix must be
**per-agent**. `HANDOFF.md` workstream B tests the two standard architectures that
address the same thing — late action injection (DDPG) and SimBa — against this.

## 7. Caveats

- **Runs 01–08 are single-context** (§2) and have not been checked against the
  varying-context variant, which is what the real scenario looks like.
  `IncDecEnv(params=[...])` takes a context list.
- **Seed counts are small** — 1 for 01/02, 2 for 03/04/05, 3 for 11/12/13, 6 for
  10. They reject easy rescues at a given horizon; they are not failure rates.
  Run 11's `gradient_steps` configs also receive different total update budgets.
- **Run 13's reward numbers are not welfare numbers** — eleven agents compete,
  fleet reward moves opposite to `diesel_0`'s in every condition, and no
  equilibrium analysis has been done.
- **Two `ReplayBuffer` defects in `assume/`, neither triggered here.** `add()`
  (`buffer.py:124`) increments `pos` before testing `pos + len_obs`, marking the
  buffer full early, after which `sample()` can return unwritten zero rows; and
  `sample()` builds `next_observations` as `observations[idx + 1]` with no terminal
  flag, so ~1 transition in 69 bootstraps across an episode boundary. Every buffer
  here stays far below the 50 000 capacity (largest 3 450), so **no result is
  affected** — but any longer run would train on zeros.
- **Environment:** `gymnasium 1.3.0`, `stable-baselines3 2.9.0`, installed into the
  `assume` conda env and *not* declared in `pyproject.toml`.
- A sibling `rl_benchmark - Kopie/` folder holds a duplicate snapshot plus
  `rwd.py`; it is not part of this archive.

## 8. Reproducing

### What survives in git

| | tracked? |
|---|---|
| this document, `HANDOFF.md`, `RUNS_Continuation.md`, `archive/` | **yes** |
| every benchmark script (~17 files) | **yes** |
| ASSUME source, incl. the commented-out shaping and the `matd3.py:618-628` prints | **yes** |
| scenario CSVs, `config.yaml`, study cases | **yes** |
| the run archive: 293 MB of `.npz` and figures | **no** (`.gitignore:143`) |
| **the starting replay buffer** `buffers/single_10ep_standard.npz` | **no** (`.gitignore:142`) |
| run 07's saved policies | **no** (same line; 532 MB total) |
| `gymnasium`, `stable-baselines3` | **no** |

The consequential one is the **starting buffer**: 24 KB, the same file for every
run 09–12 trial, SHA256-guarded by both sweep runners, not in git. **Tracking those
24 KB is the single cheapest thing that would make runs 09–12 re-runnable.**

### Recreating the starting buffer

**It contains no policy.** `update_policy()` is only called once
`episodes_done >= episodes_collecting_initial_experience`, so there are **zero**
gradient steps during collection, and `get_actions` returns
`th.rand_like(noise) * 2 - 1` while `collect_initial_experience_mode` is set, so
the actor is never queried. The archived file's actions are indistinguishable from
`Uniform(-1, 1)` (KS D = 0.051, p = 0.076, n = 620). A regenerated buffer is
therefore **statistically equivalent but not bit-identical** — runs 09–12 would
have to be re-run and the two `BUFFER_SHA256` guards updated.

`inc_dec_collect_buffer` exists for this. Two things its comment block does *not*
say, both established by re-running it: the horizon must be **72 h**, not the 24 h
it inherits (products per episode is `hours − 10`, so 24 h gives 14 and 72 h gives
62); and the archived file is the state after **episode 10**, not 11 (620 = 62 ×
10; all 11 gives 682).

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

Everything else degrades gracefully: figures regenerate from the `.npz` files, the
`.npz` files regenerate by re-running the sweeps, and the sweeps are deterministic
given a seed and a fixed thread count — with the caveat of run 08 for the SB3
surrogate, which is chaotic in the BLAS thread count and therefore not reproducible
across machines at all.

### Commands

New runs write to `examples/outputs/2_nodes_paper_small/rl_benchmark/` by default,
beside `runs/`, never into the tracked input folder.

```bash
cd examples/inputs/2_nodes_paper_small/rl_benchmark
O=../../../outputs/2_nodes_paper_small/rl_benchmark

# headline figures, all redrawn from archived data (--critic-seed 3 is TD3's failing seed)
python sweeps/run_benchmark.py --replot --critic-seed 3 \
    --results $O/runs/data/01b-best-known/headline_comparison.npz \
    --out    $O/runs/01-algorithms-best-known-settings.png
python sweeps/td3_stability.py --replot --results <runs>/data/08-stability/td3_stability.npz
python real_matd3/assume_film.py                  # run 09
python real_matd3/assume_stability.py --report    # run 10, from the archive
python real_matd3/assume_config_sweep.py --phase broad --report-only \
    --critic-out $O/runs/img/11-assume-config-critic-evolution-broad.png

# mechanism figures
python analysis/descent_window.py         # when the descent path is open
python analysis/activation_comparison.py  # tanh vs softsign
python analysis/critic_landscape.py       # final critic vs true reward
python analysis/actor_saturation.py       # regenerates actor_saturation.md

# fresh surrogate runs
python sweeps/td3_stability.py --seeds 8 --workers 15      # ~40 min

# runs 09/10. One condition per invocation: the shaping is a source edit, so
# uncomment learning_strategies.py:1583-1589 between the two and put it back.
python real_matd3/assume_training_probe.py --episodes 40 --label <what-reward>
python real_matd3/assume_stability.py --condition unshaped --seeds 42 1 2 3 4 5

# run 11's broad screen: 30 configs x 3 seeds, six one-thread workers
python real_matd3/assume_config_sweep.py --phase broad --workers 6
python real_matd3/assume_run_diagnostics.py --results <run-11.npz> --out <plot.png>

# run 12. Both levers are patches installed in the child before the scenario
# loads, so assume/ is untouched -- but the shaping must stay commented out and
# preflight() refuses to start otherwise.
python real_matd3/assume_actshare_sweep.py --workers 5
python real_matd3/assume_actshare_film.py         # both figures, from the archive

# the offline gamma=0 harness: no simulation, ~15 min, reproduces the live
# failure. This is where new critic architectures get screened first.
python real_matd3/assume_offline_critic.py

# run 13. No shared starting buffer exists at N > 1, so each trial collects its
# own 5 exploration episodes. Locally memory-bound: ~0.85 GB per trial, so
# --workers 4 on a 16 GB machine. Leave --threads at 1.
python real_matd3/assume_multiagent_actshare.py
python real_matd3/assume_multiagent_actshare.py --report-only
python real_matd3/assume_multiagent_grids.py      # critic grid, bid grid, summary
python real_matd3/assume_multiagent_window.py     # run 06's window statistics
```

For running any of this on the cluster instead, see
[`RUNS_Continuation.md`](RUNS_Continuation.md).

## 9. Layout

```
runs/
├── 01-algorithms-best-known-settings.png   ┐
├── 02-td3-stability-sweep.png              ├ the four headline figures
├── 03-assume-matd3-films.png               │
├── 12-actshare-dose-response.png           ┘
├── data/
│   ├── 01-baseline/   01b-best-known/   02-critic/ (+ models/)
│   ├── 03-sac-entropy/   04-td3-policy-delay/   05-softsign/   06-window/
│   ├── 08-stability/   09-assume-films/   10-assume-stability/
│   ├── 11-assume-config-stability/broad/   30 configs × 3 seeds, Q1/Q2 films
│   ├── 12-actshare/                        5 conditions × 3 seeds
│   └── 13-multiagent-actshare/             6 conditions × 3 seeds, 11 agents
└── img/                                    detail figures, numbered by run
```

Runs 01b, 08, 09 and 12 have no `img/NN-*` entry because their figure **is** one of
the four headline figures — that is the usual reason a run looks like it has no
plot. Run 07 has no data folder; it reads ASSUME's saved networks in place.

Each `.npz` holds `steps`, `greedy/<algo>` (probe timesteps × seeds, the noise-free
policy), `placed/<algo>` (every bid actually placed) and, except for 01, `cfg/*`.
Runs recorded with `--critic-grid N` additionally carry `critic_bids` and
`critic_q/<algo>` / `critic_grad/<algo>`, shaped `(seeds, probes, N)` — the actor's
own objective and its **autograd** gradient at every probe. PPO and random search
never have these.

### Source files

| file | role |
|---|---|
| `_layout.py` | folders, `OUT_DIR`, `resolve()` — imported by every script |
| `surrogate/incdec_reward.py` / `incdec_env.py` | the closed-form landscape; the Gymnasium env |
| `sweeps/run_benchmark.py` | training driver, CLI, plotting, **the house palette** |
| `sweeps/td3_stability.py` | run 08 — the configuration sweep and its figure |
| `test_rl_benchmark.py` | the four things that would fail silently |
| `test_exploitability.py` | 22 unit tests of the clearing / exploitability maths; `_main()` is an uncollected scratch driver |
| `exploitability_two_bid_walkthrough.py` | why the exploitability search is exhaustive, not a heuristic — prints its own derivation, needs no data |
| `analysis/critic_probe.py` | reads a trained critic: `actor_objective`, autograd `critic_curve` |
| `analysis/critic_coherence.py` | `argmax_disagreement` / `argmax_range` — **the one definition runs 10–13 share** |
| `analysis/critic_landscape.py` / `critic_evolution.py` / `activation_comparison.py` | final-critic figure; gradient field over training; tanh vs softsign |
| `analysis/actor_saturation.md` / `.py` | the tanh/softsign derivation and its tables |
| `analysis/descent_window.py` | run 06 — when the descent path is open, and for how long |
| `real_matd3/assume_critic_probe.py` | run 07 — ASSUME's own saved MATD3 networks |
| `real_matd3/assume_training_probe.py` | films both critics over a live run, no edit to `assume/` |
| `real_matd3/assume_film.py` / `assume_stability.py` | runs 09, 10 — a pair of films; the same across seeds and both conditions |
| `real_matd3/assume_config_sweep.py` / `assume_run_diagnostics.py` | run 11 — the guarded parallel config sweep; per-trial diagnostics |
| `real_matd3/assume_actshare_sweep.py` / `assume_actshare_film.py` | run 12 — the ladder and its two patches; its figures |
| `real_matd3/assume_offline_critic.py` | run 12 — the γ = 0 offline harness, **where architectures get screened** |
| `real_matd3/assume_multiagent_actshare.py` | run 13 — runner, `act-own`/`act-all` patches, recorder, `--measure` |
| `real_matd3/assume_multiagent_film.py` / `_grids.py` / `_window.py` | run 13 — its four figures |
