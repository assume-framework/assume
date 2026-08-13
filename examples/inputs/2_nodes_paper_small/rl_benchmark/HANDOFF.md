# Handoff — inc-dec RL benchmark

Read this to resume in a fresh session. The detail is deliberately not in here.

| file | what it is | read it |
|---|---|---|
| **`HANDOFF.md`** (this) | the state of the question and the forward plan | always, first |
| [`RUNS.md`](RUNS.md) | runs 01–13 — one block per run, its command, its numbers | for a specific run |
| [`RUNS_Continuation.md`](RUNS_Continuation.md) | **new runs go here**, incl. every cluster run | when adding a run |
| [`archive/`](archive/) | the full-length originals, 2 600 lines | **`grep` only** — reading them costs ~35 k tokens |

Findings live here. Numbers live in `RUNS.md`. `archive/` is frozen.

## What this is

Can standard RL find the optimum of the `2_nodes_paper_small` inc-dec reward
landscape (`reward_landscape.png`, unit `diesel_0`), and if not, why not? Two
halves: a closed-form **surrogate** driving SB3 (runs 01–08, ~3 min per
10 000-step run, no market clearing), and probes of **ASSUME's own MATD3**
(runs 07–13, live training).

**Code** (tracked): `examples/inputs/2_nodes_paper_small/rl_benchmark/`.
**Results** (gitignored): `examples/outputs/2_nodes_paper_small/rl_benchmark/runs/`.
Needs `gymnasium` + `stable-baselines3` in the `assume` conda env (installed, not
in `pyproject.toml`); run scripts with `conda run -n assume python`.

**A fresh clone has no starting buffer.**
`learned_strategies/buffers/single_10ep_standard.npz` — 24 KB, SHA256 `5f1b80b4…`
— is gitignored and both sweep runners refuse to start without it. It is
recreatable and contains no policy; `RUNS.md` §8 has the recipe (needs a **72 h**
horizon and the file written after **episode 10**).

## The landscape

```
b > 49          not dispatched            reward  0        (27 % of action space, flat)
30 <= b <= 49   dispatched, then dec'd    reward (49-b)/100 -> peak +0.190 at b=30
b < 30          dispatched, not dec'd     reward -0.170    (62 % of action space, flat)
```

The optimum sits **on a cliff edge** — 0.36 drop one tick below — and only 9.5 %
of the action space carries any gradient. So a stochastic policy should **not**
aim at 30: at σ = 1 EUR, centring on 30 earns 0.006 against 0.163 at the
constrained optimum of 32.31.

## Findings

1. **Exploration is never the bottleneck.** 1000 uniform warmup steps put ~10 % of
   samples in the band and nothing is evicted; ASSUME's frozen buffer has 620
   transitions, 10 % in band, best in-band reward +0.199.
2. **SB3's tanh actor breaks surrogate TD3/DDPG** — float32 `tanh` pins to exactly
   1.0 at `z ≥ 9.011`, so every actor gradient is exactly zero, 8/8 seeds frozen at
   100.00. **ASSUME's `Actor` defaults to softsign** and is not affected. Softsign
   unfreezes the actor without solving the task (baseline 3/8) and introduces a new
   failure mode: **overshooting through the band to −100**. **Step size is then the
   lever, not activation** — `lr-1e-4` solves 8/8 at 32.10 ± 0.37, because inside a
   19-EUR band bounded by a 0.36 cliff a full-sized Adam step is too big to stay
   there. A *small* gradient is not a problem; Adam is scale-invariant, only a hard
   zero kills it.
3. **In the surrogate the escape is a window, not a standing option.** The critic
   first learns "higher is better" (true of 89 % of the space) and every actor
   runs to the ceiling — a correct gradient on an incomplete critic. The plateau
   slope flips at step **1400 in 11 of 12 runs**, identical for TD3/DDPG/SAC,
   opening an unbroken descent path for ~600 steps, then fragments because a
   *correctly* learned flat region has a noise gradient. **`policy_delay = 2` is
   why DDPG beats TD3 here** — half the actor updates per environment step,
   against a window counted in environment steps.
4. **SAC needs its entropy floor lowered and is then the best surrogate learner**
   (`--ent-coef 0.001` → 31.35 ± 0.31, 4/4). Best-known ordering: SAC 31.35, DDPG
   31.86 ± 0.51, TD3 1.59 ± 52.26, PPO 100.00.
5. **Real single-agent MATD3 learns nothing usable from the true reward**, not even
   at 4× budget: the critic's field stays incoherent for all 2560 updates and 5 of
   6 seeds never place a probed bid in the band. Run 11's 30-configuration
   hyperparameter screen is **0/3 solved in every cell**. The reward shaping fixes
   the symptom but onto a **decoy** — `learning_strategies.py:1583-1589` fires only
   when `reward <= 0`, so it re-enters at full height at bid 49, giving two maxima
   (+0.190 at 30, +0.170 at 49 where the true reward is 0.000). All 6 shaped seeds
   settle at the band's *rim*, 50.6 ± 3.0.
6. **The standing hypothesis is that the action is 1 of 75 critic inputs**, and
   raising its share solves the task 3/3 at the same 800-update budget
   (`act-x30` → bid 33.0 ± 0.2, measured reward +0.167, against 0/90 in run 11).
   Two mechanically unrelated levers land on one curve, so the *ordering* variable
   is real. **But `act_share` is my own quantity, not a literature one**, and
   both levers are experiment monkeypatches rather than an API. **Do not build
   further on it before the workstream B tests below** — the literature has two
   standard mechanisms that address the same thing and one of them predicts the
   opposite of run 12's observation-scaling ladder.
7. **At 11 agents it buys rate, not feasibility** (run 13, `inc_dec_learning`, 105
   critic inputs). The own-action share halves to **0.016** as predicted and orders
   the outcome monotonically at each budget — but **the untouched baseline is slow,
   not stuck**: 97.1 ± 0.2 at 1200 updates, 80.1 ± 12.3 by 2700. Worth roughly a
   **2.25× budget multiplier**, not the difference between learning and not
   learning. **Attach a budget to any "MATD3 cannot learn X" claim**; every such
   statement in this archive is single-agent.
8. **It is the *own* action's share, not the action block's.** Scaling the whole
   action vector caps each agent's own share at **1/N** (0.091 at N = 11) for any
   S. **Any fix inside `CriticTD3` has to be per-agent.**
9. **Run 10's incoherence statistic inverts at N = 11 — do not carry it over.**
    With eleven agents the critic's preferred bid genuinely should depend on the
    observation, so disagreement between probed observations stops being evidence
    of a broken critic.
10. **On a plain energy-only market the same learner works, and `act_share` does
    not predict it** (run 14, `sb02a`–`sb02c`, single bid, cluster). All 9 trials
    converge, seeds agree to **0.6–1 EUR** against run 13's 15–17, and the single
    pivotal unit finds a **+17.2 EUR markup** over marginal cost with reward rising
    monotonically. `act_share` is **0.004–0.010** — an order of magnitude *below*
    run 13's failing baseline — and the task is solved anyway. So `act_share`
    orders outcomes *within* the inc-dec landscape; it is not a general predictor
    of whether MATD3 can learn. **The failure in runs 09–13 is a property of that
    reward landscape, not of MATD3, the critic architecture, or the input count.**
    Not a single-variable contrast: scenario, bid structure, `lr` and horizon
    all differ from run 13 at once.
11. **Markup tracks learning *capacity*, not learner count.** 1 × 2500 MW → +17.2;
    5 × 500 MW → +0.8 (competed to marginal cost); 10 × 500 MW → **+7.8**, back up.
    What changes between the last two is 5000 MW of learners against the same naive
    fleet, not the agent count. Do not read the ladder as "more agents ⇒ more
    competition".
12b. **The stage game has two equilibria and MATD3 learns only one of them.**
    Because the naive fleet bids marginal cost, the NE switches with demand:
    where a learner is left **undispatched** it undercuts and NE is **marginal
    cost** (`bertrand`); where every learner runs and one is only **partly**
    dispatched, that unit is marginal and irreplaceable and NE is the **backup's
    marginal cost, 85.7** (`pivotal`). `sb02b` contains both (67 % / 13 % of
    hours) and ends **6.4× more exploitable in the pivotal hours** — 2698 against
    424 EUR/h — with the pivotal trace flat from about episode 5 while the
    Bertrand trace keeps falling. `sb02c` (all Bertrand) reaches 0.02× of its
    starting exploitability; `sb02a` (all pivotal) only 0.20 ×, and its residual
    7209 EUR/h is almost exactly the markup it failed to take.
    **The explanation originally attached to this finding — "bidding down to
    cost is easy, bidding up to a rival's cost has no local gradient" — is
    refuted by run 15 (finding 14). The measurements above stand; the direction
    of the deviation is not what makes the difference.**
    `analysis/eom_exploitability.py`.
12. **Exploitability is the sharpest of the three statistics, and it is the one
    that separates market power from bad play.** Per MW of learner capacity it
    falls **2.3 → 1.3 → 0.29 EUR/MWh** across `sb02a`/`sb02b`/`sb02c`: the
    ten-learner case ends **closest to Nash** while bidding *further above*
    marginal cost than the five-learner case. It measures distance to best
    response, not distance to competitive pricing. Monotone decreasing over
    training in all 9 runs.
13. **ASSUME's MATD3 is chaotic on `inc_dec_learning`, and per-seed bids are not a
    reproducible quantity across machines** (run 13-rerun). Local vs. cluster, same
    seed and `--threads 1` on both: divergence starts at **frame 0 at float32
    rounding scale** (≈1e-7 relative, entering through the simulator and torch, not
    the RNG stream) and amplifies to **177 EUR** on a 200-wide action range. **What
    reproduces is the condition means and their ordering** — ≤ 4.1 EUR apart, well
    inside the seed spreads, monotone in the own share at both budgets, with
    `act_share` matching to three decimals. Report conditions; treat per-seed
    columns as one draw.

14. **What a shared critic fails to fit is whatever is RARE, not whatever
    direction it points** (run 15, the pivotal-frequency ladder). `p1`–`p7` are
    `sb02b`'s fleet and market with only the demand series changed, so the
    pivotal share runs 13 → 88 % (`p1` is the undistorted series, i.e. `p1` **is**
    `sb02b`, and reproduces it to the decimal). Exploitability per regime at the
    last evaluation episode:

    | case | pivotal hrs | pivotal | change | bertrand hrs | bertrand | change |
    |---|---:|---:|---:|---:|---:|---:|
    | `p1` | 13 % | 2698 | 0.34× | 67 % | 424 | **0.08×** |
    | `p3` | 38 % | 659 | 0.50× | 62 % | 1871 | 0.21× |
    | `p5` | 63 % | 841 | 0.23× | 37 % | 5200 | 0.56× |
    | `p7` | 87 % | **260** | **0.04×** | 13 % | 6313 | 0.67× |

    **The two columns cross.** The same regime converges when common and stays
    exploitable when rare, in both directions — so finding 12b's "bidding up has
    no local gradient" explanation is wrong, and the variable is the regime's
    share of the replay buffer. Fleet outcome moves with it: mean clearing price
    52.73 → 72.26 (against mc 55.71, backup 85.71), profit 234 → 7415 EUR per
    learner, reward +0.002 → +0.146.
    `p2`'s pivotal rises 5.85× against the trend, off an anomalously low ep1;
    not chased down. Exploitability is not monotone in `k` — read the crossing,
    not the individual rungs.
15. **At high pivotal share the learners find the ASYMMETRIC equilibrium, and it
    is a learned role assignment.** From `p3` up, in all three seeds at every
    rung, **exactly one unit** marks up to ~70–80 and the other four sit at the
    action floor. That is correct play, not collapse: under pay-as-clear an
    infra-marginal unit is paid the clearing price whatever it bids, so the floor
    is a best response for whoever is dispatched anyway while the marginal unit
    sets the price. The per-unit critic films show it directly — the marked-up
    unit's field has an interior optimum on the backup line, the others' are
    monotone decreasing across the whole bid range, which is the price-taker's
    value function. **Never read this fleet with a median over units**: the
    median of a bimodal equilibrium is −100 and means nothing.
    `runs/img/15/14-eom-regimes-p7-seed42.png`.
16. **Correction 17 is real and scales with how much the agents matter to each
    other** (run 14c). Both sweeps are now recorded: `diag` holds the other
    agents at their current actors' greedy actions, `diag:stored` at the action
    they actually played, which is what `matd3.py:704` feeds the critic.

    | case | learners | mean \|Δgrad\| / mean \|grad\| | median argmax shift |
    |---|---:|---:|---:|
    | `sb02a` | 1 | **0.0000** | 0.0 EUR |
    | `sb02b` | 5 | 0.407 | 8.3 EUR |
    | `sb02c` | 10 | 0.356 | 7.0 EUR |
    | `p4` / `p7` | 5 | 0.83 / 0.73 | 12.7 / 11.3 EUR |

    `sb02a` at exactly zero is the implementation's own control — one learner
    means no other agents, so the two definitions must coincide, and they do
    bit-for-bit. Everywhere else the gap is 36–83 % of the gradient magnitude and
    moves the critic's peak 7–13 EUR, **largest in the pivotal-heavy cases**,
    where the joint action sets the price. Any single-number critic reading from
    runs 09–13 carries an error of this size.
17. **RSNorm is disqualifying, and SimBa's residual trunk is worth ~15× the
    parameters** (run 17, the offline γ = 0 architecture screen, 15 critic
    architectures × 5 seeds on the frozen buffer). Two effects:

    | group | n | mean `in_band` | best |
    |---|---:|---:|---:|
    | carries RSNorm | 6 | **0.005** | 0.03 |
    | no RSNorm | 7 | **0.253** | 0.57 |

    **RSNorm acts as a gate, not a gradient.** Every variant carrying it is
    pinned at `argmax` exactly 100.0 at *every* width tested — 143 k, 548 k and
    8.5 M alike — so it does not merely fail to help, it **cancels the capacity
    effect entirely**. `simba` against `simba-nornorm` is a single-variable
    contrast (identical width, seeds, budget, 8,483,842 params both) and is the
    whole difference between 100.0 and 34.1. This confirms run 12's "z-scoring
    the observation was the worst cell" was not an artifact of crude z-scoring,
    and the residual path does **not** repair it.

    **SimBa's residual trunk shifts the whole scaling curve left by ~15×.**
    Both families measured at the same three widths, `in_band` (argmax):

    | params | plain MLP | SimBa trunk |
    |---:|---:|---:|
    | 105 k | 0.10 (94.5) | — |
    | 143 k | 0.10 (91.5) | **0.27** (70.5) |
    | 548 k | 0.10 (80.2) | **0.40** (41.0) |
    | 8.5 M | 0.43 (44.4) | **0.57** (34.1) |

    **The MLP curve is flat at 0.10 from 105 k to 548 k** and only breaks
    through at 8.5 M. The trunk rises monotonically from the smallest width
    tested. At matched outcome the trunk needs **548 k against the MLP's
    8.48 M — 15.5× fewer parameters for a better argmax** (41.0 vs 44.4).

    So capacity alone buys almost nothing until 8.5 M; what the trunk does is
    make capacity *usable*, which is exactly the claim SimBa's Fig. 2b makes
    (an MLP degrades as it widens, SimBa improves). **Read this as parameter
    efficiency at matched outcome, not as a gap at matched parameters** — the
    matched-parameter reading understates it roughly 15-fold and is what an
    earlier draft of this finding got wrong.

    **The simplicity score anti-predicts the outcome here**: Pearson **−0.57**,
    Spearman **−0.67** between simplicity and `in_band` across the 15, against
    the +0.79 the paper reports between simplicity and return. The simplest
    critics in the table are exactly the RSNorm-gated ones.
    **Caveats:** γ = 0 removes bootstrapping and actor feedback, this is one
    reward landscape and 620 transitions, and `in_band` at 5 seeds × 6 probes
    is a coarse measure. It is a screen. **Nothing here has been tried live**,
    and finding 10 says this landscape does not generalise.

## Refuted or revised — do not re-test

- ❌ "The critic smooths the cliff into a ramp." It doesn't; the critics learn it.
- ❌ "A 1e-5 gradient is too small to move the actor." Adam is scale-invariant.
- "`policy_delay` fixes it." Raising it does not (8 → 0/8 with softsign) — but
  the knob is not inert; run 04 turned it the **wrong way** (finding 3). Lowering
  it is untested in the surrogate and 0/3 on ASSUME at 800 updates.
- ❌ "The surrogate's `lr-1e-4` transfers to ASSUME." 95.5 ± 0.3, 0/3 (run 11).
- ❌ "The real critic fails because it must fit `Q(s,a)` across 548 observations."
  Run 12 has the controls: **shuffled** observations fail identically, and a critic
  that memorises harder learns the band correctly. The observation *count* stood in
  for the input *dimension* count.
- "Without shaping or a raised `act_share`, MATD3 never forms a preference" —
  single-agent and budget-dependent (finding 7).
- "Any single result here is a sample from a bimodal distribution, because BLAS
  thread count alone flipped a seed." True of the SB3 surrogate. For ASSUME the
  claim was "runs 10 and 13 reproduce bit-identically across thread-count
  changes" — **that holds only within one machine.** The run 13 cluster rerun
  reproduces *none* of the 18 trajectories (finding 13). Thread count was pinned
  at 1 on both sides, so it is not the lever; the learner is simply chaotic and
  float32 rounding is enough to separate two runs.
- ❌ "`act_share` predicts whether MATD3 can learn." It orders outcomes on the
  inc-dec landscape. Run 14 solves a plain EOM at `act_share` **0.004–0.010**,
  below run 13's failing baseline (finding 10). This is a stronger reason to
  finish workstream B than the one B was written for.

## Traps that cost time

- **`--train-freq 1h` on this scenario dies** with `No rewards were collected
  during evaluation run`. Leave it at **12h**; that caps film resolution at one
  frame per training block.
- **`algorithm.n_updates` is useless as a time axis** — the world is rebuilt every
  episode, so the counter restarts. Keep your own cumulative count.
- **A fresh `run_learning` rmtree's `trained_policies_save_path`**, and with no TTY
  the confirmation raises `EOFError` instead of defaulting. Relatedly, a
  path-valued `learning_config` override needs **both** forms — relative on
  `world.scenario_data`, absolute on the live `learning_role.learning_config` —
  and getting it wrong exits with `no buffer file found` and no traceback.
- **`conda run` buffers the child's stdout to the end** (a backgrounded run looks
  hung), **two concurrent calls collide** on `%TEMP%\__conda_tmp_*`, and
  `conda run python -c` rejects newlines. For anything parallel call the
  interpreter directly: `C:/Users/finnr/miniconda3/envs/assume/python.exe`.
- **Shaped and unshaped cannot run concurrently** — the shaping is a source edit,
  so the whole process tree is in one condition. It is currently **commented out**,
  deliberately: the repo is in "true reward" state.
- **Record `--critic-grid 401`** on anything you may later want to explain; it is
  nearly free at training time and run 01b cost a ~55 min re-run to add it.
- **The `matd3.py:618-628` debug prints are commented out** (run 13 did that) —
  they drew `th.rand(1)` per gradient step per agent, and runs 09–12 were recorded
  with them **live**.
- **A film's x axis carries two coordinates at once, and `--every` can alias them.**
  Every episode replays the same calendar, and `train_freq` is snapped by
  `learning_role.sync_train_freq_with_simulation_horizon` to divide the horizon
  evenly (`100h` over 31 days → **93h**, 8 blocks). So "critic updates" mixes
  training progress with position in the horizon, and if `blocks_per_episode /
  every` is small the frame index samples a *fixed handful of calendar phases
  forever*. Run 14 hit exactly this: `--every 4` against 8 blocks gave two frames
  per episode, so the reward trace alternated between the two halves of March
  (6026 vs 5818 MW) and **reversed direction on 77–95 % of consecutive frames** —
  which reads as instability and is a calendar. **Check
  `blocks_per_episode / every` before believing any oscillation in a film**, and
  prefer a value that divides the block count evenly or equals it.
  `analysis/eom_critic_evolution.py:frame_schedule` derives the mapping for
  already-recorded runs; new runs carry `frame_time` and `frame_episode`.
- **Exploration never finishes, and the films do not show it.** Two phases:
  episodes 0–4 are **pure noise** (`curr_action = noise`, sd ≈ 0.20 ≈ 20 EUR/MWh,
  the actor is not consulted), then actor + Gaussian noise decaying from 0.10.
  Measured at the last training episode it is still **0.039 (3.9 EUR/MWh)**
  against the 0.001 the linear schedule implies — a free fit puts the decay on
  course to reach zero at **episode ~129**, and training stops at 100. Likely
  cause: `get_progress_remaining`'s `episodes_done` counts the 19 evaluation
  episodes while the denominator is `training_episodes - collecting = 95`
  (`95 + 19` fits much better than `95`, though not perfectly). **No finding is
  affected** — the films record the actor's deterministic output and
  exploitability is read on evaluation episodes, both noise-free — but every
  critic here was fitted on a buffer carrying ≥ 3.9 EUR of action jitter, which
  is a floor on how sharply any of these policies could converge.
  Why no exploration marker appears in the films: `cfg/warmup = 0` and
  `frame_episode[0] = 5`, because the recorder hangs off `update_policy`, which
  only runs once collection is over. The window is not in frame, not missing.
  Runs ≤ 06 filmed the single-agent surrogate from step 0 and so included it.
- **The replay buffer wraps two-thirds of the way through.** One ring for the
  whole run, never reset per episode (`buffer_fill` monotone), but it saturates
  at 50 000 at **frame 124 = episode 67** and overwrites from there — so the
  last third of training has forgotten the earliest episodes, including the
  entire pure-noise phase. Finding 14's "share of the buffer" is therefore a
  share of the last ~50 000 transitions, not of all training. Identical
  trajectory across 14c and 15, so cases stay comparable.
- **The figure scripts do not run on the cluster.** The house palette lives in
  `sweeps/run_benchmark.py`, which imports `stable_baselines3` at module level,
  and SB3 is a surrogate-only dependency that is not in `pyproject.toml` and not
  in the cluster env. Both collectors are `|| true` throughout so this costs you
  nothing but the `img/` folder — **redraw locally after unpacking**. Runs 13-rerun
  and 14 both came back figure-less for this reason.

## Caveats on the current results

Two of these are the reason workstream A exists.

- **Run 13's recorded critic field is not matd3's actor objective.**
  `matd3.py:704` holds the other agents at their **stored** actions; the recorder
  holds them at their **current actors'** greedy outputs. A valid critic slice, but
  the window / `pulled left` / coherence readings describe the response to the
  current joint policy, not the gradient the actor actually climbs. Empty at N = 1,
  so runs 09–12 are untouched. `RUNS.md` correction 17.
- **The evaluation database holds two hours per episode** (10:00 and 11:00 of
  14), an unflushed async write at shutdown. Best-policy selection, early stopping
  and any reward read from `rl_params` are an early-hours sample; training is
  unaffected. `RUNS.md` correction 16.
- **The surrogate is not the scenario's reward** — `reward_from_bid` agrees with
  the frozen buffer's stored rewards on **24.8 %** of transitions. Bids, critics,
  `act_share` and the offline fits stand; every reconstructed reward, `regret`, the
  `+0.15 solved` bar and the `32.31` optimum do not. Run 12's headline survives on
  *measured* reward. Do not retune the surrogate — it is exact for runs 01–08 by
  construction. Correction 15.
- **Run 13 used a working-tree `inc_dec_learning`** (72 h horizon, lr 1e-4, 50
  episodes, `train_freq` 12h), table in `RUNS.md` §3. **Restore or commit it
  before re-running.**
- **Seed counts are small** (3 per condition for runs 11–13, 6 for run 10). They
  reject easy fixes; they are not success rates.
- **`RUNS.md` §13's per-seed bid columns are one draw, not the numbers.** The
  cluster rerun reproduces the condition means and their ordering but none of the
  18 trajectories (finding 13). Anything quoted per seed off that table — or off
  runs 09–12, which have never been re-run on a second machine — carries the same
  caveat.
- **Two `ReplayBuffer` defects in `assume/` are open and untriggered** — an early
  `full` wrap and episode-boundary bootstrapping. Fixing them is a library change.

---

# Forward plan

Three workstreams, in order. **A** repairs the measurement so anything built on it
survives; **B** replaces `act_share` with mechanisms that exist in the literature;
**C** asks whether any of it generalises off this scenario. All of it is
seed-and-budget hungry, i.e. cluster work — see [`RUNS_Continuation.md`](RUNS_Continuation.md)
for the harness, the preflight checklist and where logs go. **No GPUs**: two small
MLPs against a CPU-bound simulator; CPU partition, `gres` unset, 4 cores / 8 GB
per task, parallelism from the array width.

## A. Fix the recording, then re-run what depends on it

Both defects are in the *measurement*, not the learner, so every number they touch
is provisional until this is done.

**Run 13-rerun does not close this and was never meant to.** It re-ran run 13's
*configuration* on the cluster and settled reproducibility (finding 13); it used
the same recorder, so both defects below were live in it too. Nothing is
outstanding on the rerun — 18/18 `COMPLETED`, written up in
[`RUNS_Continuation.md`](RUNS_Continuation.md) § 13-rerun. A is a code change to
`MultiAgentRecorder`, not a batch of jobs.

✅ **A.1 is DONE for the EOM recorder** (`real_matd3/eom_critic_film.py`), and it
measured the defect rather than caveating it — finding 16. Every sweep is
recorded twice, `diag` and `diag:stored`, at one extra forward/backward per
probe. The stored actions are read **once**, at the same instant the probe
observations are frozen: the buffer is a ring that fills and wraps here
(`buffer_fill` saturates at 50 000), so a later read would pair a frozen
observation with another transition's action. Figures take `--sweep`.
**Not yet ported to `MultiAgentRecorder`** — step 1 below is what run 13 needs,
and the EOM implementation is the template for it.

1. **Record the actor's actual objective.** Give `MultiAgentRecorder` a second
   sweep that holds the other agents at the **replay batch's stored actions**, the
   way `matd3.py:704` does, alongside the current-policy sweep it already takes.
   Recording both is cheap (one extra forward/backward per probe) and settles
   whether run 13's window and `pulled left` readings change at all. `RUNS.md`
   correction 17, pinned by `test_rl_benchmark.py`.
2. **Record rewards from the buffer, not the database.** Give the single-agent
   `Recorder` the buffer-reward snapshot `MultiAgentRecorder` already has. That
   removes the two-hours-per-episode sample (correction 16) *and* the surrogate
   reconstruction (correction 15) in one change, since buffer rewards are what the
   simulator actually paid.
3. **Then re-run** runs 10 and 12's reward columns and run 13's window table on
   the cluster. Runs 10–11's reward columns are currently labelled `recon` and have
   never been recomputed; run 12's measured column exists but is the early-hours
   sample. **Budget for condition means, not per-seed comparison**: finding 13
   showed the trajectories do not reproduce across machines, so a fixed-recorder
   run 13 can only be compared to the archive at the condition level, and any
   per-seed difference is uninterpretable at 3 seeds. `cluster/rerun_run13.sh`
   already runs the right 18 tasks once the recorder change lands.

Optionally also fix the two `ReplayBuffer` defects (§caveats) while re-running —
untriggered at these buffer sizes, but a longer cluster run is exactly where the
early-`full` wrap would start returning zero rows.

## B. Literature-grounded critic changes, instead of `act_share`

✅ **Built 2026-08-13.** The code is in and tested; the offline screen is the
thing to read. What exists now:

| file | what it is |
|---|---|
| [`real_matd3/critic_architectures.py`](real_matd3/critic_architectures.py) | 32 drop-in `CriticTD3` replacements + a `REGISTRY`, **width and depth** knobs, `match_width`, and the run-18 parameter ladder |
| [`analysis/simplicity_bias.py`](analysis/simplicity_bias.py) | SimBa's Fourier simplicity score, two estimators |
| `assume_offline_critic.py --round arch` | the γ = 0 screen over any subset of them |
| `eom_critic_film.py --critic-arch X` | the live sweep, one flag |

**No `assume/` change was needed.** `install_critic_arch` rebinds
`matd3.CriticTD3` before the world loads, which is enough because
`create_critics` builds the class by name out of its own module globals.

Three things a fresh session should not have to rediscover:

1. **A normalizer's running statistics must be `nn.Parameter(requires_grad=
   False)`, never `register_buffer`.** `polyak_update` zips `parameters()`,
   which does not yield buffers, so a buffer-held running mean never reaches
   the target critic: `Q` standardizes its inputs, `Q_target` does not, and the
   TD target is computed on a different input scale for the entire run, in
   silence. `_stats_to_params` does the promotion; `test_rl_benchmark.py`
   pins it for every registered variant.
2. **RSNorm must not update inside `forward`.** `matd3.py:713` calls
   `q1_forward` on the same batch `forward` just saw, so an update there
   would count every batch twice *and* make the actor's objective differ from
   the critic's own Q1. The statistics advance in one place: the twin-Q path,
   once per replay batch.
3. **Parameter counts must be matched before architectures can be compared.**
   SimBa's Fig. 4(a) holds all twelve of their architectures within 1 % of
   4.5 M (their Appendix D). `simba` at the paper's `d_h = 512` is 8.5 M
   against the baseline's 93 k, so an unmatched table moves architecture and
   capacity in the same column. `match_width` bisects the width to a target;
   `--match-params` turns it on.

Three checks that were done when this was scoped, kept because they still hold:

* **There is no critic-architecture config knob.** `actor_architecture_aliases`
  exists ([`algorithms/__init__.py:12`](../../../../assume/reinforcement_learning/algorithms/__init__.py#L12))
  but `CriticTD3` is constructed directly at
  [`matd3.py:388`](../../../../assume/reinforcement_learning/algorithms/matd3.py#L388)
  and `:396`. A live sweep therefore needs a one-line `matd3.CriticTD3 = MyCritic`
  patch before the world loads — the same pattern the probes already use — and
  **no `assume/` change at all**. Mirroring the actor's alias dict upstream would
  be the tidier fix and is PR-able, but is not required.
* ✅ **The offline harness no longer hardcodes the critic** — `fit(..., arch=)`
  plus `--round arch`. Still to build: `cluster/critic_arch.sh` ≈ 150 lines,
  ~90 % copied from the two existing launchers. Re-evaluating runs 09–13 under
  whatever wins is the open-ended part.

Run 15 also gives B a target it did not have when it was written: **the failure
mode is a rare regime being unfitted** (finding 14). RSNorm's per-dimension
standardization and SimBa's "scaling the critic helps" both act on exactly that,
and the ladder is now a ready-made benchmark for it — `p7`'s bertrand hours
(13 %, 0.67×) are a clean, reproducible instance of the defect to fix.

`act_share` is a quantity I invented; it orders the outcome but it is not a
mechanism anyone else uses, and both levers are monkeypatches. The literature has
two standard designs that act on the same thing. **`CriticTD3` currently
concatenates observation and action at the *input* layer**
([`neural_network_architecture.py:76`](../../../../assume/reinforcement_learning/neural_network_architecture.py#L76)),
hidden sizes `[256, 128]`, Xavier, ReLU, no normalization anywhere.

1. **Late action injection** — the DDPG default since Lillicrap et al. (2016),
   whose supplementary §7 states plainly: *"Actions were not included until the 2nd
   hidden layer of Q."* Feeding the action into layer 2 gives it its own weight
   matrix instead of 1 of 75 columns of the first one, which is the literature's
   version of "make the critic notice the action" — with no free scale parameter
   and no discarded observation dimensions. **This is the cheapest and most
   defensible test in the whole plan**: a variant `CriticTD3` plus one line in
   `forward`, testable offline first. The same paper also puts **batch
   normalization on the state input and on every Q layer prior to the action
   input**, i.e. it normalizes observations *and* keeps the action out of that
   path — worth running as its own condition.
2. **SimBa** (Lee et al., ICLR 2025) — three components, all with existing torch
   pieces except one:
   * **RSNorm**: per-dimension standardization of the observation by *running*
     mean/variance. ~15 lines; do **not** substitute `BatchNorm1d` or an
     env-wrapper normalizer — SimBa §7.1 tests exactly those and both
     underperform, the env-wrapper because off-policy buffers then hold the same
     observation under different statistics.
   * **Pre-LayerNorm residual feedforward blocks** with an inverted bottleneck
     (`nn.LayerNorm` + `nn.Linear` ×2 + ReLU, hidden 4·d_h), keeping a linear
     path from input to output.
   * **Post-LayerNorm** before the output head.
   Their setup: critic 2 blocks at width 512, actor 1 block at width 128, AdamW
   with weight decay 1e-2, lr 1e-4. Two of their results are directly on point
   here: **scaling the critic helps and scaling the actor hurts**, and **the
   benefit grows with input dimensionality** (their Fig. 9) — which is the same
   axis as `act_share` arrived at from the other side.
3. **The sharp test, and it is nearly free.** SimBa's most important single
   component is standardizing the observation. Run 12's offline ladder found that
   **z-scoring the observation was the worst cell in the table** — `act_share`
   0.008, `argmax Q1` pinned at exactly 100.0 in 5/5 seeds. Those two cannot both
   be the whole story. Either `act_share` is measuring the wrong thing, or
   normalization is only safe in company with the residual path and the
   LayerNorms. ✅ **Wired and runnable:**

   ```bash
   python real_matd3/assume_offline_critic.py --round arch --seeds 5 --threads 8
   ```

   ~40 s per variant for the eight narrow ones; the `d_h = 512` rows
   (`simba`, `simba-nornorm`, `simba+late`) are the expensive ones. That
   harness already reproduces the live failure, which is what makes it the
   right place to screen architectures.

## B'. The simplicity-bias measure

`analysis/simplicity_bias.py`. SimBa's Fourier complexity
`c(f) = Σ|f~(k)|·k / Σ|f~(k)|`, simplicity `s = E[1/c]`, **higher is simpler**.

**The definition is general in `n`; their estimator is not.** §2.1 quantifies
`f : X ⊆ R^n → Y ⊆ R^m` with no dimensional restriction, but the estimator is a
DFT on a uniform grid over the input domain, costing `G^n` points — 90 000 at
their `G = 300` and `n = 2`, and nothing at all at our `n ≈ 57`. Hence two
estimators here:

* **`grid`** — Appendix B step for step: the variant rebuilt with a
  2-input/1-output head, random init, `300 × 300` on `[-100,100]²`, 100 seeds.
  6–27 s per variant. This is the only number comparable to *their* protocol.
  Note their Fig. 4(a) is measured on architecture *templates at
  initialization*, not on a trained RL critic, so it has no time axis in the
  paper.
* **`lines`** — random 1-D directions through the buffer mean at full input
  dimension, `300 × 300 = 90 000` evaluations, the same budget as one grid.
  Sub-second. Works on the real critic, so it is the one with a time axis.
  1-D restrictions of an n-D function are **not** the n-D spectrum measured
  cheaply. Compare `lines` to `lines` only, never to `grid`.

Two conventions, both load-bearing: the output is **mean-centred** before the
transform (the DC term otherwise carries the function's mean and swamps
everything), and frequency is **normalized so Nyquist = 1**. So absolute values
are *not* comparable to the paper's printed 5.8–6.5 — **the ordering is the
claim**.

**The RSNorm rows of `grid` are the least trustworthy in that table.** An
unfitted RSNorm is exactly the identity, so it cannot be scored at all; fitting
it on `[-100,100]²` makes it divide the input by ~58, so most of what that row
measures is an input rescaling. `--no-fit-rsnorm` gives the other reading.
The paper hits this too — its Appendix C.1 reports RSNorm's score moving
*opposite* to its effect on return, for this reason.

**Over time**: `eom_critic_film.py` records `lines` per agent per frame
(`--simplicity-lines`, default 128, `0` disables), saved as `simplicity` in
the `.npz` alongside the critic field, so it shares the film's time axis. The
offline `--round arch` table carries it per fitted critic, which makes that
table a direct test of the paper's own central correlation (their Fig. 17
reports Pearson 0.79 between simplicity score and return) on a case where the
outcome is known to be a failure.

## B''. Run 18 — the live sweep, and a hyperparameter grid

✅ **Built 2026-08-13, not yet run.** Whatever survived offline now goes to a
live sweep on `inc_dec_learning_single`, against run 11's `BASELINE` and run
12's `act-x30` as the two reference points. Everything below is wired,
smoke-tested end to end, and waiting on cluster time.

| file | what it is |
|---|---|
| [`real_matd3/assume_arch_sweep.py`](real_matd3/assume_arch_sweep.py) | run 18a/b — both rounds on inc-dec, one runner |
| [`real_matd3/hpo_grid.py`](real_matd3/hpo_grid.py) | the 20-cell grid, shared by both hosts |
| [`real_matd3/optim_patches.py`](real_matd3/optim_patches.py) | weight decay, the one axis with no config field |
| [`cluster/critic_arch.sh`](cluster/critic_arch.sh) | 57 tasks (arch) / 60 (hpo) |
| [`cluster/hpo_eom.sh`](cluster/hpo_eom.sh) | 60 tasks — the same grid on `p1`, per Nash equilibrium |
| [`analysis/live_arch_film.py`](analysis/live_arch_film.py) | one panel per architecture, ordered by parameter count |
| [`analysis/hpo_grid_film.py`](analysis/hpo_grid_film.py) | one panel per hyperparameter cell, one figure per regime |

**Round 18a — architectures.** 19 cells: `baseline`, `late`, `split`, and both
families run at **two depths up the same four-rung parameter ladder** (100k /
500k / 2M / 8M). The six RSNorm carriers and `bn+late` are deliberately absent —
run 17 measured them at a mean `in_band` of 0.005 with the argmax pinned at
exactly 100.0 at every width, so re-running them buys 24 tasks of the same
answer.

The ladder is a **grid, not a line**, and that is the correction to run 17.
Run 17 moved width at fixed depth, so "capacity" and "width" were one variable
and neither could be blamed. Now: if the ladder has the same shape at depth 2
and depth 4, capacity is what matters; if the deeper column wins at equal
parameters, depth is. One SimBa block is two `Linear` layers, so the two
families' depth units differ — **read each family's curve against itself**, not
`d4` against `d4`.

`split` is the new architecture. Late injection gives the action its own weight
matrix, but it still arrives *raw* and outnumbered at layer 2. `split` gives it
its own hidden layer at the same width as the observation's, so from layer 2 on
neither can dominate by how many rows it owns. That separates **equal count**
from **equal scale** — and equal scale is exactly what `act_share` moved, so
this is the cell that decides whether run 12's lever has a literature-grounded
counterpart or was measuring something else.

**Rounds 18b / 18c — hyperparameters.** The same 20 cells on both hosts, so the
inc-dec and `p1` tables can be laid side by side: 3 learning rates ×
{const, linear, cosine}, batch size, policy delay, weight decay. A coordinate
sweep, not a 108-cell cross — nothing in runs 09–17 suggests these four
interact, and the failure they are aimed at is a critic that never develops a
slope. Learning rate and schedule *are* crossed, because a schedule is a
statement about the rate. `lr0.001-const` reproduces `default` and is the grid's
internal control.

**One thing worth knowing before reading any archived run.** `matd3.py:366` and
`:407` construct `AdamW(params, lr=...)` and pass nothing else, so **every run in
this benchmark's archive trained with torch's default `weight_decay = 0.01`** —
not with none. That is a real prior toward small weights on a critic fitting a
piecewise-constant reward, and it has never been varied. It is not a
`LearningConfig` field, so the sweep moves it by rebinding `matd3.AdamW`
(`optim_patches.py`); cells at 0.01 are left unpatched so their films stay
comparable with the run 14/15 batches. Adding the field upstream is a decision
for whoever reads the sweep.

`hpo_eom.sh` runs `OBS_REGIMES=1` by default. Run 14b found the critic fits the
regime it sees often and leaves the rare one as noise, so the question is not
"does a setting help" but **"does it help in the regime the critic is currently
ignoring"** — a setting that lifts the aggregate by fitting the common regime
harder is not an improvement for this benchmark's purpose.

```bash
bash cluster/critic_arch.sh                        # 18a, 57 tasks
ROUNDS=hpo bash cluster/critic_arch.sh             # 18b, 60 tasks
bash cluster/hpo_eom.sh                            # 18c, 60 tasks on p1
```

## C. Does any of this generalise? — plain EOM scenarios

✅ **Answered, and the answer changes the priority of A and B.** Everything in
runs 01–13 is one redispatch/multi-market scenario with a 90 %-flat reward, and
**none of the failure picture survives on an ordinary energy-only market**: run 14
solves all nine trials at an `act_share` an order of magnitude below run 13's
failing baseline (findings 10–12). So the thing being explained is *that reward
landscape*, not MATD3 and not `CriticTD3`'s input layout.

What that implies for the rest of the plan:

* **B is still worth doing, for a better reason than it was written for.**
  `act_share` is now known not to be a general predictor — run 14 falsifies it as
  one. Late action injection and SimBa are worth screening on the offline γ = 0
  harness on their own merits; "raise `act_share`" is not the mechanism.
* **Whatever B finds has a second bar to clear:** it must not *break* the plain
  EOM, which currently works. `cluster/eom_critic_evolution.sh` is the regression
  test — 9 tasks, ~15 min each.
* **A is unchanged and still first.** Run 14 predates it too.

**The per-regime critic films are DONE** — runs 14b, 14c and 15 all ran with
`OBS_REGIMES=1`, and findings 15 and 16 come out of them. What follows is kept
because it documents the three views and how to record them, not because
anything is outstanding. **One item is genuinely open: the two-bid ladder, at
the end of this section.**

* **The per-regime critic films.** Finding 12b says the learners solve the
  Bertrand equilibrium and not the pivotal one; it does **not** say whether the
  failure is in the critic (it never learns that the pivotal hours pay more) or
  in the actor (the critic knows, the actor cannot get there). Probed
  observations sampled without regard to demand cannot answer it.
  `eom_critic_film.py --obs-regimes` stratifies them by regime — the recorder
  ranks the buffer on the observation's scaled residual load, which is demand
  here. Three views then read it, and they answer different halves of the
  question:

  * `--only regime-heatmap` — the critic's **field** in each regime side by side
    on one shared colour scale, plus their difference. This is the one that
    distinguishes a critic failure from an actor failure: a peak can move while
    the field stays flat, and a flat field is a gradient the actor cannot climb
    however right the peak is.
  * `--only separation` — the same reduced to lines, actor bid and `argmax Q1`
    per regime against both equilibrium prices.
  * `--regime each --only film` — the full per-unit film, one PNG per regime.

  **One call to record it:**

  ```bash
  OBS_REGIMES=1 bash .../cluster/eom_critic_evolution.sh
  ```

  `sb02b` is the case that matters — it is the only one carrying both regimes,
  which 14b confirmed (bands found: `sb02a` idle+pivotal, `sb02b` all three,
  `sb02c` idle+bertrand).
* **STILL OPEN — the two-bid ladder** (`02a`–`02c`, act_dim 2), the A/B that
  isolates bid structure. 9 tasks, `CASES="02a 02b 02c"`. Every EOM finding so
  far is single-bid, so "one bid for the whole `max_power`" is an uncontrolled
  variable in all of them.

* **Port the probes to `example_02a`–`02c`** (`examples/inputs/example_02a` …
  `02c`). ✅ **Done** — `real_matd3/eom_critic_film.py` films the critics of the
  **learning units only** (1, 5 and 10 of them), `analysis/eom_critic_evolution.py`
  draws them, and `cluster/eom_critic_evolution.sh` runs the 3 × 3 batch in one
  call. Two things had to change and are worth knowing: `MultiAgentRecorder`
  refuses `act_dim != 1`, and `EnergyLearningStrategy` is **two** actions per unit
  (inflexible / flexible block), so the new recorder is generic in `act_dim` and
  takes `act_dim + 1` sweeps per agent — one per component plus the **diagonal**,
  which is the unit moving its whole bid and the only axis comparable with runs
  09–13. The offline harness is still single-bid-axis and is **not** ported.
* **The single-bid ladder is the one that runs by default.**
  `examples/inputs/example_02_single_bid/` is the same three fleets bidding with
  `EnergyLearningSingleBidStrategy` (`act_dim` 1, one bid for the whole
  `max_power`), as three study cases of one folder differing only in
  `powerplant_units`; demand, fuel prices, naive fleet and market are
  byte-identical to the originals, so each `sb02x` is an A/B against `02x`. At
  `act_dim` 1 the film has a single bid axis — the shape runs 09–13 recorded —
  and each unit submits one bid, which is exactly what the exploitability probe
  handles without its ordered-bids decomposition. **The strategy also defaults
  `foresight` to 24 against the two-bid strategy's 12**, so its observation is
  50-dimensional against 26: an `act_share` comparison *across* the two ladders
  moves two things at once. Within a ladder it is clean.
* **Bring exploitability tracking across from the `exploitability` branch.**
  ✅ **Done** (2026-08-11, ported not merged): `assume/reinforcement_learning/
  exploitability.py`, the `WriteOutput` hooks in `assume/common/outputs.py`, the
  22 unit tests and the walkthrough beside this file, and
  `examples/inputs/exploit_example/`. It needed a `world.py` change so evaluation
  episodes write market output at all — `RUNS_Continuation.md` has the three
  consequences for cluster runs.
  **It is only correct for a single day-ahead market with no storages**, so it
  can be read on `example_02a`–`02c` and **not** on any inc-dec scenario:
  `_write_exploitability` selects the orderbook without filtering by market, so
  EOM and Redispatch bids are grouped by `start_time` and cleared as if they were
  one auction, and even filtered to the EOM a unit's day-ahead profit is not its
  total profit. Storages are scored by a volume-preserving Tier-1 rule that holds
  the SoC path fixed, which is a lower bound and not comparable with the thermal
  units it would be averaged against.
  ✅ **It works end to end on a live learning run** (run 14): 19 evaluation
  episodes per trial, 96 k–226 k rows, no gaps, monotone decreasing in all nine,
  and it separates market power from bad play where the bid level alone cannot
  (finding 12). Read it **learning units only** — the naive fleet is exploitable
  by construction and dilutes the average — and **per MW**, since the scored units
  differ 5× in capacity. Ratios to `best_profit` do not work: summed over
  timesteps the denominator goes small or negative.
* **Run 13's open equilibrium question is now explicitly out of reach.** Fleet
  reward moves *opposite* to `diesel_0`'s in every run 13 condition, so nothing
  there distinguishes "learned to bid better" from "competed the price down", and
  an exploitability metric is exactly the missing measurement — but the metric as
  written **cannot be read on that scenario at all** (two markets; see above).
  Answering it needs either a market-filtered, multi-market-aware exploitability
  or an argument from the 02x examples by analogy. Do not close the question with
  a number off `rl_exploitability` on an inc-dec run.

## Parked

Cheap, still open, not on the critical path:

- Where `act_share` saturates (the ladder jumps 0.234 → 0.479 with a bimodal point
  between), and whether `act-own-x30` continues run 13's ladder. Only worth doing
  if workstream B leaves `act_share` standing.
- Does `foresight-3` solve it with more episodes? It was still descending at the
  budget's end (`pulled left` 100 %, bid 40.4); 128 episodes settles it.
- Why the single- and multi-agent budgets differ so much — more transitions per
  episode (62 vs 14) and a non-stationary opponent set, neither isolated.
- Fix the shaping decoy: make the shaped branch continuous with the true reward at
  both band edges so both ramps point *into* the band. Superseded in priority by B,
  which aims to remove the need for shaping entirely.
- Does surrogate TD3 at `--policy-delay 1` match DDPG?

## Layout and tests

```
rl_benchmark/
├── HANDOFF.md  RUNS.md  RUNS_Continuation.md
├── archive/         the full originals — grep only
├── _layout.py       sys.path + OUT_DIR + resolve(); every script imports it
├── surrogate/       the closed-form landscape and the Gymnasium env
├── sweeps/          training drivers: run_benchmark.py, td3_stability.py
├── analysis/        reads a recorded run and explains it; makes the figures
│                    (simplicity_bias.py is the exception — it needs no run)
├── real_matd3/      probes ASSUME's own MATD3 (07–12 single, 13 multi-agent,
│                    eom_critic_film.py for example_02a–c,
│                    critic_architectures.py for workstream B,
│                    assume_arch_sweep.py + hpo_grid.py + optim_patches.py
│                    for run 18)
├── cluster/         one-call SLURM launchers; see its README
│                    (`analysis/eom_exploitability.py` reads the scratch
│                     databases, not the .npz films — it is the only script
│                     here that does)
├── test_rl_benchmark.py
├── test_exploitability.py               22 unit tests of the clearing /
│                                        exploitability maths (workstream C)
└── exploitability_two_bid_walkthrough.py  why the search is exhaustive rather
                                          than a heuristic; prints its own
                                          derivation, needs no data
```

The exploitability code itself lives in `assume/` — `reinforcement_learning/
exploitability.py` and the `WriteOutput` hooks in `common/outputs.py`. Only its
test and its explanation are kept here. **The measure is only correct for a
single day-ahead market with no storages** — the SCOPE note at the top of
`exploitability.py` says why, and it rules out every inc-dec scenario in this
archive.

Anything cluster-bound goes through [`cluster/`](cluster/), one file per
experiment, each its own submitter, array body and collector:

```bash
bash .../rl_benchmark/cluster/rerun_run13.sh            # 18 tasks + tarball
bash .../rl_benchmark/cluster/eom_critic_evolution.sh   #  9 tasks + tarball
bash .../rl_benchmark/cluster/critic_arch.sh            # 57 tasks + tarball
bash .../rl_benchmark/cluster/hpo_eom.sh                # 60 tasks + tarball
```

Every script runs from any working directory and resolves archived runs
automatically, so figures redraw with no arguments. `sweeps/run_benchmark.py`
owns the house palette; `analysis/critic_coherence.py` owns the
observation-disagreement statistics runs 10–13 share.

```bash
R=examples/inputs/2_nodes_paper_small/rl_benchmark
conda run -n assume python -m pytest $R/test_rl_benchmark.py $R/test_exploitability.py -v
```

~30 s, no simulation, no archive, and both run from any working directory.
`test_rl_benchmark.py` covers things that would fail *silently* — the figures
would keep rendering from the wrong input: `MultiAgentRecorder` against a real
`TD3.update_policy` gradient step (**extend this when workstream A lands**);
the run-13 action-scale lever; `act_share`; the coherence statistic and the
per-episode transition count. Its § 5 adds workstream B: every critic variant
is a drop-in for `CriticTD3`, `q1_forward` agrees with `forward`, running
statistics survive `polyak_update`, and the Fourier measure recovers a known
frequency (which is what catches the DC term being left in — an un-centred
image scores every architecture as near-zero complexity, in the right order).
Its § 6 adds run 18: the depth knob really moves layers and leaves the width
alone, the parameter ladder lands on its rungs at both depths in both families,
every generated ladder name is registered, `split`'s two encoders are the same
width, the hyperparameter grid is the grid it documents, and the weight-decay
patch actually reaches the optimizer (without it the `wd` cells would run
happily and measure torch's default three times over).
`test_exploitability.py` adds 22 pure unit tests of
the clearing and exploitability maths — no scenario, no database. (Its `_main()`
below the tests is a scratch driver for plotting against a Postgres run; it is not
collected by pytest and is partly dead code.)

Commands for every run are in `RUNS.md`. Results always write to the **outputs**
folder, never the tracked input folder.
