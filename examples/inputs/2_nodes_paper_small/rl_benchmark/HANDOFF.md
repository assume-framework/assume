# Handoff — inc-dec RL benchmark

Read this to resume in a fresh session. The detail is deliberately not in here.

| file | what it is | read it |
|---|---|---|
| **`HANDOFF.md`** (this) | the state of the question and the forward plan | always, first |
| [`RUNS.md`](RUNS.md) | runs 01–13 — one block per run, its command, its numbers | for a specific run |
| [`RUNS_Continuation.md`](RUNS_Continuation.md) | **new runs go here**, incl. every cluster run | when adding a run |
| [`archive/`](archive/) | the full-length originals, 2 600 lines | ⚠️ **`grep` only** — reading them costs ~35 k tokens |

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

⚠️ **A fresh clone has no starting buffer.**
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
   is real. ⚠️ **But `act_share` is my own quantity, not a literature one**, and
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

## Refuted or revised — do not re-test

- ❌ "The critic smooths the cliff into a ramp." It doesn't; the critics learn it.
- ❌ "A 1e-5 gradient is too small to move the actor." Adam is scale-invariant.
- ⚠️ "`policy_delay` fixes it." Raising it does not (8 → 0/8 with softsign) — but
  the knob is not inert; run 04 turned it the **wrong way** (finding 3). Lowering
  it is untested in the surrogate and 0/3 on ASSUME at 800 updates.
- ❌ "The surrogate's `lr-1e-4` transfers to ASSUME." 95.5 ± 0.3, 0/3 (run 11).
- ❌ "The real critic fails because it must fit `Q(s,a)` across 548 observations."
  Run 12 has the controls: **shuffled** observations fail identically, and a critic
  that memorises harder learns the band correctly. The observation *count* stood in
  for the input *dimension* count.
- ⚠️ "Without shaping or a raised `act_share`, MATD3 never forms a preference" —
  single-agent and budget-dependent (finding 7).
- ⚠️ "Any single result here is a sample from a bimodal distribution, because BLAS
  thread count alone flipped a seed." **True of the SB3 surrogate, not of ASSUME**
  — runs 10 and 13 reproduce bit-identically across thread-count changes.

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

## Caveats on the current results

Two of these are the reason workstream A exists.

- ⚠️ **Run 13's recorded critic field is not matd3's actor objective.**
  `matd3.py:704` holds the other agents at their **stored** actions; the recorder
  holds them at their **current actors'** greedy outputs. A valid critic slice, but
  the window / `pulled left` / coherence readings describe the response to the
  current joint policy, not the gradient the actor actually climbs. Empty at N = 1,
  so runs 09–12 are untouched. `RUNS.md` correction 17.
- ⚠️ **The evaluation database holds two hours per episode** (10:00 and 11:00 of
  14), an unflushed async write at shutdown. Best-policy selection, early stopping
  and any reward read from `rl_params` are an early-hours sample; training is
  unaffected. `RUNS.md` correction 16.
- ⚠️ **The surrogate is not the scenario's reward** — `reward_from_bid` agrees with
  the frozen buffer's stored rewards on **24.8 %** of transitions. Bids, critics,
  `act_share` and the offline fits stand; every reconstructed reward, `regret`, the
  `+0.15 solved` bar and the `32.31` optimum do not. Run 12's headline survives on
  *measured* reward. Do not retune the surrogate — it is exact for runs 01–08 by
  construction. Correction 15.
- ⚠️ **Run 13 used a working-tree `inc_dec_learning`** (72 h horizon, lr 1e-4, 50
  episodes, `train_freq` 12h), table in `RUNS.md` §3. **Restore or commit it
  before re-running.**
- **Seed counts are small** (3 per condition for runs 11–13, 6 for run 10). They
  reject easy fixes; they are not success rates.
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
   sample.

Optionally also fix the two `ReplayBuffer` defects (§caveats) while re-running —
untriggered at these buffer sizes, but a longer cluster run is exactly where the
early-`full` wrap would start returning zero rows.

## B. Literature-grounded critic changes, instead of `act_share`

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
   LayerNorms. **Run RSNorm alone, RSNorm + late injection, and full SimBa on the
   offline γ = 0 harness** (`real_matd3/assume_offline_critic.py`, 5 seeds, ~15 min,
   no simulation) before spending any cluster time. That harness already
   reproduces the live failure, which is what makes it the right place to screen
   architectures.

Then take whatever survives offline to a live sweep on `inc_dec_learning_single`
and `inc_dec_learning`, against run 11's `BASELINE` and run 12's `act-x30` as the
two reference points.

## C. Does any of this generalise? — plain EOM scenarios

Everything above is one redispatch/multi-market scenario with a 90 %-flat reward.
Nothing says the conclusions survive on an ordinary energy-only market.

* **Port the probes to `example_02a`–`02c`** (`examples/inputs/example_02a` …
  `02c`). The recorder and the offline harness both need the scenario only through
  `create_observation` and `CriticTD3`, so the work is in the study-case config and
  the buffer collection, not in the analysis.
* **Bring exploitability tracking across from the `exploitability` branch**, which
  carries `assume/reinforcement_learning/exploitability.py`,
  `examples/test_exploitability.py`, `examples/exploitability_two_bid_walkthrough.py`
  and `examples/inputs/exploit_example/`. Cherry-pick those onto the working branch
  rather than merging the branch — its diff against `main` is ~176 files and
  deletes test fixtures.
* **This is also the answer to run 13's open equilibrium question.** Fleet reward
  moves *opposite* to `diesel_0`'s in every run 13 condition, so nothing there
  distinguishes "learned to bid better" from "competed the price down". An
  exploitability metric is exactly the missing measurement, and it is worth having
  on the 02x examples before trying to read it on the 11-agent redispatch case.

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
├── real_matd3/      probes ASSUME's own MATD3 (07–12 single, 13 multi-agent)
└── test_rl_benchmark.py
```

Every script runs from any working directory and resolves archived runs
automatically, so figures redraw with no arguments. `sweeps/run_benchmark.py`
owns the house palette; `analysis/critic_coherence.py` owns the
observation-disagreement statistics runs 10–13 share.

```bash
conda run -n assume python -m pytest \
    examples/inputs/2_nodes_paper_small/rl_benchmark/test_rl_benchmark.py -v
```

~8 s, no simulation, no archive. Four groups, all covering things that would fail
*silently* — the figures would keep rendering from the wrong input:
`MultiAgentRecorder` against a real `TD3.update_policy` gradient step (**extend
this when workstream A lands**); the run-13 action-scale lever; `act_share`; the
coherence statistic and the per-episode transition count.

Commands for every run are in `RUNS.md`. Results always write to the **outputs**
folder, never the tracked input folder.
