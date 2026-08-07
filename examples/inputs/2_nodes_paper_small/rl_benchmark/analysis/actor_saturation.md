# Actor output saturation: tanh vs softsign

Why the choice of output squashing decides whether a deterministic-policy actor
(DDPG / TD3 / MATD3) can still move, and why a "small" gradient is usually not
the problem.

All numbers below are measured in **float32**, the dtype these networks actually
train in. Reproduce with `python actor_saturation.py`.

---

## 1. Where the actor's update comes from

The actor is trained by ascending the critic: `L = -Q(s, pi(s))`. With the policy
written as `a = f(z)`, where `z` is the last linear layer's output and `f` the
squashing function, the chain rule gives

```
dL/dtheta  =  dQ/da  ·  f'(z)  ·  dz/dtheta
              ^^^^^     ^^^^^
              critic    squashing
```

Three separate things can make this vanish, and they are usually confused:

| factor | vanishes when | can Adam compensate? |
|---|---|---|
| `dQ/da` | the critic is flat at `a` | **yes** — see §2 |
| `f'(z)` | the actor's output is saturated | **no** if it reaches exactly 0 |
| `dz/dtheta` | dead ReLUs upstream | no |

## 2. Adam makes gradient *magnitude* almost irrelevant

Adam steps by `lr · m̂ / (sqrt(v̂) + eps)`. For a consistently-signed gradient of
magnitude `G`, both moments settle near `G`, so the step is

```
step  =  lr · G / (G + eps)          eps = 1e-8 (torch default, unchanged by SB3)
```

which is scale-**in**variant while `G >> eps`:

| `G` | step, as a fraction of `lr` |
|---|---|
| 1e-2 | 100.0 % |
| 1e-5 | 99.9 % |
| 1e-7 | 90.9 % |
| 1e-8 | 50.0 % |
| 1e-9 | 9.1 % |
| **0** | **0 %** |

So `dQ/da = 1e-5` is *not* a small gradient — it still yields full-size steps.
Damping only begins within about two orders of magnitude of `eps`, and only a
**hard zero** is unrecoverable. This is why "the critic is flat, so the actor is
stuck" is not by itself a valid inference.

**Caveat (the important one):** this argument requires the sign to be *consistent*
over many consecutive updates. A gradient that reverses direction gets its Adam
moment `m̂` cancelled and produces small steps no matter how large `G` is. See §6.

## 3. Why tanh dies and softsign does not

Both map to `(-1, 1)`. What differs is **how fast they approach the asymptote**,
and that single property sets both the saturation threshold and the derivative.

|  | tanh | softsign |
|---|---|---|
| definition | `(e^z - e^-z)/(e^z + e^-z)` | `z / (1 + |z|)` |
| distance to the asymptote | `1 - f(z) ≈ 2·e^(-2z)` — **exponential** | `1 - f(z) = 1/(1+z)` — **polynomial** |
| derivative | `1 - f(z)²` | `1/(1+|z|)²` |
| derivative decay | exponential, `≈ 4·e^(-2z)` | polynomial, `~ z^-2` |

The derivative is the killer, but the *cause* is the approach rate. float32 has a
24-bit mantissa, so a value rounds to exactly `1.0` once it is within
`2^-25 ≈ 2.98e-8` of it. Solving each row above for that threshold:

* tanh: `2·e^(-2z) < 2.98e-8`  ->  **z > 9.01**
* softsign: `1/(1+z) < 2.98e-8`  ->  **z > 3.4e7**

The tanh estimate is exact (`1-tanh(9) = 3.046e-08`, matching `2·e^(-2z)` to four
figures). The softsign estimate is within a factor of ~1.6 of the measurement
below, the difference coming from float32 rounding inside the division itself.
Measured by bisection:

| | float32 `f(z) == 1.0` exactly for |
|---|---|
| **tanh** | **z ≥ 9.011** |
| **softsign** | **z ≥ 2.1e7** |

A factor of **2.3 million** in how far the pre-activation can drift before the
output pins. And once `f(z)` rounds to exactly `1.0`, autograd evaluates
`1 - f(z)²` as `1 - 1 = 0` — the gradient is not merely small, it is
*identically zero*, and every upstream parameter gradient is zeroed with it.

## 4. The gradient at a given pre-activation

`da/dz`, float32:

| `z` | tanh | softsign | ratio |
|---:|---|---|---|
| 2 | 7.065e-02 | 1.111e-01 | 1.6× |
| 5 | 1.816e-04 | 2.778e-02 | 153× |
| **10** | **0** | 8.264e-03 | ∞ |
| 20 | **0** | 2.268e-03 | ∞ |
| 50 | **0** | 3.845e-04 | ∞ |
| 100 | **0** | 9.803e-05 | ∞ |
| 1000 | **0** | 9.979e-07 | ∞ |

## 5. Combining the two: what actually reaches the weights

Taking the measured critic gradient on the inc-dec landscape, `|dQ/da| ≈ 3.4e-5`,
and applying §2's `G/(G+eps)`:

| `z` | tanh: `G` | Adam step | softsign: `G` | Adam step |
|---:|---|---|---|---|
| 5 | 6.17e-09 | 38.2 % | 9.44e-07 | 99.0 % |
| **10** | **0** | **0 %** | 2.81e-07 | 96.6 % |
| 20 | **0** | **0 %** | 7.71e-08 | 88.5 % |
| 50 | **0** | **0 %** | 1.31e-08 | 56.7 % |
| 100 | **0** | **0 %** | 3.33e-09 | 25.0 % |
| 1000 | **0** | **0 %** | 3.39e-11 | 0.3 % |

A softsign actor keeps taking near-full-speed steps in exactly the regime where a
tanh actor has already stopped permanently. Softsign degrades gracefully; tanh
falls off a cliff at `z ≈ 9`.

## 6. Necessary, not sufficient

A live gradient only helps if the actor can follow it *somewhere useful*. Two
distinct consistency requirements, both needed:

* **temporal** — the sign at the actor's own action must persist across many
  consecutive updates, or Adam's momentum cancels;
* **spatial** — the sign must persist *along the whole path* from where the actor
  is to where it should go. If `dQ/da` reverses partway, the actor descends into
  the nearest stationary point of the critic and stops there, however alive its
  gradient is.

On this landscape the spatial condition is the binding one: see the heatmaps in
`critic_evolution.png`, where the field between bid 100 and the optimum at 30 is
mottled rather than uniformly signed, and §7 of the run archive for the measured
sign-reversal counts.

## 7. Who uses what

| | output squashing |
|---|---|
| **SB3** `TD3`/`DDPG` | `nn.Tanh()`, hardcoded via `create_mlp(..., squash_output=True)` — no constructor argument |
| **ASSUME** `Actor` | `softsign` by default ([`neural_network_architecture.py:131`](../../../../assume/reinforcement_learning/neural_network_architecture.py#L131)), with `tanh`/`sigmoid`/`relu` selectable |

This means the TD3/DDPG failures recorded in runs 01–04 of the archive are **partly
an artifact of SB3's tanh actor** and do not transfer unqualified to ASSUME's
MATD3. `run_benchmark.py --actor-activation softsign` swaps SB3's output layer to
make the comparison directly; run 05 in the archive reports the result.

If you switch ASSUME's `Actor.activation` to `"tanh"`, expect this failure mode.
