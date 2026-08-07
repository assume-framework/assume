# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Regenerate every table in ``actor_saturation.md``.

Run this rather than trusting the transcribed numbers::

    python actor_saturation.py

All values are float32 -- the dtype the actor networks actually train in -- and
the derivatives come from ``torch.autograd``, so they are exactly what
``backward()`` would propagate, including any underflow to zero.
"""

from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F

#: torch's Adam default; SB3 does not override it for TD3/DDPG/SAC.
ADAM_EPS = 1e-8
#: |dQ/da| measured at the actor's own action on the inc-dec landscape.
DQ_DA = 3.4e-5

SQUASH = {"tanh": th.tanh, "softsign": F.softsign}


def derivative(fn, z: float) -> float:
    """f'(z) in float32, via autograd -- zero if the forward pass saturated."""
    t = th.tensor([z], dtype=th.float32, requires_grad=True)
    SQUASH[fn](t).backward()
    return t.grad.item()


def saturation_threshold(fn, hi: float = 1e9) -> float:
    """Smallest z for which float32 f(z) rounds to exactly 1.0."""
    lo = 1.0
    for _ in range(300):
        mid = (lo + hi) / 2
        if SQUASH[fn](th.tensor([mid], dtype=th.float32)).item() == 1.0:
            hi = mid
        else:
            lo = mid
    return hi


def adam_step(g: float) -> float:
    """Step as a fraction of lr for a consistently-signed gradient of size g."""
    return g / (g + ADAM_EPS)


def main() -> None:
    print("\n## 2. Adam step vs gradient magnitude (eps = 1e-8)\n")
    print(f"  {'G':>10} {'step':>10}")
    for g in (1e-2, 1e-5, 1e-7, 1e-8, 1e-9, 0.0):
        print(f"  {g:>10.0e} {adam_step(g):>9.1%}")

    print("\n## 3. float32 saturation threshold\n")
    for fn in SQUASH:
        print(f"  {fn:<10} f(z) == 1.0 exactly for z >= {saturation_threshold(fn):.4g}")
    print(f"  (ulp below 1.0 in float32 is 2^-24 = {2**-24:.3e})")

    print("\n## 4. da/dz\n")
    print(f"  {'z':>6} {'tanh':>13} {'softsign':>13} {'ratio':>10}")
    for z in (2, 5, 10, 20, 50, 100, 1000):
        t, s = derivative("tanh", z), derivative("softsign", z)
        ratio = "inf" if t == 0 else f"{s / t:.1f}x"
        print(f"  {z:>6} {t:>13.3e} {s:>13.3e} {ratio:>10}")

    print(f"\n## 5. What reaches the weights, with |dQ/da| = {DQ_DA:.1e}\n")
    print(
        f"  {'z':>6} {'tanh: G':>12} {'step':>8}   {'softsign: G':>13} {'step':>8}"
    )
    for z in (5, 10, 20, 50, 100, 1000):
        gt = DQ_DA * derivative("tanh", z)
        gs = DQ_DA * derivative("softsign", z)
        print(
            f"  {z:>6} {gt:>12.2e} {adam_step(gt):>7.1%}   "
            f"{gs:>13.2e} {adam_step(gs):>7.1%}"
        )

    # The asymptotic approximations quoted in the prose, checked against reality.
    print("\n  approach-rate check (prose in section 3):")
    for z in (5.0, 9.0, 12.0):
        exact_t = 1 - np.tanh(z)
        exact_s = 1 - z / (1 + z)
        print(
            f"    z={z:<5} 1-tanh: {exact_t:.3e} vs 2e^-2z = {2 * np.exp(-2 * z):.3e}"
            f"   |  1-softsign: {exact_s:.3e} vs 1/(1+z) = {1 / (1 + z):.3e}"
        )


if __name__ == "__main__":
    main()
