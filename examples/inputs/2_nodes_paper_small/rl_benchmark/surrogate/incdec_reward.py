# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Analytic surrogate of the inc-dec reward landscape.

This reproduces, in closed form, the curve that ``reward_landscape.png`` measures
by sweeping the real EOM + redispatch clearing for the learning diesel unit of the
``2_nodes_paper_small`` scenario. It exists so that RL algorithms can be compared
on *this landscape shape* without paying for a HiGHS solve per candidate action.

The landscape
-------------
A single northern generator with marginal cost ``mc`` bids a price ``b`` into a
pay-as-clear EOM and is then exposed to a pay-as-bid redispatch (dec) market. The
rest of the fleet is fixed, which makes three things constant from the agent's
point of view: the EOM clearing price ``p_eom`` set by the competing fleet, the
dec price of the marginal competing northern unit ``p_dec``, and the volume ``q``.

That gives three regimes::

    b > p_eom              -> not dispatched at all           -> reward 0
    p_dec <= b <= p_eom    -> dispatched, then dec'd          -> reward > 0
    b < p_dec              -> dispatched, NOT dec'd           -> reward < 0 (flat)

    eom_profit = (p_eom - mc) * q       (only if dispatched; negative, mc > p_eom)
    dec_profit = (mc - b)   * q         (only if dispatched and b >= p_dec)
    reward     = (eom_profit + dec_profit) / (max_bid_price * q)

The dec leg is what pays: under pay-as-bid dec the unit buys its energy back at
its own bid ``b`` and saves ``mc``, so a *higher* bid means the system operator
prefers to dec this unit -- below ``p_dec`` the competing northern units are dec'd
instead and the agent is left running at a loss.

Why it is hard
--------------
The optimum sits exactly on a cliff edge at ``b = p_dec``, one tick above a
discontinuity of depth ``(mc - p_dec) * q``. Everything above ``p_eom`` is a zero
plateau with no gradient, and the only informative slope spans
``p_eom - p_dec`` of the ``2 * max_bid_price`` the tanh actor can reach.
For the default parameters that is 19 EUR out of 200 -- 9.5% of the action space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "IncDecParams",
    "PAPER_SMALL",
    "profits_from_bid",
    "reward_from_bid",
    "reward_from_action",
    "sweep",
]


@dataclass(frozen=True)
class IncDecParams:
    """Everything the agent cannot influence, held fixed by the rest of the fleet.

    Attributes
    ----------
    marginal_cost:
        Marginal cost of the learning unit in EUR/MWh.
    eom_price:
        EOM clearing price set by the competing fleet in EUR/MWh. The agent is
        dispatched only while it bids at or below this.
    dec_threshold:
        Price of the marginal competing dec offer in the agent's zone, in
        EUR/MWh. The agent is selected for downward redispatch only at or above
        this.
    volume:
        Dispatched volume in MW -- ``max_power`` of the unit.
    max_bid_price:
        Action scaling. A tanh actor output of ``a`` becomes a bid of
        ``a * max_bid_price``, so the reachable bid range is
        ``[-max_bid_price, +max_bid_price]``.
    """

    marginal_cost: float = 66.0
    eom_price: float = 49.0
    dec_threshold: float = 30.0
    volume: float = 1000.0
    max_bid_price: float = 100.0

    @property
    def reward_scale(self) -> float:
        """``1 / (max_bid_price * max_power)``, as in ``calculate_redispatch_reward``."""
        return 1.0 / (self.max_bid_price * self.volume)

    @property
    def optimal_bid(self) -> float:
        """The cliff edge -- the highest-paying bid, in EUR/MWh."""
        return self.dec_threshold

    @property
    def optimal_reward(self) -> float:
        """Reward at ``optimal_bid``. Reduces to ``(p_eom - p_dec) / max_bid_price``."""
        return reward_from_bid(self.optimal_bid, self)

    @property
    def cliff_depth(self) -> float:
        """Reward lost by bidding one tick below the optimum."""
        return self.optimal_reward - reward_from_bid(self.dec_threshold - 1e-9, self)

    @property
    def signal_width(self) -> float:
        """Width of the informative (non-flat) region as a share of the action space."""
        return (self.eom_price - self.dec_threshold) / (2 * self.max_bid_price)


#: Parameters read off the measured sweep for ``diesel_0`` in ``2_nodes_paper_small``.
PAPER_SMALL = IncDecParams()


def profits_from_bid(
    bid: float | np.ndarray, params: IncDecParams = PAPER_SMALL
) -> tuple[np.ndarray, np.ndarray]:
    """Split the reward into its EOM and redispatch legs, in EUR.

    Returns
    -------
    (eom_profit, dec_profit)
        Arrays shaped like ``bid``. Useful for reproducing the lower-right panel
        of ``reward_landscape.png``.
    """
    bid = np.asarray(bid, dtype=float)

    dispatched = bid <= params.eom_price
    dec_d = dispatched & (bid >= params.dec_threshold)

    eom_profit = np.where(
        dispatched, (params.eom_price - params.marginal_cost) * params.volume, 0.0
    )
    dec_profit = np.where(dec_d, (params.marginal_cost - bid) * params.volume, 0.0)

    return eom_profit, dec_profit


def reward_from_bid(
    bid: float | np.ndarray, params: IncDecParams = PAPER_SMALL
) -> float | np.ndarray:
    """Reward for a bid in EUR/MWh, scaled exactly as the ASSUME strategy scales it."""
    eom_profit, dec_profit = profits_from_bid(bid, params)
    reward = (eom_profit + dec_profit) * params.reward_scale
    return float(reward) if np.isscalar(bid) or np.ndim(bid) == 0 else reward


def reward_from_action(
    action: float | np.ndarray, params: IncDecParams = PAPER_SMALL
) -> float | np.ndarray:
    """Reward for a raw actor output in ``[-1, 1]``."""
    action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    return reward_from_bid(action * params.max_bid_price, params)


def sweep(
    params: IncDecParams = PAPER_SMALL, n: int = 4001
) -> tuple[np.ndarray, np.ndarray]:
    """Dense (bid, reward) sweep over the full reachable bid range, for plotting."""
    bids = np.linspace(-params.max_bid_price, params.max_bid_price, n)
    return bids, reward_from_bid(bids, params)


if __name__ == "__main__":
    p = PAPER_SMALL
    print(f"optimal bid      {p.optimal_bid:.2f} EUR/MWh")
    print(f"optimal reward   {p.optimal_reward:+.3f}")
    print(f"cliff depth      {p.cliff_depth:.3f}")
    print(f"signal width     {p.signal_width:.1%} of the action space")
    for bid in (-100, 0, 29.95, 30.0, 40.0, 49.0, 60.0, 100.0):
        print(f"  bid {bid:>7.2f} -> reward {reward_from_bid(bid, p):+.3f}")
