# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Gymnasium environment wrapping the inc-dec reward landscape.

The environment is deliberately a *contextual bandit* dressed as an MDP, because
that is what the ASSUME bidding problem is: the action is a bid price, the reward
is that hour's profit, and the agent's bid does not move the next hour's state.
Keeping the MDP shell means any off-the-shelf continuous-control algorithm runs
on it unchanged.

Two things are mirrored from ``EnergyLearningSingleBidRedispatchStrategy`` so that
results transfer back to the real simulation:

* the action is a single tanh output in ``[-1, 1]``, scaled to a bid price by
  ``max_bid_price`` -- so the agent can reach ``[-100, +100]`` EUR/MWh;
* the observation is a 74-vector laid out like ``create_observation``:
  24 residual-load forecast, 24 price forecast, 24 price history, 2 unit-specific.
  Only a handful of those entries carry signal, exactly as in the real scenario.

Usage
-----
>>> from incdec_env import IncDecEnv
>>> env = IncDecEnv()
>>> obs, _ = env.reset(seed=0)
>>> obs, reward, terminated, truncated, info = env.step([0.30])
>>> round(reward, 3), round(info["bid"], 2)
(0.19, 30.0)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from incdec_reward import PAPER_SMALL, IncDecParams, profits_from_bid, reward_from_bid

__all__ = ["IncDecEnv"]


class IncDecEnv(gym.Env):
    """Bid one price per hour into the inc-dec landscape.

    Parameters
    ----------
    params:
        The landscape, or a list of landscapes to cycle through within an episode
        (a contextual variant -- each "hour" then has its own clearing price). The
        observation encodes the price forecast and marginal cost, both of which
        the real strategy observes; ``dec_threshold`` stays hidden, as it is in
        the real scenario.
    episode_length:
        Steps per episode. Defaults to 24, one simulated day.
    obs_noise:
        Standard deviation of Gaussian noise added to the observation, in scaled
        units. Zero by default.
    foresight:
        Forecast window length, driving the observation dimension
        (``3 * foresight + 2``). Defaults to 24, i.e. ``obs_dim = 74``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        params: IncDecParams | list[IncDecParams] = PAPER_SMALL,
        episode_length: int = 24,
        obs_noise: float = 0.0,
        foresight: int = 24,
    ):
        super().__init__()

        self.contexts = [params] if isinstance(params, IncDecParams) else list(params)
        self.episode_length = episode_length
        self.obs_noise = obs_noise
        self.foresight = foresight

        self.max_bid_price = self.contexts[0].max_bid_price
        if any(c.max_bid_price != self.max_bid_price for c in self.contexts):
            raise ValueError("all contexts must share the same max_bid_price")

        self.obs_dim = 3 * foresight + 2
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        self._t = 0

        #: Every bid actually placed, in EUR/MWh, in order -- the behaviour policy
        #: including exploration noise. Survives ``reset``; use ``clear_history``.
        self.bid_history: list[float] = []
        #: The reward earned for each of those bids.
        self.reward_history: list[float] = []

    def clear_history(self) -> None:
        """Drop the recorded bid/reward history."""
        self.bid_history.clear()
        self.reward_history.clear()

    # ------------------------------------------------------------------ helpers

    def _params_at(self, t: int) -> IncDecParams:
        return self.contexts[t % len(self.contexts)]

    def observation_for(self, params: IncDecParams) -> np.ndarray:
        """Build the 74-vector the real strategy would hand the actor.

        Layout matches ``TorchLearningStrategy.create_observation``: forecast
        residual load, forecast price, price history, then the unit-specific
        entries (scaled marginal cost and availability).
        """
        scaled_price = params.eom_price / params.max_bid_price
        # residual load is not directly observable as a price, but it is what
        # drives the clearing price -- encode it as a monotone proxy so a
        # contextual agent has something to condition on.
        scaled_res_load = scaled_price

        obs = np.concatenate(
            [
                np.full(self.foresight, scaled_res_load),
                np.full(self.foresight, scaled_price),
                np.full(self.foresight, scaled_price),
                [params.marginal_cost / params.max_bid_price, 1.0],
            ]
        ).astype(np.float32)

        if self.obs_noise:
            obs = obs + self.np_random.normal(0.0, self.obs_noise, obs.shape).astype(
                np.float32
            )
        return obs

    # --------------------------------------------------------------- gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._t = 0
        return self.observation_for(self._params_at(self._t)), {}

    def step(self, action):
        params = self._params_at(self._t)

        raw = float(np.clip(np.asarray(action, dtype=float).ravel()[0], -1.0, 1.0))
        bid = raw * params.max_bid_price
        reward = float(reward_from_bid(bid, params))
        eom_profit, dec_profit = profits_from_bid(bid, params)

        self.bid_history.append(bid)
        self.reward_history.append(reward)

        info = {
            "bid": bid,
            "eom_profit": float(eom_profit),
            "dec_profit": float(dec_profit),
            "regret": params.optimal_reward - reward,
            "dispatched": bid <= params.eom_price,
            "dec_d": params.dec_threshold <= bid <= params.eom_price,
        }

        self._t += 1
        truncated = self._t >= self.episode_length
        obs = self.observation_for(self._params_at(self._t))

        return obs, reward, False, truncated, info
