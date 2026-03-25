import os
import tempfile

import numpy as np
import torch as th

from assume.reinforcement_learning.buffer import ReplayBuffer


def test_replay_buffer_save_load_preserves_shapes_and_metadata():
    # create a small buffer
    buffer_size = 10
    n_rl_units = 2
    obs_dim = 3
    act_dim = 1

    rb = ReplayBuffer(
        buffer_size=buffer_size,
        obs_dim=obs_dim,
        act_dim=act_dim,
        n_rl_units=n_rl_units,
        device="cpu",
        float_type=th.float32,
    )

    # populate with some fake transitions
    for i in range(5):
        obs = np.ones((1, n_rl_units, obs_dim), dtype=rb.np_float_type) * i
        acts = np.ones((1, n_rl_units, act_dim), dtype=rb.np_float_type) * (i + 0.1)
        # ReplayBuffer.add expects rewards with trailing singleton dimension
        # (len, n_rl_units, 1) so that np.squeeze(..., axis=-1) yields
        # shape (len, n_rl_units).
        rews = np.ones((1, n_rl_units, 1), dtype=rb.np_float_type) * (i + 0.5)
        rb.add(obs, acts, rews)

    # set some internal state
    rb.pos = 3
    rb.full = False

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "replay_buffer.npz")
        rb.save(path)

        # load back
        loaded = ReplayBuffer.load(path, device="cpu", float_type=th.float32)

        # assert shapes
        assert loaded.observations.shape == rb.observations.shape
        assert loaded.actions.shape == rb.actions.shape
        assert loaded.rewards.shape == rb.rewards.shape

        # assert metadata
        assert loaded.pos == rb.pos
        assert loaded.full == rb.full
