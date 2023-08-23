import logging

import torch as th
from torch.nn import functional as F
from torch.optim import Adam

logger = logging.getLogger(__name__)

from assume.reinforcement_learning.algorithms.base_algorithm import RLAlgorithm
from assume.reinforcement_learning.learning_utils import CriticTD3 as Critic
from assume.reinforcement_learning.learning_utils import polyak_update


class TD3(RLAlgorithm):
    def __init__(
        self,
        learning_role,
        learning_rate=1e-4,
        learning_starts=100,
        batch_size=1024,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=-1,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
    ):
        self.learning_role = learning_role
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.train_freq = train_freq
        self.gradient_steps = (
            self.train_freq if gradient_steps == -1 else gradient_steps
        )

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        self.n_rl_agents = self.learning_role.buffer.n_rl_units

        self.obs_dim = self.learning_role.obs_dim
        self.act_dim = self.learning_role.act_dim

        self.device = self.learning_role.device
        self.float_type = self.learning_role.float_type

        self.unique_obs_len = 8

        # define critic and target critic per agent

    async def update_policy(self):
        logger.info(f"Updating Policy")
        for _ in range(self.gradient_steps):
            # loop over all agents based on number of agents in sel.n_rl_agents
            for i in range(self.n_rl_agents + 1):
                # why the modulo 100?
                # if i % 100 == 0:
                # sample replay buffer
                transitions = self.learning_role.buffer.sample(self.batch_size)
                states = transitions.observations
                actions = transitions.actions
                next_states = transitions.next_observations
                rewards = transitions.rewards
