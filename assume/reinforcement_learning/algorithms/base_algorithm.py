# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 15:43:00 2021

@author: Nick_SimPC
"""

import os
import pickle

import torch as th
from buffer import ReplayBuffer
from torch.nn import functional as F
from torch.optim import Adam


class RLAlgorithm:
    def __init__(
        self,
        env=None,
        learning_rate=1e-4,
        buffer_size=1e6,
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
        self.env = env
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        # get train frequency from learning role
        self.train_freq = len(env.snapshots) if train_freq == -1 else train_freq
        self.gradient_steps = (
            self.train_freq if gradient_steps == -1 else gradient_steps
        )
        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        self.rl_agents = [
            agent for agent in env.rl_powerplants + env.rl_storages if agent.learning
        ]
        self.n_rl_agents = len(self.rl_agents)

        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        self.device = env.device
        self.float_type = env.float_type

        self.buffer = ReplayBuffer(
            buffer_size=int(buffer_size),
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            n_rl_units=self.n_rl_agents,
            device=self.device,
        )

        self.unique_obs_len = 8

        for agent in self.rl_agents:
            agent.critic = Critic(
                self.n_rl_agents,
                self.obs_dim,
                self.act_dim,
                self.float_type,
                self.unique_obs_len,
            )

            agent.critic_target = Critic(
                self.n_rl_agents,
                self.obs_dim,
                self.act_dim,
                self.float_type,
                self.unique_obs_len,
            )

            agent.critic.optimizer = Adam(
                agent.critic.parameters(), lr=self.learning_rate
            )

            agent.critic_target.load_state_dict(agent.critic.state_dict())
            agent.critic_target.train(mode=False)

            agent.critic = agent.critic.to(self.device)
            agent.critic_target = agent.critic_target.to(self.device)

        if self.env.load_params:
            self.load_params(self.env.load_params)

        self.steps_done = 0
        self.n_updates = 0

    def update_policy(self):
        self.logger.error(
            "No policy update function of the used Rl algorithm was defined. Please dinfe how the policies should be updated in the specific algorithm you use"
        )

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        for agent in self.rl_agents:
            agent.critic = agent.critic.train(mode)
            agent.actor = agent.actor.train(mode)

        self.training = mode

    def save_params(self, dir_name="best_policy"):
        save_dir = self.env.save_params["save_dir"]

        def save_obj(obj, directory, agent):
            path = f"{directory}critic_{str(agent)}"
            th.save(obj, path)

        directory = save_dir + self.env.simulation_id + "/" + dir_name + "/"

        if not os.path.exists(directory):
            os.makedirs(directory)

        for agent in self.rl_agents:
            obj = {
                "critic": agent.critic.state_dict(),
                "critic_target": agent.critic_target.state_dict(),
                "critic_optimizer": agent.critic.optimizer.state_dict(),
            }

            save_obj(obj, directory, agent.name)

    def load_params(self, load_params):
        if not load_params["load_critics"]:
            return None

        sim_id = load_params["id"]
        load_dir = load_params["dir"]

        self.env.logger.info("Loading critic parameters...")

        def load_obj(directory, agent):
            path = f"{directory}critic_{str(agent)}"
            return th.load(path, map_location=self.device)

        directory = load_params["policy_dir"] + sim_id + "/" + load_dir + "/"

        if not os.path.exists(directory):
            raise FileNotFoundError(
                "Specified directory for loading the critics does not exist!"
            )

        for agent in self.rl_agents:
            try:
                params = load_obj(directory, agent.name)
                agent.critic.load_state_dict(params["critic"])
                agent.critic_target.load_state_dict(params["critic_target"])
                agent.critic.optimizer.load_state_dict(params["critic_optimizer"])
            except Exception:
                self.world.logger.info(
                    "No critic values loaded for agent {}".format(agent.name)
                )
