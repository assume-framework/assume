import logging
import os
from datetime import datetime, timedelta

import torch as th
from dateutil import rrule as rr
from mango import Role
from torch.optim import Adam

from assume.common.base import LearningConfig, LearningStrategy
from assume.reinforcement_learning.algorithms.matd3 import TD3
from assume.reinforcement_learning.buffer import ReplayBuffer
from assume.reinforcement_learning.learning_utils import Actor, CriticTD3

logger = logging.getLogger(__name__)


class Learning(Role):
    def __init__(
        self,
        learning_config: LearningConfig,
        start: datetime,
        end: datetime,
    ):
        self.simulation_start = start
        self.simulation_end = end

        # how many learning roles do exist and how are they named
        self.buffer: ReplayBuffer = None
        self.obs_dim = learning_config["observation_dimension"]
        self.act_dim = learning_config["action_dimension"]
        self.episodes_done = 0
        self.rl_strats: dict[int, LearningStrategy] = {}
        self.rl_algorithm = learning_config["algorithm"]
        self.critics = {}
        self.target_critics = {}

        # define whether we train model or evaluate it
        self.training_episodes = learning_config["training_episodes"]
        self.learning_mode = learning_config["learning_mode"]

        cuda_device = (
            learning_config["device"]
            if "cuda" in learning_config.get("device", "cpu")
            else "cpu"
        )
        self.device = th.device(cuda_device if th.cuda.is_available() else "cpu")
        self.float_type = th.float16 if "cuda" in cuda_device else th.float
        th.backends.cuda.matmul.allow_tf32 = True

        self.learning_rate = learning_config.get("learning_rate", 1e-4)
        self.episodes_collecting_initial_experience = learning_config.get(
            "episodes_collecting_initial_experience", 5
        )
        self.train_freq = learning_config.get("train_freq", 1)
        self.gradient_steps = (
            self.train_freq
            if learning_config.get("gradient_steps", -1) == -1
            else learning_config["gradient_steps"]
        )
        self.batch_size = learning_config.get("batch_size", 128)
        self.gamma = learning_config.get("gamma", 0.99)

        self.eval_episodes_done = 0
        self.max_eval_reward = -1e9
        self.max_eval_regret = 1e9
        self.max_eval_profit = -1e9

        # function that initializes learning, needs to be an extra function so that it can be called after buffer is given to Role
        self.create_learning_algorithm(self.rl_algorithm)

        # store evaluation values
        self.rl_eval_rewards = []
        self.rl_eval_profits = []
        self.rl_eval_regrets = []

        if learning_config.get("continue_learning", False):
            load_directory = learning_config.get("load_model_path")
            if load_directory is not None:
                self.logger.warning(
                    "You have specified continue learning as True but no load_model_path was given!"
                )
                self.info("Continuing learning with randomly initialized values!")
            else:
                self.load_params(load_directory)

    def setup(self):
        # subscribe to messages for handling the training process
        self.context.subscribe_message(
            self,
            self.handle_message,
            lambda content, meta: content.get("context") == "rl_training",
        )

        recurrency_task = rr.rrule(
            freq=rr.HOURLY,
            interval=self.train_freq,
            dtstart=self.simulation_start,
            until=self.simulation_end,
            cache=True,
        )

        self.context.schedule_recurrent_task(self.update_policy, recurrency_task)

    def handle_message(self, content, meta):
        """
        Handles the incoming messages and performs corresponding actions.

        Args:
            content (dict): The content of the message.
            meta: The metadata associated with the message. (not needed yet)
        """

        if content.get("type") == "replay_buffer":
            data = content["data"]
            self.buffer.add(
                obs=data[0],
                actions=data[1],
                reward=data[2],
            )

    def turn_off_initial_exploration(self):
        for _, unit in self.rl_strats.items():
            unit.collect_initial_experience_mode = False

    def create_learning_algorithm(self, algorithm):
        if algorithm == "matd3":
            self.rl_algorithm = TD3(
                learning_role=self,
                learning_rate=self.learning_rate,
                episodes_collecting_initial_experience=self.episodes_collecting_initial_experience,
                gradient_steps=self.gradient_steps,
                batch_size=self.batch_size,
                gamma=self.gamma,
            )
        else:
            self.logger.error(f"Learning algorithm {algorithm} not implemented!")

    async def update_policy(self):
        if self.episodes_done > self.episodes_collecting_initial_experience:
            self.rl_algorithm.update_policy()

    # TODO: add evaluation function
    def compare_and_save_policies(self):
        modes = ["reward", "profit", "regret"]
        for mode in modes:
            value = None

            if mode == "reward" and self.rl_eval_rewards[-1] > self.max_eval_reward:
                self.max_eval_reward = self.rl_eval_rewards[-1]
                dir_name = "highest_reward"
                value = self.max_eval_reward
            elif mode == "profit" and self.rl_eval_profits[-1] > self.max_eval_profit:
                self.max_eval_profit = self.rl_eval_profits[-1]
                dir_name = "highest_profit"
                value = self.max_eval_profit
            elif (
                mode == "regret"
                and self.rl_eval_regrets[-1] < self.max_eval_regret
                and self.rl_eval_regrets[-1] != 0
            ):
                self.max_eval_regret = self.rl_eval_regrets[-1]
                dir_name = "lowest_regret"
                value = self.max_eval_regret

            if value is not None:
                self.rl_algorithm.save_params(dir_name=dir_name)
                for unit in self.rl_powerplants + self.rl_storages:
                    if unit.learning:
                        unit.save_params(dir_name=dir_name)

                self.logger.info(
                    f"Policies saved, episode: {self.eval_episodes_done + 1}, mode: {mode}, value: {value:.2f}"
                )

    def save_params(self, directory):
        self.save_critic_params(directory=f"{directory}/critics")
        self.save_actor_params(directory=f"{directory}/actors")

    def save_critic_params(self, directory):
        def save_obj(obj, directory, agent):
            path = f"{directory}/critic_{str(agent)}.pt"
            th.save(obj, path)

        os.makedirs(directory, exist_ok=True)
        for u_id in self.rl_strats.keys():
            obj = {
                "critic": self.critics[u_id].state_dict(),
                "critic_target": self.target_critics[u_id].state_dict(),
                "critic_optimizer": self.critics[u_id].optimizer.state_dict(),
            }
            save_obj(obj, directory, u_id)

    def save_actor_params(self, directory):
        def save_obj(obj, directory, agent):
            path = f"{directory}/actor_{str(agent)}.pt"
            th.save(obj, path)

        os.makedirs(directory, exist_ok=True)
        for u_id in self.rl_strats.keys():
            obj = {
                "actor": self.rl_strats[u_id].actor.state_dict(),
                "actor_target": self.rl_strats[u_id].actor_target.state_dict(),
                "actor_optimizer": self.rl_strats[u_id].actor.optimizer.state_dict(),
            }
            save_obj(obj, directory, u_id)

    def load_obj(self, directory):
        return th.load(directory, map_location=self.device)

    def load_params(self, directory):
        self.load_critic_params(directory)
        self.load_actor_params(directory)

    def load_critic_params(self, directory):
        self.logger.info("Loading critic parameters...")

        if not os.path.exists(directory):
            self.logger.warning(
                "Specified directory for loading the critics does not exist! Starting with randomly initialized values!"
            )
            return

        for u_id in self.rl_strats.keys():
            try:
                critic_params = self.load_obj(
                    directory=f"{directory}/critics/critic_{str(u_id)}.pt"
                )
                self.critics[u_id].load_state_dict(critic_params["critic"])
                self.target_critics[u_id].load_state_dict(
                    critic_params["critic_target"]
                )
                self.critics[u_id].optimizer.load_state_dict(
                    critic_params["critic_optimizer"]
                )
            except Exception:
                self.logger.warning(f"No critic values loaded for agent {u_id}")

    def load_actor_params(self, directory):
        self.logger.info("Loading actor parameters...")
        if not os.path.exists(directory):
            self.logger.warning(
                "Specified directory for loading the actors does not exist! Starting with randomly initialized values!"
            )
            return

        for u_id in self.rl_strats.keys():
            try:
                actor_params = self.load_obj(
                    directory=f"{directory}/actors/actor_{str(u_id)}.pt"
                )
                self.rl_strats[u_id].actor.load_state_dict(actor_params["actor"])
                self.rl_strats[u_id].actor_target.load_state_dict(
                    actor_params["actor_target"]
                )
                self.rl_strats[u_id].actor.optimizer.load_state_dict(
                    actor_params["actor_optimizer"]
                )
            except Exception:
                self.logger.warning(f"No actor values loaded for agent {u_id}")

    def extract_actors_and_critics(self):
        actors = {}
        actor_targets = {}

        for u_id, unit_strategy in self.rl_strats.items():
            actors[u_id] = unit_strategy.actor
            actor_targets[u_id] = unit_strategy.actor_target

        actors_and_critics = {
            "actors": actors,
            "actor_targets": actor_targets,
            "critics": self.critics,
            "target_critics": self.target_critics,
        }

        return actors_and_critics

    def create_actors_and_critics(self, actors_and_critics):
        if actors_and_critics is None:
            self.create_actors()
            self.create_critics()

        else:
            self.critics = actors_and_critics["critics"]
            self.target_critics = actors_and_critics["target_critics"]
            for u_id, unit_strategy in self.rl_strats.items():
                unit_strategy.actor = actors_and_critics["actors"][u_id]
                unit_strategy.actor_target = actors_and_critics["actor_targets"][u_id]

    def create_actors(self):
        for _, unit_strategy in self.rl_strats.items():
            unit_strategy.actor = Actor(self.obs_dim, self.act_dim, self.float_type).to(
                self.device
            )

            unit_strategy.actor_target = Actor(
                self.obs_dim, self.act_dim, self.float_type
            ).to(self.device)
            unit_strategy.actor_target.load_state_dict(unit_strategy.actor.state_dict())
            unit_strategy.actor_target.train(mode=False)

            unit_strategy.actor.optimizer = Adam(
                unit_strategy.actor.parameters(), lr=self.learning_rate
            )

    def create_critics(self):
        n_agents = len(self.rl_strats)
        strategy: LearningStrategy

        for u_id, strategy in self.rl_strats.items():
            self.critics[u_id] = CriticTD3(
                n_agents, strategy.obs_dim, strategy.act_dim, self.float_type
            )
            self.target_critics[u_id] = CriticTD3(
                n_agents, strategy.obs_dim, strategy.act_dim, self.float_type
            )

            self.critics[u_id].optimizer = Adam(
                self.critics[u_id].parameters(), lr=self.learning_rate
            )

            self.target_critics[u_id].load_state_dict(self.critics[u_id].state_dict())
            self.target_critics[u_id].train(mode=False)

            self.critics[u_id] = self.critics[u_id].to(self.device)
            self.target_critics[u_id] = self.target_critics[u_id].to(self.device)
