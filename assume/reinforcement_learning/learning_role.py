import logging
import os
from datetime import datetime, timedelta

import torch as th
from dateutil import rrule as rr
from mango import Role
from torch.optim import Adam

from assume.common.base import LearningStrategy
from assume.reinforcement_learning.algorithms.matd3 import TD3
from assume.reinforcement_learning.buffer import ReplayBuffer
from assume.reinforcement_learning.learning_utils import CriticTD3

logger = logging.getLogger(__name__)


class Learning(Role):
    def __init__(
        self,
        learning_config,
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
        self.rl_units: dict[int, LearningStrategy] = {}
        self.rl_algorithm = learning_config["algorithm"]
        self.critics = {}
        self.target_critics = {}

        # define whether we train model or evaluate it
        self.training_episodes = learning_config["training_episodes"]
        self.learning_mode = learning_config["learning_mode"]
        self.cuda_device = learning_config["device"]
        self.float_type = th.float

        th.backends.cuda.matmul.allow_tf32 = True

        if not self.learning_mode:
            # no training means GPU not necessary as calculations are not that expensive
            self.device = th.device("cpu")

            # check if we already trained a model that can be used
            rl_models_path = f"/outputs/rl_models/{str(self.algorithm)}.py"
            if not os.path.exists(rl_models_path):
                self.logger.error(
                    "Reinforcement Learning training is deactivated and no pretrained model were provided, activate learning or specify model to use in config"
                )
            # TODO implement loading models! check Nick

        else:
            cuda_device = f"cuda:{str(self.cuda_device)}"
            self.device = th.device(cuda_device if th.cuda.is_available() else "cpu")

            self.learning_rate = learning_config["learning_rate"]
            self.learning_starts = learning_config["collect_inital_experience"]
            self.train_freq = learning_config["train_freq"]
            self.gradient_steps = learning_config["gradient_steps"]
            self.batch_size = learning_config["batch_size"]
            self.gamma = learning_config["gamma"]
            self.training_episodes = learning_config["training_episodes"]
            self.eval_episodes_done = 0
            self.max_eval_reward = -1e9
            self.max_eval_regret = 1e9
            self.max_eval_profit = -1e9

        self.float_type = th.float16 if self.cuda_device == "cuda" else th.float

    def init_learning(self):
        # function that initializes learning, needs to be an extra function so that it can be called after buffer is given to Role
        self.create_learning_algorithm(self.rl_algorithm)

        # store evaluation values
        self.rl_eval_rewards = []
        self.rl_eval_profits = []
        self.rl_eval_regrets = []

        n_agents = len(self.rl_units)
        strategy: LearningStrategy
        for u_id, strategy in self.rl_units.items():
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
            dtstart=self.simulation_start + timedelta(hours=self.learning_starts),
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
            start = content.get("start")
            data = content["data"]
            self.buffer.add(
                obs=data[0],
                actions=data[1],
                reward=data[2],
            )

    def create_learning_algorithm(self, algorithm):
        if algorithm == "matd3":
            self.rl_algorithm = TD3(
                learning_role=self,
                learning_starts=self.learning_starts,
                train_freq=self.train_freq,
                gradient_steps=self.gradient_steps,
                batch_size=self.batch_size,
                gamma=self.gamma,
            )
        else:
            self.logger.error(
                f"you specified an reinforcement learning algorithm {algorithm}, for which no files where provided"
            )

    async def update_policy(self):
        self.rl_algorithm.update_policy()

    # in assume self
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

    def set_buffer(self, buffer: ReplayBuffer):
        self.buffer = buffer
        self.init_learning()

    def save_params(self, dir_name: str = "best_policy"):
        def save_obj(obj, directory, agent):
            path = f"{directory}critic_{str(agent)}"
            th.save(obj, path)

        directory = f"output/{simulation_id}/{dir_name}/"
        os.makedirs(directory, exist_ok=True)

        for u_id in self.rl_units.keys():
            obj = {
                "critic": self.critics[u_id].state_dict(),
                "critic_target": self.target_critics[u_id].state_dict(),
                "critic_optimizer": self.critics[u_id].optimizer.state_dict(),
            }
            save_obj(obj, directory, u_id)

    def load_params(self, load_params: dict):
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

        for u_id in self.rl_units.keys():
            try:
                params = load_obj(directory, agent.name)
                self.critics[u_id].load_state_dict(params["critic"])
                self.target_critics[u_id].load_state_dict(params["critic_target"])
                self.critics[u_id].optimizer.load_state_dict(params["critic_optimizer"])
            except Exception:
                self.world.logger.info(
                    "No critic values loaded for agent {}".format(u_id)
                )
