import os

import torch as th
from mango import Role

# need
# self.world.dt
# self.world.snapshots


class Learning(Role):
    def __init__(
        self,
        learning_config,
    ):
        self.buffer = None
        self.obs_dim = learning_config["observation_dimension"]
        self.act_dim = learning_config["action_dimension"]
        self.episodes_done = 0
        self.n_rl_units = 0
        self.rl_units = []
        self.rl_algorithm = learning_config["algorithm"]

        # define wheter we train model or evaluate it
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

            # store evaluation values
            self.rl_eval_rewards = []
            self.rl_eval_profits = []
            self.rl_eval_regrets = []

            # shedule policy updates
            # TODO define frequency and stuff
            """
            recurrency_task = rr.rrule(
                freq=rr.HOURLY,
                interval=self.train_freq,
                dtstart=self.learning_starts,
                until=self.training_episodes,
                cache=True,
            )
        self.context.schedule_recurrent_task(self.store_dfs, recurrency_task)
        """

        # self.float_type = th.float16 if self.device.type == "cuda" else th.float

    def setup(self):
        # subscribe to messages for handling the training process
        self.context.subscribe_message(
            self,
            self.handle_message,
            lambda content, meta: content.get("context") == "rl_training",
        )

    def handle_message(self, content, meta):
        """
        Handles the incoming messages and performs corresponding actions.

        Args:
            content (dict): The content of the message.
            meta: The metadata associated with the message. (not needed yet)
        """

        if content.get("type") == "replay_buffer":
            start = content.get("start")
            data = content.get("data")
            self.buffer.add(
                obs=data[0],
                actions=data[1],
                reward=data[2],
            )
