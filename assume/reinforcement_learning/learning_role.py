# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch as th
from mango import Role

from assume.common.base import LearningConfig, LearningStrategy
from assume.common.utils import datetime2timestamp
from assume.reinforcement_learning.algorithms.matd3 import TD3
from assume.reinforcement_learning.algorithms.ppo import PPO
from assume.reinforcement_learning.buffer import ReplayBuffer, RolloutBuffer
from assume.reinforcement_learning.learning_utils import linear_schedule_func

logger = logging.getLogger(__name__)


class Learning(Role):
    """
    This class manages the learning process of reinforcement learning agents, including initializing key components such as
    neural networks, replay buffer, and learning hyperparameters. It handles both training and evaluation modes based on
    the provided learning configuration.

    Args:
        learning_config (LearningConfig): The configuration for the learning process.

    """

    # TD3 and PPO (Replay buffer, gradient steps, early stopping, self.eval_episodes_done potentiall irrelevant for PPO)
    def __init__(
        self,
        learning_config: LearningConfig,
        start: datetime = None,
        end: datetime = None,
    ):
        # General parameters
        self.rl_algorithm_name = learning_config.get("algorithm", "matd3")
        self.early_stopping_steps = learning_config.get(self.rl_algorithm_name, {}).get(
            "early_stopping_steps", 10
        )
        self.early_stopping_threshold = learning_config.get(
            self.rl_algorithm_name, {}
        ).get("early_stopping_threshold", 0.05)
        self.episodes_done = 0
        # dict[key, value]
        self.rl_strats: dict[int, LearningStrategy] = {}

        # For centralized critic in MATD3
        self.critics = {}

        # define whether we train model or evaluate it
        self.learning_mode = learning_config["learning_mode"]
        self.continue_learning = learning_config["continue_learning"]
        self.perform_evaluation = learning_config["perform_evaluation"]
        self.trained_policies_save_path = learning_config["trained_policies_save_path"]
        self.trained_policies_load_path = learning_config.get(
            "trained_policies_load_path", self.trained_policies_save_path
        )


        self.learning_rate = learning_config["learning_rate"]
        self.actor_architecture = learning_config.get(self.rl_algorithm_name, {}).get(
            "actor_architecture", "mlp"
        )
        self.training_episodes = learning_config[
            "training_episodes"
        ]
        self.train_freq = learning_config.get(self.rl_algorithm_name, {}).get(
            "train_freq"
        )
        self.batch_size = learning_config.get(self.rl_algorithm_name, {}).get(
            "batch_size", 128
        )
        if start is not None:
            self.start = datetime2timestamp(start)
        if end is not None:
            self.end = datetime2timestamp(end)

        self.gamma = learning_config.get(self.rl_algorithm_name, {}).get("gamma", 0.99)
        self.learning_rate_schedule = learning_config.get(
            "learning_rate_schedule", None
        )
        if self.learning_rate_schedule == "linear":
            self.calc_lr_from_progress = linear_schedule_func(self.learning_rate)
        else:
            self.calc_lr_from_progress = lambda x: self.learning_rate

        noise_dt = learning_config.get("noise_dt", 1)
        self.action_noise_schedule = learning_config.get("action_noise_schedule", None)
        if self.action_noise_schedule == "linear":
            self.calc_noise_from_progress = linear_schedule_func(noise_dt)
        else:
            self.calc_noise_from_progress = lambda x: noise_dt
        
        # if we do not have initial experience collected we will get an error as no samples are available on the
        # buffer from which we can draw experience to adapt the strategy, hence we set it to minimum one episode
        self.episodes_collecting_initial_experience = max(
            learning_config.get(self.rl_algorithm_name, {}).get(
                "episodes_collecting_initial_experience", 5
            ),
            1,
        )

                # if early_stopping_steps are not provided then set default to no early stopping (early_stopping_steps need to be greater than validation_episodes)
        self.early_stopping_steps = learning_config.get(
            "early_stopping_steps",
            int(
                self.training_episodes
                / learning_config.get("validation_episodes_interval", 5)
                + 1
            ),
        )
        self.early_stopping_threshold = learning_config.get(
            "early_stopping_threshold", 0.05
        )

        # if early_stopping_steps are not provided then set default to no early stopping (early_stopping_steps need to be greater than validation_episodes)
        self.early_stopping_steps = learning_config.get(
            "early_stopping_steps",
            int(
                self.training_episodes
                / learning_config.get("validation_episodes_interval", 5)
                + 1
            ),
        )
        self.early_stopping_threshold = learning_config.get(
            "early_stopping_threshold", 0.05
        )

        self.gradient_steps = (
            int(self.train_freq[:-1])
            if learning_config.get("gradient_steps", -1) == -1
            else learning_config["gradient_steps"]
        )

        # Algorithm-specific parameters
        if self.rl_algorithm_name == "matd3":
            self.buffer: ReplayBuffer = None
            self.target_critics = {}
            self.noise_sigma = learning_config["matd3"]["noise_sigma"]
            self.noise_scale = learning_config["matd3"]["noise_scale"]

        elif self.rl_algorithm_name == "ppo":
            self.buffer: RolloutBuffer = None

            # Potentially more parameters for PPO
            self.steps_per_epoch = learning_config["ppo"].get("steps_per_epoch", 10)
            self.clip_ratio = learning_config["ppo"].get("clip_ratio", 0.2)
            self.entropy_coeff = learning_config["ppo"].get("entropy_coeff", 0.02)
            self.value_coeff = learning_config["ppo"].get("value_coeff", 0.5)
            self.max_grad_norm = learning_config["ppo"].get("max_grad_norm", 0.5)
            self.gae_lambda = learning_config["ppo"].get("gae_lambda", 0.95)

        
        
        cuda_device = (
            learning_config["device"]
            if "cuda" in learning_config.get("device", "cpu")
            else "cpu"
        )
        
        self.device = th.device(cuda_device if th.cuda.is_available() else "cpu")

        # future: add option to choose between float16 and float32
        # float_type = learning_config.get("float_type", "float32")
        self.float_type = th.float

        th.backends.cuda.matmul.allow_tf32 = True
        th.backends.cudnn.allow_tf32 = True



        # Initialize the algorithm depending on the type
        self.create_learning_algorithm(self.rl_algorithm_name)

        # Initialize evaluation metrics
        self.eval_episodes_done = 0
        self.max_eval = defaultdict(lambda: -1e9)
        self.rl_eval = defaultdict(list)
        # List of avg changes
        self.avg_rewards = []

    # TD3 and PPO
    def load_inter_episodic_data(self, inter_episodic_data):
        """
        Load the inter-episodic data from the dict stored across simulation runs.

        Args:
            inter_episodic_data (dict): The inter-episodic data to be loaded.

        """
        # TODO: Make this function of algorithm so that we loose case sensitivity here
        self.episodes_done = inter_episodic_data["episodes_done"]
        self.eval_episodes_done = inter_episodic_data["eval_episodes_done"]
        self.max_eval = inter_episodic_data["max_eval"]
        self.rl_eval = inter_episodic_data["all_eval"]
        self.avg_rewards = inter_episodic_data["avg_all_eval"]
        self.buffer = inter_episodic_data["buffer"]

        if self.rl_algorithm_name == "matd3":
            # if enough initial experience was collected according to specifications in learning config
            # turn off initial exploration and go into full learning mode
            if self.episodes_done > self.episodes_collecting_initial_experience:
                self.turn_off_initial_exploration()

        self.initialize_policy(inter_episodic_data["actors_and_critics"])

    # TD3 and PPO
    def get_inter_episodic_data(self):
        """
        Dump the inter-episodic data to a dict for storing across simulation runs.

        Returns:
            dict: The inter-episodic data to be stored.
        """

        return {
            "episodes_done": self.episodes_done,
            "eval_episodes_done": self.eval_episodes_done,
            "max_eval": self.max_eval,
            "all_eval": self.rl_eval,
            "avg_all_eval": self.avg_rewards,
            "buffer": self.buffer,
            "actors_and_critics": self.rl_algorithm.extract_policy(),
        }

    # TD3 and PPO
    def setup(self) -> None:
        """
        Set up the learning role for reinforcement learning training.

        Notes:
            This method prepares the learning role for the reinforcement learning training process. It subscribes to relevant messages
            for handling the training process and schedules recurrent tasks for policy updates based on the specified training frequency.
        """
        # subscribe to messages for handling the training process

        if not self.perform_evaluation:
            self.context.subscribe_message(
                self,
                self.save_buffer_and_update,
                lambda content, meta: content.get("context") == "rl_training",
            )

    def save_buffer_and_update(self, content: dict, meta: dict) -> None:
        """
        Handles the incoming messages and performs corresponding actions.

        Args:
            content (dict): The content of the message.
            meta (dict): The metadata associated with the message. (not needed yet)
        """

        if self.rl_algorithm_name == "matd3":
            if content.get("type") == "save_buffer_and_update":
                data = content["data"]
                self.buffer.add(
                    obs=data[0],
                    actions=data[1],
                    reward=data[2],
                )

            self.update_policy()

        elif self.rl_algorithm_name == "ppo":
            
            logger.debug("save_buffer_and_update in learning_role.py")

            if content.get("type") == "save_buffer_and_update":
                data = content["data"]
                self.buffer.add(
                    obs=data[0],
                    actions=data[1],
                    reward=data[2],
                    log_probs=data[3],
                )

            self.update_policy()

            # since the PPO is an on-policy algorithm it onyl uses the expercience collected with the current policy
            # after the policy-update which ultimately changes the policy, theb buffer needs to be cleared 
            self.buffer.reset()

    # TD3
    def turn_off_initial_exploration(self) -> None:
        """
        Disable initial exploration mode for all learning strategies.

        Notes:
            This method turns off the initial exploration mode for all learning strategies associated with the learning role. Initial
            exploration is often used to collect initial experience before training begins. Disabling it can be useful when the agent
            has collected sufficient initial data and is ready to focus on training.
        """
        for _, unit in self.rl_strats.items():
            unit.collect_initial_experience_mode = False

    def get_progress_remaining(self) -> float:
        """
        Get the remaining learning progress from the simulation run.

        """
        for _, unit in self.rl_strats.items():
            unit.action_noise.scale = stored_scale

    def get_noise_scale(self) -> None:
        """
        Get the noise scale from the first learning strategy (unit) in rl_strats.

        Notes:
            The noise scale is the same for all learning strategies (units) in rl_strats, so we only need to get it from one unit.
            It is only depended on the number of updates done so far, which is determined by the number of episodes done and the update frequency.

        """
        stored_scale = list(self.rl_strats.values())[0].action_noise.scale

        return stored_scale

    def create_learning_algorithm(self, algorithm: str):
        """
        Create and initialize the reinforcement learning algorithm, based on defined algorithm type.

        This method creates and initializes the reinforcement learning algorithm based on the specified algorithm name. The algorithm
        is associated with the learning role and configured with relevant hyperparameters.

        Args:
            algorithm (str): The name of the reinforcement learning algorithm.
        """
        if algorithm == "matd3":
            self.rl_algorithm = TD3(
                learning_role=self,
                learning_rate=self.learning_rate,
                episodes_collecting_initial_experience=self.episodes_collecting_initial_experience,
                gradient_steps=self.gradient_steps,
                batch_size=self.batch_size,
                gamma=self.gamma,
                actor_architecture=self.actor_architecture,
            )
        elif algorithm == "ppo":
            self.rl_algorithm = PPO(
                learning_role=self,
                learning_rate=self.learning_rate,
                gamma=self.gamma,  # Discount factor
                gradient_steps=self.gradient_steps,  # Number of epochs for policy updates
                clip_ratio=self.clip_ratio,  # PPO-specific clipping parameter
                vf_coef=self.value_coeff,  # Coefficient for value function loss
                entropy_coef=self.entropy_coeff,  # Coefficient for entropy to encourage exploration
                max_grad_norm=self.max_grad_norm,  # Maximum gradient norm for clipping
                gae_lambda=self.gae_lambda,  # Lambda for Generalized Advantage Estimation (GAE)
                actor_architecture=self.actor_architecture,  # Actor network architecture
            )
        else:
            logger.error(f"Learning algorithm {algorithm} not implemented!")

        # Loop over rl_strats
        # self.rl_algorithm an die Learning Strategy übergeben
        # Damit die Learning Strategy auf act/get_actions zugreifen kann

    # TD3
    def initialize_policy(self, actors_and_critics: dict = None) -> None:
        """
        Initialize the policy of the reinforcement learning agent considering the respective algorithm.

        This method initializes the policy (actor) of the reinforcement learning agent. It tests if we want to continue the learning process with
        stored policies from a former training process. If so, it loads the policies from the specified directory. Otherwise, it initializes the
        respective new policies.
        """

        self.rl_algorithm.initialize_policy(actors_and_critics)

        if self.continue_learning is True and actors_and_critics is None:
            directory = self.trained_policies_load_path
            if Path(directory).is_dir():
                logger.info(f"Loading pretrained policies from {directory}!")
                self.rl_algorithm.load_params(directory)
            else:
                logger.warning(
                    f"Folder with pretrained policies {directory} does not exist"
                )

    # TD3 and PPO
    def update_policy(self) -> None:
        """
        Update the policy of the reinforcement learning agent.

        This method is responsible for updating the policy (actor) of the reinforcement learning agent asynchronously. It checks if
        the number of episodes completed is greater than the number of episodes required for initial experience collection. If so,
        it triggers the policy update process by calling the `update_policy` method of the associated reinforcement learning algorithm.

        Notes:
            This method is typically scheduled to run periodically during training to continuously improve the agent's policy.
        """
        if self.rl_algorithm_name == "ppo":
            self.rl_algorithm.update_policy()
        else:
            if self.episodes_done > self.episodes_collecting_initial_experience:
                self.rl_algorithm.update_policy()

    def compare_and_save_policies(self, metrics: dict) -> bool:
        """
        Compare evaluation metrics and save policies based on the best achieved performance according to the metrics calculated.

        This method compares the evaluation metrics, such as reward, profit, and regret, and saves the policies if they achieve the
        best performance in their respective categories. It iterates through the specified modes, compares the current evaluation
        value with the previous best, and updates the best value if necessary. If an improvement is detected, it saves the policy
        and associated parameters.

        metrics contain a metric key like "reward" and the current value.
        This function stores the policies with the highest metric.
        So if minimize is required one should add for example "minus_regret" which is then maximized.

        Returns:
            bool: True if early stopping criteria is triggered.

        Notes:
            This method is typically used during the evaluation phase to save policies that achieve superior performance.
            Currently the best evaluation metric is still assessed by the development team and preliminary we use the average rewards.
        """
        if not metrics:
            logger.error("tried to save policies but did not get any metrics")
            return
        # if the current values are a new max in one of the metrics - we store them in the default folder

        # add current reward to list of all rewards
        for metric, value in metrics.items():
            self.rl_eval[metric].append(value)

            # check if the current value is the best value
            if self.rl_eval[metric][-1] > self.max_eval[metric]:
                self.max_eval[metric] = self.rl_eval[metric][-1]

                # use first metric as default
                if metric == list(metrics.keys())[0]:
                    # store the best for our current metric in its folder
                    self.rl_algorithm.save_params(
                        directory=f"{self.trained_policies_save_path}/{metric}_eval_policies"
                    )

                    logger.info(
                        f"New best policy saved, episode: {self.eval_episodes_done + 1}, {metric=}, value={value:.2f}"
                    )

            # if we do not see any improvement in the last x evaluation runs we stop the training
            if len(self.rl_eval[metric]) >= self.early_stopping_steps:
                self.avg_rewards.append(
                    sum(self.rl_eval[metric][-self.early_stopping_steps :])
                    / self.early_stopping_steps
                )

                if len(self.avg_rewards) >= self.early_stopping_steps:
                    avg_change = (
                        max(self.avg_rewards[-self.early_stopping_steps :])
                        - min(self.avg_rewards[-self.early_stopping_steps :])
                    ) / min(self.avg_rewards[-self.early_stopping_steps :])

                    if avg_change < self.early_stopping_threshold:
                        logger.info(
                            f"Stopping training as no improvement above {self.early_stopping_threshold} in last {self.early_stopping_steps} evaluations for {metric}"
                        )
                        if (
                            self.learning_rate_schedule or self.action_noise_schedule
                        ) is not None:
                            logger.info(
                                f"Learning rate schedule ({self.learning_rate_schedule}) or action noise schedule ({self.action_noise_schedule}) were scheduled to decay, further learning improvement can be possible. End value of schedule may not have been reached."
                            )

                        self.rl_algorithm.save_params(
                            directory=f"{self.trained_policies_save_path}/last_policies"
                        )

                        return True
            return False
