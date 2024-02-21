# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import torch as th

logger = logging.getLogger(__name__)


class RLAlgorithm:
    """
    The base RL model class. To implement your own RL algorithm, you need to subclass this class and implement the `update_policy` method.

    Args:
        learning_role (Learning Role object): Learning object
        learning_rate (float): learning rate for adam optimizer
        episodes_collecting_initial_experience (int): how many steps of the model to collect transitions for before learning starts
        batch_size (int): Minibatch size for each gradient update
        tau (float): the soft update coefficient ("Polyak update", between 0 and 1)
        gamma (float): the discount factor
        gradient_steps (int): how many gradient steps to do after each rollout (if -1, no gradient step is done)
        policy_delay (int): Policy and target networks will only be updated once every policy_delay steps per training steps. The Q values will be updated policy_delay more often (update every training step)
        target_policy_noise (float): Standard deviation of Gaussian noise added to target policy (smoothing noise)
        target_noise_clip (float): Limit for absolute value of target policy smoothing noise
    """

    def __init__(
        self,
        # init learning_role as object of Learning class
        learning_role,
        learning_rate=1e-4,
        episodes_collecting_initial_experience=100,
        batch_size=1024,
        tau=0.005,
        gamma=0.99,
        gradient_steps=-1,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
    ):
        super().__init__()

        self.learning_role = learning_role
        self.learning_rate = learning_rate
        self.episodes_collecting_initial_experience = (
            episodes_collecting_initial_experience
        )
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.gradient_steps = gradient_steps

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        self.obs_dim = self.learning_role.obs_dim
        self.act_dim = self.learning_role.act_dim

        self.device = self.learning_role.device
        self.float_type = self.learning_role.float_type

        self.unique_obs_len = 8

    def update_policy(self):
        logger.error(
            "No policy update function of the used Rl algorithm was defined. Please define how the policies should be updated in the specific algorithm you use"
        )

    def load_obj(self, directory: str):
        """
        Load an object from a specified directory.

        This method loads an object, typically saved as a checkpoint file, from the specified
        directory and returns it. It uses the `torch.load` function and specifies the device for loading.

        Args:
            directory (str): The directory from which the object should be loaded.

        Returns:
            object: The loaded object.
        """
        return th.load(directory, map_location=self.device)

    def load_params(self, directory: str) -> None:
        """
        Load learning params - abstract method to be implemented by the Learning Algorithm
        """
        pass
