import logging
import os

import torch as th
from torch.optim import Adam

logger = logging.getLogger(__name__)


class RLAlgorithm:
    def __init__(
        self,
        learning_role=None,
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
        super().__init__()

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

        self.n_rl_agents = self.learning_role.buffer.n_rl_agents

        self.obs_dim = self.learning_role.obs_dim
        self.act_dim = self.learning_role.act_dim

        self.device = self.learning_role.device
        self.float_type = self.learning_role.float_type

        self.unique_obs_len = 8

        # define critic and target critic per agent

    def update_policy(self):
        self.logger.error(
            "No policy update function of the used Rl algorithm was defined. Please dinfe how the policies should be updated in the specific algorithm you use"
        )

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode

        for agent in self.rl_agents:
            agent.critic = agent.critic.train(mode)
            agent.actor = agent.actor.train(mode)

        self.training = mode
        """
        pass

    def save_params(self, dir_name="best_policy"):
        """
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
        """
        pass

    def load_params(self, load_params):
        """
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
        """
        pass
