import logging

logger = logging.getLogger(__name__)


class RLAlgorithm:
    def __init__(
        self,
        # init learning_role as object of Learning class
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

    def update_policy(self):
        self.logger.error(
            "No policy update function of the used Rl algorithm was defined. Please define how the policies should be updated in the specific algorithm you use"
        )
