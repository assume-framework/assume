import pandas as pd

from assume.strategies.base_strategy import BaseStrategy
from assume.units.base_unit import BaseUnit


# TODO learning role übergben die nur intiaiert wird wenn learning aktiv ist und dann entsprechend weiß
# strategie ob sie gerade lernt oder nur ausführt
class RLStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()

        self.foresight = pd.Timedelta("12h")
        self.current_time = None

        # RL agent parameters
        self.obs_dim = self.world.obs_dim
        self.act_dim = self.world.act_dim

        self.device = self.world.device
        self.float_type = self.world.float_type

        self.actor = Actor(self.obs_dim, self.act_dim, self.float_type).to(self.device)

        if self.world.training:
            self.learning_rate = self.world.learning_rate
            self.actor_target = Actor(self.obs_dim, self.act_dim, self.float_type).to(
                self.device
            )
            self.actor_target.load_state_dict(self.actor.state_dict())
            # Target networks should always be in eval mode
            self.actor_target.train(mode=False)

            self.actor.optimizer = Adam(self.actor.parameters(), lr=self.learning_rate)
            self.action_noise = NormalActionNoise(
                mu=0.0, sigma=0.2, action_dimension=self.act_dim, scale=1.0, dt=1.0
            )  # dt=.999996)

        if self.world.load_params:
            self.load_params(self.world.load_params)

    def reset(
        self,
        unit: BaseUnit = None,
    ):
        unit.total_capacity = [0.0 for _ in self.world.snapshots]
        unit.total_capacity[-1] = unit.minPower + (unit.maxPower - unit.minPower) / 2

        unit.scaled_total_capacity = np.array(unit.total_capacity).reshape(-1, 1)
        unit.scaled_total_capacity[-1] = unit.total_capacity[-1] / unit.maxPower

        self.bids_mr = {n: (0.0, 0.0) for n in self.world.snapshots}
        self.bids_flex = {n: (0.0, 0.0) for n in self.world.snapshots}

        self.sent_bids = []

        self.current_downtime = 0

        # TODO needs to come from unit
        self.curr_obs = self.create_obs(self.world.currstep)
        self.next_obs = None

        self.curr_action = None
        self.curr_reward = None
        self.curr_experience = []

        self.rewards = [0.0 for _ in self.world.snapshots]
        self.regrets = [0.0 for _ in self.world.snapshots]
        self.profits = [0.0 for _ in self.world.snapshots]

    def calculate_bids(
        self,
        observation,
        unit: BaseUnit = None,
    ):
        """
        Take an action based on actor network, add exlorarion noise if needed

        Returns
        -------
            action (PyTorch Variable): Actions for this agent

        """

        self.curr_obs = observation

        if self.world.does_training:
            if self.world.episodes_done < self.world.learning_starts:
                self.curr_action = (
                    th.normal(
                        mean=0.0, std=0.2, size=(1, self.act_dim), dtype=self.float_type
                    )
                    .to(self.device)
                    .squeeze()
                )

                self.curr_action += th.tensor(
                    self.scaled_marginal_cost[self.world.currstep],
                    device=self.device,
                    dtype=self.float_type,
                )
            else:
                self.curr_action = self.actor(self.curr_obs).detach()
                self.curr_action += th.tensor(
                    self.action_noise.noise(), device=self.device, dtype=self.float_type
                )
        else:
            self.curr_action = self.actor(self.curr_obs).detach()

        return self.curr_action

    # TODO change to interconnect to send dispatch in unit operator
    def step(
        self,
        unit: BaseUnit = None,
    ):
        t = self.context.current_timestamp

        # Calculates market success
        if unit.total_power_output[t] < unit.min_power:
            unit.total_power_output[t] = 0

        unit.total_scaled_capacity[t] = unit.total_power_output[t] / unit.max_power

        price_difference = self.world.mcp[t] - unit.marginal_cost[t]
        profit = price_difference * unit.total_capacity[t] * self.world.dt
        opportunity_cost = (
            price_difference * (unit.max_power - unit.total_capacity[t]) * self.world.dt
        )
        opportunity_cost = max(opportunity_cost, 0)

        scaling = 0.1 / unit.max_power
        regret_scale = 0.2

        if unit.total_power_output[t] != 0 and unit.total_power_output[t - 1] == 0:
            profit = profit - unit.hot_start_cost / 2
        elif unit.total_power_output[t] == 0 and unit.total_power_output[t - 1] != 0:
            profit = profit - unit.hot_start_cost / 2

        self.rewards[t] = (profit - regret_scale * opportunity_cost) * scaling
        self.profits[t] = profit
        self.regrets[t] = opportunity_cost

        self.curr_reward = self.rewards[t]
        self.next_obs = self.create_obs(t + 1)

        self.curr_experience = [
            self.curr_obs,
            self.next_obs,
            self.curr_action,
            self.curr_reward,
        ]

        self.curr_obs = self.next_obs

        self.sent_bids = []

        # TODO where to update stuff, if in each policy then done for every agent which does not make sense
        # due to central crtici
        # if done in overall simualtioon it will be based on events as well? so when? after what event?
        if self.training:
            obs, actions, rewards = self.collect_experience()
            self.rl_algorithm.buffer.add(obs, actions, rewards)
            self.rl_algorithm.update_policy()

    def save_params(self, dir_name="best_policy"):
        save_dir = self.world.save_params["save_dir"]

        def save_obj(obj, directory):
            path = directory + self.name
            th.save(obj, path)

        obj = {
            "policy": self.actor.state_dict(),
            "target_policy": self.actor_target.state_dict(),
            "policy_optimizer": self.actor.optimizer.state_dict(),
        }

        directory = save_dir + self.world.simulation_id + dir_name + "/"

        if not os.path.exists(directory):
            os.makedirs(directory)

        save_obj(obj, directory)

    def load_params(self, load_params):
        sim_id = load_params["id"]
        load_dir = load_params["dir"]

        def load_obj(directory):
            path = directory + self.name
            return th.load(path, map_location=self.device)

        directory = load_params["policy_dir"] + sim_id + load_dir + "/"

        if not os.path.exists(directory):
            raise FileNotFoundError(
                "Specified directory for loading the actors policy does not exist!"
            )

        params = load_obj(directory)

        self.actor.load_state_dict(params["policy"])
        if self.world.training:
            self.actor_target.load_state_dict(params["target_policy"])
            self.actor.optimizer.load_state_dict(params["policy_optimizer"])

    # everything below is TODO

    def collect_experience(self):
        total_units = self.rl_algorithm.n_rl_agents
        obs = th.zeros((2, total_units, self.obs_dim), device=self.device)
        actions = th.zeros((total_units, self.act_dim), device=self.device)
        rewards = []

        for i, pp in enumerate(self.rl_algorithm.rl_agents):
            obs[0][i] = pp.curr_experience[0]
            obs[1][i] = pp.curr_experience[1]
            actions[i] = pp.curr_experience[2]
            rewards.append(pp.curr_experience[3])

        return obs, actions, rewards

    # this function is used in flexable but I want this in the RL Units operator
    # or does it have to be there because I intialize the MATD3 there and there it is needed
    # since it is overarching for different agents?

    # TODO check wiritng and if it should be in units operator and tensorboard
    def extract_rl_episode_info(self):
        total_rewards = 0
        total_profits = 0
        total_regrets = 0

        for unit in self.rl_powerplants:
            if unit.learning:
                total_rewards += sum(unit.rewards)
                total_profits += sum(unit.profits)
                total_regrets += sum(unit.regrets)

        for unit in self.rl_storages:
            if unit.learning:
                total_rewards += sum(unit.rewards)
                total_rewards += sum(unit.rewards)
                total_profits += sum(unit.profits)

        total_rl_units = (
            self.rl_algorithm.n_rl_agents
            if self.rl_algorithm is not None
            else len(self.rl_powerplants + self.rl_storages)
        )
        average_reward = total_rewards / total_rl_units / len(self.snapshots)
        average_profit = total_profits / total_rl_units / len(self.snapshots)
        average_regret = total_regrets / total_rl_units / len(self.snapshots)

        if self.training:
            self.tensorboard_writer.add_scalar(
                "Train/Average Reward", average_reward, self.episodes_done
            )
            self.tensorboard_writer.add_scalar(
                "Train/Average Profit", average_profit, self.episodes_done
            )
            self.tensorboard_writer.add_scalar(
                "Train/Average Regret", average_regret, self.episodes_done
            )
        else:
            self.rl_eval_rewards.append(average_reward)
            self.rl_eval_profits.append(average_profit)
            self.rl_eval_regrets.append(average_regret)

            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar(
                    "Eval/Average Reward", average_reward, self.eval_episodes_done
                )
                self.tensorboard_writer.add_scalar(
                    "Eval/Average Profit", average_profit, self.eval_episodes_done
                )
                self.tensorboard_writer.add_scalar(
                    "Eval/Average Regret", average_regret, self.eval_episodes_done
                )
