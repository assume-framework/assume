from assume.strategies.base_strategy import BaseStrategy
from assume.units.base_unit import BaseUnit


class RLStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()

    def load_strategy():
        """
        In case the strategy is learned with RL the policy (mapping of states to actions) it needs
        to be loaded from current model

        Return: ?
        """

        raise NotImplementedError()

    def calculate_bids(
        self,
        unit: BaseUnit = None,
        operational_window: dict = None,
    ):
        bid_quantity_inflex, bid_price_inflex = 0, 0
        bid_quantity_flex, bid_price_flex = 0, 0

        if operational_window is not None:
            self.current_time = operational_window["window"]["start"]
            # =============================================================================
            # Powerplant is either on, or is able to turn on
            # Calculating possible bid amount
            # =============================================================================
            bid_quantity_inflex = operational_window["min_power"]["power"]

            marginal_cost_mr = operational_window["min_power"]["marginal_cost"]
            marginal_cost_flex = operational_window["max_power"]["marginal_cost"]
            # =============================================================================
            # Calculating possible price
            # =============================================================================
            if unit.current_status:
                bid_price_inflex = self.calculate_EOM_price_if_on(
                    unit, marginal_cost_mr, bid_quantity_inflex
                )
            else:
                bid_price_inflex = self.calculate_EOM_price_if_off(
                    unit, marginal_cost_flex, bid_quantity_inflex
                )

            if unit.total_heat_output[self.current_time] > 0:
                power_loss_ratio = (
                    unit.power_loss_chp[self.current_time]
                    / unit.total_heat_output[self.current_time]
                )
            else:
                power_loss_ratio = 0.0

            # Flex-bid price formulation
            if unit.current_status:
                bid_quantity_flex = (
                    operational_window["max_power"]["power"] - bid_quantity_inflex
                )
                bid_price_flex = (1 - power_loss_ratio) * marginal_cost_flex

        bids = [
            {"price": bid_price_inflex, "volume": bid_quantity_inflex},
            {"price": bid_price_flex, "volume": bid_quantity_flex},
        ]

        return bids

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

    def reset(self):
        self.total_capacity = [0.0 for _ in self.world.snapshots]
        self.total_capacity[-1] = self.minPower + (self.maxPower - self.minPower) / 2

        self.scaled_total_capacity = np.array(self.total_capacity).reshape(-1, 1)
        self.total_scaled_capacity[-1] = self.total_capacity[-1] / self.maxPower

        self.bids_mr = {n: (0.0, 0.0) for n in self.world.snapshots}
        self.bids_flex = {n: (0.0, 0.0) for n in self.world.snapshots}

        self.sent_bids = []

        self.current_downtime = 0

        self.curr_obs = self.create_obs(self.world.currstep)
        self.next_obs = None

        self.curr_action = None
        self.curr_reward = None
        self.curr_experience = []

        self.rewards = [0.0 for _ in self.world.snapshots]
        self.regrets = [0.0 for _ in self.world.snapshots]
        self.profits = [0.0 for _ in self.world.snapshots]

    def formulate_bids(self):
        """
        Take an action based on actor network, add exlorarion noise if needed

        Returns
        -------
            action (PyTorch Variable): Actions for this agent

        """
        if self.world.training:
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

    def step(self):
        t = self.world.currstep
        self.total_capacity[t] = 0

        for bid in self.sent_bids:
            if "mrEOM" in bid.ID:
                self.total_capacity[t] += bid.confirmedAmount
                self.bids_mr[t] = (bid.confirmedAmount, bid.price)
            elif "flexEOM" in bid.ID:
                self.total_capacity[t] += bid.confirmedAmount
                self.bids_flex[t] = (bid.confirmedAmount, bid.price)

        # Calculates market success
        if self.total_capacity[t] < self.minPower:
            self.total_capacity[t] = 0

        self.total_scaled_capacity[t] = self.total_capacity[t] / self.maxPower

        price_difference = self.world.mcp[t] - self.marginal_cost[t]
        profit = price_difference * self.total_capacity[t] * self.world.dt
        opportunity_cost = (
            price_difference * (self.maxPower - self.total_capacity[t]) * self.world.dt
        )
        opportunity_cost = max(opportunity_cost, 0)

        scaling = 0.1 / self.maxPower
        regret_scale = 0.2

        if self.total_capacity[t] != 0 and self.total_capacity[t - 1] == 0:
            profit = profit - self.hotStartCosts / 2
        elif self.total_capacity[t] == 0 and self.total_capacity[t - 1] != 0:
            profit = profit - self.hotStartCosts / 2

        self.rewards[t] = (profit - regret_scale * opportunity_cost) * scaling
        self.profits[t] = profit
        self.regrets[t] = opportunity_cost

        self.curr_reward = self.rewards[t]
        self.next_obs = self.create_obs(self.world.currstep + 1)

        self.curr_experience = [
            self.curr_obs,
            self.next_obs,
            self.curr_action,
            self.curr_reward,
        ]

        self.curr_obs = self.next_obs

        self.sent_bids = []

    def feedback(self, bid):
        if bid.status in ["Confirmed", "PartiallyConfirmed"]:
            t = self.world.currstep

            if "CRMPosDem" in bid.ID:
                self.confQtyCRM_pos.update(
                    {t + _: bid.confirmedAmount for _ in range(self.crm_timestep)}
                )

            if "CRMNegDem" in bid.ID:
                self.confQtyCRM_neg.update(
                    {t + _: bid.confirmedAmount for _ in range(self.crm_timestep)}
                )

            if "steam" in bid.ID:
                self.confQtyDHM_steam[t] = bid.confirmedAmount

        if "steam" in bid.ID:
            self.powerLossFPP(t, bid)

        self.sent_bids.append(bid)

    def create_obs(self, t):
        obs_len = 4
        obs = self.agent.obs.copy()

        # get the marginal cost
        if t < len(self.world.snapshots) - obs_len:
            obs.extend(self.scaled_marginal_cost[t : t + obs_len])
        else:
            obs.extend(self.scaled_marginal_cost[t:])
            obs.extend(
                self.scaled_marginal_cost[: obs_len - (len(self.world.snapshots) - t)]
            )

        if t < obs_len:
            obs.extend(
                self.total_scaled_capacity[len(self.world.snapshots) - obs_len + t :]
            )
            obs.extend(self.total_scaled_capacity[:t])
        else:
            obs.extend(self.total_scaled_capacity[t - obs_len : t])

        obs = np.array(obs)
        obs = (
            th.tensor(obs, dtype=self.float_type)
            .to(self.device, non_blocking=True)
            .view(-1)
        )

        return obs.detach().clone()

    # powerplant function
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
