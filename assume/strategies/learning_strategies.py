# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch as th

from assume.common.base import (
    BaseUnit,
    LearningStrategy,
    MinMaxChargeStrategy,
    MinMaxStrategy,
    SupportsMinMax,
    SupportsMinMaxCharge,
)
from assume.common.fast_pandas import FastSeries
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.common.utils import min_max_scale
from assume.reinforcement_learning.algorithms import actor_architecture_aliases
from assume.reinforcement_learning.learning_utils import NormalActionNoise

logger = logging.getLogger(__name__)


class TorchLearningStrategy(LearningStrategy):
    """
    A strategy to enable machine learning with pytorch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.unit_id = kwargs["unit_id"]

        # defines bounds of actions space
        self.min_bid_price = self.learning_config.min_bid_price
        self.max_bid_price = self.learning_config.max_bid_price

        # tells us whether we are training the agents or just executing per-learning strategies
        self.learning_mode = self.learning_config.learning_mode
        self.evaluation_mode = self.learning_config.evaluation_mode

        # based on learning config
        self.algorithm = self.learning_config.algorithm
        self.actor_architecture = self.learning_config.actor_architecture

        # check if actor architecture is available
        if self.actor_architecture in actor_architecture_aliases.keys():
            self.actor_architecture_class = actor_architecture_aliases[
                self.actor_architecture
            ]
        else:
            raise ValueError(
                f"Policy '{self.actor_architecture}' unknown. Supported architectures are {list(actor_architecture_aliases.keys())}"
            )

        # sets the device of the actor network
        self.device = self.learning_role.device

        # future: add option to choose between float16 and float32
        # float_type = kwargs.get("float_type", "float32")
        self.float_type = th.float

        # define standard deviation for the initial exploration noise
        self.exploration_noise_std = self.learning_config.exploration_noise_std

        if self.learning_mode or self.evaluation_mode:
            # learning role overwrites this if loaded from file or after initial experience episodes
            self.collect_initial_experience_mode = True

            self.action_noise = NormalActionNoise(
                mu=0.0,
                sigma=self.learning_config.noise_sigma,
                action_dimension=self.act_dim,
                scale=self.learning_config.noise_scale,
                dt=self.learning_config.noise_dt,
            )

            self.learning_role.register_strategy(self)

        # actor policies are only loaded here from file if learning mode is off (otherwise handled by learning_role)
        # i.e., when loading pre-trained strategies without training ("learning_mode: false" and "trained_policies_load_path" specified in config)
        # or final simulation run after training (terminate_learning == true)
        elif Path(self.learning_config.trained_policies_load_path).is_dir():
            self.load_actor_params(
                load_path=self.learning_config.trained_policies_load_path
            )
        else:
            raise FileNotFoundError(
                f"No policies were provided for DRL unit {self.unit_id}!. Please provide a valid path to the trained policies. Expected them under filepath '{self.learning_config.trained_policies_load_path}'."
            )

    def load_actor_params(self, load_path):
        """
        Load actor parameters.

        Args:
            load_path (str): The path to load parameters from.
        """
        directory = f"{load_path}/actors/actor_{self.unit_id}.pt"

        params = th.load(directory, map_location=self.device, weights_only=True)

        self.actor = self.actor_architecture_class(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            float_type=self.float_type,
            unique_obs_dim=self.unique_obs_dim,
            num_timeseries_obs_dim=self.num_timeseries_obs_dim,
        ).to(self.device)
        self.actor.load_state_dict(params["actor"])
        self.actor.eval()  # set the actor to evaluation mode

    def prepare_observations(self, unit, market_id):
        # scaling factors for the observations
        # Note: These scaling factors could be interpreted as information leakage. However as we are in a simulation environment and not a purley forecasting setting
        # we assume that the agent has access to this information already
        upper_scaling_factor_price = max(unit.forecaster.price[market_id])
        lower_scaling_factor_price = min(unit.forecaster.price[market_id])
        residual_load = unit.forecaster.residual_load.get(
            market_id, FastSeries(index=unit.index, value=0)
        )
        upper_scaling_factor_res_load = max(residual_load)
        lower_scaling_factor_res_load = min(residual_load)

        self.scaled_res_load_obs = min_max_scale(
            residual_load,
            lower_scaling_factor_res_load,
            upper_scaling_factor_res_load,
        )

        self.scaled_prices_obs = min_max_scale(
            unit.forecaster.price[market_id],
            lower_scaling_factor_price,
            upper_scaling_factor_price,
        )

    def create_observation(
        self, unit: BaseUnit, market_id: str, start: datetime, end: datetime
    ):
        """
        Constructs a scaled observation tensor based on the unit's forecast data and internal state.

        Args
        ----
        unit : BaseUnit
            The unit providing forecast and internal state data.
        market_id : str
            Identifier for the specific market.
        start : datetime
            Start time for the observation period.
        end : datetime
            End time for the observation period.

        Returns
        -------
        torch.Tensor
            Observation tensor with data on forecasted residual load, price, and unit-specific values.

        Notes
        -----
        Observations are constructed from forecasted residual load and price over the foresight period,
        scaled by maximum demand and bid price. The last values in the observation vector represent
        unit-specific values, depending on the strategy and unit-type.
        """

        # ensure scaled observations are prepared
        if not hasattr(self, "scaled_res_load_obs") or not hasattr(
            self, "scaled_prices_obs"
        ):
            self.prepare_observations(unit, market_id)

        # =============================================================================
        # 1.1 Get the Observations, which are the basis of the action decision
        # =============================================================================

        # --- 1. Forecasted residual load and price (forward-looking) ---
        scaled_res_load_forecast = self.scaled_res_load_obs.window(
            start, self.foresight, direction="forward"
        )
        scaled_price_forecast = self.scaled_prices_obs.window(
            start, self.foresight, direction="forward"
        )

        # --- 2. Historical actual prices (backward-looking) ---
        # Note: We scale with the max_bid_price here in comparison to the scaling of the forecast where we use the max price of the forecast period
        # this is not consistent but has worked well so far. Future work could look into this in more detail.
        scaled_price_history = (
            unit.outputs["energy_accepted_price"].window(
                start, self.foresight, direction="backward"
            )
            / self.max_bid_price
        )

        # --- 3. Individual observations ---
        individual_observations = self.get_individual_observations(unit, start, end)

        # concat all observations into one array
        observation = np.concatenate(
            [
                scaled_res_load_forecast,
                scaled_price_forecast,
                scaled_price_history,
                individual_observations,
            ]
        )

        # transfer array to GPU for NN processing
        observation = th.as_tensor(
            observation, dtype=self.float_type, device=self.device
        ).flatten()

        if self.learning_mode:
            self.learning_role.add_observation_to_cache(
                self.unit_id, start, observation
            )

        return observation

    def get_individual_observations(
        self, unit: BaseUnit, start: datetime, end: datetime
    ):
        """
        Retrieves the unit-specific observations.

        Args
        ----
        unit : BaseUnit
            The unit providing forecast and internal state data.
        start : datetime
            Start time for the observation period.

        Returns
        -------
        individual_observations : np.ndarray
            Strategy and unit-specific observations.
        """

        return np.array([])

    def get_actions(self, next_observation):
        """
        Determines actions based on the current observation, applying noise for exploration if in learning mode.

        Args
        ----
        next_observation : torch.Tensor
            Observation data influencing bid price and direction.

        Returns
        -------
        torch.Tensor
            Actions that include bid price and direction.
        torch.Tensor
            Noise component which is already added to actions for exploration, if applicable.

        Notes
        -----
        In learning mode, actions incorporate noise for exploration. Initial exploration relies
        solely on noise to cover the action space broadly.
        """

        # distinction whether we are in learning mode or not to handle exploration realised with noise
        if self.learning_mode and not self.evaluation_mode:
            # if we are in learning mode the first x episodes we want to explore the entire action space
            # to get a good initial experience, in the area around the costs of the agent
            if self.collect_initial_experience_mode:
                # define current action as solely noise
                noise = th.normal(
                    mean=0.0,
                    std=self.exploration_noise_std,
                    size=(self.act_dim,),
                    dtype=self.float_type,
                    device=self.device,
                )

                # =============================================================================
                # 2.1 Get Actions and handle exploration
                # =============================================================================
                # only use noise as the action to enforce exploration
                curr_action = noise

            else:
                # if we are not in the initial exploration phase we chose the action with the actor neural net
                # and add noise to the action
                curr_action = self.actor(next_observation).detach()
                noise = self.action_noise.noise(
                    device=self.device, dtype=self.float_type
                )
                curr_action += noise

                # make sure that noise adding does not exceed the actual output of the NN as it pushes results in a direction that actor can't even reach
                curr_action = th.clamp(
                    curr_action, self.actor.min_output, self.actor.max_output
                )
        else:
            # if we are not in learning mode we just use the actor neural net to get the action without adding noise
            curr_action = self.actor(next_observation).detach()
            # noise is an tensor with zeros, because we are not in learning mode
            noise = th.zeros_like(curr_action, dtype=self.float_type)

        return curr_action, noise


class EnergyLearningStrategy(TorchLearningStrategy, MinMaxStrategy):
    """
    Reinforcement Learning Strategy that enables the agent to learn optimal bidding strategies
    on an Energy-Only Market.

    The agent submits two price bids: one for the inflexible component (P_min) and another for
    the flexible component (P_max - P_min) of its capacity. This strategy utilizes a set of 38
    observations to generate actions, which are then transformed into market bids. The observation
    space comprises two unique values: the marginal cost and the current capacity of the unit.

    The observation space for this strategy consists of 38 elements, drawn from both the forecaster
    and the unit's internal state. Observations include the following components:

    - **Forecasted Residual Load**: Forecasted load over the foresight period, scaled by the maximum
      demand of the unit, indicating anticipated grid conditions.
    - **Forecasted Price**: Price forecast over the foresight period, scaled by the maximum bid price,
      providing a sense of expected market prices.
    - **Total Capacity and Marginal Cost**: The last two elements of the observation vector, representing
      the unique state of the unit itself. Here, `total capacity` is scaled by the unit's maximum
      power, while `marginal cost` is scaled by the maximum bid price. These specific values reflect the
      unit's operational capacity and production costs, helping the agent determine the optimal bid.

    Actions are formulated as two values, representing bid prices for both the inflexible and flexible
    components of the unit's capacity. Actions are scaled from a range of [-1, 1]
    to real bid prices in the `calculate_bids` method, where they translate into specific bid volumes
    for the inflexible (P_min) and flexible (P_max - P_min) components.

    Rewards are based on profit from transactions, minus operational and opportunity costs. Key components include:

    - **Profit**: Determined from the income generated by accepted bids, calculated as the product of
      accepted price, volume, and duration.
    - **Operational Costs**: Includes marginal costs and start-up costs when the unit transitions between
      on and off states.
    - **Opportunity Cost**: Calculated as the potential income lost when the unit is not running at full
      capacity. High opportunity costs result in a penalty, encouraging full utilization of capacity.
    - **Scaling and Regret Term**: The final reward combines profit, opportunity costs, and a regret term
      to penalize missed revenue opportunities. The reward is scaled to guide learning, with a regret term
      applied to reduce high opportunity costs.

    Attributes
    ----------
    foresight : int
        Number of time steps for which the agent forecasts market conditions. Defaults to 12.
    max_bid_price : float
        Maximum allowable bid price. Defaults to 100.
    max_demand : float
        Maximum demand capacity of the unit. Defaults to 10e3.
    device : str
        Device for computation, such as "cpu" or "cuda". Defaults to "cpu".
    float_type : str
        Data type for floating-point calculations, typically "float32". Defaults to "float32".
    learning_mode : bool
        Indicates whether the agent is in learning mode. Defaults to False.
    algorithm : str
        Name of the RL algorithm in use. Defaults to "matd3".
    actor_architecture_class : type[torch.nn.Module]
        Class of the neural network architecture used for the actor network. Defaults to MLPActor.
    actor : torch.nn.Module
        Actor network for determining actions.
    order_types : list[str]
        Types of market orders supported by the strategy. Defaults to ["SB"].
    action_noise : NormalActionNoise
        Noise model added to actions during learning to encourage exploration. Defaults to None.
    collect_initial_experience_mode : bool
        Whether the agent is collecting initial experience through exploration. Defaults to True.

    Args
    ----
    *args : Variable length argument list.
    **kwargs : Arbitrary keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        # 'foresight' represents the number of time steps into the future that we will consider
        # when constructing the observations.
        foresight = kwargs.pop("foresight", 12)
        act_dim = kwargs.pop("act_dim", 2)
        unique_obs_dim = kwargs.pop("unique_obs_dim", 2)
        super().__init__(
            foresight=foresight,
            act_dim=act_dim,
            unique_obs_dim=unique_obs_dim,
            *args,
            **kwargs,
        )

        # define allowed order types
        self.order_types = kwargs.get("order_types", ["SB"])

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Calculates bids based on the current observations and actions derived from the actor network.

        Args
        ----
        unit : SupportsMinMax
            The unit for which to calculate bids, with details on capacity constraints.
        market_config : MarketConfig
            The configuration settings of the energy market.
        product_tuples : list[Product]
            List of products with start and end times for bidding.
        **kwargs : Additional keyword arguments.

        Returns
        -------
        Orderbook
            Contains bid entries for each product, including start time, end time, price, and volume.

        Notes
        -----
        This method structures bids in two parts:
        - **Inflexible Bid** (P_min): A bid for the minimum operational capacity.
        - **Flexible Bid** (P_max - P_min): A bid for the flexible capacity available after P_min.
        The actions are scaled to reflect real bid prices and volumes, which are then converted into
        orderbook entries.
        """

        bid_quantity_inflex, bid_price_inflex = 0, 0
        bid_quantity_flex, bid_price_flex = 0, 0

        start = product_tuples[0][0]
        end = product_tuples[0][1]
        # get technical bounds for the unit output from the unit
        min_power, max_power = unit.calculate_min_max_power(start, end)
        min_power = min_power[0]
        max_power = max_power[0]

        # =============================================================================
        # 1. Get the Observations, which are the basis of the action decision
        # =============================================================================
        next_observation = self.create_observation(
            unit=unit,
            market_id=market_config.market_id,
            start=start,
            end=end,
        )

        # =============================================================================
        # 2. Get the Actions, based on the observations
        # =============================================================================
        actions, noise = self.get_actions(next_observation)

        # =============================================================================
        # 3. Transform Actions into bids
        # =============================================================================
        # actions are in the range [-1,1], we need to transform them into actual bids
        # we can use our domain knowledge to guide the bid formulation
        bid_prices = actions * self.max_bid_price

        # 3.1 formulate the bids for Pmin
        # Pmin, the minimum run capacity is the inflexible part of the bid, which should always be accepted
        bid_quantity_inflex = min_power
        bid_price_inflex = th.min(bid_prices)

        # 3.1 formulate the bids for Pmax - Pmin
        # Pmin, the minimum run capacity is the inflexible part of the bid, which should always be accepted
        bid_quantity_flex = max_power - bid_quantity_inflex
        bid_price_flex = th.max(bid_prices)

        # actually formulate bids in orderbook format
        bids = [
            {
                "start_time": start,
                "end_time": end,
                "only_hours": None,
                "price": bid_price_inflex,
                "volume": bid_quantity_inflex,
                "node": unit.node,
            },
            {
                "start_time": start,
                "end_time": end,
                "only_hours": None,
                "price": bid_price_flex,
                "volume": bid_quantity_flex,
                "node": unit.node,
            },
        ]

        if self.learning_mode:
            self.learning_role.add_actions_to_cache(self.unit_id, start, actions, noise)

        return bids

    def get_actions(self, next_observation):
        """
        Compute actions based on the current observation.

        Args
        ----
        next_observation : torch.Tensor
            The current observation, where the last element is assumed to be the marginal cost.

        Returns
        -------
        tuple of torch.Tensor
            A tuple containing: Actions to be taken (with or without noise). The noise component (if any), useful for diagnostics.

        Notes
        -----
        During learning, exploratory noise is applied and already part of the curr_action unless in evaluation mode. In initial exploration mode, actions are sampled around the marginal cost to explore its vicinity. We assume the final element of `next_observation` is the marginal cost.
        """

        # Get the base action and associated noise from the parent implementation
        curr_action, noise = super().get_actions(next_observation)

        if self.learning_mode and not self.evaluation_mode:
            if self.collect_initial_experience_mode:
                # Assumes last dimension of the observation corresponds to marginal cost
                marginal_cost = next_observation[
                    -1
                ].detach()  # ensure no gradients flow through
                # Add marginal cost to the action directly for initial random exploration
                curr_action += marginal_cost

        return curr_action, noise

    def get_individual_observations(
        self, unit: SupportsMinMax, start: datetime, end: datetime
    ):
        """
        Retrieves the unit-specific observations. For dispatchable units, this includes the last dispatched volume and the current marginal costs.

        Args
        ----
        unit : SupportsMinMax
            The unit providing forecast and internal state data.
        start : datetime.datetime
            Start time for the observation period.

        Returns
        -------
        individual_observations : np.ndarray
            Scaled total dispatched capacity and marginal cost.

        Notes
        -----
            The last two values in the observation vector represent the total capacity
            and marginal cost, scaled by maximum power and bid price, respectively.
        """

        # --- Current volume & marginal cost ---
        current_volume = unit.get_output_before(start)
        current_costs = unit.calculate_marginal_cost(start, current_volume)

        scaled_total_dispatch = current_volume / unit.max_power
        scaled_marginal_cost = current_costs / self.max_bid_price

        individual_observations = np.array(
            [scaled_total_dispatch, scaled_marginal_cost]
        )

        return individual_observations

    def calculate_reward(
        self,
        unit,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ):
        """
        Calculates the reward for the unit based on profits, costs, and opportunity costs from market transactions.

        Args
        ----
        unit : SupportsMinMax
            The unit for which to calculate the reward.
        marketconfig : MarketConfig
            Market configuration settings.
        orderbook : Orderbook
            Orderbook containing executed bids and details.

        Notes
        -----
        The reward is computed by combining the following:
        **Profit**: Income from accepted bids minus marginal and start-up costs.
        **Opportunity Cost**: Penalty for underutilizing capacity, calculated as potential lost income.
        **Regret Term**: A scaled regret term penalizes high opportunity costs to guide effective bidding.

        The reward is scaled and stored along with other outputs in the unit’s data to support learning.
        """
        # Function is called after the market is cleared, and we get the market feedback,
        # allowing us to calculate profit based on the realized transactions.

        product_type = marketconfig.product_type

        start = orderbook[0]["start_time"]
        end = orderbook[0]["end_time"]
        # end includes the end of the last product, to get the last products' start time we deduct the frequency once
        end_excl = end - unit.index.freq

        # Depending on how the unit calculates marginal costs, retrieve cost values.
        marginal_cost = unit.calculate_marginal_cost(
            start, unit.outputs[product_type].at[start]
        )
        market_clearing_price = orderbook[0]["accepted_price"]

        duration = (end - start) / timedelta(hours=1)

        income = 0.0
        operational_cost = 0.0

        accepted_volume_total = 0
        offered_volume_total = 0

        # Iterate over all orders in the orderbook to calculate order-specific profit.
        for order in orderbook:
            accepted_volume = order.get("accepted_volume", 0)
            accepted_volume_total += accepted_volume

            offered_volume_total += order["volume"]

            # Calculate profit as income minus operational cost for this event.
            order_income = market_clearing_price * accepted_volume * duration
            order_cost = marginal_cost * accepted_volume * duration

            # Accumulate income and operational cost for all orders.
            income += order_income
            operational_cost += order_cost

        # Consideration of start-up costs, divided evenly between upward and downward regulation events.
        if (
            unit.outputs[product_type].at[start] != 0
            and unit.outputs[product_type].at[start - unit.index.freq] == 0
        ):
            operational_cost += unit.hot_start_cost / 2
        elif (
            unit.outputs[product_type].at[start] == 0
            and unit.outputs[product_type].at[start - unit.index.freq] != 0
        ):
            operational_cost += unit.hot_start_cost / 2

        profit = income - operational_cost

        # Stabilizing learning: Limit positive profit to 10% of its absolute value.
        # This reduces variance in rewards and avoids extreme profit-seeking behavior.
        # However, this does NOT prevent the agent from exploiting market inefficiencies if they exist.
        # This leads to the agent learning to bid close to marginal costs to ensure acceptance,
        # while still being able to capitalize on any market inefficiencies that may arise.
        # However this will lead the learning agents to converge to the market price they should bid from below marginal costs.
        # We only advise using this if profits can spike extremely high due to market conditions, or many learning units enter tactic collusion.
        # IMPORTANT: This is a clear case of reward_tuning to stabilize learning - Use with caution!
        # profit_scale= 0.1

        profit_scale = 1
        profit = min(profit, profit_scale * abs(profit))

        # Opportunity cost: The income lost due to not operating at full capacity.
        opportunity_cost = (
            (market_clearing_price - marginal_cost)
            * (unit.max_power - accepted_volume_total)
            * duration
        )

        # If opportunity cost is negative, no income was lost, so we set it to zero.
        opportunity_cost = max(opportunity_cost, 0)

        # Dynamic regret scaling:
        # - If accepted volume is positive, apply lower regret (0.1) to avoid punishment for being on the edge of the merit order.
        # - If no dispatch happens, apply higher regret (0.5) to discourage idle behavior, if it could have been profitable.
        regret_scale = 0.1 if accepted_volume_total > unit.min_power else 0.5

        # --------------------
        # 4.1 Calculate Reward
        # Instead of directly setting reward = profit, we incorporate a regret term (opportunity cost penalty).
        # This guides the agent toward strategies that maximize accepted bids while minimizing lost opportunities.

        # scaling factor to normalize the reward to the range [-1,1]
        scaling = 1 / (self.max_bid_price * unit.max_power)
        regret = regret_scale * opportunity_cost
        reward = scaling * (profit - regret)

        # Store results in unit outputs
        # Note: these are not learning-specific results but stored for all units for analysis
        unit.outputs["profit"].loc[start:end_excl] += profit
        unit.outputs["total_costs"].loc[start:end_excl] += operational_cost

        # write rl-rewards to buffer
        if self.learning_mode:
            self.learning_role.add_reward_to_cache(
                unit.id, start, reward, regret, profit
            )


class EnergyLearningSingleBidStrategy(EnergyLearningStrategy, MinMaxStrategy):
    """
    Reinforcement Learning Strategy with Single-Bid Structure for Energy-Only Markets.

    This strategy is a simplified variant of the standard `EnergyLearningStrategy`, which typically submits two
    separate price bids for inflexible (P_min) and flexible (P_max - P_min) components. Instead,
    `EnergyLearningSingleBidStrategy` submits a single bid that always offers the unit's maximum power,
    effectively treating the full capacity as inflexible from a bidding perspective.

    The core reinforcement learning mechanics, including the observation structure, actor network
    architecture, and reward formulation, remain consistent with the two-bid `EnergyLearningStrategy`. However,
    this strategy modifies the action space to produce only a single bid price, and omits the
    decomposition of capacity into flexible and inflexible parts.

    Attributes
    ----------
    Inherits all attributes from EnergyLearningStrategy, with the exception of:
    - act_dim : int
        Reduced to 1 to reflect single bid pricing.
    - foresight : int
        Set to 24 to match typical storage strategy setups.

    """

    def __init__(self, *args, **kwargs):
        # we select 24 to be in line with the storage strategies
        foresight = kwargs.pop("foresight", 24)
        act_dim = kwargs.pop("act_dim", 1)
        unique_obs_dim = kwargs.pop("unique_obs_dim", 2)
        super().__init__(
            foresight=foresight,
            act_dim=act_dim,
            unique_obs_dim=unique_obs_dim,
            *args,
            **kwargs,
        )

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Generates a single price bid for the full available capacity (max_power).

        The method observes market and unit state, derives an action (bid price) from
        the actor network, and constructs one bid covering the entire capacity, without
        distinguishing between flexible and inflexible components.

        Returns
        -------
        Orderbook
            A list containing one bid with start/end time, full volume, and calculated price.
        """

        start = product_tuples[0][0]
        end = product_tuples[0][1]
        # get technical bounds for the unit output from the unit
        _, max_power = unit.calculate_min_max_power(start, end)
        max_power = max_power[0]

        # =============================================================================
        # 1. Get the Observations, which are the basis of the action decision
        # =============================================================================
        next_observation = self.create_observation(
            unit=unit,
            market_id=market_config.market_id,
            start=start,
            end=end,
        )

        # =============================================================================
        # 2. Get the Actions, based on the observations
        # =============================================================================
        actions, noise = self.get_actions(next_observation)

        # =============================================================================
        # 3. Transform Actions into bids
        # =============================================================================
        # actions are in the range [-1,1] + noise, we need to transform them into actual bids
        # we can use our domain knowledge to guide the bid formulation
        bid_price = actions[0] * self.max_bid_price

        # actually formulate bids in orderbook format
        bids = [
            {
                "start_time": start,
                "end_time": end,
                "only_hours": None,
                "price": bid_price,
                "volume": max_power,
                "node": unit.node,
            },
        ]

        if self.learning_mode:
            self.learning_role.add_actions_to_cache(self.unit_id, start, actions, noise)

        return bids


class StorageEnergyLearningStrategy(TorchLearningStrategy, MinMaxChargeStrategy):
    """
    Reinforcement Learning Strategy for a storage unit that enables the agent to learn
    optimal bidding strategies on an Energy-Only Market.

    The observation space for this strategy consists of 74 elements. Key components include:

    - **State of Charge**: Represents the current level of energy in the storage unit,
      influencing the bid direction and capacity.
    - **Energy Cost**: The cost associated with the energy content in the storage unit,
      which helps determine bid prices and profitability.
    - **Price Forecasts**
    - **Residual Load Forecasts**

    The agent's actions are formulated as two values, representing the bid price and the bid direction.
    These actions are scaled and interpreted to form actionable market bids, with specific conditions
    dictating the bid type. The storage agent can also decide to stay inactive by submitting a zero bid
    as this can be a valid strategy in some market conditions and also improves the learning process.

    - **Bid Price**: The one action value determines the price at which the agent will bid.
    - **Bid Direction**: This is implicitly set based on the action:
        - If `action < 0`: The agent submits a **buy bid**.
        - If `action >= 0`: The agent submits a **sell bid**.

    Rewards are based on the profit generated by the agent's market bids, with sell bids contributing
    positive profit and buy bids contributing negative profit. Additional components in the reward
    calculation include:

    - **Profit**: Calculated from the income of successful sell bids minus costs from buy bids.
    - **Fixed Costs**: Charges associated with storage operations, including charging and discharging
      costs, which are deducted from the reward.

    Attributes
    ----------
    foresight : int
        Number of time steps for forecasting market conditions. Defaults to 24.
    max_bid_price : float
        Maximum allowable bid price. Defaults to 100.
    max_demand : float
        Maximum demand capacity of the storage. Defaults to 10e3.
    device : str
        Device used for computation ("cpu" or "cuda"). Defaults to "cpu".
    float_type : str
        Data type for floating-point calculations. Defaults to "float32".
    learning_mode : bool
        Whether the agent is in learning mode. Defaults to False.
    algorithm : str
        RL algorithm used by the agent. Defaults to "matd3".
    actor_architecture_class : type[torch.nn.Module]
        Class of the neural network for the actor network. Defaults to MLPActor.
    actor : torch.nn.Module
        The neural network used to predict actions.
    order_types : list[str]
        Types of market orders used by the strategy. Defaults to ["SB"].
    action_noise : NormalActionNoise
        Noise model added to actions during learning for exploration. Defaults to None.
    collect_initial_experience_mode : bool
        Whether the agent is in an exploration mode for initial experience. Defaults to True.

    Args
    ----
    *args : Variable length argument list.
    **kwargs : Arbitrary keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        # 'foresight' represents the number of time steps into the future that we will consider
        # when constructing the observations.
        foresight = kwargs.pop("foresight", 24)
        act_dim = kwargs.pop("act_dim", 1)
        unique_obs_dim = kwargs.pop("unique_obs_dim", 2)
        super().__init__(
            foresight=foresight,
            act_dim=act_dim,
            unique_obs_dim=unique_obs_dim,
            *args,
            **kwargs,
        )

        # define allowed order types
        self.order_types = kwargs.get("order_types", ["SB"])

    def get_individual_observations(
        self, unit: SupportsMinMaxCharge, start: datetime, end: datetime
    ):
        """
        Retrieves the unit-specific observations for storage units. For storages we use the state of charge and cost of currently stored energy as the individual observations.
        We define the latter as the average volume weighted procurement costs of the currently stored energy.

        Args
        ----
        unit : SupportsMinMaxCharge
            Storage unit providing forecasted and current state data.
        start : datetime.datetime
            Start time for the observation period.

        Returns
        -------
        individual_observations: np.ndarray
            Array containing state of charge and energy cost.

        Notes
        -----
        Observations are scaled by the unit's max state of charge and energy costs, creating input for
        the agent's action selection.
        """
        # get the current soc and energy cost value
        soc = unit.outputs["soc"].at[start]
        cost_stored_energy_scaled = (
            unit.outputs["cost_stored_energy"].at[start] / self.max_bid_price
        )

        individual_observations = np.array([soc, cost_stored_energy_scaled])

        return individual_observations

    def calculate_bids(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Generates market bids based on the unit's current state and observations.

        Args
        ----
        unit : SupportsMinMaxCharge
            The storage unit with information on charging/discharging capacity.
        market_config : MarketConfig
            Configuration of the energy market.
        product_tuples : list[Product]
            List of market products to bid on, each containing start and end times.
        **kwargs : Additional keyword arguments.

        Returns
        -------
        Orderbook
            Structured bids including price, volume, and bid direction.

        Notes
        -----
        Observations are used to calculate bid actions, which are then scaled and processed
        into bids for submission in the market.
        """

        start = product_tuples[0][0]
        end_all = product_tuples[-1][1]

        next_observation = self.create_observation(
            unit=unit,
            market_id=market_config.market_id,
            start=start,
            end=end_all,
        )
        # =============================================================================
        # Get the Actions, based on the observations
        # =============================================================================
        actions, noise = self.get_actions(next_observation)

        # =============================================================================
        # 3. Transform Actions into bids
        # =============================================================================
        # the absolute value of the action determines the bid price
        bid_price = abs(actions[0]) * self.max_bid_price
        # the sign of the action determines the bid direction
        if actions[0] < 0:
            bid_direction = "buy"
        elif actions[0] >= 0:
            bid_direction = "sell"

        _, max_discharge = unit.calculate_min_max_discharge(start, end_all)
        _, max_charge = unit.calculate_min_max_charge(start, end_all)

        bid_quantity_supply = max_discharge[0]
        bid_quantity_demand = max_charge[0]

        bids = []

        if bid_direction == "sell":
            bids.append(
                {
                    "start_time": start,
                    "end_time": end_all,
                    "only_hours": None,
                    "price": bid_price,
                    "volume": bid_quantity_supply,
                    "node": unit.node,
                }
            )

        elif bid_direction == "buy":
            bids.append(
                {
                    "start_time": start,
                    "end_time": end_all,
                    "only_hours": None,
                    "price": bid_price,
                    "volume": bid_quantity_demand,  # negative value for demand
                    "node": unit.node,
                }
            )

        if self.learning_mode:
            self.learning_role.add_actions_to_cache(self.unit_id, start, actions, noise)

        return bids

    def calculate_reward(
        self,
        unit: SupportsMinMaxCharge,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ):
        """
        Calculates the reward based on profit generated by bids after market feedback.

        Args
        ----
        unit : SupportsMinMaxCharge
            The storage unit associated with the agent.
        marketconfig : MarketConfig
            Configuration of the energy market.
        orderbook : Orderbook
            Contains executed bids and transaction details.

        Notes
        -----
        Rewards are based on profit and include fixed costs for charging and discharging.
        """
        product_type = marketconfig.product_type
        reward = 0

        # check if orderbook contains only one order and raise an error if not and notify the user
        # that the strategy is not designed for multiple orders and the market configuration should be adjusted
        if len(orderbook) > 1:
            raise ValueError(
                "StorageEnergyLearningStrategy is not designed for multiple orders. Please adjust the market configuration or the strategy."
            )

        order = orderbook[0]
        start = order["start_time"]
        end = order["end_time"]
        # end includes the end of the last product, to get the last products' start time we deduct the frequency once
        end_excl = end - unit.index.freq

        next_time = start + unit.index.freq
        duration_hours = (end - start) / timedelta(hours=1)

        # Calculate marginal and starting costs
        marginal_cost = unit.calculate_marginal_cost(
            start, unit.outputs[product_type].at[start]
        )
        marginal_cost += unit.get_starting_costs(int(duration_hours))

        accepted_volume = order.get("accepted_volume", 0)
        # ignore very small volumes due to calculations
        accepted_volume = accepted_volume if abs(accepted_volume) > 1 else 0
        accepted_price = order.get("accepted_price", 0)

        # Calculate profit and cost for the order
        order_profit = accepted_price * accepted_volume * duration_hours
        order_cost = abs(marginal_cost * accepted_volume * duration_hours)

        current_soc = unit.outputs["soc"].at[start]
        next_soc = unit.outputs["soc"].at[next_time]

        # Calculate and clip the energy cost for the start time
        # cost_stored_energy = average volume-weighted procurement costs of the currently stored energy
        if next_soc * unit.capacity < 1:
            unit.outputs["cost_stored_energy"].at[next_time] = 0
        elif accepted_volume < 0:
            # increase costs of current SoC by price for buying energy
            # not fully representing the true cost per MWh (e.g. omitting discharge efficiency losses), but serving as a proxy for it
            unit.outputs["cost_stored_energy"].at[next_time] = (
                unit.outputs["cost_stored_energy"].at[start]
                * current_soc
                * unit.capacity
                - (accepted_price + marginal_cost) * accepted_volume * duration_hours
            ) / (next_soc * unit.capacity)
        else:
            unit.outputs["cost_stored_energy"].at[next_time] = unit.outputs[
                "cost_stored_energy"
            ].at[start]

        unit.outputs["cost_stored_energy"].at[next_time] = np.clip(
            unit.outputs["cost_stored_energy"].at[next_time],
            -self.max_bid_price,
            self.max_bid_price,
        )

        profit = order_profit - order_cost

        # scaling factor to normalize the reward to the range [-1,1]
        scaling_factor = 1 / (self.max_bid_price * unit.max_power_discharge)

        reward += scaling_factor * profit

        # Store results in unit outputs
        # Note: these are not learning-specific results but stored for all units for analysis
        unit.outputs["profit"].loc[start:end_excl] += profit
        unit.outputs["total_costs"].loc[start:end_excl] += order_cost

        # write rl-rewards to buffer
        if self.learning_mode:
            self.learning_role.add_reward_to_cache(unit.id, start, reward, 0, profit)


class RenewableEnergyLearningSingleBidStrategy(EnergyLearningSingleBidStrategy):
    """
    Reinforcement Learning Strategy for a renewable unit that enables the agent to learn
    optimal bidding strategies on an Energy-Only Market.

    The observation space for this strategy consists of 75 elements. Key components include:
    - **Price Forecasts**
    - **Residual Load Forecasts**
    - **Historic Prices**
    As individual observations we provide the
    - **Renewable Generation Forecasts**
    - **Marginal Costs**
    - **Previous Time Step Generation**

    The agent's actions are formulated as one value, representing the bid price at which it bids its full capacity.
    These actions are scaled and interpreted to form actionable market bids, with specific conditions
    dictating the bid type.

    - **Bid Price**: The one action value determines the price at which the agent will bid.

    Rewards are based on the profit generated by the agent's market bids.

    Attributes
    ----------
    foresight : int
        Number of time steps for forecasting market conditions. Defaults to 24.
    max_bid_price : float
        Maximum allowable bid price. Defaults to 100.
    device : str
        Device used for computation ("cpu" or "cuda"). Defaults to "cpu".
    float_type : str
        Data type for floating-point calculations. Defaults to "float32".
    learning_mode : bool
        Whether the agent is in learning mode. Defaults to False.
    algorithm : str
        RL algorithm used by the agent. Defaults to "matd3".
    actor_architecture_class : type[torch.nn.Module]
        Class of the neural network for the actor network. Defaults to MLPActor.
    actor : torch.nn.Module
        The neural network used to predict actions.
    order_types : list[str]
        Types of market orders used by the strategy. Defaults to ["SB"].
    action_noise : NormalActionNoise
        Noise model added to actions during learning for exploration. Defaults to None.
    collect_initial_experience_mode : bool
        Whether the agent is in an exploration mode for initial experience. Defaults to True.

    Args
    ----
    *args : Variable length argument list.
    **kwargs : Arbitrary keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        # 'foresight' represents the number of time steps into the future that we will consider
        # when constructing the observations.
        foresight = kwargs.pop("foresight", 24)
        act_dim = kwargs.pop("act_dim", 1)
        unique_obs_dim = kwargs.pop("unique_obs_dim", 3)
        super().__init__(
            foresight=foresight,
            act_dim=act_dim,
            unique_obs_dim=unique_obs_dim,
            *args,
            **kwargs,
        )

        # define allowed order types
        self.order_types = kwargs.get("order_types", ["SB"])

    def get_individual_observations(
        self, unit: SupportsMinMaxCharge, start: datetime, end: datetime
    ):
        """
        Retrieves the unit-specific observations for renewable units. For them we add the availabilities of the generation in the observation space.

        Args
        ----
        unit : SupportsMinMaxCharge
            Storage unit providing forecasted and current state data.
        start : datetime.datetime
            Start time for the observation period.
        end : datetime.datetime
            End time for the observation period.

        Returns
        -------
        individual_observations: np.ndarray
            Array containing state of charge and energy cost.

        Notes
        -----
        Observations are scaled by the unit's max power and the max bid price.
        """
        # --- Current volume & marginal cost ---
        current_volume = unit.get_output_before(start)
        current_costs = unit.calculate_marginal_cost(start, current_volume)
        _, available_power = unit.calculate_min_max_power(start, end)

        scaled_total_dispatch = current_volume / unit.max_power
        scaled_marginal_cost = current_costs / self.max_bid_price
        scaled_available_power = available_power[0] / unit.max_power

        individual_observations = np.array(
            [scaled_total_dispatch, scaled_marginal_cost, scaled_available_power]
        )

        return individual_observations

    def calculate_reward(
        self,
        unit: SupportsMinMaxCharge,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ):
        """
        Calculates the reward for the renewable unit based on profits, costs, and opportunity costs from market transactions.

        Args
        ----
        unit : SupportsMinMax
            The unit for which to calculate the reward.
        marketconfig : MarketConfig
            Market configuration settings.
        orderbook : Orderbook
            Orderbook containing executed bids and details.

        Notes
        -----
        The reward is computed by combining the following:
        **Profit**: Income from accepted bids minus marginal and start-up costs.
        **Opportunity Cost**: Penalty for underutilizing capacity, calculated as potential lost income.
        **Regret Term**: A scaled regret term penalizes high opportunity costs to guide effective bidding.

        The reward is scaled and stored along with other outputs in the unit’s data to support learning.
        """
        # Function is called after the market is cleared, and we get the market feedback,
        # allowing us to calculate profit based on the realized transactions.

        product_type = marketconfig.product_type

        start = orderbook[0]["start_time"]
        end = orderbook[0]["end_time"]
        # `end_excl` marks the last product's start time by subtracting one frequency interval.
        end_excl = end - unit.index.freq

        # Depending on how the unit calculates marginal costs, retrieve cost values.
        marginal_cost = unit.calculate_marginal_cost(
            start, unit.outputs[product_type].at[start]
        )
        market_clearing_price = orderbook[0]["accepted_price"]

        duration = (end - start) / timedelta(hours=1)

        income = 0.0
        operational_cost = 0.0

        accepted_volume_total = 0
        offered_volume_total = 0

        # Iterate over all orders in the orderbook to calculate order-specific profit.
        for order in orderbook:
            accepted_volume = order.get("accepted_volume", 0)
            accepted_volume_total += accepted_volume

            offered_volume_total += order["volume"]

            # Calculate profit as income minus operational cost for this event.
            order_income = market_clearing_price * accepted_volume * duration
            order_cost = marginal_cost * accepted_volume * duration

            # Accumulate income and operational cost for all orders.
            income += order_income
            operational_cost += order_cost

        # Consideration of start-up costs, divided evenly between upward and downward regulation events.
        if (
            unit.outputs[product_type].at[start] != 0
            and unit.outputs[product_type].at[start - unit.index.freq] == 0
        ):
            operational_cost += unit.hot_start_cost / 2
        elif (
            unit.outputs[product_type].at[start] == 0
            and unit.outputs[product_type].at[start - unit.index.freq] != 0
        ):
            operational_cost += unit.hot_start_cost / 2

        profit = income - operational_cost

        # Stabilizing learning: Limit positive profit to 50% of its absolute value.
        # This reduces variance in rewards and prevents overfitting to extreme profit-seeking behavior.
        # However, this does NOT prevent the agent from exploiting market inefficiencies if they exist.
        # RL by nature identifies and exploits system weaknesses if they lead to higher profit.
        # This is not a price cap but rather a stabilizing factor to avoid reward spikes affecting learning stability.
        # IMPORTANT: This is a clear case of reward_tuning to stabilize learning - Use with caution!
        # profit_scale = 0.5

        profit_scale = 1
        profit = min(profit, profit_scale * abs(profit))

        # get potential maximum infeed according to availability from order volume
        # Note: this will only work as the correct reference point when the volume is not defined by an action
        # using a call unit_calculate_min_max_power here would be false since the dispatch of the order is already set, leading to no available power anymore
        available_power = offered_volume_total

        # Opportunity cost: The income lost due to not operating at full capacity.
        opportunity_cost = (
            (market_clearing_price - marginal_cost)
            * (available_power - accepted_volume_total)
            * duration
        )

        # If opportunity cost is negative, no income was lost, so we set it to zero.
        opportunity_cost = max(opportunity_cost, 0)

        # Dynamic regret scaling:
        # - If accepted volume is positive, apply lower regret (0.1) to avoid punishment for being on the edge of the merit order.
        # - If no dispatch happens, apply higher regret (0.5) to discourage idle behavior, if it could have been profitable.
        regret_scale = 0.1 if accepted_volume_total > unit.min_power else 0.5

        # --------------------
        # 4.1 Calculate Reward
        # Instead of directly setting reward = profit, we incorporate a regret term (opportunity cost penalty).
        # This guides the agent toward strategies that maximize accepted bids while minimizing lost opportunities.

        # scaling factor to normalize the reward to the range [-1,1]
        if available_power == 0:
            scaling = 0
        else:
            scaling = 1 / (self.max_bid_price * available_power)

        regret = regret_scale * opportunity_cost
        reward = scaling * (profit - regret)

        # Store results in unit outputs
        # Note: these are not learning-specific results but stored for all units for analysis
        unit.outputs["profit"].loc[start:end_excl] += profit
        unit.outputs["total_costs"].loc[start:end_excl] += operational_cost

        # write rl-rewards to buffer
        if self.learning_mode:
            self.learning_role.add_reward_to_cache(
                unit.id, start, reward, regret, profit
            )
