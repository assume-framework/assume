# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import numpy as np
import pandas as pd
import torch as th
from torch.nn import functional as F
from torch.optim import AdamW

from assume.reinforcement_learning.algorithms.base_algorithm import (
    ActorCriticAlgorithm,
    RLAlgorithm,
)
from assume.reinforcement_learning.buffer import RolloutBuffer
from assume.reinforcement_learning.learning_utils import transform_buffer_data
from assume.reinforcement_learning.neural_network_architecture import (
    ActorPPO,
    CriticPPO,
    LSTMActorPPO,
)

logger = logging.getLogger(__name__)


class PPO(ActorCriticAlgorithm):
    """
    Proximal Policy Optimization (PPO) Algorithm.

    A policy gradient method that alternates between sampling data through
    interaction with the environment, and optimizing a surrogate objective
    function using stochastic gradient ascent. It is an on-policy algorithm.

    Attributes:
        clip_range: The epsilon parameter for PPO clipping.
        clip_range_vf: The epsilon parameter for value function clipping.
        n_epochs: Number of optimization epochs per rollout.
        entropy_coef: Coefficient for entropy term in loss calculation.
        vf_coef: Coefficient for value function term in loss calculation.
        max_grad_norm: Maximum gradient norm for clipping.
        n_updates: Counter for gradient updates performed.
        actor_architecture_class: Actor network architecture class.
        critic_architecture_class: Critic network architecture class.

    Example:
        >>> ppo = PPO(learning_role)
        >>> ppo.update_policy()
    """

    # On-policy: also cache value estimates, log-probs, and done flags
    # collected per time-step for GAE computation.
    buffer_fields = RLAlgorithm.buffer_fields + ("values", "log_probs", "dones")

    def __init__(
        self,
        learning_role,
        clip_range=None,
        clip_range_vf=None,
        n_epochs=None,
        entropy_coef=None,
        vf_coef=None,
        max_grad_norm=None,
    ):
        """Initialize PPO algorithm with specific hyperparameters.

        Args:
            learning_role: The primary learning role object.
            clip_range: The epsilon parameter for PPO policy clipping.
            clip_range_vf: The epsilon parameter for value function clipping.
            n_epochs: Number of optimization epochs per rollout.
            entropy_coef: Coefficient for entropy term in loss.
            vf_coef: Coefficient for value function term in loss.
            max_grad_norm: Maximum gradient norm for clipping.
        """
        super().__init__(learning_role)

        # Set PPO-specific architecture classes
        self.actor_architecture_class = ActorPPO
        self.critic_architecture_class = CriticPPO

        config = self.learning_config
        on_policy_config = config.on_policy

        # Using on-policy config unless explicitly overridden via constructor args.
        self.clip_range = (
            clip_range if clip_range is not None else on_policy_config.clip_ratio
        )
        self.clip_range_vf = clip_range_vf
        self.n_epochs = n_epochs if n_epochs is not None else on_policy_config.n_epochs
        self.entropy_coef = (
            entropy_coef if entropy_coef is not None else on_policy_config.entropy_coef
        )
        self.vf_coef = vf_coef if vf_coef is not None else on_policy_config.vf_coef
        self.max_grad_norm = (
            max_grad_norm
            if max_grad_norm is not None
            else on_policy_config.max_grad_norm
        )

        # Update counter
        self.n_updates = 0

    # =========================================================================
    # CHECKPOINT SAVING METHODS
    # =========================================================================

    uses_target_networks: bool = False

    # Note: save_params, save_critic_params, save_actor_params, load_params,
    # load_critic_params, load_actor_params, initialize_policy are inherited from A2CAlgorithm

    def get_action(
        self, strategy, obs: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, dict[str, object] | None]:
        """Sample a stochastic action.

        In learning mode the actor's Gaussian policy is sampled and the
        value estimate and log-probability are returned as `extra_data` for
        the caller to cache alongside the action (see
        ``Learning.add_actions_to_cache``), for later use in
        _store_to_buffer_and_update_sync. In evaluation mode the
        deterministic mean action is returned instead, with no extra_data.

        PPO does *not* have an initial-exploration phase — the stochastic
        policy provides sufficient exploration from the very first episode.

        Note: the value estimate returned here is a placeholder (``0.0``) —
        for MAPPO it is recomputed centrally in
        ``_store_to_buffer_and_update_sync`` using the centralized critic.
        """
        if strategy.learning_mode and not strategy.evaluation_mode:
            action, log_prob = strategy.actor.get_action_and_log_prob(obs.unsqueeze(0))
            action = action.squeeze(0).detach()
            log_prob = log_prob.squeeze(0).detach()
            if hasattr(log_prob, "item"):
                log_prob = log_prob.item()
            noise = th.zeros_like(action, dtype=strategy.float_type)
            extra_data = {"values": 0.0, "log_probs": log_prob, "dones": 0.0}
            return action, noise, extra_data

        # Evaluation
        action = strategy.actor(obs, deterministic=True).detach()
        noise = th.zeros_like(action, dtype=strategy.float_type)
        return action, noise, None

    def create_buffer(self, time_step) -> RolloutBuffer:
        """Create the rollout buffer holding exactly one update window.

        On-policy learning discards its experience after every update, so the
        buffer only has to span the `train_freq` window between two updates —
        `train_freq / time_step` transitions. At least 2, since
        ``update_policy`` reserves the last transition to bootstrap
        V(s_{t+1}) and trains on the rest.
        """
        train_freq = pd.Timedelta(str(self.learning_config.train_freq))
        rollout_buffer_size = max(2, int(train_freq / pd.Timedelta(time_step)))

        return RolloutBuffer(
            buffer_size=rollout_buffer_size,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            n_rl_units=len(self.learning_role.rl_strats),
            device=self.device,
            float_type=self.float_type,
            gamma=self.learning_config.gamma,
            gae_lambda=self.learning_config.on_policy.gae_lambda,
        )

    def _centralized_values(self, obs: np.ndarray) -> np.ndarray:
        """Evaluate every agent's centralized critic on one joint observation.

        Args:
            obs: Joint observation for a single time-step,
                shape (n_agents, obs_dim).

        Returns:
            Value estimate per agent, shape (n_agents,), in the agent order of
            ``learning_role.rl_strats``.
        """
        strategies = list(self.learning_role.rl_strats.values())
        values = np.zeros(len(strategies))

        # agent-specific slice of every agent's observation
        unique_obs_all = obs[:, self.obs_dim - self.unique_obs_dim :]

        with th.no_grad():
            for i, strategy in enumerate(strategies): # TODO: does this need to be ordered by unit_id?
                other_unique = np.concatenate(
                    (unique_obs_all[:i], unique_obs_all[i + 1 :]), axis=0
                )
                centralized_obs = np.concatenate(
                    (obs[i : i + 1], other_unique.reshape(1, -1)), axis=1
                )
                obs_tensor = th.as_tensor(
                    centralized_obs, device=self.device, dtype=self.float_type
                )
                values[i] = strategy.critics(obs_tensor).cpu().numpy().reshape(-1)[0]

        return values

    def store_experience(self, cache: dict, device) -> None:
        """Append the update window to the rollout buffer, one time-step at a time.

        Unlike the replay buffer, the rollout buffer stores single transitions,
        and each of them needs a value estimate V(s_t) from the centralized
        critic. That value cannot be produced at action time — a centralized
        value needs *all* agents' observations for the same timestep, and an
        agent only has its own — so ``get_action`` caches a placeholder and the
        real value is computed here, where the joint observation first exists.
        The critic is unchanged over the whole window (``update_policy`` runs
        only after this method), so these values are still the behaviour
        policy's, as PPO requires.
        """
        unit_id_order = list(self.learning_role.rl_strats.keys())

        for timestamp in sorted(cache["obs"].keys()):
            missing_units = [
                u
                for u in unit_id_order
                if u not in cache["obs"][timestamp]
                or u not in cache["actions"][timestamp]
                or u not in cache["rewards"][timestamp]
                or u not in cache["log_probs"][timestamp]
                or u not in cache["dones"][timestamp]
            ]
            if missing_units:
                logger.warning(
                    "Skipping on-policy rollout step at %s: missing data for units %s. "
                    "This usually means a learning unit failed to report an "
                    "observation/action/reward/log_prob/done for this timestep, "
                    "and we do not fill the buffer with default values instead.",
                    timestamp,
                    missing_units,
                )
                continue

            step = {
                field: transform_buffer_data(
                    {timestamp: cache[field][timestamp]}, device, unit_id_order
                )
                for field in ("obs", "actions", "rewards", "dones", "log_probs")
            }

            # Recompute V(s_t) centrally, overriding the placeholder that
            # get_action cached, and reshape to the buffer's (1, n_agents, 1).
            values_data = (
                self._centralized_values(step["obs"][0])
                .reshape(1, -1, 1)
                .astype(np.float32)
            )

            self.buffer.add(
                obs=step["obs"],
                action=step["actions"],
                reward=step["rewards"],
                done=step["dones"],
                value=values_data,
                log_prob=step["log_probs"],
            )

    def compute_gradient_step_range(
        self, unit_params_list: list[dict]
    ) -> tuple[range, int]:
        """On-policy step counting.

        PPO/MAPPO have no "initial experience" phase and no fixed configured
        `gradient_steps` — the number of steps performed in this update
        simply equals `len(unit_params_list)`. But `Learning.update_steps`
        still resets to 0 every episode, so — just like the off-policy
        default — we still need a cross-episode offset, here based on
        `episodes_done` (no initial-experience subtraction needed).
        """
        actual_gradient_steps = len(unit_params_list)
        gradient_step_range = range(actual_gradient_steps)

        # steps performed in previous training episodes
        steps_done_in_previous_episodes = (
            self.learning_role.episodes_done
            * self._updates_per_episode()
            * actual_gradient_steps
        )
        base_step = (
            steps_done_in_previous_episodes
            + self.learning_role.update_steps * actual_gradient_steps
        )
        return gradient_step_range, base_step

    def create_actors(self) -> None:
        """Create stochastic actor networks for all agents.

        Initializes the ActorPPO or LSTMActorPPO network based on the configuration,
        as well as its optimizer for each agent strategy.

        Example:
            >>> ppo.create_actors()
            >>> # Creates actor network and optimizer for each strategy
        """
        actor_architecture = self.learning_config.on_policy.actor_architecture

        for strategy in self.learning_role.rl_strats.values():
            # Create PPO Actor
            if actor_architecture == "lstm":
                strategy.actor = LSTMActorPPO(
                    obs_dim=self.obs_dim,
                    act_dim=self.act_dim,
                    float_type=self.float_type,
                    unique_obs_dim=self.unique_obs_dim,
                    num_timeseries_obs_dim=strategy.num_timeseries_obs_dim,
                ).to(self.device)
            else:
                strategy.actor = ActorPPO(
                    obs_dim=self.obs_dim,
                    act_dim=self.act_dim,
                    float_type=self.float_type,
                ).to(self.device)

            # Create Optimizer
            strategy.actor.optimizer = AdamW(
                strategy.actor.parameters(),
                lr=self.learning_role.calc_lr_from_progress(1),
            )

            strategy.actor.loaded = False

    def create_critics(self) -> None:
        """Create value networks for all agents.

        Initializes the CriticPPO network (Centralized Critic) and its optimizer
        for each registered agent strategy.

        Example:
            >>> ppo.create_critics()
            >>> # Creates critic networks and optimizers for each strategy
        """
        n_agents = len(self.learning_role.rl_strats)

        for strategy in self.learning_role.rl_strats.values():
            # Create value network
            strategy.critics = CriticPPO(
                n_agents=n_agents,
                obs_dim=self.obs_dim,
                unique_obs_dim=self.unique_obs_dim,
                float_type=self.float_type,
            ).to(self.device)

            # Create optimizer
            strategy.critics.optimizer = AdamW(
                strategy.critics.parameters(),
                lr=self.learning_role.calc_lr_from_progress(1),
            )

    def extract_policy(self) -> dict:
        """Extract all actor and critic networks into a dictionary.

        Collects actor and critic networks from all learning strategies into
        a structured dictionary.

        Returns:
            Dictionary containing all network components organized by type:
                - 'actors': Primary actor networks
                - 'critics': Primary critic networks
                - Dimension information for reconstruction

        Example:
            >>> policy_dict = ppo.extract_policy()
            >>> # Contains all networks ready for saving or transfer
        """
        actors = {}
        critics = {}

        for u_id, strategy in self.learning_role.rl_strats.items():
            actors[u_id] = strategy.actor
            critics[u_id] = strategy.critics

        return {
            "actors": actors,
            "critics": critics,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "unique_obs_dim": self.unique_obs_dim,
        }

    # =========================================================================
    # CORE TRAINING: POLICY UPDATE
    # =========================================================================

    def update_policy(self) -> None:
        """Update actor and critic networks using Proximal Policy Optimization (PPO).

        Performs one complete training iteration consisting of:
        1. Checking if enough data is collected in the rollout buffer.
        2. Computing Generalized Advantage Estimation (GAE) and Returns using the last value estimate.
        3. Updating the Actor and Critic networks over multiple epochs using mini-batches.
        4. Calculating the surrogate objective with clipping.
        5. Calculating value function loss (MSE) and entropy bonus.
        6. Logging metrics and gradients.
        7. Clearing the on-policy buffer after the update.
        """
        logger.debug("Updating Policy (PPO)")

        # Keeping strategy order aligned with rollout-buffer column order.
        strategies = [strategy for strategy in self.learning_role.rl_strats.values()]
        n_rl_agents = len(strategies)

        # Getting the buffer, this will be a RolloutBuffer for on-policy algorithms.
        rollout_buffer = self.buffer

        # Check if rollout buffer has data
        if rollout_buffer is None or rollout_buffer.pos == 0:
            logger.debug("Rollout buffer is empty, skipping policy update")
            return

        # Require at least two transitions because we reserve the final one
        # for bootstrapping V(s_{t+1}) and train on the remaining rollout.
        if rollout_buffer.pos < 2:
            logger.debug(
                "Rollout buffer has fewer than 2 samples, skipping policy update."
            )
            return

        # Update learning rate
        progress_remaining = self.learning_role.get_progress_remaining()
        learning_rate = self.learning_role.calc_lr_from_progress(progress_remaining)

        for strategy in strategies:
            for param_group in strategy.critics.optimizer.param_groups:
                param_group["lr"] = learning_rate
            for param_group in strategy.actor.optimizer.param_groups:
                param_group["lr"] = learning_rate

        # Get last values for advantage computation
        last_values = np.zeros(n_rl_agents)
        dones = np.zeros(n_rl_agents)

        # Get the buffer size to index into the last stored state
        buffer_size = (
            rollout_buffer.pos
            if not rollout_buffer.full
            else rollout_buffer.buffer_size
        )

        if buffer_size > 0:
            # Use the LAST observation as the bootstrap for the REST of the buffer.
            # We sacrifice the last step (pos-1) to serve as s_{t+1} for the step before it.
            # This ensures V(s_{t+1}) is calculated using the REAL next state, not a self-
            # referential V(s_{t}).
            last_idx = buffer_size - 1
            last_obs = rollout_buffer.observations[last_idx]

            if last_idx > 0:
                last_dones = rollout_buffer.dones[last_idx - 1]
            else:
                last_dones = rollout_buffer.dones[last_idx]

            # Reduce buffer size by 1 so as to not train on the bootstrap step
            rollout_buffer.pos -= 1
            if rollout_buffer.full:
                rollout_buffer.full = False  # If it was full, it's not anymore

            # Bootstrap value, from the same centralized critics that produced
            # the V(s_t) already stored in the buffer by store_experience.
            last_values = self._centralized_values(last_obs)
            dones = last_dones.copy() # TODO: is this the correct behavior?

        # Compute advantages and returns
        rollout_buffer.compute_returns_and_advantages(last_values, dones)

        # Initialize metrics storage
        all_actor_losses = []
        all_critic_losses = []
        all_entropy_losses = []

        # Initialize unit_params for gradient logging
        # Use an empty list that will be dynamically extended
        unit_params = []
        step_count = 0

        # Helper to create a new step entry
        def create_step_entry():
            return {
                u_id: {
                    "actor_loss": None,
                    "actor_total_grad_norm": None,
                    "actor_max_grad_norm": None,
                    "critic_loss": None,
                    "critic_total_grad_norm": None,
                    "critic_max_grad_norm": None,
                }
                for u_id in self.learning_role.rl_strats.keys()
            }

        effective_batch_size = min(
            self.learning_config.batch_size,
            rollout_buffer.pos
            if not rollout_buffer.full
            else rollout_buffer.buffer_size,
        )

        for epoch in range(self.n_epochs):
            for batch in rollout_buffer.get(effective_batch_size):
                current_batch_size = batch.observations.shape[0]

                # Precompute unique observation parts for centralized critic
                unique_obs_from_others = batch.observations[
                    :, :, self.obs_dim - self.unique_obs_dim :
                ].reshape(current_batch_size, n_rl_agents, -1)

                for i, strategy in enumerate(strategies):
                    actor = strategy.actor
                    critic = strategy.critics

                    obs_i = batch.observations[:, i, :]

                    # Construct centralized state
                    other_unique_obs = th.cat(
                        (
                            unique_obs_from_others[:, :i],
                            unique_obs_from_others[:, i + 1 :],
                        ),
                        dim=1,
                    )
                    all_states = th.cat(
                        (
                            obs_i.reshape(current_batch_size, -1),
                            other_unique_obs.reshape(current_batch_size, -1),
                        ),
                        dim=1,
                    )

                    actions_i = batch.actions[:, i, :]
                    old_log_probs_i = batch.old_log_probs[:, i]
                    advantages_i = batch.advantages[:, i]
                    returns_i = batch.returns[:, i]
                    old_values_i = batch.old_values[:, i]

                    # Normalize advantages across the entire batch, not per-mini-batch
                    # This provides more stable training
                    advantages_flat = advantages_i.flatten()
                    advantages_i = (advantages_i - advantages_flat.mean()) / (
                        advantages_flat.std() + 1e-8
                    )

                    log_probs, entropy = actor.evaluate_actions(obs_i, actions_i)
                    values = critic(all_states).flatten()

                    # Importance sampling ratio
                    ratio = th.exp(log_probs - old_log_probs_i)

                    # Clipped surrogate objective
                    policy_loss_1 = advantages_i * ratio
                    policy_loss_2 = advantages_i * th.clamp(
                        ratio, 1 - self.clip_range, 1 + self.clip_range
                    )
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                    # Entropy loss
                    entropy_loss = -self.entropy_coef * entropy.mean()

                    if self.clip_range_vf is not None:
                        # Clipped value function loss
                        values_clipped = old_values_i + th.clamp(
                            values - old_values_i,
                            -self.clip_range_vf,
                            self.clip_range_vf,
                        )
                        value_loss_1 = F.mse_loss(values, returns_i)
                        value_loss_2 = F.mse_loss(values_clipped, returns_i)
                        value_loss = th.max(value_loss_1, value_loss_2)
                    else:
                        value_loss = F.mse_loss(values, returns_i)

                    loss = policy_loss + entropy_loss + self.vf_coef * value_loss

                    # Actor update
                    actor.optimizer.zero_grad()
                    critic.optimizer.zero_grad()
                    loss.backward()

                    # Calculate gradient norms BEFORE clipping
                    actor_params = list(actor.parameters())
                    critic_params = list(critic.parameters())

                    actor_max_grad_norm = max(
                        (
                            p.grad.norm().item()
                            for p in actor_params
                            if p.grad is not None
                        ),
                        default=0.0,
                    )
                    critic_max_grad_norm = max(
                        (
                            p.grad.norm().item()
                            for p in critic_params
                            if p.grad is not None
                        ),
                        default=0.0,
                    )

                    # Gradient clipping
                    actor_total_grad_norm = th.nn.utils.clip_grad_norm_(
                        actor.parameters(), self.max_grad_norm
                    )
                    critic_total_grad_norm = th.nn.utils.clip_grad_norm_(
                        critic.parameters(), self.max_grad_norm
                    )

                    actor.optimizer.step()
                    critic.optimizer.step()

                    # Store metrics
                    all_actor_losses.append(policy_loss.item())
                    all_critic_losses.append(value_loss.item())
                    all_entropy_losses.append(entropy_loss.item())

                    # Ensure we have an entry for this step
                    if step_count >= len(unit_params):
                        unit_params.append(create_step_entry())

                    # Store per-unit gradient params for this step
                    unit_params[step_count][strategy.unit_id]["actor_loss"] = (
                        policy_loss.item()
                    )
                    unit_params[step_count][strategy.unit_id]["critic_loss"] = (
                        value_loss.item()
                    )
                    unit_params[step_count][strategy.unit_id][
                        "actor_total_grad_norm"
                    ] = (
                        actor_total_grad_norm.item()
                        if isinstance(actor_total_grad_norm, th.Tensor)
                        else actor_total_grad_norm
                    )
                    unit_params[step_count][strategy.unit_id]["actor_max_grad_norm"] = (
                        actor_max_grad_norm
                    )
                    unit_params[step_count][strategy.unit_id][
                        "critic_total_grad_norm"
                    ] = (
                        critic_total_grad_norm.item()
                        if isinstance(critic_total_grad_norm, th.Tensor)
                        else critic_total_grad_norm
                    )
                    unit_params[step_count][strategy.unit_id][
                        "critic_max_grad_norm"
                    ] = critic_max_grad_norm

                step_count += 1

        self.n_updates += 1

        # Write gradient params to output
        self.learning_role.write_rl_grad_params_to_output(learning_rate, unit_params)

        # Clear rollout buffer
        rollout_buffer.reset()

        logger.debug(
            f"PPO update complete. Actor loss: {np.mean(all_actor_losses):.4f}, "
            f"Value loss: {np.mean(all_critic_losses):.4f}"
        )
