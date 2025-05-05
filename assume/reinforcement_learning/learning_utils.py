# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from collections.abc import Callable
from datetime import datetime
from typing import TypedDict

import numpy as np
import torch as th
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class ObsActRew(TypedDict):
    observation: list[th.Tensor]
    action: list[th.Tensor]
    reward: list[th.Tensor]


observation_dict = dict[list[datetime], ObsActRew]

# A schedule takes the remaining progress as input
# and outputs a scalar (e.g. learning rate, action noise scale ...)
Schedule = Callable[[float], float]


# Ornstein-Uhlenbeck Noise
# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    """
    A class that implements Ornstein-Uhlenbeck noise.
    """

    def __init__(self, action_dimension, mu=0, sigma=0.5, theta=0.15, dt=1e-2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.noise_prev = np.zeros(self.action_dimension)
        self.noise_prev = (
            self.initial_noise
            if self.initial_noise is not None
            else np.zeros(self.action_dimension)
        )

    def noise(self):
        noise = (
            self.noise_prev
            + self.theta * (self.mu - self.noise_prev) * self.dt
            + self.sigma
            * np.sqrt(self.dt)
            * np.random.normal(size=self.action_dimension)
        )
        self.noise_prev = noise

        return noise


class NormalActionNoise:
    """
    A Gaussian action noise that supports direct tensor creation on a given device.
    """

    def __init__(self, action_dimension, mu=0.0, sigma=0.1, scale=1.0, dt=0.9998):
        self.act_dimension = action_dimension
        self.mu = mu
        self.sigma = sigma
        self.scale = scale
        self.dt = dt

    def noise(self, device=None, dtype=th.float):
        """
        Generates noise using torch.normal(), ensuring efficient execution on GPU if needed.

        Args:
        - device (torch.device, optional): Target device (e.g., 'cuda' or 'cpu').
        - dtype (torch.dtype, optional): Data type of the tensor (default: torch.float32).

        Returns:
        - torch.Tensor: Noise tensor on the specified device.
        """
        return (
            self.dt
            * self.scale
            * th.normal(
                mean=self.mu,
                std=self.sigma,
                size=(self.act_dimension,),
                dtype=dtype,
                device=device,
            )
        )

    def update_noise_decay(self, updated_decay: float):
        self.dt = updated_decay


def polyak_update(params, target_params, tau: float):
    """
    Perform a Polyak average update on ``target_params`` using ``params``:
    target parameters are slowly updated towards the main parameters.
    ``tau``, the soft update coefficient controls the interpolation:
    ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
    The Polyak update is done in place, with ``no_grad``, and therefore does not create intermediate tensors,
    or a computation graph, reducing memory cost and improving performance.  We scale the target params
    by ``1-tau`` (in-place), add the new weights, scaled by ``tau`` and store the result of the sum in the target
    params (in place).
    See https://github.com/DLR-RM/stable-baselines3/issues/93

    Args:
        params: parameters to use to update the target params
        target_params: parameters to update
        tau: the soft update coefficient ("Polyak update", between 0 and 1)
    """
    with th.no_grad():
        for param, target_param in zip(params, target_params):
            target_param.lerp_(param, tau)  # More efficient in-place operation


def linear_schedule_func(
    start: float, end: float = 0, end_fraction: float = 1
) -> Schedule:
    """
    Create a function that interpolates linearly between start and end
    between ``progress_remaining`` = 1 and ``progress_remaining`` = 1 - ``end_fraction``.

    Args:
        start: value to start with if ``progress_remaining`` = 1
        end: value to end with if ``progress_remaining`` = 0
        end_fraction: fraction of ``progress_remaining``
            where end is reached e.g 0.1 then end is reached after 10%
            of the complete training process.

    Returns:
        Linear schedule function.

    Note:
        Adapted from SB3: https://github.com/DLR-RM/stable-baselines3/blob/512eea923afad6f6da4bb53d72b6ea4c6d856e59/stable_baselines3/common/utils.py#L100

    """

    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) > end_fraction:
            return end
        else:
            return start + (1 - progress_remaining) * (end - start) / end_fraction

    return func


def constant_schedule(val: float) -> Schedule:
    """
    Create a function that returns a constant. It is useful for learning rate schedule (to avoid code duplication)

    Args:
        val: constant value
    Returns:
        Constant schedule function.

    Note:
        From SB3: https://github.com/DLR-RM/stable-baselines3/blob/512eea923afad6f6da4bb53d72b6ea4c6d856e59/stable_baselines3/common/utils.py#L124

    """

    def func(_):
        return val

    return func


def get_hidden_sizes(state_dict: dict, prefix: str) -> list[int]:
    sizes = []
    i = 0
    while f"{prefix}.{i}.weight" in state_dict:
        weight = state_dict[f"{prefix}.{i}.weight"]
        out_dim = weight.shape[0]
        sizes.append(out_dim)
        i += 1
    return sizes[:-1]  # exclude the final output layer if needed


def copy_layer_data(dst, src):
    for k in dst:
        if k in src and dst[k].shape == src[k].shape:
            dst[k].data.copy_(src[k].data)


def transfer_weights(
    model: th.nn.Module,
    old_state: dict,
    old_id_order: list[str],
    new_id_order: list[str],
    obs_base: int,
    act_dim: int,
    unique_obs: int,
) -> dict | None:
    """
    Copy only those obs_ and action-slices for matching IDs.
    New IDs keep their original (random) weights.
    """
    # 1) Architecture check
    new_state = model.state_dict()
    old_hidden = get_hidden_sizes(old_state, prefix="q1_layers")
    new_hidden = get_hidden_sizes(new_state, prefix="q1_layers")
    if old_hidden != new_hidden:
        logger.warning(
            f"Cannot transfer weights: architecture mismatch.\n"
            f"Old sizes: {old_hidden}, New sizes: {new_hidden}."
        )
        return None

    # 2) Compute total dims
    old_n = len(old_id_order)
    new_n = len(new_id_order)
    old_obs_tot = obs_base + unique_obs * max(0, old_n - 1)
    new_obs_tot = obs_base + unique_obs * max(0, new_n - 1)

    # 3) Clone new state
    new_state_copy = {k: v.clone() for k, v in new_state.items()}

    # 4) Transfer per-prefix
    for prefix in ("q1_layers", "q2_layers"):
        w_old = old_state[f"{prefix}.0.weight"]
        b_old = old_state[f"{prefix}.0.bias"]
        w_new = new_state_copy[f"{prefix}.0.weight"]
        b_new = new_state_copy[f"{prefix}.0.bias"]
        orig_w = new_state[f"{prefix}.0.weight"].clone()

        # a) shared obs_base
        w_new[:, :obs_base] = w_old[:, :obs_base]

        # b) matched-ID blocks
        for new_idx, u in enumerate(new_id_order):
            if u not in old_id_order:
                continue
            old_idx = old_id_order.index(u)

            # unique_obs for agents beyond the first
            if new_idx > 0 and old_idx > 0:
                ns = obs_base + unique_obs * (new_idx - 1)
                os_ = obs_base + unique_obs * (old_idx - 1)
                w_new[:, ns : ns + unique_obs] = w_old[:, os_ : os_ + unique_obs]

            # action blocks for every agent
            nact = new_obs_tot + act_dim * new_idx
            oact = old_obs_tot + act_dim * old_idx
            w_new[:, nact : nact + act_dim] = w_old[:, oact : oact + act_dim]

        # c) restore unmatched agents’ unique_obs
        for new_idx, u in enumerate(new_id_order):
            if new_idx == 0 or u in old_id_order:
                continue
            ns = obs_base + unique_obs * (new_idx - 1)
            w_new[:, ns : ns + unique_obs] = orig_w[:, ns : ns + unique_obs]
            # actions untouched

        # d) bias and deeper layers
        b_new.copy_(b_old)
        for i in range(1, len(new_hidden) + 1):
            new_state_copy[f"{prefix}.{i}.weight"].copy_(
                old_state[f"{prefix}.{i}.weight"]
            )
            new_state_copy[f"{prefix}.{i}.bias"].copy_(old_state[f"{prefix}.{i}.bias"])

    return new_state_copy


def cluster_strategies_k_means(strategies: list, n_groups: int, random_state: int):
    contexts = np.array(
        [strategy.context.detach().cpu().numpy() for strategy in strategies]
    )

    kmeans = KMeans(n_clusters=n_groups, random_state=random_state)
    labels = kmeans.fit_predict(contexts)

    clusters = {i: [] for i in range(n_groups)}
    for strategy, label in zip(strategies, labels):
        clusters[label].append(strategy)

    return clusters


def cluster_strategies_max_size(strategies: list, max_size: int, random_state: int):
    """
    Clusters strategies such that no cluster exceeds `max_size`.

    Args:
        strategies (list): List of strategy objects with `.context` attributes.
        max_size (int): Maximum allowed number of strategies per cluster.
        random_state (int): Seed for reproducible KMeans clustering.

    Returns:
        dict[int, list]: Dictionary of clustered strategies.

    Raises:
        ValueError: If a cluster still exceeds `max_size` after processing.
        RecursionError: If too many recursive splits occur.
    """

    n_clusters = max(1, len(strategies) // max_size)

    clusters = cluster_strategies_k_means(
        strategies=strategies, n_groups=n_clusters, random_state=random_state
    )

    def split_clusters(clusters_dict):
        final_clusters = {}
        cluster_id_counter = 0

        for cluster in clusters_dict.values():
            if len(cluster) <= max_size:
                final_clusters[cluster_id_counter] = cluster
                cluster_id_counter += 1
            else:
                # Check for identical contexts before attempting split
                sub_contexts = np.array(
                    [s.context.detach().cpu().numpy() for s in cluster]
                )
                unique_sub_contexts = np.unique(sub_contexts, axis=0)

                if len(unique_sub_contexts) <= 1:
                    logger.warning(
                        f"Cannot split cluster of size {len(cluster)} further "
                        f"as contexts are identical. Keeping oversized cluster (violates max_size)."
                    )
                    final_clusters[cluster_id_counter] = cluster
                    cluster_id_counter += 1
                    continue

                # Split cluster further
                n_subclusters = max(2, len(cluster) // max_size)
                sub_clusters = cluster_strategies_k_means(
                    strategies=cluster,
                    n_groups=n_subclusters,
                    random_state=random_state
                    + len(cluster),  # add cluster size to random state for uniqueness
                )

                # Recursively process sub-clusters
                sub_result = split_clusters(sub_clusters)
                for sub_cluster in sub_result.values():
                    final_clusters[cluster_id_counter] = sub_cluster
                    cluster_id_counter += 1

        return final_clusters

    try:
        final_clusters = split_clusters(clusters)
    except RecursionError:
        logger.error(
            "Recursion limit reached while splitting clusters. "
            "Consider increasing the max_number_of_actors parameter"
        )
        raise

    # Final safety check
    for cluster in final_clusters.values():
        if len(cluster) > max_size:
            raise ValueError(
                f"Cluster size exceeds max size: {len(cluster)} > {max_size}"
            )

    return final_clusters


def create_clusters(
    strategies,
    actor_clustering_method,
    clustering_method_kwargs,
):
    """
    Create clusters of strategies based on the specified clustering method and return them.

    Supports two clustering methods:
    - 'max-size': Uses Agglomerative Clustering to create groups with a maximum number of strategies.
    - 'k-means': Uses KMeans to create a predefined number of clusters.

    Args:
        strategies (list): List of strategy objects.
        actor_clustering_method (str): Clustering method to use ('max-size' or 'k-means').
        max_number_of_actors (int): Maximum number of strategies per cluster for 'max-size'.
        n_clusters (int): Number of clusters for 'k-means'.
        random_state (int): Seed for reproducibility.

    Returns:
        dict[int, list]: A dictionary where keys are cluster indices and values are lists of strategies.

    Raises:
        ValueError: If the required parameters are missing or if the clustering method is unknown.
    """

    n_clusters = clustering_method_kwargs.get("n_clusters", 2)
    max_number_of_actors = clustering_method_kwargs.get("max_number_of_actors", 10)
    random_state = clustering_method_kwargs.get("random_state", 42)

    n_strategies = len(strategies)

    if not strategies:
        raise ValueError("No strategies available for clustering.")

    # return a single cluster if only one strategy is available
    if n_strategies == 1:
        return {0: strategies}

    # Check if all strategies have the same context
    all_contexts = np.array([s.context.detach().cpu().numpy() for s in strategies])
    if len(np.unique(all_contexts, axis=0)) == 1:
        logger.warning(
            "All strategy contexts are identical — clustering is not possible."
        )
        return {0: strategies}  # Return a single cluster with all strategies

    if actor_clustering_method == "max-size":
        # Validate 'max-size' method
        if not isinstance(max_number_of_actors, int) or max_number_of_actors <= 0:
            raise ValueError(
                "'max_number_of_actors' must be a positive integer when using 'max-size' clustering method."
            )

        clusters = cluster_strategies_max_size(
            strategies=strategies,
            max_size=max_number_of_actors,
            random_state=random_state,
        )

    elif actor_clustering_method == "k-means":
        # Validate 'k-means' method
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError(
                "'n_clusters' must be a positive integer when using 'k-means' clustering method."
            )

        if n_clusters > n_strategies:
            raise ValueError(
                f"'n_clusters' ({n_clusters}) cannot be greater than the number of strategies ({n_strategies})."
            )

        clusters = cluster_strategies_k_means(
            strategies=strategies,
            n_groups=n_clusters,
            random_state=random_state,
        )

    else:
        raise ValueError(
            f"Unknown clustering method '{actor_clustering_method}'. Supported methods are 'max-size' and 'k-means'."
        )

    return clusters


def merge_and_cluster(
    strategies: list,
    old_mapping: dict[str, int],
    requested_n_clusters: int,
    threshold_factor: float = 1.2,
) -> dict[str, int]:
    """
    1) Graft new units onto old clusters if they're 'close enough'.
    2) Cluster the remainder into exactly `additional = requested_n_clusters - old_n` groups
       (or 1 if `additional < 1`), warning if the user didn't actually increase clusters.

    Returns a full unit_id → cluster_id map.
    """
    # 1) Old‐cluster centroids & radii
    old_clusters = sorted(set(old_mapping.values()))
    centroids, radii = {}, {}
    for cl in old_clusters:
        members = [s for s in strategies if old_mapping.get(s.unit_id) == cl]
        X = np.vstack([m.context.detach().cpu().numpy() for m in members])
        centroids[cl] = X.mean(axis=0)
        radii[cl] = np.max(np.linalg.norm(X - centroids[cl], axis=1))

    # 2) Assign close units to existing clusters
    final_map = dict(old_mapping)
    leftovers = []
    for s in strategies:
        uid = s.unit_id
        if uid in old_mapping:
            continue
        x = s.context.detach().cpu().numpy()
        # distance to each old centroid
        dists = {cl: np.linalg.norm(x - centroids[cl]) for cl in old_clusters}
        best = min(dists, key=dists.get)
        if dists[best] <= radii[best] * threshold_factor:
            final_map[uid] = best
        else:
            leftovers.append(s)

    # 3) Cluster leftovers if needed
    if leftovers:
        # Figure out how many new clusters
        old_n = len(old_clusters)
        additional = requested_n_clusters - old_n
        if additional < 1:
            if any(s.unit_id not in old_mapping for s in strategies):
                logger.warning(
                    f"Requested {requested_n_clusters} total clusters ≤ existing {old_n}; "
                    "will form 1 new cluster for the truly novel units."
                )
                additional = 1

        X_new = np.vstack([s.context.detach().cpu().numpy() for s in leftovers])
        km = KMeans(n_clusters=additional, random_state=0, n_init=10).fit(X_new)
        base = max(old_clusters) + 1 if old_clusters else 0
        for s, lbl in zip(leftovers, km.labels_):
            final_map[s.unit_id] = base + int(lbl)

    return final_map
