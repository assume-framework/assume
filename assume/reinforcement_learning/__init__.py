# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from assume.reinforcement_learning.buffer import ReplayBuffer
from assume.reinforcement_learning.learning_role import Learning


"""
class PolicyRegistry:
    policy_aliases: ClassVar[Dict[str, Type[nn.Module]]] = {
            "mlp": MLPActor,
            "lstm": LSTMActor,
        }

    def get_policy_from_name(self, policy_name: str) -> Type[nn.Module]:
        
        Get a policy class from its name representation.

        The goal here is to standardize policy naming, e.g.
        all algorithms can call upon "mlp" or "lstm",
        and they receive respective policies that work for them.

        (cf. https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/base_class.html)

        :param policy_name: Alias of the policy
        :return: A policy class (type)
        

        if policy_name in self.policy_aliases:
            return self.policy_aliases[policy_name]
        else:
            raise ValueError(f"Policy {policy_name} unknown")
"""