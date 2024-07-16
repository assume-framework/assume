# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from torch import nn

from assume.reinforcement_learning.network_architecture import MLPActor, LSTMActor

policy_aliases: dict[str, type[nn.Module]] = {
    "mlp": MLPActor,
    "lstm": LSTMActor,
}
