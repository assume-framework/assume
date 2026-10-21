# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from torch import nn

from assume.reinforcement_learning.neural_network_architecture import (
    MLPActor,
    LSTMActor,
)

actor_architecture_aliases: dict[str, type[nn.Module]] = {
    "mlp": MLPActor,
    "lstm": LSTMActor,
}
