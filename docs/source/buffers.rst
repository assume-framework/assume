.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

###############################
Buffers
###############################

This chapter gives you an insight into the general usage of buffers in reinforcement learning and how they are implemented in ASSUME.


Why do we need buffers?
=======================

In reinforcement learning, a buffer, often referred to as a replay buffer, is a crucial component in algorithms like for Experience Replay.
It serves as a memory for the agent's past experiences, storing tuples of observations, actions, rewards, and subsequent observations.

Instead of immediately using each new experience for training, the experiences are stored in the buffer. During the training process,
a batch of experiences is randomly sampled from the replay buffer. This random sampling breaks the temporal correlation in the data, contributing to a more stable learning process.

The replay buffer improves sample efficiency by allowing the agent to reuse and learn from past experiences multiple times.
This reduces the reliance on new experiences and makes better use of the available data. It also helps mitigate the effects of non-stationarity in the environment,
as the agent is exposed to a diverse set of experiences.

Overall, the replay buffer is instrumental in stabilizing the learning process in reinforcement learning algorithms,
enhancing their robustness and performance by providing a diverse and non-correlated set of training samples.


How are they used in Assume?
============================
In principal Assume allows for different buffers to be implemented. They just need to adhere to the structure presented in the base buffer. Here we will present the different buffers already implemented, which is only one, yet.


The simple replay buffer
------------------------

The replay buffer is currently implemented as a simple circular buffer, where the oldest experiences are discarded when the buffer is full. This ensures that the agent is always learning from the most recent experiences.
Yet, the buffer is quite large to store all observations also from multiple agents. It is initialised with zeros and then gradually filled. Basically after every step of the environment the data is collected in the learning role which sends it to the replay buffer by calling its add function.

After a certain round of training runs which is defined in the config file the RL strategy is updated by calling the update function of the respective algorithms which calls the sample function of the replay buffer.
The sample function returns a batch of experiences which is then used to update the RL strategy.
For more information on the learning capabilities of ASSUME, see :doc:`learning`.
