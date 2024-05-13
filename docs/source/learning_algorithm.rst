.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

##################################
Reinforcement Learning Algorithms
##################################

In the chapter :doc:`learning` we got an general overview about how RL is implementes for a multi-agent setting in Assume. In the case one wants to apply these RL algorithms
to a new problem, one does not necessarly need to understand how the RL algorithms are are working in detail. The only thing needed is the adaptation of the bidding startegies,
which is covered in the tutorial. Yet, for the interested reader we will give a short overview about the RL algorithms used in Assume. We start with the learning role which is the core of the leanring implementation.


The Learning Role
=================

The learning role orchestrates the learning process. It initializes the training process and manages the experiences gained in a buffer.
Furthermore, it schedules the policy updates and, hence, brings the critic and the actor together during the learning process.
Particularly this means, that at the beginning of the simulation, we schedule recurrent policy updates, where the output of the critic is used as a loss
of the actor, which then updates its weights using backward propagation.

With the learning role, we can also choose which RL algorithm should be used. The algorithm and the buffer have base classes and can be customized if needed.
But without touching the code there are easy adjustments to the algorithms that can and eventually need to be done in the config file.
The following table shows the options that can be adjusted and gives a short explanation. For more advanced users is the functionality of the algorithm also documented below.



 ======================================== ==========================================================================================================
  learning config item                    description
 ======================================== ==========================================================================================================
  continue_learning                       Whether to use pre-learned strategies and then continue learning.
  trained_policies_save_path              Where to store the newly trained rl strategies - only needed when learning_mode is set
  trained_policies_load_path              If pre-learned strategies should be used, where are they stored? - only needed when continue_learning
  max_bid_price                           The maximum bid price which limits the action of the actor to this price.
  learning_mode                           Should we use learning mode at all? If not, the learning bidding strategy is overwritten with a default strategy.
  algorithm                               Specifies which algorithm to use. Currently, only MATD3 is implemented.
  learning_rate                           The learning rate, also known as step size, which specifies how much the new policy should be considered in the update.
  training_episodes                       The number of training episodes, whereby one episode is the entire simulation horizon specified in the general config.
  episodes_collecting_initial_experience  The number of episodes collecting initial experience, whereby this means that random actions are chosen instead of using the actor network
  train_freq                              Defines the frequency in time steps at which the actor and critic are updated.
  gradient_steps                          The number of gradient steps.
  batch_size                              The batch size of experience considered from the buffer for an update.
  gamma                                   The discount factor, with which future expected rewards are considered in the decision-making.
  device                                  The device to use.
  noise_sigma                             The standard deviation of the distribution used to draw the noise, which is added to the actions and forces exploration.  noise_scale
  noise_dt                                Determines how quickly the noise weakens over time.
  noise_scale                             The scale of the noise, which is multiplied by the noise drawn from the distribution.
 ======================================== ==========================================================================================================


The Algorithms
==============

TD3 (Twin Delayed DDPG)
-----------------------

TD3 is a direct successor of DDPG and improves it using three major tricks: clipped double Q-Learning, delayed policy update and target policy smoothing.
We recommend reading OpenAI Spinning guide or the original paper to undertsand the algorithm in detail.

Original paper: https://arxiv.org/pdf/1802.09477.pdf

OpenAI Spinning Guide for TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

Original Implementation: https://github.com/sfujim/TD3

In general the TD3 works in the following way. It maintains a pair of critics and a single actor. For each step so after every time interval in our simulation, we update both critics towards the minimum
target value of actions selected by the current target policy:


.. math::
    & y=r+\gamma \min _{i=1,2} Q_{\theta_i^{\prime}}\left(s^{\prime}, \pi_{\phi^{\prime}}\left(s^{\prime}\right)+\epsilon\right), \\
    & \epsilon \sim \operatorname{clip}(\mathcal{N}(0, \sigma),-c, c)



Every :math:`d` iterations, which is implemented with the train_freq, the policy is updated with respect to :math:`Q_{\theta_1}` following the deterministic policy gradient algorithm (Silver et al., 2014).
TD3 is summarized in the following picture from the others of the original paper (Fujimoto, Hoof and Meger, 2018).


.. image:: img/TD3_algorithm.jpeg
    :align: center
    :width: 500px


The steps in the algorithm are translated to implementations in ASSUME in the following way.
The initialization of the actors and critics is done by the :func:`assume.reinforcement_learning.algorithms.matd3.TD3.initialize_policy` function, which is called
in the learning role. The replay buffer needs to be stable across different episodes, which corresponds to runs of the entire simulation, hence it needs to be detached from the
entities of the simualtion that are killed after each episode, like the elarning role. Therefore, it is initialized independently and given to the learning role
at the beginning of each episode. For more information regarding the buffer see :doc:`buffers`.

The core of the algorithm is embodied by the :func:`assume.reinforcement_learning.algorithms.matd3.TD3.update_policy` in the learning algorithms. Here the critic and the actor are updated according to the algorithm.
