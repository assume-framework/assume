.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

Example Simulations
=====================

While the modeller can define her own input data for the simulation, we provide some example simulations to get started.
Here you can find an overview of the different exampels provided. Below you find an exhaustive table explaining the different examples.


 ============================= ============================= =====================================================
  example name                 input files                   description
 ============================= ============================= =====================================================
  small                         example_01a                     Small Simulation (4 actors) with single hour bidding.
  small_dam                     example_01a                     Small simulation with 24 hour bidding.
  small_with_opt_clearing       example_01a                     Small simulation with optimization clearing instead of pay_as_clear.
  small_with_BB                 example_01d                     Small Simulation with Block Bids and complex clearing.
  small_with_vre                example_01b                     Small simulation with variable renewable energy.
  small_learning_1              example_02a                     A small study with roughly 10 powerplants, where one powerplant is equiped with a learning bidding strategy and can learn to exert market power.
  small_learning_2              example_02b                     A small study with roughly 10 powerplants, where multiple powerplants are equiped with a learning bidding strategy and learn that they do not have market power anymore.
 ============================= ============================= =====================================================

The following table categorizes the different provided examples in a more detailed manner. We included the main features of ASSUME in the table.


============================== =============== =============== =================== ====================== ============= ============= ================= ============== =============
example name                   Country         Generation Tech Generation Volume   Demand Tech            Demand Volume Markets       Bidding Strategy  Grid           Further Infos
============================== =============== =============== =================== ====================== ============= ============= ================= ============== =============
small_learning_1               Germany         conventional    12,500 MW           fixed inflexible       1,000,000 MW  EoM           Learning, Naive   No             Resembles Case 1 from Harder et.al. 2023
small_learning_2               Germany         conventional    12,500 MW           fixed inflexible       1,000,000 MW  EoM           Learning, Naive   No             Resembles Case 2 from Harder et.al. 2023
============================== =============== =============== =================== ====================== ============= ============= ================= ============== =============


References
-----------
Harder, Nick & Qussous, Ramiz & Weidlich, Anke. (2023). Fit for purpose: Modeling wholesale electricity markets realistically with multi-agent deep reinforcement learning. Energy and AI. 14. 100295. 10.1016/j.egyai.2023.100295.
