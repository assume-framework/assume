.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

Example Simulations
===================

ASSUME provides a range of example simulations to help users understand and explore various market scenarios. These examples demonstrate different features and configurations, from small-scale setups to large, real-world simulations. Below is an overview of the available examples, followed by a more detailed explanation of their key features.

Overview of Example Simulations
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Example Name
     - Input Files
     - Description
   * - small
     - example_01a
     - Basic simulation with 4 actors and single-hour bidding.
   * - small_dam
     - example_01a
     - Day-ahead market simulation with 24-hour bidding.
   * - small_with_opt_clearing
     - example_01a
     - Demonstrates optimization-based market clearing.
   * - small_with_vre
     - example_01b
     - Introduces variable renewable energy sources.
   * - small_with_vre_and_storage
     - example_01c
     - Showcases renewable energy and storage units.
   * - small_with_BB_and_LB
     - example_01c
     - Illustrates block bids and linked bids usage.
   * - small_with_vre_and_storage_and_complex_clearing
     - example_01c
     - Combines VRE, storage, and complex clearing mechanisms.
   * - small_with_crm
     - example_01c
     - Includes Control Reserve Market (CRM).
   * - small_with_redispatch
     - example_01d
     - Demonstrates redispatch scenarios.
   * - small_with_nodal_clearing
     - example_01d
     - Features nodal market clearing.
   * - small_with_zonal_clearing
     - example_01d
     - Implements zonal market clearing.
   * - small_learning_1
     - example_02a
     - 7 power plants, 1 with learning bidding strategy. Case 1 in [1]
   * - small_learning_2
     - example_02b
     - 11 power plants, 5 with learning bidding strategy. Case 2 in [1]
   * - small_learning_3
     - example_02c
     - 16 power plants, 10 with learning bidding strategy. Case 3 in [1]
   * - learning_with_complex_bids
     - example_02d
     - Learning strategies with complex bidding.
   * - large_2019_eom
     - example_03
     - Full-year German power market simulation (EOM only). [2]
   * - large_2019_eom_crm
     - example_03
     - Full-year German power market simulation (EOM + CRM). [2]
   * - large_2019_day_ahead
     - example_03
     - Full-year German day-ahead market simulation. [2]
   * - large_2019_rl
     - example_03a
     - Modified 2019 simulation for reinforcement learning. [1][2]

Detailed Features of Example Simulations
----------------------------------------

The following table provides a more in-depth look at key examples, highlighting their specific characteristics and configurations.

.. list-table::
   :header-rows: 1
   :widths: 15 10 15 15 15 10 10 15 10 15

   * - Example Name
     - Country
     - Generation Tech
     - Generation Volume
     - Demand Tech
     - Demand Volume
     - Markets
     - Bidding Strategy
     - Grid
     - Further Info
   * - small_learning_1
     - Germany
     - Conventional
     - 12,500 MW
     - Fixed inflexible
     - 1,000,000 MW
     - EOM
     - Learning, Naive
     - No
     - Case 1 from [1]
   * - small_learning_2
     - Germany
     - Conventional
     - 12,500 MW
     - Fixed inflexible
     - 1,000,000 MW
     - EOM
     - Learning, Naive
     - No
     - Case 2 from [1]
   * - small_learning_3
     - Germany
     - Conventional
     - 12,500 MW
     - Fixed inflexible
     - 1,000,000 MW
     - EOM
     - Learning, Naive
     - No
     - Case 3 from [1]
   * - large_2019_eom
     - Germany
     - Conv., VRE
     - Full 2019 data
     - Fixed inflexible
     - Full 2019 data
     - EOM
     - Various
     - No
     - Based on [2]
   * - large_2019_eom_crm
     - Germany
     - Conv., VRE
     - Full 2019 data
     - Fixed inflexible
     - Full 2019 data
     - EOM, CRM
     - Various
     - No
     - Based on [2]
   * - large_2019_day_ahead
     - Germany
     - Conv., VRE
     - Full 2019 data
     - Fixed inflexible
     - Full 2019 data
     - DAM
     - Various
     - No
     - Based on [2]
   * - large_2019_rl
     - Germany
     - Conv., VRE
     - Modified 2019 data
     - Fixed inflexible
     - Modified 2019 data
     - EOM
     - RL, Various
     - No
     - Based on [1][2]

Note: Conv. = Conventional, VRE = Variable Renewable Energy, EOM = Energy-Only Market, CRM = Control Reserve Market, DAM = Day-Ahead Market, RL = Reinforcement Learning

Key Features of Example Simulations
-----------------------------------

1. Small-scale examples (small_*):

   - Designed for easier understanding of specific features and configurations.
   - Demonstrate various market mechanisms, bidding strategies, and technologies.
   - Useful for learning ASSUME's basic functionalities and exploring specific market aspects.

2. Learning-enabled examples (small_learning_*, learning_with_complex_bids):

   - Showcase the integration of learning algorithms in bidding strategies.
   - Illustrate how agents can adapt their behavior in different market conditions.
   - small_learning_1, small_learning_2, and small_learning_3 directly correspond to Cases 1, 2, and 3, respectively, in the publication by Harder et al. [1].
   - Demonstrate practical applications of reinforcement learning in energy markets.

3. Large-scale examples (large_2019_*):

   - Represent real-world scenarios based on the German power market in 2019.
   - Include full demand and renewable generation profiles, major generation units, and storage facilities.
   - Demonstrate different market configurations (EOM, CRM, DAM) and their impacts.
   - Based on comprehensive research presented in [2], offering insights into complex market dynamics.

These examples provide a diverse range of scenarios, allowing users to explore various aspects of energy market simulation, from basic concepts to complex, real-world applications.

References
----------
[1] Harder, Nick & Qussous, Ramiz & Weidlich, Anke. (2023). Fit for purpose: Modeling wholesale electricity markets realistically with multi-agent deep reinforcement learning. Energy and AI. 14. 100295. 10.1016/j.egyai.2023.100295.

[2] Qussous, R.; Harder, N.; Weidlich, A. Understanding Power Market Dynamics by Reflecting Market Interrelations and Flexibility-Oriented Bidding Strategies. *Energies* **2022**, *15*, 494. https://doi.org/10.3390/en15020494
