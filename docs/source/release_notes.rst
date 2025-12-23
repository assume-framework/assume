.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

#############
Release Notes
#############

Upcoming Release
================
.. warning::
  The features in this section are not released yet, but will be part of the next release! To use the features already you have to install the main branch,
  e.g. ``pip install git+https://github.com/assume-framework/assume``

0.5.6 - (23th December 2025)
============================

**Bug Fixes:**

- **Changed action clamping**: The action clamping was changed to extreme values defined by dicts. Instead of using the min and max of a forward pass in the NN, the clamping is now based on the activation function of the actor network. Previously, the output range was incorrectly assumed based only on the input, which failed when weights were negative due to Xavier initialization.
- **Adjusted reward scaling**: Reward scaling now considers current available power instead of the unit’s max_power, reducing reward distortion when availability limits capacity. Available power is now derived from offered_order_volume instead of unit.calculate_min_max_power. Because dispatch is set before reward calculation, the previous method left available power at 0 whenever the unit was dispatched.
- **Update pytest dependency**: Tests now run with Pytest 9
- **Add new docs feature**: dependencies to build docs can now be installed with `pip install -e .[docs]`
- **Fix tests on Windows**: One test was always failing on Windows, which is fixed so that all tests succeed on all archs

**Improvements:**

- **Application of new naming convention for bidding strategies**: [unit]_[market]_[method]_[comment] for bidding strategy keys (in snake_case) and [Unit][Market][Method][Comment]Strategy for bidding strategy classes (in PascalCase for classes)
- **Changed SoC Definition**: The state of charge (SoC) for storage units is now defined to take values between 0 and 1, instead of absolute energy content (MWh). This change ensures consistency with other models and standard definition. The absolute energy content can still be calculated by multiplying SoC with the unit's capacity. The previous 'max_soc' is renamed to 'capacity'. 'max_soc' and 'min_soc' can still be used to model allowed SoC ranges, but are now defined between 0 and 1 as well.
- **Restructured learning_role tasks**: Major learning changes that make learning application more generalizable across the framework.
  - **Simplified learning data flow:** Removed the special ``learning_unit_operator`` that previously aggregated unit data and forwarded it to the learning role. Eliminates the single-sender dependency and avoids double bookkeeping across units and operators.
  - **Direct write access:** All learning-capable entities (units, unit operators, market agents) now write learning data directly to the learning role.
  - **Centralized logic:** Learning-related functionality is now almost always contained within the learning role, improving maintainability.
  - **Automatic calculation of obs_dim:** The observation dimension is now automatically calculated based on the definition of the foresight, num_timeseries_obs_dim and unique_obs_dim in the learning configuration. This avoids inconsistencies between the defined observation space and the actual observation dimension used in the actor network. However, if assumes the rational that 'self.obs_dim = num_timeseries_obs_dim * foresight + unique_obs_dim', if this is not the case the calculation of obs_dim needs to be adjusted in the learning strategy.
  - **Note:** Distributed learning across multiple machines is no longer supported, but this feature was not in active use.
- **Restructured learning configuration**: All learning-related configuration parameters are now contained within a single `learning_config` dictionary in the `config.yaml` file. This change simplifies configuration management and avoids ambiguous setting of defaults.

  .. note::
    ``learning_mode`` is moved from the top-level config to `learning_config`. Existing config files need to be updated accordingly.

- **Learning_role in all cases involving DRL**: The `learning_role` is now available in all simulations involving DRL, also if pre-trained strategies are loaded and no policy updates are performed. This change ensures consistent handling of learning configurations and simplifies the codebase by removing special cases.
- **Final DRL simulation with last policies**: After training, the final simulation now uses the last trained policies instead of the best policies. This change provides a more accurate representation of the learned behavior, as the last policies reflect the most recent training state. Additionally, multi-agent simulations do not always converge to the maximum reward. E.g. competing agents may underbid each other to gain market share, leading to lower overall rewards while reaching a stable state nevertheless.


**New Features:**

- **Unit Operator Portfolio Strategy**: A new bidding strategy type that enables portfolio optimization, where the default is called `UnitsOperatorEnergyNaiveDirectStrategy`. This strategy simply passes through bidding decisions of individual units within a portfolio, which was the default behavior beforehand as well. Further we added 'UnitsOperatorEnergyHeuristicCournotStrategy' which allows to model bidding behavior of a portfolio of units in a day-ahead market. The strategy calculates the optimal bid price and quantity for each unit in the portfolio, taking into account markup and the production costs of the units. This enables users to simulate and analyze the impact of strategic portfolio bidding on market outcomes and unit profitability.
- **Nodal Market Clearing Algorithm**: A new market clearing algorithm that performs electricity market clearing using an optimal power flow (OPF) approach, considering grid constraints and nodal pricing. This algorithm utilizes PyPSA to solve the OPF problem, allowing for a physics based representation of network constraints.

0.5.5 - (13th August 2025)
==========================

**New Features:**

- **Realtime simulation**: A simulation which runs in real-time instead of simulation time can be created and used for hardware-in-the-loop with real assets. Manual configuration of the agents is required for this.

**Improvements:**

- **Changed Logging for DRL metrics**: TensorBoard logging was restructured to separate metrics collected per gradient step and per simulation time step. This avoids unnecessary padding, ensures consistency, and prevents data loss across different logging frequencies.
- **Improve checking for available solvers**: Defines the list of available solvers only once
- **More notebooks in CI**: add notebook 10 to CI for functional validation of Demand-Side-Units (DSU).
- **Correctly suppress highs output for linopy**: this fixes an error when running the redispatch market clearing on Google collab
- **Add JOSS paper**
- **Pin pip-tools to fix the docker build**
- **Additional learning strategy for renewables**: Introduced a new learning strategy specifically designed for renewable energy sources. Most of the functionalities can just be inherited, we chose to add the availability of the unit into the individual observations and calculate the opportunity costs based on the available generation as well.
- **Actor Output Clamping:** The action outputs of the actor neural network + noise are now always clamped to the valid output range, which is dynamically determined based on the actor architecture and its activation function. This prevents exploration noise from pushing actions outside the achievable output space of the actor, ensuring that bids remain within the intended limits.

0.5.4 - (9th July 2025)
=======================

**New Features:**

- **ThermalStorage with Scheduling:** Introduced a `ThermalStorage` class that extends `GenericStorage` to support both short-term (freely cycling) and long-term (schedule-driven) storage operation. Long-term mode allows users to define a binary schedule to restrict charging and discharging to specific periods, enabling realistic modeling of industrial or seasonal thermal storage behavior. To use this feature, set `storage_type` to `"long-term"` and provide a `storage_schedule_profile` (0: charging allowed, 1: discharging allowed). Hydrogen fuel type included for the Boiler
- **Hydrogen_plant:** The HydrogenPlant master class has been refactored for modularity. Technologies such as the electrolyser and (optionally) the SeasonalHydrogenStorage are now connected in a flexible manner, supporting both per-timestep and cumulative hydrogen demand balancing. The plant model now robustly accommodates both storage and non-storage configurations, ensuring correct mass balances across all scenarios.
- **Steam Generation Plant:** Introduced a 'SteamGenerationPlant' class to model steam generation processes. This class supports both electric and thermal inputs, allowing for flexible operation based on available resources. The plant can be configured with various components, such as heat pumps and boilers, to optimize steam production.
- **New Demand Side Flexibility Measure** Implemented 'symmetric_flexible_block' flexibility measure for demand side units. This measure allows users to define a symmetric block of flexibility, enabling to construct a load profile based on which the block bids for CRM amrket can be formulated.
- **Positive and Negative Flexibility for DSM Units** Introduced the bidding strategies 'CapacityHeuristicBalancingPosStrategy' and 'CapacityHeuristicBalancingPosStrategy' to define positive and negative flexibility for demand side management (DSM) units. This feature allows users to participate DSM units in a Control Reserve Market (CRM).
- **Electricity price signal based Flexibility Signal for DSM**: Implemented'electricity_price_signal' flexibility measure for demand side units, Thus measure allows to shift the load based on the electricity price signal, enabling users to perform this operation based on a reference load profile.
- **Documentation**: Fullscale DSM Tutorial and adjusted learning tutorials to include new bidding strategy and one particularly for storages.
- **New Redispatch Tutorial**: Provide a new tutorial referencing ongoing dveelopment on an extra branch.

**Improvements:**

- **Initial State of Charge (SOC) Enforcement:** The initial SOC is now explicitly enforced as a model constraint for all storage units. This guarantees that simulations always start from the intended state, ensuring scientific reproducibility and correctness.
- **Enhanced Test Suite for Storage Units:** Comprehensive unit tests have been added for both `GenericStorage` and `ThermalStorage`, including short-term and long-term scheduling, efficiency losses, ramping, power limits, schedule adherence, and initial SOC.
  - Tests verify economic cycling (charging at low price, discharging at high price), round-trip efficiency, and no simultaneous charge/discharge.
- **SeasonalHydrogenStorage:** The framework of SeasonalHydrogenStorage is now consistent with the framework of Thermal storage.
- **Refactored Learning Strategies:** Much of the code for generating observations and actions was redundant across different unit types. This redundancy has been removed by introducing the function in the common base class, making it easier to extend the learning strategies in the future. As a result, new functions such as `get_individual_observations`, which are specific to each unit type, have been added.
- **Change energy_cost Observation in Storage Learning:**  The cost of stored energy for the learning storage is now tracked solely based on acquisition cost while charging, independent of discharging revenues. This change prevents negative cost values, ensures a consistent economic interpretation of stored energy, and improves the guiding properties of the observations of reinforcement learning according to shap value experiments.
    Marginal costs are now included as well. Storage marginal costs currently only consist of additional charge or discharge costs, e.g. to include fixed volumetric grid fees. Revising and comparing the mc logic to the Powerplant implementation resulted in removing the efficiency correction factor of the additional costs for consistency.
- **Component connection in hydrogen plant:** Fixed a bug regarding the connection of the components in the hydrogen plant.


**Bug Fixes:**

- **Correct Schedule Enforcement in ThermalStorage:** Fixed an error in the schedule constraint logic for long-term storage. The model now strictly enforces that charging and discharging can only occur during their respective scheduled periods.
- **Initial SOC Bug:** Fixed a bug where the initial SOC could be ignored or set to an unintended value by the solver. The initial SOC is now always fixed at model initialization.
- **Boiler Upper Bound:** Added a missing upper bound constraint to `natural_gas_in` in the `Boiler` component to ensure that the fuel input never exceeds the specified maximum power. This change prevents the solver from assigning unphysical input values and brings consistency with the handling of other fuel types.
- **Missing init:** Steam generation plant and CRM bidding strategies for dsm was missing in 'init'.

0.5.3 - (4th June 2025)
=======================

**New Features:**

- **Add single bid RL strategy:** Added a new reinforcement learning strategy that allows agents to submit bids based on one action value only that determines the price at which the full capacity is offered.
- **Bidding Strategy for Elastic Demand**: The new `EnergyHeuristicElasticStrategy` enables demand units to submit multiple bids that approximate a marginal utility curve, using
  either linear or isoelastic price elasticity models. Unlike other strategies, it does **not** rely on predefined volumes—bids are dynamically generated based on the
  unit’s elasticity configuration. To use this strategy, set `bidding_strategy` to `"demand_energy_heuristic_elastic"` in the `demand_units.csv` file and specify the following
  parameters: `elasticity` (must be negative), `elasticity_model` (`"linear"` or `"isoelastic"`), `num_bids`, and `price` (which acts as `max_price`). The `elasticity_model`
  defines the shape of the demand curve, with `"linear"` producing a straight-line decrease and `"isoelastic"` generating a hyperbolic curve. `num_bids` determines how many
  bid steps are submitted, allowing control over the granularity of demand flexibility.


**Improvements:**

- **Flexible Agent Count in `continue_learning` Mode:** You can now change the number of learning agents between training runs while reusing previously trained critics.
  This enables flexible workflows like training power plants first and adding storage units later. When the core architectures match, critic weights are partially transferred when possible, ensuring smoother transitions.

**Bug Fixes:**

- **Last policy loading**: Fixed a bug where the last policy loaded after a training run was not the best policy, but rather the last policy.
- **Negative accepted volume in block bids**: Fixed a bug where accepted volume from block bids was converted to negative.
- **Grafana Dashboard adjustments**: Fixed a bug where the Grafana dashboard was wrongly summing values due to time bucketing. The dashboard now consistently displays the average per time bucket which does underestimate
  variance in the data, but a note was added to explain this.
- **Changed market price in rejected orders**: Fixed a bug where the wrong market price was written in the rejected orders, namely any auction with more than 1 product had the price of the last product written as the
  market price instead of the price of the respective hour. This was, however, only a mistake for the rejected orders.

0.5.2 - (21st March 2025)
=========================

**New Features:**

- **TensorBoard Integration:** To enable better monitoring of the learning progress and comparison between different runs, we have added the possibility to use TensorBoard for logging
  the learning progress. To use this feature, please follow the instructions in the README.
- **Building Class:** Introduced a new ``Building`` class to represent residential and tertiary buildings. This enhancement allows users to define a building type along with
  associated technology components, facilitating a more detailed investigation of energy consumption and flexibility potential. The building can also be defined as a prosumer or consumer.
  When a building is defined as prosumer, it actively participates in electricity trading, allowing the operator/resident to sell excess energy to the grid. In contrast,
  a consumer represents a traditional energy consumer focusing solely on energy consumption without trading capabilities.

**Improvements:**

- **Changed SoC Definition**: The state of charge (SoC) for storage units is now defined as the SoC at the beginning of the respective timestep, reflecting the entire available capacity before having submitted any bids.
  This change ensures that the SoC is consistently interpretable. Discharging and charging action in the respective hour are then reflected by the next SoC.
- **Multi-market participation configuration**: Respect the `eligible_obligations_lambda` set in the `MarketConfig` to only bid on markets where the UnitsOperator fulfills the requirements.
  Changes the behavior to not participate on markets when no unit has a matching bidding strategy for this market.
- **Learning Performance:** The learning performance for large multi-agent learning setups has been significantly improved by introducing several learning stabilization techniques.
  This leads to a more stable learning process and faster convergence. It also allows for running simulations with a larger number of agents that achieve comparable results to historical data.
  For example, running example_03a for the year 2019, one can achieve an RMSE of 10.22 EUR/MWh and MAE of 6.52 EUR/MWh for hourly market prices, and an RMSE of 6.8 EUR/MWh and MAE of 4.6 EUR/MWh when
  using daily average prices. This is a significant improvement compared to the previous version of the framework.

**Bug Fixes:**

- **Storage Learning Strategy:** Fixed a bug in the storage learning strategy that caused the learning process to fail or perform poorly. The bug was related to the way the storage was updating the state of charge.
  This has been fixed, and the learning process for storage units is now stable and performs well. It also improved the performance of non-learning bidding strategies for storage units. Further reduced the actions number to one which reflects discharge and charge actions.
- **Wrong train_freq Handling:** Fixed a bug where, if the simulation length was not a multiple of the train_freq, the remaining simulation steps were not used for training, causing the training to fail.
  This has been fixed, and now the train_freq is adjusted dynamically to fit the simulation length. The user is also informed about the adjusted train_freq in the logs.
- **Logging of Learning Parameters:** Fixed the way learning parameters were logged, which previously used a different simulation_id for each episode, leading to very slow performance of the learning Grafana dashboard.
  Now, the learning parameters are logged using the same simulation_id for each episode, which significantly improves the performance of the learning Grafana dashboard.
- **Learning Reward Writing:** Fixed a bug where the reward was wrongly transformed with a reshape instead of a transpose when writing the reward to the database. This caused the reward to be written in the wrong format when working with multiple units.
  The bug did affect learning process with heterogeneous agents mainly. This has been fixed, and now the reward is written in the correct format.

**Code Refactoring**

  - Moved common functions to DSMFlex.
  - Added tests for the ``Building`` class.
  - Refactored variable names for better readability and consistency.
  - Restructured the process sequence for improved efficiency.

v0.5.1 - (3rd February 2025)
===========================================
**New Features:**

- **Exchange Unit**: A new unit type for modeling **energy trading** between market participants. It supports **buying (importing) and selling (exporting) energy**, with user-defined prices.
  Check **example_01a**, **example_03**, and the files **"exchange_units.csv"** and **"exchanges_df.csv"** for usage examples.
- **Market Contracts and Support Policies**: it is now possible to simulate the auctioning of support policies, like feed-in tariff, PPA, CfD or a market premium.
  The contracts are auctioned and then have a regular contract execution, to compensate according to the contracts dynamic, based on the historic market price and unit dispatch (#542).
- **Merit Order Plot** on the default Grafana Dashboard - showing a deeper view into the bidding behavior of the market actors.
  Additionally, a graph showing the market result per generation technology has been added (#531).

**Improvements:**

- **Multi-agent DRL fix**: Addressed a critical bug affecting action sampling, ensuring correct multi-agent learning.
- **Performance boost**: Optimized training efficiency, achieving **2x overall speedup** and up to **5x on CUDA devices**.
- **Learning Observation Space Scaling:** Instead of the formerly used max scaling of the observation space, we added a min-max scaling to the observation space.
  This allows for a more robust scaling of the observation space for future analysis (#508).
- **Allow Multi-Market Bidding Strategies**: Added the possibility to define a bidding strategy for multiple markets. Now when the same bidding strategy is used for two or more markets,
  the strategy is only created once and the same instance is used for all of these markets.
- **Improve Storage Behavior**: Storages were using the current unmodified SoC instead of the final SoC of last hour, leading to always using the initial value to calculate discharge possibility.(#524)
- **OEDS Loader**: when using the OEDS as a database, the queries have been adjusted to the latest update of the MarktStammDatenRegister. Time-sensitive fuel costs for gas, coal and oil are available from the OEDS as well.
  This also includes various fixes to the behavior of the DMAS market and complex powerplant strategies (#532).

**Bug Fixes:**

- **Update PyPSA Version:** Fixes example "small_with_redispatch"; adjustments to tutorials 10 and 11 to remove DeprecationWarnings.
- **Fixes to the documentation** documentation and example notebooks were updated to be compatible with the latest changes to the framework (#530, #537, #543)
- **postgresql17** - using the docker container in the default compose.yml requires to backup or delete the existing assume-db folder. Afterwards, no permission changes should be required anymore when setting up the DB (#541)

v0.5.0 - (10th December 2024)
===========================================

**New Features:**

- **Learning Rate and Noise Scheduling**: Added the possibility to schedule the learning rate and action noise in the learning process. This feature
  enables streamlining the learning progress. Currently, only "linear" decay available by setting the `learning_rate_schedule` and
  `action_noise_schedule` in the learning config to "linear". Defaults to no decay if not provided. It decays `learning_rate`/ `noise_dt`
  linearly from starting value to 0 over given `training_episodes` which can be adjusted by the user. The schedule parameters (e.g. end value
  and end fraction) are not adjustable in the config file, but can be set in the code.
- **Hydrogen Plant:** A new demand side unit representing a hydrogen plant has been added. The hydrogen plant consists of an
  electrolyzer and a seasonal hydrogen storage unit. The electrolyzer converts electricity into hydrogen, which can be
  stored in the hydrogen storage unit and later used.
- **Seasonal Hydrogen Storage:** A new storage unit representing a seasonal hydrogen storage has been added. The seasonal hydrogen
  storage unit can store hydrogen over long periods and release it when needed. It has specific constraints to avoid charging or
  discharging during off-season or on-season time as well as a target level to be reached at the end of the season.

**Improvements:**

- **Timeseries Performance Optimization:** Switched to a custom `FastIndex` and `FastSeries` class, which is based on the pandas Series
  but utilizes NumPy arrays for internal data storage and indexing. This change significantly improves the
  performance of read and write operations, achieving an average speedup of **2x to 3x** compared to standard
  pandas Series. The `FastSeries` class retains a close resemblance to the pandas Series, including core
  functionalities like indexing, slicing, and arithmetic operations. This ensures seamless integration,
  allowing users to work with the new class without requiring significant code adaptation.
- **Outputs Role Performance Optimization:** Output role handles dict data directly and only converts to DataFrame on Database write.
- **Overall Performance Optimization:** The overall performance of the framework has been improved by a factor of 5x to 12x
  depending on the size of the simulation (number of units, markets, and time steps).

**Bugfixes:**

- **Tutorials**: General fixes of the tutorials, to align with updated functionalitites of Assume
- **Tutorial 07**: Aligned Amiris loader with changes in format in Amiris compare (https://gitlab.com/fame-framework/fame-io/-/issues/203 and https://gitlab.com/fame-framework/fame-io/-/issues/208)
- **Powerplant**: Remove duplicate `Powerplant.set_dispatch_plan()` which broke multi-market bidding
- **CSV scenario loader**: Fixed issue when one extra day was being added to the index, which lead to an error in the simulation when additional data was not available in the input data.
- **Market opening schedule**: Fixed issue where the market opening was scheduled even though the simulation was ending before the required products. Now the market opening is only scheduled
  if the total duration of the market products plus first delivery time fits before the simulation end.
- **Loader fixes**: Fixes for PyPSA, OEDS and AMIRIS loaders

**Full Changelog**: `v0.4.3...v0.5.0 <https://github.com/assume-framework/assume/compare/v0.4.2...v0.5.0>`_

v0.4.3 - (11th November 2024)
===========================================

**Improvements:**

- **Documentation**: added codespell hook to pre-commit which checks for spelling errors in documentation and code

**Bugfixes:**

- **Simulation**: Delete simulation results for same simulation prior to run (as before v0.4.2)

**Full Changelog**: `v0.4.2...v0.4.3 <https://github.com/assume-framework/assume/compare/v0.4.2...v0.4.3>`_

v0.4.2 - (5th November 2024)
===========================================

**New Features:**

- **Residential Components**: Added new residential DST components including PV, EV, Heat Pump, and Boiler, now with enhanced docstrings for better usability.
- **Modular DST Components**: DST components have been converted from functions to classes, improving modularity and reusability.
- **Generic Storage Class**: Introduced a `GenericStorage` class for storage components. Specific classes, such as EV and Hydrogen Storage, now inherit from it.
- **Storage Learning Strategy**: Added a new DRL-based learning strategy for storage units. To use it, set `storage_energy_learning` in the `bidding_EOM` column of `storage_units.csv`. Refer to the `StorageEnergyLearningStrategy` documentation for more details.
- **Mango 2.x Update**: Upgraded to mango 2.x, enabling synchronous world creation. To upgrade an existing environment, run:
  ```
  pip uninstall -y mango-agents mango-agents-assume && pip install assume-framework --upgrade
  ```
- **Distributed Simulation Enhancements**: Improved distributed simulation for TCP and MQTT, allowing containers to wait for each other during simulations.
- **Integrated Optimization with Pyomo and HIGHS Solver**: The Pyomo library and HIGHS solver are now installed by default, removing the need to install `assume-framework[optimization]` separately. The HIGHS solver is used as the default, replacing the older GLPK solver for improved optimization performance and efficiency.

**Improvements:**

- **Documentation**: Refined tutorial notebooks and added bug fixes.
- **Saving Frequency Logic**: Refactored the saving frequency in the `WriteOutput` class for improved efficiency.

**Bug Fixes:**

- **Solver Compatibility**: Addressed undefined `solver_options` when using solvers other than Gurobi or HIGHS.
- **Cashflow Calculation**: Corrected cashflow calculations for single-digit orders.
- **Simulation Execution**: Enabled simulations to synchronize and wait for each other.
- **Edge Case Handling**: Fixed edge cases in `pay_as_clear` and `pay_as_bid`.

**New Contributor:**

- @HafnerMichael made their first contribution with improvements to cashflow calculations and development of residential DST components.

**Full Changelog**: `v0.4.1...v0.4.2 <https://github.com/assume-framework/assume/compare/v0.4.1...v0.4.2>`_


v0.4.1 (8th October 2024)
===========================================

**New Features:**

- improve LSTM learning strategy (#382)
- add python 3.12 compatibility (#334)
- manual strategy for interactive market simulation (#403)

**Improvements:**

- add the ability to define the solver for the optimization-based market clearing inside the param_dict of the config file (#432)
- shallow clone in Jupyter notebooks so that cloning is faster (#433)
- fixes in storage operation bidding (#417)
- update GitHub Actions versions (#402)

**Bug Fixes:**

- add compatibility with pyyaml-include (#421)
- make complex clearing compatible to RL (#430)
- pin PyPSA to remove DeprecationWarnings for now (#431)

**Full Changelog**: `v0.4.0...v0.4.1 <https://github.com/assume-framework/assume/compare/v0.4.0...v0.4.1>`_

v0.4.0 (8th August 2024)
=========================================

**New Features:**

- **Market Coupling:** Users can now perform market clearing for different market zones with given transmission capacities. This feature
  allows for more realistic simulation of market conditions across multiple interconnected regions, enhancing the accuracy of market
  analysis and decision-making processes. A tutorial on how to use this feature is coming soon.

- **Adjust the Framework to Schedule Storing to the Learning Role:** This enhancement enables Learning agents to participate in sequential
  markets, such as day-ahead and intraday markets. The rewards are now written after the last market, ensuring that the learning process
  accurately reflects the outcomes of all market interactions. This improvement supports more sophisticated and realistic agent training scenarios.
  A tutorial on how to use this feature is coming soon.

- **Multiprocessing:** Using a command line option, it is now possible to use run each simulation agent in its own process to speed up larger simulations.
  You can read more about it in :doc:`distributed_simulation`

- **Steel Plant Demand Side Management Unit**: A new unit type has been added to the framework, enabling users to model the demand side management
  of a steel plant. This feature allows for more detailed and accurate simulations of industrial energy consumption patterns and market interactions.
  This unit can be configured with different components, such as the electric arc furnace, electrolyzer, and hot storage, to reflect the specific
  characteristics of steel production processes. The process can be optimized to minimize costs or to maximize the available flexibility, depending
  on the user's requirements. A tutorial and detailed documentation on how to use this feature are coming soon.

- **LSTM Actor Architectures:** The framework now supports long short-term memory (LSTM) networks as actor architectures for reinforcement learning.
  This feature enables users to apply more advanced neural network architectures to their learning agents, enhancing the learning process and
  enabling more accurate and efficient decision-making especially with time series data.

**Improvements:**

- Significant speed up of the framework and especially of the learning process
- Separated scenario loader function to improve speed and reduce unrequired operations
- Refactored unit operator by adding a separate unit operator for learning units
- Enhanced learning output and path handling
- Updated dashboard for better storage view
- Improved clearing with shuffling of bids, to avoid bias in clearing of units early in order book
- Introduced a mechanism to clear the market according to defined market zones while maintaining information about
  individual nodes, enabling the establishment of specific market zones within the energy market and subsequent
  nodal-based markets such as redispatch.
- Added `zones_identifier` to the configuration file and `zone_id` to the `buses.csv`, and refactored the complex market
  clearing algorithm to incorporate zone information, ensuring that bids submitted with a specific node are
  matched to the corresponding market zone.
- If any values in the availability_df.csv file are larger than 1, the framework will now warn the user
  and run a method to normalize the values to [0, 1].
- Examples have been restructured to easier orientation and understanding: example_01.. cover all feature demonstration examples,
  example_02.. cover all learning examples, example_03.. cover all full year examples
- Added the option of integrating different actor network architectures to the reinforcement learning algorithm, currently a multilayer perceptron (mlp) and long short-term memory (lstm) are implemented
- Added storing of network flows for complex clearing

**Bug Fixes:**

- Fix learning when action dimension equals one
- Fixed Tutorial 5
- Correctly calculated timezone offsets
- Improved handling of rejected bids
- Fix the error that exploration mode is used during evaluation
- Fix double dispatch writing
- Fixed complex clearing with pyomo>=6.7
- Resolved various issues with learning and policy saving
- Fixed missing market dispatch values in day-ahead markets
- Added a check for availability_df.csv file to check for any values larger than 1
- Fixed compatibility issues between new pyomo and RL due to tensor handling

**Other Changes:**

- Added closing word and final dashboard link to interoperability tutorial


**Full Changelog**: `v0.3.7...v0.4.0 <https://github.com/assume-framework/assume/compare/v0.3.7...v0.4.0>`_

v0.3.7 (21st March 2024)
=========================

**New Features:**

- Added Contract Market with feed-in policy and market premium (#248)
- Introduced basic grid visualization (#305)
- Added PyPSA loader (#311)
- Implemented interoperability tutorial (#323)

**Improvements:**

- Updated how Pyomo markets are imported (#310)
- Added ARM docker platform support (#312)
- Updated Grafana docker version to latest (#316)
- Adjusted scenario loaders (#317)
- Prepared ASSUME for proper nodal pricing integration (#304)

**Bug Fixes:**

- Fixed bugs in tutorial 6 (#324)
- Set correct compose.yml mount for docker (#320)

**Other Changes:**

- Added Code of Conduct (#313)
- Added fixed Pyomo version to avoid warnings (#325)
- Increased version to 0.3.7 for latest release (#327)


v0.3.6 (22nd February 2024)
===========================

**Improvements:**

- Updated GitHub actions (#296, #297)
- Silenced output of Gurobi by specifying a non-logging environment (#300)
- Fixed writing of market_dispatch and dispatch for other product types (#301)
- Fixed datetime warning (#302)

**Bug Fixes:**

- Fixed Tutorial 2 (#299)
- Fixed string conversion of paths (#307)

**Documentation:**

- Added a tutorial for advanced order types and documentation for complex clearing (#303)

**Other Changes:**

- Moved DMAS bidding strategies into try-except block since Pyomo is not a required dependency (#308)


v0.3.5 (14th February 2024)
===========================

**New Features:**

- Introduced the redispatch module for congestion management
- Implemented cost-based and market-based redispatch strategies
- Added support for "pay as bid" and "pay as clear" market methods in redispatch

**Improvements:**

- Changed strategy allocation to use market names instead of product types (#289)
- Implemented overall scenario loading improvements

**Bug Fixes:**

- Fixed issues with storage operations (#291)
- Removed empty bid as a method of bidding strategy (#293)
- Cleaned up hard-coded EOM references (#294)


v0.3 (6th February 2024)
=========================

**New Features:**

- Added Data Request mechanism (#247)
- Implemented block order and linked order with respective market clearing mechanism (#269)
- Added MASTR based OEDS loader
- Introduced AMIRIS Scenario loader

**Improvements:**

- Added "Open in Colab" to notebooks (#258)
- Improved data_dict usage (#274)

**Bug Fixes:**

- Fixed calculation of marginal cost and output_before (#250)
- Adjusted query of reward during training (#256)
- Fixed calculation of flexible storage bids (#260)
- Fixed RL evaluations (#280)

**Documentation:**

- Added basic tutorials 01 and 02 (#257)
- Created Custom Unit and Custom Strategy tutorial (#262)
- Added tutorial for EOM and LTM comparison (#265)
- Updated dependencies and installation instructions (#282)
- Added additional clearing and strategy docs (#283)

**Other Changes:**

- Added reuse compliance
- Moved scenario loaders to separate folder (#264)
- Added automatic assignment of RL units to one RL unit operator (#276)


v0.2.1 (3rd November 2023)
===========================

**Improvements:**

- Improved distribution of current time to agents running in shadow container in different processes (#199)

**Bug Fixes:**

- Fixed loading of learned strategies (#219)

**Documentation:**

- Added RL Documentation (#221)

**Other Changes:**

- Added AMIRIS scenario loader (#224)
- Added shields badges to README (#223)
- Fixed issues for running distributed scenario with MQTT (#222)


v0.2.0 (30th September 2023)
=============================

**New Features:**

- Added support for CUDA-enabled devices for learning
- Implemented tracking of evaluation periods for better learning performance evaluation
- Added capability to start several simulations in parallel

**Improvements:**

- Enhanced learning performance
- Addressed storage units behavior bugs

**Other Changes:**

- Added new Grafana dashboard definitions for easier analysis
- Updated Docker compose file to include Renderer for saving plots directly from Grafana dashboards


v0.1.0 - Initial Release (12th September 2023)
==============================================

This is the initial release of the ASSUME Framework, published to PyPi.

**Key Features:**

- Ability to define different energy market designs
- Includes reinforcement learning capabilities

The ASSUME Framework allows users to model and simulate various energy market designs while incorporating reinforcement learning techniques for advanced analysis and optimization.
