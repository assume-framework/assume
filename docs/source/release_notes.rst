.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

#######################
Release Notes
#######################

Upcoming Release
=======================

.. warning::
  The features in this section are not released yet, but will be part of the next release! To use the features already you have to install the main branch,
  e.g. ``pip install git+https://github.com/assume-framework/assume``

**New Features:**
- Converted DST components from functions to classes for improved modularity and reusability.
- Added new residential DST components like PV, EV, Heat Pump, and Boiler, with refined docstrings for better usability.
- Created a `GenericStorage` class for storage components, with classes like EV and Hydrogen Storage inheriting from it.


v0.4.1 - latest release (8th October 2024)
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

v0.4.0 - latest release (8th August 2024)
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
- Refactored unit operator by adding a seperate unit operator for learning units
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
- Examples have been restructed to easier orientation and understanding: example_01.. cover all feature demonstration examples,
  example_02.. cover all learning examples, example_03.. cover all full year examples
- Added the option of integrating different actor network architectures to the reinforcement learning algorithm, currently a multilayer perceptron (mlp) and long short-term memory (lstm) are implemented

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
