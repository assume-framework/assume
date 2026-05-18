.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later


Demand-Side Agents & Management
===============================

This section describes the demand-side units and management strategies available in the toolbox. The module enables the modeling of demand-side units such as industrial facilities (e.g., steel plants) and residential buildings, with detailed optimization of electricity consumption and load-shifting capabilities.

Overview
--------

Demand-Side Units (DSUs) represent entities that manage energy consumption and contribute to the electricity market. These entities are optimized based on specific objectives, such as minimizing costs or maximizing operational flexibility. The two main types of demand-side agents modeled in this module are **Industrial Agents** (e.g., Steel Plant) and **Residential Agents** (e.g., Buildings).

Modeling Demand-Side Units (DSUs)
---------------------------------

Demand-Side Units (DSUs) are modeled using a component-based approach, where each DSU is a combination of multiple technologies or processes. Each DSU can either be an industrial facility or a residential building. The DSU model is designed to automatically create the process flow by connecting the appropriate components based on user inputs.

For example:

- **Industrial DSU**: The Steel Plant agent connects components like electrolysers, hydrogen storage, DRI plants, and electric arc furnaces to form a complete production process. This is automatically done by providing input components, and the system handles the rest.
- **Residential DSU**: The Building agent connects components like heat pumps, thermal storage, and EV chargers to optimize the energy consumption of a residential building.

DSM Flexibility and Load Shifting
---------------------------------

The :py:class:`assume.units.dsm_load_shift` module integrates load-shifting capabilities into agents. This allows for:
- Defining cost tolerance for flexibility.
- Shifting loads between time steps based on operational flexibility.
- Recalculating operations based on market offers and accepted bids.

Rolling-Horizon Optimisation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DSM units support both a full-horizon solve and a rolling-horizon solve.
Without additional configuration, units use the full simulation horizon.
When rolling horizon is enabled, the unit repeatedly optimises a shorter
look-ahead window, commits only the first part of that solution, and then
re-optimises after the next market step with updated internal states.

This is useful when the optimisation should stay responsive to market
clearing results while still accounting for storage states, ramping limits,
and process-coupling constraints over a wider future horizon.

Configuration
~~~~~~~~~~~~~

Rolling-horizon behaviour is configured through
``dsm_optimisation_config``. In scenario-based runs this configuration is
passed from ``config.yaml`` to the DSM units by the CSV scenario loader.

The supported keys are:

- ``horizon_mode``: Either ``full_horizon`` or ``rolling_horizon``.
- ``look_ahead_horizon``: Length of the optimisation window, for example ``72h``.
- ``commit_horizon``: Number of steps committed before the next re-optimisation, for example ``24h``.
- ``rolling_step``: Shift applied between consecutive windows, for example ``24h``.

.. note::

    If ``horizon_mode`` is set to ``rolling_horizon``, the values for
    ``look_ahead_horizon``, ``commit_horizon``, and ``rolling_step`` must all
    be provided.

Example YAML configuration:

.. code-block:: yaml

    industrial_dsm_optimisation:
       horizon_mode: rolling_horizon
       look_ahead_horizon: 4h
       commit_horizon: 2h
       rolling_step: 2h

    residential_dsm_optimisation:
       horizon_mode: rolling_horizon
       look_ahead_horizon: 6h
       commit_horizon: 2h
       rolling_step: 2h

Operational flow
~~~~~~~~~~~~~~~~

In rolling-horizon mode, the optimisation proceeds as follows:

1. A look-ahead window is built from the current simulation step.
2. The unit solves that window with the same process and flexibility constraints used in the full-horizon model.
3. Only the first ``commit_horizon`` part of the solution is committed to the unit's schedule.
4. End-of-window component states, such as storage state of charge or operational status, are carried into the next window.
5. After the market advances, the unit re-optimises the next window starting from the updated state.

This means DSM units can preserve inter-temporal feasibility while reacting
to accepted bids and updated market conditions.

Steel plant strategies in rolling horizon
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For :class:`~assume.units.steel_plant.SteelPlant`, rolling-horizon
optimisation can combine the rolling window logic with different production
guidance modes from the forecaster.

- **Cost-optimized**: If no additional production guidance is provided, the steel plant minimizes cost freely across each rolling window.
- **Profile-guided**: If ``normalized_load_profile`` is provided, the rolling-horizon solver keeps committed production close to the profile shape using soft profile-tracking constraints.
- **Min-demand**: If per-timestep ``steel_demand`` is provided, the rolling-horizon solver enforces hourly minimum production levels in the committed part of each window.

The steel plant forecaster accepts either generic forecast columns or
unit-specific columns:

- ``normalized_load_profile`` or ``<unit_id>_normalized_load_profile``
- ``steel_demand`` or ``<unit_id>_steel_demand``

These signals are copied to the unit during forecaster initialization and are
used automatically by the rolling-horizon solver.

Attributes
^^^^^^^^^^^

- **Cost Tolerance**: Defines how much additional cost can be tolerated for load-shifting.
- **Load Shift**: Adjusts the total power input based on flexibility constraints and available storage or generation resources.

Example
^^^^^^^^

For a steel plant, the load-shifting mechanism can balance power input between the electrolyser, DRI plant, and EAF, adjusting production to minimize costs while meeting production targets.


DSU Components
--------------

Each component (e.g., Electrolyser, Heat Pump, DRI Plant) is modeled in detail with specific constraints and parameters, such as:
- Power input/output limits
- Ramp rates
- Efficiency
- Operating costs
- Load flexibility

For full references see :mod:`assume.units.dst_components`.

Bidding and Optimization
------------------------

Once the DSU is defined, it can participate in the electricity market by optimizing its operation to either minimize costs or maximize flexibility. The bidding strategy depends on the agent type and market conditions.

Industrial Agent: Steel Plant
-----------------------------

The **Industrial Agent** represents facilities such as steel plants, where key components like electrolysers, DRI plants, hydrogen storage, and electric arc furnaces are modeled. These components form a process flow that optimizes electricity consumption based on the agent’s objectives.

Attributes
^^^^^^^^^^^

- **Technology**: Represents components such as Electrolyser, DRI Plant, Hydrogen Storage, and Electric Arc Furnace.
- **Node**: Connection point in the energy system.
- **Bidding Strategy**: Defines how the steel plant bids in electricity markets. Example: `DsmEnergyOptimizationStrategy`.
- **Objective**: Optimization target, such as minimizing operational costs.
- **Flexibility Measure**: Load-shifting to optimize flexibility.

Components
^^^^^^^^^^^

- **Electrolyser**: Produces hydrogen via electrolysis with constraints on power limits, ramp rates, operating time, and costs.
- **DRI Plant**: Models direct reduced iron production with fuel and power consumption constraints.
- **Electric Arc Furnace (EAF)**: Produces steel from DRI with power consumption, ramp rates, and CO2 emissions constraints.

Example Workflow
^^^^^^^^^^^^^^^^^^

1. Instantiate the SteelPlant agent with required components (Electrolyser, DRI Plant, etc.).
2. Define objectives, such as minimizing electricity costs.
3. Run the optimization to generate bids for the electricity market.

.. note::

   DSM units (e.g. :class:`~assume.units.steel_plant.SteelPlant`,
   :class:`~assume.units.building.Building`,
   :class:`~assume.units.hydrogen_plant.HydrogenPlant`,
   :class:`~assume.units.steam_generation_plant.SteamGenerationPlant`) build their internal
   optimisation model inside the forecaster's ``initialize`` step.  When running the
   simulation through the normal world / scenario loader this happens automatically.  However,
   if you create a unit programmatically and do **not** call
   :meth:`~assume.common.forecaster.UnitForecaster.initialize` (or
   :func:`~assume.world.World.init_forecasts`), you must call
   ``unit.setup_model()`` explicitly before running optimisation on the unit.

Residential Agent: Building
----------------------------

The **Residential Agent** models buildings containing technologies like heat pumps, electric vehicles (EVs), and thermal storage. These components allow the building to participate in the electricity market as both a consumer and prosumer.

Attributes
^^^^^^^^^^^

- **Technology**: Components such as heat pumps, boilers, thermal storage, and EV charging.
- **Node**: Connection to the electrical grid.
- **Bidding Strategy**: Example: `NaiveDABuildingStrategy`.
- **Objective**: Optimization target, such as minimizing energy expenses.
- **Flexibility Measure**: Load-shifting to adjust energy usage based on market conditions.

Components
^^^^^^^^^^^

- **Heat Pump**: Provides heating/cooling and interacts with thermal storage.
- **EV Charging**: Manages EV battery charging based on availability periods and electricity prices.
- **Thermal Storage**: Buffers thermal energy for flexible heating/cooling operations.

Example Workflow
^^^^^^^^^^^^^^^^^

1. Create a Building agent with components such as a heat pump and thermal storage.
2. Define objectives like minimizing electricity usage.
3. Execute optimization to determine the best dispatch plan based on market prices.

Strategies for Bidding
----------------------

Several naive bidding strategies are implemented for managing how agents participate in electricity markets. These strategies define how energy bids are created for different agent types.

- **DsmEnergyOptimizationStrategy**: Optimizes steel plant operations and generates bids based on marginal costs.
- **NaiveDABuildingStrategy**: Manages the bidding process for residential agents, calculating bids based on optimal energy use and load-shifting.
- **Redispatch Strategies**: Adjust operations based on redispatch market signals, focusing on reducing operational costs or maximizing flexibility.
