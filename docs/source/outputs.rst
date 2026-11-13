.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: MIT

###############
Storing Outputs
###############

This documentation describes the functionality of the output processing system used to store simulation outputs either in a database or as CSV files. It covers the configuration, data buffering, conversion, and storage mechanisms of the main class as well as the support class for database maintenance.


WriteOutput Class
==================

The ``WriteOutput`` class is responsible for:

- Collecting simulation output messages.
- Buffering data until either a time-based interval or a memory limit is reached.
- Converting raw data into structured pandas DataFrames.
- Storing data into a database and/or exporting it as CSV files.
- Managing performance trade-offs by allowing configuration of buffer sizes and storage intervals.

This class is part of the :mod:`assume.common.outputs` module. For more details, refer to :py:class:`assume.common.outputs.WriteOutput`.

Data Processing Workflow
--------------------------
1. **Message Handling:**

   - The method ``handle_output_message`` receives asynchronous messages containing simulation outputs.
   - Data is buffered in lists corresponding to the intended database tables based on content type.

2. **Buffer Size Management:**

   - The memory usage of buffered data is monitored continuously.
   - If memory usage exceeds the ``outputs_buffer_size_mb`` threshold, buffered data is flushed to storage.

3. **Data Conversion:**

   - Helper methods like ``convert_rl_params`` and ``convert_market_results`` transform raw data into DataFrames suitable for storage.

4. **Data Storage:**

   - The method ``store_dfs`` flushes buffered DataFrames to storage based on time intervals (``save_frequency_hours``) or buffer size limits.

Performance Considerations
---------------------------
- **Buffering Strategy:** Increasing ``outputs_buffer_size_mb`` can improve performance by reducing write frequency. Default: 300 MB.
- **Write Intervals:** Setting ``save_frequency_hours`` facilitates real-time observation with Grafana dashboards. Disabling this by setting it to ``null`` improves performance but prevents real-time monitoring.
- **Optimal Configuration:** Increase ``outputs_buffer_size_mb`` and disable ``save_frequency_hours`` for maximum performance. Ensure sufficient memory is available.

.. note::
  When storing results as CSV files, ``save_frequency`` is automatically disabled, meaning data is only saved at the end of the simulation or if the buffer size limit is reached. This prevents real-time observation through Grafana dashboards.


Database Tables Overview
=========================
The simulation outputs are stored in various tables. Below is a breakdown of tables, their purpose, and what columns they contain.
When the outputs are stored in CSV files, the data is organized in a similar structure, with each table represented as a separate CSV file.

.. list-table::
   :header-rows: 1

   * - **Table Name**
     - **Purpose**
     - **Columns**

   * - market_meta
     - Market-related data.
     - simulation, market_id, time, node, product_start, product_end, only_hours, price, max_price, min_price, supply_volume, supply_volume_energy, demand_volume, demand_volume_energy

   * - market_dispatch
     - Dispatch data for markets.
     - simulation, market_id, datetime, unit_id, power

   * - unit_dispatch
     - Unit-level dispatch data.
     - simulation, unit, power, heat, soc, energy_generation_costs, starting_costs, energy_cashflow, total_costs

   * - dsm_schedule
     - Planned operating schedule per technology of the demand-side units, in long format. See :ref:`dsm-output-tables`.
     - simulation, time, unit, technology, variable, carrier, direction, unit_of_measure, baseline, flex

   * - dsm_specs
     - Technical specification of the demand-side units per technology. See :ref:`dsm-output-tables`.
     - simulation, unit, technology, parameter, value

   * - market_orders
     - Market order details.
     - simulation, market_id, node, bid_id, unit_id, parent_bid_id, bid_type, end_time, price, volume, accepted_price, accepted_volume, min_acceptance_ratio

   * - power_plant_meta
     - Power plant metadata.
     - simulation, unit_id, unit_type ("power_plant"), max_power, min_power, emission_factor, efficiency

   * - storage_meta
     - Storage unit metadata.
     - simulation, unit_id, unit_type ("storage"), capacity, max_soc, min_soc, max_power_charge, max_power_discharge, min_power_charge, min_power_discharge, efficiency_charge, efficiency_discharge

   * - rl_params
     - Reinforcement learning parameters.
     - simulation, unit, datetime, evaluation_mode, episode, profit, reward, regret, actions, exploration_noise, critic_loss, total_grad_norm, max_grad_norm, learning_rate

   * - rl_meta
     - Reinforcement learning metadata.
     - simulation, episode, eval_episode, learning_mode, evaluation_mode

.. note::
  The columns for all unit_meta like `power_plant_meta` and `storage_meta` are not listed here. Their structure is dictated by the ``as_dict`` method in the respective unit classes.

.. _dsm-output-tables:

DSM Output Tables
=================

The tables ``dsm_schedule`` and ``dsm_specs`` are written whenever demand-side units
(steel plant, cement plant, hydrogen plant, steam plant, building) are part of the
simulation.

Planned schedule and realised dispatch
--------------------------------------

``dsm_schedule`` holds the schedule the unit's own optimisation determined, **not** the
realised dispatch after market clearing. If a bid is only partly accepted, the realised
plant-level operation is in ``unit_dispatch``, and the bid and accepted volumes per
market are in ``market_orders``. Join them on the unit id and the time to compare planned
and accepted operation.

``baseline`` is the optimal operation without flexibility, ``flex`` the flexible
operation. ``flex`` stays empty as long as the unit did not determine it, for example in
rolling-horizon runs, where the committed windows are written to ``baseline``.

Reported variables
------------------

Every non-binary variable of a technology block is reported, e.g. ``power_in``, material
and fuel flows, ``co2_emission``, ``operating_cost`` or storage levels. The series of the
unit itself (``total_power_input``, ``variable_cost``, the load shift variables and the
``electricity_price`` the solve used) are reported under the pseudo technology ``unit``.

Each variable is classified by ``carrier`` (electricity, heat, hydrogen, natural_gas,
steel, co2, ...), ``direction`` (``in``, ``out``, ``state``, ``cost`` or ``price``) and
``unit_of_measure``, so that flows can be aggregated per energy carrier, e.g. for Sankey
diagrams.

.. warning::
  The cement plant's ``fossil_in`` is the sum of ``natural_gas_in`` and ``coal_in``.
  Do not add it to those two again when aggregating fuel inputs.

Technical specification
-----------------------

``dsm_specs`` holds one row per unit, technology and scalar parameter of that technology
(rated powers, capacities, efficiencies, specific consumptions, ...), plus the name of
the component class as ``component_class``. It is sent once at the start of the
simulation, together with the other unit meta data.

DatabaseMaintenance Class
==========================

The ``DatabaseMaintenance`` class provides utility functions for managing simulation data stored in a database. It includes:

- Retrieving unique simulation IDs from the database.
- Deleting specific simulations.
- Performing bulk deletion of simulation data with optional exclusions.

This class is part of the :mod:`assume.common.outputs` module. For more details, refer to :py:class:`assume.common.outputs.DatabaseMaintenance`.

Usage Example
==============
Below is an example YAML configuration snippet for a simulation configuring the save frequency and buffer size:

.. code-block:: yaml

  example_simulation:
    simulation_id: example_simulation
    start: 2025-01-01 00:00:00
    end: 2025-01-02 00:00:00
    save_frequency_hours: 24 # Time interval for saving data in hours
    outputs_buffer_size_mb: 300  # Buffer size in MB

Example usage of the ``DatabaseMaintenance`` class:

.. code-block:: python

  from assume.common import DatabaseMaintenance

  # Select to store the simulation results in a local database or in TimescaleDB.
  # When using TimescaleDB, ensure Docker is installed and the Grafana dashboard is accessible.
  data_format = "timescale"  # Options: "local_db" or "timescale"

  if data_format == "local_db":
      db_uri = "sqlite:///./examples/local_db/assume_db.db"
  elif data_format == "timescale":
      db_uri = "postgresql://assume:assume@localhost:5432/assume"

  maintenance = DatabaseMaintenance(db_uri=db_uri)

  # 1. Retrieve unique simulation IDs:
  unique_ids = maintenance.get_unique_simulation_ids()
  print("Unique simulation IDs:", unique_ids)

  # 2. Delete specific simulations:
  maintenance.delete_simulations(["example_01", "example_02"])

  # 3. Delete all simulations except a few:
  maintenance.delete_all_simulations(exclude=["example_01"])
