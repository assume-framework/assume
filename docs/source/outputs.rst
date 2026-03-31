.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

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
     - simulation, unit, power, heat, soc, energy_generation_costs, energy_cashflow, total_costs

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
