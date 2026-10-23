.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: MIT

###########
Unit Types
###########

In power system modeling, various unit types are used to represent the components responsible for generating, storing, and consuming electricity. These units are essential for simulating the operation of the grid and ensuring a reliable balance between supply and demand.

The primary unit types in this context include:

1. **Power Plants**: These are conventional or renewable energy generators, such as coal, gas, wind, or solar power plants. They are responsible for supplying electricity to the grid based on the system's demand.

2. **Storage Units**: These units, like batteries or pumped hydro storage, can store electricity when supply exceeds demand and release it when needed, adding flexibility to the grid.

3. **Demand Units**: These represent consumers of electricity, such as households, industries, or commercial buildings, whose electricity consumption is typically fixed and not easily adjustable based on real-time grid conditions. Demand units will therefore be modelled with inelastic demand most often. However, representation of elastic bidding is possible with this unit type.

Each unit type has specific characteristics that affect how the power system operates, and understanding these is key to modeling and optimizing grid performance.


Start-up costs and operating times of power plants
==================================================

Power plants which track their operation can have start-up costs and minimum operating and down times. The time parameters are given in **hours**, independent of the simulation resolution, and are converted to time steps of the simulation index. The minimum times are rounded up to whole time steps.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Unit
     - Meaning
   * - ``hot_start_cost``, ``warm_start_cost``, ``cold_start_cost``
     - €/MW of ``max_power``
     - Cost of one start-up, scaled with the installed capacity of the unit.
   * - ``downtime_hot_start``
     - h
     - A start after a downtime of up to this duration is a hot start.
   * - ``downtime_warm_start``
     - h
     - A start after a downtime of more than ``downtime_hot_start`` and up to this duration is a warm start. A longer downtime is a cold start.
   * - ``min_operating_time``, ``min_down_time``
     - h
     - Minimum time the unit has to run after a start, or stay off after a shutdown.

The start-up costs are written to the ``starting_costs`` column of the unit dispatch, once per start at the first time step in which the unit produces again. They are part of ``total_costs``. Storage units have no start-up costs.


.. include:: demand_side_agent.rst
