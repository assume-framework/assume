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


Operating state at the start of the simulation

No dispatch history exists before the simulation's first time step, so each unit type needs an assumption for its initial condition.

Power plants
------------

For units based on :class:`~assume.common.base.SupportsMinMax` (power plants and other units that track their operating time), the first time step is consistently treated as if the unit had been switched off before the simulation started. It is assumed the current downtime already exceeds every configured threshold, for example min_down_time.

As a result, if a unit is dispatched to produce power in the first time step, this is a genuine start-up: the ramp-up limit applies from zero, ``min_down_time`` cannot block the start (the assumed downtime already satisfies it), and start-up costs are booked like for any later cold start.

Storage units
-------------

Storage units do not track an operating time. Their initial condition is the explicit ``initial_soc`` parameter (default ``0.5``, unless csv loader is used and column is defined and not filled, than the value ``0.0`` is assumed), which is used as the state of charge of the first time step. As with power plants, ``get_output_before`` returns ``0`` for the first time step, so the first charge or discharge ramp is calculated from zero.


Impact of initial conditions on your results
--------------------------------------------

The initial conditions of units can significantly influence the simulation results, especially in the early stages. For power plants, starting from a cold state may lead to higher initial costs and slower response times as they need to ramp up. For storage units, the initial state of charge can affect their ability to provide energy during peak demand periods. This might lead to high prices in early hours. We therefore advice to start you simulation time a couple days before the time frame of interest, to allow the units to even out before the results are evaluated.


Start-up costs and operating times of power plants
==================================================

Power plants can have start-up costs and minimum operating and down times. The time parameters are given in **hours**, independent of the simulation resolution, and are converted to time steps of the simulation index. Fractions of an hour are allowed, e.g. ``1.5``. As a unit can only switch at the start of a time step, the minimum up/down times are rounded up and the down time thresholds are rounded down to whole time steps. For example, in a simulation with 15-minute steps, ``min_down_time: 1.1`` becomes 5 steps (1.25 h), and ``downtime_hot_start: 1.1`` becomes 4 steps (1 h).

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

Both thresholds default to 0, so without them every start is a cold start. The start-up costs are written to the ``starting_costs`` column of the unit dispatch, at the first time step in which the unit produces again. They are part of ``total_costs``. Storage units have no start-up costs right now, as get_operation_time is not implemented for them and falls back to 0.


.. include:: demand_side_agent.rst
