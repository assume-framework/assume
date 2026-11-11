.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: MIT


Power Plant Units
=================

The :class:`~assume.units.powerplant.PowerPlant` unit models a generator that feeds power into the grid between ``min_power`` and ``max_power``.

Start-up Costs and Operating Times
----------------------------------

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
