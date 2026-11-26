.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: MIT


Power Plant Units
=================

The :class:`~assume.units.powerplant.PowerPlant` unit models a generator that feeds power into the grid between ``min_power`` and ``max_power``.

General Definitions
-------------------

This page defines the parameterisation and explains how different limits change powerplant behavior.
Throughout the code and this page, the same sign convention is used everywhere:

- **Feeding electricity to the grid is positive.** It is bounded between ``min_power`` and
  ``max_power`` (both ``>= 0``).
- A power of ``0`` means the unit is idle.

On top of the minimum and maximum power output limits, the unit has **ramping constraints** that limit how much the power
output is allowed to change in one minute. The MW/min ramps are internally converted to simulation timesteps:

- ``ramp_up`` / ``ramp_down`` (both ``>= 0``): the maximum increase/decrease of the power in MW/min.

Because a unit can be active on several markets in the same time step, every ramping check distinguishes between three power values:

- **Previous power**: the power the unit was dispatched at in the *previous* time step (``previous_power``).
- **Current power**: the power already committed/sold on *other* markets for the *same* time step (``current_power``).
- **Requested power**: the bid being checked for a *new* market product (``requested_power`` / ``requested_power``). This always answers the question "what do I want to offer on *this* market right now?" -- it is a standalone power level for this one market, **not** a delta relative to ``current_power`` and **not** the combined total across all markets.

Ramping (and the max-/min power limits) are always checked on the *combination* of ``current_power`` and the requested power, because that combination is what the unit physically has to deliver. The function then returns the **feasible bid**: the largest/smallest power level for *this* market that keeps ``current_power + feasible_bid`` within what the unit can actually ramp to and what the min/max power limits are. That feasible bid is, again, a value for this one market -- ``current_power`` is only used internally to net the check, it is never added into the returned number.


Ramping Behaviour
------------------

Ramping is evaluated by
:meth:`~assume.common.base.SupportsMinMax.calculate_ramp` in
:class:`~assume.common.base.SupportsMinMax`. The method takes ``previous_power``, ``requested_power`` and ``current_power`` and returns the power level that is actually feasible given the ramping constraints.


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

Both thresholds default to 0, so without them every start is a cold start. The start-up costs are written to the ``starting_costs`` column of the unit dispatch, at the first time step in which the unit produces again. They are part of ``total_costs``. Storage units have no start-up costs right now, as ``get_operation_time`` is not implemented for them and falls back to 0.

.. _powerplant-initial-state:

Operating State at the Start of the Simulation
----------------------------------------------

For units based on :class:`~assume.common.base.SupportsMinMax` (power plants and other units that track their operating time), the first time step is consistently treated as if the unit had been switched off before the simulation started. It is assumed the current downtime already exceeds every configured threshold, for example ``min_down_time``.

As a result, if a unit is dispatched to produce power in the first time step, this is a genuine start-up: the ramp-up limit applies from zero, ``min_down_time`` cannot block the start (the assumed downtime already satisfies it), and start-up costs are booked like for any later cold start.
