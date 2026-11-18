.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: MIT

####################
Real-Time Simulation
####################

Hardware in the loop simulations, using a real-time clock instead of the simulation clock are possible as well.
This is possible by using a :py:class:`mango.util.clock.AsyncioClock` instead of the standard :py:class:`mango.util.clock.ExternalClock` used for simulation time.

Switching the clock is done by adding the `real_time=True` parameter during the setup of the world:


.. code-block:: python

    world.setup(
        start=start,
        end=end,
        save_frequency_hours=1,
        simulation_id=simulation_id,
        # set real_time to true here
        real_time=True,
    )

The start of the simulation should be configured to be the next full minute.
Note that the time must be provided in UTC, as ASSUME treats all naive datetimes (without ``tzinfo``) as UTC, independent of the timezone of the machine.
The mango clock itself counts seconds since 1970-01-01 00:00 UTC, but Python's ``datetime.timestamp()`` and ``datetime.fromtimestamp()`` as well as mango's recurrence rules interpret naive datetimes in the local time of the machine.
Convert between datetimes and clock timestamps with :func:`assume.common.utils.datetime2timestamp` and :func:`assume.common.utils.timestamp2datetime` instead.

This simulation starts directly and ends after one hour.

Take a look at the `real_time/world_rt.py` example for a full working real time simulation which can be run using

    python examples/real_time/world_rt.py

Here, we have a simple market which is scheduled every single minute and creates similar market results.
