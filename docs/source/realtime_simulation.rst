.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

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
Note that the time must be provided in UTC as every timestamp in mango is interpreted as UTC/zulu time to be timezone agnostic.

This simulation starts directly and ends after one hour.

Take a look at the `world_rt.py` example for a full working real time simulation which can be run using

    python examples/world_rt.py

Here, we have a simple market which is scheduled every single minute and creates similar market results.
