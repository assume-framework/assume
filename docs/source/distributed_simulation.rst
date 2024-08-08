.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

######################
Distributed Simulation
######################

Through the use of mango, it is possible to run ASSUME in a distributed way.

This is not supported on Windows for now.

There are different scenarios in which this might be wanted or needed.

- Improve performance of large simulation by utilizing multiple cores, avoiding the `GIL <https://wiki.python.org/moin/GlobalInterpreterLock>`
- Run simulation across multiple computer nodes to avoid exposing proprietary strategies
- Run simulation with real time clock
- Scalability simulation

The feature can be enabled for a single node using the `-p` command line parameter.
This sets the distributed_role = True and opens a port for connections on 9010
As documented in :doc:`command_line_interface`.

World Distributed Role
----------------------

When creating a :doc:`assume.world`, one can specify the optional parameter `distributed_role`.
This can have three values: `True`, `False` and `None` (default)

- None: no distributed role behavior is established. The whole simulation is run in a single process, utilizing one core.
- True: distributed behavior is used. Every Agent is created with its own mango container in a separate process. The mango containers communicate the current time between each other through the DistributedClockManager.
- False: specifies the distributed_role as an agent which does not manage its own Clock but uses a `DistributedClockAgent` which connects to a `manager_address` and receives clock updates from it.

Distributed Example
-------------------

Besides the command line interface param, there is the option to run distributed simulations from the example.

The example contains the following structure:

- main.py (contains script to run distributed simulation in single terminal using subprocesses)
- world_manager.py (contains manager script)
- world_agent.py (contains agent script)
- config.py (contains common config used by the other modules)

This makes it possible to run the simulation either using:

    python -m distributed_simulation.main

which starts the simulation in your terminal.

Or use two separate terminals, start the first one using:

    python -m distributed_simulation.world_agent

The script tries to connect to the manager now.
Next step is to start the manager.
Open another terminal, activate your python environment and run

    python -m distributed_simulation.world_manager

Now we are starting a simulation with two different terminals available.

Docker Example
--------------

Of course it is also possible to run this using docker to emulate having different machines.

This can be done using:

    docker compose --profile mqtt up

you should see a MQTT message broker being created as well as the simulation containers.
