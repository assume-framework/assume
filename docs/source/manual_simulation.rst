.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

#################
Manual Simulation
#################

You can also be part of the simulation by submitting the bids yourself.
This can be done to test a specific behavior, challenge yourself or create a game using a distributed simulation, where everyone can participate.

Single Manual Terminal Client
-----------------------------

You can test this by opening the `example_01a/powerplant_units.csv` and change the bidding_strategy of a single powerplant from `powerplant_energy_naive` to `manual_strategy`.

Now run `assume -c tiny` in your terminal to have a small local simulation with assume, which prompts you for the bids of this single agent.

Distributed Game
----------------

As shown in :doc:`distributed_simulation` - one can also run a distributed simulation where each participant manages a single unit operator and is asked for the bids of a unit.

To now use the manual strategy in a distributed simulation, you can edit the `world_agent.py` and replace `powerplant_energy_naive` with `manual_strategy`.
Then you can first start a terminal for the agent:

    python -m distributed_simulation.world_agent

And run the manager in a separate window:

    python -m distributed_simulation.world_manager

This asks for the bids in the world_agent terminal while the manager waits for all agents to submit the bids.
This can be extended to include better multi user support, which would make this usable in a game environment.

Future Work
-----------

A common idea in university is to have an interactive market simulation where users can simulate market participation.
By extending this small setup, one could create a webserver or GUI for each client, which would also allow to embed Grafana results for decision making.
