.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

###########
Unit Types
###########

In power system modeling, various unit types are used to represent the components responsible for generating, storing, and consuming electricity. These units are essential for simulating the operation of the grid and ensuring a reliable balance between supply and demand.

The primary unit types in this context include:

1. **Power Plants**: These are conventional or renewable energy generators, such as coal, gas, wind, or solar power plants. They are responsible for supplying electricity to the grid based on the system's demand.

2. **Storage Units**: These units, like batteries or pumped hydro storage, can store electricity when supply exceeds demand and release it when needed, adding flexibility to the grid.

3. **Demand Units**: These represent consumers of electricity, such as households, industries, or commercial buildings, whose electricity consumption is typically fixed and not easily adjustable based on real-time grid conditions. Demand units will therefore be modelled with inelastic demand most often. However, representation of elastic bidding is possible with this unit type.

Each unit type has specific characteristics that affect how the power system operates, and understanding these is key to modeling and optimizing grid performance.


.. include:: demand_side_agent.rst
