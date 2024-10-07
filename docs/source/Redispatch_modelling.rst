.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

===============================
Congestion management and Redispatch Modelling
===============================

This section demonstrates modelling and simulation of redispatch mechanism using PyPSA as a plug and play module in ASSUME-framework. 
The model is created mainly taking grid constraints into consideration to identify grid bottlenecks with dispatches from EOM and resolve them using the redispatch algorithm.

Concept of Redispatch
========
The locational mismatch in demand and generation of electricity needs transmission of electricity from low demand regions to high demand regions. The transmission capacity limits the maximum amounts of electricity which can be transmitted at any point in time. 
If there is no enough capacity to transmit the required amount of electricity then there is a need of ramping down of generation at the locations of low demand and ramping up of generation at the locations of higher demand. 
This is typically called as Redispatch. Apart from spot markets there is redispatch mechanism to regulate this grid flows to avoid congestion issues. It is operated and controlled by the System operators (SO).

--------------------------------------------
Overview of Redispatch Modelling in PyPSA
--------------------------------------------
PyPSA network model can be created to visualize line flows with the EOM clearning outcomes of generation and loads at different nodes(locations).

PyPSA has following terminology to define grid infrastructure:
Attributes:
-----------
1. **Bus**: Nodes(locations) of powerplants and Loads
    - **name**: Unique name of Bus
    - **v_nom**: Nominal voltage of the node
    - **carrier**: Energy carrier: can be “AC” or “DC” for electricity buses
    - **x**: Position longitude
    - **y**: Position latitude

2. **Generator**: Powerplants which generates electricity
    - **name**: Unique name of Generator
    - **p_nom**: Nominal power for limits in optimization
    - **p_max_pu**: The maximum output for each snapshot per unit of p_nom
    - **p_min_pu**: The minimum output for each snapshot per unit of p_nom
    - **p_set**: Active power set point (for Power flow analyis)
    - **marginal_cost**: Marginal cost of production of 1 MWh
    - **carrier**: Energy carrier: can be “AC” or “DC” for electricity buses
    - **x**: Position longitude
    - **y**: Position latitude
    - **sign**: Power sign(when positive represents generation of power/when negative consumption of power)

3. **Load**: Demand units which consumes electricity
    - **name**: Unique name of Load
    - **p_set**: Active power set point (for inflexible demand)
    - **marginal_cost**: Marginal cost of production of 1 MWh
    - **x**: Position longitude
    - **y**: Position latitude
    - **sign**: Power sign(when positive represents generation of power/when negative consumption of power)

4. **Line**: Transmission grids which transmit electrity
    - **name**: Unique name of Line
    - **s_nom**: Nominal transmission capacity
    - **capital_cost**: Capitial cost of extending lines by 1 MVA
    - **r**: Resistance in Ohm
    - **bus0**: First node to which the line is connected
    - **bus1**: Second node to which the line is connected
    - **x**:Series reactance
    - **s_nom_extendable**: Switch to enable s_nom expansion

5. **Network**: PyPSA network to which all above components are integrated to
    - **name**: Unique name of Network
    - **snapshots**: List of timesteps.(e.g. Hourly, Quarter-hourly, Daily etc.)

A network model using PyPSA can be created by defining nodes as a locations of power generation and consumption locations which can then be interconnected with the transmission lines with nominal trasmission capacity (s_nom). 
These components can defined with further more operational constraints (such as nominal_power, efficiency, ramp rates and many more)

The limitation of current PyPSA model is that, it is not possible to define flexible Load. 

Modelling of redispatch in ASSUME-framework using PyPSA framework mainly includes two parts:

--------------------------------------------
1. **Congestion identification**
--------------------------------------------

In the first step first checks for congestion in the network. The linear flow method is particularly useful for quick assessments of congestion and redispatch needs. PyPSA provides the network.lpf() function for running linear power flow. It is much faster than full non-linear AC power flow, making it suitable for real-time analysis or large network studies.
The assumptions to set up lpf includes no restrictions on power flows in terms of MVA (MW) and there are no line losses. PyPSA network needs to be set up with all the necessary components: buses, lines, loads, generators, and any other relevant elements. A small trick here is to add generators as Load with positive sign to let the generations not going up to satifsy both demand and network constraints. 
Loads are normally added as Load since they are anyways inflexible. So this model doesn't allow any ramping up/down of generators nor ramping up/down of loads and simulated flow of power for given timesteps.

network.lines_t.p0 gives Active power flows through Lines which then can be compared to Nominal capacity of Lines (s_nom) to determine whether there is congestion. 

 ``line_loading = network.lines_t.p0 / network.lines.s_nom``

For the line loading with higher than value 1 suggests that there is congestion. 
---

--------------------------------------------
2. **Redispatch of powerplants**
--------------------------------------------
Once it is identified that there is congestion in any of the lines at any timestep, the redispatch mechanism is established to relieve this congestion.

The idea here is to  and add upward redispatch and downward redispatch on top of that.

        ``p_max_pu_down = (volume - min_power)/(max_power)``

- The redispatch is modelled in following steps in ASSUME-framwork::
    1. **Fixing dispatches from EOM market**: The dispatches from EOM market are fixed in order to model redispatches from the powerplants with accurate cost coniderations.
        The EOM dispatches are fixed by adding them as a ``Load`` in the network with dispatches as ``p_set``.Generators are added with positive sign and demands are added with negative sign.
    
    2. **Upward redispatch from market and reserved powerplants**:
        Modelling redispatch using PyPSA is not straightforward due to the limitations of PyPSA to model flexibility of ramping up and ramping down of loads. 
        The upward redispatches are then added as ``Generator`` with positive sign. The maximum available capacity to redispatch upward is restricted with the factor ``p_max_pu``. This fraction is estimated as a difference between the current generation and maximum power of the powerplant.
        
        ``p_max_pu_up = (max_power - volume)/(max_power)``

    3. **Downward redispatch from market powerplants**: 
        Similarly, the upward redispatches are then modelled as ``Generator`` with negative sign. The maximum available capacity to redispatch downward is restricted with the factor ``p_max_pu``. This fraction is estimated as a difference between the current generation and maximum power of the powerplant.
    
    4. **Upward and Downward redispatch from other flexibilites**: 
        Simlarly flexibilities for redispatch are also added as generators with positive sign for upward redispatch and with negative sign for downward redispatch respectively.

The transmission line capacity is also restricted to exapand by setting ``s_nom_extendable=False``.

Objective:
The aim of redispatch is to reduce the overall cost of Redispatch(starting up, shuting down, ramping up, ramping down etc.).



