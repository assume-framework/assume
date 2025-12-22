.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later


Congestion Management and Redispatch Modeling
===============================================

This section demonstrates the modeling and simulation of the redispatch mechanism using PyPSA as a plug-and-play module within the ASSUME framework.
The model primarily considers grid constraints to identify bottlenecks in the grid, resolve them using the redispatch algorithm, and account for dispatches from the EOM (Energy-Only Market).

Concept of Redispatch
----------------------

The locational mismatch between electricity demand and generation requires the transmission of electricity from low-demand regions to high-demand regions. The transmission capacity limits the maximum amount of electricity that can be transmitted at any point in time.

When transmission capacity is insufficient to meet demand, generation must be reduced at locations with low demand and increased at locations with high demand. This process is known as **Redispatch**. In addition to spot markets, the redispatch mechanism is used to regulate grid flows and avoid congestion issues. It is operated and controlled by the system operators (SO).


Overview of Redispatch Modeling in PyPSA
------------------------------------------

The PyPSA network model can be created to visualize line flows using EOM clearing outcomes of generation and loads at different nodes (locations).

PyPSA uses the following terminology to define grid infrastructure:

Attributes
-----------

1. **Bus**: Nodes (locations) where power plants and loads are connected
    - **name**: Unique identifier of the bus
    - **v_nom**: Nominal voltage of the node
    - **carrier**: Energy carrier, which can be “AC” or “DC” for electricity buses
    - **x**: Longitude coordinate
    - **y**: Latitude coordinate

2. **Generator**: Power plants that generate electricity
    - **name**: Unique identifier of the generator
    - **p_nom**: Nominal power capacity, used as a limit in optimization
    - **p_max_pu**: Maximum output for each snapshot as a fraction of p_nom
    - **p_min_pu**: Minimum output for each snapshot as a fraction of p_nom
    - **p_set**: Active power set point (for power flow analysis)
    - **marginal_cost**: Marginal cost of producing 1 MWh
    - **carrier**: Energy carrier, such as “AC” or “DC”
    - **x**: Longitude coordinate
    - **y**: Latitude coordinate
    - **sign**: Power sign (positive for generation, negative for consumption)

3. **Load**: Demand units that consume electricity
    - **name**: Unique identifier of the load
    - **p_set**: Active power set point (for inflexible demand)
    - **marginal_cost**: Marginal cost of producing 1 MWh
    - **x**: Longitude coordinate
    - **y**: Latitude coordinate
    - **sign**: Power sign (positive for generation, negative for consumption)

4. **Line**: Transmission grids that transmit electricity
    - **name**: Unique identifier of the line
    - **s_nom**: Nominal transmission capacity
    - **capital_cost**: Capital cost for extending lines by 1 MVA
    - **r**: Resistance in Ohms
    - **bus0**: First node the line is connected to
    - **bus1**: Second node the line is connected to
    - **x**: Series reactance
    - **s_nom_extendable**: Flag to enable s_nom expansion

5. **Network**: The PyPSA network to which all the components are integrated
    - **name**: Unique identifier of the network
    - **snapshots**: List of timesteps (e.g., hourly, quarter-hourly, daily, etc.)

A PyPSA network model can be created by defining nodes as locations for power generation and consumption, interconnected by transmission lines with nominal transmission capacity (`s_nom`). These components can be further constrained operationally, for instance, by nominal power, efficiency, ramp rates, and other factors.

Currently, a limitation of the PyPSA model is the inability to define flexible loads.

Modeling Redispatch in ASSUME
--------------------------------

Modeling redispatch in the ASSUME framework using PyPSA primarily includes two parts:

Congestion Identification
--------------------------

The first step is to check for congestion in the network. The linear power flow (LPF) method is particularly useful for quick assessments of congestion and redispatch needs. PyPSA provides the `network.lpf()` function for running linear power flow. This method is significantly faster than a full non-linear AC power flow, making it suitable for real-time analysis or large network studies.

The active power flows through the lines can be retrieved using `network.lines_t.p0`. These can be compared to the nominal capacity of the lines (`s_nom`) to determine whether there is congestion.

```python
line_loading = network.lines_t.p0 / network.lines.s_nom
```

If line loading exceeds 1, it suggests there is congestion.

Redispatch of Power Plants
---------------------------

Once congestion is identified at any line or timestep, the redispatch mechanism is applied to alleviate it.

Steps for Redispatch
^^^^^^^^^^^^^^^^^^^^^

1. **Fixing Dispatches from the EOM Market**
   EOM market dispatches are fixed to model redispatch from power plants with accurate cost considerations. EOM dispatches are treated as a `Load` in the network, with dispatches specified via `p_set`. Generators are assigned a positive sign, and demands are given a negative sign.

2. **Upward Redispatch from Market and Reserved Power Plants**
   Due to PyPSA’s limitations in modeling load flexibility, upward redispatch is added as a `Generator` with a positive sign. The maximum available capacity for upward redispatch is restricted using the `p_max_pu` factor, estimated as the difference between the current generation and the maximum power of the power plant.

   ```python
   p_max_pu_up = (max_power - volume) / max_power
   ```

3. **Downward Redispatch from Market Power Plants**
   Similarly, downward redispatch is modelled as a `Generator` with a negative sign. The maximum available capacity for downward redispatch is restricted by the `p_max_pu` factor.

4. **Upward and Downward Redispatch from Other Flexibilities**
   Flexibility for redispatch is also modelled as generators, with positive signs for upward redispatch and negative signs for downward redispatch.

Objective
---------

The aim of redispatch is to minimize the overall cost of redispatch, including costs for starting up, shutting down, ramping up, ramping down, and other related actions.
