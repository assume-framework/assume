.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later


Modelling Redispatch in ASSUME
==============================

This section demonstrates the modeling and simulation of the redispatch mechanism using PyPSA as a plug-and-play module within the ASSUME framework.
Modeling redispatch in ASSUME using PyPSA consists of two main steps: first, identifying network congestion based on the cleared Energy-Only Market (EOM) dispatch, and second, resolving congestion by optimizing upward and downward redispatch actions.

Congestion Identification
-------------------------
The first step is to check whether the cleared EOM dispatch leads to congestion in the network.

Overview of Redispatch Modeling in PyPSA
----------------------------------------

The PyPSA network model can be created to visualize line flows using EOM clearing outcomes of generation and loads at different nodes (locations).

PyPSA uses the following terminology to define grid infrastructure:

Attributes
----------

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
    - **sign**: Power sign (negative by default)

4. **Line**: Transmission grids that transmit electricity
    - **name**: Unique identifier of the line
    - **s_nom**: Nominal transmission capacity of apparent power in MVA
    - **s_max_pu**: Maximum loading as a fraction of s_nom (value between 0 and 1)
    - **capital_cost**: Capital cost for extending lines by 1 MVA
    - **r**: Resistance in Ohms
    - **bus0**: First node the line is connected to
    - **bus1**: Second node the line is connected to
    - **x**: Series reactance
    - **s_nom_extendable**: Flag to enable s_nom expansion

5. **Network**: The PyPSA network to which all the components are integrated
    - **name**: Unique identifier of the network
    - **snapshots**: List of timesteps (e.g., hourly, quarter-hourly, daily, etc.)

ASSUME uses PyPSA's linear power-flow method ``network.lpf()`` for this purpose. The method is computationally efficient and suitable for congestion checks after market simulations.
The cleared EOM generation is assigned to the corresponding PyPSA generators via ``generators_t.p_set``. Demand is assigned to normal PyPSA loads via ``loads_t.p_set``. Since demand bids in ASSUME are commonly represented with negative volumes, these values are converted to positive load values before being passed to PyPSA.

.. code-block:: python

    redispatch_network.generators_t.p_set = gen_p_set
    redispatch_network.loads_t.p_set = load_p_set.abs()

The resulting line flows are retrieved from network.lines_t.p0 and compared with the available transmission capacity:

.. code-block:: python

    line_loading = network.lines_t.p0.abs() / ( network.lines.s_nom * network.lines.s_max_pu )

If the line loading exceeds 1, the corresponding line is congested and redispatch optimization is triggered.

Redispatch of Power Plants
--------------------------
Once congestion is detected, ASSUME optimizes redispatch actions to relieve overloaded lines.
In the updated redispatch formulation, cleared EOM generator dispatch is represented directly on the generator side. The cleared generator schedule is assigned through generators_t.p_set for linear power-flow analysis and is also reflected in the generator bounds using p_min_pu = p_max_pu.
This ensures that the pre-redispatch dispatch is treated as fixed while redispatch flexibility is represented separately through additional upward and downward redispatch generators.

Steps for Redispatch
^^^^^^^^^^^^^^^^^^^^^

1. Fixing cleared EOM generator dispatch

The cleared EOM dispatch of each generator is fixed using two complementary representations. First, generators_t.p_set is used for the linear power-flow calculation. Second, the same dispatch is converted into per-unit values and assigned as equal lower and upper bounds:

.. code-block:: python

    p_set_pu = cleared_dispatch / p_nom
    generators_t.p_min_pu = p_set_pu
    generators_t.p_max_pu = p_set_pu

This prevents the base generator dispatch from being freely changed during redispatch optimization.

2. Representing upward redispatch

Upward redispatch is modelled through additional generator components with positive sign. The available upward redispatch capacity is limited by the difference between the availability-adjusted maximum power and the cleared EOM dispatch:

.. code-block:: python

    p_max_pu_up = (max_power - gen_p_set) / p_nom

Here, max_power represents the available maximum generation capacity in the respective timestep, gen_p_set is market cleared capacity and max_power is fraction of p_nom taking availability factor into account.

3. Representing downward redispatch

Downward redispatch is modelled through additional generator components with negative sign. The available downward redispatch capacity is limited by the difference between the cleared EOM dispatch and the minimum generation level:

.. code-block:: python

    p_max_pu_down = (gen_p_set - min_power) / p_nom

A negative generator sign means that dispatching this component reduces the physical generation of the corresponding unit.

4. Backup redispatch

Additional upward and downward backup generators are added at each node. These backup units have high marginal costs and ensure that the optimization remains feasible if market-based redispatch bids are insufficient to resolve congestion.

Objective
---------

The redispatch optimization minimizes the net cost of redispatch actions while satisfying network constraints. Upward redispatch is assigned a positive marginal cost. Downward redispatch is assigned the negative of the submitted redispatch price, because reducing generation avoids the corresponding generation cost.

Therefore, the net redispatch cost is evaluated using signed accepted redispatch volumes multiplied by their accepted prices:

.. math::

    C_{redispatch} = \sum_{o \in O} V^{accepted}_{o} \cdot P^{accepted}_{o}

where :math:`V^{accepted}_{o}` is the signed accepted redispatch volume and :math:`P^{accepted}_{o}` is the corresponding accepted redispatch price.

Positive accepted volumes correspond to upward redispatch, while negative accepted volumes correspond to downward redispatch.
