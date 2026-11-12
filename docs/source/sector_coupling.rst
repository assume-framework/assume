.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: MIT


Sector-Coupled Systems
======================

This section describes the sector-coupled unit type in ASSUME. A
:class:`~assume.units.sector_coupled_system.SectorCoupledSystem` represents a
physical installation with several internal energy-converters and storages
across different energy carriers — for example a heat pump combined with a gas
boiler and a thermal store, or an electrolyser combined with hydrogen storage
and a fuel cell — as **one single market participant**.

Instead of bidding a fixed or heuristically shifted demand profile, the unit
runs an internal Pyomo optimisation that decides how the internal components
serve the required carrier demand, and the resulting aggregate electrical
exchange is then offered on the energy-only market (EOM) and, optionally, in
the redispatch market as a flexible load.

.. note::

    Sector-coupled systems complement the single-technology demand-side units
    described above (buildings, hydrogen plants, steam plants). They are meant
    for installations whose economics are only defined at the *system* level,
    i.e. where the split between internal technologies must be decided by the
    optimisation rather than fixed in advance.

Unit Architecture
-----------------

A sector-coupled system is built from a **parent unit** and its **components**:

- Each row in ``sector_coupled_units.csv`` defines one component (e.g.
  ``hs_north_heat_pump``).
- Rows sharing the same ``parent_unit`` (e.g. ``hs_north``) are grouped by the
  scenario loader into a single ``SectorCoupledSystem`` parent.
- The parent is the only object the market sees in the default
  ``market_representation: aggregate`` mode.

**The components are never optimised independently.** Carrier balances,
storage states and all component constraints live inside one parent Pyomo
optimisation (see
:py:class:`assume.units.sector_coupled_system.SectorCoupledSystem`). This
guarantees that every market position the parent bids is physically feasible
for the whole installation.

Two presets are provided via the ``system_type`` column:

- ``heat``: ``heat_pump`` + ``boiler*`` (e.g. ``boiler_gas``) +
  ``thermal_storage``
- ``hydrogen``: ``electrolyser`` + ``hydrogen_seasonal_storage`` + ``fuel_cell``

Component names are matched against the preset by prefix, so the exact suffix
is free (``boiler_gas``, ``heat_pump_a``, ...). New sectors can be added
without a new unit class by passing a custom ``carrier_balances`` description
(see `Extending`_).

Carrier Balances and Sign Convention
------------------------------------

Each preset defines one or more **carrier balances**, one per non-electric
carrier. For every time step the model requires that the carrier supply (sum
of producer outputs) equals the carrier demand:

- **Heat preset**: heat demand must be met exactly (``sense: eq``). Heat may
  come from the heat pump and/or the boiler, and may be buffered in the
  thermal storage.
- **Hydrogen preset**: hydrogen is conserved exactly (``sense: eq``).
  Electrolyser production must be allocated to hydrogen demand, storage
  charging, or another explicitly modelled hydrogen-consuming component (e.g.
  the fuel cell). Unmodelled hydrogen surplus is not permitted.

The electrical coupling of the parent to the grid is described by the
variable ``total_power_input``:

- **positive** values: electricity import / consumption,
- **negative** values: electricity export / generation.

Negative values are required for systems that contain electricity-producing
components such as fuel cells; such systems can therefore be net exporters in
some time steps.

On the market side, ASSUME's convention is the opposite sign: electricity
demand is bid with negative volumes. The unit handles the conversion
internally, so a consumption of 4 MW in the internal model appears as a bid
volume of -4 MW in the market.

Internal Optimisation
---------------------

The parent model contains, per component, a Pyomo block with the
technology-appropriate variables (electrical input, carrier outputs, fuel
input, charge/discharge/SOC for storages) and the following coupling
structures:

- carrier balances per time step (see above),
- storage dynamics with charge/discharge power limits, SOC bounds,
  efficiencies and storage losses,
- component-specific constraints (e.g. heat pump power/heat ratio bounded by
  the coefficient of performance),
- fuel consumption and variable operating costs per component.

The default objective minimises the total variable cost over the horizon
(``objective: min_variable_cost``).

Additional modelling options:

- ``enforce_terminal_soc`` (default ``True``): in full-horizon operation the
  storage SOC is returned to its initial/reference terminal level.
- ``absolute_demands``: optional horizon-wide (instead of per-step) demand
  constraints per carrier, useful for monthly or seasonal balances.
- The hydrogen preset optionally consumes a
  ``seasonal_storage_schedule`` forecast for long-term hydrogen storage.

Market Participation
--------------------

The market interface is built around a **reference solution**: the parent
solves the cost-optimal model for the full optimisation horizon. From this
reference, the two market positions are derived.

EOM
^^^

The strategy registered under ``sector_coupled_energy_optimization``
(:py:class:`assume.strategies.naive_strategies.SectorCoupledEnergyNaiveStrategy`)
bids the reference consumption profile on the EOM:

- consumption time steps are bid with negative volumes,
- currently every non-zero bid uses a must-take price (3000 EUR/MWh in the
  current implementation).

.. note::

    The must-take pricing is an interim solution. It makes the EOM
    price signal not influence *when* the system consumes; the internal
    schedule is cost-optimal with respect to fuel and marginal costs only.
    An opportunity-cost based bidding strategy that prices flexibility from
    the internal model is the planned next step.

Redispatch
^^^^^^^^^^

For the redispatch market, the parent offers a **costed flexibility
envelope** around the reference schedule, built by the strategy registered
under ``sector_coupled_energy_redispatch``
(:py:class:`assume.strategies.naive_strategies.SectorCoupledRedispatchNaiveStrategy`):

- the base volume is the EOM-cleared reference schedule,
- the upward and downward headroom at each time step is computed by
  re-solving the internal model and accepting a power level only if the
  resulting cost increase stays within ``cost_tolerance``
  (``flexibility_measure: cost_based_load_shift``),
- the physical consumption interval ``[min_consumption, max_consumption]`` is
  converted to the market sign convention:
  ``min_power = -max_consumption``, ``max_power = -min_consumption``,
- the bid price is the marginal cost of the internal system at the base
  consumption.

On the clearing side, the redispatch market accepts these flexible loads and
integrates them into the PyPSA network model (see
:doc:`redispatch_modeling`): each registered flexible load appears as a load
node with additional upward/downward flexibility variables, so congestion can
be relieved by shifting the system's consumption within its feasible and
costed envelope. The market configuration must provide the additional fields
``node``, ``min_power`` and ``max_power`` (payment mechanism
``pay_as_bid``).

Aggregate vs. Component Market Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``market_representation`` column controls what the market sees:

- ``aggregate`` (default): only the parent unit appears in the order books.
- ``component``: the parent additionally exposes child market IDs
  (``{parent}_{component}``) linked back via ``parent_unit_id``. In this mode
  the parent's EOM volume and redispatch headroom are decomposed across the
  components using cost-based weights.

.. note::

    The component decomposition is an **accounting attribution**, not an
    independent feasibility guarantee: the child positions always sum exactly
    to the parent position, but each child is not optimised on its own.
    Component-level redispatch clearing is not implemented yet — the
    redispatch market always clears the physical parent aggregate.

Scenario Inputs
---------------

Sector-coupled systems are loaded from ``sector_coupled_units.csv`` (one row
per component) plus the usual forecast files. The relevant columns are:

System-level columns (read from the parent's rows):

- ``parent_unit``: groups component rows into one parent system.
- ``unit_type``: must be ``sector_coupled_system``.
- ``system_type``: preset, ``heat`` or ``hydrogen``.
- ``unit_operator``, ``node``: operator and grid node of the parent.
- ``bidding_EOM``, ``bidding_redispatch``: strategy registry keys
  (``sector_coupled_energy_optimization``,
  ``sector_coupled_energy_redispatch``).
- ``objective`` (default ``min_variable_cost``),
  ``flexibility_measure`` (default ``cost_based_load_shift``),
  ``cost_tolerance`` (default 10).
- ``market_representation``: ``aggregate`` or ``component``.
- ``enforce_terminal_soc``: ``TRUE``/``FALSE``.

Component-level columns (per technology row):

- ``name``, ``technology``: component name and technology
  (``heat_pump``, ``boiler_gas``, ``thermal_storage``, ``electrolyser``,
  ``hydrogen_seasonal_storage``, ``fuel_cell``).
- Converter columns: ``max_power``, ``min_power``, ``cop`` (heat pump),
  ``efficiency``, ``fuel_type`` (e.g. ``natural_gas``), ``capacity``.
- Storage columns: ``max_power_charge``, ``max_power_discharge``,
  ``min_soc``, ``max_soc``, ``efficiency_charge``, ``efficiency_discharge``,
  ``initial_soc``, ``storage_loss_rate``, ``storage_type``.

Forecast files:

- ``forecasts_df.csv`` must contain the electricity price forecast (e.g. the
  ``price_EOM`` column) **and** one demand column per carrier and node, using
  the convention ``heat_demand_{node}`` for heat systems and
  ``h2_demand_{node}`` for hydrogen systems.
- ``fuel_prices_df.csv`` provides the fuel prices (e.g. ``natural_gas``).

For the redispatch grid integration, the loader additionally registers each
parent as a flexible load (``build_flexible_loads``): the electrical nominal
capacity ``p_nom`` is derived as the sum of the ``max_power`` values of all
directly electricity-consuming components (heat pumps, electrolysers, electric
boilers).

Examples
--------

Two example scenarios are included:

- ``examples/inputs/example_04b``: three grid nodes, each with one heat
  system (heat pump + gas boiler + thermal storage), an EOM
  (``pay_as_clear``) and a redispatch market (``pay_as_bid``).
- ``examples/inputs/example_04c``: the same setup extended with one hydrogen
  system (electrolyser + hydrogen storage + fuel cell) per node.

Both are registered in ``examples/examples.py``:

.. code-block:: python

    example = "sector_coupled_3_nodes_heat"      # example_04b, heat only
    example = "sector_coupled_3_nodes_heat_h2"   # example_04c, heat + hydrogen

Extending
---------

- **New sectors/carriers**: pass a custom ``carrier_balances`` dict to the
  ``SectorCoupledSystem`` constructor (or via the loader) instead of relying
  on a preset. Each carrier entry defines its demand attribute, the balance
  ``sense`` (``eq`` or ``ge``), the component name prefixes that produce or
  store the carrier, and the supply variables.
- **Opportunity-cost bidding**: the current EOM strategy uses must-take
  prices (see above). The planned replacement prices each time step's
  flexibility with the actual opportunity cost from the internal model, so
  the EOM price signal reshapes the consumption schedule.
