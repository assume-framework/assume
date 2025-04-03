.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

Units Overview
==============

This document provides an overview of the various units available within the ASSUME framework, detailing their respective submodules.

Submodules
----------

Demand Module
-------------

The `assume.units.demand` module contains classes and functions for modeling demand in energy systems.

.. automodule:: assume.units.demand
   :members:
   :undoc-members:
   :show-inheritance:

Power Plant Module
------------------

The `assume.units.powerplant` module includes definitions for different types of power plants and their operational characteristics.

.. automodule:: assume.units.powerplant
   :members:
   :undoc-members:
   :show-inheritance:

Storage Module
--------------

The `assume.units.storage` module provides classes for various energy storage solutions, including batteries and other storage technologies.

.. automodule:: assume.units.storage
   :members:
   :undoc-members:
   :show-inheritance:

Demand Side Technology Module
--------------------------------
The `assume.units.dsm_load_shift` module integrates load-shifting capabilities into agents, allowing them to optimize their energy consumption based on market conditions.

.. automodule:: assume.units.dsm_load_shift
   :members:
   :undoc-members:
   :show-inheritance:


Demand Side Technology Components
---------------------------------

The `assume.units.dst_components` module focuses on components used in Demand Side Technology systems such as electrolizers or storage units.

.. automodule:: assume.units.dst_components
   :undoc-members:
   :show-inheritance:

The following classes are defined within the `assume.units.dst_components` module:

Electrolyser
^^^^^^^^^^^^^

A class representing an electrolyser component for hydrogen production.

.. autoclass:: assume.units.dst_components.Electrolyser
   :members:
   :undoc-members:


Hydrogen Buffer Storage
^^^^^^^^^^^^^^^^^^^^^^^

A class representing a hydrogen storage unit.

.. autoclass:: assume.units.dst_components.HydrogenBufferStorage
   :members:
   :undoc-members:

Seasonal Hydrogen Storage
^^^^^^^^^^^^^^^^^^^^^^^^^

A class representing a seasonal hydrogen storage unit.

.. autoclass:: assume.units.dst_components.SeasonalHydrogenStorage
   :members:
   :undoc-members:

Direct Reduced Iron Plant
^^^^^^^^^^^^^^^^^^^^^^^^^

A class representing a Direct Reduced Iron (DRI) plant.

.. autoclass:: assume.units.dst_components.DRIPlant
   :members:
   :undoc-members:

Direct Reduced Iron Storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A class representing DRI storage solutions.

.. autoclass:: assume.units.dst_components.DRIStorage
   :members:
   :undoc-members:

Electric Arc Furnace
^^^^^^^^^^^^^^^^^^^^

A class representing an electric arc furnace component.

.. autoclass:: assume.units.dst_components.ElectricArcFurnace
   :members:
   :undoc-members:

Heat Pump
^^^^^^^^^

A class representing a heat pump component in energy systems.

.. autoclass:: assume.units.dst_components.HeatPump
   :members:
   :undoc-members:

Boiler
^^^^^^^

A class representing a boiler component in energy systems.

.. autoclass:: assume.units.dst_components.Boiler
   :members:
   :undoc-members:

Electric Vehicle
^^^^^^^^^^^^^^^^

A class representing an electric vehicle component.

.. autoclass:: assume.units.dst_components.ElectricVehicle
   :members:
   :undoc-members:

Generic Storage
^^^^^^^^^^^^^^^

A class representing a generic storage unit for various applications.

.. autoclass:: assume.units.dst_components.GenericStorage
   :members:
   :undoc-members:

PV Plant
^^^^^^^^

A class representing a photovoltaic (PV) power plant component.

.. autoclass:: assume.units.dst_components.PVPlant
   :members:
   :undoc-members:

Module Contents
---------------

The following section provides an overview of the primary contents within the `assume.units` module, summarizing its main functionalities.

.. automodule:: assume.units
   :members:
   :undoc-members:
   :show-inheritance:
