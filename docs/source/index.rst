.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

ASSUME: Agent-Based Electricity Markets Simulation Toolbox
===========================================================

**ASSUME** is an open-source toolbox for agent-based simulations of European electricity
markets, with a primary focus on the German market setup. Developed as an open-source model,
its primary objectives are to ensure usability and customizability for a wide range of
users and use cases in the energy system modeling community.

The unique feature of the ASSUME tool-box is the integration of **Deep Reinforcement
Learning** methods into the behavioural strategies of market agents.
The model offers various predefined agent representations for both the demand and
generation sides, which can be used as plug-and-play modules, simplifying the
reinforcement of learning strategies. This setup enables research into new market
designs and dynamics in energy markets.

Developers
----------
The project is `developed by <https://assume.readthedocs.io/en/latest/developers.html>`_
a collaborative team of researchers from INATECH at the University of Freiburg,
IISM at Karlsruhe Institute of Technology, Fraunhofer Institute for Systems and Innovation Research,
Fraunhofer Institution for Energy Infrastructures and Geothermal Energy,
and FH Aachen - University of Applied Sciences. Each contributor brings valuable
expertise in electricity market modeling, deep reinforcement learning, demand side
flexibility, and infrastructure modeling.

Funding
-------
ASSUME is funded by the Federal Ministry for Economic Affairs and Climate Action (BMWK).
We are grateful for their support in making this project possible.

Check out the :doc:`quick_start` section for further information, including
how to :doc:`installation` the project.

.. note::

   This project is under active development.

Documentation
=============

**Getting Started**

* :doc:`introduction`
* :doc:`installation`
* :doc:`quick_start`

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   introduction
   installation
   quick_start

**Examples**

* :doc:`examples_basic`
* :doc:`example_simulations`

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Examples

   examples_basic
   example_simulations

**User Guide**

User Guide
==========

* :doc:`market_config`
* :doc:`market_mechanism`
* :doc:`scenario_loader`
* :doc:`outputs`
* :doc:`unit_operator`
* :doc:`units`
* :doc:`learning`
* :doc:`learning_algorithm`
* :doc:`support_policies`
* :doc:`distributed_simulation`
* :doc:`manual_simulation`
* :doc:`realtime_simulation`
* :doc:`command_line_interface`
* :doc:`assume`

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: User Guide

   market_config
   market_mechanism
   scenario_loader
   outputs
   unit_operator
   units
   learning
   learning_algorithm
   support_policies
   distributed_simulation
   manual_simulation
   realtime_simulation
   command_line_interface
   assume


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


**Help & References**

* :doc:`release_notes`
* :doc:`developers`

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Help & References

   release_notes
   developers
