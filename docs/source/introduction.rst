.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

##########################################
 Introduction
##########################################


Target user group
=================

ASSUME in general is intended for researchers, planners, utilities and everyone
searching to understand market dynamics of energy markets.
It provides an easy-to-use tool-box as a free software that can be tailored
to the specific use case of the user.


Architecture
============
In the following figure the architecture of the framework is depicted. It can be roughly divided into two parts.
On the left side of the world class the markets are located and on the right side the market participants,
which are here named units. Both world are connected via the orders that market participants place on the markets.
The learning capability is sketched out with the yellow classes on the right side, namely the units side.

.. image:: img/architecture.svg
    :align: center
    :width: 600px



Key Features and Definitions
============================

Markets
---------

Under the term **market**, we generally understand a place that facilitates the exchange of goods and services
among different agents (or market participants). One of the primary goals of the ASSUME framework is to allow
users to directly work with existing market designs provided with the framework or to quickly implement their
own market with a specific design.

It is planned to include the energy-only market (EOM), representing the combination of day-ahead (DA) and
intraday (ID) markets, during the initial release of the framework. A control-reserve market (CRM) will be
implemented during the following development steps. In the future, the EOM should be split into separate DA
and ID markets to allow for even more detailed market representation if there is a demand for this feature.


Modularity
----------
To facilitate the ease of work with different markets, we focus on providing a highly modular implementation
of the market modules. The implementation consists of the parent market class, which provides the required
syntax for market implementations and has integrated auxiliary functions. Using the parent market class and
following the necessary syntax, all child classes representing different market types are guaranteed to be
fully supported by the basic functionality of the framework. Further, we integrate a market operator
class that can operate one or more markets. It is responsible for their coordination and can be used
to integrate post-clearing analysis or steps. One example would be the classical redispatch after an
assessment of the feasibility of the clearing.


Coupling
--------
Modern energy systems span many countries and incorporate multiple sectors. Therefore, it is crucial to
integrate market coupling, which links different control and market areas to facilitate energy exchange and
harmonize the grid. During the project's initial phase, this coupling incorporates only the electricity
structure, extending the market coupling to other areas, such as natural gas or hydrogen markets, is possible.
This extension is planned for the later stages of the project.


Interchangeable market clearing algorithms
------------------------------------------
Similar to the modularity of the markets, the market clearing algorithms are also easily interchangeable.
The user can test markets using different pricing schemes, such as uniform or pay-as-bid or other clearing
mechanisms. Utilizing the given parent market clearing algorithm, the user can also implement a specific
algorithm fit for research purposes.


Market Participants
===================

The market participants, here labeled units, comprise all entities acting in the respective markets and are at
the core of any agent-based simulation model. The entirety of their behavior leads to the market and system
outcome as a bottom-up simulation model, respectively.

Modularity of Units
-------------------

As described before, the ASSUME toolbox should allow many analyses and be adaptable to the researchers' needs.
Therefore, parent classes are used to derive units for both supply and demand sides. These parent classes provide
the basic structure, functionality, and syntax. This implementation, when followed, guarantees full integration
into the framework, offers coherence between different agent types, and enables ease of implementing new agent types.

Further there is a distinction established between the unit and the unit operator. In the simplest setting, each
unit operator has one unit. Then the unit just communicates its operational windows according to its technical
restrictions, and based on this, the unit's operator places bids on the market according to the prevalent
bidding strategy. To enable portfolio optimizations, a unit operator can, however, have multiple units for which
they define a bid together considering all the technical constraints and costs.


.. _exchangeable_bidding_strategy:

Exchangeable Bidding Strategy
------------------------------

Bidding strategy dictates the bidding behavior of the units in different markets. They map certain states and
technical constraints to bidding decisions, respectively. In line with the modularity of the units, their respective
bidding strategies are also interchangeable and customizable. The framework allows us to integrate different
strategies for bid formulations, such as rule-based, optimization-based strategies, or such that are derived by
learning algorithms. This feature lets users quickly test different pre-implemented bidding strategies delivered
with ASSUME or implement their own bidding strategy. Further, the used bidding strategy can be held by either the
unit itself or the unit operator, depending on whether or not a portfolio optimization is used. For example, each
unit has a bidding strategy, a unit operator holds multiple units and has a bidding strategy on its own. Assuming
that the portfolio optimization is not activated, the unit operator formulates bids according to the bidding
strategy of each unit and places them as orders on the market. If the portfolio optimization is activated, the
unit operator ignores the bidding strategy of the units and instead uses their own to formulate bids for the
entirety of their portfolio.


Learning Capabilities
----------------------

The deep reinforcement learning capability of the modeled agents in a multi-agent setting is one of the core features
of the toolbox. Given specific state, action spaces, and a particular reward function, the unit operators learn a
bidding policy to maximize their reward by interacting in the market environment. The learning is here connected to
the units operator, since learning will need to be centralized to a certain extent, for example by technology and
power plant classes, if the simulation reaches a certain size. However, similar to the bidding strategies, the
learning can also be placed in each unit.

The initial release of the model includes (several) learning algorithms and some pre-learned policies for basic
types of agents.


Enabling More Sophisticated Bid Types
--------------------------------------

Besides the typical hourly bids, the day-ahead market also accepts loop, curtailable, linked, and exclusive
bid blocks [link]. For example, link blocks mean an offer is only accepted in one hour if its parent block was
also accepted in the previous hour. This trading behavior gets important when technical restrictions such as
shut-down and start-up times and costs are included. The markets are designed in a way that they allow for an
integration of such bids. It is planned to incorporate such bid types into the units and their strategies
(especially into the learning capabilities) at later stages of the framework development.



General
=======
This chapter comprises features that span the complete model.

Network
-------
The relation and interconnection between different markets and their participants are represented using graphs.
This is established from the very beginning of the model implementation and can later be utilized by the
communication layer to facilitate coordination between the markets and agents. In addition, an actual electrical
grid can be integrated into the respective markets to account for technical constraints or for re-dispatch calculations.

Communication layer
-------------------
Throughout the model, a variety of data exchanges need to take place. This exchange can have several forms. It
can be between the agents and the markets for order submission and market feedback. Moreover, it has to take
place between the global environment and the agents for the weather and scenario data. To handle that in a
standardized way, we facilitate the mango framework (`mango – Modular Python Agent Framework <https://gitlab.com/mango-agents/mango>`_)
which is an open-source agent-based simulation framework by University Oldenburg.
It is a Python framework for the development of multi-agent systems which can be used to build a
single intelligent software agent, provides simple interfaces for communication between agents, and enables
modularization of complex agents. A container mechanism is used to accelerate message exchange for agents
that reside within a dedicated process. Please note that the whole simulation is timed by the clock function
of mango as well. This means that every market opening and order placement is triggered by this clock.

Input-/Output Formats
---------------------
One major part of the model interoperability with other open-source tools and models is the possibility of
exchanging data. Hence, the Input and output formats are chosen to align with commonly used standards. TBD

Scalability
-----------
Scalability is the software's ability to operate effectively with different levels of workloads without
the need for a redesign. Under workload in the framework context, we understand the number of markets,
agents, simulation horizon, or any other parameter that can influence the system performance. It is intended
to incorporate the features allowing scalability from the initial phases of the project.

Parallel execution
------------------
Similar to real-life electricity markets, our simulation comprises multiple decentralized agents participating
on a centralized market platform. To ensure proper computational performance, the framework should be capable
of decentralized and parallel execution. This functionality would allow for an execution of the framework on
multiple systems, where different market participants operate independently both in terms of software as well as hardware.


Licence
=======

Copyright 2022-2025 :doc:`developers`

ASSUME is licensed under the `GNU Affero General Public License v3.0 <https://github.com/assume-framework/assume/blob/main/LICENSES/AGPL-3.0-or-later.txt>`_.
