.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

Unit Operator
==============

Assume is created using flexible and usable abstractions, while still providing flexibility to cover most use cases of market modeling. This is also true for the unit operator class.
In general the task of this calls can range from simple passing the bids of the technical units through to complex portfolio optimization of all units assigned to one operator. This text aims
to explain its current functionalities and the possibilities the unit


Basic Functionalities
----------------------

In general, the unit operator is the operator of the technical units and can either have one unit or multiple units assigned to it.
The unit operator is responsible for the following tasks:

- **Registering to the respective markets that it wants to participate in**
- **Handling the opening message from a market by asking for bids from its units**
- **Passing or processing the bids of the technical units through to the market**
- **Handling the market feedback and communicate actual dispatch of technical units**
- **Collects a variety of data and sends it to the output unit which writes it in coordinated way to the database**

As one can see from all the task the unit oporator covers, that it orchestrates and coordinates the technical units and the markets.


Portfolio Strategies
----------------------

The main flexibility of a unit operator is that all bids and technical constraints from its units can be processed
jointly before being sent to the market. This enables portfolio-level bidding strategies, where the operator acts
as a single strategic agent across all units it manages.

ASSUME provides two levels of general portfolio strategies, both defined in :mod:`assume.strategies.portfolio_strategies`:

- :class:`~assume.strategies.portfolio_strategies.UnitsOperatorDirectStrategy` — passes each unit's individual bids through unchanged, simply aggregating them into a single orderbook.
- :class:`~assume.strategies.portfolio_strategies.UnitsOperatorEnergyHeuristicCournotStrategy` — applies a Cournot-style markup on top of each unit's marginal cost bid, scaled by the total capacity of the operator's portfolio.

Portfolio Reinforcement Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A reinforcement learning variant is available in :class:`~assume.strategies.portfolio_learning_strategies.PortfolioLearningStrategy`.
Instead of bidding each unit independently, the RL agent observes the full portfolio state and learns a joint bidding
policy. The observation space includes:

- **Cyclical hour-of-day encoding** (sin/cos)
- **Forecasted residual load** over the foresight horizon, scaled by maximum demand
- **Forecasted price** over the foresight horizon, scaled by maximum bid price
- **Inframarginal generation forecast**: fraction of portfolio capacity that is cheaper than the forecasted price
- **Marginal cost curve**: flexible capacity and upper cost bound per cost quantile (``nbins`` bins)

The agent outputs a markup multiplier per cost bin per time step. Units are sorted by marginal cost, and each unit's
flexible capacity is bid at ``marginal_cost × markup[bin]``. Inflexible (must-run) generation is always bid at
marginal cost. The reward signal is the difference between realised profit and a competitive benchmark (all units
bidding at marginal cost given the price forecast).
