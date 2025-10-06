.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

Bidding Strategies
==================

Bidding strategies are a core concept of ASSUME which describe how agents bid their assets on a market.

Overview
---------

As described in the :ref:`exchangeable_bidding_strategy`, a Bidding Strategy dictates the bidding behavior of units in different markets, whereby it maps certain states and technical constraints to bidding decisions.
In general, there is a distinction between two kinds of strategy classes:

- :py:meth:`assume.strategies.portfolio_strategies.UnitOperatorStrategy`, which indicate the use in a UnitsOperator and can be used to provide a Portfolio optimization.
- :py:meth:`assume.strategies.naive_strategies.UnitStrategy`, which indicate the strategy used for a single unit of :doc:`unit` and

Both types have a function `calculate_bids` which is called with the information of the market and bids to bid on.

UnitOperatorStrategy
--------------------

The UnitsOperatorStrategies can be used to adjust the behavior of the UnitsOperator.
The default is the :py:meth:`assume.strategies.portfolio_strategies.DirectUnitOperatorStrategy`.
It formulates the bids to the market according to the bidding strategy of the each unit individually.
This calls `calculate_bids`` of each unit and returns the aggregated list of all individual bids of all units.
This is the default for all UnitsOperators which do not have a separate strategy configured.

Another implementation includes the :py:meth:`assume.strategies.portfolio_strategies.CournotPortfolioStrategy` that adds a markup to the marginal cost of each unit of the units operator.
The marginal cost is computed with NaiveSingleBidStrategy and the markup depends on the total capacity of the unit operator.

UnitStrategies
--------------

The ASSUME framework provides multiple options in terms of Bidding Strategy methodologies:

 ============================== =============================================================
  Bidding Strategy Methodology   Description
 ============================== =============================================================
  Heuristic                      Basic methodology to form bids, based on participating in a market mechanism that follows the Merit Order principle. These strategies do not utilise forecasting
                                 or consider finer details such as the effects of changing a power plant's operational state (start-up costs etc.),
                                 so the bid volume is of the order of its max. power capacity (given ramping constraints) and the bid price is set to the marginal cost.
  Optimization                   This methodology, based on the flexABLE methodology [`Qussous et al. 2022 <https://doi.org/10.3390/en15020494>`_],
                                 offers more refined strategising compared to the Naive methods. Applicable to power plant and storage units, it incorporates market dynamics to a
                                 greater degree by forecasting market prices, as well as accounting for operational history,
                                 potential power loss due to heat production,
                                 and the fact that a plant can make bids for multiple markets at the same time.
  DMAS-Optimization              Optimization using forecasts and avoided cost calculation used for smart bids coming from the DMAS methodology <https://github.com/NOWUM/dmas/>`_ .
  Learning                       A `reinforcement learning <https://assume.readthedocs.io/en/latest/learning.html>`_ (RL) approach to formulating bids, for an Energy-Only Market.
                                 Agents perform actions (choose bid price(s) (and for storage a direction)) informed by observations
                                 (including forecasted residual load, forecasted price, marginal cost). Bid volumes are fixed (to the maximum possible volume).
                                 Based on the reward (profits) from accepted bids, agents learn to optimise bids to maximise profits.
  Miscellaneous                  Other bidding strategies not belonging to a specific methodology.
 ============================== =============================================================

For each Bidding Strategy methodology there are multiple Bidding Strategy options depending on the market that the bid is intended for,
as well as the type of unit making the bid.

Accordingly, each Bidding Strategy has an associated ID which takes the form "methodology_market_unit". The "market" and/or "unit" components are omitted if
the strategy is applicable across multiple markets and/or unit types, e.g. :code:`"naive"`.
This "bidding_strategy_id" needs to be entered when defining a unit's bidding strategy. Each Bidding Strategy and associated ID for each methodology is defined and described further below.

When constructing a units CSV file, the bidding strategies are set using :code:`"bidding_*"` columns, where the market type the bidding strategy is applied to
follows the underscore (the market names need to match with those in the config file).
