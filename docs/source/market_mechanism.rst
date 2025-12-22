.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

##################
Market Mechanisms
##################

A Market Mechanism is used to execute the clearing, scheduled by the MarketRole in base_market.py

The method signature for the market_mechanism is given as::

  def clearing_mechanism_name(
    market_agent: MarketRole,
    market_products: list[MarketProduct],
  ):
    accepted_orders: Orderbook = []
    rejected_orders: Orderbook = []
    meta: list[Meta] = []
    return accepted_orders, rejected_orders, meta

The :code:`market_mechanism` is called by the MarketRole, which is the agent that is responsible for the market.
It is called with the :code:`market_agent` and the :code:`market_products`, which are the products that are traded in the current opening of the market.
This gives maximum flexibility as it allows to access properties from the MarketRole directly.
The :code:`market_mechanism` returns a list of accepted orders, a list of rejected orders and a list of meta information (for each tradable market product or trading zone, if needed).
The meta information is used to store information about the clearing, e.g. the min and max price, the cleared demand volume and supply volume, as well as the information about the cleared product.

In the Market Mechanism, the MarketRole is available to access the market configuration with :code:`market_agent.marketconfig` and the available Orders from previous clearings through :code:`market_agent.all_orders`.
In the future, the MarketMechanism will be a class which contains the additional information like grid information without changing the MarketRole.

The available market mechanisms are the following:

1. :py:meth:`assume.markets.clearing_algorithms.simple.PayAsClearRole`
2. :py:meth:`assume.markets.clearing_algorithms.simple.PayAsBidRole`
3. :py:meth:`assume.markets.clearing_algorithms.complex_clearing.ComplexClearingRole`
4. :py:meth:`assume.markets.clearing_algorithms.complex_clearing_dmas.ComplexDmasClearingRole`
5. :py:meth:`assume.markets.clearing_algorithms.redispatch.RedispatchMarketRole`
6. :py:meth:`assume.markets.clearing_algorithms.nodal_clearing.NodalClearingRole`
7. :py:meth:`assume.markets.clearing_algorithms.contracts.PayAsBidContractRole`


The :code:`PayAsClearRole` performs an electricity market clearing using a pay-as-clear mechanism.
This means that the clearing price is the highest price that is still accepted.
This price is then valid for all accepted orders.
For this, the demand and supply are separated, before the demand is sorted from highest to lowest order price
and the supply lowest to highest order price.
Where those two curves in a price over power plot meet, the market is cleared for the price at the intersection.
All supply orders with a price below and all demand orders above are accepted.
Where the price is equal, only partial volume is accepted.

The :code:`PayAsBidRole` clears the market in the same manner as the pay-as-clear mechanism, but the accepted_price is
the price of the supply order for both the demand order and the supply orders that meet this demand.

Complex clearing
=================

The :code:`ComplexClearingRole` performs an electricity market clearing using an optimization to clear the market.
Here, also profile block and linked orders are supported.
The objective function is a social welfare maximization, which is equivalent to a cost minimization:

.. math:: \min \left( {\sum_{b \in \mathcal{B}}\quad{u_b \: C_{b} \: P_{b, t}} \: T} \right),

where :math:`\mathcal{B}` is the set of all submitted bids, :math:`C_{b}` is the bid price,
:math:`P_{b, t}` is the volume offered (demand is negative)
and :math:`T` is the clearing horizon of 24 hours.
Decision variables are the acceptance ratio :math:`u_b` with :math:`u_b \in [0, 1] \quad \forall \: b \in \mathcal{B}`,
and the clearing status :math:`x_b` with :math:`x_b \in \{0, 1\} \: \forall \: b \in \mathcal{B}`.

The optimization problem is subject to the following constraints:

The energy balance constraint: :math:`\quad \sum_{b \in \mathcal{B}} P_{b, t} \: u_b = 0 \quad \forall \: t \in \mathcal{T}`,

The minimum acceptance ratio constraint: :math:`\quad u_{b} \geq U_{b} \: x_{b} \quad \mathrm{and} \quad u_{b} \leq x_{b} \quad \forall \: b \in \mathcal{B}`,

with the minimum acceptance ratio :math:`U_{b}` defined for each bid b.

The linked bid constraint, ensuring that the acceptance of child bids c is below the acceptance of their parent bids p
is given by: :math:`\mathbf{a}_{c, p} \: u_c \leq u_{p} \quad \forall \: c, p \in \mathcal{B}`,

with the incidence matrix :math:`\mathbf{a}_{c, p}` defining the links between bids as 1, if c is linked as child to p, 0 else.

Flows in the network are limited by the Net Transfer Capacity ('s_nom') of each line l: :math:`\quad -NTC_{l} \leq F_{l, t} \leq NTC_{l} \quad \forall \: l \in \mathcal{L}, t \in \mathcal{T}`,

Because with this algorithm, paradoxically accepted bids (PABs) can occur, the objective is solved in an iterative manner:

1. The optimization problem is solved with the objective function and all constraints.
2. The binary variables :math:`x_b` are fixed to the current solution.
3. The optimization problem is solved again without the minimum acceptance ratio constraint.
4. The market clearing prices are given as the dual variables of the energy balance constraint.
5. The surplus of each bid is calculated as the difference between the bid price and the market clearing price.
6. If the surplus for one or more bids is negative, the clearing status :math:`x_b` for those bids is set to 0 and the algorithm starts again with step 1.


If you want a hands-on use-case of the complex clearing check out the prepared tutorial in Colab: https://colab.research.google.com/github/assume-framework/assume

Nodal clearing
=================

The :code:`NodalClearingRole` performs an electricity market clearing of the bids submitted by market participants using an optimal power flow (OPF) approach.
Profile, block and linked orders are not supported.
The algorithm utilizes PyPSA to solve the OPF problem, allowing for a physics based representation of network constraints.

.. include:: redispatch_modeling.rst
