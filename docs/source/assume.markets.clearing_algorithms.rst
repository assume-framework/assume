.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: MIT

Market Clearing Algorithms
==========================

Overview
--------

This document provides an overview of various market clearing algorithms used in the ASSUME framework.

Simple Market Clearing Algorithms
---------------------------------

Merit-Order Kernel
^^^^^^^^^^^^^^^^^^
Both simple clearing algorithms and the naive price forecast use this function, which clears a single product on numpy arrays of bid prices and volumes. It can also be used directly.

.. autofunction:: assume.markets.clearing_algorithms.simple.clear_merit_order

Pay-as-Bid
^^^^^^^^^^
This class implements the pay-as-bid mechanism. The accepted volumes are determined with :func:`~assume.markets.clearing_algorithms.simple.clear_merit_order`, just like for pay-as-clear. Accepted supply orders receive their own bid price, accepted demand orders pay the prices of the supply orders they are matched with in the merit order, i.e. the volume-weighted average of these supply prices.

.. autoclass:: assume.markets.clearing_algorithms.simple.PayAsBidRole
   :members:
   :undoc-members:

Pay-as-Clear
^^^^^^^^^^^^
This class implements the pay-as-clear mechanism with :func:`~assume.markets.clearing_algorithms.simple.clear_merit_order`, which sorts the supply bids ascending and the demand bids descending by price and then matches them one by one until the market is cleared. In general, the last accepted supply bid sets the market clearing price. Only if the first demand bid which is not fully served would pay more than that, e.g. during supply scarcity, this demand bid sets the price.

.. autoclass:: assume.markets.clearing_algorithms.simple.PayAsClearRole
   :members:
   :undoc-members:

Complex Clearing Algorithm
--------------------------
This class implements a market clearing algorithm relying on optimization.

.. automodule:: assume.markets.clearing_algorithms.complex_clearing
   :members:
   :undoc-members:
   :show-inheritance:

Complex Clearing from DMAS model
--------------------------------

.. automodule:: assume.markets.clearing_algorithms.complex_clearing_dmas
   :members:
   :undoc-members:
   :show-inheritance:

Redispatch Clearing Algorithm
------------------------------

.. automodule:: assume.markets.clearing_algorithms.redispatch
   :members:
   :undoc-members:
   :show-inheritance:

Nodal Clearing Algorithm
------------------------
.. automodule:: assume.markets.clearing_algorithms.nodal_clearing
   :members:
   :undoc-members:
   :show-inheritance:

Additional Details
------------------

For more details on the market clearing algorithms, refer to the module contents.

Module Contents
---------------

.. automodule:: assume.markets.clearing_algorithms
   :members:
   :undoc-members:
   :show-inheritance:
