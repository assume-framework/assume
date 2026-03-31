.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

Market Clearing Algorithms
==========================

Overview
--------

This document provides an overview of various market clearing algorithms used in the ASSUME framework.

Simple Market Clearing Algorithms
---------------------------------

Pay-as-Bid
^^^^^^^^^^
This class implements the pay-as-bid mechanism in a simple way using iterative clearing. It is a simple clearing algorithm that clears the market by sorting the bids and offers in ascending order and then matching them one by one until the market is cleared.

.. autoclass:: assume.markets.clearing_algorithms.simple.PayAsBidRole
   :members:
   :undoc-members:

Pay-as-Clear
^^^^^^^^^^^^
This class implements the pay-as-clear mechanism within the merit order clearing algorithm. It is a simple clearing algorithm that clears the market by sorting the bids and offers in ascending order and then matching them one by one until the market is cleared.

.. autoclass:: assume.markets.clearing_algorithms.simple.PayAsClearRole
   :members:
   :undoc-members:

Complex Clearing Algorithm
--------------------------

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
