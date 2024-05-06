.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

######################
Support Policies
######################

Support Policies are a very important feature when considering different energy market designs.
A support policy allows to influence the cash flow of unit, making decisions more profitable.

One can differentiate between support policies which influence the available market capacity (product_type=`energy``) and those which do not.

If the product_type is `energy`, the volume used for the contract can not be additionally bid on the EOM.

All the support policies are only available when using a Market with the MarketMechanism :meth:`assume.markets.clearing_algorithms.contracts.PayAsBidContractRole`


Example Policies
=====================================


Feed-In-Tariff - FIT
--------------------

To create a Feed-In-Tariff (Einspeisevergütung) one has a contract which sets a fixed price for all produced energy.
The energy can not be additionally sold somewhere else (product_type=`energy`).

The Tariff is contracted at the beginning of the simulation and is valid for X days (1 year).

The payout is executed on a different repetition schedule (monthly).
For this, the output_agent is asked how much energy an agent produced in the timeframe.

This is essentially the same as a Power Purchase Agreement (PPA), except that the payment of FIT is continuous and not monthly or yearly.


Fixed Market Premium - MPFIX
----------------------------

A market premium is paid on top of the market results, based on the results.
As the volume does not influcence the market bidding, the product_type is `financial_support`
So a Market premium is contracted at the beginning of the simulation and is valid for X days (1 year).

The payout is executed on a different repetition schedule (monthly).
For this, the output_agent is asked how much energy an agent produced in the timeframe and what the clearing price of the market with name "EOM" was.
The differences are then calculated and paid out on a monthly base.

This mechanism is also known as One-Sided market premium

Variable Market Premium - MPVAR
-------------------------------

The Idea of the variable market premium is to be based on some kind of market index (like ID3) received from the output agent.


Capacity Premium - CP
---------------------

A capacity premium is paid on a yearly basis for a technology.
This is done in € per installed MW of capacity.
It allows to influence the financial flow of plants which would not be profitable.

Contract for Differences - CfD
------------------------------

A fixed LCoE (Levelized Cost of Energy) is set as a price, if an Agent accepts the CfD contract,
it has to bid at the hourly EOM - the difference of the market result is paid/received to/from the contractor.


Swing Contract
--------------

Actor
^^^^^
