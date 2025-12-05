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


Portfolio Optimization
----------------------

The main felxibility a unit oporator is, that we can process all the bids and technical constraints a the unit oporator gets from its technical units
however we want, before sending them to the market. This allows us to implement a portfolio optimization for the technical units assigned to the unit operator.
A respective function is in place for that. Yet, it is not used in the current version of the unit operator. The function is called :func:`assume.common.units_operator.UnitsOperator.formulate_bids`.
For example, one could think of coordinating the bids of a battery and a PV unit to maximise the self-consumption of the PV unit under one unit operator.
