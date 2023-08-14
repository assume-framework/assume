Units Operator
==============

Assume is created using flexible and usable abstractions, while still providing flexbility to cover most use cases of market modeling.

Main Use-Cases for this are:

- Editing the Market Design
- Changing Input Data
- Comparing different Market Designs according to selected KPIs
- Changing the Bidding behavior
- Evaluating different Reinforcement Methods and Learning Results

The main feature is the flexbility of the Market definitions as well as the bidding behavior, which is split into multiple parts.
In the following the data model of the Units Operator is described in detail.

Data Model
----------

The general Idea of Bidding Agents is to include many Units while each Unit contains the restrictions and base information of its units.
A Units Operator manages the bidding and the general behavior of one or multiple units, and can use Portfolio Optimization (Work-In-Progress) to send an optimized orderbook to the market.
Alternatively, the single bidding options of each unit can be send to the market individually.

For Units, two Interfaces were abstracted for general Units which can charge something (SupportsMinMaxCharge) and for general Units which can provide or consume a given Amount of Power (SupportMinMax).
Both are part of :doc:`assume.common`.

A Unit includes a Forecaster which gives Data on Forecasts, as well as a BiddingStrategy, which can make use of the information given in the unit, as well as its forecasts to create a valid Orderbook for the given market.

This three Classes give flexbility regarding where the data for forecasts, availability or prices comes from (Forecaster), what Operation Parameters are given (Unit) as well as the actual bidding behavior which can either be rule-based or based on Reinforcement Learning (BiddingStrategy).


.. mermaid::

    classDiagram
        Forecaster <-- Unit
        BiddingStrategy <-- Unit
        BiddingStrategy <-- UnitsOperator
        BiddingStrategy <-- UnitsOperator
        Unit <-- UnitsOperator
        SupportsMinMax --|> Unit
        SupportsMinMaxCharge --|> Unit
        Demand --|> SupportsMinMax
        Powerplant --|> SupportsMinMax
        Battery --|> SupportsMinMaxCharge
        HydroStorage --|> SupportsMinMaxCharge
