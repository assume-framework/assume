.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

Market Configurations
=====================

The market configuration allows an extensive configuration of multiple bidding options.
A full list of configurations and explanation of the :meth:`assume.common.market_objects.MarketConfig` is given here.


 ============================= =====================================================
  market config item            description
 ============================= =====================================================
  market_id                     string name
  product_type                  energy or capacity or heat
  market_products               list of available products to be traded
  opening_hours                 recurrence rule of openings
  opening_duration              time delta
  market mechanism              name of method used for clearing
  maximum_bid_price             max allowed bidding price
  minimum_bid_price             min allowed bidding price
  maximum_bid_volume            the largest valid volume for a single bid
  additional_fields             list of additional fields to base bid
  volume_tick                   step increments of volume
  price_tick                    step increments of price
  volume_unit                   string for visualization of volume
  price_unit                    string for visualization of price
  supports_get_unmatched        boolean
  maximum_gradient              max allowed change between bids
  eligible_obligations_lambda   function checking if agent is allowed to trade here
  param_dict                    additional dict with params for grid and other configs
 ============================= =====================================================


Here, a description of all options and the usage is given.

The `opening_hours` is a recurrence rule which defines when the market is open.
The `opening_duration` is a timedelta which defines how long the market is open. This can be used to leave time for negotiation in the agents.
The `market_mechanism` is the name of the clearing method used for this market, and is mapped to the actual clearing_function in the initialization.
The `minimum_bid_price`, `maximum_bid_price` and `maximum_bid_volume` are constraints for bids. All Bids of the sent orderbook are rejected and not considered for clearing if one or more are not in these bounds.
An agent receives a "Rejected" message if this is the case.
The `additional_fields` is a list of additional fields which are used to base the bid on.
The `price_tick` and `volume_tick` is the step increment of volume, which ensures that only integers are used for calculation.
The `price_unit` and `volume_unit` are strings for visualization of price.
The `supports_get_unmatched` is a boolean which defines if the market supports the handle_get_unmatched method, which allows agents to look into the current market orderbook, as it is the case, mostly on continuous markets.
The `maximum_gradient` is the maximum allowed change between bids from one hour to the next one - only relevant if the count of market products is greater than 1.
The `eligible_obligations_lambda` allows to configure additional requirements for agents operating on the market. It can be a string representing one of the preconfigured functions by name (`only_renewables` and `only_co2emissionless`) - or a lambda function checking on the units information.

Most important, the `market_products` are a list of MarketProduct objects.

MarketProduct
-------------

Each :meth:`assume.common.market_objects.MarketProduct` contains the three information:

- duration (a relative timedelta or recurrency rule)
- count (how many consecutive products are available for trading)
- first_delivery (a relative timedelta of the first delivery in relation to market start)
- only_hours (tuple of hours from which this product is available, on multi day products)

For example, if the duration is 1 hour, the count is 4 and first_delivery is 2 hours, then a market which opens at 12:00 will have the following products:

- 14:00 - 15:00
- 15:00 - 16:00
- 16:00 - 17:00
- 17:00 - 18:00

It then makes sense to reschedule the market clearing all 4 hours, but it would also be possible to schedule the same clearing interval an hour later too, which would look like this:

.. mermaid::

  gantt
    title Market Schedule Simple count 4
    dateFormat  YYY-MM-DD HH:mm
    axisFormat %H:%M
    section First
    Bidding00 EOM          :a1, 2019-01-01 12:00, 1h
    Delivery01 EOM         :2019-01-01 14:00, 1h
    Delivery02 EOM         :2019-01-01 15:00, 1h
    Delivery03 EOM         :2019-01-01 16:00, 1h
    Delivery04 EOM         :2019-01-01 17:00, 1h
    section Second
    Bidding10 EOM          :a2, 2019-01-01 13:00, 1h
    Delivery11 EOM         :2019-01-01 15:00, 1h
    Delivery12 EOM         :2019-01-01 16:00, 1h
    Delivery13 EOM         :2019-01-01 17:00, 1h
    Delivery14 EOM         :2019-01-01 18:00, 1h

Please note the following Trade Convention
------------------------------------------

In our market trading system, we follow this convention for representing the volume and price of trades in any market:

- **Volume Representation**:
    - The sign of the volume indicates the direction of the trade:
    - **Positive Volume**: Indicates that the volume is being sold to the market.
    - **Negative Volume**: Indicates that the volume is being bought from the market.

- **Price Representation**:
  - The price of a trade can be negative and positive.
  - The price does not change based on the direction of the trade (Providing energy/power or procurring energy/power) but potentially the financial flow.
  - **Positive Volume and Positive Price**: Indicates that electricity is sold to the market, and money is received for it.
  - **Positive Volume and Negative Price**: Indicates that electricity is sold to the market, but money has to be paid for it.
  - **Negative Volume and Positive Price**: Indicates that electricity is bought from the market, and money is paid for it.
  - **Negative Volume and Negative Price**: Indicates that electricity is bought from the market, and money is received for it.


This convention ensures clarity and consistency in how trades are represented and interpreted within the market. By using positive and negative volumes to indicate the direction of trades, we can easily distinguish between buying and selling activities while maintaining a straightforward and unambiguous pricing structure.

Please note the following limitation
------------------------------------

All currently implemented `bidding_strategies` in ASSUME do not handle feasibility constraints with regard to the dispatch in market mechanisms with multiple products (count > 1).
This means that for markets with multiple products, the bidding strategies do consider the technical feasibility of dispatching units across all products when formulating bids, but not after the market clearing when some bids are rejected.
Therefore, we do not advise to use markets with multiple products in combination with units that require strong time couling such as storages or dispatchable units when `start_up_times` are considered.
This is a known limitation in agent-based modelling and does underestimate the risk of infeasible dispatches for power plant operators.

Example Configuration - CRM Market
----------------------------------

An example of a EOM and CRM market is shown here.
It is possible to trade at the EOM and sell positive capacity on the CRM too::

   markets_config:
    EOM:
      operator: EOM_operator
      product_type: energy
      start_date: 2019-01-01 01:00
      products:
        - duration: 1h
          count: 1
          first_delivery: 1h
      opening_frequency: 1h
      opening_duration: 1h
      market_mechanism: pay_as_clear

    CRM_pos:
      operator: CRM_operator
      product_type: capacity_pos
      start_date: 2019-01-01 00:00
      products:
        - duration: 4h
          count: 1
          first_delivery: 2h
      opening_frequency: 4h
      opening_duration: 30m
      market_mechanism: pay_as_bid

Due to the configuration of the market opening frequency and duration, the timetable for the opening and closing of the markets, as well as the delivery periods are shown below

.. mermaid::

  gantt
    title Market Schedule
    dateFormat  YYY-MM-DD HH:mm
    axisFormat %H:%M
    section EOM
    Bidding01 EOM          :a1, 2019-01-01 01:00, 1h
    Delivery01 EOM         :2019-01-01 01:00, 1h
    Bidding02 EOM          :a2, 2019-01-01 02:00, 1h
    Delivery02 EOM         :2019-01-01 02:00, 1h
    Bidding03 EOM          :a3, 2019-01-01 03:00, 1h
    Delivery03 EOM         :2019-01-01 03:00, 1h
    Bidding04 EOM          :a4, 2019-01-01 04:00, 1h
    Delivery04 EOM         :2019-01-01 04:00, 1h
    section CRM
    Bidding CRM            :crm01, 2019-01-01 00:00, 30m
    Delivery CRM           :crm02, 2019-01-01 01:00, 4h
    Bidding CRM            :crm03, 2019-01-01 04:00, 30m
    Delivery CRM           :crm04, 2019-01-01 05:00, 4h


Example Configuration - Eligible Obligations Lambda
---------------------------------------------------

If not all agents are allowed to bid on a market, one can configure this in the market as well.
For example, because only agents with a given minimum or maximum power are allowed or only agents with renewable generation:

.. code-block:: yaml

    markets_config:
      EOM:
        operator: EOM_operator
        product_type: energy
        start_date: 2019-01-01 01:00
        products:
          duration: 1h
          count: 1
          first_delivery: 1h
        opening_frequency: 1h
        opening_duration: 1h
        market_mechanism: pay_as_clear
        eligible_obligations_lambda: only_renewables

When configuring the market as a Python object, it is also possible to configure a customized lambda function for the market object to reflect to special conditions.

The built-in lambda functions are:

- :py:meth:`assume.common.market_objects.only_renewables`
- :py:meth:`assume.common.market_objects.only_co2emissionless`
- :py:meth:`assume.common.market_objects.power_plant_not_negative`
