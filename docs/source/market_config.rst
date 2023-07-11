Market Configurations
=====================

The market configuration allows an extensive configuration of multiple bidding options.
A full list of configurations and explanation is given here.


 ============================= =====================================================
  market config item            description
 ============================= =====================================================
  name                          string name
  product type                  energy or capacity or heat
  market products               list of available products to be traded
  opening hours                 recurrence rule of openings
  opening duration              time delta
  market mechanism              name of method used for clearing
  maximum bid                   max allowed bidding price
  minimum bid                   min allowed bidding price
  maximum volume                the largest valid volume for a single bid
  additional fields             list of additional fields to base bid
  volume tick size              step increments of volume
  price tick size               step increments of price
  volume unit                   string for visualization
  price unit                    string for visualization
  supports get unmatched        boolean
  maximum gradient              max allowed change between bids
  eligible obligations lambda   function checking if agent is allowed to trade here
 ============================= =====================================================


Here, a description of all options and the usage will be given.


Example Configuration - CRM Market
==================================

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
