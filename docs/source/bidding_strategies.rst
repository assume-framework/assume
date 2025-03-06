.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

Bidding Strategies
=====================

Overview
-------------

The ASSUME framework provides multiple options in terms of Bidding Strategy methodologies:

 ============================= =============================================================
  Bidding Strategy Category     Description
 ============================= =============================================================
  Naive                         Basic form of bids, based on participating in market mechanism that follows the Merit Order principle.
                                External market factors or forecasts are not accounted for, so the bid volume is of the order of its max. power capacity (given ramping constraints)
                                and the price is its marginal cost.
  Standard                      These bidding strategies, based on the flexABLE methodology [`Qussous et al. 2022 <https://doi.org/10.3390/en15020494>`_],
                                are more refined compared to the Naive strategy. They incorporate market dynamics to a greater degree by forecasting market prices,
                                operational history, potential power loss due to heat production,
                                and taking into account the fact that a plant can make bids for multiple markets at the same time.
  DMAS                          TODO
  Learning                      A reinforcement learning (RL) approach designed to optimize bidding strategies for an Energy-Only Market. The agent's actions are two bid prices: one for
                                the inflexible component (P_min) and another for the flexible component (P_max - P_min) of a unit's capacity.
                                The bids are informed by 50 observations, which include forecasted residual load, forecasted price, total capacity,
                                and marginal cost, all contributing to decision-making. Noise can be added to the action to encourage exploration and novelty.

                                The reward system includes profits from accepted bids, operational costs, opportunity costs (penalizing underutilized capacity),
                                and a regret term to minimize missed revenue opportunities. This approach encourages full utilization of the unit's capacity.
  Other                         Other bidding methods not belonging to a specific category.
 ============================= =============================================================

For each Bidding Strategy Category, there are different Bidding Strategy Types based on the unit type (Power Plant, Storage, Demand-Side Management (DSM)) that is making the bid,
and the Market Type the bid is made for (Energy-Only Market (EOM), Positive/Negative Control Reserve Market (PCRM/NCRM), Redispatch Market). The following sections describe these
Bidding Strategy Types per category.

Naive
-------------

 ================================================= ================================================ =============================================================
  Bidding Strategy Type                             Bidding Strategy ID                              Description
 ================================================= ================================================ =============================================================
  Naive Energy-Only Market Power Plant Bid          naive_eom_powerplant                             This bid is the generic formulation of a bid in a merit order
                                                                                                     market configuration at any one timepoint (hour) for the EOM,
                                                                                                     where it uses marginal cost for its bid price, and max. power
                                                                                                     output (given ramping constraints) as its volume.
  Naive Energy-Only Market Demand Bid               naive_eom_demand                                 The demand bid uses the same strategy as :code:`naive_eom_powerplant`, it is
                                                                                                     realised as a price-inelastic demand bid by setting
                                                                                                     the bid price very high, and the volume is negative as it is consuming rather than producing power.
  Naive Day-Ahead Market Bid                        naive_eom_block_powerplant                       Similar to :code:`naive_eom_powerplant`, however it is a block bid for 24 hours to
                                                                                                     simulate a bid for the Day-Ahead market.
  Naive Positive Control Reserve Market Bid         naive_pcrm                                       This bid uses the same strategy as :code:`naive_eom_powerplant`,
                                                                                                     however the bid is placed in the positive CRM.
  Naive Negative Control Reserve Market Bid         naive_ncrm                                       This bid uses the same strategy as :code:`naive_eom_powerplant`,
                                                                                                     however the bid is placed in the negative CRM.
  Naive Redispatch Market Bid                       naive_redispatch                                 This bid uses the same strategy as :code:`naive_eom_powerplant`,
                                                                                                     however the bid is placed in the Redispatch market.
  Naive Day-Ahead DSM Market Bid                    naive_eom_block_dsm                              Demand Side Management (DSM) unit bid, for 24-hour period on the EOM,
                                                                                                     where the bid volume is the unit's optimal power requirement
                                                                                                     at the product's start time, and the bid price is set to a fixed marginal cost (3000 €/MWh).
  Naive Redispatch DSM Market Bid                   naive_redispatch_dsm                             DSM unit bid of its available flexibility, for 24-hour period on the Redispatch market,
                                                                                                     where the bid volume is the unit's flexible power requirement
                                                                                                     at the product's start time, and the bid price is set to its
                                                                                                     marginal cost at the product's start time.
 ================================================= ================================================ =============================================================

Naive method descriptions:

- :meth:`assume.strategies.naive_strategies.NaiveSingleBidStrategy`
- :meth:`assume.strategies.naive_strategies.NaiveProfileStrategy`
- :meth:`assume.strategies.naive_strategies.NaiveRedispatchStrategy`
- :meth:`assume.strategies.naive_strategies.NaiveDADSMStrategy`
- :meth:`assume.strategies.naive_strategies.NaiveRedispatchDSMStrategy`

Standard
-------------

 ================================================= ========================== =============================================================
  Bidding Strategy Type                             Bidding Strategy ID        Description
 ================================================= ========================== =============================================================
  Energy-Only Market Power Plant Bid                eom_powerplant             A more refined approach to bidding on the EOM compared to :code:`naive_eom_powerplant`.
                                                                               A unit submits both inflexible and flexible bids per hour.
                                                                               The inflexible bid represents the minimum power output, priced at marginal cost plus startup costs,
                                                                               while the flexible bid covers additional power up to the maximum capacity at marginal cost.
                                                                               It incorporates price forecasting and accounts for ramping constraints, operational history,
                                                                               and power loss due to heat production.
  Energy-Only Market Power Plant Block Bid          eom_block_powerplant       Formulated similarly to :code:`eom_powerplant`, however it is a block bid for multiple hours.
                                                                               A minimum acceptance ratio (MAR) defines how to handle the possibility of rejected bids
                                                                               within individual hours of the block.
                                                                               It set to 1, meaning that all bids within the block must be accepted otherwise the whole block bid is rejected.
                                                                               See the (`Advanced Orders tutorial <https://assume.readthedocs.io/en/latest/examples/06_advanced_orders_example.html#1.-Basics>`_)
                                                                               for a more detailed description.
  Energy-Only Market Linked Bid                     eom_linked_powerplant      Similar to :code:`eom_block_powerplant`, however the MAR for children (flexible) bids can be less than that of the parent (inflexible) bids.
  Negative Control Reserve Market Bid               ncrm_powerplant            A bid on the negative Capacity or Energy CRM, volume is determined by calculating how much it can reduce power. The capacity price is
                                                                               found by comparing the revenue it could receive if it bid this volume on the EOM, the energy price is the negative of marginal cost.
  Positive Control Reserve Market Bid               pcrm_powerplant            A bid on the positive Capacity or Energy CRM, volume is determined by calculating how much it can increase power. The capacity price is
                                                                               found by comparing the revenue it could receive if it bid this volume on the EOM, the energy price is the marginal cost.
  Energy-Only Market Storage Bid                    eom_storage                Determines strategy of Storage unit bidding on the EOM. The unit acts as generator or load based on average price forecast.
                                                                               If the current price forecast is greater than the average price, the Storage unit will bid to discharge at a price
                                                                               equal to the average price divided by the discharge efficiency. Otherwise, it will bid to charge at the average price
                                                                               multiplied by the charge efficiency. Calculates ramping constraints for charging and discharging based on theoretical state of charge (SOC),
                                                                               ensuring that power output is feasible.
  Negative Control Reserve Market Storage Bid       ncrm_storage               Analogous to :code:`eom_storage`, but bids either on the negative capacity CRM or energy CRM.
  Positive Control Reserve Market Storage Bid       pcrm_storage               Analogous to :code:`eom_storage`, but bids either on the positive capacity CRM or energy CRM.
 ================================================= ========================== =============================================================

Standard method descriptions:

- :meth:`assume.strategies.standard_powerplant.EOMPowerplant`
- :meth:`assume.strategies.standard_advanced_orders.EOMBlockPowerplant`
- :meth:`assume.strategies.standard_advanced_orders.EOMLinkedPowerplant`
- :meth:`assume.strategies.standard_powerplant.NCRMPowerplant`
- :meth:`assume.strategies.standard_powerplant.PCRMPowerplant`
- :meth:`assume.strategies.standard_storage.EOMStorage`
- :meth:`assume.strategies.standard_storage.NCRMStorage`
- :meth:`assume.strategies.standard_storage.PCRMStorage`

DMAS
-------------

 ================================================= ================================================ =============================================================
  Bidding Strategy Type                             Bidding Strategy ID                              Description
 ================================================= ================================================ =============================================================
  DMAS Powerplant Bid                               dmas_powerplant                                  TODO
  DMAS Storage Bid                                  dmas_storage                                     TODO
 ================================================= ================================================ =============================================================

DMAS method descriptions:

- :meth:`assume.strategies.dmas_powerplant.DmasPowerplantStrategy`
- :meth:`assume.strategies.dmas_storage.DmasStorageStrategy`

Learning
-------------

 ================================================= ========================== =============================================================
  Bidding Strategy Type                             Bidding Strategy ID        Description
 ================================================= ========================== =============================================================
  Reinforcement Learning Powerplant Bid             learning_powerplant        A reinforcement learning (RL) approach designed to optimize bidding strategies for an Energy-Only Market. The agent's actions are
                                                                               two bid prices: one for the inflexible component (P_min) and another for the flexible component (P_max - P_min) of a unit's capacity.
                                                                               The bids are informed by 50 observations, which include forecasted residual load, forecasted price, total capacity, and marginal cost,
                                                                               all contributing to decision-making. Noise can be added to the action to encourage exploration and novelty.

                                                                               The reward system includes profits from accepted bids, operational costs, opportunity costs (penalizing underutilized capacity),
                                                                               and a regret term to minimize missed revenue opportunities. This approach encourages full utilization of the unit's capacity.
  Reinforcement Learning Storage Bid                learning_storage           Similar to `learning_powerplant`, taking into account parameters of a Storage unit such as State-of-Charge (SOC).
 ================================================= ========================== =============================================================

Learning method descriptions:

- :meth:`assume.strategies.learning_strategies.RLStrategy`
- :meth:`assume.strategies.learning_strategies.StorageRLStrategy`

Other
-------------

 ================================================= ======================== =============================================================
  Bidding Strategy Type                             Bidding Strategy ID      Description
 ================================================= ======================== =============================================================
  Naive Exchange (Import/Export) Bid                naive_exchange           This bidding strategy is forumlated so as to incorporate cross-border trading into the market mechanism.
                                                                             An export and an import bid are made.
                                                                             Export bids have negative volumes and are treated as demand
                                                                             (with bidding price close to maximum to virtually guarantee acceptance) on the market.
                                                                             Import bids have positive volumes and are treated as supply
                                                                             (with bidding price close to minimum to virtually guarantee acceptance) on the market.
  Over the Counter Market Bid                       otc_strategy             Similar to `naive_eom_powerplant`, however it is bid on the OTC market, representing bilateral trades.
  Manual Bid                                        manual_strategy          The bidding volume and price is manually entered.
 ================================================= ======================== =============================================================

Other method descriptions:

- :meth:`assume.strategies.naive_strategies.NaiveExchangeStrategy`
- :meth:`assume.strategies.extended.OTCStrategy`
- :meth:`assume.strategies.manual_strategies.SimpleManualTerminalStrategy`
