.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

Bidding Strategies
=====================

Overview
-------------

As is described in the `Introduction <https://assume.readthedocs.io/en/latest/introduction.html#exchangeable-bidding-strategy>`_,
a Bidding Strategy dictates the bidding behavior of units in different markets, whereby it maps certain states and technical constraints to bidding decisions.

The ASSUME framework provides multiple options in terms of Bidding Strategy methodologies:

 ============================== =============================================================
  Bidding Strategy Methodology   Description
 ============================== =============================================================
  Naive                          Basic methodology to form bids, based on participating in a market mechanism that follows the Merit Order principle. These strategies do not utilise forecasting
                                 or consider finer details such as the effects of changing a power plant's operational state (start-up costs etc.),
                                 so the bid volume is of the order of its max. power capacity (given ramping constraints) and the price tends to be the marginal cost.
  Standard                       This methodology, based on the flexABLE methodology [`Qussous et al. 2022 <https://doi.org/10.3390/en15020494>`_],
                                 is more refined compared to the Naive strategy. It incorporates market dynamics to a greater degree by forecasting market prices,
                                 as well as accounting for operational history, potential power loss due to heat production,
                                 and the fact that a plant can make bids for multiple markets at the same time.
  DMAS                           TODO
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

When constructing a units CSV file, the bidding strategies are set using :code:`"bidding_"` columns, where the market type the bidding strategy is applied to
follows the underscore (the market names need to match with those in the config file).:

 ======================= ================== ========================= ============================= ============================= ===========
  name                    technology        bidding_EOM                bidding_CRM_pos               bidding_CRM_neg               max_power
 ======================= ================== ========================= ============================= ============================= ===========
  Naive-Bidding Unit      hydro              naive                     naive                         naive                         1000
  Standard-Bidding Unit   hydro              standard_eom_powerplant   standard_pos_crm_powerplant   standard_neg_crm_powerplant   1000
 ======================= ================== ========================= ============================= ============================= ===========

We'll now take a look at the different Bidding Strategy types within each methodology, and their associated "bidding_strategy_id".

Naive
-------------

 ================================================ =============================================================
  bidding_strategy_id                              Description
 ================================================ =============================================================
  naive                                            The basic bidding strategy formulated for participating in a merit order
                                                   market configuration, at any one timepoint (hour). Can be used for placing bids on the EOM, negative CRM or
                                                   positive CRM.

                                                   When used by a powerplant unit, it uses marginal cost for its bid price, and max. possible power
                                                   output (given ramping constraints) as its volume.

                                                   A demand unit can realise a "price-inelastic" demand bid by setting
                                                   the bid price very high and volume equalling demand at the timepoint.
  naive_profile                                    Similar to :code:`naive`, however it is a block bid for 24 hours to
                                                   simulate a bid for the Day-Ahead market, where bid price is set to the marginal cost
                                                   at the starting timepoint.
  naive_profile_dsm                                Strategy for a Demand-Side Management (DSM) unit bid, for a 24-hour period,
                                                   where the bid volume is the unit's optimal power requirement
                                                   at the product's start time, and the bid price is set to a fixed marginal cost (3000 €/MWh).
  naive_exchange                                   This bidding strategy is formulated so as to incorporate cross-border trading into the market mechanism.
                                                   An export and an import bid are made.
                                                   Export bids have negative volumes and are treated as demand
                                                   (with bidding price close to maximum to virtually guarantee acceptance) on the market.
                                                   Import bids have positive volumes and are treated as supply
                                                   (with bidding price close to minimum to virtually guarantee acceptance) on the market.
  naive_redispatch                                 A naive strategy that simply submits all information about the unit and
                                                   currently dispatched power for the following hours to the redispatch market.
                                                   Information includes the marginal cost, the ramp up and down values, and the dispatch.
  naive_redispatch_dsm                             A naive strategy of a Demand Side Management (DSM) unit that bids the available flexibility of
                                                   the unit on the redispatch market.
                                                   The bid volume is the flexible power requirement of the unit at the start time of the product.
                                                   The bid price is the marginal cost of the unit at the start time of the product.
 ================================================ =============================================================

Naive method API references:

- :meth:`assume.strategies.naive_strategies.NaiveSingleBidStrategy`
- :meth:`assume.strategies.naive_strategies.NaiveProfileStrategy`
- :meth:`assume.strategies.naive_strategies.NaiveExchangeStrategy`
- :meth:`assume.strategies.naive_strategies.NaiveRedispatchStrategy`
- :meth:`assume.strategies.naive_strategies.NaiveDADSMStrategy`
- :meth:`assume.strategies.naive_strategies.NaiveRedispatchDSMStrategy`

Standard
-------------

 ================================= =============================================================
  bidding_strategy_id               Description
 ================================= =============================================================
  standard_eom_powerplant           A more refined approach to bidding on the EOM compared to :code:`naive`.
                                    A unit submits both inflexible and flexible bids per hour.
                                    The inflexible bid represents the minimum power output, priced at marginal cost plus startup costs,
                                    while the flexible bid covers additional power up to the maximum capacity at marginal cost.
                                    It incorporates price forecasting and accounts for ramping constraints, operational history,
                                    and power loss due to heat production.
  standard_profile_eom_powerplant   Formulated similarly to :code:`eom_powerplant`, however the bid is for a block of multiple hours
                                    instead of being for a single hour.
                                    A minimum acceptance ratio (MAR) defines how to handle the possibility of rejected bids
                                    within individual hours of the block. For the inflexible bid, the MAR is set to 1,
                                    meaning that all bids within the block must be accepted otherwise the whole block bid is rejected.
                                    A separate MAR can be set for children (flexible) bids.
                                    See the `Advanced Orders tutorial <https://assume.readthedocs.io/en/latest/examples/06_advanced_orders_example.html#1.-Basics>`_
                                    for a more detailed description.
  standard_neg_crm_powerplant       A bid on the negative Capacity or Energy CRM, volume is determined by calculating how much it can reduce power. The capacity price is
                                    found by comparing the revenue it could receive if it bid this volume on the EOM, the energy price is the negative of marginal cost.
  standard_pos_crm_powerplant       A bid on the positive Capacity or Energy CRM, volume is determined by calculating how much it can increase power. The capacity price is
                                    found by comparing the revenue it could receive if it bid this volume on the EOM, the energy price is the marginal cost.
  standard_eom_storage              Determines strategy of Storage unit bidding on the EOM. The unit acts as a generator or load based on average price forecast.
                                    If the current price forecast is greater than the average price, the Storage unit will bid to discharge at a price
                                    equal to the average price divided by the discharge efficiency. Otherwise, it will bid to charge at the average price
                                    multiplied by the charge efficiency. Calculates ramping constraints for charging and discharging based on theoretical state of charge (SOC),
                                    ensuring that power output is feasible. The bid volume is subject to the charge/discharge capacity of the unit.
  standard_neg_crm_storage          Analogous to :code:`standard_eom_storage`, but bids either on the negative capacity CRM or energy CRM.
  standard_pos_crm_storage          Analogous to :code:`standard_eom_storage`, but bids either on the positive capacity CRM or energy CRM.
 ================================= =============================================================

Standard method API references:

- :meth:`assume.strategies.standard_powerplant.StandardEOMPowerplantStrategy`
- :meth:`assume.strategies.standard_advanced_orders.EOMBlockPowerplant`
- :meth:`assume.strategies.standard_advanced_orders.StandardProfileEOMPowerplantStrategy`
- :meth:`assume.strategies.standard_powerplant.StandardNegCRMPowerplantStrategy`
- :meth:`assume.strategies.standard_powerplant.StandardPosCRMPowerplantStrategy`
- :meth:`assume.strategies.standard_storage.StandardEOMStorageStrategy`
- :meth:`assume.strategies.standard_storage.StandardNegCRMStorageStrategy`
- :meth:`assume.strategies.standard_storage.StandardPosCRMStorageStrategy`

DMAS
-------------

 ==================================== =============================================================
  bidding_strategy_id                  Description
 ==================================== =============================================================
  dmas_powerplant                      TODO
  dmas_storage                         TODO
 ==================================== =============================================================

DMAS method API references:

- :meth:`assume.strategies.dmas_powerplant.DmasPowerplantStrategy`
- :meth:`assume.strategies.dmas_storage.DmasStorageStrategy`

Learning
-------------

 ================================= =============================================================
  bidding_strategy_id               Description
 ================================= =============================================================
  learning_eom_powerplant           A `reinforcement learning <https://assume.readthedocs.io/en/latest/learning_algorithm.html#td3-twin-delayed-ddpg>`_ (RL) approach to formulating bids for a
                                    Power Plant in an Energy-Only Market. The agent's actions are
                                    two bid prices: one for the inflexible component (P_min) and another for the flexible component (P_max - P_min) of a unit's capacity.
                                    The bids are informed by 50 observations, which include forecasted residual load, forecasted price, total capacity, and marginal cost,
                                    all contributing to decision-making. Noise is added to the action, especially towards the beginning of the learning, to encourage exploration and novelty.

                                    The reward is calculated based on profits from executed bids, operational costs, opportunity costs (penalizing underutilized capacity),
                                    and a regret term to minimize missed revenue opportunities. This approach encourages full utilization of the unit's capacity.
  learning_eom_storage              Similar RL approach as :code:`learning_eom_powerplant`, for a Storage unit. The make-up of the observations is similar to those for
                                    :code:`learning_eom_powerplant`, with an additional observation being the State-of-Charge (SOC) of the storage unit. The agent has 2 actions -
                                    a bid price, and a bid direction (to buy, sell or do nothing). The bid volume is subject to the charge/discharge capacity of the unit.

                                    The reward is calculated based on profits from executed bids, with fixed costs for charging/discharging incorporated.
  learning_profile_eom_powerplant   An RL strategy for bidding in an EOM using different order types (simple hourly, block, and linked orders).
                                    Based on :code:`standard_profile_eom_powerplant`, however uses the trained actor network (as with the other RL bidding strategies)
                                    to determine bid prices instead of relying on marginal costs. Once again there are two bid prices, a lower price for inflexible component,
                                    and a higher price for flexible compoenent.

                                    Order types are set implicitly, not by the RL agent itself, and the bid structure
                                    based on allowed order types (SB - Simple (Hourly) Bid, BB - Block Bid, LB - Linked Bid):

                                    - SB only: Both power types use SB.
                                    - SB & LB: Inflexible uses SB, flexible uses LB.
                                    - SB & BB: Inflexible uses BB, flexible uses SB.
                                    - SB, BB & LB: Inflexible uses BB, flexible uses LB (or SB if inflexible power is 0, like VREs).

 ================================= =============================================================

Learning method API references:

- :meth:`assume.strategies.learning_strategies.LearningEOMPowerplantStrategy`
- :meth:`assume.strategies.learning_strategies.LearningEOMStorageStrategy`

Other
-------------

 ======================== =============================================================
  bidding_strategy_id      Description
 ======================== =============================================================
  misc_otc                 Similar to `naive`, however it is bid on the OTC market, representing bilateral trades.
  misc_manual              The bidding volume and price is manually entered.
 ======================== =============================================================

Miscellaneous method API references:

- :meth:`assume.strategies.extended.OTCStrategy`
- :meth:`assume.strategies.manual_strategies.SimpleManualTerminalStrategy`
