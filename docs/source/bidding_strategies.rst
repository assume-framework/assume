.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

Bidding Strategies
==================

Bidding strategies are a core concept of ASSUME which describe how agents bid their assets on a market.

Overview
---------

As described in the :ref:`exchangeable_bidding_strategy`, a Bidding Strategy dictates the bidding behavior of units in different markets, whereby it maps certain states and technical constraints to bidding decisions.
In general, there is a distinction between two kinds of strategy classes:

- :py:meth:`assume.strategies.portfolio_strategies.UnitOperatorStrategy`, which indicate the use in a UnitsOperator and can be used to provide a Portfolio optimization.
- Strategies used for a single unit of :doc:`units`, which range from naive (e.g. :py:meth:'assume.strategies.naive_strategies.EnergyNaiveStrategy')to advanced (e.g. :py:meth:'assume.strategies.advanced_orders.EnergyHeuristicFlexableLinkedStrategy').

Both types have a function `calculate_bids` which is called with the information of the market and bids to bid on.

UnitOperatorStrategy
--------------------

The UnitsOperatorStrategies can be used to adjust the behavior of the UnitsOperator.
The default is the :py:meth:`assume.strategies.portfolio_strategies.UnitsOperatorDirectStrategy`.
It formulates the bids to the market according to the bidding strategy of the each unit individually.
This calls `calculate_bids`` of each unit and returns the aggregated list of all individual bids of all units.
This is the default for all UnitsOperators which do not have a separate strategy configured.

Another implementation includes the :py:meth:`assume.strategies.portfolio_strategies.UnitsOperatorEnergyHeuristicCournotStrategy` that adds a markup to the marginal cost of each unit of the units operator.
The marginal cost is computed with EnergyNaiveStrategy and the markup depends on the total capacity of the unit operator.

UnitStrategies
--------------

The ASSUME framework provides multiple options in terms of Bidding Strategy methodologies:

==============================  =============================================================
Bidding Strategy Methodology    Description
==============================  =============================================================
Naive                           Simple bidding strategies for participating in a market mechanism that follows the Merit Order principle. No additional dependencies needed.
Heuristic                       Basic methodology to form bids, based on participating in a market mechanism that follows the Merit Order principle. These strategies do not utilise
                                forecasting or consider finer details such as the effects of changing a power plant's operational state (start-up costs etc.), so the bid volume is
                                of the order of its maximum power capacity (given ramping constraints), and the bid price is set to the marginal cost.
Optimization                    Methodology, based on the flexABLE methodology [`Qussous et al. 2022 <https://doi.org/10.3390/en15020494>`_], offers more refined strategising
                                compared to the Naive methods. Applicable to power plant and storage units, it incorporates market dynamics to a greater degree by forecasting market
                                prices, as well as accounting for operational history, potential power loss due to heat production, and the fact that a plant can make bids for
                                multiple markets at the same time.
Learning                        A :doc:`reinforcement learning <learning>` (RL) approach to formulating bids for an Energy-Only Market.
                                Agents perform actions (choose bid price(s) and for storage a direction) informed by observations (including forecasted residual load, forecasted
                                price, marginal cost). Bid volumes are fixed to the maximum possible volume. Based on the reward (profits) from accepted bids, agents learn to optimise
                                bids to maximise profits. This requires to have PyTorch installed.
Interactive                     Strategies which let a user handle input through a terminal or other interface.
==============================  =============================================================


For each Bidding Strategy methodology there are multiple Bidding Strategy options depending on the product and market that the bid is intended for,
as well as the type of unit making the bid.

Accordingly, each Bidding Strategy has an associated ID which takes the form "unit_product_method_comment", the comment being optional.
This "bidding_strategy_id" needs to be entered when defining a unit's bidding strategy. Each Bidding Strategy and associated ID for each methodology is defined and described further below.

When constructing a units CSV file, the bidding strategies are set using :code:`"bidding_*"` columns, where the market type the bidding strategy is applied to
follows the underscore (the market names need to match with those in the config file).

======================  ===========  ==================================  ============================================  ============================================  ===========
name                    technology   bidding_EOM                         bidding_CRM_pos                               bidding_CRM_neg                               max_power
======================  ===========  ==================================  ============================================  ============================================  ===========
Naive-Bidding Unit      hydro        powerplant_energy_naive             powerplant_capacity_heuristic_balancing_pos   powerplant_capacity_heuristic_balancing_neg   1000
Advanced-Bidding Unit   hydro        powerplant_energy_heuristic_block   powerplant_capacity_heuristic_balancing_pos   powerplant_capacity_heuristic_balancing_neg   1000
======================  ===========  ==================================  ============================================  ============================================  ===========

We'll now take a look at the different Bidding Strategies within each methodology, their associated "bidding_strategy_id", and for which unit and market type(s) they are valid.

Naive
-----

==================================  =======================  ==================================================================================================================================
bidding_strategy_id                 For Market Types         Description
==================================  =======================  ==================================================================================================================================
powerplant_energy_naive             EOM, CRM_pos, CRM_neg    Basic strategy for merit order markets at a single timepoint (hour). Uses marginal cost as bid price and the maximum feasible
                                                             power (respecting ramping constraints) as bid volume.
demand_energy_naive                 EOM, CRM_pos, CRM_neg    Basic naive strategy for demand units. Can realise a price-inelastic demand by setting a very high bid price and volume equal
                                                             to demand at the timepoint.
powerplant_energy_naive_otc         OTC                      Similar to powerplant_energy_naive but for OTC (bilateral) trades.
demand_energy_naive_otc             OTC                      Similar to demand_energy_naive but for OTC (bilateral) trades.
powerplant_energy_naive_profile     EOM, CRM_pos, CRM_neg    Similar to powerplant_energy_naive but submitted as a 24-hour block (Day-Ahead). Bid price is set to the marginal cost at the
                                                             starting timepoint.
powerplant_energy_naive_redispatch  redispatch               Submits unit info and currently dispatched power for upcoming hours to the redispatch market (includes marginal cost, ramping,
                                                             and dispatch information).
demand_energy_naive_redispatch      redispatch               Submits unit info and currently dispatched power for upcoming hours to the redispatch market (includes marginal cost, ramping,
                                                             and dispatch information).
household_energy_naive_redispatch   redispatch               Naive DSM strategy for industry/household units; bids available flexibility on redispatch. Volume equals flexible power at product
                                                             start; price equals marginal cost at product start.
industry_energy_naive_redispatch    redispatch               Naive DSM strategy for industry/household units; bids available flexibility on redispatch. Volume equals flexible power at product
                                                             start; price equals marginal cost at product start.
exchange_energy_naive               EOM                      Incorporates cross-border trading: submits export (negative volume) and import bids. Exports are treated as demand with very high
                                                             bid price; imports as supply with a very low bid price to virtually guarantee acceptance.
==================================  =======================  ==================================================================================================================================

Naive method API references:

- :py:meth:`assume.strategies.naive_strategies.EnergyNaiveStrategy`
- :py:meth:`assume.strategies.extended.EnergyNaiveOtcStrategy`
- :py:meth:`assume.strategies.naive_strategies.EnergyNaiveProfileStrategy`
- :py:meth:`assume.strategies.naive_strategies.EnergyNaiveRedispatchStrategy`
- :py:meth:`assume.strategies.naive_strategies.DsmEnergyNaiveRedispatchStrategy`
- :py:meth:`assume.strategies.naive_strategies.ExchangeEnergyNaiveStrategy`

Heuristic
---------

============================================  ==================  ==================================================================================================================
bidding_strategy_id                           For Market Types    Description
============================================  ==================  ==================================================================================================================
demand_energy_heuristic_elastic               EOM                 This bidding strategy is formulated for a demand unit to realise a "price-elastic" demand bid, approximating a
                                                                  marginal utility curve. The elasticity can be set to "linear" or "isoelastic".
powerplant_energy_heuristic_flexable          EOM                 A more refined approach to bidding on the EOM compared to naive. A unit submits both inflexible and flexible
                                                                  bids per hour. The inflexible bid represents the minimum power output, priced at marginal cost plus startup
                                                                  costs, while the flexible bid covers additional power up to the maximum capacity at marginal cost. It
                                                                  incorporates price forecasting and accounts for ramping constraints, operational history, and power loss
                                                                  due to heat production.
powerplant_energy_heuristic_block             EOM                 A power plant strategy valid for complex market clearing which bids dict blocks to the market. The bid is for a
                                                                  block of multiple hours instead of being for a single hour. A minimum acceptance ratio (MAR) defines how to
                                                                  handle rejected bids within individual hours of the block. For the inflexible bid, the MAR is set to 1,
                                                                  meaning all bids within the block must be accepted otherwise the whole block bid is rejected. A separate MAR
                                                                  can be set for children (flexible) bids. See the Advanced Orders tutorial:
                                                                  https://assume.readthedocs.io/en/latest/examples/06_advanced_orders_example.html#1.-Basics
powerplant_energy_heuristic_linked            EOM                 A power plant strategy which handles block and linked bids on a market with these fields as a dict. The strategy
                                                                  is similar to :code:`powerplant_energy_heuristic_block` but allows to integrate the flexible bids as linked bids.
                                                                  See the Advanced Orders tutorial:
                                                                  https://assume.readthedocs.io/en/latest/examples/06_advanced_orders_example.html#1.-Basics
powerplant_capacity_heuristic_balancing_neg   CRM_neg             A bid on the negative Capacity or Energy Control Reserve Market (CRM); volume is determined by calculating how
                                                                  much it can reduce power. The capacity price is found by comparing the revenue it could receive if it bid this
                                                                  volume on the EOM; the energy price is the negative of marginal cost.
powerplant_capacity_heuristic_balancing_pos   CRM_pos             A bid on the positive Capacity or Energy CRM; volume is determined by calculating how much it can increase
                                                                  power. The capacity price is found by comparing the revenue it could receive if it bid this volume on
                                                                  the EOM; the energy price is the positive of marginal cost.
storage_energy_heuristic_flexable             EOM                 Determines strategy of a Storage unit bidding on the EOM. The unit acts as a generator or load based on the
                                                                  average price forecast. If the current price forecast is greater than the average price, the Storage unit
                                                                  will bid to discharge at a price equal to the average price divided by the discharge efficiency. Otherwise,
                                                                  it will bid to charge at the average price multiplied by the charge efficiency. Calculates ramping
                                                                  constraints for charging and discharging based on theoretical state of charge (SOC), ensuring that power
                                                                  output is feasible. The bid volume is subject to the charge/discharge capacity of the unit.
storage_capacity_heuristic_balancing_neg      CRM_neg             Analogous to :code:`storage_energy_heuristic_flexable`, but bids either on the negative capacity CRM or
                                                                  energy CRM.
storage_capacity_heuristic_balancing_pos      CRM_pos             Analogous to :code:`storage_energy_heuristic_flexable`, but bids either on the positive capacity CRM or
                                                                  energy CRM.
============================================  ==================  ==================================================================================================================

Heuristic method API references:

- :py:meth:`assume.strategies.naive_strategies.EnergyHeuristicElasticStrategy`

- :py:meth:`assume.strategies.advanced_orders.EnergyHeuristicFlexableBlockStrategy`
- :py:meth:`assume.strategies.advanced_orders.EnergyHeuristicFlexableLinkedStrategy`
- :py:meth:`assume.strategies.flexable.CapacityHeuristicBalancingNegStrategy`
- :py:meth:`assume.strategies.flexable.CapacityHeuristicBalancingPosStrategy`

- :py:meth:`assume.strategies.flexable_storage.StorageEnergyHeuristicFlexableStrategy`
- :py:meth:`assume.strategies.flexable_storage.StorageCapacityHeuristicBalancingNegStrategy`
- :py:meth:`assume.strategies.flexable_storage.StorageCapacityHeuristicBalancingPosStrategy`

Optimization
------------

===========================================  ==================  ============
bidding_strategy_id                          For Market Types    Description
===========================================  ==================  ============
household_energy_optimization                EOM                 An energy strategy of a Household DSM unit. The bid volume is the optimal power requirement of the optimization.
industry_energy_optimization                 EOM                 An energy strategy of a Industry DSM unit. The bid volume is the optimal power requirement of the optimization.
industrial_hybrid_eom                         EOM                 Rule-based EOM strategy for rolling-horizon industrial hybrid plants. It supports compatible cement routes and a steel
                                                                  electrolyser plus hydrogen-buffer route. Gas-hybrid cement uses natural-gas and CO2 forecasts; fully-electric cement
                                                                  with E-TES bids its electricity forecast; steel requires ``electricity_wtp``. After clearing, deterministic rules apply.
industrial_hybrid_capacity_neg                CRM_neg             Capacity-only negative CRM strategy for the supported cement routes and steel electrolyser-buffer route. It bids firm block
                                                                  capacity and reservation opportunity cost; accepted awards preserve available electric capacity and relevant storage space
                                                                  in one pre-EOM rolling solve. Fully-electric E-TES prices reservation from constrained forecast schedule cost. No activation energy is modelled.
industrial_hybrid_otc                         LTM_OTC             Physical long-term electricity-procurement strategy. It bids a firm auxiliary-security tranche for compatible cement routes
                                                                  and the supported steel route, plus a firm process tranche for direct electric-plus-natural-gas cement. Accepted pay-as-bid
                                                                  contracts reduce the overlapping EOM residual demand; E-TES routes bid auxiliary security only.
household_capacity_heuristic_balancing_neg   CRM_neg             A negative capacity strategy of a Household DSM unit. The bid volume is the optimal power requirement of the optimization.
household_capacity_heuristic_balancing_pos   CRM_pos             A positive capacity strategy of a Industry DSM unit. The bid volume is the optimal power requirement of the optimization.
industry_capacity_heuristic_balancing_neg    CRM_neg             A negative capacity strategy of a Household DSM unit. The bid volume is the optimal power requirement of the optimization.
industry_capacity_heuristic_balancing_pos    CRM_pos             A positive capacity strategy of a Industry DSM unit. The bid volume is the optimal power requirement of the optimization.
powerplant_energy_optimization_dmas          EOM                 Power plant strategy using forecast optimization and avoided cost calculation used for smart bids coming from the DMAS methodology
storage_energy_optimization_dmas             EOM                 Storage strategy using forecasts and avoided cost calculation used for smart bids coming from the DMAS methodology
===========================================  ==================  ============

Optimization method API references:

- :py:meth:`assume.strategies.naive_strategies.DsmEnergyOptimizationStrategy`
- :py:meth:`assume.strategies.industrial_hybrid.IndustrialHybridEomStrategy`
- :py:meth:`assume.strategies.industrial_hybrid.IndustrialHybridCapacityNegStrategy`
- :py:meth:`assume.strategies.industrial_hybrid.IndustrialHybridOtcStrategy`
- :py:meth:`assume.strategies.naive_strategies.DsmCapacityHeuristicBalancingStrategy`
- :py:meth:`assume.strategies.dmas_powerplant.EnergyOptimizationDmasStrategy`
- :py:meth:`assume.strategies.dmas_storage.StorageEnergyOptimizationDmasStrategy`

Industrial hybrid steel route
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first steel route is deliberately narrow: a hydrogen-fuelled DRI plant,
an electrolyser, and ``hydrogen_buffer_storage``. It offers negative CRM
capacity as additional electrolyser electricity that can be stored as hydrogen
for the full capacity block. The offer is limited jointly by electrolyser power
headroom and hydrogen-buffer space. Its reservation opportunity cost is the
forecast external hydrogen value, converted through electrolyser and buffer
efficiencies, minus the forecast EOM electricity price.

For the steel EOM strategy configure ``electricity_wtp`` in
``bidding_strategy_params``. It is the scenario's value of electricity for the
planned steel-production schedule; ASSUME's current steel model does not expose
a single technology-independent marginal production value from which this could
be derived. If an EOM bid is rejected, this V1 rule records the corresponding
``unserved_steel``; it does not invent a gas, hydrogen-import, or re-optimised
fallback.

``industrial_hybrid_otc`` supports the steel route's firm DRI auxiliary
electricity. The DRI component consumes this electricity alongside hydrogen,
so its firm volume is the lowest hourly ``steel_demand ×
specific_dri_demand × specific_electricity_consumption`` in the delivery block.
It requires a complete per-timestep ``steel_demand`` forecast. The strategy
does not yet bid an electrolyser-process tranche because that additional power
would create hydrogen that may exceed buffer capacity over a long product.

Industrial hybrid OTC procurement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``industrial_hybrid_otc`` is a physical demand strategy for the existing
``LTM_OTC`` pay-as-bid energy market. Configure
``auxiliary_security_value`` in ``bidding_strategy_params``. It is the demand
bid ceiling for the firm auxiliary-electricity tranche, capped by the market's
configured maximum price. It is a scenario value for guaranteed auxiliary
supply, not the internal clinker-shortfall penalty. The market assigns
``accepted_price`` as the contract strike; the strategy submits no strike
price.

For every long-term product the strategy uses the lowest forecast auxiliary
load throughout the delivery block. A direct electric-plus-natural-gas route
also bids the lowest convertible calciner load. Its ceiling is the
duration-weighted mean of ``min(EOM price forecast, electric WTP)``. An E-TES
plus natural-gas route bids only the auxiliary tranche because a continuous
long-term charger commitment requires a separate storage-activation model.

Fully-electric cement with E-TES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fully-electric cement route requires an electric calciner and a
``short-term_with_generator`` thermal storage. Its EOM demand-bid ceiling is
the electricity-price forecast for each delivery interval; it does not require
natural-gas, CO2, or ``electricity_wtp`` forecasts. If procurement is short,
the rule serves auxiliary and direct calciner electricity first, then charges
E-TES with any remaining electricity. Missing stored heat is recorded as
``unserved_clinker`` and ``gas_fallback`` remains zero. This is accounting
after clearing, not a new Pyomo optimisation.

The route offers negative CRM capacity as additional E-TES charging power. It
checks charger headroom and storage space for continuous activation throughout
the product. Its reservation bid price is the non-negative forecast objective
increase between the ordinary rolling schedule and a non-committing schedule
that preserves the offered charger capacity and state-of-charge space.

For OTC it bids the firm auxiliary load only. The ceiling is the highest valid
market price below the delivery product's duration-weighted EOM forecast and
never exceeds ``auxiliary_security_value``. If no valid price lies below that
forecast, it submits no OTC bid. This means the OTC contract is only procured
when it is offered below the expected EOM cost; residual demand is bid in EOM.

Industrial hybrid market sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The market openings must be ordered so that awards are known before the next
market is bid: ``LTM_OTC`` first, then ``CRM_neg`` when it overlaps delivery,
then ``EOM``, then delivery. For example, a 30-minute OTC window can clear at
00:30 before CRM opens, and before EOM opens at 01:00; a seven-day product is
still configured solely through its product duration::

   bidding_strategy_params:
     auxiliary_security_value: 800

   markets_config:
     LTM_OTC:
       operator: LTM_operator
       product_type: energy
       start_date: 2019-01-01 00:00
       products:
         - duration: 7d
           count: 1
           first_delivery: 2h
       opening_frequency: 7d
       opening_duration: 30m
       market_mechanism: pay_as_bid
     CRM_neg:
       operator: CRM_operator
       product_type: capacity_neg
       start_date: 2019-01-01 00:30
       opening_duration: 30m
     EOM:
       operator: EOM_operator
       product_type: energy
       start_date: 2019-01-01 01:00

The product duration is not embedded in the strategy. It must be fully covered
by the route's required forecasts; otherwise that OTC product is not bid.
Accepted quantities, including partial acceptance, are retained as a MW profile.
Before EOM bidding the rolling solve imposes that procurement as minimum electric
load (and, for steel, minimum DRI electricity) and combines it with any accepted
capacity_neg commitment reservation. EOM demand then covers only the remaining
planned load.

Learning
--------

===================================== ================== =============================================================
bidding_strategy_id                   For Market Types   Description
===================================== ================== =============================================================
powerplant_energy_learning            EOM                A :ref:`reinforcement learning <td3learning>` (RL) approach to formulating bids for a
                                                         Power Plant in an Energy-Only Market. The agent's actions are
                                                         two bid prices: one for the inflexible component (P_min) and another for the flexible component (P_max - P_min) of a unit's capacity.
                                                         The bids are informed by 50 observations, which include forecasted residual load, forecasted price, total capacity, and marginal cost,
                                                         all contributing to decision-making. Noise is added to the action, especially towards the beginning of the learning, to encourage exploration and novelty.
                                                         The reward is calculated based on profits from executed bids, operational costs, opportunity costs (penalizing underutilized capacity),
                                                         and a regret term to minimize missed revenue opportunities. This approach encourages full utilization of the unit's capacity.
storage_energy_learning               EOM                Similar RL approach as :code:`learning_eom_powerplant`, for a Storage unit. The make-up of the observations is similar to those for
                                                         :code:`learning_eom_powerplant`, with an additional observation being the State-of-Charge (SOC) of the storage unit. The agent has 2 actions -
                                                         a bid price, and a bid direction (to buy, sell or do nothing). The bid volume is subject to the charge/discharge capacity of the unit.
                                                         The reward is calculated based on profits from executed bids, with fixed costs for charging/discharging incorporated.
powerplant_energy_learning_single_bid EOM                Reinforcement Learning Strategy with Single-Bid Structure for Energy-Only Markets.
                                                         This strategy is a simplified variant of the standard `EnergyLearningStrategy`, which typically submits two
                                                         separate price bids for inflexible (P_min) and flexible (P_max - P_min) components. Instead,
                                                         `EnergyLearningSingleBidStrategy` submits a single bid that always offers the unit's maximum power,
                                                         effectively treating the full capacity as inflexible from a bidding perspective.
renewable_energy_learning_single_bid  EOM                Reinforcement Learning Strategy for a renewable unit that enables the agent to learn
                                                         optimal bidding strategies on an Energy-Only Market.
===================================== ================== =============================================================

Learning method API references:

- :py:meth:`assume.strategies.learning_strategies.EnergyLearningStrategy`
- :py:meth:`assume.strategies.learning_strategies.EnergyLearningSingleBidStrategy`
- :py:meth:`assume.strategies.learning_strategies.StorageEnergyLearningStrategy`
- :py:meth:`assume.strategies.learning_strategies.RenewableEnergyLearningSingleBidStrategy`

Other
-----

=============================== ================ ================== =============================================================
bidding_strategy_id             For Unit Types   For Market Types   Description
=============================== ================ ================== =============================================================
powerplant_energy_interactive   Any              Any                The bidding volume and price is manually entered in the terminal.
=============================== ================ ================== =============================================================

Miscellaneous method API references:

- :py:meth:`assume.strategies.interactive_strategies.EnergyInteractiveStrategy`
