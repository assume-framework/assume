.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

Base Unit Classes
=================

This section describes the base unit classes that are used to derive all units. These classes provide multiple methods
used for dispatch and bidding. To create a custom unit, one of these classes has to be used as a base class, as their methods
ensure the integration of the unit into the simulation model.


BaseUnit Class
==============

A base class for a all units. This class is used as foundation for two main subclasses: SupportsMinMax and SupportsMinMaxCharge.

Initialization
--------------

The class is initialized with the following parameters:

- id (str): Identifier for the unit.
- unit_operator (str): Operator of the unit.
- technology (str): Technology used by the unit.
- bidding_strategies (dict[str, BaseStrategy]): Strategies for bidding.
- index (pd.DatetimeIndex): Index for time-related operations.
- node (str, optional): Node associated with the unit.
- forecaster (Forecaster, optional): Forecaster for predicting unit behavior.
- kwargs: Additional keyword arguments.

Essential Methods
-----------------

calculate_bids
~~~~~~~~~~~~~~

Calculates the bids for the next time step. This function forwards the call to the bidding strategies.

Parameters:
- market_config (MarketConfig): The market configuration.
- product_tuples (list[tuple]): Product tuples.

Returns:
- Orderbook: The bids.

calculate_marginal_cost
~~~~~~~~~~~~~~~~~~~~~~~

Calculates the marginal cost for the given power.

Parameters:
- start (pd.Timestamp): Start time of the dispatch.
- power (float): Power output of the unit.

Returns:
- float: The marginal cost for the given power.

set_dispatch_plan
~~~~~~~~~~~~~~~~~

Adds dispatch plan from current market result to total dispatch plan. Is performed after each market clearing.

Parameters:
- marketconfig (MarketConfig): The market configuration.
- orderbook (Orderbook): The orderbook.

Returns:
- None

execute_current_dispatch
~~~~~~~~~~~~~~~~~~~~~~~~

Checks if the total dispatch plan is feasible. Is performed after all markets are cleared for a given time step.

Parameters:
- start (pd.Timestamp): Start time of the dispatch.
- end (pd.Timestamp): End time of the dispatch.

Returns:
- pd.Series: The volume of the unit within the given time range.

calculate_generation_cost
~~~~~~~~~~~~~~~~~~~~~~~~~

Calculates the generation cost for a given power output.

Parameters:
- power_output (float): Power output of the unit.

Returns:
- float: The generation cost for the given power output.

calculate_cashflow
~~~~~~~~~~~~~~~~~~

Calculates the cashflow for a given product type based on the orderbook. Takes into account the marginal cost and generation cost
and the traded volume and price.

Parameters:
- product_type (str): Type of product for cashflow calculation.

Returns:
- float: The calculated cashflow for the given product type.


SupportsMinMax Class
====================

A subclass of the BaseUnit class, used to derive units that have a minimum and maximum power output
and do not have a storage.

Initialization
--------------

The class is initialized with the following parameters:

- min_power (float): The minimum power output of the unit.
- max_power (float): The maximum power output of the unit.
- ramp_down (float): How much power can be decreased in one time step.
- ramp_up (float): How much power can be increased in one time step.
- efficiency (float): The efficiency of the unit.
- emission_factor (float): The emission factor of the unit.
- min_operating_time (int): The minimum time the unit has to be on (in hours).
- min_down_time (int): The minimum time the unit has to be off (in hours).

Essential Methods
-----------------

calculate_min_max_power
~~~~~~~~~~~~~~~~~~~~~~~

Calculates the min and max power or capacity for the given time period.
The behavior depends on the passed product type. It can be either 'power' or 'capacity'
or 'heat' if the unit supports heat decoupling.

Parameters:
- start (pd.Timestamp): The start time of the dispatch.
- end (pd.Timestamp): The end time of the dispatch.
- product_type (str): The product type of the unit.

Returns:
- tuple[pd.Series, pd.Series]: The min and max power for the given time period.

calculate_ramp
~~~~~~~~~~~~~~

Updates the power taking into account previous power, current power and ramping constraints.
Returns the same passed power if it can be fulfilled.
Returns an adjusted value if ramping constraints are violated.

Parameters:
- previous_power (float): The previous power output of the unit.
- power (float): The power output of the unit.
- current_power (float, optional): The current power output of the unit.

Returns:
- float: The ramp for the given power.

get_operation_time
~~~~~~~~~~~~~~~~~~

Returns the current operation time. Can be used to check if the unit is on for the minimum
operating time.

Parameters:
- start (datetime): The start time.

Returns:
- int: The operation time.

get_average_operation_times
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculates the average uninterrupted operation time. Check how long on average the unit was
on and off in the past. This method can be helpful to determine the bid price when the unit
is turned off.

Parameters:
- start (datetime): The current time.

Returns:
- tuple[float, float]: avg_op_time, avg_down_time.

get_starting_costs
~~~~~~~~~~~~~~~~~~

Returns the costs if start-up is planned. Takes into account the total down time and the
hot start, warm start and cold start times and their respective costs.

Parameters:
- op_time (int): Operation time.

Returns:
- float: The start-up costs.


SupportsMinMaxCharge Class
==========================

A subclass of the BaseUnit class, used for Storage derived classes.

Key Differences from SupportsMinMax
-----------------------------------

The SupportsMinMaxCharge class introduces additional parameters and methods specific to storage units, including charging and discharging operations.

Initialization
--------------

The class is initialized with the following parameters:

- initial_soc (float): The initial state of charge of the storage.
- min_power_charge (float): Minimum power that must be charged in one time step.
- max_power_charge (float): Maximum power that can be charged in one time step.
- min_power_discharge (float): Minimum power that must be discharged in one time step.
- max_power_discharge (float): Maximum power that can be discharged in one time step.
- ramp_up_discharge (float): Maximum increase in discharging power in one time step.
- ramp_down_discharge (float): Maximum decrease in discharging power in one time step.
- ramp_up_charge (float): Maximum increase in charging power in one time step.
- ramp_down_charge (float): Maximum decrease in charging power in one time step.
- max_volume (float): The maximum volume of the storage.
- efficiency_charge (float): The efficiency of charging.
- efficiency_discharge (float): The efficiency of discharging.

Essential Methods
-----------------

calculate_min_max_charge
~~~~~~~~~~~~~~~~~~~~~~~~

Calculates the min and max charging power for the given time period.
Takes into account the current state of charge and the ramping constraints.
Can be also used for 'power' or 'capacity', depending on the product type.

Parameters:
- start (pd.Timestamp): The start time of the dispatch.
- end (pd.Timestamp): The end time of the dispatch.
- product_type (str): The product type of the unit.

Returns:
- tuple[pd.Series, pd.Series]: The min and max charging power for the given time period.

calculate_min_max_discharge
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculates the min and max discharging power for the given time period.
Takes into account the current state of charge and the ramping constraints.
Can be also used for 'power' or 'capacity', depending on the product type.

Parameters:
- start (pd.Timestamp): The start time of the dispatch.
- end (pd.Timestamp): The end time of the dispatch.
- product_type (str): The product type of the unit.

Returns:
- tuple[pd.Series, pd.Series]: The min and max discharging power for the given time period.

get_soc_before
~~~~~~~~~~~~~~

Returns the state of charge before the given datetime. Can be used for a bidding strategy that depends on the current state of charge.

Parameters:
- dt (datetime): The datetime.

Returns:
- float: The state of charge before the given datetime.
