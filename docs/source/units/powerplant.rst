.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

Power Plant
===========

Overview
--------
The PowerPlant class represents a unit for power generation. It models the behavior of a power plant, including its operational constraints
such as minimum and maximum power output, ramping rates, minimum up and down times, and minimum operating time. It also returns the
marginal cost of the unit for different power outputs.

Initialization Parameters
-------------------------------
.. csv-table::
   :class: full-width
   :header-rows: 1
   :file: powerplant_params.csv

Methods
-------

1. ``init_marginal_cost(self)``
   - Initializes the marginal cost of the unit for the whole simulation period if partial efficiency is False. This saves computational time.

2. ``calc_simple_marginal_cost(self) -> float``
   - Calculates the marginal cost of the unit using a simple method without considering partial efficiency.

   - **Outputs:** float

3. ``calc_marginal_cost_with_partial_eff(self, power_output: float, timestep: pd.Timestamp) -> float | pd.Series``
   - Calculates the marginal cost of the unit based on power output and timestamp, considering partial efficiency.

   - **Inputs:** power_output (float), timestep (pd.Timestamp)
   - **Outputs:** float or pd.Series

4. ``calculate_marginal_cost(self, start: datetime, power: float) -> float``
   - Calculates the marginal cost of the unit based on the provided start time, power output, and value of partial efficiency. Overrides the base class method. Call to get the marginal cost of the unit.

   - **Inputs:** start (datetime), power (float)
   - **Outputs:** float

5. ``execute_current_dispatch(self, start: pd.Timestamp, end: pd.Timestamp) -> float``
   - Executes the current dispatch of the unit. Check for maximal power output, ramping constraints, and minimum up and down times. Corrects the dispatch if necessary.

   - **Inputs:** start (pd.Timestamp), end (pd.Timestamp)
   - **Outputs:** float

6. ``calculate_min_max_power(self, start: pd.Timestamp, end: pd.Timestamp, product_type: str) -> tuple[pd.Series, pd.Series]``
   - Calculates the minimum and maximum power output of the unit for a particular product type (power, capacity, heat). Takes into account ramping constraints, minimum and maximum power, and delivery of other products (e.g. heat).

   - **Inputs:** start (pd.Timestamp), end (pd.Timestamp), product_type (str)
   - **Outputs:** tuple[pd.Series, pd.Series]
