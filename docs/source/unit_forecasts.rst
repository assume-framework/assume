.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

##############
Unit Forecasts
##############

Each unit in ASSUME has an associated **forecaster** that provides it with static input data such as
availability, fuel prices, or market price predictions. Forecast data is typically provided as time
series, either loaded from a CSV file or calculated internally from simulation data.

***********************
Forecaster Types
***********************

Each unit type has its own forecaster class. All forecasters inherit from
:class:`~assume.common.forecaster.UnitForecaster`, which provides the base lifecycle interface and
common attributes (``price``, ``residual_load``, ``availability``).

============================= ========================================= ===========================================
Unit Type                     Forecaster Class                          Additional attributes
============================= ========================================= ===========================================
Power Plants                  ``PowerplantForecaster``                  ``fuel_prices``
Storage Units                 ``UnitForecaster``                        *(base only)*
Demand Units                  ``DemandForecaster``                      ``demand``
Exchange Units                ``ExchangeForecaster``                    ``volume_import``, ``volume_export``
DSM Units                     ``DsmUnitForecaster``                     ``congestion_signal``, ``renewable_utilisation_signal``, ``electricity_price``
Steelplant Units              ``SteelplantForecaster``                  DSM attrs + ``fuel_prices``
Steam Generation Units        ``SteamgenerationForecaster``             DSM attrs + ``fuel_prices``, ``demand``, ``thermal_demand``, ``thermal_storage_schedule``, ``electricity_price_flex``
Building Units                ``BuildingForecaster``                    DSM attrs + ``fuel_prices``, ``heat_demand``, ``ev_load_profile``, ``battery_load_profile``, ``pv_profile``, ``load_profile``
Hydrogen Units                ``HydrogenForecaster``                    DSM attrs + ``hydrogen_demand``, ``seasonal_storage_schedule``
Custom Units (CSV import)     ``CustomUnitForecaster``                  *(arbitrary keyword attributes)*
============================= ========================================= ===========================================

***********************
Forecast Lifecycle
***********************

Every forecaster exposes three lifecycle methods:

1. :code:`preprocess` — prepares intermediate information needed by the other two steps.
2. :code:`initialize` — computes all forecast timeseries. Called **once** after all units are created.
3. :code:`update` — revises forecasts during runtime (e.g. during bid calculation in bidding strategies).

These methods are called automatically by the simulation framework at the appropriate points.

*************************************
Algorithm Resolution and Registries
*************************************

Each lifecycle method decides *which function to run* through two layers of indirection:

1. **``forecast_algorithms``** (``dict[str, str]``) — maps a **key** to an **algorithm ID**.
   Keys follow the pattern ``{prefix}_{forecast_metric}`` where *prefix* is ``preprocess``,
   ``update``, or empty for ``initialize``.

2. **``forecast_registries``** (``dict[str, dict]``) — maps each **algorithm ID** to the actual
   Python callable. It contains three sub-dicts:

   - ``"init"`` — callables for the ``initialize`` step
   - ``"preprocess"`` — callables for the ``preprocess`` step
   - ``"update"`` — callables for the ``update`` step

   The default registries are provided by
   :func:`~assume.common.forecast_algorithms.get_forecast_registries` and contain all built-in
   algorithms listed below.

The resolution flow is:

.. code-block:: text

   forecast_algorithms["price"]  →  "price_naive_forecast"  (algorithm ID)
                                          ↓
   forecast_registries["init"]["price_naive_forecast"]  →  calculate_naive_price  (callable)

This design lets you swap algorithms via configuration without touching code.

***********************************
Configuration
***********************************

Via config.yaml
===============

Specify which algorithms to use in the ``forecast_algorithms`` section of your study case:

.. code-block:: yaml

    example_study_case:

        other_stuff:
            ...

        forecast_algorithms:
            # initialize algorithms have no prefix
            price: price_naive_forecast
            residual_load: residual_load_naive_forecast

            # preprocess and update algorithms take the corresponding prefix
            preprocess_price: price_default
            update_price: price_default
            update_congestion_signal: congestion_signal_default

Via unit CSV files
==================

You can also override algorithms per unit using columns in the unit CSV files.
The column name follows the pattern ``forecast_{prefix}_{forecast_metric}`` and the cell value
is the ``algorithm_id``. (Prefix is empty for initialize algorithms, and ``preprocess`` or ``update`` for the other two steps.)

.. note::
   CSV file specifications **overwrite** config.yaml specifications for that unit
   (only when the CSV cell is not empty/None).

If neither CSV nor config specifies a certain algorithm, a default is chosen automatically.

***********************************
Available Algorithms
***********************************

Initialize algorithms
=====================

These are used by the ``initialize`` method. Their key in ``forecast_algorithms`` has **no prefix**
(e.g. ``price``, ``residual_load``).

Price forecast algorithms:

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Type
     - algorithm_id
     - Description
   * - Naive (default)
     - ``price_naive_forecast``
     - Merit-order dispatch against demand (excl. storages and DSM units) for all timesteps. Automatically uses elastic or inelastic clearing depending on demand unit types. Columns in ``forecast_df`` take precedence over calculated values.
   * - Keep
     - ``price_keep_given``
     - Keeps the forecast series provided at instantiation unchanged.
   * - Test
     - ``price_default_test``
     - Returns a constant price of 50 for the EOM market (for testing only).

Residual load forecast algorithms:

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Type
     - algorithm_id
     - Description
   * - Naive (default)
     - ``residual_load_naive_forecast``
     - Total demand (incl. exchange volumes) minus renewable generation (wind/solar) for all timesteps.
   * - Keep
     - ``residual_load_keep_given``
     - Keeps the forecast series provided at instantiation unchanged.
   * - Test
     - ``residual_load_default_test``
     - Returns an empty dict (for testing only).

Congestion signal forecast algorithms (DSM units only):

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Type
     - algorithm_id
     - Description
   * - Naive (default)
     - ``congestion_signal_naive_forecast``
     - Per-node congestion severity: for each node, takes the max ratio of net load to line capacity across all connected transmission lines. Returns empty dict if grid data is unavailable.
   * - Keep
     - ``congestion_signal_keep_given``
     - Keeps the forecast series provided at instantiation unchanged.
   * - Test
     - ``congestion_signal_default_test``
     - Returns a constant zero signal (for testing only).

Renewable utilisation forecast algorithms (DSM units only):

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Type
     - algorithm_id
     - Description
   * - Naive (default)
     - ``renewable_utilisation_naive_forecast``
     - Per-node renewable generation (availability × max_power) plus an all-nodes aggregate. Returns empty dict if grid data is unavailable.
   * - Keep
     - ``renewable_utilisation_keep_given``
     - Keeps the forecast series provided at instantiation unchanged.
   * - Test
     - ``renewable_utilisation_default_test``
     - Returns a constant zero signal (for testing only).

Preprocess algorithms
=====================

These are used by the ``preprocess`` method. Their key uses the ``preprocess_`` prefix
(e.g. ``preprocess_price``).

.. list-table::
   :widths: 30 30 40
   :header-rows: 1

   * - algorithm_id
     - Applies to
     - Description
   * - ``price_default``
     - price
     - No-op (returns None). This is the default.
   * - ``residual_load_default``
     - residual_load
     - No-op (returns None). This is the default.
   * - ``residual_load_prepare_multiple``
     - residual_load
     - Extracts unit-specific residual load columns from ``forecast_df`` for use in the update step.
   * - ``congestion_signal_default``
     - congestion_signal
     - No-op (returns None). This is the default.
   * - ``renewable_utilisation_default``
     - renewable_utilisation
     - No-op (returns None). This is the default.

Update algorithms
=================

These are used by the ``update`` method. Their key uses the ``update_`` prefix
(e.g. ``update_price``).

.. list-table::
   :widths: 30 30 40
   :header-rows: 1

   * - algorithm_id
     - Applies to
     - Description
   * - ``price_default``
     - price
     - No-op (keeps current forecast unchanged). This is the default.
   * - ``residual_load_default``
     - residual_load
     - No-op (keeps current forecast unchanged). This is the default.
   * - ``residual_load_set_preloaded``
     - residual_load
     - Swaps the current residual load forecast with a named series from ``preprocess_information`` (requires ``residual_load_prepare_multiple`` in preprocess).
   * - ``congestion_signal_default``
     - congestion_signal
     - No-op (keeps current forecast unchanged). This is the default.
   * - ``renewable_utilisation_default``
     - renewable_utilisation
     - No-op (keeps current forecast unchanged). This is the default.

***********************************
Other Ways to Provide Forecasts
***********************************

Besides configuring algorithms, there are two additional ways to supply forecast data:

1. **Via forecast_df.csv** — create a CSV file with columns matching the expected keys
   (e.g. ``price_EOM``, ``residual_load_EOM``, ``{node}_congestion_signal``).
   Default forecast algorithms will **not** overwrite columns found in this file.

2. **Direct instantiation** — when creating a simulation programmatically (without scenario loaders),
   you can pass forecast series directly to the forecaster constructor. In this case, set the
   corresponding algorithm to ``{forecast_metric}_keep_given`` to prevent the initialize step
   from overwriting your data.

***********************************
Adding Custom Algorithms
***********************************

To add a custom forecast algorithm:

1. Write your function following the signature of existing algorithms in
   :mod:`~assume.common.forecast_algorithms`.
2. Register it by adding an entry to the appropriate registry dict
   (``forecast_algorithms``, ``forecast_preprocess_algorithms``, or ``forecast_update_algorithms``).
3. Reference your new ``algorithm_id`` in the config.yaml or unit CSV files.

For custom units, there are two options to handle forecasts:

1. Create a new forecaster class inheriting from :class:`~assume.common.forecaster.UnitForecaster`
   with the necessary fields and methods.
2. Use :class:`~assume.common.forecaster.CustomUnitForecaster`, which accepts arbitrary keyword
   attributes. This is used automatically when importing custom units via CSV files.
