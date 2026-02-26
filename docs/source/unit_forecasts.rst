.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

##############
Unit Forecasts
##############

To provide units with static input data such as availability, fuel prices or market price prediction data, each unit has an associated unit forecast.

The data of forecasts is typically provided as time series data and either loaded from a csv file or calculated internally from imported data.

Each unit type has typically its own forecast type for example:

=============== ====================
 Unit Type      Forecast Type
=============== ====================
 Power Plants   PowerplantForecaster
 Storage Units  UnitForecaster
 Demand Units   DemandForecaster
=============== ====================

Each forecast type contains specific fields relevant to the respective unit type. For example, a PowerplantForecaster may include fields such as fuel prices, while a DemandForecaster may include the expected demand profile.
Further, each Forecaster has access to three methods that act as the forecast interface:

1. `preprocess`: this method prepares information / forecasts / etc. for steps 2. and 3.
2. `initialize`: this method initializes all forecast timeseries, after all units are created
3. `update`: this method is used to change forecasts during runtime (shall be called during calculate bids of bidding strategies)

Generally, those three methods call functions that can be specified in the config.yaml or the unit csv files.
In the config.yaml they have to be specified at `forecast_algorithms` as `{prefix}_{forecast_metric}`: `algorithm_id`.
`prefix` denotes which of the three methods `preprocess`, `update`, `initialize` is meant (`initialize` takes no prefix).
`forecast_metric` denotes which of the four forecast metrics `price`, `residual_load`, `congestion_signal` and `renewable_utilisation` is meant.
`algorithm_id` is a specific identifier for the chosen function.

.. code-block:: yaml

    example_study_case:

        other_stuff:
            ...

        forecast_algorithms:
            # functions for 2. initialize have no prefix
            price: price_naive_forecast
            residual_load: residual_load_naive_forecast

            # preprocess and update keys take the corresponding prefix
            preprocess_price: price_default
            update_price: price_default
            update_congestion_signal: congestion_signal_default

For specification in the unit csv files the mapping is done with another fixed `forecast_` in the beginning:
`forecast_{prefix}_{forecast_metric}` as column name and `algorithm_id` as value for the corresponding unit.
Note: The csv file specification will overwrite possible config.yaml specifications (iff csv file specification is not empty/None type at this row)

If neither csv nor config specifies a certain forecast, a default is chosen.

For 2. initialize, the following algorithm ids are currently available.

Price forecast algorithms:

.. list-table:: 
   :widths: 20 30 50
   :header-rows: 1

   * - Type
     - algorithm_id
     - Description
   * - Naive (default)
     - price_naive_forecast
     - Calculates the price forecast based on merit order of the existing units (excl. storages) for all time steps (if not specified in forecast_df)
   * - Keep
     - price_keep_given
     - Keeps the given forecast series (see 2.)
   * - Test
     - price_default_test
     - Provides trivial forecast series for test purposes.

Residual load forecast algorithms:

.. list-table:: 
   :widths: 20 30 50
   :header-rows: 1

   * - Type
     - algorithm_id
     - Description
   * - Naive (default)
     - residual_load_naive_forecast
     - Calculates the residual load forecast by summing the demand and subtracting (expected) renewable generation (wind and solar) for all time steps (if not specified in forecast_df)
   * - Keep
     - residual_load_keep_given
     - Keeps the given forecast series (see 2.)
   * - Test
     - residual_load_default_test
     - Provides trivial forecast series for test purposes.

Congestion signal forecast algorithms:

.. list-table:: 
   :widths: 20 30 50
   :header-rows: 1

   * - Type
     - algorithm_id
     - Description
   * - Naive (default)
     - congestion_signal_naive_forecast
     - Retrieves node-wise congestion forecast by calculating max line congestion from corresponding load of nodes and lince capacity
   * - Keep
     - keep_given
     - Keeps the given forecast series (see 2.)
   * - Test
     - default_test
     - Provides trivial forecast series for test purposes.

Renewable utilisation forecast algorithms:

.. list-table:: 
   :widths: 20 30 50
   :header-rows: 1

   * - Type
     - algorithm_id
     - Description
   * - Naive (default)
     - renewable_utilisation_naive_forecast
     - Retrieves node-wise amount of renewable energy production
   * - Keep
     - keep_given
     - Keeps the given forecast series (see 2.)
   * - Test
     - default_test
     - Provides trivial forecast series for test purposes.

Note, there are other ways a simulation can get access to forecasts.

1. Forecasts can be given to the simulation by creating a forecast_df.csv with corresponding columns. Default forecast algorithms will not overwrite those!
2. Forecast series can be directly given to the forecasters on instantiation when creating a simulation by hand (i.e. not using specific scenario loaders). Note: the forecasting_algorithms should be set to `{forecast_metric}_keep_given` for all forecasting_metrics that use this way! Otherwise the forecast will get overwritten.


Regarding preprocess and update algorithms:
The default preprocess functions do nothing.
Currently there are no other preprocess functions implemented. Feel free to add some for yourself =D

The default update functions does nothing (same forecast series for the whole simulation).
Currently there are no other update functions implemented. Feel free to add some for yourself =D


For custom units, there are two options to handle forecasts:

1. Add a new forecaster for the custom unit type by creating a new class that inherits from `UnitForecaster` and implements the necessary fields and methods.
2. Use the `CustomUnitForecaster`, which allows for flexible definition of forecast fields without creating a new class.

Option 2 is used when importing custom units via csv files.
