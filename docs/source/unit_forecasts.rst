.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

##############
Unit Forecasts
##############

To provide units with static input data such as availability, fuel prices or market price prediction data, each unit has an associated unit forecast.

The data of forecasts is typically provided as time series data and either loaded from a csv file or calculated internally from imported data.

Each unit type has typically its own forecast type:

=============== ====================
 Unit Type      Forecast Type
=============== ====================
 Power Plants   PowerplantForecaster
 Storage Units  UnitForecaster
 Demand Units   DemandForecaster
=============== ====================

Each forecast type contains specific fields relevant to the respective unit type. For example, a PowerplantForecaster may include fields such as fuel prices, while a DemandForecaster may include the expected demand profile.

There are multiple ways a simulation can get access to forecasts.
1. Forecasts can be given to the simulation by creating a forecast_df.csv with corresponding columns.
2. Give forecast series directly to the forecasters on instantiation when creating a simulation by hand (i.e. not using specific scenario loaders). Note: the forecasting_algorithms should be set to `keep_given` for all forecasting_metrics that use this way! Otherwise the forecast will get overwritten.
3. A forecast can be initialized automatically by specifying forecast initialization algorithms in the config. Notably, for all of the four forecast metrics (`price`, `residual_load`, `congestion_signal` and `renewable_utilisation`) there are the algorithm_ids `keep_given` and `default_test`. Currently available algorithm_ids are:
+---------------------------+-----------------------+-------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|   Forecast Metric id      |   Type                |   algorithm_id                            |   Description                                                                                                                                                                             |
+===========================+=======================+===========================================+===========================================================================================================================================================================================+
|   price                   |   Naive (default)     |   price_naive_forecast                    |   Calculates the price forecast based on merit order of the existing units (excl. storages) for all time steps (if not specified in forecast_df)                                          |
|                           +-----------------------+-------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                           |                       |   keep_given                              |   Keeps the given forecast series (see 2.)                                                                                                                                                |
|                           +-----------------------+-------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                           |                       |   default_test                            |   Provides trivial forecast series for test purposes.                                                                                                                                     |
+---------------------------+-----------------------+-------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|   residual_load Load      |   Naive (default)     |   residual_load_naive_forecast            |   Calculates the residual load forecast by summing the demand and subtracting (expected) renewable generation (wind and solar) for all time steps (if not specified in forecast_df)       |
|                           +-----------------------+-------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                           |                       |   keep_given                              |   Keeps the given forecast series (see 2.)                                                                                                                                                |
|                           +-----------------------+-------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                           |                       |   default_test                            |   Provides trivial forecast series for test purposes.                                                                                                                                     |
+---------------------------+-----------------------+-------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|   congestion_signal       |   Naive (default)     |   congestion_signal_naive_forecast        |   Retrieves node-wise congestion forecast by calculating max line congestion from corresponding load of nodes and lince capacity                                                          |
|                           +-----------------------+-------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                           |                       |   keep_given                              |   Keeps the given forecast series (see 2.)                                                                                                                                                |
|                           +-----------------------+-------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                           |                       |   default_test                            |   Provides trivial forecast series for test purposes.                                                                                                                                     |
+---------------------------+-----------------------+-------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|   renewable_utilisation   |   Naive (default)     |   renewable_utilisation_naive_forecast    |   Retrieves node-wise amount of renewable energy production                                                                                                                               |
|                           +-----------------------+-------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                           |                       |   keep_given                              |   Keeps the given forecast series (see 2.)                                                                                                                                                |
|                           +-----------------------+-------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                           |                       |   default_test                            |   Provides trivial forecast series for test purposes.                                                                                                                                     |
+---------------------------+-----------------------+-------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


To update calculated forecast fields at runtime, you need to specify an update function in the config that will be used.

For custom units, there are two options to handle forecasts:

1. Add a new forecaster for the custom unit type by creating a new class that inherits from `UnitForecaster` and implements the necessary fields and methods.
2. Use the `CustomUnitForecaster`, which allows for flexible definition of forecast fields without creating a new class.

Option 2 is used when importing custom units via csv files.
