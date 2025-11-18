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

To update calculated forecast fields at runtime, create and run a new `ForecastInitialisation` with the updated data and then use the results to update the respective unit forecasts.

For custom units, there are two options to handle forecasts:

1. Add a new forecaster for the custom unit type by creating a new class that inherits from `UnitForecaster` and implements the necessary fields and methods.
2. Use the `CustomUnitForecaster`, which allows for flexible definition of forecast fields without creating a new class.

Option 2 is used when importing custom units via csv files.
