# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import numpy as np
import pandas as pd

from assume.common.fast_pandas import FastIndex, FastSeries


class Forecaster:
    """
    Forecaster represents a base class for forecasters based on existing files,
    random noise, or actual forecast methods. It initializes with the provided index. It includes methods
    to retrieve forecasts for specific columns, availability of units, and prices of fuel types, returning
    the corresponding timeseries as pandas Series.

    Attributes:
        index (pandas.Series): The index of the forecasts.

    Args:
        index (pandas.Series): The index of the forecasts.

    Example:
        >>> forecaster = Forecaster(index=pd.Series([1, 2, 3]))
        >>> forecast = forecaster['temperature']
        >>> print(forecast)

    """

    def __init__(self, index: FastIndex):
        self.index = index

    def __getitem__(self, column: str) -> FastSeries:
        """
        Returns the forecast for a given column.

        Args:
            column (str): The column of the forecast.

        Returns:
            FastSeries: The forecast.

        This method returns the forecast for a given column as a pandas Series based on the provided index.
        """
        return FastSeries(value=0.0, index=self.index)

    def get_availability(self, unit: str) -> FastSeries:
        """
        Returns the availability of a given unit as a pandas Series based on the provided index.

        Args:
            unit (str): The unit.

        Returns:
            FastSeries: The availability of the unit.

        Example:
        >>> forecaster = Forecaster(index=pd.Series([1, 2, 3]))
        >>> availability = forecaster.get_availability('unit_1')
        >>> print(availability)
        """

        return self[f"availability_{unit}"]

    def get_price(self, fuel_type: str) -> FastSeries:
        """
        Returns the price for a given fuel type as a pandas Series or zeros if the type does
        not exist.

        Args:
            fuel_type (str): The fuel type.

        Returns:
            FastSeries: The price of the fuel.

        Example:
            >>> forecaster = Forecaster(index=pd.Series([1, 2, 3]))
            >>> price = forecaster.get_price('lignite')
            >>> print(price)
        """

        return self[f"fuel_price_{fuel_type}"]


class CsvForecaster(Forecaster):
    """
    This class represents a forecaster that provides timeseries for forecasts derived from existing files.

    It initializes with the provided index. It includes methods to retrieve forecasts for specific columns,
    availability of units, and prices of fuel types, returning the corresponding timeseries as pandas Series.

    Attributes:
        index (pandas.Series): The index of the forecasts.
        powerplants_units (dict[str, pandas.Series]): The power plants.

    Args:
        index (pandas.Series): The index of the forecasts.
        powerplants_units (dict[str, pandas.Series]): The power plants.

    Example:
        >>> forecaster = CsvForecaster(index=pd.Series([1, 2, 3]))
        >>> forecast = forecaster['temperature']
        >>> print(forecast)

    """

    def __init__(
        self,
        index: pd.Series,
        powerplants_units: pd.DataFrame,
        demand_units: pd.DataFrame,
        market_configs: dict = {},
        save_path: str = "",
        *args,
        **kwargs,
    ):
        super().__init__(index, *args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.powerplants_units = powerplants_units
        self.demand_units = demand_units
        self.market_configs = market_configs
        self.forecasts = pd.DataFrame(index=index)
        self.save_path = save_path

    def __getitem__(self, column: str) -> FastSeries:
        """
        Returns the forecast for a given column.

        If the column does not exist in the forecasts, a Series of zeros is returned. If the column contains "availability", a Series of ones is returned.

        Args:
            column (str): The column of the forecast.

        Returns:
            FastSeries: The forecast for the given column.

        """

        if column not in self.forecasts.keys():
            if "availability" in column:
                self.forecasts[column] = FastSeries(
                    value=1.0, index=self.index, name=column
                )
            else:
                self.forecasts[column] = FastSeries(
                    value=0.0, index=self.index, name=column
                )

        return self.forecasts[column]

    def set_forecast(self, data: pd.DataFrame | pd.Series | None, prefix=""):
        """
        Sets the forecast for a given column.
        If data is a DataFrame, it sets the forecast for each column in the DataFrame. If data is
        a Series, it sets the forecast for the given column. If data is None, no action is taken.

        Args:
            data (pd.DataFrame | pd.Series | None): The forecast data.
            prefix (str): The prefix of the column.

        Example:
            >>> forecaster = CsvForecaster(index=pd.Series([1, 2, 3]))
            >>> forecaster.set_forecast(pd.Series([22, 25, 17], name='temperature'), prefix='location_1_')
            >>> print(forecaster['location_1_temperature'])
        """

        if data is None:
            return
        elif isinstance(data, pd.DataFrame):
            if prefix:
                # set prefix for columns to set
                columns = [prefix + column for column in data.columns]
                data.columns = columns
            if len(data.index) == 1:
                # if we have a single value which should be set for the whole series
                for column in data.columns:
                    self.forecasts[column] = data[column].item()
            else:
                # Add new columns to the existing DataFrame, overwriting any existing columns with the same names
                new_columns = set(data.columns) - set(self.forecasts.columns)
                self.forecasts = pd.concat(
                    [self.forecasts, data[list(new_columns)]], axis=1
                )
        else:
            self.forecasts[prefix + data.name] = data

    def calc_forecast_if_needed(self):
        """
        Calculates the forecasts if they are not already calculated.

        This method calculates price forecast and residual load forecast for available markets, if
        these don't already exist.
        """

        cols = []
        for pp in self.powerplants_units.index:
            col = f"availability_{pp}"
            if col not in self.forecasts.columns:
                s = pd.Series(1, index=self.forecasts.index)
                s.name = col
                cols.append(s)
        cols.append(self.forecasts)
        self.forecasts = pd.concat(cols, axis=1).copy()

        for market_id, config in self.market_configs.items():
            if config["product_type"] != "energy":
                self.logger.warning(
                    f"Price forecast could be calculated for {market_id}. It can only be calculated for energy only markets for now"
                )
                continue

            if f"price_{market_id}" not in self.forecasts.columns:
                self.forecasts[f"price_{market_id}"] = (
                    self.calculate_market_price_forecast(market_id=market_id)
                )

            if f"residual_load_{market_id}" not in self.forecasts.columns:
                self.forecasts[f"residual_load_{market_id}"] = (
                    self.calculate_residual_load_forecast(market_id=market_id)
                )

    def get_registered_market_participants(self, market_id):
        """
        Retrieves information about market participants to make accurate price forecasts.

        Currently, the functionality for using different markets and specified registration for
        the price forecast is not implemented, so it returns the power plants as a DataFrame.

        Args:
            market_id (str): The market ID.

        Returns:
            pd.DataFrame: The registered market participants.

        """

        self.logger.warning(
            "Functionality of using the different markets and specified registration for the price forecast is not implemented yet"
        )
        return self.powerplants_units

    def calculate_residual_load_forecast(self, market_id) -> pd.Series:
        """
        This method calculates the residual demand forecast by subtracting the total available power from renewable energy (VRE) power plants from the overall demand forecast for each time step.

        Returns:
            pd.Series: The residual demand forecast.

        Notes:
            1. Selects VRE power plants (wind_onshore, wind_offshore, solar) from the powerplants_units data.
            2. Creates a DataFrame, vre_feed_in_df, with columns representing VRE power plants and initializes it with zeros.
            3. Calculates the power feed-in for each VRE power plant based on its availability and maximum power.
            4. Calculates the residual demand by subtracting the total VRE power feed-in from the overall demand forecast.
        """

        vre_powerplants_units = self.powerplants_units[
            self.powerplants_units["technology"].isin(
                ["wind_onshore", "wind_offshore", "solar"]
            )
        ].copy()

        vre_feed_in_df = pd.DataFrame(
            index=self.index, columns=vre_powerplants_units.index, data=0.0
        )

        for pp, max_power in vre_powerplants_units["max_power"].items():
            vre_feed_in_df[pp] = self.forecasts[f"availability_{pp}"] * max_power

        demand_units = self.demand_units[
            self.demand_units[f"bidding_{market_id}"].notnull()
        ]
        sum_demand = self.forecasts[demand_units.index].sum(axis=1)

        res_demand_df = sum_demand - vre_feed_in_df.sum(axis=1)

        return res_demand_df

    def calculate_market_price_forecast(self, market_id):
        """
        Calculates the merit order price forecast for the entire time horizon at once.

        The method considers the infeed of renewables, residual demand, and the marginal costs of power plants to derive the price forecast.

        Returns:
            pd.Series: The merit order price forecast.

        Notes:
            1. Calculates the marginal costs for each power plant based on fuel costs, efficiencies, emissions, and fixed costs.
            2. Sorts the power plants based on their marginal costs and availability.
            3. Computes the cumulative power of available power plants.
            4. Determines the price forecast by iterating through the sorted power plants, setting the price for times that can still be provided with a specific technology and cheaper ones.

        TODO:
            Extend price forecasts for all markets, not just specified for the DAM.
            Consider the inclusion of storages in the price forecast calculation.

        """

        # calculate infeed of renewables and residual demand_df
        # check if max_power is a series or a float

        # select only those power plant units, which have a bidding strategy for the specific market_id
        powerplants_units = self.powerplants_units[
            self.powerplants_units[f"bidding_{market_id}"].notnull()
        ]

        marginal_costs = powerplants_units.apply(self.calculate_marginal_cost, axis=1).T
        sorted_columns = marginal_costs.loc[self.index[0]].sort_values().index
        col_availabilities = self.forecasts.columns[
            self.forecasts.columns.str.startswith("availability")
        ]
        availabilities = self.forecasts[col_availabilities]
        availabilities.columns = col_availabilities.str.replace("availability_", "")

        power = self.powerplants_units.max_power * availabilities
        cumsum_power = power[sorted_columns].cumsum(axis=1)

        demand_units = self.demand_units[
            self.demand_units[f"bidding_{market_id}"].notnull()
        ]
        sum_demand = self.forecasts[demand_units.index].sum(axis=1)

        # initialize empty price_forecast
        price_forecast = pd.Series(index=self.index, data=0.0)

        # start with most expensive type (highest cumulative power)
        for col in sorted_columns[::-1]:
            # find times which can still be provided with this technology
            # and cheaper once
            cheaper = cumsum_power[col] > sum_demand
            # set the price of this technology as the forecast price
            # for these times
            price_forecast.loc[cheaper] = marginal_costs[col].loc[cheaper]
            # repeat with the next cheaper technology

        return price_forecast

    def calculate_marginal_cost(self, pp_series: pd.Series) -> pd.Series:
        """
        Calculates time series of marginal costs for a power plant.

        This method calculates the marginal cost of a power plant by taking into account the following factors: \n
        - Fuel costs based on the fuel type and efficiency of the power plant. \n
        - Emissions costs considering the CO2 price and emission factor of the power plant. \n
        - Fixed costs, if specified for the power plant.

        Args:
            pp_series (pandas.Series): Series containing power plant data.

        Returns:
            pandas.Series: The marginal cost of the power plant.

        Notes:
            1. Determines the fuel price based on the fuel type of the power plant.
            2. Calculates the fuel cost by dividing the fuel price by the efficiency of the power plant.
            3. Calculates the emissions cost based on the CO2 price and emission factor, adjusted by the efficiency of the power plant.
            4. Considers any fixed costs specified for the power plant.
            5. Aggregates the fuel cost, emissions cost, and fixed cost to obtain the marginal cost of the power plant.
        """

        fp_column = f"fuel_price_{pp_series.fuel_type}"
        if fp_column in self.forecasts.columns:
            fuel_price = self.forecasts[fp_column]
        else:
            fuel_price = pd.Series(0.0, index=self.index)

        emission_factor = pp_series["emission_factor"]
        co2_price = self.forecasts["fuel_price_co2"]

        fuel_cost = fuel_price / pp_series["efficiency"]
        emissions_cost = co2_price * emission_factor / pp_series["efficiency"]
        additional_cost = (
            pp_series["additional_cost"] if "additional_cost" in pp_series else 0.0
        )

        marginal_cost = fuel_cost + emissions_cost + additional_cost

        return marginal_cost

    def save_forecasts(self, path=None):
        """
        Saves the forecasts to a csv file located at the specified path.

        Args:
            path (str): The path to save the forecasts to.

        Raises:
            ValueError: If no forecasts are provided, an error message is logged.
        """

        path = path or self.save_path

        merged_forecasts = pd.DataFrame(self.forecasts)
        merged_forecasts.index = pd.date_range(
            start=self.index[0], end=self.index[-1], freq=self.index.freq
        )
        merged_forecasts.to_csv(f"{path}/forecasts_df.csv", index=True)

    def convert_forecasts_to_fast_series(self):
        """
        Converts all forecasts in self.forecasts (DataFrame) into FastSeries and saves them
        in a dictionary. It also converts the self.index to a FastIndex.
        """
        # Convert index to FastIndex
        inferred_freq = pd.infer_freq(self.index)
        if inferred_freq is None:
            raise ValueError("Frequency could not be inferred from the index.")

        self.index = FastIndex(
            start=self.index[0], end=self.index[-1], freq=inferred_freq
        )

        # Initialize an empty dictionary to store FastSeries
        fast_forecasts = {}

        # Convert each column in the forecasts DataFrame to a FastSeries
        for column_name in self.forecasts.columns:
            # Convert each column in self.forecasts to FastSeries
            forecast_series = self.forecasts[column_name]
            fast_forecasts[column_name] = FastSeries.from_pandas_series(forecast_series)

        # Replace the DataFrame with the dictionary of FastSeries
        self.forecasts = fast_forecasts


class RandomForecaster(CsvForecaster):
    """
    This class represents a forecaster that generates forecasts using random noise. It inherits
    from the `CsvForecaster` class and initializes with the provided index, power plants, and
    standard deviation of the noise.

    Attributes:
        index (pandas.Series): The index of the forecasts.
        powerplants_units (pandas.DataFrame): The power plants.
        sigma (float): The standard deviation of the noise.

    Args:
        index (pandas.Series): The index of the forecasts.
        powerplants_units (pandas.DataFrame): The power plants.
        sigma (float): The standard deviation of the noise.

    Example:
        >>> forecaster = RandomForecaster(index=pd.Series([1, 2, 3]))
        >>> forecaster.set_forecast(pd.Series([22, 25, 17], name='temperature'), prefix='location_1_')
        >>> print(forecaster['location_1_temperature'])

    """

    def __init__(
        self,
        index: pd.Series,
        powerplants_units: pd.DataFrame,
        demand_units: pd.DataFrame,
        market_configs: dict = {},
        sigma: float = 0.02,
        *args,
        **kwargs,
    ):
        super().__init__(
            index, powerplants_units, demand_units, market_configs, *args, **kwargs
        )

        self.index = FastIndex(start=index[0], end=index[-1], freq=pd.infer_freq(index))
        self.sigma = sigma

    def __getitem__(self, column: str) -> FastSeries:
        """
        Retrieves forecasted values modified by random noise.

        This method returns the forecast for a given column as a pandas Series modified
        by random noise based on the provided standard deviation of the noise and the existing
        forecasts. If the column does not exist in the forecasts, a Series of zeros is returned.

        Args:
            column (str): The column of the forecast.

        Returns:
            FastSeries: The forecast modified by random noise.

        """

        if column not in self.forecasts.columns:
            return FastSeries(value=0.0, index=self.index)
        noise = np.random.normal(0, self.sigma, len(self.index))
        forecast_data = self.forecasts[column].values * noise
        return FastSeries(index=self.index, value=forecast_data)


class NaiveForecast(Forecaster):
    """
    This class represents a forecaster that generates forecasts using naive methods.

    It inherits from the `Forecaster` class and initializes with the provided index and optional parameters
    for availability, fuel price, CO2 price, demand, and price forecast.
    If the optional parameters are constant values, they are converted to pandas Series with the
    provided index. If the optional parameters are lists, they are converted to pandas Series with
    the provided index and the corresponding values.

    Attributes:
        index (pandas.Series): The index of the forecasts.
        availability (float | list, optional): The availability of the power plants.
        fuel_price (float | list, optional): The fuel price.
        co2_price (float | list, optional): The CO2 price.
        demand (float | list, optional): The demand.
        price_forecast (float | list, optional): The price forecast.

    Args:
        index (pandas.Series): The index of the forecasts.
        availability (float | list, optional): The availability of the power plants.
        fuel_price (float | list, optional): The fuel price.
        co2_price (float | list, optional): The CO2 price.
        demand (float | list, optional): The demand.
        price_forecast (float | list, optional): The price forecast.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Example:
        >>> forecaster = NaiveForecast(demand=100, co2_price=10, fuel_price=10, availability=1, price_forecast=50)
        >>> print(forecaster['demand'])

        >>> forecaster = NaiveForecast(index=pd.Series([1, 2, 3]), demand=[100, 200, 300], co2_price=[10, 20, 30])
        >>> print(forecaster["demand"][2])
    """

    def __init__(
        self,
        index: pd.Series,
        availability: float | list = 1,
        fuel_price: float | list = 10,
        co2_price: float | list = 10,
        demand: float | list = 100,
        price_forecast: float | list = 50,
        *args,
        **kwargs,
    ):
        super().__init__(index)
        self.index = FastIndex(start=index[0], end=index[-1], freq=pd.infer_freq(index))

        # Convert attributes to FastSeries if they are not already Series
        self.fuel_price = FastSeries(
            index=self.index, value=fuel_price, name="fuel_price"
        )
        self.availability = FastSeries(
            index=self.index, value=availability, name="availability"
        )
        self.co2_price = FastSeries(index=self.index, value=co2_price, name="co2_price")
        self.demand = FastSeries(index=self.index, value=demand, name="demand")
        self.price_forecast = FastSeries(
            index=self.index, value=price_forecast, name="price_forecast"
        )

        self.data_dict = {}

        for key, value in kwargs.items():
            self.data_dict[key] = FastSeries(index=self.index, value=value, name=key)

    def __getitem__(self, column: str) -> FastSeries:
        """
        Retrieves forecasted values.

        This method retrieves the forecasted values for a specific column based on the
        provided parameters such as availability, fuel price, CO2 price, demand, and price
        forecast. If the column matches one of the predefined parameters, the corresponding
        value is returned as a pandas Series. If the column does not match, a Series of zeros is returned.

        Args:
            column (str): The column for which forecasted values are requested.

        Returns:
            FastSeries: The forecasted values for the specified column.

        """

        if "availability" in column:
            return self.availability
        elif column == "fuel_price_co2":
            return self.co2_price
        elif "fuel_price" in column:
            return self.fuel_price
        elif "demand" in column:
            return self.demand
        elif column == "price_EOM":
            return self.price_forecast
        elif column in self.data_dict.keys():
            return self.data_dict[column]
        else:
            return FastSeries(value=0.0, index=self.index)
