# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import numpy as np
import pandas as pd


class Forecaster:
    """
    Forecaster represents a base class for forecasters based on existing files,
    random noise, or actual forecast methods. It initializes with the provided index. It includes methods
    to retrieve forecasts for specific columns, availability of units, and prices of fuel types, returning
    the corresponding timeseries as pandas Series.

    Parameters:
        index (pd.Series): The index of the forecasts.

    Args:
        index (pd.Series): The index of the forecasts.

    Example:
        >>> forecaster = Forecaster(index=pd.Series([1, 2, 3]))
        >>> forecast = forecaster['temperature']
        >>> print(forecast)

    """

    def __init__(self, index: pd.Series):
        self.index = index

    def __getitem__(self, column: str) -> pd.Series:
        """
        Returns the forecast for a given column.

        Args:
            column (str): The column of the forecast.

        Returns:
            pd.Series: The forecast.

        This method returns the forecast for a given column as a pandas Series based on the provided index.
        """
        return pd.Series(0.0, self.index)

    def get_availability(self, unit: str) -> pd.Series:
        """
        Returns the availability of a given unit as a pandas Series based on the provided index.

        Args:
            unit (str): The unit.

        Returns:
            pd.Series: The availability of the unit.

        Example:
        >>> forecaster = Forecaster(index=pd.Series([1, 2, 3]))
        >>> availability = forecaster.get_availability('unit_1')
        >>> print(availability)
        """

        return self[f"availability_{unit}"]

    def get_price(self, fuel_type: str) -> pd.Series:
        """
        Returns the price for a given fuel type as a pandas Series or zeros if the type does
        not exist.

        Args:
            fuel_type (str): The fuel type.

        Returns:
            pd.Series: The price of the fuel.

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

    Parameters:
        index (pd.Series): The index of the forecasts.
        powerplants (dict[str, pd.Series]): The power plants.

    Args:
        index (pd.Series): The index of the forecasts.
        powerplants (dict[str, pd.Series]): The power plants.

    Example:
        >>> forecaster = CsvForecaster(index=pd.Series([1, 2, 3]))
        >>> forecast = forecaster['temperature']
        >>> print(forecast)

    """

    def __init__(
        self, index: pd.Series, powerplants: dict[str, pd.Series] = {}, *args, **kwargs
    ):
        super().__init__(index, *args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.powerplants = powerplants
        self.forecasts = pd.DataFrame(index=index)

    def __getitem__(self, column: str) -> pd.Series:
        """
        Returns the forecast for a given column.

        If the column does not exist in the forecasts, a Series of zeros is returned. If the column contains "availability", a Series of ones is returned.

        Args:
            column (str): The column of the forecast.

        Returns:
            pd.Series: The forecast for the given column.

        """

        if column not in self.forecasts.columns:
            if "availability" in column:
                return pd.Series(1, self.index)
            return pd.Series(0.0, self.index)

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
                # if some columns already exist, just add the new columns
                new_columns = set(data.columns) - set(self.forecasts.columns)
                self.forecasts = pd.concat(
                    [self.forecasts, data[list(new_columns)]], axis=1
                )
        else:
            self.forecasts[prefix + data.name] = data

    def calc_forecast_if_needed(self):
        """
        Calculates the forecasts if they are not already calculated.

        This method calculates additional forecasts if they do not already exist, including
        "price_EOM" and "residual_load_forecast".

        Args:
        """

        cols = []
        for pp in self.powerplants.index:
            col = f"availability_{pp}"
            if col not in self.forecasts.columns:
                s = pd.Series(1, index=self.forecasts.index)
                s.name = col
                cols.append(s)
        cols.append(self.forecasts)
        self.forecasts = pd.concat(cols, axis=1).copy()
        if "price_EOM" not in self.forecasts.columns:
            self.forecasts["price_EOM"] = self.calculate_EOM_price_forecast()
        if "residual_load_EOM" not in self.forecasts.columns:
            self.forecasts[
                "residual_load_EOM"
            ] = self.calculate_residual_demand_forecast()

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

        self.logger.warn(
            "Functionality of using the different markets and specified registration for the price forecast is not implemented yet"
        )
        return self.powerplants

    def calculate_residual_demand_forecast(self):
        """
        This method calculates the residual demand forecast by subtracting the total available power from renewable energy (VRE) power plants from the overall demand forecast for each time step.

        Returns:
            pd.Series: The residual demand forecast.

        Notes:
        1. Selects VRE power plants (wind_onshore, wind_offshore, solar) from the powerplants data.
        2. Creates a DataFrame, vre_feed_in_df, with columns representing VRE power plants and initializes it with zeros.
        3. Calculates the power feed-in for each VRE power plant based on its availability and maximum power.
        4. Calculates the residual demand by subtracting the total VRE power feed-in from the overall demand forecast.

        """

        vre_powerplants = self.powerplants[
            self.powerplants["technology"].isin(
                ["wind_onshore", "wind_offshore", "solar"]
            )
        ].copy()

        vre_feed_in_df = pd.DataFrame(
            index=self.index, columns=vre_powerplants.index, data=0.0
        )

        for pp, max_power in vre_powerplants["max_power"].items():
            vre_feed_in_df[pp] = self.forecasts[f"availability_{pp}"] * max_power

        res_demand_df = self.forecasts["demand_EOM"] - vre_feed_in_df.sum(axis=1)

        return res_demand_df

    def calculate_EOM_price_forecast(self):
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
        marginal_costs = self.powerplants.apply(self.calculate_marginal_cost, axis=1).T
        sorted_columns = marginal_costs.loc[self.index[0]].sort_values().index
        col_availabilities = self.forecasts.columns[
            self.forecasts.columns.str.startswith("availability")
        ]
        availabilities = self.forecasts[col_availabilities]
        availabilities.columns = col_availabilities.str.replace("availability_", "")

        power = self.powerplants.max_power * availabilities
        cumsum_power = power[sorted_columns].cumsum(axis=1)
        # initialize empty price_forecast
        price_forecast = pd.Series(index=self.index, data=0.0)

        # start with most expensive type (highest cumulative power)
        for col in sorted_columns[::-1]:
            # find times which can still be provided with this technology
            # and cheaper once
            cheaper = cumsum_power[col] > self.forecasts["demand_EOM"]
            # set the price of this technology as the forecast price
            # for these times
            price_forecast.loc[cheaper] = marginal_costs[col].loc[cheaper]
            # repeat with the next cheaper technology

        return price_forecast

    def calculate_marginal_cost(self, pp_series: pd.Series):
        """
        Calculates time series of marginal costs for a power plant.

        This method calculates the marginal cost of a power plant by taking into account the following factors: \n
        - Fuel costs based on the fuel type and efficiency of the power plant. \n
        - Emissions costs considering the CO2 price and emission factor of the power plant. \n
        - Fixed costs, if specified for the power plant.

        Args:
            pp_series (pd.Series): Series containing power plant data.

        Returns:
            pd.Series: The marginal cost of the power plant.

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
        fixed_cost = pp_series["fixed_cost"] if "fixed_cost" in pp_series else 0.0
        variable_cost = (
            pp_series["variable_cost"] if "variable_cost" in pp_series else 0.0
        )

        marginal_cost = fuel_cost + emissions_cost + fixed_cost + variable_cost

        return marginal_cost

    def save_forecasts(self, path):
        """
        Saves the forecasts to a csv file located at the specified path.

        Args:
            path (str): The path to save the forecasts to.

        Raises:
            ValueError: If no forecasts are provided, an error message is logged.
        """

        try:
            self.forecasts.to_csv(f"{path}/forecasts_df.csv", index=True)
        except ValueError:
            self.logger.error(
                f"No forecasts for {self.market_id} provided, so none saved."
            )


class RandomForecaster(CsvForecaster):
    """
    This class represents a forecaster that generates forecasts using random noise. It inherits
    from the `CsvForecaster` class and initializes with the provided index, power plants, and
    standard deviation of the noise.

    Parameters:
        index (pd.Series): The index of the forecasts.
        powerplants (dict[str, pd.Series]): The power plants.
        sigma (float): The standard deviation of the noise.

    Args:
        index (pd.Series): The index of the forecasts.
        powerplants (dict[str, pd.Series]): The power plants.
        sigma (float): The standard deviation of the noise.

    Example:
        >>> forecaster = RandomForecaster(index=pd.Series([1, 2, 3]))
        >>> forecaster.set_forecast(pd.Series([22, 25, 17], name='temperature'), prefix='location_1_')
        >>> print(forecaster['location_1_temperature'])

    """

    def __init__(
        self,
        index: pd.Series,
        powerplants: dict[str, pd.Series] = {},
        sigma: float = 0.02,
        *args,
        **kwargs,
    ):
        self.sigma = sigma
        super().__init__(index, powerplants, *args, **kwargs)

    def __getitem__(self, column: str) -> pd.Series:
        """
        Retrieves forecasted values modified by random noise.

        This method returns the forecast for a given column as a pandas Series modified
        by random noise based on the provided standard deviation of the noise and the existing
        forecasts. If the column does not exist in the forecasts, a Series of zeros is returned.

        Args:
            column (str): The column of the forecast.

        Returns:
            pd.Series: The forecast modified by random noise.

        """

        if column not in self.forecasts.columns:
            return pd.Series(0.0, self.index)
        noise = np.random.normal(0, self.sigma, len(self.index))
        return self.forecasts[column] * noise


class NaiveForecast(Forecaster):
    """
    This class represents a forecaster that generates forecasts using naive methods.

    It inherits from the `Forecaster` class and initializes with the provided index and optional parameters
    for availability, fuel price, CO2 price, demand, and price forecast.
    If the optional parameters are constant values, they are converted to pandas Series with the
    provided index. If the optional parameters are lists, they are converted to pandas Series with
    the provided index and the corresponding values.

    Parameters:
        index (pd.Series): The index of the forecasts.
        availability (float | list, optional): The availability of the power plants.
        fuel_price (float | list, optional): The fuel price.
        co2_price (float | list, optional): The CO2 price.
        demand (float | list, optional): The demand.
        price_forecast (float | list, optional): The price forecast.

    Args:
        index (pd.Series): The index of the forecasts.
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
        self.fuel_price = fuel_price
        self.availability = availability
        self.co2_price = co2_price
        self.demand = demand
        self.price_forecast = price_forecast

    def __getitem__(self, column: str) -> pd.Series:
        """
        Retrieves forecasted values.

        This method retrieves the forecasted values for a specific column based on the
        provided parameters such as availability, fuel price, CO2 price, demand, and price
        forecast. If the column matches one of the predefined parameters, the corresponding
        value is returned as a pandas Series. If the column does not match, a Series of zeros is returned.

        Args:
            column (str): The column for which forecasted values are requested.

        Returns:
            pd.Series: The forecasted values for the specified column.

        """

        if "availability" in column:
            value = self.availability
        elif column == "fuel_price_co2":
            value = self.co2_price
        elif "fuel_price" in column:
            value = self.fuel_price
        elif "demand" in column:
            value = self.demand
        elif column == "price_EOM":
            value = self.price_forecast
        else:
            value = 0
        if isinstance(value, pd.Series):
            value.index = self.index
        return pd.Series(value, self.index)
