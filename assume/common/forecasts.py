import logging

import numpy as np
import pandas as pd


class Forecaster:
    """
    A Forecaster can provide timeseries for forecasts which are derived either from existing files, random noise or actual forecast methods.

    :param index: the index of the forecasts
    :type index: pd.Series

    Methods
    -------
    """

    def __init__(self, index: pd.Series):
        self.index = index

    def __getitem__(self, column: str) -> pd.Series:
        """
        Returns the forecast for a given column.

        :param column: the column of the forecast
        :type column: str
        :return: the forecast
        :rtype: pd.Series
        """
        return pd.Series(0, self.index)

    def get_availability(self, unit: str):
        """
        Returns the availability of a given unit.

        :param unit: the unit
        :type unit: str
        :return: the availability of the unit
        :rtype: pd.Series
        """
        return self[f"availability_{unit}"]

    def get_price(self, fuel_type: str):
        """
        Returns the price for a given fuel_type
        or zeros if type does not exist

        :param fuel_type: the fuel type
        :type fuel_type: str
        :return: the price of the fuel
        :rtype: pd.Series
        """
        return self[f"fuel_price_{fuel_type}"]


class CsvForecaster(Forecaster):
    """
    A Forecaster that reads forecasts from csv files.

    :param index: the index of the forecasts
    :type index: pd.Series
    :param powerplants: the powerplants
    :type powerplants: dict[str, pd.Series]

    Methods
    -------
    """

    def __init__(
        self, index: pd.Series, powerplants: dict[str, pd.Series] = {}, *args, **kwargs
    ):
        super().__init__(index, *args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.powerplants = powerplants
        self.forecasts = pd.DataFrame(index=index)

    def __getitem__(self, column: str) -> pd.Series:
        if column not in self.forecasts.columns:
            if "availability" in column:
                return pd.Series(1, self.index)
            return pd.Series(0, self.index)
        return self.forecasts[column]

    def set_forecast(self, data: pd.DataFrame | pd.Series | None, prefix=""):
        """
        Sets the forecast for a given column.

        :param data: the forecast
        :type data: pd.DataFrame | pd.Series | None
        :param prefix: the prefix of the column
        :type prefix: str
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
        get information about market participants to make accurate price forecast

        :param market_id: the market id
        :type market_id: str
        :return: the registered market participants
        :rtype: pd.DataFrame
        """
        self.logger.warn(
            "Functionality of using the different markets and specified registration for the price forecast is not implemented yet"
        )
        return self.powerplants

    def calculate_residual_demand_forecast(self):
        """
        Calculates the residual demand forecast.

        :return: the residual demand forecast
        :rtype: pd.Series
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
        Function that calculates the merit order price, which is given as a price forecast to the Rl agents
        Here for the entire time horizon at once
        TODO make price forecasts for all markets, not just specified for the DAM like here
        TODO consider storages?

        :return: the merit order price forecast
        :rtype: pd.Series
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
        Calculates the marginal cost of a power plant based on the fuel costs and efficiencies of the power plant.

        :param pp_series: Series with power plant data
        :type pp_series: pd.Series
        :return: the marginal cost of the power plant
        :rtype: float
        """
        fp_column = f"fuel_price_{pp_series.fuel_type}"
        if fp_column in self.forecasts.columns:
            fuel_price = self.forecasts[fp_column]
        else:
            fuel_price = pd.Series(0, index=self.index)

        emission_factor = pp_series["emission_factor"]
        co2_price = self.forecasts["fuel_price_co2"]

        fuel_cost = fuel_price / pp_series["efficiency"]
        emissions_cost = co2_price * emission_factor / pp_series["efficiency"]
        fixed_cost = pp_series["fixed_cost"] if "fixed_cost" in pp_series else 0.0

        marginal_cost = fuel_cost + emissions_cost + fixed_cost

        return marginal_cost

    def save_forecasts(self, path):
        """
        Saves the forecasts to a csv file.

        :param path: the path to save the forecasts to
        :type path: str
        """
        try:
            self.forecasts.to_csv(f"{path}/forecasts_df.csv", index=True)
        except ValueError:
            self.logger.error(
                f"No forecasts for {self.market_id} provided, so none saved."
            )


class RandomForecaster(CsvForecaster):
    """
    A forecaster that generates forecasts using random noise.

    :param index: the index of the forecasts
    :type index: pd.Series
    :param powerplants: the powerplants
    :type powerplants: dict[str, pd.Series]
    :param sigma: the standard deviation of the noise
    :type sigma: float

    Methods
    -------
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
        if column not in self.forecasts.columns:
            return pd.Series(0, self.index)
        noise = np.random.normal(0, self.sigma, len(self.index))
        return self.forecasts[column] * noise


class NaiveForecast(Forecaster):
    """
    A forecaster that generates forecasts using naive methods.

    :param index: the index of the forecasts
    :type index: pd.Series
    :param availability: the availability of the power plants
    :type availability: float | list
    :param fuel_price: the fuel price
    :type fuel_price: float | list
    :param co2_price: the co2 price
    :type co2_price: float | list
    :param demand: the demand
    :type demand: float | list
    :param price_forecast: the price forecast
    :type price_forecast: float | list

    Methods
    -------
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
