import logging
from datetime import datetime

import numpy as np
import pandas as pd


class Forecaster:
    """
    A Forecaster can provide timeseries for forecasts which are derived either from existing files, random noise or actual forecast methods.
    """

    def __init__(self, index: pd.Series):
        self.index = index

    def __getitem__(self, column: str) -> pd.Series:
        return pd.Series(0, self.index)

    def get_availability(self, unit: str):
        """
        returns the price for a given fuel_type
        or zeros if type does not exist
        """
        return self[f"availability_{unit}"]

    def get_price(self, fuel_type: str):
        """
        returns the price for a given fuel_type
        or zeros if type does not exist
        """
        return self[f"fuel_price_{fuel_type}"]


class CsvForecaster(Forecaster):
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
        if data is None:
            return
        elif isinstance(data, pd.DataFrame):
            if prefix:
                columns = [prefix + column for column in data.columns]
                data.columns = columns
            if len(data.index) == 1:
                for column in data.columns:
                    self.forecasts[column] = data[column].item()
            else:
                new_columns = set(data.columns) - set(self.forecasts.columns)
                self.forecasts = pd.concat(
                    [self.forecasts, data[list(new_columns)]], axis=1
                )
        else:
            self.forecasts[prefix + data.name] = data

    def calc_forecast_if_needed(self):
        for pp in self.powerplants.index:
            col = f"availability_{pp}"
            if col not in self.forecasts.columns:
                self.forecasts[col] = 1
        if "price_EOM" not in self.forecasts.columns:
            self.forecasts["price_EOM"] = self.calculate_EOM_price_forecast()
        if "residual_load_EOM" not in self.forecasts.columns:
            self.forecasts[
                "residual_load_EOM"
            ] = self.calculate_residual_demand_forecast()

    def get_registered_market_participants(self, market_id):
        """
        get information about market participants to make accurate price forecast
        """
        self.logger.warn(
            "Functionality of using the different markets and specified registration for the price forecast is not implemented yet"
        )
        return self.powerplants

    def calculate_residual_demand_forecast(self):
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

        Parameters
        ----------
        pp_series : dict
            Series with power plant data.
        fuel_prices : dict
            Dictionary of fuel data.
        emission_factors : dict
            Dictionary of emission factors.

        Returns
        -------
        marginal_cost : float
            Marginal cost of the power plant.

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
        variable_cost = pp_series["var_cost"] if "var_cost" in pp_series else 0.0

        marginal_cost = fuel_cost + emissions_cost + variable_cost

        return marginal_cost

    def save_forecasts(self, path):
        """
        Saves the forecasts to a csv file.

        Parameters
        ----------
        path : str

        """
        try:
            self.forecasts.to_csv(f"{path}/forecasts_df.csv", index=True)
        except ValueError:
            self.logger.error(
                f"No forecasts for {self.market_id} provided, so none saved."
            )


class RandomForecaster(CsvForecaster):
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
    def __init__(
        self,
        index: pd.Series,
        availability: float | list = 1,
        fuel_price: float | list = 10,
        co2_price: float | list = 10,
        demand: float | list = 100,
        *args,
        **kwargs,
    ):
        super().__init__(index)
        self.fuel_price = fuel_price
        self.availability = availability
        self.co2_price = co2_price
        self.demand = demand

    def __getitem__(self, column: str) -> pd.Series:
        if "availability" in column:
            value = self.availability
        elif column == "fuel_price_co2":
            value = self.co2_price
        elif "fuel_price" in column:
            value = self.fuel_price
        elif "demand" in column:
            value = self.demand
        else:
            value = 0
        return pd.Series(value, self.index)
