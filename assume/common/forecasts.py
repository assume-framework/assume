import logging
import os
from datetime import datetime
from itertools import groupby
from operator import itemgetter
from pathlib import Path

import pandas as pd
from mango import Role
from mango.messages.message import Performatives

from assume.common.market_objects import (
    ClearingMessage,
    MarketConfig,
    OpeningMessage,
    Order,
    Orderbook,
)
from assume.common.utils import aggregate_step_amount
from assume.strategies import BaseStrategy
from assume.units import BaseUnit


class ForecastProvider(Role):
    def __init__(
        self,
        available_markets: list[MarketConfig],
        fuel_prices: dict[str, pd.Series] = {},
        co2_price: float or pd.Series = 0.0,
        capacity_factors: dict[str, pd.Series] = {},
        powerplants: dict[str, pd.Series] = {},
        demand: float or pd.Series = 0.0,
    ):
        super().__init__()

        self.logger = logging.getLogger(__name__)

        self.bids_map = {}
        self.available_markets = available_markets
        self.registered_markets: dict[str, MarketConfig] = {}
        self.fuel_prices = fuel_prices
        self.co2_price = co2_price
        self.capacity_factors = capacity_factors
        self.all_power_plants = powerplants
        self.demand = demand

    def get_registered_market_participants(self, market_id):
        """
        get information about market aprticipants to make accurate price forecast
        """

        raise NotImplementedError(
            "Functionality of using the different markets and specified registration for the price forecast is not implemented yet"
        )

        # calculate price forecast

    def calculate_price_forecast(
        self,
        market,
        path,
        demand,
        renewable_capacity_factors,
        registered_power_plants,
        fuel_prices,
        co2_prices,
    ):
        """
        Function that calculates the merit order price, which is given as a price forecast to the Rl agents
        Here for the entire time horizon at once
        TODO make price forecasts for all markets, not just specified for the DAM like here
        TODO consider storages?
        """

        # initialize price forecast
        if renewable_capacity_factors is None:
            return None
        price_resdemand_forecast = pd.DataFrame(
            index=renewable_capacity_factors.index, columns=["mcp", "residual_demand"]
        )

        self.demand = list(demand.values)
        forecast_file = path + "/price_forecast_EOM.csv"

        # intialize potential orders
        forecasted_orders = pd.DataFrame(columns=["mcp", "max_power"])

        if market == "EOM":
            # Check if the forecast file already exists
            if os.path.isfile(forecast_file):
                # If inputs haven't changed, load the forecast from the CSV file and exit
                price_resdemand_forecast = pd.read_csv(
                    forecast_file, index_col="Timestamp"
                )
                self.logger.info("Price forecast loaded from the existing file.")
                price_resdemand_forecast.index = pd.to_datetime(
                    price_resdemand_forecast.index
                )

                return price_resdemand_forecast

            # calculate infeed of renewables and residual demand
            # check if max_power is a series or a float
            self.logger.info("Preparing market forecasts")

            for t in renewable_capacity_factors.index:
                i = 0
                res_demand = demand.at[t, "demand_DE"]
                # reset potential orders
                forecasted_orders = forecasted_orders.iloc[0:0]

                for unit_name, unit_params in registered_power_plants.iterrows():
                    # pp = pp[1]
                    if unit_name in renewable_capacity_factors.columns:
                        # adds availabel renewables to merrit order
                        capacity_factor = renewable_capacity_factors[unit_name]

                        max_power = capacity_factor.at[t] * unit_params["max_power"]
                        mcp = 0

                        res_demand = res_demand - max_power

                    else:
                        max_power = unit_params.max_power

                        # calculate simplified marginal costs for each power plant
                        mcp = (
                            fuel_prices[unit_params.fuel_type].at[t]
                            / unit_params["efficiency"]
                            + co2_prices.at[t]
                            * unit_params["emission_factor"]
                            / unit_params["efficiency"]
                            + unit_params["fixed_cost"]
                        )

                        forecasted_orders.loc[i] = {"mcp": mcp, "max_power": max_power}
                        i += 1

                # Sort the DataFrame by the "mcp" column
                forecasted_orders = forecasted_orders.sort_values("mcp")

                # Cumulate the "max_power" column
                forecasted_orders["cumulative_max_power"] = forecasted_orders[
                    "max_power"
                ].cumsum()

                # Find the row where "cumulative_max_power" exceeds a demand value
                filtered_forecasted_orders = forecasted_orders[
                    forecasted_orders["cumulative_max_power"] > res_demand
                ]
                if not filtered_forecasted_orders.empty:
                    mcp = filtered_forecasted_orders.iloc[0]["mcp"]

                else:
                    # demand cannot be supplied by power plant fleet
                    mcp = 3000

                price_resdemand_forecast.at[t, "mcp"] = mcp
                price_resdemand_forecast.at[t, "residual_demand"] = res_demand

        else:
            raise NotImplementedError(
                "For this market the price forecast is not implemented yet"
            )

        self.logger.info("Finished market forecasts")
        price_resdemand_forecast.to_csv(forecast_file, index_label="Timestamp")

        price_resdemand_forecast.index = pd.to_datetime(price_resdemand_forecast.index)

        return price_resdemand_forecast
