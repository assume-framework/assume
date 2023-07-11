import logging
from itertools import groupby

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
        market_id: str,
        price_forecast_df: pd.DataFrame = None,
        powerplants: dict[str, pd.Series] = None,
        fuel_prices_df: dict[str, pd.Series] = None,
        demand_df: float or pd.Series = 0.0,
        vre_cf_df: dict[str, pd.Series] = None,
    ):
        if fuel_prices_df is None:
            fuel_prices_df = {}
        if vre_cf_df is None:
            vre_cf_df = {}
        if powerplants is None:
            powerplants = {}

        super().__init__()

        self.logger = logging.getLogger(__name__)

        self.market_id = market_id
        self.price_forecast_df = price_forecast_df
        self.fuel_prices_df = fuel_prices_df
        self.powerplants = powerplants
        self.demand_df = demand_df
        self.vre_cf_df = vre_cf_df

        if self.price_forecast_df is None:
            self.price_forecast_df = self.calculate_price_forecast(
                market_id=self.market_id
            )

    def get_registered_market_participants(self, market_id):
        """
        get information about market aprticipants to make accurate price forecast
        """

        raise NotImplementedError(
            "Functionality of using the different markets and specified registration for the price forecast is not implemented yet"
        )

        # calculate price forecast

    def calculate_price_forecast(self, market_id):
        if market_id == "EOM":
            self.logger.info(f"Preparing price forecast for {market_id}")
            return self.calculate_EOM_price_forecast()
        else:
            self.logger.warning(
                f"No price forecast for {market_id} is implemented yet. Please provide an external price forecast."
            )

    def calculate_EOM_price_forecast(self):
        """
        Function that calculates the merit order price, which is given as a price forecast to the Rl agents
        Here for the entire time horizon at once
        TODO make price forecasts for all markets, not just specified for the DAM like here
        TODO consider storages?
        """

        # calculate infeed of renewables and residual demand_df
        # check if max_power is a series or a float
        powerplants = self.powerplants[
            self.powerplants["fuel_type"] != "renewable"
        ].copy()

        price_forecast_df = pd.DataFrame(
            columns=["mcp"], index=self.demand_df.index, data=0.0
        )

        marginal_costs = powerplants.apply(
            self.calculate_marginal_cost, axis=1, fuel_prices=self.fuel_prices_df
        )
        marginal_costs = marginal_costs.T
        if len(marginal_costs) == 1:
            marginal_costs = marginal_costs.iloc[0].to_dict()

        vre_feed_in_df = pd.DataFrame(
            index=self.demand_df.index, columns=self.vre_cf_df.keys(), data=0.0
        )

        for vre in self.vre_cf_df.keys():
            try:
                vre_feed_in_df[vre] = (
                    self.vre_cf_df[vre] * self.powerplants.at[vre, "max_power"]
                )
            except KeyError:
                # no unit with this type
                vre_feed_in_df[vre] = 0

        for i in range(len(self.demand_df)):
            pp_df = powerplants.copy()
            pp_df["marginal_cost"] = (
                marginal_costs.iloc[i]
                if type(marginal_costs) == pd.DataFrame
                else marginal_costs
            )
            mcp = self.calc_market_clearing_price(
                powerplants=pp_df,
                demand=self.demand_df.iat[i],
                vre_feed_in=vre_feed_in_df.iloc[i].sum(),
            )
            price_forecast_df["mcp"].iat[i] = mcp

        return price_forecast_df

    def calculate_marginal_cost(self, pp_dict, fuel_prices):
        """
        Calculates the marginal cost of a power plant based on the fuel costs and efficiencies of the power plant.

        Parameters
        ----------
        pp_dict : dict
            Dictionary of power plant data.
        fuel_prices : dict
            Dictionary of fuel data.
        emission_factors : dict
            Dictionary of emission factors.

        Returns
        -------
        marginal_cost : float
            Marginal cost of the power plant.

        """

        fuel_price = fuel_prices[pp_dict["fuel_type"]]
        emission_factor = pp_dict["emission_factor"]
        co2_price = fuel_prices["co2"]

        fuel_cost = fuel_price / pp_dict["efficiency"]
        emissions_cost = co2_price * emission_factor / pp_dict["efficiency"]
        variable_cost = pp_dict["var_cost"] if "var_cost" in pp_dict else 0.0

        marginal_cost = fuel_cost + emissions_cost + variable_cost

        return marginal_cost

    def calc_market_clearing_price(self, powerplants, demand, vre_feed_in):
        """
        Calculates the market clearing price of the merit order model.

        Parameters
        ----------
        powerplants : pandas.DataFrame
            Dataframe containing the power plant data.
        demand : float
            Demand of the system.

        Returns
        -------
        mcp : float
            Market clearing price.

        """

        # Sort the power plants on marginal cost
        powerplants.sort_values("marginal_cost", inplace=True)

        # Calculate the cumulative capacity
        powerplants["cum_cap"] = powerplants["max_power"].cumsum() + vre_feed_in

        # Calculate the market clearing price
        if powerplants["cum_cap"].iat[-1] < demand:
            mcp = powerplants["marginal_cost"].iat[-1]
        elif vre_feed_in > demand:
            mcp = 0.0
        else:
            mcp = powerplants.loc[
                powerplants["cum_cap"] >= demand, "marginal_cost"
            ].iat[0]

        return mcp
