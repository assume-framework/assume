import logging
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

        return NotImplementedError

    def calculate_marginal_price(self, demand, registered_power_plants):
        """
        Function that calculates the merit order price, which is given as a price forecast to the Rl agents
        Here for the entire time horizon at once
        TODO make price forecasts for all markets, not just specified for the DAM like here
        """

        self.demand = list(demand.values)

        powerplants = sorted(
            registered_power_plants, key=lambda x, t=t: x.marginal_cost[t]
        )
        contractedSupply = 0

        registered_power_plants["marginal_costs"] = (
            self.fuel_prices / registered_power_plants["efficiency"]
            + self.co2_price
            * registered_power_plants["emission_factor"]
            / registered_power_plants["efficiency"]
            + registered_power_plants["fixed_cost"]
        )

        registered_power_plants.sort_values(
            "marginal_costs", ascending=True, axi=0, inplace=True
        )

        for pp in powerplants:
            contractedSupply += pp.maxPower
            mcp = pp.marginal_cost[t]
            if contractedSupply >= res_load:
                break
        else:
            mcp = 100

        return mcp
