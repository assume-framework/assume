# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime, timedelta

import numpy as np
from dateutil import rrule as rr

from assume import World
from assume.common.base import BaseUnit, SupportsMinMax
from assume.common.fast_pandas import FastIndex
from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct, Orderbook, Product
from assume.strategies.naive_strategies import NaiveSingleBidStrategy

log = logging.getLogger(__name__)


class UpdateForecastBidStrategy(NaiveSingleBidStrategy):
    """
    A naive strategy that bids the marginal cost of the unit on the market.

    Methods
    -------
    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market.

        Args:
            unit (SupportsMinMax): The unit to be dispatched.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): The list of all products the unit can offer.

        Returns:
            Orderbook: The bids consisting of the start time, end time, only hours, price and volume.
        """
        start = product_tuples[0][0]  # start time of the first product
        end_all = product_tuples[-1][1]  # end time of the last product

        min_power_values, max_power_values = unit.calculate_min_max_power(
            start, end_all
        )  # minimum and maximum power output of the unit between the start time of the first product and the end time of the last product

        bids = []
        for product, min_power, max_power in zip(
            product_tuples, min_power_values, max_power_values
        ):
            # for each product, calculate the marginal cost of the unit at the start time of the product
            # and the volume of the product. Dispatch the order to the market.
            start = product[0]
            
            bids.append(
                {
                    "start_time": start,
                    "end_time": product[1],
                    "only_hours": product[2],
                    # the price we are bidding on the gas market
                    "price": 10.0 + np.random.randn()*5,
                    "volume": -unit.max_power, # buy as much as we need
                    "node": unit.node,
                }
            )
        return self.remove_empty_bids(bids)

    def calculate_reward(
        self,
        unit: BaseUnit,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ):
        # set the accepted price of one market as the forecasting price of another market        
        acc = unit.outputs[f"{marketconfig.product_type}_accepted_price"]
        
        min_start = unit.index[-1]
        max_end = unit.index[0]
        for order in orderbook:
            if order["start_time"] < min_start:
                min_start = order["start_time"]
            if order["end_time"] > max_end:
                max_end = order["end_time"]
        max_end = max_end - unit.index.freq
        unit.forecaster["fuel_price_{marketconfig.product_type}"][min_start:max_end] = acc[min_start:max_end]


def init(world, n=1, months=1):
    start = datetime(2019, 1, 1)
    end = datetime(2019, 1, 1) + timedelta(days=30*months)

    index = FastIndex(start, end, freq="h")
    sim_id = "fuel_market"

    world.setup(
        start=start,
        end=end,
        save_frequency_hours=480,
        simulation_id=sim_id,
    )

    world.bidding_strategies["update_forecast"] = UpdateForecastBidStrategy

    marketdesign = [
        MarketConfig(
            market_id="gas_market",
            opening_hours=rr.rrule(
                rr.HOURLY, interval=1, dtstart=start, until=end, cache=True
            ),
            opening_duration=timedelta(hours=1),
            market_mechanism="pay_as_clear",
            market_products=[MarketProduct(timedelta(hours=1), 1, timedelta(hours=2))],
            product_type="gas",
        ),
        MarketConfig(
            market_id="EOM",
            opening_hours=rr.rrule(
                rr.HOURLY, interval=1, dtstart=start+timedelta(hours=1), until=end, cache=True
            ),
            opening_duration=timedelta(hours=1),
            market_mechanism="pay_as_clear",
            market_products=[MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
        ),
    ]
    
    for market_config in marketdesign:
        mo_id = market_config.market_id + "_operator"
        
        world.add_market_operator(id=mo_id)
        world.add_market(mo_id, market_config)

    world.add_unit_operator("my_demand1")
    world.add_unit(
        "demand1",
        "demand",
        "my_demand1",
        # the unit_params have no hints
        {
            "min_power": 0,
            "max_power": 1000,
            "bidding_strategies": {"EOM": "naive_eom"},
            "technology": "demand",
            "price": 20,
        },
        NaiveForecast(index, demand=1000),
    )

    world.add_unit_operator("gas_source")
    world.add_unit(
        "gas_source1",
        "power_plant",
        "gas_source",
        # the unit_params have no hints
        {
            "min_power": 0,
            "max_power": 1000,
            "bidding_strategies": {"gas_market": "naive_eom"},
            "technology": "gas",
        },
        NaiveForecast(index, demand=1000, fuel_price=10),
    )

    gas_forecast = NaiveForecast(index, availability=1, fuel_price=3, co2_price=0.1)
    i = 1
    world.add_unit_operator(f"my_operator{i}")
    world.add_unit(
        f"gas{i}",
        "power_plant",
        f"my_operator{i}",
        {
            "min_power": 200,
            "max_power": 1000,
            "bidding_strategies": {"EOM": "naive_eom", "gas_market": "update_forecast"},
            "technology": "gas",
        },
        gas_forecast,
    )

    i = 2
    coal_forecast = NaiveForecast(index, availability=1, fuel_price=8, co2_price=0.1)
    world.add_unit_operator(f"my_operator{i}")
    world.add_unit(
        f"coal{i}",
        "power_plant",
        f"my_operator{i}",
        {
            "min_power": 200,
            "max_power": 1000,
            "bidding_strategies": {"EOM": "naive_eom"},
            "technology": "coal",
        },
        coal_forecast,
    )

if __name__ == "__main__":
    db_uri = "postgresql://assume:assume@localhost:5432/assume"
    world = World(database_uri=db_uri)
    init(world, months=i)
    world.run()
