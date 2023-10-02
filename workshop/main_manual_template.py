# %% import packages
import logging
import os
from datetime import datetime, timedelta

import pandas as pd
from dateutil import rrule as rr

from assume import World
from assume.common.forecasts import CsvForecaster, NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct

# import new electrolyser class
# Co-code.. .... .... ... ...

# import new bidding strategy
# Co-code.. .... .... ... ...

# %%
logger = logging.getLogger(__name__)

csv_path = "workshop/outputs"
os.makedirs(csv_path, exist_ok=True)

# create world isntance
world = World(export_csv_path=csv_path)

# add new unit type to world
# Co-code.. .... .... ... ...
# add new bidding strategy to world
# Co-code.. .... .... ... ...


# %%
async def init():
    # define simulation period and ID
    start = None  # Co-code.. .... .... ... ...
    end = None  # Co-code.. .... .... ... ...
    index = pd.date_range(
        start=start,
        end=end + timedelta(hours=24),
        freq="H",
    )
    sim_id = "electrolyser_demo"

    # run world setup to create simulation and different roles
    # this creates the clock and the outputs role
    await world.setup(
        start=start,
        end=end,
        save_frequency_hours=None,
        simulation_id=sim_id,
        index=index,
    )

    # Co-code: define market design and add it to a market
    marketdesign = [
        MarketConfig(
            name=None,  # Co-code,
            opening_hours=rr.rrule(rr.HOURLY, interval=1, dtstart=start, until=end),
            opening_duration=timedelta(hours=1),
            market_mechanism=None,  # Co-code,
            market_products=[
                MarketProduct(
                    duration=timedelta(hours=1),
                    count=1,
                    first_delivery=timedelta(hours=1),
                )
            ],
        )
    ]

    mo_id = "market_operator"
    world.add_market_operator(id=mo_id)
    for market_config in marketdesign:
        world.add_market(
            market_operator_id=mo_id,
            market_config=market_config,
        )

    # add unit operator
    world.add_unit_operator(id="power_plant_operator")

    # define a simple forecaster
    simple_forecaster = NaiveForecast(index, availability=1, fuel_price=0, co2_price=50)

    # add a unit to the world
    world.add_unit(
        id="power_plant_01",
        unit_type="power_plant",
        unit_operator_id="power_plant_operator",
        unit_params={
            "min_power": 0,
            "max_power": 100,
            "bidding_strategies": {"energy": "naive"},
            "fixed_cost": 5,
            "technology": "wind turbine",
        },
        forecaster=simple_forecaster,
    )

    # repeat for demand unit
    world.add_unit_operator("demand_operator")
    world.add_unit(
        id="demand_unit_1",
        unit_type="demand",
        unit_operator_id="demand_operator",
        unit_params={
            "min_power": 0,
            "max_power": 1000,
            "bidding_strategies": {"energy": "naive"},
            "technology": "demand",
        },
        forecaster=NaiveForecast(index, demand=50),
    )

    # load forecasts for hydrogen demand and hydrogen price
    hydrogen_forecasts = pd.read_csv(
        "workshop/inputs/simple_scenario/forecasts_df.csv",
        index_col=0,
        parse_dates=True,
    )

    # add the electrolyser unit to the world
    world.add_unit_operator(id="electrolyser_operator")
    hydrogen_plant_forecaster = CsvForecaster(index=index)
    hydrogen_plant_forecaster.set_forecast(data=hydrogen_forecasts)

    # parameterise electrolyser
    world.add_unit(
        id="elektrolyser_01",
        unit_type="electrolyser",
        unit_operator_id="electrolyser_operator",
        unit_params={
            "min_power": None,  # Co-code,
            "max_power": None,  # Co-code,
            "min_hydrogen": None,  # Co-code,
            "max_hydrogen": None,  # Co-code,
            "bidding_strategies": None,  # Co-code,
            "technology": None,  # Co-code,
            "fixed_cost": None,  # Co-code,
        },
        forecaster=hydrogen_plant_forecaster,
    )


# %%
# run the simulation
world.loop.run_until_complete(init())
world.run()
