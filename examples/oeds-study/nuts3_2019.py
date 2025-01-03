#!/usr/bin/env python3

import os
from datetime import datetime, timedelta

from dateutil import rrule as rr

from assume import World
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.scenario.loader_oeds import load_oeds

db_uri = "postgresql://assume:assume@localhost:5432/assume"
world = World(database_uri=db_uri)
scenario = "world_mastr"
# FH Aachen internal server
infra_uri = os.getenv(
    "INFRASTRUCTURE_URI",
    "postgresql://readonly:readonly@timescale.nowum.fh-aachen.de:5432/opendata",
)

# default_nuts_config = "DE1"
default_nuts_config = "DE1, DEA, DEB, DEC, DED, DEE, DEF"
nuts_config = os.getenv("NUTS_CONFIG", default_nuts_config).split(",")
nuts_config = [n.strip() for n in nuts_config]
nuts_config = "nuts3"
year = 2019
random = True
type = "random" if random else "static"
if isinstance(nuts_config, str):
    study_case = f"{nuts_config}_{type}_{year}"
else:
    study_case = f"custom_{type}_{year}"
start = datetime(year, 1, 1)
end = datetime(year, 12, 31) - timedelta(hours=1)
marketdesign = [
    MarketConfig(
        "EOM",
        rr.rrule(rr.HOURLY, interval=24, dtstart=start, until=end),
        timedelta(hours=1),
        "pay_as_clear",
        [MarketProduct(timedelta(hours=1), 24, timedelta(hours=1))],
        additional_fields=["block_id", "link", "exclusive_id"],
        maximum_bid_volume=1e9,
        maximum_bid_price=1e9,
    )
]

default_strategy = {mc.market_id: "naive_eom" for mc in marketdesign}
default_naive_strategy = {mc.market_id: "naive_eom" for mc in marketdesign}

bidding_strategies = {
    "hard coal": default_strategy,
    "lignite": default_strategy,
    "oil": default_strategy,
    "gas": default_strategy,
    "biomass": default_strategy,
    "hydro": default_strategy,
    "nuclear": default_strategy,
    "wind": default_naive_strategy,
    "solar": default_naive_strategy,
    "demand": default_naive_strategy,
    "storage": {mc.market_id: "flexable_eom_storage" for mc in marketdesign},
}
load_oeds(
    world,
    scenario,
    study_case,
    start,
    end,
    infra_uri,
    marketdesign,
    bidding_strategies,
    nuts_config,
)

world.run()