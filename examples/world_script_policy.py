# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime, timedelta

import pandas as pd
from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd

from assume import World
from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.markets.clearing_algorithms.contracts import PayAsBidContractRole

log = logging.getLogger(__name__)


def init(world: World):
    """
    Init function of the Policy Script Scenario
    """
    start = datetime(2019, 1, 1)
    end = datetime(2019, 3, 3)
    index = pd.date_range(
        start=start,
        end=end + timedelta(hours=24),
        freq="h",
    )
    sim_id = "world_script_policy"

    world.clearing_mechanisms["pay_as_bid_contract"] = PayAsBidContractRole
    from assume.strategies.extended import SupportStrategy

    world.bidding_strategies["support"] = SupportStrategy

    world.setup(
        start=start,
        end=end,
        save_frequency_hours=48,
        simulation_id=sim_id,
    )
    contract_types = ["MPFIX"]

    marketdesign = [
        MarketConfig(
            "EOM",
            rr.rrule(
                rr.HOURLY, interval=24, dtstart=start + timedelta(hours=2), until=end
            ),
            timedelta(hours=1),
            "pay_as_clear",
            [MarketProduct(timedelta(hours=1), 24, timedelta(hours=1))],
            product_type="energy",
        ),
        MarketConfig(
            "Support",
            rr.rrule(rr.MONTHLY, dtstart=start, until=end),
            timedelta(hours=1),
            "pay_as_bid_contract",
            [MarketProduct(rd(months=1), 1, timedelta(hours=1))],
            additional_fields=[
                "sender_id",
                "contract",
                "eligible_lambda",
                "evaluation_frequency",  # monthly
            ],
            product_type="financial_support",
            supports_get_unmatched=True,
            param_dict={"allowed_contracts": ["MPVAR", "MPFIX", "CFD"]},
        ),
        MarketConfig(
            "SupportEnergy",
            rr.rrule(rr.MONTHLY, dtstart=start, until=end),
            timedelta(hours=1),
            "pay_as_bid_contract",
            [MarketProduct(rd(days=28), 1, timedelta(days=0))],
            additional_fields=[
                "sender_id",
                "contract",  # one of FIT, PPA
                "eligible_lambda",
                "evaluation_frequency",  # monthly
            ],
            # it needs to be the same product_type to interfere with output
            product_type="energy",
            supports_get_unmatched=True,
            param_dict={"allowed_contracts": ["FIT", "PPA"]},
        ),
    ]

    mo_id = "market_operator"
    world.add_market_operator(id=mo_id)
    for market_config in marketdesign:
        world.add_market(mo_id, market_config)

    world.add_unit_operator("my_operator")
    world.add_unit_operator("brd")
    world.add_unit(
        "demand1",
        "demand",
        "brd",
        # the unit_params have no hints
        {
            "min_power": 0,
            "max_power": 1000,
            "bidding_strategies": {
                "EOM": "support",
                "Support": "support",
                "SupportEnergy": "support",
            },
            "bidding_params": {
                "contract_amount_fraction": 0.5,  # 0.5,
                "contract_types": contract_types,
                "evaluation_frequency": rr.WEEKLY,
            },  # Feed-In-Tariff
            "technology": "demand",
        },
        NaiveForecast(index, demand=1000),
    )

    nuclear_forecast = NaiveForecast(index, availability=1, fuel_price=3, co2_price=0.1)
    world.add_unit(
        "nuclear1",
        "power_plant",
        "my_operator",
        {
            "min_power": 200,
            "max_power": 1000,
            "bidding_strategies": {
                "EOM": "naive_eom",
                "Support": "support",
                "SupportEnergy": "support",
            },
            "bidding_params": {
                "contract_amount_fraction": 0.5,
                "contract_types": contract_types,
                "evaluation_frequency": rr.WEEKLY,
                "contract_value": 6,  # make profit if contract is above EOM
            },  # Feed-In-Tariff
            "technology": "nuclear",
        },
        nuclear_forecast,
    )


if __name__ == "__main__":
    db_uri = "postgresql://assume:assume@localhost:5432/assume"
    world = World(database_uri=db_uri)
    init(world)
    world.run()
