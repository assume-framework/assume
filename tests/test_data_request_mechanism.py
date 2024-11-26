# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import asyncio
from datetime import datetime

import pandas as pd
from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd
from mango import (
    Agent,
    RoleAgent,
    activate,
    addr,
    create_acl,
    create_ec_container,
)
from mango.util.clock import ExternalClock

from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.units_operator import UnitsOperator
from assume.markets.base_market import MarketRole
from assume.strategies.naive_strategies import NaiveSingleBidStrategy
from assume.units.demand import Demand

start = datetime(2020, 1, 1)
end = datetime(2020, 12, 2)


class DataRequester(Agent):
    def __init__(self):
        super().__init__()
        self.await_message: asyncio.Future = None

    async def send_data_request(
        self, receiver_addr, receiver_id, content: dict, reply_with
    ):
        self.await_message = asyncio.Future()
        await self.send_message(
            create_acl(
                content,
                receiver_addr=addr(receiver_addr, receiver_id),
                sender_addr=self.addr,
                acl_metadata={
                    "reply_with": reply_with,
                },
            ),
            receiver_addr=addr(receiver_addr, receiver_id),
        )

        return await self.await_message

    def handle_message(self, content, meta):
        self.await_message.set_result((content, meta))


async def test_request_messages():
    market_id = "Test"
    marketconfig = MarketConfig(
        market_id=market_id,
        opening_hours=rr.rrule(rr.HOURLY, dtstart=start, until=end),
        opening_duration=rd(hours=1),
        market_mechanism="pay_as_clear",
        market_products=[MarketProduct(rd(hours=1), 1, rd(hours=1))],
    )
    clock = ExternalClock(0)
    container = create_ec_container(
        addr="world", connection_type="external_connection", clock=clock
    )
    units_agent = RoleAgent()
    container.register(units_agent, suggested_aid="test_operator")
    units_role = UnitsOperator(available_markets=[marketconfig])
    units_agent.add_role(units_role)

    index = pd.date_range(start=start, end=end + pd.Timedelta(hours=4), freq="1h")
    forecaster = NaiveForecast(index, demand=1000)
    params_dict = {
        "bidding_strategies": {"EOM": NaiveSingleBidStrategy()},
        "technology": "energy",
        "unit_operator": "test_operator",
        "max_power": 1000,
        "min_power": 0,
        "forecaster": forecaster,
    }
    unit = Demand("testdemand", index=forecaster.index, **params_dict)
    units_role.add_unit(unit)

    market_role = MarketRole(marketconfig)
    market_agent = RoleAgent()
    container.register(market_agent, suggested_aid="market")
    market_agent.add_role(market_role)

    dr = DataRequester()

    container.register(dr, suggested_aid="data_requester")

    market_content = {
        "context": "data_request",
        "market_id": "Test",
        "metric": "price",
        "start_time": index[0],
        "end_time": index[1],
    }
    unit_content = {
        "context": "data_request",
        "unit": "testdemand",
        "metric": "energy",
        "start_time": index[0],
        "end_time": index[3],
    }
    async with activate(container):
        # market results are empty for now
        content, meta = await dr.send_data_request(
            "world", "market", market_content, "market_request"
        )
        assert meta["in_reply_to"] == "market_request"
        assert content["context"] == "data_response"
        assert content["data"].empty

        market_role.results.append({"time": index[0], "price": 12})
        market_role.results.append({"time": index[1], "price": 18})
        content, meta = await dr.send_data_request(
            "world", "market", market_content, "market_request"
        )
        # price is now returned correctly
        assert content["data"][index[0]] == 12

        unit.outputs["energy"][index[1]] = 100
        unit.outputs["energy"][index[3]] = 200

        content, meta = await dr.send_data_request(
            "world", "test_operator", unit_content, "unit_request"
        )
        assert meta["in_reply_to"] == "unit_request"
        assert content["context"] == "data_response"
        assert isinstance(content["data"], pd.Series)
        assert content["data"][index[1]] == 100
        assert content["data"][index[2]] == 0
        assert content["data"][index[3]] == 200
        clock.set_time(end.timestamp() + 1)
