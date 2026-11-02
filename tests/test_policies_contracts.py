# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import asyncio
from datetime import datetime, timedelta

import pandas as pd
from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd
from mango import (
    RoleAgent,
    activate,
    addr,
    create_ec_container,
)
from mango.util.clock import ExternalClock

from assume.common.forecaster import DemandForecaster
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.units_operator import UnitsOperator
from assume.markets.base_market import MarketRole
from assume.markets.clearing_algorithms.contracts import (
    PayAsBidContractRole,
    available_contracts,
    market_premium,
)
from assume.strategies.extended import is_co2emissionless
from assume.strategies.naive_strategies import EnergyNaiveStrategy
from assume.units.demand import Demand

# module level start and end for the settlement test
start = datetime(2020, 1, 1)
end = datetime(2020, 12, 2)


def test_contract_functions():
    start = datetime(2019, 1, 1)
    end = datetime(2019, 2, 1)

    index = pd.date_range(
        start=start,
        end=end,
        freq="h",
    )

    contract = {
        "start_time": start,
        "end_time": end,
        "only_hours": None,
        "price": 10,
        "volume": 1000,
        "sender_id": "nuclear1",
        "eligible_lambda": is_co2emissionless,
        "evaluation_frequency": rr.WEEKLY,
        "agent_addr": addr("world", "my_operator"),
        "bid_id": "nuclear1_1",
        "unit_id": "nuclear1",
        "accepted_volume": 1000,
        "accepted_price": 0,
        "contract_price": 4.5,
        "contractor_unit_id": "demand1",
        "contractor_addr": addr("world", "brd"),
        "market_id": "Support",
    }

    market_idx = pd.Series(3, index)
    gen_series = pd.Series(1000, index)

    for c_function in available_contracts.values():
        result = c_function(contract, market_idx, gen_series, start, end)
        assert result

    result = market_premium(contract, market_idx, gen_series, start, end)
    assert result


def make_contract(contract_type: str, unit_id: str, buyer_id: str) -> dict:
    return {
        "start_time": start,
        "end_time": end,
        "only_hours": None,
        "price": 10,
        "volume": 1000,
        "sender_id": unit_id,
        "contract": contract_type,
        "eligible_lambda": is_co2emissionless,
        "evaluation_frequency": rr.WEEKLY,
        "agent_addr": addr("world", "seller_operator"),
        "bid_id": f"{unit_id}_1",
        "unit_id": unit_id,
        "accepted_volume": 1000,
        "accepted_price": 0,
        "contract_price": 4.5,
        "contractor_unit_id": buyer_id,
        "contractor_addr": addr("world", "buyer_operator"),
        "market_id": "Support",
    }


async def test_concurrent_contract_settlement():
    """
    Contracts which share the same buyer and start time (e.g. because they were
    cleared in the same auction) must be settled independently with the
    generation data of their own selling unit.
    """
    eom_config = MarketConfig(
        market_id="EOM",
        opening_hours=rr.rrule(rr.HOURLY, dtstart=start, until=end),
        opening_duration=rd(hours=1),
        market_mechanism="pay_as_clear",
        market_products=[MarketProduct(rd(hours=1), 1, rd(hours=1))],
    )
    support_config = MarketConfig(
        market_id="Support",
        opening_hours=rr.rrule(rr.MONTHLY, dtstart=start, until=end),
        opening_duration=rd(hours=1),
        market_mechanism="pay_as_bid_contract",
        market_products=[MarketProduct(rd(months=1), 1, rd(hours=1))],
        product_type="financial_support",
        additional_fields=[
            "sender_id",
            "contract",
            "eligible_lambda",
            "evaluation_frequency",
        ],
    )

    clock = ExternalClock(start.timestamp())
    container = create_ec_container(
        addr="world", connection_type="external_connection", clock=clock
    )

    # the contract market and the EOM share one market operator agent,
    # as it is the case in the World - this way the contract market can
    # request the EOM price signal from the EOM market role
    market_agent = RoleAgent()
    container.register(market_agent, suggested_aid="market_operator")
    market_agent.add_role(MarketRole(eom_config))
    contract_role = PayAsBidContractRole(support_config)
    market_agent.add_role(contract_role)

    # operator of the selling units
    seller_agent = RoleAgent()
    container.register(seller_agent, suggested_aid="seller_operator")
    seller_role = UnitsOperator(available_markets=[eom_config, support_config])
    seller_agent.add_role(seller_role)

    # operator of the buying demand unit
    buyer_agent = RoleAgent()
    container.register(buyer_agent, suggested_aid="buyer_operator")
    buyer_role = UnitsOperator(available_markets=[eom_config, support_config])
    buyer_agent.add_role(buyer_role)

    index = pd.date_range(start=start, end=end + pd.Timedelta(days=9), freq="1h")
    forecaster = DemandForecaster(index, demand=-1000)

    for unit_id in ["seller1", "seller2", "demand1"]:
        unit = Demand(
            unit_id,
            index=forecaster.index,
            **{
                "bidding_strategies": {
                    "EOM": EnergyNaiveStrategy(),
                    "Support": EnergyNaiveStrategy(),
                },
                "technology": "energy",
                "unit_operator": "seller_operator"
                if unit_id != "demand1"
                else "buyer_operator",
                "max_power": -1000,
                "min_power": 0,
                "forecaster": forecaster,
            },
        )
        if unit_id == "demand1":
            buyer_role.add_unit(unit)
        else:
            seller_role.add_unit(unit)

    # generation of the selling units - different values to detect mixups
    evaluation_begin = start + timedelta(days=8) - timedelta(days=7)
    evaluation_end = start + timedelta(days=8) - timedelta(hours=1)
    seller_role.units["seller1"].outputs["energy"].loc[evaluation_begin:evaluation_end] = 100
    seller_role.units["seller2"].outputs["energy"].loc[evaluation_begin:evaluation_end] = 200

    # EOM price results used for the CFD settlement
    eom_role = market_agent.roles[0]
    for t in pd.date_range(start=evaluation_begin, end=evaluation_end, freq="1h"):
        eom_role.results.append({"time": t, "price": 50})

    # the clock is advanced by one week - the contracts are settled weekly
    clock.set_time((start + timedelta(days=8)).timestamp())

    contract_ppa = make_contract("PPA", "seller1", "demand1")
    contract_cfd = make_contract("CFD", "seller2", "demand1")

    async with activate(container):
        # wait until both operators have registered with the markets
        for _ in range(100):
            if (
                "Support" in seller_role.registered_markets
                and "Support" in buyer_role.registered_markets
            ):
                break
            await asyncio.sleep(0.01)
        assert (
            "Support" in seller_role.registered_markets
            and "Support" in buyer_role.registered_markets
        )

        await asyncio.wait_for(
            asyncio.gather(
                contract_role.execute_contract(contract_ppa),
                contract_role.execute_contract(contract_cfd),
            ),
            timeout=30,
        )
        # wait until the settlement messages are processed
        for _ in range(100):
            cashflows = [
                seller_role.units[unit_id]
                .outputs["financial_support_cashflow"]
                .loc[evaluation_begin:evaluation_end]
                .sum()
                for unit_id in ["seller1", "seller2"]
            ]
            if all(cf != 0 for cf in cashflows):
                break
            await asyncio.sleep(0.01)

    # evaluation window is [begin, end-1h] -> 167 hourly values
    hours = 167
    # PPA: generation of seller1 * contract price
    expected_ppa = 100 * 4.5 * hours
    # CFD: (contract price - market price) * generation of seller2
    expected_cfd = (4.5 - 50) * 200 * hours

    ppa_cashflow = (
        seller_role.units["seller1"]
        .outputs["financial_support_cashflow"]
        .loc[evaluation_begin:evaluation_end]
        .sum()
    )
    cfd_cashflow = (
        seller_role.units["seller2"]
        .outputs["financial_support_cashflow"]
        .loc[evaluation_begin:evaluation_end]
        .sum()
    )

    assert ppa_cashflow == expected_ppa
    assert cfd_cashflow == expected_cfd
