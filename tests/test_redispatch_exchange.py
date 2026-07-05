# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for cross-border exchange in redispatch.

An exchange unit's net position (import/export) is a committed cross-border schedule
that the national TSO does not redispatch. It must nonetheless be present in the
redispatch power flow as a fixed boundary injection: an export needs a sink and an
import needs a source. Without it the optimisation ramps domestic units down (or up)
to fake the missing exchange away, producing a spurious net redispatch imbalance.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest
from dateutil import rrule as rr

from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.utils import get_available_products

pytest.importorskip("pypsa")

from assume.markets.clearing_algorithms import RedispatchMarketRole

eps = 1e-4


@pytest.fixture
def exchange_redispatch_market_config():
    return MarketConfig(
        market_id="exchange_redispatch",
        market_products=[MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
        additional_fields=["node", "min_power", "max_power"],
        opening_hours=rr.rrule(
            rr.HOURLY,
            dtstart=datetime(2005, 6, 1),
            until=datetime(2005, 6, 2),
            cache=True,
        ),
        opening_duration=timedelta(hours=1),
        volume_unit="MW",
        volume_tick=0.1,
        maximum_bid_volume=None,
        price_unit="€/MW",
        market_mechanism="redispatch",
    )


buses_3 = pd.DataFrame(
    {"name": ["node1", "node2", "node3"], "v_nom": [380.0, 380.0, 380.0]}
).set_index("name")


def _lines(s_nom_1_2: float, s_nom_2_3: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["line_1_2", "line_2_3"],
            "bus0": ["node1", "node2"],
            "bus1": ["node2", "node3"],
            "s_nom": [s_nom_1_2, s_nom_2_3],
            "x": [0.01, 0.01],
            "r": [0.001, 0.001],
        }
    ).set_index("name")


def _grid_export() -> dict:
    # gen1@node1 and gen3@node3 feed load@node2; an exchange at node3 exports 50 MW.
    # line_1_2 is the bottleneck (s_nom 80): relieving it needs gen1 down / gen3 up,
    # while the 50 MW export must stay honoured as a withdrawal at node3.
    return {
        "buses": buses_3,
        "lines": _lines(80.0, 2000.0),
        "generators": pd.DataFrame(
            {
                "name": ["gen1", "gen3"],
                "node": ["node1", "node3"],
                "max_power": [300.0, 300.0],
            }
        ).set_index("name"),
        "loads": pd.DataFrame(
            {"name": ["dem2"], "node": ["node2"], "max_power": [-300.0]}
        ).set_index("name"),
        "exchange_units": pd.DataFrame(
            {"name": ["exch3"], "node": ["node3"]}
        ).set_index("name"),
    }


def _bid(products, unit_id, node, volume, min_power, max_power, price, p_nom=None):
    return {
        "start_time": products[0][0],
        "end_time": products[0][1],
        "unit_id": unit_id,
        "bid_id": f"{unit_id}_bid",
        "volume": volume,
        "min_power": min_power,
        "max_power": max_power,
        "p_nom": p_nom if p_nom is not None else max(abs(min_power), abs(max_power)),
        "price": price,
        "only_hours": None,
        "node": node,
    }


def _run(market_config, grid_data, bids):
    market_config.param_dict["grid_data"] = grid_data
    market_config.param_dict["log_flows"] = True
    next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
    products = get_available_products(market_config.market_products, next_opening)
    orderbook = bids(products)
    rmr = RedispatchMarketRole(market_config)
    accepted_orders, rejected_orders, meta, flows = rmr.clear(orderbook, products)
    accepted = {o["unit_id"]: o["accepted_volume"] for o in accepted_orders}
    rejected = {o["unit_id"]: o["accepted_volume"] for o in rejected_orders}
    return rmr, accepted, rejected, meta, flows


def test_exchange_added_as_fixed_injection(exchange_redispatch_market_config):
    """An exchange unit becomes a single fixed generator with no _up/_down companions."""
    mc = exchange_redispatch_market_config
    mc.param_dict["grid_data"] = _grid_export()
    rmr = RedispatchMarketRole(mc)

    gens = set(rmr.network.generators.index)
    assert "exch3" in gens
    # fixed injection -> it is never redispatched, so no up/down flexibility generators
    assert "exch3_up" not in gens
    assert "exch3_down" not in gens
    # generator sign is +1 (positive p = import, negative p = export)
    assert rmr.network.generators.at["exch3", "sign"] == pytest.approx(1.0)


def test_no_exchange_units_still_initializes(exchange_redispatch_market_config):
    """A grid without an exchange_units key must initialize unchanged (guard works)."""
    mc = exchange_redispatch_market_config
    grid = _grid_export()
    del grid["exchange_units"]
    mc.param_dict["grid_data"] = grid

    rmr = RedispatchMarketRole(mc)
    assert not any(g.startswith("exch3") for g in rmr.network.generators.index)


def test_export_is_honored_not_redispatched(exchange_redispatch_market_config):
    """
    With a 50 MW export honoured at node3, congestion on line_1_2 is relieved by a
    *balanced* generator redispatch (gen1 down, gen3 up). The net accepted redispatch
    is exactly zero -- proving the export is not faked away by a spurious net
    down-regulation (which is what happened when the exchange was absent).
    """

    def bids(products):
        return [
            # committed 100 MW each; flexible 0..300
            _bid(products, "gen1", "node1", 100.0, 0.0, 300.0, 10.0, p_nom=300.0),
            _bid(products, "gen3", "node3", 100.0, 0.0, 300.0, 20.0, p_nom=300.0),
            _bid(products, "dem2", "node2", -150.0, 0.0, -150.0, 1000.0, p_nom=150.0),
            # exchange: net export of 50 MW (negative), fixed (min == max == volume)
            _bid(products, "exch3", "node3", -50.0, -50.0, -50.0, 0.0, p_nom=10e4),
        ]

    _, accepted, rejected, _, flows = _run(
        exchange_redispatch_market_config, _grid_export(), bids
    )

    # the exchange is never redispatched
    assert "exch3" not in accepted
    assert rejected.get("exch3") == pytest.approx(0.0, abs=eps)

    # congestion relieved: gen1 turned down, gen3 turned up
    assert accepted["gen1"] < 0
    assert accepted["gen3"] > 0

    # DECISIVE: the real redispatch is balanced (sum == 0). Without the export sink the
    # optimisation would net-down by the 50 MW export -> sum == -50.
    assert sum(accepted.values()) == pytest.approx(0.0, abs=eps)

    # the bottleneck line is no longer overloaded
    assert abs(flows[(0, "line_1_2")]) <= 80.0 + eps


def test_import_is_honored_as_source(exchange_redispatch_market_config):
    """
    A net import (positive) is honoured as a source: the fixed injection covers demand
    that domestic generation alone would otherwise be ramped up to serve. The exchange
    itself stays out of the accepted redispatch and the real redispatch nets to zero.
    """

    def bids(products):
        # gen1 100 + gen3 100 + import 50 = load 250; line_1_2 (s_nom 80) forces a
        # balanced gen1-down / gen3-up redispatch around the honoured import.
        return [
            _bid(products, "gen1", "node1", 100.0, 0.0, 300.0, 10.0, p_nom=300.0),
            _bid(products, "gen3", "node3", 100.0, 0.0, 300.0, 20.0, p_nom=300.0),
            _bid(products, "dem2", "node2", -250.0, 0.0, -250.0, 1000.0, p_nom=250.0),
            # exchange: net import of 50 MW (positive), fixed
            _bid(products, "exch3", "node3", 50.0, 50.0, 50.0, 0.0, p_nom=10e4),
        ]

    _, accepted, rejected, _, _ = _run(
        exchange_redispatch_market_config, _grid_export(), bids
    )

    assert "exch3" not in accepted
    assert rejected.get("exch3") == pytest.approx(0.0, abs=eps)
    # balanced redispatch around the honoured import
    assert sum(accepted.values()) == pytest.approx(0.0, abs=eps)
