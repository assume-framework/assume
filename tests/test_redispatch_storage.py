# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for bidirectional storage redispatch (charge + discharge).

A storage unit sitting in a 3-node radial network offers a symmetric power band
around its committed schedule. Depending on where the congestion is, the redispatch
market activates it upward (extra discharge) or downward (extra charge).
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
def storage_redispatch_market_config():
    return MarketConfig(
        market_id="storage_redispatch",
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

# radial: node1 -- node2 -- node3, storage sits at the middle node2
storage_3 = pd.DataFrame(
    {
        "name": ["stor2"],
        "node": ["node2"],
        "max_power_discharge": [500.0],
        "max_power_charge": [500.0],
    }
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


def _grid_up() -> dict:
    # gen at node1 feeds a load at node2; the node1->node2 line is the bottleneck
    return {
        "buses": buses_3,
        "lines": _lines(600.0, 2000.0),
        "generators": pd.DataFrame(
            {"name": ["gen1"], "node": ["node1"], "max_power": [1000.0]}
        ).set_index("name"),
        "loads": pd.DataFrame(
            {"name": ["dem2"], "node": ["node2"], "max_power": [-1000.0]}
        ).set_index("name"),
        "storage_units": storage_3,
    }


def _grid_down() -> dict:
    # a must-run gen at node2 exports to a load at node1 over a tight line; the only
    # way to relieve it is to raise local generation at node1 and soak the surplus by
    # charging the storage at node2
    return {
        "buses": buses_3,
        "lines": _lines(400.0, 2000.0),
        "generators": pd.DataFrame(
            {
                "name": ["gen1", "gen2"],
                "node": ["node1", "node2"],
                "max_power": [1000.0, 2000.0],
            }
        ).set_index("name"),
        "loads": pd.DataFrame(
            {
                "name": ["dem1", "dem3"],
                "node": ["node1", "node3"],
                "max_power": [-600.0, -400.0],
            }
        ).set_index("name"),
        "storage_units": storage_3,
    }


def _bid(products, unit_id, node, volume, min_power, max_power, p_nom, price):
    return {
        "start_time": products[0][0],
        "end_time": products[0][1],
        "unit_id": unit_id,
        "bid_id": f"{unit_id}_bid",
        "volume": volume,
        "min_power": min_power,
        "max_power": max_power,
        "p_nom": p_nom,
        "price": price,
        "only_hours": None,
        "node": node,
    }


def _run(market_config, grid_data, bids):
    market_config.param_dict["grid_data"] = grid_data
    market_config.param_dict["log_flows"] = True
    next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
    products = get_available_products(market_config.market_products, next_opening)
    # bids are built lazily so callers can use the resolved product times
    orderbook = bids(products)
    rmr = RedispatchMarketRole(market_config)
    accepted_orders, rejected_orders, meta, flows = rmr.clear(orderbook, products)
    accepted = {o["unit_id"]: o["accepted_volume"] for o in accepted_orders}
    return rmr, accepted, meta, flows


def test_storage_units_added_to_network(storage_redispatch_market_config):
    """The storage unit becomes a base + up + down generator triple with p_nom = span."""
    mc = storage_redispatch_market_config
    mc.param_dict["grid_data"] = _grid_up()
    rmr = RedispatchMarketRole(mc)

    gens = rmr.network.generators.index
    assert {"stor2", "stor2_up", "stor2_down"}.issubset(set(gens))
    # p_nom is the full charge+discharge span (500 + 500)
    assert rmr.network.generators.at["stor2_up", "p_nom"] == pytest.approx(1000.0)
    assert rmr.network.generators.at["stor2_down", "p_nom"] == pytest.approx(1000.0)
    # the down generator injects with reversed sign
    assert rmr.network.generators.at["stor2_down", "sign"] == pytest.approx(-1.0)


def test_no_storage_units_still_initializes(storage_redispatch_market_config):
    """A grid without a storage_units key must initialize unchanged (guards work)."""
    mc = storage_redispatch_market_config
    grid = _grid_up()
    del grid["storage_units"]
    mc.param_dict["grid_data"] = grid

    rmr = RedispatchMarketRole(mc)
    assert not any(g.startswith("stor2") for g in rmr.network.generators.index)


def test_storage_redispatch_upward(storage_redispatch_market_config):
    """
    Congestion between the generator (node1) and the load (node2) is relieved by the
    storage discharging locally at node2 -> upward redispatch (accepted_volume > 0).
    """

    def bids(products):
        return [
            _bid(products, "gen1", "node1", 1000.0, 0.0, 1000.0, 1000.0, 10.0),
            _bid(products, "dem2", "node2", -1000.0, 0.0, -1000.0, -1000.0, 1000.0),
            # storage priced above the down-source so the LP only relieves the overload
            _bid(products, "stor2", "node2", 0.0, -500.0, 500.0, 1000.0, 30.0),
        ]

    _, accepted, _, _ = _run(storage_redispatch_market_config, _grid_up(), bids)

    # storage discharges (up), the far generator is turned down to keep balance
    assert accepted["stor2"] > 0
    assert accepted["stor2"] == pytest.approx(400.0, abs=eps)
    assert accepted["gen1"] == pytest.approx(-400.0, abs=eps)
    assert sum(accepted.values()) == pytest.approx(0.0, abs=eps)


def test_storage_redispatch_downward(storage_redispatch_market_config):
    """
    A must-run generator at node2 overloads its export line; the storage charges at
    node2 to absorb the surplus created by raising local generation at node1
    -> downward redispatch (accepted_volume < 0).
    """

    def bids(products):
        return [
            # must-run generator at node2 (min == max == volume -> no flexibility)
            _bid(products, "gen2", "node2", 1000.0, 1000.0, 1000.0, 2000.0, 10.0),
            # flexible (expensive) upward generator at node1
            _bid(products, "gen1", "node1", 0.0, 0.0, 1000.0, 1000.0, 50.0),
            _bid(products, "dem1", "node1", -600.0, 0.0, -600.0, -600.0, 1000.0),
            _bid(products, "dem3", "node3", -400.0, 0.0, -400.0, -400.0, 1000.0),
            # storage priced below the up-source so it is the cheapest way to soak surplus
            _bid(products, "stor2", "node2", 0.0, -500.0, 500.0, 1000.0, 10.0),
        ]

    _, accepted, _, _ = _run(storage_redispatch_market_config, _grid_down(), bids)

    # storage charges (down), local generation at node1 is raised to keep balance
    assert accepted["stor2"] < 0
    assert accepted["stor2"] == pytest.approx(-200.0, abs=eps)
    assert accepted["gen1"] == pytest.approx(200.0, abs=eps)
    assert sum(accepted.values()) == pytest.approx(0.0, abs=eps)


def _up_orderbook(products):
    return [
        _bid(products, "gen1", "node1", 1000.0, 0.0, 1000.0, 1000.0, 10.0),
        _bid(products, "dem2", "node2", -1000.0, 0.0, -1000.0, -1000.0, 1000.0),
        _bid(products, "stor2", "node2", 0.0, -500.0, 500.0, 1000.0, 30.0),
    ]


def test_line_loading_recorded(storage_redispatch_market_config):
    """With log_line_loading enabled, clear() records pre-redispatch loading per line."""
    mc = storage_redispatch_market_config
    mc.param_dict["grid_data"] = _grid_up()
    mc.param_dict["log_line_loading"] = True

    products = get_available_products(
        mc.market_products, mc.opening_hours.after(datetime(2005, 6, 1))
    )
    rmr = RedispatchMarketRole(mc)
    rmr.clear(_up_orderbook(products), products)

    records = rmr.line_loading_results
    assert records is not None
    # one record per line per snapshot (2 lines x 1 product)
    assert len(records) == 2 * len(products)

    by_line = {r["line"]: r for r in records}
    assert set(by_line) == {"line_1_2", "line_2_3"}
    # pre-redispatch: gen1 (1000 MW) -> dem2 (1000 MW) over line_1_2 (s_nom 600) is
    # overloaded => loading = 1000 / 600 > 1 (recorded BEFORE optimize() resolves it)
    assert by_line["line_1_2"]["line_loading"] == pytest.approx(1000.0 / 600.0, abs=eps)
    assert by_line["line_1_2"]["line_loading"] > 1
    # loading == |flow| / (s_nom * s_max_pu), with s_max_pu defaulting to 1
    for r in records:
        assert r["line_loading"] == pytest.approx(
            abs(r["flow_mw"]) / r["s_nom"], abs=eps
        )
        assert "datetime" in r


def test_line_loading_not_recorded_by_default(storage_redispatch_market_config):
    """Without the flag, clear() must not populate line_loading_results."""
    mc = storage_redispatch_market_config
    mc.param_dict["grid_data"] = _grid_up()

    products = get_available_products(
        mc.market_products, mc.opening_hours.after(datetime(2005, 6, 1))
    )
    rmr = RedispatchMarketRole(mc)
    rmr.clear(_up_orderbook(products), products)

    assert getattr(rmr, "line_loading_results", None) is None
