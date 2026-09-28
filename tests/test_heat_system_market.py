# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest
from dateutil import rrule as rr

from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.utils import get_available_products
from assume.markets.clearing_algorithms import PayAsClearRole
from assume.strategies.naive_strategies import DsmEnergyOptimizationStrategy


TOL = 1e-5


@pytest.fixture
def heat_system_market_config():
    """
    One-hour copper-plate EOM used to validate the visible market effect of
    adding the three HeatSystems.
    """
    return MarketConfig(
        market_id="heat_system_validation_eom",
        market_products=[
            MarketProduct(
                timedelta(hours=1),
                1,
                timedelta(hours=1),
            )
        ],
        additional_fields=["node"],
        opening_hours=rr.rrule(
            rr.HOURLY,
            dtstart=datetime(2023, 1, 1),
            until=datetime(2023, 1, 2),
            cache=True,
        ),
        opening_duration=timedelta(hours=1),
        volume_unit="MW",
        volume_tick=0.1,
        maximum_bid_volume=100000,
        maximum_bid_price=3000,
        minimum_bid_price=-500,
        price_unit="EUR/MWh",
        market_mechanism="pay_as_clear",
    )


@pytest.fixture
def validation_product(heat_system_market_config):
    next_opening = heat_system_market_config.opening_hours.after(
        datetime(2023, 1, 1)
    )
    products = get_available_products(
        heat_system_market_config.market_products,
        next_opening,
    )
    assert len(products) == 1
    return products


def _order(product, unit_id, volume, price, node):
    """Build one simple EOM order."""
    return {
        "start_time": product[0],
        "end_time": product[1],
        "only_hours": product[2],
        "unit_id": unit_id,
        "bid_id": f"{unit_id}_bid",
        "volume": float(volume),
        "price": float(price),
        "node": node,
    }


def _base_electricity_orders(product):
    """
    Reference 3-node electricity system.

    Base demand:
        north = 10 MW
        east  = 10 MW
        west  = 40 MW
        total = 60 MW

    Merit order:
        wind north = 65 MW @ 20 EUR/MWh
        coal east  = 60 MW @ 40 EUR/MWh
        gas west   = 80 MW @ 60 EUR/MWh

    With no HeatSystem, wind alone supplies all 60 MW and the expected
    clearing price is 20 EUR/MWh.
    """
    return [
        # Inelastic base electricity demand
        _order(product, "demand_north", -10, 3000, "north"),
        _order(product, "demand_east", -10, 3000, "east"),
        _order(product, "demand_west", -40, 3000, "west"),
        # Generation supply
        _order(product, "powerplant_north", 65, 20, "north"),
        _order(product, "powerplant_east_cheap", 60, 40, "east"),
        _order(product, "powerplant_west_cheap", 80, 60, "west"),
    ]


def _heat_system_orders(product):
    """
    Expected HeatSystem EOM bids for the validation case:

        North: 4 MW_th / COP 2 = 2 MW_el
        East:  6 MW_th / COP 2 = 3 MW_el
        West:  8 MW_th / COP 2 = 4 MW_el

    Demand bids are negative in ASSUME.
    """
    return [
        _order(product, "hs_north", -2, 3000, "north"),
        _order(product, "hs_east", -3, 3000, "east"),
        _order(product, "hs_west", -4, 3000, "west"),
    ]


def _accepted_by_unit(accepted_orders):
    return {order["unit_id"]: order for order in accepted_orders}


@pytest.mark.parametrize(
    "unit_id,node,power_requirement,expected_volume",
    [
        ("hs_north", "north", 2.0, -2.0),
        ("hs_east", "east", 3.0, -3.0),
        ("hs_west", "west", 4.0, -4.0),
    ],
)
def test_heat_system_energy_strategy_creates_expected_demand_bid(
    heat_system_market_config,
    validation_product,
    unit_id,
    node,
    power_requirement,
    expected_volume,
):
    """
    The current HeatSystem EOM strategy uses opt_power_requirement as quantity
    and submits it with the demand sign convention (negative volume).

    This deliberately tests the current implementation where the EOM bid price
    is fixed at 3000 EUR/MWh.
    """
    start = validation_product[0][0]

    unit = SimpleNamespace(
        id=unit_id,
        node=node,
        horizon_mode="full_horizon",
        optimisation_counter=1,
        opt_power_requirement=pd.Series(
            [power_requirement],
            index=[start],
            dtype=float,
        ),
    )

    strategy = DsmEnergyOptimizationStrategy()
    bids = strategy.calculate_bids(
        unit,
        heat_system_market_config,
        validation_product,
    )

    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(expected_volume, abs=TOL)
    assert bids[0]["price"] == pytest.approx(3000.0, abs=TOL)


def test_market_without_heat_system_clears_at_wind_price(
    heat_system_market_config,
    validation_product,
):
    """
    Without HeatSystems:

        total demand = 60 MW
        wind capacity = 65 MW @ 20 EUR/MWh

    Therefore wind alone is marginal and the clearing price must be 20 EUR/MWh.
    """
    product = validation_product[0]
    orderbook = _base_electricity_orders(product)

    market = PayAsClearRole(heat_system_market_config)
    accepted, rejected, meta, _ = market.clear(
        orderbook,
        validation_product,
    )

    accepted_by_unit = _accepted_by_unit(accepted)

    assert meta[0]["demand_volume"] == pytest.approx(60.0, abs=TOL)
    assert meta[0]["supply_volume"] == pytest.approx(60.0, abs=TOL)
    assert meta[0]["price"] == pytest.approx(20.0, abs=TOL)

    assert accepted_by_unit["powerplant_north"]["accepted_volume"] == pytest.approx(
        60.0, abs=TOL
    )

    # Coal and gas must not be needed in the no-HeatSystem reference case.
    assert "powerplant_east_cheap" not in accepted_by_unit
    assert "powerplant_west_cheap" not in accepted_by_unit


def test_market_with_heat_system_clears_at_coal_price(
    heat_system_market_config,
    validation_product,
):
    """
    With HeatSystems:

        base electricity demand = 60 MW
        HeatSystem demand       = 2 + 3 + 4 = 9 MW
        total demand            = 69 MW

    Wind supplies its full 65 MW and coal supplies the remaining 4 MW.
    Therefore coal becomes marginal and the clearing price must be 40 EUR/MWh.
    """
    product = validation_product[0]
    orderbook = _base_electricity_orders(product)
    orderbook.extend(_heat_system_orders(product))

    market = PayAsClearRole(heat_system_market_config)
    accepted, rejected, meta, _ = market.clear(
        orderbook,
        validation_product,
    )

    accepted_by_unit = _accepted_by_unit(accepted)

    assert meta[0]["demand_volume"] == pytest.approx(69.0, abs=TOL)
    assert meta[0]["supply_volume"] == pytest.approx(69.0, abs=TOL)
    assert meta[0]["price"] == pytest.approx(40.0, abs=TOL)

    assert accepted_by_unit["powerplant_north"]["accepted_volume"] == pytest.approx(
        65.0, abs=TOL
    )
    assert accepted_by_unit[
        "powerplant_east_cheap"
    ]["accepted_volume"] == pytest.approx(4.0, abs=TOL)
    assert "powerplant_west_cheap" not in accepted_by_unit

    # Every HeatSystem demand bid should be fully accepted.
    expected_heat_bids = {
        "hs_north": -2.0,
        "hs_east": -3.0,
        "hs_west": -4.0,
    }
    for unit_id, expected_volume in expected_heat_bids.items():
        order = accepted_by_unit[unit_id]
        assert order["accepted_volume"] == pytest.approx(
            expected_volume, abs=TOL
        )
        assert order["accepted_price"] == pytest.approx(40.0, abs=TOL)


def test_heat_system_changes_market_outcome(
    heat_system_market_config,
    validation_product,
):
    """
    Regression test for the central integration result:

        without HeatSystem -> 60 MW demand, 20 EUR/MWh
        with HeatSystem    -> 69 MW demand, 40 EUR/MWh
    """
    product = validation_product[0]

    market_without_heat = PayAsClearRole(heat_system_market_config)
    _, _, meta_without_heat, _ = market_without_heat.clear(
        _base_electricity_orders(product),
        validation_product,
    )

    market_with_heat = PayAsClearRole(heat_system_market_config)
    with_heat_orders = _base_electricity_orders(product)
    with_heat_orders.extend(_heat_system_orders(product))
    _, _, meta_with_heat, _ = market_with_heat.clear(
        with_heat_orders,
        validation_product,
    )

    assert meta_without_heat[0]["demand_volume"] == pytest.approx(
        60.0, abs=TOL
    )
    assert meta_with_heat[0]["demand_volume"] == pytest.approx(
        69.0, abs=TOL
    )

    assert meta_without_heat[0]["price"] == pytest.approx(20.0, abs=TOL)
    assert meta_with_heat[0]["price"] == pytest.approx(40.0, abs=TOL)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
