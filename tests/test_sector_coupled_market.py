# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import pytest
from dateutil import rrule as rr

from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.utils import get_available_products
from assume.markets.clearing_algorithms import PayAsClearRole

TOL = 1e-5


@pytest.fixture
def sector_coupled_market_config():
    return MarketConfig(
        market_id="sector_coupled_validation_eom",
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
def validation_product(sector_coupled_market_config):
    next_opening = sector_coupled_market_config.opening_hours.after(
        datetime(2023, 1, 1)
    )
    products = get_available_products(
        sector_coupled_market_config.market_products,
        next_opening,
    )
    assert len(products) == 1
    return products


def _order(product, unit_id, volume, price, node):
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
    return [
        _order(product, "demand_north", -10, 3000, "north"),
        _order(product, "demand_east", -10, 3000, "east"),
        _order(product, "demand_west", -40, 3000, "west"),
        _order(product, "powerplant_north", 65, 20, "north"),
        _order(product, "powerplant_east_cheap", 60, 40, "east"),
        _order(product, "powerplant_west_cheap", 80, 60, "west"),
    ]


def _sector_coupled_orders(product):
    # Aggregate electrical demand of three heat-sector SectorCoupledSystems:
    # north 4 MW_th / COP 2 = 2 MW_el
    # east  6 MW_th / COP 2 = 3 MW_el
    # west  8 MW_th / COP 2 = 4 MW_el
    return [
        _order(product, "hs_north", -2, 3000, "north"),
        _order(product, "hs_east", -3, 3000, "east"),
        _order(product, "hs_west", -4, 3000, "west"),
    ]


def _accepted_by_unit(accepted_orders):
    return {order["unit_id"]: order for order in accepted_orders}


def test_market_without_sector_coupled_system_clears_at_wind_price(
    sector_coupled_market_config,
    validation_product,
):
    product = validation_product[0]
    orderbook = _base_electricity_orders(product)

    market = PayAsClearRole(sector_coupled_market_config)
    accepted, _, meta, _ = market.clear(
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
    assert "powerplant_east_cheap" not in accepted_by_unit
    assert "powerplant_west_cheap" not in accepted_by_unit


def test_market_with_sector_coupled_system_clears_at_coal_price(
    sector_coupled_market_config,
    validation_product,
):
    product = validation_product[0]
    orderbook = _base_electricity_orders(product)
    orderbook.extend(_sector_coupled_orders(product))

    market = PayAsClearRole(sector_coupled_market_config)
    accepted, _, meta, _ = market.clear(
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

    expected_sector_bids = {
        "hs_north": -2.0,
        "hs_east": -3.0,
        "hs_west": -4.0,
    }

    for unit_id, expected_volume in expected_sector_bids.items():
        order = accepted_by_unit[unit_id]
        assert order["accepted_volume"] == pytest.approx(
            expected_volume, abs=TOL
        )
        assert order["accepted_price"] == pytest.approx(40.0, abs=TOL)


def test_sector_coupled_system_changes_market_outcome(
    sector_coupled_market_config,
    validation_product,
):
    product = validation_product[0]

    market_without_sector = PayAsClearRole(sector_coupled_market_config)
    _, _, meta_without_sector, _ = market_without_sector.clear(
        _base_electricity_orders(product),
        validation_product,
    )

    market_with_sector = PayAsClearRole(sector_coupled_market_config)
    with_sector_orders = _base_electricity_orders(product)
    with_sector_orders.extend(_sector_coupled_orders(product))
    _, _, meta_with_sector, _ = market_with_sector.clear(
        with_sector_orders,
        validation_product,
    )

    assert meta_without_sector[0]["demand_volume"] == pytest.approx(
        60.0, abs=TOL
    )
    assert meta_with_sector[0]["demand_volume"] == pytest.approx(
        69.0, abs=TOL
    )
    assert meta_without_sector[0]["price"] == pytest.approx(
        20.0, abs=TOL
    )
    assert meta_with_sector[0]["price"] == pytest.approx(
        40.0, abs=TOL
    )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
