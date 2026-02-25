# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import pandas as pd
import pytest
from dateutil import rrule as rr

from assume.common.forecaster import ExchangeForecaster
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.strategies import ExchangeEnergyNaiveStrategy
from assume.units.exchange import Exchange

# === FIXTURES ===


@pytest.fixture
def market_config():
    """Fixture for market configuration."""
    return MarketConfig(
        market_id="EOM",
        opening_hours=rr.rrule(rr.HOURLY),
        opening_duration=timedelta(hours=1),
        market_mechanism="not needed",
        market_products=[MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
    )


@pytest.fixture
def full_forecast():
    """Fixture for forecasted import/export values with both import and export data."""
    index = pd.date_range(start=datetime(2024, 1, 1), periods=24, freq="1h")
    return ExchangeForecaster(
        index=index,
        volume_import=100,
        volume_export=50,
        availability=1,
        market_prices={},
    )  # Both import and export


@pytest.fixture
def import_only_forecast():
    """Fixture for forecasted values where only import is available."""
    index = pd.date_range(start=datetime(2024, 1, 1), periods=24, freq="1h")
    return ExchangeForecaster(index, volume_import=100)  # Only import, no export


@pytest.fixture
def export_only_forecast():
    """Fixture for forecasted values where only export is available."""
    index = pd.date_range(start=datetime(2024, 1, 1), periods=24, freq="1h")
    return ExchangeForecaster(index, volume_export=50)  # Only export, no import


@pytest.fixture
def exchange_unit(full_forecast):
    """Fixture for an Exchange unit with full import/export forecast."""
    return Exchange(
        id="test_unit",
        unit_operator="test_operator",
        bidding_strategies={"EOM": ExchangeEnergyNaiveStrategy()},
        forecaster=full_forecast,
        price_import=10.0,
        price_export=2000.0,
    )


@pytest.fixture
def exchange_import_only(import_only_forecast):
    """Fixture for an Exchange unit that only has import forecasted."""
    return Exchange(
        id="test_unit",
        unit_operator="test_operator",
        bidding_strategies={"EOM": ExchangeEnergyNaiveStrategy()},
        forecaster=import_only_forecast,
        price_import=10.0,
        price_export=2000.0,
    )


@pytest.fixture
def exchange_export_only(export_only_forecast):
    """Fixture for an Exchange unit that only has export forecasted."""
    return Exchange(
        id="test_unit",
        unit_operator="test_operator",
        bidding_strategies={"EOM": ExchangeEnergyNaiveStrategy()},
        forecaster=export_only_forecast,
        price_import=10.0,
        price_export=2000.0,
    )


# === TESTS ===


def test_exchange_initialization(exchange_unit):
    """Test if the Exchange unit initializes correctly with both import and export forecast."""
    assert exchange_unit.id == "test_unit"
    assert exchange_unit.unit_operator == "test_operator"
    assert exchange_unit.price_import == 10.0
    assert exchange_unit.price_export == 2000.0

    # Validate correct forecasting data
    assert exchange_unit.volume_import.at["2024-01-01 00:00:00"] == 100, (
        "Incorrect import volume"
    )
    assert exchange_unit.volume_export.at["2024-01-01 00:00:00"] == -50, (
        "Incorrect export volume"
    )  # Export is negative


def test_exchange_import_only(exchange_import_only):
    """Test Exchange unit when only import data is available in the forecast."""
    assert exchange_import_only.volume_import.at["2024-01-01 00:00:00"] == 100, (
        "Incorrect import volume"
    )

    # Export should be zero when missing from forecast
    assert exchange_import_only.volume_export.at["2024-01-01 00:00:00"] == 0, (
        "Export should default to 0 when missing"
    )


def test_exchange_export_only(exchange_export_only):
    """Test Exchange unit when only export data is available in the forecast."""
    assert exchange_export_only.volume_export.at["2024-01-01 00:00:00"] == -50, (
        "Incorrect export volume"
    )

    # Import should be zero when missing from forecast
    assert exchange_export_only.volume_import.at["2024-01-01 00:00:00"] == 0, (
        "Import should default to 0 when missing"
    )


def test_exchange_energy_naive_strategy_bidding(exchange_unit, market_config):
    """Test if ExchangeEnergyNaiveStrategy generates correct bids when both import and export exist."""
    strategy = ExchangeEnergyNaiveStrategy()
    product_tuple = (datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 1), None)

    bids = strategy.calculate_bids(exchange_unit, market_config, [product_tuple])

    assert len(bids) == 2, "Should generate two bids (import/export)"

    for bid in bids:
        assert isinstance(bid, dict), "Each bid should be a dictionary"
        assert "start_time" in bid, "Missing start_time in bid"
        assert "end_time" in bid, "Missing end_time in bid"
        assert "price" in bid, "Missing price in bid"
        assert "volume" in bid, "Missing volume in bid"

        if bid["volume"] > 0:  # Import case
            assert bid["price"] == exchange_unit.price_import, "Incorrect import price"
        else:  # Export case
            assert bid["price"] == exchange_unit.price_export, "Incorrect export price"


def test_exchange_energy_naive_strategy_import_only(
    exchange_import_only, market_config
):
    """Test if the strategy only generates import bids when export is unavailable."""
    strategy = ExchangeEnergyNaiveStrategy()
    product_tuple = (datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 1), None)

    bids = strategy.calculate_bids(exchange_import_only, market_config, [product_tuple])

    assert len(bids) == 1, "Only import bid should be generated"
    assert bids[0]["volume"] > 0, "Import volume should be positive"
    assert bids[0]["price"] == exchange_import_only.price_import, (
        "Incorrect import price"
    )


def test_exchange_energy_naive_strategy_export_only(
    exchange_export_only, market_config
):
    """Test if the strategy only generates export bids when import is unavailable."""
    strategy = ExchangeEnergyNaiveStrategy()
    product_tuple = (datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 1), None)

    bids = strategy.calculate_bids(exchange_export_only, market_config, [product_tuple])

    assert len(bids) == 1, "Only export bid should be generated"
    assert bids[0]["volume"] < 0, "Export volume should be negative"
    assert bids[0]["price"] == exchange_export_only.price_export, (
        "Incorrect export price"
    )
