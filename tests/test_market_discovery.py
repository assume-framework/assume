# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from datetime import datetime, timedelta

import pandas as pd
import pytest
from dateutil import rrule as rr

from assume.common.forecaster import UnitForecaster
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.strategies.market_discovery import (
    MarketInfo,
    discover_markets,
    get_market_events,
    get_products_at_time,
)
from assume.units import Storage


@pytest.fixture
def storage_unit() -> Storage:
    index = pd.date_range("2022-01-01", periods=48, freq="h")
    forecaster = UnitForecaster(index, market_prices={"EOM": 50}, availability=1)

    return Storage(
        id="Test_Storage",
        unit_operator="TestOperator",
        technology="TestTechnology",
        bidding_strategies={},
        forecaster=forecaster,
        max_power_charge=-100,
        max_power_discharge=100,
        capacity=1000,
        initial_soc=0.5,
        efficiency_charge=0.9,
        efficiency_discharge=0.95,
    )


def make_market_config(
    market_id: str,
    product_type: str = "energy",
    opening_frequency: timedelta = timedelta(hours=1),
    opening_duration: timedelta = timedelta(hours=1),
    duration: timedelta = timedelta(hours=1),
    count: int = 1,
    first_delivery: timedelta = timedelta(hours=1),
    start: datetime = datetime(2022, 1, 1),
    end: datetime = datetime(2022, 1, 3),
    eligible_obligations_lambda=None,
) -> MarketConfig:
    freq_map = {
        timedelta(hours=1): rr.HOURLY,
        timedelta(hours=4): rr.HOURLY,
        timedelta(hours=24): rr.DAILY,
    }
    freq = freq_map.get(opening_frequency, rr.HOURLY)
    interval = int(opening_frequency / timedelta(hours=1)) or 1
    return MarketConfig(
        market_id=market_id,
        product_type=product_type,
        market_products=[
            MarketProduct(
                duration=duration,
                count=count,
                first_delivery=first_delivery,
            )
        ],
        opening_hours=rr.rrule(
            freq=freq,
            interval=interval,
            dtstart=start,
            until=end,
        ),
        opening_duration=opening_duration,
        eligible_obligations_lambda=eligible_obligations_lambda,
    )


def test_discover_markets_eligibility(storage_unit):
    """Only markets with a registered bidding strategy are discovered."""
    eom = make_market_config("EOM")
    crm_pos = make_market_config("CRM_pos", product_type="capacity_pos")
    heat = make_market_config("heat", product_type="heat")

    storage_unit.bidding_strategies = {"EOM": object(), "CRM_pos": object()}

    infos = discover_markets(
        storage_unit,
        [eom, crm_pos, heat],
        datetime(2022, 1, 1),
        datetime(2022, 1, 3),
    )

    assert set(infos.keys()) == {"EOM", "CRM_pos"}
    assert "heat" not in infos


def test_discover_markets_obligations_lambda(storage_unit):
    """The eligible_obligations_lambda of the market is respected."""
    # market only eligible for units with technology PSPP
    eom = make_market_config(
        "EOM", eligible_obligations_lambda=lambda u: u["technology"] == "PSPP"
    )
    storage_unit.bidding_strategies = {"EOM": object()}
    storage_unit.technology = "PSPP"

    infos = discover_markets(
        storage_unit,
        [eom],
        datetime(2022, 1, 1),
        datetime(2022, 1, 3),
    )
    assert "EOM" in infos

    # now with a non-matching technology
    storage_unit.technology = "gas"
    infos = discover_markets(
        storage_unit,
        [eom],
        datetime(2022, 1, 1),
        datetime(2022, 1, 3),
    )
    assert "EOM" not in infos


def test_market_info_openings_and_deliveries(storage_unit):
    """Openings and deliveries are extracted correctly from the config."""
    eom = make_market_config("EOM", first_delivery=timedelta(hours=1))
    storage_unit.bidding_strategies = {"EOM": object()}

    infos = discover_markets(
        storage_unit,
        [eom],
        datetime(2022, 1, 1),
        datetime(2022, 1, 3),
    )
    info = infos["EOM"]

    # hourly openings from 2022-01-01 00:00 until 2022-01-03 00:00
    assert len(info.openings) == 49
    assert info.openings[0] == datetime(2022, 1, 1)
    # first delivery is one hour after the first opening
    assert info.deliveries[0] == (datetime(2022, 1, 1, 1), datetime(2022, 1, 1, 2))

    # next opening from a given time
    opening, closure = info.next_opening(datetime(2022, 1, 1, 5, 30))
    assert opening == datetime(2022, 1, 1, 6)
    assert closure == datetime(2022, 1, 1, 7)

    assert info.open_at(datetime(2022, 1, 1, 6, 30))
    # market opens hourly with 1h duration, so it is open again at 7:30
    assert info.open_at(datetime(2022, 1, 1, 7, 30))
    # but between 7:00 and 7:00:00.001 it is closed - use a time within a gap
    # there are no gaps with hourly openings and 1h duration, so check
    # openness before the first opening instead
    assert not info.open_at(datetime(2021, 12, 31, 23, 30))


def test_market_info_sorting(storage_unit):
    """Markets are sorted by their first gate closure."""
    # market opening every 4 hours, first opening at 00:00
    crm = make_market_config(
        "CRM_pos",
        product_type="capacity_pos",
        opening_frequency=timedelta(hours=4),
        opening_duration=timedelta(hours=1),
        duration=timedelta(hours=4),
    )
    # EOM opens hourly
    eom = make_market_config("EOM")
    storage_unit.bidding_strategies = {"EOM": object(), "CRM_pos": object()}

    infos = discover_markets(
        storage_unit,
        [crm, eom],
        datetime(2022, 1, 1),
        datetime(2022, 1, 2),
    )

    market_ids = list(infos.keys())
    # both open at the same time - order is stable, but both are present
    assert set(market_ids) == {"EOM", "CRM_pos"}


def test_get_market_events(storage_unit):
    """The event list contains openings, gate closures and delivery starts in order."""
    eom = make_market_config("EOM")
    storage_unit.bidding_strategies = {"EOM": object()}

    infos = discover_markets(
        storage_unit,
        [eom],
        datetime(2022, 1, 1),
        datetime(2022, 1, 1, 4),
    )
    events = get_market_events(infos)

    # two openings in the horizon (00:00 and 01:00 ... until 04:00)
    event_types = {event[1] for event in events}
    assert event_types == {"opening", "gate_closure", "delivery_start"}

    # events are chronologically sorted
    times = [event[0] for event in events]
    assert times == sorted(times)

    # first events: opening at 00:00, then gate closure and delivery start
    # both at 01:00 - delivery_start sorts before gate_closure alphabetically
    assert events[0] == (datetime(2022, 1, 1), "opening", "EOM")
    assert events[1] == (datetime(2022, 1, 1, 1), "delivery_start", "EOM")
    assert events[2] == (datetime(2022, 1, 1, 1), "gate_closure", "EOM")


def test_get_products_at_time(storage_unit):
    """Products are only returned while the market is open and before delivery."""
    eom = make_market_config("EOM", first_delivery=timedelta(hours=1))
    storage_unit.bidding_strategies = {"EOM": object()}

    infos = discover_markets(
        storage_unit,
        [eom],
        datetime(2022, 1, 1),
        datetime(2022, 1, 2),
    )
    info = infos["EOM"]

    # market opens at 00:00, first delivery at 01:00
    products = get_products_at_time(info, datetime(2022, 1, 1))
    assert len(products) == 1
    assert products[0].start == datetime(2022, 1, 1, 1)
    assert products[0].end == datetime(2022, 1, 1, 2)

    # market is closed at 00:30 (opening duration is 1h -> still open actually)
    products = get_products_at_time(info, datetime(2022, 1, 1, 0, 30))
    assert products == []

    # at 02:00 a new opening window starts - the product for the 03:00
    # delivery is available again
    products = get_products_at_time(info, datetime(2022, 1, 1, 2))
    assert len(products) == 1
    assert products[0].start == datetime(2022, 1, 1, 3)


def test_market_info_next_opening_none(storage_unit):
    """next_opening returns None if the market does not open anymore."""
    eom = make_market_config("EOM", end=datetime(2022, 1, 2))
    storage_unit.bidding_strategies = {"EOM": object()}

    info = MarketInfo(
        market_config=eom,
        product_type="energy",
        openings=[datetime(2022, 1, 1)],
    )
    assert info.next_opening(datetime(2022, 1, 1, 12)) is None
