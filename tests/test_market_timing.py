# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from datetime import datetime, timedelta

import dateutil.rrule as rr
from dateutil.relativedelta import relativedelta as rd

from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.market_timing import check_market_timing, render_market_timing

# the diagrams are printed in each test, pytest shows them if a test fails
start = datetime(2019, 1, 1)


def daily_market(market_id, first_opening, opening_hours, first_delivery_hours):
    """A market opening daily, trading 24 hourly products."""
    return MarketConfig(
        market_id=market_id,
        opening_hours=rr.rrule(
            rr.DAILY, dtstart=first_opening, until=start + timedelta(days=3)
        ),
        opening_duration=timedelta(hours=opening_hours),
        market_products=[
            MarketProduct(rd(hours=1), 24, rd(hours=first_delivery_hours))
        ],
    )


def test_valid_setup():
    """
    Setup as in example_01d: the day-ahead market is cleared at 20:00,
    redispatch opens afterwards at 21:00 with the day-ahead results available.
    Both markets are cleared before their delivery starts at 00:00.
    """
    configs = [
        daily_market("EOM", start, opening_hours=20, first_delivery_hours=24),
        daily_market(
            "redispatch",
            start + timedelta(hours=21),
            opening_hours=2,
            first_delivery_hours=3,
        ),
    ]
    text = render_market_timing(configs, use_ascii=True)
    print(text)

    assert check_market_timing(configs) == []
    assert "EOM" in text and "redispatch" in text
    assert "WARNING" not in text and "ERROR" not in text


def test_opening_while_other_market_is_cleared():
    """
    The day-ahead market is cleared at 12:00, redispatch opens at the same time.
    As agents bid directly on opening, the day-ahead results are not available yet,
    although redispatch trades the same delivery hours, so a warning is raised.
    """
    configs = [
        daily_market("EOM", start, opening_hours=12, first_delivery_hours=12),
        daily_market(
            "redispatch",
            start + timedelta(hours=12),
            opening_hours=1,
            first_delivery_hours=1,
        ),
    ]
    text = render_market_timing(configs, use_ascii=True)
    print(text)

    issues = check_market_timing(configs)
    assert [(i.level, i.market_id, i.opening) for i in issues] == [
        ("warning", "redispatch", datetime(2019, 1, 1, 12))
    ]
    # the clash is shown in the graph and the warning is printed below
    assert "timing clashes" in text
    assert "WARNING" in text


def test_opening_while_other_market_is_cleared_for_other_delivery():
    """
    Same as above, but redispatch trades the day after the cleared day-ahead hours.
    The day-ahead results are not needed, so the same time opening is fine.
    """
    configs = [
        daily_market("EOM", start, opening_hours=12, first_delivery_hours=12),
        daily_market(
            "redispatch",
            start + timedelta(hours=12),
            opening_hours=1,
            first_delivery_hours=48,
        ),
    ]
    print(render_market_timing(configs, use_ascii=True))

    assert check_market_timing(configs) == []


def test_bidding_after_delivery_started():
    """
    The market is cleared at 13:00, but its delivery already starts at 12:00.
    This is physically impossible and an error.
    """
    configs = [daily_market("EOM", start, opening_hours=13, first_delivery_hours=12)]
    print(render_market_timing(configs, use_ascii=True))

    issues = check_market_timing(configs)
    assert [(i.level, i.market_id, i.clash_start, i.clash_end) for i in issues] == [
        ("error", "EOM", datetime(2019, 1, 1, 12), datetime(2019, 1, 1, 13))
    ]
