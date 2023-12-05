# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import pytest
from dateutil import rrule as rr

from assume.common.market_objects import MarketConfig, MarketProduct
from assume.strategies import OTCStrategy

start = datetime(2023, 7, 1)
end = datetime(2023, 7, 2)


def test_otc_strategy(mock_supports_minmax):
    strategy = OTCStrategy(scale_firm_power_capacity=0.1)

    mc = MarketConfig(
        "Test",
        rr.rrule(rr.HOURLY),
        timedelta(hours=1),
        "not needed",
        [MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
    )

    unit = mock_supports_minmax
    start = datetime(2023, 7, 1)
    end = datetime(2023, 7, 2)
    product_tuples = [(start, end, None)]

    bids = strategy.calculate_bids(
        unit=unit, market_config=mc, product_tuples=product_tuples
    )

    assert bids[0]["price"] == 3
    assert bids[0]["volume"] == 400


@pytest.mark.parametrize("scale", [0.1, 0.2, 1.0, 0.0])
def test_otc_strategy_scaled(scale, mock_supports_minmax):
    strategy = OTCStrategy(scale_firm_power_capacity=scale)

    mc = MarketConfig(
        "OTC",
        rr.rrule(rr.HOURLY),
        timedelta(hours=1),
        "not needed",
        [MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
    )

    unit = mock_supports_minmax
    start = datetime(2023, 7, 1)
    end = datetime(2023, 7, 2)
    product_tuples = [(start, end, None)]

    bids = strategy.calculate_bids(
        unit=unit, market_config=mc, product_tuples=product_tuples
    )

    power = 400 * scale

    assert bids[0]["price"] == 3
    assert bids[0]["volume"] == power


if __name__ == "__main__":
    test_otc_strategy()
    test_otc_strategy_scaled()
