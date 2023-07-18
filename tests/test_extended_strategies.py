from datetime import datetime, timedelta

from dateutil import rrule as rr

from assume.common.market_objects import MarketConfig, MarketProduct
from assume.strategies import OperationalWindow, OTCStrategy

start = datetime(2023, 7, 1)
end = datetime(2023, 7, 2)
operational_window: OperationalWindow = {
    "window": (start, end),
    "states": {
        "current_power": {
            "volume": 100,
            "cost": 1,
        },
        "min_power": {
            "volume": 80,
            "cost": 2,
        },
        "max_power": {
            "volume": 400,
            "cost": 3,
        },
        "neg_reserve": {
            "volume": 10,
            "cost": 4,
        },
        "pos_reserve": {
            "volume": 20,
            "cost": 5,
        },
    },
}


def test_otc_strategy():
    strategy = OTCStrategy(scale_firm_power_capacity=0.1)

    mc = MarketConfig(
        "Test",
        rr.rrule(rr.HOURLY),
        timedelta(hours=1),
        "not needed",
        [MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
    )

    bids = strategy.calculate_bids(
        unit=None, market_config=mc, operational_window=operational_window
    )

    assert bids[0]["price"] == operational_window["states"]["max_power"]["cost"]
    assert bids[0]["volume"] == operational_window["states"]["max_power"]["volume"]
    assert bids == [{"price": 3, "volume": 400}]


import pytest


@pytest.mark.parametrize("scale", [0.1, 0.2, 1.0, 0.0])
def test_otc_strategy_scaled(scale):
    strategy = OTCStrategy(scale_firm_power_capacity=scale)

    mc = MarketConfig(
        "OTC",
        rr.rrule(rr.HOURLY),
        timedelta(hours=1),
        "not needed",
        [MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
    )

    bids = strategy.calculate_bids(
        unit=None, market_config=mc, operational_window=operational_window
    )

    power = operational_window["states"]["max_power"]["volume"] * scale

    assert bids[0]["price"] == operational_window["states"]["max_power"]["cost"]
    assert bids[0]["volume"] == power
    assert bids == [{"price": 3, "volume": power}]


if __name__ == "__main__":
    test_otc_strategy()
    test_otc_strategy_scaled()
