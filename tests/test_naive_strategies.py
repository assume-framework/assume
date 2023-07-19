from datetime import datetime

from assume.strategies import (
    NaiveNegReserveStrategy,
    NaivePosReserveStrategy,
    NaiveStrategy,
    OperationalWindow,
)

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


def test_naive_strategy():
    strategy = NaiveStrategy()
    bids = strategy.calculate_bids(None, operational_window, None)
    mp = operational_window["states"]["max_power"]
    assert bids[0]["price"] == mp["cost"]
    assert bids[0]["volume"] == mp["volume"]
    assert bids == [{"price": 3, "volume": 400}]


def test_naive_pos_strategy():
    """
    calculates bids for
    """
    strategy = NaivePosReserveStrategy()
    bids = strategy.calculate_bids(None, operational_window, None)

    assert bids[0]["price"] == 0
    assert bids[0]["volume"] == operational_window["states"]["pos_reserve"]["volume"]
    assert bids == [{"price": 0, "volume": 20}]


def test_naive_neg_strategy():
    strategy = NaiveNegReserveStrategy()
    bids = strategy.calculate_bids(None, operational_window, None)

    assert bids[0]["price"] == 0
    assert bids[0]["volume"] == operational_window["states"]["neg_reserve"]["volume"]
    assert bids == [{"price": 0, "volume": 10}]


if __name__ == "__main__":
    test_naive_strategy()
    test_naive_neg_strategy()
    test_naive_pos_strategy()
