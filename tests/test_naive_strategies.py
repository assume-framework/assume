from datetime import datetime

from assume.strategies import (
    NaiveNegReserveStrategy,
    NaivePosReserveStrategy,
    NaiveStrategy,
)

start = datetime(2023, 7, 1)
end = datetime(2023, 7, 2)
operational_window = {
    "window": {"start": start, "end": end},
    "current_power": {
        "power": 100,
        "marginal_cost": 1,
    },
    "min_power": {
        "power": 80,
        "marginal_cost": 2,
    },
    "max_power": {
        "power": 400,
        "marginal_cost": 3,
    },
    "neg_reserve": {
        "capacity": 10,
        "marginal_cost": 4,
    },
    "pos_reserve": {
        "capacity": 20,
        "marginal_cost": 5,
    },
}


def test_naive_strategy():
    strategy = NaiveStrategy()
    bids = strategy.calculate_bids(None, None, operational_window)

    assert bids[0]["price"] == operational_window["max_power"]["marginal_cost"]
    assert bids[0]["volume"] == operational_window["max_power"]["power"]
    assert bids == [{"price": 3, "volume": 400}]


def test_naive_pos_strategy():
    """
    calculates bids for
    """
    strategy = NaivePosReserveStrategy()
    bids = strategy.calculate_bids(None, None, operational_window)

    assert bids[0]["price"] == 0
    assert bids[0]["volume"] == operational_window["pos_reserve"]["capacity"]
    assert bids == [{"price": 0, "volume": 20}]


def test_naive_neg_strategy():
    strategy = NaiveNegReserveStrategy()
    bids = strategy.calculate_bids(None, None, operational_window)

    assert bids[0]["price"] == 0
    assert bids[0]["volume"] == operational_window["neg_reserve"]["capacity"]
    assert bids == [{"price": 0, "volume": 10}]


if __name__ == "__main__":
    test_naive_strategy()
    test_naive_neg_strategy()
    test_naive_pos_strategy()
