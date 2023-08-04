from datetime import datetime

import pandas as pd

from assume.common.base import SupportsMinMax
from assume.strategies import (
    NaiveNegReserveStrategy,
    NaivePosReserveStrategy,
    NaiveStrategy,
)


class MockMarketConfig:
    product_type = "energy"


class MockUnit(SupportsMinMax):
    def __init__(self, index):
        super().__init__("", "", "", {}, index, None)
        self.max_power = 1000
        self.min_power = 0
        self.ramp_down = 200
        self.ramp_up = 400

    def calculate_min_max_power(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type="energy"
    ) -> tuple[pd.Series, pd.Series]:
        min = pd.Series(100, self.index).loc[start:start]
        max = pd.Series(400, self.index).loc[start:start]
        return min, max

    def calculate_marginal_cost(self, start: pd.Timestamp, power: float) -> float:
        return 3


start = datetime(2023, 7, 1)
end = datetime(2023, 7, 2)
index = pd.date_range(
    start=datetime(2023, 7, 1),
    end=datetime(2023, 7, 2),
    freq="1h",
)
unit = MockUnit(index)


def test_naive_strategy():
    strategy = NaiveStrategy()
    mc = MockMarketConfig()
    product_tuples = [(start, end, None)]
    bids = strategy.calculate_bids(unit, mc, product_tuples=product_tuples)
    assert bids[0]["price"] == 3
    assert bids[0]["volume"] == 400


def test_naive_pos_strategy():
    """
    calculates bids for
    """
    strategy = NaivePosReserveStrategy()
    mc = MockMarketConfig()
    product_tuples = [(start, end, None)]
    bids = strategy.calculate_bids(unit, mc, product_tuples=product_tuples)
    assert bids[0]["price"] == 0
    assert bids[0]["volume"] == 400
    assert len(bids) == 1


def test_naive_neg_strategy():
    strategy = NaiveNegReserveStrategy()
    mc = MockMarketConfig()
    product_tuples = [(start, end, None)]
    bids = strategy.calculate_bids(unit, mc, product_tuples=product_tuples)
    assert bids[0]["price"] == 0
    assert bids[0]["volume"] == 100
    assert len(bids) == 1


if __name__ == "__main__":
    # run pytest and enable prints
    import pytest

    pytest.main(["-s", __file__])
