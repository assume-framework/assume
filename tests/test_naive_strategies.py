from datetime import datetime

from assume.strategies import (
    NaiveNegReserveStrategy,
    NaivePosReserveStrategy,
    NaiveStrategy,
)

start = datetime(2023, 7, 1)
end = datetime(2023, 7, 2)


def test_naive_strategy(mock_market_config, mock_supports_minmax):
    strategy = NaiveStrategy()
    mc = mock_market_config
    product_tuples = [(start, end, None)]
    bids = strategy.calculate_bids(
        mock_supports_minmax, mc, product_tuples=product_tuples
    )
    assert bids[0]["price"] == 3
    assert bids[0]["volume"] == 400


def test_naive_pos_strategy(mock_market_config, mock_supports_minmax):
    """
    calculates bids for
    """
    strategy = NaivePosReserveStrategy()
    mc = mock_market_config
    product_tuples = [(start, end, None)]
    bids = strategy.calculate_bids(
        mock_supports_minmax, mc, product_tuples=product_tuples
    )
    assert bids[0]["price"] == 0
    assert bids[0]["volume"] == 400
    assert len(bids) == 1


def test_naive_neg_strategy(mock_market_config, mock_supports_minmax):
    strategy = NaiveNegReserveStrategy()
    mc = mock_market_config
    unit = mock_supports_minmax
    product_tuples = [(start, end, None)]
    bids = strategy.calculate_bids(unit, mc, product_tuples=product_tuples)
    assert bids[0]["price"] == 0
    assert bids[0]["volume"] == 100
    assert len(bids) == 1


if __name__ == "__main__":
    # run pytest and enable prints
    import pytest

    pytest.main(["-s", __file__])
