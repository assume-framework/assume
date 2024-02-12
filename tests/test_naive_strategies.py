# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta as rd

from assume.common.market_objects import MarketProduct
from assume.common.utils import get_available_products
from assume.strategies import (
    NaiveNegReserveStrategy,
    NaivePosReserveStrategy,
    NaiveProfileStrategy,
    NaiveSingleBidStrategy,
)
from tests.conftest import MockMinMaxUnit

start = datetime(2023, 7, 1)
end = datetime(2023, 7, 2)


def test_naive_strategy(mock_market_config, mock_supports_minmax):
    strategy = NaiveSingleBidStrategy()
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


def test_naive_da_strategy(mock_market_config, mock_supports_minmax):
    # test with mock market
    strategy = NaiveProfileStrategy()
    mc = mock_market_config
    unit = mock_supports_minmax
    product_tuples = [(start, end, None)]
    bids = strategy.calculate_bids(unit, mc, product_tuples=product_tuples)
    assert bids[0]["price"] == 3
    assert bids[0]["volume"] == {start: 400}

    # test with dam
    mc.market_products = [MarketProduct(rd(hours=+1), 24, rd(hours=24))]
    next_opening = start
    products = get_available_products(mc.market_products, next_opening)
    assert len(products) == 24
    index = pd.date_range(
        start=products[0][0],
        end=products[-1][0],
        freq="1h",
    )
    unit = MockMinMaxUnit(index)

    bids = strategy.calculate_bids(unit, mc, product_tuples=products)
    assert bids[0]["price"] == 3
    assert len(bids[0]["volume"]) == 24
    assert bids[0]["volume"][products[0][0]] == 400
    assert bids[0]["volume"][products[-1][0]] == 400


if __name__ == "__main__":
    # run pytest and enable prints
    import pytest

    pytest.main(["-s", __file__])
