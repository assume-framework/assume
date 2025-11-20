# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime

import pytest
from dateutil.relativedelta import relativedelta as rd

from assume.common.market_objects import MarketProduct
from assume.common.utils import get_available_products
from assume.strategies import (
    EnergyHeuristicElasticStrategy,
    EnergyNaiveProfileStrategy,
    EnergyNaiveStrategy,
)

start = datetime(2023, 7, 1)
end = datetime(2023, 7, 2)


def test_naive_strategy(mock_market_config, mock_supports_minmax):
    strategy = EnergyNaiveStrategy()
    mc = mock_market_config
    product_tuples = [(start, end, None)]
    bids = strategy.calculate_bids(
        mock_supports_minmax, mc, product_tuples=product_tuples
    )
    assert bids[0]["price"] == 3
    assert bids[0]["volume"] == 400


def test_naive_da_strategy(mock_market_config, mock_supports_minmax):
    # test with mock market
    strategy = EnergyNaiveProfileStrategy()
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

    bids = strategy.calculate_bids(unit, mc, product_tuples=products)
    assert bids[0]["price"] == 3
    assert len(bids[0]["volume"]) == 24
    assert bids[0]["volume"][products[0][0]] == 400
    assert bids[0]["volume"][products[-1][0]] == 400


class MockElasticDemand:
    """
    Minimal mock object to simulate an elastic demand unit.
    """

    def __init__(
        self,
        elasticity=-0.1,
        max_price=100.0,
        num_bids=4,
        max_power=500,
        min_power=0,
        node="nodeX",
        elasticity_model="isoelastic",
    ):
        self.elasticity = elasticity
        self.max_price = max_price
        self.min_price = 0.0
        self.elasticity_model = elasticity_model
        self.num_bids = num_bids
        self.max_power = max_power
        self.min_power = min_power
        self.node = node


@pytest.mark.parametrize(
    "elasticity, model",
    [
        (-0.1, "isoelastic"),
        (-0.5, "isoelastic"),
        (-0.1, "linear"),
        (-0.5, "linear"),
    ],
)
def test_demand_energy_heuristic_elastic_strategy_robust_range(
    mock_market_config, elasticity, model
):
    """
    Parametric test to check multiple elasticity values and model types.
    Ensures bid count, price monotonicity, and volume validity.
    """
    strategy = EnergyHeuristicElasticStrategy()
    unit = MockElasticDemand(
        elasticity=elasticity,
        max_price=100.0,
        num_bids=4,
        max_power=500,
        min_power=0,
        elasticity_model=model,
    )
    product_tuples = [(start, end, None)]
    bids = strategy.calculate_bids(unit, mock_market_config, product_tuples)

    assert len(bids) == unit.num_bids
    prices = [bid["price"] for bid in bids]
    volumes = [-bid["volume"] for bid in bids]

    assert all(prices[i] >= prices[i + 1] for i in range(len(prices) - 1))
    assert sum(volumes) == pytest.approx(unit.max_power, rel=1e-2)
    assert prices[0] == unit.max_price


def test_demand_energy_heuristic_elastic_strategy_raises_when_volume_exceeds_max(
    mock_market_config,
):
    """
    Tests that a ValueError is raised when first block bid volume exceeds max_power.
    """
    strategy = EnergyHeuristicElasticStrategy()
    unit = MockElasticDemand(
        elasticity=0.1,  # Extreme elasticity to trigger large volume
        max_price=500.0,
        num_bids=3,
        max_power=100,
        min_power=0,
        elasticity_model="isoelastic",
    )
    product_tuples = [(start, end, None)]

    with pytest.raises(ValueError, match="exceeds max power"):
        strategy.calculate_bids(unit, mock_market_config, product_tuples)


if __name__ == "__main__":
    # run pytest and enable prints
    import pytest

    pytest.main(["-s", __file__])
