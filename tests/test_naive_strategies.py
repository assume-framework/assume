# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import pandas as pd
import pytest
from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd
from dateutil import rrule as rr

from assume.common.forecaster import UnitForecaster
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.utils import get_available_products
from assume.strategies import (
    EnergyHeuristicElasticStrategy,
    EnergyNaiveProfileStrategy,
    EnergyNaiveStrategy,
)
from assume.strategies.naive_strategies import EnergyNaiveRedispatchStrategy
from tests.conftest import MockMinMaxUnit

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

redispatch_start = datetime(2005, 6, 1, 0)
redispatch_end = datetime(2005, 6, 1, 3)
simple_redispatch_market_config = MarketConfig(
    market_id="simple_redispatch",
    market_products=[MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
    additional_fields=["node", "min_power", "max_power"],
    opening_hours=rr.rrule(
        rr.HOURLY,
        dtstart=redispatch_start,
        until=redispatch_end,
        cache=True,
    ),
    opening_duration=timedelta(hours=1),
    volume_unit="MW",
    volume_tick=0.1,
    maximum_bid_volume=None,
    price_unit="€/MW",
    market_mechanism="redispatch",
)

index = pd.date_range(start=redispatch_start, end=redispatch_end, freq="1h")
forecaster = UnitForecaster(index=index)
unit_for_redispatch_with_availability = MockMinMaxUnit(forecaster=forecaster)
unit_for_redispatch_with_availability.max_power = 1000
unit_for_redispatch_with_availability.min_power = 100


@pytest.mark.parametrize(
    "dummy_availability, dummy_current_power, expected_bids_volume, expected_bids_max_power, expected_min_power",
    [
        (
            pd.Series(0, index=index),
            pd.Series([0, 0, 0, 0], index=index),
            pd.Series([0, 0, 0, 0], index=index),
            pd.Series(0, index=index),
            pd.Series(0, index=index),
        ),  # no availability, not running on EOM -> no volume to offer
        (
            pd.Series(0.0, index=index),
            pd.Series([0, 0, 0, 0], index=index),
            pd.Series([0, 0, 0, 0], index=index),
            pd.Series(0, index=index),
            pd.Series(0, index=index),
        ),  # same as above, but with float availability
        (
            pd.Series(0.5, index=index),
            pd.Series([0, 100, 200, 500], index=index),
            pd.Series([0, 100, 200, 500], index=index),
            pd.Series(500, index=index),
            pd.Series(100, index=index),
        ),  # 0.5 availability
        (
            pd.Series(1.0, index=index),
            pd.Series([0, 100, 200, 1000], index=index),
            pd.Series([0, 100, 200, 1000], index=index),
            pd.Series(1000, index=index),
            pd.Series(100, index=index),
        ),  # full availability
    ],
)
def test_naive_redispatch_strategy(
    dummy_availability,
    dummy_current_power,
    expected_bids_volume,
    expected_bids_max_power,
    expected_min_power,
):
    # test with simple redispatch market
    strategy = EnergyNaiveRedispatchStrategy()
    mc = simple_redispatch_market_config
    unit = unit_for_redispatch_with_availability

    mc.market_products = [
        MarketProduct(rd(hours=+1), 4, rd(hours=0))
    ]  # 4 products with 1h duration, starting at redispatch_start
    next_opening = redispatch_start
    products = get_available_products(mc.market_products, next_opening)
    assert len(products) == 4

    unit.forecaster.availability = dummy_availability
    unit.outputs["energy"] = dummy_current_power
    bids = strategy.calculate_bids(unit, mc, product_tuples=products)

    assert [bids[i]["volume"] for i in range(len(bids))] == pytest.approx(
        expected_bids_volume
    )
    assert [bids[i]["max_power"] for i in range(len(bids))] == pytest.approx(
        expected_bids_max_power
    )
    assert [bids[i]["min_power"] for i in range(len(bids))] == pytest.approx(
        expected_min_power
    )
    assert [bids[i]["p_nom"] for i in range(len(bids))] == pytest.approx(
        [unit.max_power] * len(bids)
    )
    assert [bids[i]["node"] for i in range(len(bids))] == pytest.approx(
        [unit.node] * len(bids)
    )
    assert [bids[i]["price"] for i in range(len(bids))] == pytest.approx(
        [3] * len(bids)
    )  # should bid at marginal cost, which are set to 3 in MockMinMaxUnit


if __name__ == "__main__":
    # run pytest and enable prints
    import pytest

    pytest.main(["-s", __file__])
